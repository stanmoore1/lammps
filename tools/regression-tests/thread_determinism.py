#!/usr/bin/env python3
"""Look for thread-safety problems by running each example twice, unchanged.

With a fixed seed and a fixed domain decomposition LAMMPS is deterministic, so
two runs of the same input with the same binary and the same number of MPI ranks
and threads have to produce byte-identical thermodynamic output.  If they do not,
the run depends on the order in which threads happened to reach a shared address.
That is either

  - an atomic floating point accumulation, whose sum depends on arrival order.
    This is a deliberate trade in the KOKKOS package: a half neighbor list with
    more than one thread accumulates forces atomically.  Expected with
    "-pk kokkos neigh half", and the reason the OpenMP regression configuration
    carries much looser tolerances than the Serial one.
  - a real data race: an unsynchronised read and write of the same address.
    Nothing makes this acceptable, and with "neigh full" (no atomics in the force
    accumulation) it is the only explanation left.

So the interesting run is "neigh full" with several threads: anything that moves
there is worth a debugger.  Comparing across thread counts instead would confound
the question, because changing the thread count changes the reduction order even
in a program with no races at all.

Why this rather than ThreadSanitizer: TSan models happens-before through atomic
operations on particular addresses and does not model a standalone
"atomic_thread_fence", which is how desul -- and therefore the whole Kokkos
OpenMP backend -- synchronises.  GCC says so at compile time ("atomic_thread_fence
is not supported with -fsanitize=thread"), and it is not a GCC limitation: a
race-free program that hands data between two threads with a relaxed flag and a
pair of fences is reported as a race by clang too.  A Kokkos program consisting
of nothing but one parallel_for writing a(i)=i, a fence, and a parallel_reduce
produced 28 "data race" reports under GCC's TSan, including one inside Kokkos
zeroing a freshly allocated View.  So a TSan report on this code proves nothing
without proving each report individually, while an unreproducible run is
evidence on its own terms.

Usage:
  thread_determinism.py --lmp-bin build-kokkos-omp/lmp --examples examples-ref \\
      --nprocs 1 --nthreads 4 --package-args "neigh full" [--repeats 2] \\
      [--inputs list.txt] [--timeout 120]
"""

import argparse
import os
import re
import subprocess
import sys

THERMO_HEADER_RE = re.compile(r'^\s*Step\s')

# below this an absolute difference is round-off on a quantity that is
# nominally zero, and its relative size carries no information
ABSOLUTE_FLOOR = 1.0e-10


def thermo_lines(text):
    """Return the thermo table rows of every run in a log, as plain strings.

    Only the numeric rows are kept: timings, memory use and the neighbor list
    summary legitimately differ between two runs of the same input.
    """
    rows = []
    in_table = False
    for line in text.split('\n'):
        if THERMO_HEADER_RE.match(line):
            in_table = True
            rows.append(line.rstrip())
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped or not re.match(r'^[-\d]', stripped):
            in_table = False
            continue
        rows.append(line.rstrip())
    return rows


def run_once(lmp, folder, script, nprocs, nthreads, package_args, timeout, extra_args):
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = str(nthreads)
    env['OMP_PROC_BIND'] = 'false'
    cmd = ['mpirun', '--host', f'localhost:{nprocs}', '-np', str(nprocs), lmp,
           '-cite', 'none', '-log', 'none', '-k', 'on', 't', str(nthreads), '-sf', 'kk']
    if package_args:
        cmd += ['-pk', 'kokkos'] + package_args.split()
    cmd += extra_args + ['-in', script]
    try:
        done = subprocess.run(cmd, cwd=folder, env=env, timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    text = done.stdout.decode(errors='replace')
    if done.returncode != 0:
        error = next((l.strip() for l in text.split('\n') if 'ERROR' in l), 'non-zero exit')
        return None, error[:120]
    return thermo_lines(text), None


def worst_relative_difference(a, b):
    """Largest relative difference between two thermo tables, and where.

    The magnitude is what separates the two explanations.  A sum whose order
    changed between runs moves the last bits of a column, so the difference is
    around 1e-16 relative and grows only as the trajectory amplifies it.  A read
    of memory another thread was writing moves whatever it moved, and lands
    orders of magnitude above that.
    """
    worst, where = 0.0, None
    for n, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        for u, v in zip(x.split(), y.split()):
            if u == v:
                continue
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                return float('inf'), (n, x.strip()[:80], y.strip()[:80])
            scale = max(abs(fu), abs(fv))
            rel = abs(fu - fv) / scale if scale else 0.0
            if rel > worst:
                worst, where = rel, (n, x.strip()[:80], y.strip()[:80])
    if len(a) != len(b):
        return float('inf'), (min(len(a), len(b)), f'{len(a)} rows', f'{len(b)} rows')
    return worst, where


def first_row_difference(a, b):
    """Relative difference of the first thermo row, which is computed at setup.

    Nothing has been integrated yet at that point, so a chaotic trajectory has
    had no opportunity to amplify anything: whatever shows up here was produced
    by the force and energy computation itself.
    """
    first_a = next((r for r in a if not r.lstrip().startswith('Step')), None)
    first_b = next((r for r in b if not r.lstrip().startswith('Step')), None)
    if first_a is None or first_b is None:
        return 0.0
    worst = 0.0
    for u, v in zip(first_a.split(), first_b.split()):
        try:
            fu, fv = float(u), float(v)
        except ValueError:
            continue
        # A thermo column that is zero up to round-off -- a net force, or a
        # pressure difference an input prints to show it is zero -- holds values
        # like 3e-14 whose ratio to each other is meaningless.  Comparing those
        # relatively reports an enormous difference for two runs that agree
        # perfectly well, which is how pair_style pace first looked like the
        # worst offender in the tree.  Require an absolute difference too.
        if abs(fu - fv) < ABSOLUTE_FLOOR:
            continue
        scale = max(abs(fu), abs(fv))
        if scale:
            worst = max(worst, abs(fu - fv) / scale)
    return worst


def first_difference(a, b):
    worst, where = worst_relative_difference(a, b)
    if where is None:
        return None
    return where[0], where[1], where[2], worst, first_row_difference(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lmp-bin', required=True)
    parser.add_argument('--examples', default='examples')
    parser.add_argument('--nprocs', type=int, default=1)
    parser.add_argument('--nthreads', type=int, default=4)
    parser.add_argument('--package-args', default='neigh full',
                        help='arguments for the "-pk kokkos" switch')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--inputs', help='file listing input scripts, one per line')
    parser.add_argument('--extra-args', default='',
                        help='further command line switches for every run')
    args = parser.parse_args()

    if args.inputs:
        with open(args.inputs) as src:
            scripts = [l.strip() for l in src if l.strip()]
    else:
        scripts = []
        for root, _, files in os.walk(args.examples):
            scripts += [os.path.join(root, f) for f in files if f.startswith('in.')]
        scripts.sort()

    extra = args.extra_args.split()
    nondeterministic, failed, ok = [], [], 0
    for script in scripts:
        folder, name = os.path.split(os.path.abspath(script))
        runs = []
        error = None
        for _ in range(args.repeats):
            rows, error = run_once(args.lmp_bin, folder, name, args.nprocs,
                                   args.nthreads, args.package_args, args.timeout, extra)
            if error:
                break
            runs.append(rows)
        if error:
            failed.append((script, error))
            print(f'  skip  {script}: {error}', flush=True)
            continue
        if not runs[0]:
            failed.append((script, 'no thermo output'))
            continue
        diff = None
        for other in runs[1:]:
            diff = first_difference(runs[0], other)
            if diff:
                break
        if diff:
            nondeterministic.append((script, diff))
            print(f'  DIFFERS {script}: row {diff[0]}, worst relative {diff[3]:.2e}, '
                  f'first row {diff[4]:.2e}\n'
                  f'      run 1: {diff[1]}\n      run 2: {diff[2]}', flush=True)
        else:
            ok += 1

    print(f'\n{len(scripts)} inputs: {ok} reproducible, '
          f'{len(nondeterministic)} nondeterministic, {len(failed)} not run')
    nondeterministic.sort(key=lambda e: -e[1][4])
    print('\nnondeterministic runs, by the difference in the FIRST thermo row.')
    print('That row is computed at setup, before anything is integrated, so a')
    print('chaotic trajectory cannot have amplified a reordered sum into it yet:')
    print('  first row    worst anywhere   input')
    for script, diff in nondeterministic:
        print(f'  {diff[4]:9.2e}    {diff[3]:9.2e}     {script}')
    early = [e for e in nondeterministic if e[1][4] > 1e-12]
    print(f'\n{len(early)} of {len(nondeterministic)} differ in the first row by more than 1e-12')
    return 1 if nondeterministic else 0


if __name__ == '__main__':
    sys.exit(main())
