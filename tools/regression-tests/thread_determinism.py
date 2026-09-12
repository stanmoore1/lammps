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


def first_difference(a, b):
    for n, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return n, x.strip()[:90], y.strip()[:90]
    if len(a) != len(b):
        return min(len(a), len(b)), f'{len(a)} rows', f'{len(b)} rows'
    return None


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
            print(f'  DIFFERS {script}: row {diff[0]}\n'
                  f'      run 1: {diff[1]}\n      run 2: {diff[2]}', flush=True)
        else:
            ok += 1

    print(f'\n{len(scripts)} inputs: {ok} reproducible, '
          f'{len(nondeterministic)} nondeterministic, {len(failed)} not run')
    for script, _ in nondeterministic:
        print(f'  nondeterministic: {script}')
    return 1 if nondeterministic else 0


if __name__ == '__main__':
    sys.exit(main())
