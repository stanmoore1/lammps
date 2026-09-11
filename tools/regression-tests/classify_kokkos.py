#!/usr/bin/env python3
"""Sort a KOKKOS regression run into the four buckets a developer cares about.

run_tests.py reports per test whether the numbers matched, which is the right
answer for a regression gate but not for hunting bugs in the KOKKOS styles: a
test that stops with "Cannot yet use fix pour with the KOKKOS package" and one
that segfaults and one whose energies drift apart all end up as failures.

This script separates them, and drops everything that already fails with the
plain CPU styles, since that is not a KOKKOS problem:

  1. crashed         - died without a LAMMPS error message, lost atoms, or NaN
  2. unsupported     - stopped with a clean LAMMPS error, so it cannot be run
                       under KOKKOS at all and belongs in the "skip" list
  3. differs         - ran to the end but the thermo output does not match
  4. agrees          - ran to the end and matched

Only (3) is a candidate bug list; (1) is usually a worse bug and (2) is a
documentation and configuration matter.
"""

import argparse
import os
import re
import sys

import yaml

# thermo output that is not a number at all
NAN_RE = re.compile(r'(?<![A-Za-z0-9_])-?(nan|inf)(?![A-Za-z0-9_])', re.IGNORECASE)
# a LAMMPS error message, as opposed to dying without one
LAMMPS_ERROR_RE = re.compile(r'^ERROR(?: on proc \d+)?:\s*(.*)$', re.MULTILINE)
# the signatures of a run that died rather than stopped
CRASH_MARKERS = ('Segmentation fault', 'signal', 'Aborted', 'core dumped',
                 'terminate called', 'bad_alloc', 'Kokkos::abort',
                 'RangePolicy bounds error', 'Assertion')
LOST_ATOMS_RE = re.compile(r'Lost atoms', re.IGNORECASE)


def load_progress(path):
    with open(path) as f:
        # every line is "<input>: {flow style mapping}", and the header lines
        # written at the start of a run are not part of the mapping
        text = '\n'.join(l for l in f.read().split('\n') if not l.startswith('~'))
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def passed(entry):
    return str(entry.get('status', '')).startswith('completed')


def read_log(tree, entry, name):
    """Return the text of the log the run just wrote, if it is still there."""
    folder = entry.get('folder')
    if not folder or not tree:
        return ''
    basename = name[3:] if name.startswith('in.') else name
    for candidate in (f'log.{basename}.4', f'log.{basename}.1'):
        path = os.path.join(tree, folder, candidate)
        if os.path.isfile(path):
            try:
                with open(path, errors='ignore') as f:
                    return f.read()
            except OSError:
                return ''
    return ''


def classify(name, entry, tree):
    """Return (bucket, detail) for one KOKKOS test result."""
    status = str(entry.get('status', ''))
    log = read_log(tree, entry, name)
    haystack = status + '\n' + log

    if LOST_ATOMS_RE.search(haystack):
        return 'crashed', 'lost atoms'
    # only look for NaN in the thermo output, where a stray "inf" in a file
    # name or a comment cannot be mistaken for a diverging number
    for line in log.split('\n'):
        stripped = line.strip()
        if stripped and NAN_RE.search(stripped) and re.match(r'^[\s\d.eE+-]*(nan|inf)', stripped, re.IGNORECASE):
            return 'crashed', f'non-finite thermo value: {stripped[:60]}'
    if any(marker in haystack for marker in CRASH_MARKERS):
        return 'crashed', 'died with a signal or abort'

    error = LAMMPS_ERROR_RE.search(haystack)
    if error:
        return 'unsupported', error.group(1).strip()
    # run_tests.py puts the error message into the status when it has one
    if 'ERROR:' in status:
        return 'unsupported', status.split('ERROR:', 1)[1].strip()

    if status.startswith('failed'):
        if 'timeout' in status:
            return 'crashed', 'timed out'
        detail = status[len('failed,'):].strip() if status.startswith('failed,') else status
        diverged_at = entry.get('diverged_at')
        if diverged_at is not None:
            detail += f' (first differs at step {diverged_at})'
        return 'differs', detail
    if passed(entry):
        return 'agrees', ''
    return 'other', status


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cpu', required=True, help='progress file of the CPU run')
    parser.add_argument('--kokkos', required=True, help='progress file of the KOKKOS run')
    parser.add_argument('--examples', default='',
                        help='examples tree the runs used, to read the generated logs')
    parser.add_argument('--output', default='-', help='where to write the report')
    args = parser.parse_args()

    cpu = load_progress(args.cpu)
    kokkos = load_progress(args.kokkos)

    buckets = {'crashed': [], 'unsupported': [], 'differs': [], 'agrees': [], 'other': []}
    cpu_broken = []
    for name, entry in sorted(kokkos.items()):
        cpu_entry = cpu.get(name)
        if cpu_entry is None:
            continue
        if not passed(cpu_entry):
            # not a KOKKOS problem: it does not work with the plain styles either
            cpu_broken.append((name, str(cpu_entry.get('status', ''))))
            continue
        bucket, detail = classify(name, entry, args.examples)
        buckets[bucket].append((os.path.join(str(entry.get('folder', '')), name), detail))

    titles = {
        'crashed': '1. Crashed (segfault, lost atoms, NaN, timeout)',
        'unsupported': '2. Stopped with a LAMMPS error: exclude from KOKKOS testing',
        'differs': '3. Ran but disagrees with the CPU reference: candidate bugs',
        'agrees': '4. Agrees with the CPU reference',
        'other': '5. Unclassified',
    }
    out = ['# KOKKOS Serial vs plain CPU, whole examples tree', '',
           f'Tests run under KOKKOS: {len(kokkos)}',
           f'Of those, already failing with the plain CPU styles (not counted below): {len(cpu_broken)}',
           '']
    out.append('| bucket | tests |')
    out.append('|---|---|')
    for key in ('crashed', 'unsupported', 'differs', 'agrees', 'other'):
        if buckets[key] or key != 'other':
            out.append(f'| {titles[key]} | {len(buckets[key])} |')
    out.append('')
    for key in ('crashed', 'unsupported', 'differs', 'agrees', 'other'):
        entries = buckets[key]
        if not entries:
            continue
        out.append(f'## {titles[key]} ({len(entries)})')
        out.append('')
        if key == 'agrees':
            for path, _ in entries:
                out.append(f'- `{path}`')
        else:
            for path, detail in entries:
                out.append(f'- `{path}`' + (f' - {detail}' if detail else ''))
        out.append('')

    text = '\n'.join(out)
    if args.output == '-':
        sys.stdout.write(text)
    else:
        with open(args.output, 'w') as f:
            f.write(text)
        print(f'wrote {args.output}')
    for key in ('crashed', 'unsupported', 'differs', 'agrees'):
        print(f'{key:12s} {len(buckets[key])}')


if __name__ == '__main__':
    main()
