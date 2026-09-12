#!/usr/bin/env python3
"""For each disagreeing KOKKOS test, how far apart are the two FIRST thermo rows?

A divergence that only shows up after some steps is chaotic amplification of
round-off; one that is already there in the first row is a different force,
energy or pressure and cannot be explained that way.  This prints the worst
relative deviation of the first row, per test, so the two can be told apart -
which is what turns a list of a hundred and fifty failing comparisons into the
handful worth reading the code over.

Run it from the top of the source tree after a KOKKOS sweep:

  tools/regression-tests/first_row_delta.py [regression-work/kokkos-progress.yaml]
"""
import os, re, sys, yaml

COLS = ('PotEng', 'TotEng', 'Press', 'Temp', 'E_vdwl')

def thermo_rows(path):
    """Return (header, first data row) of the first thermo block in a log."""
    header = None
    with open(path, errors='ignore') as f:
        for line in f:
            parts = line.split()
            if header is None:
                if parts and parts[0] == 'Step' and len(parts) > 1:
                    header = parts
                continue
            try:
                row = [float(x) for x in parts]
            except ValueError:
                header = None
                continue
            if len(row) == len(header):
                return header, row
    return None, None

def main():
    progress = sys.argv[1] if len(sys.argv) > 1 else 'regression-work/kokkos-progress.yaml'
    rows = {}
    for line in open(progress):
        if line.startswith('~') or ': ' not in line:
            continue
        name, _, val = line.partition(': ')
        try:
            e = yaml.safe_load(val)
        except yaml.YAMLError:
            continue
        if isinstance(e, dict):
            rows[(e.get('folder', ''), name)] = e

    out = []
    for (folder, name), e in rows.items():
        checks = e.get('failed_checks')
        if not isinstance(checks, dict):
            continue
        if not (checks.get('abs_diff_failed') or checks.get('rel_diff_failed')):
            continue
        base = name[3:] if name.startswith('in.') else name
        kk = os.path.join(folder, f'log.{base}.4')
        # the reference log written by --gen-ref is log.<date>.<base>.<compiler>.4,
        # next to the log.<base>.4 the KOKKOS run just wrote
        refs = [os.path.join(folder, f) for f in os.listdir(folder)
                if re.match(rf'log\.[^.]+\.{re.escape(base)}\.[^.]+\.4$', f)] if os.path.isdir(folder) else []
        if not os.path.isfile(kk) or not refs:
            out.append((None, folder, name, 'no log pair'))
            continue
        h1, r1 = thermo_rows(kk)
        h2, r2 = thermo_rows(refs[0])
        if not r1 or not r2 or h1 != h2:
            out.append((None, folder, name, 'thermo tables not comparable'))
            continue
        worst, worstcol = 0.0, ''
        for col in COLS:
            if col not in h1:
                continue
            i = h1.index(col)
            a, b = r1[i], r2[i]
            scale = max(abs(a), abs(b))
            if scale == 0:
                continue
            rel = abs(a - b) / scale
            if rel > worst:
                worst, worstcol = rel, col
        out.append((worst, folder, name, f'{worstcol} first row rel {worst:.2e}'))

    known = [o for o in out if o[0] is not None]
    known.sort(key=lambda o: -o[0])
    print(f'{len(out)} disagreeing tests, {len(known)} with comparable first rows\n')
    print('worst first-row relative deviation, largest first:')
    for worst, folder, name, detail in known:
        print(f'  {worst:9.2e}  {folder}/{name}')
    buckets = {'> 1e-3': 0, '1e-6 .. 1e-3': 0, '1e-10 .. 1e-6': 0, 'first row identical': 0}
    for worst, *_ in known:
        if worst > 1e-3: buckets['> 1e-3'] += 1
        elif worst > 1e-6: buckets['1e-6 .. 1e-3'] += 1
        elif worst > 0: buckets['1e-10 .. 1e-6'] += 1
        else: buckets['first row identical'] += 1
    print('\nsummary:')
    for k, v in buckets.items():
        print(f'  {v:4d}  {k}')
    for o in out:
        if o[0] is None:
            print(f'  (skipped) {o[1]}/{o[2]}: {o[3]}')

main()
