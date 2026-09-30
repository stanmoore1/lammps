#!/usr/bin/env python3
"""Summarise eval_traj.py outputs: usage analyze.py <summary.txt> [...]
Per trajectory: mean total energies of hbond (hb), cross stacking (xstk) and coaxial
stacking (cx) in standalone oxDNA and in LAMMPS, and the per-frame differences."""
import sys
import numpy as np
for f in sys.argv[1:]:
    hdr = open(f).readline().split()[1:]
    d = np.loadtxt(f, ndmin=2); c = {k: d[:, i] for i, k in enumerate(hdr)}
    n = len(d)
    print(f'== {f}  ({n} frames)')
    tags = [t[3:] for t in hdr if t.startswith('hb_') and t != 'hb_ox']
    for term in ('hb', 'xstk', 'cx'):
        ox = c[f'{term}_ox']
        line = f'  {term:5s} standalone <E> = {ox.mean():9.4f}'
        for t in tags:
            dd = c[f'{term}_{t}'] - ox
            big = np.sum(np.abs(dd) > 1e-3)
            line += f' | {t}: <E> {c[f"{term}_{t}"].mean():9.4f}, <dE> {dd.mean():+.4f}, max|dE| {np.abs(dd).max():.4f}, frames |dE|>1e-3: {big}/{n}'
        print(line)
