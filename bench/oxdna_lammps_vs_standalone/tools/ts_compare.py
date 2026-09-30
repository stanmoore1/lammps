#!/usr/bin/env python3
"""Q1: compare the oxDNA3 hbond / xstk angular modulations f4 of the LAMMPS potential file
(oxdna3_lj.cgdna, smoothing point dtheta_ast = ts as written in the file) with standalone
oxDNA3 (DNA3Interaction::init(), 'enslaved parameters' loop: ts = sqrt(0.81225 / A) for every
f4 term, overriding model.h and the parameter file).
f4(dt) = 1 - A dt^2 (|dt| < ts), B (tc - |dt|)^2 (ts <= |dt| < tc), 0 beyond;
tc = 1 / (A ts), B = A ts / (tc - ts) (continuity of value and slope at ts).
usage: ts_compare.py [oxdna3_lj.cgdna]"""
import sys, math
import numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else 'oxdna3_lj.cgdna'
L = {}
for l in open(P):
    p = l.split()
    if len(p) > 3 and not l.startswith('#'): L[(p[0], p[1], p[2])] = [float(x) for x in p[3:]]


def f4(dt, A, ts):
    tc = 1 / (A * ts); B = A * ts / (tc - ts); dt = abs(dt)
    return np.where(dt < ts, 1 - A * dt * dt, np.where(dt < tc, B * (tc - dt) ** 2, 0.0))


def report(name, A, ts_l):
    ts_s = math.sqrt(0.81225 / A)
    dt = np.linspace(0, 1.5, 150001)
    d = np.abs(f4(dt, A, ts_l) - f4(dt, A, ts_s))
    nz = dt[d > 1e-12]
    rng = f'{nz.min():.4f} - {nz.max():.4f}' if len(nz) else '-'
    print(f'{name:22s} A={A:6.3f}  ts: LAMMPS {ts_l:.6f}  standalone {ts_s:.6f}  '
          f'cutoff tc: {1/(A*ts_l):.4f} / {1/(A*ts_s):.4f}  f4 differs for |dtheta| in {rng} rad, max |df4| = {d.max():.4f} '
          f'(f4(ts_LAMMPS) = {1 - A*ts_l*ts_l:.4f} vs standalone {1 - 0.81225:.4f} at its own ts)')


h = L[('*', '*', 'hbond')][6:]
for n, k in [('hbond theta1', 0), ('hbond theta2', 3), ('hbond theta3', 6), ('hbond theta4 (A-T)', 9),
             ('hbond theta4 (C-G)', 12), ('hbond theta7', 15), ('hbond theta8', 18)]:
    report(n, h[k], h[k + 2])
v = L[('*', '*', 'xstk')]; o = 1 + 8 * 256
for k, n in enumerate(['xstk theta1', 'xstk theta2', 'xstk theta3']):
    report(n, v[o + 3 * k], v[o + 3 * k + 2])
o += 9 + 1536
report('xstk theta7', v[o], v[o + 3]); report('xstk theta8', v[o + 4], v[o + 7])
