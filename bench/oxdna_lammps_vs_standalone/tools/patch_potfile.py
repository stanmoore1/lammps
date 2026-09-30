#!/usr/bin/env python3
"""Write oxdna3_lj_ts.cgdna: LAMMPS oxDNA3 potential file with every angular-modulation
smoothing point dtheta_ast replaced by sqrt(0.81225/a), as upstream DNA3Interaction::init()
enforces for all sequence-dependent f4 modulations (the 'enslaved parameters' loop).
Only hbond and xstk entries change (stk already satisfies it; upstream DNA3 coaxial
stacking uses the non-SD oxDNA2 f4 tables, so coaxstk is left unchanged)."""
import numpy as np
ts = lambda a: repr(float(np.sqrt(0.81225 / a)))
out = []
for l in open('oxdna3_lj.cgdna'):
    p = l.split()
    if len(p) > 2 and p[2] == 'hbond':
        v = p[3:]
        for off in (6, 9, 12, 15, 18, 21, 24):   # (a, t0, ts) triplets: theta1,2,3,4at,4cg,7,8
            v[off + 2] = ts(float(v[off]))
        l = ' '.join(p[:3] + v) + '\n'
    elif len(p) > 2 and p[2] == 'xstk':
        v = p[3:]; o = 1 + 8 * 256
        for k in range(3): v[o + 3*k + 2] = ts(float(v[o + 3*k]))       # theta1,2,3
        o += 9
        for base in (o, o + 768):                                         # theta4 33/55: a[256] t0[256] ts[256]
            for k in range(256): v[base + 512 + k] = ts(float(v[base + k]))
        o += 1536
        v[o + 3] = ts(float(v[o])); v[o + 7] = ts(float(v[o + 4]))        # theta7, theta8: a t0_33 t0_55 ts
        l = ' '.join(p[:3] + v) + '\n'
    out.append(l)
open('oxdna3_lj_ts.cgdna', 'w').writelines(out)
