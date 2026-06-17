#!/usr/bin/env python3
"""Overlay P0 vs rho (IK + H contours) from the three field-ladder runs at one
temperature.  Each field strength sweeps a different density range; together they
trace the homogeneous van der Waals loop P0(rho).

Usage: python3 p0_loop_combined.py [Tstr]   (default 0.980)"""
import sys
import numpy as np
import contour_pressure as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

Lz = 30.734555644208398917
Lx = 10.244851881402800231
Tstr = sys.argv[1] if len(sys.argv) > 1 else '0.980'
T = float(Tstr)
sm = 10
runs = [('T%s_d0.2' % Tstr, 0.2, 'tab:green'),
        ('T%s_d0.4' % Tstr, 0.4, 'tab:orange'),
        ('T%s_d0.8' % Tstr, 0.8, 'tab:red')]

fig, ax = plt.subplots(figsize=(7.5, 6))
for tag, du, c in runs:
    rik, PNik, PTik = cp.ik_profile(tag + '_ikstress.out', Lz, sm)
    P0ik = cp.p0_ik(PNik, PTik)
    ax.plot(rik, P0ik, color=c, lw=1.7, label=r'IK, $\Delta U=%.1f$' % du)
    rh, PNh, PTh = cp.h_profile(tag + '_hstress.out', tag + '_dens.out', Lz, Lx * Lx, T, sm)
    P0h = cp.p0_ik(PNh, PTh)
    ax.plot(rh, P0h, color=c, lw=1.0, ls='--', alpha=0.6)

ax.plot([], [], 'k-', lw=1.7, label='IK (solid)')
ax.plot([], [], 'k--', lw=1.0, label='H (dashed)')
ax.set_xlabel(r'$\rho^*$'); ax.set_ylabel(r'$P_0^*$')
ax.set_title(r'CPP LJTS $T^*=%s$:  $P_0$ vs $\rho$ van der Waals loop (field ladder)' % Tstr)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
plt.tight_layout()
out = 'p0_vs_rho_combined_T%s.png' % Tstr
plt.savefig(out, dpi=140)
print('wrote ' + out)
