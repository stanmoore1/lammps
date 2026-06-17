#!/usr/bin/env python3
"""Quick diagnostics + P0(z) contour plot for one CPP LJTS field-ladder run.
Checks: density range, interface wander (per-block slab center from the 1st cosine
mode -- PBC safe), and P0 = 3/2 PT - 1/2 PN agreement between the IK and Harasima
contours.  Plots P0(z) for both contours together with rho(z) vs z.

Usage: python3 check_contours.py [tag]   (default tag T0.980_d0.8)
"""
import sys
import numpy as np
import contour_pressure as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else 'T0.980_d0.8'
T = float(tag.split('_')[0][1:])
Lx = 10.244851881402800231        # N=1000, rho=0.31, aspect=3 (same box every run)
Lz = 30.734555644208398917
area = Lx * Lx
sm = 10

# smoothed contour profiles (cosine-fit inside ik_profile / h_profile)
rik, PNik, PTik = cp.ik_profile(tag + '_ikstress.out', Lz, sm)
rh, PNh, PTh = cp.h_profile(tag + '_hstress.out', tag + '_dens.out', Lz, area, T, sm)
P0ik = cp.p0_ik(PNik, PTik)
P0h = cp.p0_ik(PNh, PTh)
nb = len(rik); z = (np.arange(nb) + 0.5) * (Lz / nb)

print('=== %s  (T*=%.3f) ===' % (tag, T))
print('density range  IK: %.4f .. %.4f   (rho_l-rho_v = %.4f)'
      % (rik.min(), rik.max(), rik.max() - rik.min()))
print('density range  H : %.4f .. %.4f' % (rh.min(), rh.max()))


def read_blocks(fn):
    blocks, cur = [], None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 3:
            cur = []; blocks.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return [np.array(b) for b in blocks]


# interface wander: slab center per 100k-step block from the phase of the 1st
# cosine mode (the mode the cos field pins).  Small spread => well pinned.
blocks = read_blocks(tag + '_dens.out')
zc = []
for b in blocks:
    r = b[:, 3]; zz = (np.arange(len(r)) + 0.5) / len(r)
    zc.append((np.angle(np.sum(r * np.exp(2j * np.pi * zz))) % (2 * np.pi)) / (2 * np.pi) * Lz)
zc = np.array(zc)
mean_ang = np.angle(np.mean(np.exp(2j * np.pi * zc / Lz)))
dev = ((zc - mean_ang / (2 * np.pi) * Lz + Lz / 2) % Lz) - Lz / 2
print('slab center / block (z*): ' + ' '.join('%.2f' % v for v in zc))
print('interface wander: std = %.3f  (%.1f%% of Lz, %.2f bins)'
      % (dev.std(), 100 * dev.std() / Lz, dev.std() / (Lz / nb)))

vap = rik < rik.min() + 0.10 * (rik.max() - rik.min())
print('P0 vapor plateau  IK = %.4f   H = %.4f' % (P0ik[vap].mean(), P0h[vap].mean()))
print('corr(P0_IK, P0_H) = %.3f   max|diff| = %.4f   rms = %.4f'
      % (np.corrcoef(P0ik, P0h)[0, 1], np.max(np.abs(P0ik - P0h)),
         np.sqrt(np.mean((P0ik - P0h) ** 2))))

# --- plot: P0(z) for IK and H (left axis) + rho(z) (right axis) vs z ---
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(z, P0ik, color='tab:blue', lw=1.9, label=r'$P_0^{\rm IK}=\frac{3}{2}P_T^{\rm IK}-\frac{1}{2}P_N$')
ax.plot(z, P0h, color='tab:green', lw=1.9, label=r'$P_0^{\rm H}=\frac{3}{2}P_T^{\rm H}-\frac{1}{2}P_N$')
ax.axhline(P0ik[vap].mean(), color='gray', ls=':', lw=0.8)
ax.set_xlabel(r'$z^*$'); ax.set_ylabel(r'$P_0^*(z)$')
ax2 = ax.twinx()
ax2.plot(z, rik, color='tab:red', lw=1.3, ls='--', alpha=0.7, label=r'$\rho(z)$')
ax2.set_ylabel(r'$\rho^*(z)$', color='tab:red'); ax2.tick_params(axis='y', colors='tab:red')
l1, t1 = ax.get_legend_handles_labels(); l2, t2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, t1 + t2, loc='upper center', fontsize=9, ncol=1)
ax.set_title(r'CPP LJTS $T^*=%.3f$, $\Delta U=%s$:  $P_0(z)$ (IK vs H) and $\rho(z)$'
             % (T, tag.split('_d')[1]))
ax.grid(alpha=0.3)
plt.tight_layout()
out = 'p0_contours_%s.png' % tag
plt.savefig(out, dpi=140)
print('wrote ' + out)

# --- plot: P0 vs rho (parametric van der Waals loop) for IK and H ---
# trace in z-order: rho sweeps vapor -> liquid -> vapor across the two interfaces,
# so (rho, P0) draws the loop directly (no sorting, which would mix the two sides).
fig2, bx = plt.subplots(figsize=(7, 5.5))
bx.plot(rik, P0ik, color='tab:blue', lw=1.6, label='IK contour')
bx.plot(rh, P0h, color='tab:green', lw=1.6, label='H contour')
bx.set_xlabel(r'$\rho^*$'); bx.set_ylabel(r'$P_0^*$')
bx.set_title(r'CPP LJTS $T^*=%.3f$, $\Delta U=%s$:  $P_0$ vs $\rho$'
             % (T, tag.split('_d')[1]))
bx.legend(fontsize=9); bx.grid(alpha=0.3)
plt.tight_layout()
out2 = 'p0_vs_rho_%s.png' % tag
plt.savefig(out2, dpi=140)
print('wrote ' + out2)
