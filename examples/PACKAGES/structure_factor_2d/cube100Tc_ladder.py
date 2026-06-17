#!/usr/bin/env python3
"""Consolidated IK/H contour summary across the cubic-N=100 field ladder, at Tc and
at 1.1 Tc.  Shows (left) the optimal mixing alpha vs field strength -- it sweeps with
gradient sharpness, NOT a universal constant -- and (right) the pure Harasima and IK
contour mu0 RMS vs field: H is best at weak field, IK at strong field, crossing over."""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, contour_pressure as cp, pets_eos as pets
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

L = 6.8582414181223398941; Lz = L; area = L * L; SM = 6
G = np.linspace(0.12, 0.57, 80)
SETS = {
    r'$T_c=1.089$': (1.089, [('cube100Tc2', 2.0), ('cube100Tc3', 3.0), ('cube100Tc4', 4.0)], 'tab:red'),
    r'$1.1\,T_c=1.198$': (1.198, [('cube100', 2.0), ('cube100u3', 3.0), ('cube100u4', 4.0)], 'tab:blue'),
}


def analyze(tag, T):
    pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
    sc = np.linspace(0.14, 0.56, 40); mp = np.array([pmu(x) for x in sc])
    rik, PNik, PTik = cp.ik_profile('%s_ikstress.out' % tag, Lz, SM)
    rh, PNh, PTh = cp.h_profile('%s_hstress.out' % tag, '%s_dens.out' % tag, Lz, area, T, SM)

    def tg(rho, y):
        z = (np.arange(len(rho)) + 0.5) / len(rho) * Lz; m = z <= Lz / 2
        r, yy = rho[m], y[m]; o = np.argsort(r)
        return np.interp(G, r[o], yy[o], left=np.nan, right=np.nan)
    P0ik = tg(rik, 1.5 * PTik - 0.5 * PNik); P0h = tg(rh, 1.5 * PTh - 0.5 * PNh)
    m = np.isfinite(P0ik) & np.isfinite(P0h)

    def rms(P0):
        rs, ps = G[m], P0[m]; cf = np.polyfit(rs, ps, 4); dP = np.polyder(np.poly1d(cf))
        rg = np.linspace(rs.min(), rs.max(), 120)
        mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
        mu += np.mean(mp - np.interp(sc, rg, mu))
        return np.sqrt(np.mean((np.interp(sc, rg, mu) - mp) ** 2))
    al = np.linspace(-0.8, 2.0, 281); r = np.array([rms(a * P0ik + (1 - a) * P0h) for a in al])
    return al[np.argmin(r)], rms(P0h), rms(P0ik), r.min()


fig, ax = plt.subplots(1, 2, figsize=(13, 5.3))
for lab, (T, rungs, col) in SETS.items():
    dU = np.array([d for _, d in rungs]); A, H, IK, MIX = [], [], [], []
    for tag, d in rungs:
        a, h, ik, mx = analyze(tag, T)
        A.append(a); H.append(h); IK.append(ik); MIX.append(mx)
        print('%-16s dU=%g: optimal a=%.2f  (H=%.4f IK=%.4f mix=%.4f)' % (tag, d, a, h, ik, mx))
    ax[0].plot(dU, A, 'o-', color=col, lw=2, ms=8, label=lab)
    ax[1].plot(dU, H, 's--', color=col, lw=1.8, ms=7, alpha=0.9, label='%s  Harasima' % lab)
    ax[1].plot(dU, IK, 'o-', color=col, lw=1.8, ms=7, label='%s  IK' % lab)
ax[0].axhline(1, color='gray', ls=':', label='pure IK'); ax[0].axhline(0, color='gray', ls='--', label='pure H')
ax[0].set_xlabel(r'field strength $\Delta U$'); ax[0].set_ylabel(r'optimal mixing $\alpha$')
ax[0].set_title(r'Best IK/H combination $P_0=\alpha\,IK+(1-\alpha)\,H$ drifts with gradient')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set_xlabel(r'field strength $\Delta U$'); ax[1].set_ylabel(r'$\mu_0$ RMS vs PeTS')
ax[1].set_title('Pure-contour accuracy crosses over: H best weak, IK best strong')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('cube100Tc_ladder.png', dpi=140)
print('wrote cube100Tc_ladder.png')
