#!/usr/bin/env python3
"""Optimal IK/H linear combination  P0 = a*P0_IK + (1-a)*P0_H  for one cube100 run.
Left: mu0(rho) for pure H, pure IK, and the best mix vs PeTS & Thol (least-squares
anchor).  Right: mu0 RMS vs the mixing coefficient a.
Usage: cube100_ikh_mix.py <tag> <dumax> <T>"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, contour_pressure as cp
import pets_eos as pets, thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else 'cube100Tc2'
dumax = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
T = float(sys.argv[3]) if len(sys.argv) > 3 else 1.089
L = 6.8582414181223398941; Lz = L; area = L * L; SM = 6
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T * np.log(r) + thol.properties(T, r)['mu_res']
sc = np.linspace(0.14, 0.56, 40); mp = np.array([pmu(x) for x in sc])
G = np.linspace(0.12, 0.57, 80)

rik, PNik, PTik = cp.ik_profile('%s_ikstress.out' % tag, Lz, SM)
rh, PNh, PTh = cp.h_profile('%s_hstress.out' % tag, '%s_dens.out' % tag, Lz, area, T, SM)


def tg(rho, y):
    z = (np.arange(len(rho)) + 0.5) / len(rho) * Lz; m = z <= Lz / 2
    r, yy = rho[m], y[m]; o = np.argsort(r)
    return np.interp(G, r[o], yy[o], left=np.nan, right=np.nan)


P0ik = tg(rik, 1.5 * PTik - 0.5 * PNik); P0h = tg(rh, 1.5 * PTh - 0.5 * PNh)
m = np.isfinite(P0ik) & np.isfinite(P0h)


def mu0_curve(P0):
    rs, ps = G[m], P0[m]; cf = np.polyfit(rs, ps, 4); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(rs.min(), rs.max(), 120)
    mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
    shift = np.mean(mp - np.interp(sc, rg, mu))           # least-squares anchor
    return rg, mu + shift


def rms_of(P0):
    rg, mu = mu0_curve(P0)
    return np.sqrt(np.mean((np.interp(sc, rg, mu) - mp) ** 2))


al = np.linspace(-0.6, 1.6, 221); rms = np.array([rms_of(a * P0ik + (1 - a) * P0h) for a in al])
ab = al[np.argmin(rms)]
print('%s (dUmax=%g, T=%g): pure H=%.4f  pure IK=%.4f  optimal a=%.2f -> %.4f'
      % (tag, dumax, T, rms_of(P0h), rms_of(P0ik), ab, rms.min()))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
rr = np.linspace(0.07, 0.58, 250)
ax[0].plot(rr, [pmu(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS')
ax[0].plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2, label='Thol 2015 EOS')
for P0, lab, col, ls in [(P0h, 'pure Harasima (a=0)', 'tab:green', '--'),
                         (P0ik, 'pure IK (a=1)', 'tab:blue', '--'),
                         (ab * P0ik + (1 - ab) * P0h, 'best mix (a=%.2f)' % ab, 'tab:red', '-')]:
    rg, mu = mu0_curve(P0)
    ax[0].plot(rg, mu, ls, color=col, lw=2.4 if ls == '-' else 1.7,
               label='%s  RMS %.4f' % (lab, rms_of(P0)))
ax[0].set_xlabel(r'$\rho^*$'); ax[0].set_ylabel(r'$\mu_0^*$ (anchored)')
ax[0].set_title(r'%s: IK/H mix $\mu_0(\rho)$ vs EOS' % tag); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(al, rms, '-', color='tab:purple', lw=2)
ax[1].plot(ab, rms.min(), 'o', color='tab:red', ms=8, label=r'optimal $a=%.2f$ (RMS %.4f)' % (ab, rms.min()))
ax[1].axvline(0, color='tab:green', ls=':', label='pure H')
ax[1].axvline(1, color='tab:blue', ls=':', label='pure IK')
ax[1].set_xlabel(r'mixing $a$  in  $a\,P_0^{IK}+(1-a)\,P_0^{H}$'); ax[1].set_ylabel(r'$\mu_0$ RMS vs PeTS')
ax[1].set_title(r'$\Delta U=%g$, $T=%g$: best linear combination' % (dumax, T))
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('%s_ikh_mix.png' % tag, dpi=140)
print('wrote %s_ikh_mix.png' % tag)
