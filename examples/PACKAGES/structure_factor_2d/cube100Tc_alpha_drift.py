#!/usr/bin/env python3
"""Where does the optimal IK/H mixing alpha drift as the Tc field is weakened?
Cubic N=100 at Tc=1.089, field ladder dUmax = 0.25,0.5,1,2,3,4 (each 2x).  For each
field: optimal a in P0=a*IK+(1-a)*H over a FIXED EOS window, with a wide scan; flag
the runs whose sampled density range no longer covers that window (extrapolation ->
a is ill-defined).  Second panel: the sampled rho-range collapses toward rho_c as the
field weakens, which is WHY a drifts off."""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, contour_pressure as cp, pets_eos as pets
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.089; L = 6.8582414181223398941; Lz = L; area = L * L; SM = 6
RHOC = 0.3092
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
WIN = (0.20, 0.42)                                      # fixed comparison window (around rho_c)
sc = np.linspace(*WIN, 40); mp = np.array([pmu(x) for x in sc])
LAD = [('cube100Tc025', 0.25), ('cube100Tc05', 0.5), ('cube100Tc1', 1.0),
       ('cube100Tc2', 2.0), ('cube100Tc3', 3.0), ('cube100Tc4', 4.0)]


def analyze(tag):
    rik, PNik, PTik = cp.ik_profile('%s_ikstress.out' % tag, Lz, SM)
    rh, PNh, PTh = cp.h_profile('%s_hstress.out' % tag, '%s_dens.out' % tag, Lz, area, T, SM)
    lo, hi = max(rik.min(), rh.min()), min(rik.max(), rh.max())
    G = np.linspace(max(lo + 0.01, 0.05), hi - 0.01, 90)

    def tg(rho, y):
        z = (np.arange(len(rho)) + 0.5) / len(rho) * Lz; mm = z <= Lz / 2
        r, yy = rho[mm], y[mm]; o = np.argsort(r)
        return np.interp(G, r[o], yy[o])
    P0ik, P0h = tg(rik, 1.5 * PTik - 0.5 * PNik), tg(rh, 1.5 * PTh - 0.5 * PNh)

    def rms(P0):
        cf = np.polyfit(G, P0, 4); dP = np.polyder(np.poly1d(cf))
        rg = np.linspace(G.min(), G.max(), 120)
        mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
        mu += np.mean(mp - np.interp(sc, rg, mu))
        return np.sqrt(np.mean((np.interp(sc, rg, mu) - mp) ** 2))
    al = np.linspace(-3.0, 2.5, 276); r = np.array([rms(a * P0ik + (1 - a) * P0h) for a in al])
    cover = (lo <= WIN[0]) and (hi >= WIN[1])           # does sampled range cover the window?
    sharp = (np.median(r) - r.min())                    # depth of the minimum (well-defined?)
    return al[np.argmin(r)], r.min(), lo, hi, cover, sharp


dU = np.array([d for _, d in LAD]); A = []; LO = []; HI = []; COV = []; SH = []
print('dUmax  opt_a   minRMS  rho[min,max]   covers[0.20,0.42]?  min-depth')
for tag, d in LAD:
    a, mn, lo, hi, cov, sh = analyze(tag)
    A.append(a); LO.append(lo); HI.append(hi); COV.append(cov); SH.append(sh)
    print('%5s  %+5.2f  %.4f  [%.3f,%.3f]   %-5s              %.4f' % (d, a, mn, lo, hi, cov, sh))
A, LO, HI, COV = np.array(A), np.array(LO), np.array(HI), np.array(COV)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.3))
ok = COV; bad = ~COV
ax[0].plot(dU, A, '-', color='gray', lw=1.5, zorder=1)
ax[0].scatter(dU[ok], A[ok], s=90, color='tab:blue', zorder=3, label='window covered (a well-defined)')
ax[0].scatter(dU[bad], A[bad], s=90, color='tab:red', marker='X', zorder=3,
              label='window NOT covered (extrapolated -> a drifts off)')
ax[0].axhline(0, color='gray', ls='--', lw=0.8); ax[0].axhline(1, color='gray', ls=':', lw=0.8)
ax[0].set_xscale('log'); ax[0].set_xticks(dU); ax[0].set_xticklabels(['0.25', '0.5', '1', '2', '3', '4'])
ax[0].set_xlabel(r'field $\Delta U$ (log)'); ax[0].set_ylabel(r'optimal mixing $\alpha$')
ax[0].set_title(r'$\alpha$ drifts down with field, then runs off below $\Delta U\approx2$')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].fill_between(dU, LO, HI, color='tab:green', alpha=0.3, label='sampled $\\rho$ range')
ax[1].plot(dU, LO, 'o-', color='tab:green', ms=5); ax[1].plot(dU, HI, 'o-', color='tab:green', ms=5)
ax[1].axhline(RHOC, color='k', ls='-', lw=1.2, label=r'$\rho_c=0.309$')
ax[1].axhspan(WIN[0], WIN[1], color='tab:blue', alpha=0.12, label='comparison window')
ax[1].set_xscale('log'); ax[1].set_xticks(dU); ax[1].set_xticklabels(['0.25', '0.5', '1', '2', '3', '4'])
ax[1].set_xlabel(r'field $\Delta U$ (log)'); ax[1].set_ylabel(r'$\rho^*$')
ax[1].set_title('Accessible density range collapses to $\\rho_c$ as field weakens')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('cube100Tc_alpha_drift.png', dpi=140)
print('wrote cube100Tc_alpha_drift.png')
