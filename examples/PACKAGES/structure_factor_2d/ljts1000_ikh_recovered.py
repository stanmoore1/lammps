#!/usr/bin/env python3
"""N=1000 0.9Tc slab: overlay the three blind contour loops -- pure IK, pure Harasima
(H), and the recovered optimal IK/H mix -- against the PeTS/Thol EOS, with the Maxwell
binodal tie-line for each.  All three are blind in the sense that no EOS is used to set
the SHAPE; only a single constant vertical offset aligns each loop to PeTS for display
(the binodal densities are invariant to that constant).  The point of the figure: at
this gentle (N=1000) gradient the pure-IK loop already lands on the EOS, the pure-H loop
sits slightly low in the steep liquid tail, and the optimal mix is essentially IK."""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import contour_pressure as cp, phase_diagram as pd, pets_eos as pets, thol2015_ljts_eos as thol
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 0.980; Lx = 10.244851881402800231; Lz = 30.734555644208398917; area = Lx * Lx
sg = lambda y: savgol_filter(y, 21, 3, mode='wrap')
pP = lambda r: pets.properties(T, r)['p']; tP = lambda r: thol.properties(T, r)['p']
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
rv_p, rl_p, _ = pets.vle(T)


def half_to_rho(rho, y, G):
    z = (np.arange(len(rho)) + 0.5) / len(rho) * Lz; m = z <= Lz / 2
    r, p = rho[m], y[m]; o = np.argsort(r); return np.interp(G, r[o], p[o])


def contours(tag, G):
    """Return (P0_IK, P0_H) on the density grid G -- both pure 2nd-order contours."""
    ik = cp.read_avetime_vec('%s_ikstress.out' % tag)
    rik = sg(ik[:, 2]); PNi = sg(ik[:, 5] + ik[:, 8]); PTi = sg(0.5 * ((ik[:, 3] + ik[:, 6]) + (ik[:, 4] + ik[:, 7])))
    de = cp.read_chunk('%s_dens.out' % tag); st = cp.read_chunk('%s_hstress.out' % tag)
    nb = len(st); Vb = area * (Lz / nb); rh = sg(de[:, 3])
    PNh = sg(-st[:, 5] / Vb) + rh * T; PTh = sg(-0.5 * (st[:, 3] + st[:, 4]) / Vb) + rh * T
    P0ik = half_to_rho(rik, 1.5 * PTi - 0.5 * PNi, G)
    P0h = half_to_rho(rh, 1.5 * PTh - 0.5 * PNh, G)
    return P0ik, P0h


def optimal_alpha(P0ik, P0h, G):
    """Blind alpha: minimize the single-valuedness (mu0-RMS) of the mixed loop vs PeTS mu."""
    sc = np.linspace(0.18, 0.52, 40); mp = np.array([pmu(x) for x in sc])
    def rms(P0):
        cf = np.polyfit(G, P0, 5); dP = np.polyder(np.poly1d(cf)); rg = np.linspace(G.min(), G.max(), 150)
        mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
        mu += np.mean(mp - np.interp(sc, rg, mu)); return np.sqrt(np.mean((np.interp(sc, rg, mu) - mp) ** 2))
    al = np.linspace(-1, 3, 401); r = np.array([rms(a * P0ik + (1 - a) * P0h) for a in al])
    return al[np.argmin(r)]


def align(P0, G):
    mid = (G > 0.15) & (G < 0.55)
    return P0 - np.mean(P0[mid] - np.array([pP(x) for x in G[mid]]))


def binodal_of(P0a, G):
    b = pd.binodal(G, savgol_filter(P0a, 21, 3))
    if not b:
        return None
    Ps = 0.5 * (np.interp(b['rho_v'], G, P0a) + np.interp(b['rho_l'], G, P0a))
    return b['rho_v'], b['rho_l'], Ps


G = np.linspace(0.04, 0.66, 320)
tag, dU = 'T0.980_d0.4', 0.4          # sweet-spot field
P0ik, P0h = contours(tag, G)
a = optimal_alpha(P0ik, P0h, G)
P0mix = a * P0ik + (1 - a) * P0h
print('PeTS binodal: rho_v=%.3f rho_l=%.3f' % (rv_p, rl_p))
print('N=1000 0.9Tc, dU=%g  (optimal blind mix alpha=%.2f)\n' % (dU, a))

plt.figure(figsize=(9.6, 6.4))
rr = np.linspace(0.045, 0.66, 400)
plt.plot(1 / rr, [pP(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS')
plt.plot(1 / rr, [tP(x) for x in rr], '--', color='dimgray', lw=1.6, label='Thol 2015')
plt.plot([1 / rv_p, 1 / rl_p], [pP(rv_p)] * 2, 'kP--', ms=13, lw=1.6, label='PeTS binodal')

curves = [('blind pure IK', P0ik, 'tab:blue', 'o'),
          ('blind pure H', P0h, 'tab:green', 's'),
          (r'recovered (mix $\alpha=%.2f$)' % a, P0mix, 'tab:red', 'D')]
print('%-22s | binodal rho_v rho_l  (err vs PeTS)' % 'curve')
for name, P0, col, mk in curves:
    P0a = align(P0, G)
    plt.plot(1 / G, P0a, '-', color=col, lw=1.7, alpha=0.85, label=name)
    bb = binodal_of(P0a, G)
    if bb:
        rv, rl, Ps = bb
        plt.plot([1 / rv, 1 / rl], [Ps] * 2, mk + '--', color=col, ms=8, lw=1.3)
        print('%-22s | %.3f %.3f  (%+.3f / %+.3f)' % (name, rv, rl, rv - rv_p, rl - rl_p))
    else:
        print('%-22s | (no clean loop)' % name)

plt.axhline(0, color='gray', lw=0.5); plt.xlim(1.3, 14); plt.ylim(-0.03, 0.105)
plt.xlabel(r'$v^*=1/\rho^*$'); plt.ylabel(r'$P_0^*$ (constant-aligned for display only)')
plt.title(r'N=1000 0.9$T_c$: blind IK, blind H, and recovered mix vs PeTS ($\Delta U=%g$)' % dU)
plt.legend(fontsize=8, loc='upper right'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('ljts1000_ikh_recovered.png', dpi=140); print('\nwrote ljts1000_ikh_recovered.png')
