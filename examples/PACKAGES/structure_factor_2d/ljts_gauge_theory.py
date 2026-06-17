#!/usr/bin/env python3
"""Redo the IK/H 4th-order-gauge gradient-expansion test on the SUBCRITICAL
0.9 Tc data (T*=0.980, LJTS N=1000 two-phase slabs, field ladder dUmax=0.2,0.4,0.8).

Same two falsifiable predictions of the strict 4th-order gradient expansion
(derived + sympy-verified for the supercritical set):
  (A) P0_IK(z) - P0_H(z) is a pure total z-derivative   ->  INT over the box = 0.
  (B) the optimal mixing in  alpha P0_IK + (1-alpha) P0_H  is the UNIVERSAL constant
      alpha = 5/3 (cancels the leading rho rho'''' term).
Here the interface is a real liquid-vapor interface (intrinsic width ~1-2 sigma),
NOT set by the field -- so the gradient is similar across the ladder, a sharper test
of whether the expansion holds.  P0 is an absolute pressure, so we score each contour
mix directly against the PeTS vdW loop P(rho) (no anchor, no mu0 integration through
the loop).  Compared also to Thol 2015.
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, contour_pressure as cp
import pets_eos as pets, thol2015_ljts_eos as thol
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 0.980
Lx = 10.244851881402800231; Lz = 30.734555644208398917; area = Lx * Lx
# Subcritical profiles have flat liquid/vapor plateaus + a sharp interface, so the
# GLOBAL cosine basis rings (Gibbs).  Use a LOCAL Savitzky-Golay filter instead.
SGW = int(sys.argv[1]) if len(sys.argv) > 1 else 15     # savgol window (odd # of bins)
SGP = 3                                                 # savgol polynomial order
LADDER = [('T0.980_d0.2', 0.2, 'tab:green'), ('T0.980_d0.4', 0.4, 'tab:orange'),
          ('T0.980_d0.8', 0.8, 'tab:red')]
pP = lambda r: pets.properties(T, r)['p']
tP = lambda r: thol.properties(T, r)['p']


def sg(y):
    return savgol_filter(y, SGW, SGP, mode='wrap')      # periodic, local -> no Gibbs ringing


def field(tag):
    ik = cp.read_avetime_vec('%s_ikstress.out' % tag)   # IK: stress/cartesian (kin+conf)
    rik = sg(ik[:, 2])
    PNik = sg(ik[:, 5] + ik[:, 8])
    PTik = sg(0.5 * ((ik[:, 3] + ik[:, 6]) + (ik[:, 4] + ik[:, 7])))
    de = cp.read_chunk('%s_dens.out' % tag); st = cp.read_chunk('%s_hstress.out' % tag)
    nb = len(de); Vbin = area * (Lz / nb)
    rh = sg(de[:, 3])                                    # Harasima: per-atom virial + rho*kT
    PNh = sg(-st[:, 5] / Vbin) + rh * T
    PTh = sg(-0.5 * (st[:, 3] + st[:, 4]) / Vbin) + rh * T
    zik = (np.arange(len(rik)) + 0.5) / len(rik) * Lz
    zh = (np.arange(nb) + 0.5) / nb * Lz
    zg = np.linspace(0, Lz, 600, endpoint=False) + 0.5 * Lz / 600
    P0ik = np.interp(zg, zik, 1.5 * PTik - 0.5 * PNik, period=Lz)
    P0h = np.interp(zg, zh, 1.5 * PTh - 0.5 * PNh, period=Lz)
    rho = np.interp(zg, zh, rh, period=Lz)
    return zg, rho, P0ik, P0h


def p0_rms(rho, P0z):
    """RMS of the contour pressure vs the PeTS EOS pressure at the local density,
    over the interfacial (loop) region where the contours actually differ."""
    m = (rho > 0.08) & (rho < 0.62)                 # drop the saturated plateaus' tails
    if m.sum() < 10:
        return np.nan
    return np.sqrt(np.mean((P0z[m] - np.array([pP(r) for r in rho[m]])) ** 2))


alphas = np.linspace(-1.0, 2.5, 71)
print('SUBCRITICAL 0.9 Tc (T*=0.980) gradient-expansion gauge test  [savgol w=%d p=%d]:' % (SGW, SGP))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
abest = {}
for tag, dU, col in LADDER:
    zg, rho, P0ik, P0h = field(tag)
    dP0 = P0ik - P0h
    Idiff = np.trapezoid(dP0, zg); Iabs = np.trapezoid(np.abs(dP0), zg)
    rms = np.array([p0_rms(rho, a * P0ik + (1 - a) * P0h) for a in alphas])
    ab = alphas[np.nanargmin(rms)]; abest[dU] = ab
    print('  dU=%g:  INT(P0_IK-P0_H)=%+.4f (%.0f%% of INT|.|; total-deriv predicts 0);  '
          'optimal alpha=%+.2f  | pure IK RMS=%.4f  pure H RMS=%.4f'
          % (dU, Idiff, 100 * Idiff / Iabs, ab, p0_rms(rho, P0ik), p0_rms(rho, P0h)))
    ax[0].plot(alphas, rms, '-', lw=2, color=col, label=r'$\Delta U=%g$ (min $\alpha=%+.2f$)' % (dU, ab))
    ax[0].plot(ab, np.nanmin(rms), 'o', color=col, ms=7)
ax[0].axvline(5 / 3, color='k', ls='--', lw=1.5, label=r'gradient theory $\alpha=5/3$')
ax[0].axvline(1.0, color='gray', ls=':', label='pure IK')
ax[0].set_xlabel(r'$\alpha$ in $\alpha P_0^{IK}+(1-\alpha)P_0^{H}$'); ax[0].set_ylabel(r'$P_0(\rho)$ RMS vs PeTS')
ax[0].set_title(r'0.9 $T_c$: optimal mixing $\alpha$ across the field ladder')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

dU_arr = np.array([d for _, d, _ in LADDER]); a_arr = np.array([abest[d] for d in dU_arr])
ax[1].plot(dU_arr, a_arr, 'o-', ms=9, color='tab:blue', lw=2, label=r'data: optimal $\alpha(\Delta U)$')
ax[1].axhline(5 / 3, color='k', ls='--', lw=1.5, label=r'gradient theory $\alpha=5/3$')
ax[1].axhline(1.0, color='gray', ls=':', label='pure IK')
ax[1].set_xlabel(r'field strength $\Delta U$'); ax[1].set_ylabel(r'optimal $\alpha$')
ax[1].set_title(r'0.9 $T_c$: is $\alpha$ universal?')
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('ljts_gauge_theory.png', dpi=140)
print('alpha(dU) =', dict((d, round(abest[d], 2)) for d in dU_arr))
print('wrote ljts_gauge_theory.png')

# --- payoff plot: the recovered vdW loop P0(rho) vs PeTS & Thol at 0.9 Tc ---
# P0(rho) is single-valued, so bin-average P0 over rho-bins (collapses the noisy
# liquid/vapor plateaus to clean points; the two interfaces reinforce each other).
def loop_binned(rho, P0, edges):
    idx = np.digitize(rho, edges) - 1
    rc = 0.5 * (edges[1:] + edges[:-1])
    m = np.array([P0[idx == k].mean() if np.any(idx == k) else np.nan for k in range(len(rc))])
    return rc, m


redges = np.linspace(0.03, 0.66, 46)
plt.figure(figsize=(8.5, 6))
rr = np.linspace(0.03, 0.66, 300)
plt.plot(rr, [pP(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS (vdW loop)')
plt.plot(rr, [tP(x) for x in rr], '--', color='dimgray', lw=2.0, label='Thol 2015 EOS')
plt.axhline(0, color='gray', lw=0.6)
for tag, dU, col in LADDER:
    zg, rho, P0ik, P0h = field(tag)
    rc, mik = loop_binned(rho, P0ik, redges)
    _, mh = loop_binned(rho, P0h, redges)
    plt.plot(rc, mik, '-', color=col, lw=1.8, alpha=0.95, label=r'IK contour, $\Delta U=%g$' % dU)
    plt.plot(rc, mh, ':', color=col, lw=1.8, alpha=0.95)
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$P_0^*$')
plt.title(r'0.9 $T_c$ ($T^*=0.980$): contour recovery of the van der Waals loop  (solid=IK, dotted=H)')
plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3); plt.ylim(-0.06, 0.18)
plt.tight_layout(); plt.savefig('ljts_vdw_loop.png', dpi=140)
print('wrote ljts_vdw_loop.png  (rho-binned)')
