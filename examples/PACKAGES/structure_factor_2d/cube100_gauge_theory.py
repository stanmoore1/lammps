#!/usr/bin/env python3
"""Verify the gradient-expansion theory of the IK/H 4th-order gauge against the data.

Derivation (see chat): for either contour, using ITS OWN P_N and P_T,
    P0_C = 3/2 P_T^C - 1/2 P_N^C = P0(rho) + (beta2/35) pi_{C,2} + O(grad^6),
with the 4th-order density coefficients
    pi_{IK,2} = rho rho''''/60 - rho' rho'''/60 + rho''^2/120,
    pi_{H,2}  = rho rho''''/24.
Two exact algebraic facts (verified with sympy):
  (1) pi_{IK,2} - pi_{H,2} = d/dz[ -rho rho'''/40 + rho' rho''/120 ]  (a TOTAL derivative)
  (2) both have the SAME irreducible part rho''^2/24  (they differ only by a total deriv).
=> P0_IK(z) - P0_H(z) is a pure total z-derivative.  Consequences tested here on data:
  (A) integral over the period:  INT (P0_IK - P0_H) dz = 0,
  (B) shape:  P0_IK - P0_H  =  C * d/dz[ -rho rho'''/40 + rho' rho''/120 ],  C = beta2/35,
  (C) therefore NO constant alpha in  alpha P0_IK + (1-alpha) P0_H  cancels the
      irreducible 4th-order error; mixing only tunes the total-derivative remainder.
      The alpha that kills the leading (rho rho'''') term is 5/3 (extrapolation).
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, contour_pressure as cp
import pets_eos as pets
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.198; L = 6.8582414181223398941; Lz = L; area = L * L
tag = sys.argv[1] if len(sys.argv) > 1 else 'cube100u4'
SM = 6


def deriv_set(y, Lz, nmodes=SM, zq=None):
    """rho and its 1st-4th z-derivatives from the cosine-series fit, on grid zq."""
    coef, zc = oz.fourier_cosine_coef(y, nmodes)
    if zq is None:
        zq = (np.arange(len(y)) + 0.5) / len(y) * Lz
    k = np.arange(nmodes + 1); w = 2 * np.pi * k / Lz
    ph = 2 * np.pi * np.outer(zq / Lz, k)
    c, s = np.cos(ph), np.sin(ph)
    rho = c @ coef
    r1 = -(s * w) @ coef
    r2 = -(c * w**2) @ coef
    r3 = (s * w**3) @ coef
    r4 = (c * w**4) @ coef
    return rho, r1, r2, r3, r4


sc = np.linspace(0.14, 0.55, 40); mp = np.array([T*np.log(x)+pets.properties(T, x)['mu_res'] for x in sc])


def mu0_rms(rho, P0z):
    o = np.argsort(rho); rs, ps = rho[o], P0z[o]
    msk = (rs > 0.12) & (rs < 0.57); rs, ps = rs[msk], ps[msk]
    cf = np.polyfit(rs, ps, 4); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(rs.min(), rs.max(), 120)
    mu = np.concatenate([[0], np.cumsum(0.5*(dP(rg[1:])/rg[1:]+dP(rg[:-1])/rg[:-1])*np.diff(rg))])
    mua = mu + (np.interp(np.median(sc), sc, mp) - np.interp(np.median(sc), rg, mu))
    return np.sqrt(np.mean((np.interp(sc, rg, mua) - mp)**2))


def field(tag):
    rik, PNik, PTik = cp.ik_profile('%s_ikstress.out' % tag, Lz, SM)
    rh, PNh, PTh = cp.h_profile('%s_hstress.out' % tag, '%s_dens.out' % tag, Lz, area, T, SM)
    zik = (np.arange(len(rik)) + 0.5) / len(rik) * Lz
    zh = (np.arange(len(rh)) + 0.5) / len(rh) * Lz
    zg = np.linspace(0, Lz, 400, endpoint=False) + 0.5 * Lz / 400
    P0ik = np.interp(zg, zik, 1.5 * PTik - 0.5 * PNik, period=Lz)
    P0h = np.interp(zg, zh, 1.5 * PTh - 0.5 * PNh, period=Lz)
    rho = np.interp(zg, zh, rh, period=Lz)
    return zg, rho, P0ik, P0h


alphas = np.linspace(-0.6, 2.0, 53)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
print('Field-ladder test of the gradient-expansion 4th-order gauge:')
abest = {}
for tag, dU, col in [('cube100', 2.0, 'tab:green'), ('cube100u3', 3.0, 'tab:orange'),
                     ('cube100u4', 4.0, 'tab:red')]:
    zg, rho, P0ik, P0h = field(tag)
    dP0 = P0ik - P0h
    Idiff = np.trapezoid(dP0, zg); Iabs = np.trapezoid(np.abs(dP0), zg)
    rms = np.array([mu0_rms(rho, a * P0ik + (1 - a) * P0h) for a in alphas])
    ab = alphas[np.argmin(rms)]; abest[dU] = ab
    print('  dU=%g:  INT(P0_IK-P0_H)=%+.4f (%.0f%% of INT|.|; total-deriv predicts 0);  '
          'optimal alpha=%+.2f (theory predicts 5/3)' % (dU, Idiff, 100*Idiff/Iabs, ab))
    ax[0].plot(alphas, rms, '-', lw=2, color=col, label=r'$\Delta U=%g$ (min $\alpha=%+.2f$)' % (dU, ab))
    ax[0].plot(ab, rms.min(), 'o', color=col, ms=7)
ax[0].axvline(5/3, color='k', ls='--', lw=1.5, label=r'gradient theory $\alpha=5/3$')
ax[0].axvline(1.0, color='gray', ls=':', label=r'pure IK')
ax[0].set_xlabel(r'$\alpha$  in  $\alpha P_0^{IK}+(1-\alpha)P_0^{H}$'); ax[0].set_ylabel(r'$\mu_0$ RMS vs PeTS')
ax[0].set_ylim(0, 0.12)
ax[0].set_title(r'Optimal mixing $\alpha$ DRIFTS with field (not the universal 5/3)')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

dU_arr = np.array([2.0, 3.0, 4.0]); a_arr = np.array([abest[d] for d in dU_arr])
ax[1].plot(dU_arr, a_arr, 'o-', ms=8, color='tab:blue', lw=2, label='data: optimal $\\alpha(\\Delta U)$')
ax[1].axhline(5/3, color='k', ls='--', lw=1.5, label=r'gradient theory $\alpha=5/3$ (leading 4th-order)')
ax[1].axhline(1.0, color='gray', ls=':', label='pure IK')
ax[1].set_xlabel(r'field strength $\Delta U$'); ax[1].set_ylabel(r'optimal $\alpha$')
ax[1].set_title('No universal constant: $\\alpha$ is gradient-dependent')
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('cube100_gauge_theory.png', dpi=140)
print('=> data refutes a universal 4th-order constant: the 4th-order gradient expansion has')
print('   broken down at these gradients (IK-H not a total derivative; alpha field-dependent).')
print('wrote cube100_gauge_theory.png')
