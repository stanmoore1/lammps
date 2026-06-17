#!/usr/bin/env python3
"""Fourth-order contour gauge (beyond the IK/H linear combination).

The second-order gauge  g2(z) = 3/2 P_T(z) - 1/2 P_N(z)  cancels the influence-
parameter (kappa) term, so g2_IK and g2_H both equal P0 + O(grad^4).  Their
difference  D(z) = g2_IK(z) - g2_H(z)  is therefore a *measured* pure >=4th-order
gradient field, and the linear combination

    P0(rho) = g2_IK(rho) - alpha0 * D(rho)                                (4th order)

is the exact 4th-order gauge IF alpha0 is a true constant.  The contour's only
remaining weakness is the extreme tails (rho<0.10, rho>0.57), where the gradient is
steepest and the NEXT order (6th) survives.  We add it with a square-gradient-
weighted coefficient,

    P0(rho) = g2_IK(rho) - (alpha0 + alpha1 * s(rho)) * D(rho),          (4th+6th)

s = (rho')^2 the local square-gradient (largest exactly in the tails, so alpha1
acts there).  alpha0, alpha1 are fit EOS-BLIND from the FIELD LADDER: the same rho
occurs at two field strengths (dUmax=2 and 4) with different gradient, but the true
P0(rho) is field-independent, so we choose (alpha0,alpha1) to make the two corrected
curves coincide.  Block-bootstrap error bars; compared to PeTS and Thol 2015.
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz
import pets_eos as pets
import thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.198; L = 6.8582414181223398941; Lz = L; area = L * L
SM = 6
GRID = np.linspace(0.09, 0.595, 140)          # common rho grid for interpolation
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T * np.log(r) + thol.properties(T, r)['mu_res']
sc = np.linspace(0.14, 0.55, 40)
mp = np.array([pmu(x) for x in sc]); mt = np.array([tmu(x) for x in sc])
rng = np.random.default_rng(0)


def read_chunk_blocks(fn):
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


def read_vec_blocks(fn):
    blocks, cur = [], None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 2:
            cur = []; blocks.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return [np.array(b) for b in blocks]


def _half_to_grid(rho_z, y_z):
    """Map a z-profile y(z) onto GRID via the monotonic lower half rho(z), z<=Lz/2."""
    nb = len(rho_z); z = (np.arange(nb) + 0.5) * Lz / nb
    m = z <= Lz / 2
    r, y = rho_z[m], y_z[m]
    o = np.argsort(r)
    return np.interp(GRID, r[o], y[o], left=np.nan, right=np.nan)


def field_profiles(ik, hst, hde):
    """Mean per-field arrays -> g2_IK(rho), g2_H(rho), s(rho)=(rho')^2 on GRID."""
    # IK (stress/cartesian): kinetic+config already included
    rik = oz.fourier_cosine_smooth(ik[:, 2], SM)
    PNik = oz.fourier_cosine_smooth(ik[:, 5] + ik[:, 8], SM)
    PTik = oz.fourier_cosine_smooth(0.5 * ((ik[:, 3] + ik[:, 6]) + (ik[:, 4] + ik[:, 7])), SM)
    g2ik = _half_to_grid(rik, 1.5 * PTik - 0.5 * PNik)
    # Harasima (per-atom virial) + ideal rho*kT
    nb = len(hst); Vbin = area * (Lz / nb)
    rh = oz.fourier_cosine_smooth(hde[:, 3], SM)
    PNh = oz.fourier_cosine_smooth(-hst[:, 5] / Vbin, SM) + rh * T
    PTh = oz.fourier_cosine_smooth(-0.5 * (hst[:, 3] + hst[:, 4]) / Vbin, SM) + rh * T
    g2h = _half_to_grid(rh, 1.5 * PTh - 0.5 * PNh)
    # square-gradient s(rho) from the density profile
    drho = oz.fourier_cosine_deriv(rh, SM, Lz)
    s = _half_to_grid(rh, drho ** 2)
    return g2ik, g2h, s


def load(tag):
    ik = read_vec_blocks('%s_ikstress.out' % tag)
    hs = read_chunk_blocks('%s_hstress.out' % tag)
    de = read_chunk_blocks('%s_dens.out' % tag)
    return ik, hs, de


u2 = load('cube100'); u4 = load('cube100u4')
p2 = field_profiles(np.mean(u2[0], 0), np.mean(u2[1], 0), np.mean(u2[2], 0))
p4 = field_profiles(np.mean(u4[0], 0), np.mean(u4[1], 0), np.mean(u4[2], 0))
SNORM = np.nanmax(np.concatenate([p2[2], p4[2]]))      # normalize s -> alpha1 ~ O(1)


def ladder_fit(p2, p4):
    """EOS-blind least squares: make P0_corr(u4)=P0_corr(u2) on the overlap.
    P0_corr_f = g2ik_f - (a0 + a1 s_f) D_f ;  fit (a0,a1)."""
    g2ik2, g2h2, s2 = p2; g2ik4, g2h4, s4 = p4
    D2, D4 = g2ik2 - g2h2, g2ik4 - g2h4
    sn2, sn4 = s2 / SNORM, s4 / SNORM
    A = g2ik4 - g2ik2                                   # known mismatch
    B = D4 - D2                                         # alpha0 regressor
    C = sn4 * D4 - sn2 * D2                             # alpha1 (6th-order) regressor
    m = np.isfinite(A) & np.isfinite(B) & np.isfinite(C)
    m &= (GRID > 0.12) & (GRID < 0.575)
    M = np.vstack([B[m], C[m]]).T
    a, *_ = np.linalg.lstsq(M, A[m], rcond=None)        # minimize ||A - a0 B - a1 C||
    return a[0], a[1]


def corrected_P0(pf, a0, a1):
    g2ik, g2h, s = pf
    return g2ik - (a0 + a1 * s / SNORM) * (g2ik - g2h)


def p0_to_mu0(P0):
    m = np.isfinite(P0) & (GRID > 0.115) & (GRID < 0.575)
    rs, ps = GRID[m], P0[m]
    cf = np.polyfit(rs, ps, 4); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(max(rs.min(), 0.12), min(rs.max(), 0.57), 120)
    mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
    return rg, mu


def anchor(rg, mu, ref):
    mua = mu + (np.interp(np.median(sc), sc, ref) - np.interp(np.median(sc), rg, mu))
    return mua, np.sqrt(np.mean((np.interp(sc, rg, mua) - ref) ** 2))


a0, a1 = ladder_fit(p2, p4)
print('ladder fit (EOS-blind):  alpha0=%.3f  alpha1=%.3f  (alpha0 only: linear combo a=%.2f)'
      % (a0, a1, 1 - a0))

# --- build the curves on GRID ---
# pure IK contour
P0_ik = 0.5 * (p2[0] + p4[0])
# 4th-order gauge (ladder alpha0, alpha1=0), field-averaged
P0_4 = 0.5 * (corrected_P0(p2, a0, 0.0) + corrected_P0(p4, a0, 0.0))
# 4th+6th-order gauge, field-averaged
P0_46 = 0.5 * (corrected_P0(p2, a0, a1) + corrected_P0(p4, a0, a1))

results = {}
for lab, P0 in [('IK contour', P0_ik), ('4th-order gauge', P0_4), ('4th+6th gauge', P0_46)]:
    rg, mu = p0_to_mu0(P0)
    mua_p, rms_p = anchor(rg, mu, mp)
    _, rms_t = anchor(rg, mu, mt)
    results[lab] = (rg, mua_p, rms_p, rms_t)
    print('  %-16s mu0 RMS vs PeTS=%.4f  vs Thol=%.4f' % (lab, rms_p, rms_t))

# --- block-bootstrap band on the 4th+6th gauge (alpha fixed) ---
nboot = 200
boot = []
for _ in range(nboot):
    def res(blocks):
        i = rng.integers(0, len(blocks), len(blocks))
        return np.mean([blocks[k] for k in i], 0)
    b2 = field_profiles(res(u2[0]), res(u2[1]), res(u2[2]))
    b4 = field_profiles(res(u4[0]), res(u4[1]), res(u4[2]))
    P0b = 0.5 * (corrected_P0(b2, a0, a1) + corrected_P0(b4, a0, a1))
    try:
        rg, mu = p0_to_mu0(P0b); mua, _ = anchor(rg, mu, mp)
        boot.append(np.interp(GRID, rg, mua, left=np.nan, right=np.nan))
    except Exception:
        pass
boot = np.array(boot)
with np.errstate(all='ignore'):
    band = np.nanstd(boot, 0)
print('block-bootstrap sigma on 4th+6th gauge: median=%.4f (statistical error; cf. systematic RMS above)'
      % np.nanmedian(band))
rg46, mua46 = results['4th+6th gauge'][0], results['4th+6th gauge'][1]
band_on = np.interp(GRID, rg46, mua46, left=np.nan, right=np.nan)

# --- plot ---
plt.figure(figsize=(9, 6))
rr = np.linspace(0.06, 0.605, 250)
plt.plot(rr, [pmu(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS')
plt.plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2.0, label='Thol 2015 EOS')
cols = {'IK contour': 'tab:blue', '4th-order gauge': 'tab:orange', '4th+6th gauge': 'tab:red'}
for lab in ['IK contour', '4th-order gauge', '4th+6th gauge']:
    rg, mua, rms_p, rms_t = results[lab]
    lw = 2.6 if lab == '4th+6th gauge' else 1.8
    plt.plot(rg, mua, 'o-', ms=3, lw=lw, color=cols[lab],
             label='%s (PeTS %.3f / Thol %.3f)' % (lab, rms_p, rms_t))
m = np.isfinite(band) & np.isfinite(band_on)
plt.fill_between(GRID[m], band_on[m] - band[m], band_on[m] + band[m], color='tab:red', alpha=0.22,
                 label='block-bootstrap $\\pm1\\sigma$')
# mark the tail regions the gauge is meant to fix
for x in (0.10, 0.57):
    plt.axvline(x, color='gray', ls=':', lw=0.8)
plt.text(0.075, -2.05, 'tail', color='gray', fontsize=8)
plt.text(0.585, -2.05, 'tail', color='gray', fontsize=8)
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored)')
plt.title(r'Fourth-order contour gauge ($\Delta U$ ladder 2+4): fixing the tails (N=100, $T^*=1.198$)')
plt.legend(fontsize=8, loc='upper left'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('cube100_contour4.png', dpi=140)
print('wrote cube100_contour4.png')
