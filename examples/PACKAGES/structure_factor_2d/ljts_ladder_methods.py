#!/usr/bin/env python3
"""Fit the LADDER EOS-recovery methods on the subcritical 0.9 Tc data (T*=0.980,
field ladder dUmax=0.2,0.4,0.8) and compare the recovered mu0(rho)/P0(rho) to the
PeTS and Thol-2015 vdW loops:
  (1) field-coupling gradient ladder  (field_coupling.local_eos -- pools rho(z))
  (2) field-coupling kernel ladder    (kernel_fit.kernel_eos    -- nonlocal C(s))
  (3) 4th-order contour gauge ladder  (EOS-blind alpha0,alpha1 from cross-field
      consistency of P0 = 3/2 P_T - 1/2 P_N), with Savitzky-Golay smoothing.
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, contour_pressure as cp, field_coupling as fc, kernel_fit as kf
import pets_eos as pets, thol2015_ljts_eos as thol
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 0.980
Lx = 10.244851881402800231; Lz = 30.734555644208398917; area = Lx * Lx
SGW, SGP = 15, 3
LAD = [('T0.980_d0.2', 0.2), ('T0.980_d0.4', 0.4), ('T0.980_d0.8', 0.8)]
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T * np.log(r) + thol.properties(T, r)['mu_res']
pP = lambda r: pets.properties(T, r)['p']
tP = lambda r: thol.properties(T, r)['p']
sg = lambda y: savgol_filter(y, SGW, SGP, mode='wrap')
GRID = np.linspace(0.06, 0.63, 130)


def anchor_mu(rg, mu):
    """absolute mu0 from a loop: pin the integration constant to PeTS at rho=0.12
    (a single, low-noise vapor-side density)."""
    return mu + (pmu(0.12) - np.interp(0.12, rg, mu))


# ---------- density profiles (savgol) for the FC methods ----------
dens = {t: sg(cp.read_chunk('%s_dens.out' % t)[:, 3]) for t, _ in LAD}
nb = len(next(iter(dens.values())))
amps = np.array([0.0] + [dU / 2 for _, dU in LAD])
prof = np.array([np.full(nb, np.mean([d.mean() for d in dens.values()]))] + [dens[t] for t, _ in LAD])

res = {}
# (1) FC gradient ladder  (mu0 carries the usual CPP anchoring constant -> anchor it)
try:
    e = fc.local_eos(amps, prof, T, Lz, deg=7, smooth=12, grad_spec={2: 0, 4: 0})
    res['FC-gradient ladder'] = (e['rho'], anchor_mu(e['rho'], e['mu0']), e['P0'], 'tab:orange')
    print('(1) FC-gradient ladder: fit ok')
except Exception as ex:
    print('(1) FC-gradient failed:', ex)
# (2) FC kernel ladder
try:
    ek = kf.kernel_eos(amps, prof, T, Lz, deg=7, smax=2.5, nmodes=3, ridge=1e-3, smooth=12)
    res['FC-kernel ladder'] = (ek['rho'], anchor_mu(ek['rho'], ek['mu0']), ek['P0'], 'tab:purple')
    print('(2) FC-kernel ladder: fit ok')
except Exception as ex:
    print('(2) FC-kernel failed:', ex)


# ---------- (3) 4th-order contour gauge ladder (savgol) ----------
def half_to_grid(rho_z, y_z):
    z = (np.arange(len(rho_z)) + 0.5) / len(rho_z) * Lz
    m = z <= Lz / 2
    r, y = rho_z[m], y_z[m]; o = np.argsort(r)
    return np.interp(GRID, r[o], y[o], left=np.nan, right=np.nan)


def gauge_profiles(tag):
    ik = cp.read_avetime_vec('%s_ikstress.out' % tag)
    rik = sg(ik[:, 2]); PNik = sg(ik[:, 5] + ik[:, 8])
    PTik = sg(0.5 * ((ik[:, 3] + ik[:, 6]) + (ik[:, 4] + ik[:, 7])))
    g2ik = half_to_grid(rik, 1.5 * PTik - 0.5 * PNik)
    de = cp.read_chunk('%s_dens.out' % tag); st = cp.read_chunk('%s_hstress.out' % tag)
    Vbin = area * (Lz / len(de)); rh = sg(de[:, 3])
    PNh = sg(-st[:, 5] / Vbin) + rh * T; PTh = sg(-0.5 * (st[:, 3] + st[:, 4]) / Vbin) + rh * T
    g2h = half_to_grid(rh, 1.5 * PTh - 0.5 * PNh)
    drho = np.gradient(rh, Lz / len(rh))
    s = half_to_grid(rh, drho ** 2)
    return g2ik, g2h, s


gp = [gauge_profiles(t) for t, _ in LAD]
SN = np.nanmax(np.concatenate([p[2] for p in gp]))


def ladder_fit(profs):
    a = np.array([p[0] for p in profs]); D = np.array([p[0] - p[1] for p in profs])
    C = np.array([(p[2] / SN) * (p[0] - p[1]) for p in profs])
    fin = np.isfinite(a) & np.isfinite(D) & np.isfinite(C)
    msk = (GRID > 0.10) & (GRID < 0.60)
    def ctr(X):
        return X - np.nanmean(np.where(fin, X, np.nan), 0)
    A, B, Cc = ctr(a), ctr(D), ctr(C)
    sel = fin.all(0)[None, :] & msk[None, :] & np.isfinite(A) & np.isfinite(B) & np.isfinite(Cc)
    M = np.vstack([B[sel], Cc[sel]]).T
    sol, *_ = np.linalg.lstsq(M, A[sel], rcond=None)
    return sol[0], sol[1]


a0, a1 = ladder_fit(gp)
print('(3) contour gauge ladder fit (EOS-blind): alpha0=%.3f alpha1=%.3f' % (a0, a1))


def p0grid_to_mu0(P0grid):
    """mu0 by Gibbs-Duhem in rho-space through the loop (deg-6 fit of P0(rho))."""
    m = np.isfinite(P0grid) & (GRID > 0.09) & (GRID < 0.61)
    cf = np.polyfit(GRID[m], P0grid[m], 6); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(GRID[m].min(), GRID[m].max(), 200)
    mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
    return rg, anchor_mu(rg, mu), np.polyval(cf, rg)


# field-averaged pure contours (the alpha=1 / alpha=0 endpoints) + the ladder gauge
P0ik = np.nanmean(np.array([p[0] for p in gp]), 0)
P0h = np.nanmean(np.array([p[1] for p in gp]), 0)
P0g = np.nanmean(np.array([p[0] - (a0 + a1 * p[2] / SN) * (p[0] - p[1]) for p in gp]), 0)
res['IK contour'] = (*p0grid_to_mu0(P0ik), 'tab:blue')
res['H contour'] = (*p0grid_to_mu0(P0h), 'tab:green')
res['4th-order contour gauge'] = (*p0grid_to_mu0(P0g), 'tab:red')


def rms_mu(rg, mu):
    mm = (rg > 0.12) & (rg < 0.58)
    return np.sqrt(np.mean((mu[mm] - np.array([pmu(x) for x in rg[mm]])) ** 2))


# ---------- plots: mu0(rho) and P0(rho) loops ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
rr = np.linspace(0.04, 0.65, 300)
ax[0].plot(rr, [pmu(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS')
ax[0].plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2, label='Thol 2015 EOS')
ax[1].plot(rr, [pP(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS')
ax[1].plot(rr, [tP(x) for x in rr], '--', color='dimgray', lw=2, label='Thol 2015 EOS')
ax[1].axhline(0, color='gray', lw=0.6)
order = ['IK contour', 'H contour', 'FC-gradient ladder', 'FC-kernel ladder', '4th-order contour gauge']
for lab in [l for l in order if l in res]:
    rg, mu, P0, col = res[lab]
    r = rms_mu(rg, mu)
    pure = 'contour' in lab and 'gauge' not in lab          # raw IK / H endpoints
    ls, lw, al = (('--', 1.6, 0.8) if pure else ('-', 2.4, 1.0))
    ax[0].plot(rg, mu, ls, color=col, lw=lw, alpha=al, label='%s (mu0 RMS %.3f)' % (lab, r))
    ax[1].plot(rg, P0, ls, color=col, lw=lw, alpha=al, label=lab)
    print('   %-26s mu0 RMS vs PeTS = %.4f' % (lab, r))
ax[0].set_xlabel(r'$\rho^*$'); ax[0].set_ylabel(r'$\mu_0^*$'); ax[0].set_title(r'0.9 $T_c$: ladder-method $\mu_0(\rho)$')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set_xlabel(r'$\rho^*$'); ax[1].set_ylabel(r'$P_0^*$'); ax[1].set_title(r'0.9 $T_c$: ladder-method $P_0(\rho)$ (vdW loop)')
ax[1].set_ylim(-0.06, 0.18); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('ljts_ladder_methods.png', dpi=140)
print('wrote ljts_ladder_methods.png')
