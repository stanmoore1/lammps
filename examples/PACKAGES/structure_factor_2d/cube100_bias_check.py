#!/usr/bin/env python3
"""Bias diagnostics for the contour mu0 fits.  The mu0 pipeline (smooth -> polyfit
P0(rho) -> d/drho -> integrate -> least-squares anchor) is NONLINEAR, so it can bias
the estimate.  We separate three things for IK and Harasima on one run:
  (1) bootstrap bias  = <mu0_boot> - mu0_point   (nonlinear-estimator bias; should be
      << bootstrap sigma if the pipeline is unbiased),
  (2) model bias      = spread of mu0 over polynomial degree 3..6,
  (3) smoothing bias  = spread of mu0 over cosine modes / savgol,
and compare all to the physical residual vs PeTS (the contour-truncation bias).
Usage: cube100_bias_check.py <tag> <dumax> <T>"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, contour_pressure as cp, pets_eos as pets
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else 'cube100Tc2'
dumax = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
T = float(sys.argv[3]) if len(sys.argv) > 3 else 1.089
L = 6.8582414181223398941; Lz = L; area = L * L
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
sc = np.linspace(0.16, 0.50, 40); mp = np.array([pmu(x) for x in sc])
rng = np.random.default_rng(0); GRID = sc


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


def smooth(y, method='cos', par=6):
    return savgol_filter(y, par, 3, mode='wrap') if method == 'sg' else oz.fourier_cosine_smooth(y, par)


def p0_profile(ik, hst, hde, which, method='cos', par=6):
    if which == 'IK':
        rho = smooth(ik[:, 2], method, par)
        PN = smooth(ik[:, 5] + ik[:, 8], method, par)
        PT = smooth(0.5 * ((ik[:, 3] + ik[:, 6]) + (ik[:, 4] + ik[:, 7])), method, par)
    else:
        nb = len(hst); Vbin = area * (Lz / nb)
        rho = smooth(hde[:, 3], method, par)
        PN = smooth(-hst[:, 5] / Vbin, method, par) + rho * T
        PT = smooth(-0.5 * (hst[:, 3] + hst[:, 4]) / Vbin, method, par) + rho * T
    return rho, 1.5 * PT - 0.5 * PN


def mu0_from(rho, P0, deg=4):
    z = (np.arange(len(rho)) + 0.5) / len(rho) * Lz; m = z <= Lz / 2
    r, p = rho[m], P0[m]; o = np.argsort(r); r, p = r[o], p[o]
    msk = (r > 0.10) & (r < 0.56); r, p = r[msk], p[msk]
    if len(r) < deg + 2:
        return None
    cf = np.polyfit(r, p, deg); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(max(r.min(), 0.12), min(r.max(), 0.55), 120)
    mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
    mu += np.mean(mp - np.interp(sc, rg, mu))           # least-squares anchor
    return np.interp(GRID, rg, mu, left=np.nan, right=np.nan)


ikb = read_vec_blocks('%s_ikstress.out' % tag)
hb = read_chunk_blocks('%s_hstress.out' % tag)
db = read_chunk_blocks('%s_dens.out' % tag)
nb = min(len(ikb), len(hb), len(db)); ikb, hb, db = ikb[:nb], hb[:nb], db[:nb]
print('%s (dUmax=%g, T=%g): %d blocks\n' % (tag, dumax, T, nb))
muP = np.array([pmu(x) for x in GRID])

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for which, col, axi in [('IK', 'tab:blue', 0), ('Harasima', 'tab:green', 1)]:
    rp, P0p = p0_profile(np.mean(ikb, 0), np.mean(hb, 0), np.mean(db, 0), which)
    mu_pt = mu0_from(rp, P0p)
    # (1) bootstrap bias
    boot = []
    for _ in range(400):
        i = rng.integers(0, nb, nb)
        r, P0 = p0_profile(np.mean([ikb[k] for k in i], 0), np.mean([hb[k] for k in i], 0),
                           np.mean([db[k] for k in i], 0), which)
        m = mu0_from(r, P0)
        if m is not None:
            boot.append(m)
    boot = np.array(boot); mu_mean = np.nanmean(boot, 0); sig = np.nanstd(boot, 0)
    bbias = mu_mean - mu_pt
    # (2) model (polynomial-degree) bias
    degs = np.array([mu0_from(rp, P0p, d) for d in (3, 4, 5, 6)])
    model_spread = np.nanmax(degs, 0) - np.nanmin(degs, 0)
    # (3) smoothing bias
    smv = []
    for meth, par in [('cos', 4), ('cos', 6), ('cos', 8), ('sg', 9), ('sg', 15)]:
        r, P0 = p0_profile(np.mean(ikb, 0), np.mean(hb, 0), np.mean(db, 0), which, meth, par)
        smv.append(mu0_from(r, P0))
    smv = np.array(smv); smooth_spread = np.nanmax(smv, 0) - np.nanmin(smv, 0)
    resid = mu_pt - muP
    print('%s:' % which)
    print('  bootstrap bias   : mean|bias|=%.4f   median sigma=%.4f   -> |bias|/sigma=%.2f'
          % (np.nanmean(np.abs(bbias)), np.nanmedian(sig), np.nanmean(np.abs(bbias)) / np.nanmedian(sig)))
    print('  poly-degree bias : mean spread=%.4f (deg 3-6)' % np.nanmean(model_spread))
    print('  smoothing bias   : mean spread=%.4f (cos4-8, sg9-15)' % np.nanmean(smooth_spread))
    print('  EOS residual     : mean=%+.4f  rms=%.4f  (sign-consistent? %s)'
          % (np.nanmean(resid), np.sqrt(np.nanmean(resid**2)),
             'YES->bias' if abs(np.nanmean(resid)) > 0.5 * np.sqrt(np.nanmean(resid**2)) else 'no->mostly shape'))
    print()
    ax[axi].axhline(0, color='k', lw=0.8)
    ax[axi].plot(GRID, resid, '-', color=col, lw=2.2, label='EOS residual (truncation bias)')
    ax[axi].plot(GRID, bbias, '--', color='tab:red', lw=1.8, label='bootstrap bias')
    ax[axi].fill_between(GRID, -sig, sig, color='gray', alpha=0.25, label=r'$\pm$bootstrap $\sigma$')
    ax[axi].fill_between(GRID, -model_spread, model_spread, color='tab:orange', alpha=0.2, label='poly-degree spread')
    ax[axi].set_title('%s contour' % which); ax[axi].set_xlabel(r'$\rho^*$'); ax[axi].set_ylabel(r'$\mu_0$ bias')
    ax[axi].legend(fontsize=8); ax[axi].grid(alpha=0.3)
plt.suptitle('Bias diagnostics: %s (dUmax=%g, T=%g)' % (tag, dumax, T))
plt.tight_layout(); plt.savefig('%s_bias.png' % tag, dpi=140)
print('wrote %s_bias.png' % tag)
