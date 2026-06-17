#!/usr/bin/env python3
"""Field-ladder comparison: the Harasima-contour mu0(rho) at every rung dUmax=2,3,4.
Stronger field -> steeper density gradient -> larger 2nd-order contour error, so the
curves fan out systematically with field strength.  The dUmax->0 trend (and the
ladder-fit 4th-order gauge from cube100_contour4) is what removes that error.
Each curve carries its block-bootstrap band; PeTS and Thol 2015 shown."""
import sys
import os
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz
import pets_eos as pets, thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

TCSET = bool(os.environ.get('TCSET'))                  # TCSET=1 -> Tc=1.089 ladder
T = 1.089 if TCSET else 1.198; L = 6.8582414181223398941; Lz = L; area = L * L
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T * np.log(r) + thol.properties(T, r)['mu_res']
sc = np.linspace(0.14, 0.55, 40); mp = np.array([pmu(x) for x in sc])
rng = np.random.default_rng(0); GRID = np.linspace(0.12, 0.57, 80)


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


def h_mu0(st, de):
    nb = len(st); Vbin = area * (Lz / nb)
    rho = oz.fourier_cosine_smooth(de[:, 3], 6)
    PN = oz.fourier_cosine_smooth(-st[:, 5] / Vbin, 6) + rho * T
    PT = oz.fourier_cosine_smooth(-0.5 * (st[:, 3] + st[:, 4]) / Vbin, 6) + rho * T
    P0 = 1.5 * PT - 0.5 * PN
    o = np.argsort(rho); rs, ps = rho[o], P0[o]
    m = (rs > 0.11) & (rs < 0.58); rs, ps = rs[m], ps[m]
    cf = np.polyfit(rs, ps, 4); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(max(rs.min(), 0.12), min(rs.max(), 0.57), 120)
    mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
    return rg, mu


def anchor(rg, mu):
    mua = mu + (np.interp(np.median(sc), sc, mp) - np.interp(np.median(sc), rg, mu))
    return mua, np.sqrt(np.mean((np.interp(sc, rg, mua) - mp) ** 2))


plt.figure(figsize=(8.5, 6))
rr = np.linspace(0.06, 0.61, 250)
plt.plot(rr, [pmu(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS')
plt.plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2.0, label='Thol 2015 EOS')
_LAD = ([('cube100Tc2', 2.0, 'tab:green'), ('cube100Tc3', 3.0, 'tab:orange'), ('cube100Tc4', 4.0, 'tab:red')]
        if TCSET else
        [('cube100', 2.0, 'tab:green'), ('cube100u3', 3.0, 'tab:orange'), ('cube100u4', 4.0, 'tab:red')])
for tag, dU, col in _LAD:
    hb = read_chunk_blocks('%s_hstress.out' % tag); db = read_chunk_blocks('%s_dens.out' % tag)
    n = min(len(hb), len(db)); hb, db = hb[:n], db[:n]
    rg, mu = h_mu0(np.mean(hb, 0), np.mean(db, 0)); mua, rms = anchor(rg, mu)
    boot = []
    for _ in range(200):
        i = rng.integers(0, n, n)
        r, m = h_mu0(np.mean([hb[k] for k in i], 0), np.mean([db[k] for k in i], 0))
        ma, _ = anchor(r, m); boot.append(np.interp(GRID, r, ma, left=np.nan, right=np.nan))
    band = np.nanstd(np.array(boot), 0); bon = np.interp(GRID, rg, mua, left=np.nan, right=np.nan)
    plt.plot(rg, mua, 'o-', ms=3, color=col, label=r'Harasima, $\Delta U=%g$ (RMS %.3f)' % (dU, rms))
    msk = np.isfinite(band) & np.isfinite(bon)
    plt.fill_between(GRID[msk], bon[msk] - band[msk], bon[msk] + band[msk], color=col, alpha=0.18)
    print('dUmax=%g: Harasima mu0 RMS=%.4f' % (dU, rms))
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored)')
plt.title(r'Field ladder: Harasima contour $\mu_0(\rho)$ vs field strength (N=100, $T^*=1.198$)')
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
_out = 'cube100Tc_ladder_compare.png' if TCSET else 'cube100_ladder_compare.png'
plt.savefig(_out, dpi=140)
print('wrote ' + _out)
