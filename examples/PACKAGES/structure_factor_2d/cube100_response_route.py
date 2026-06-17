#!/usr/bin/env python3
"""Route 2: the exact density-response / compressibility route, done in k-space.

The exact static relation is  delta rho(z)/delta(beta U(z')) = -[rho delta + rho^2 h]
= -S(z,z'); its k->0 limit is the local compressibility, and (Gibbs-Duhem)

    d mu0/d rho |_{rho(z)} = kT / (rho(z) S(0;rho(z))).

The occupancy-fluctuation method used Var(N)/<N> for S(0) and FAILED, because the
canonical fixed-N constraint kills the k=0 mode and slab boundaries leak.  Here we
take the *lateral* structure factor S_ii(k) of each z-bin (the in-plane k, which is
NOT constrained by total N) and extrapolate k->0 with the Ornstein-Zernike form
S_ii(k)/<N_i> = S0 + b k^2.  S0 is the local homogeneous S(0;rho_i); no z-direction
correlations enter, so the slab-boundary leak is gone.  Integrate over rho -> mu0(rho).
Block-bootstrap error bars; compared to PeTS and Thol 2015.
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz
import pets_eos as pets, thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.198; L = 6.8582414181223398941; Lz = L; area = L * L
SFBIN = 16; vbin = area * (Lz / SFBIN)
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T * np.log(r) + thol.properties(T, r)['mu_res']
sc = np.linspace(0.14, 0.55, 40)
mp = np.array([pmu(x) for x in sc]); mt = np.array([tmu(x) for x in sc])
rng = np.random.default_rng(0)
QFIT = 2.6                                     # use lateral k below this for the OZ fit


def s0_per_bin(sfm):
    """Local S(0;rho_i) per bin from the OZ small-k extrapolation of the lateral
    diagonal structure factor S_ii(k)/<N_i>."""
    qs, Sm, rho = oz.assemble_matrices(sfm, SFBIN)
    qs = np.array(qs)
    S0 = np.full(SFBIN, np.nan)
    for i in range(SFBIN):
        Ni = rho[i] * vbin
        if Ni < 0.5:
            continue
        sd = np.array([Sm[q][i, i] for q in qs]) / Ni        # -> S(k)
        m = qs < QFIT
        if m.sum() >= 3:
            b = np.polyfit(qs[m] ** 2, sd[m], 1)             # S = S0 + b k^2
            S0[i] = b[1]
        else:
            S0[i] = sd[m][0]
    return rho, S0


def mu0_curve(sfm):
    rho, S0 = s0_per_bin(sfm)
    good = np.isfinite(S0) & (S0 > 0.02) & (rho > 0.05) & (rho < 0.66)
    r, s = rho[good], S0[good]
    dmu = T / (r * s)                                         # d mu0 / d rho
    o = np.argsort(r); r, dmu = r[o], dmu[o]
    # collapse duplicate densities (two slabs per rho) by averaging
    ru, idx = np.unique(np.round(r, 4), return_inverse=True)
    du = np.array([dmu[idx == k].mean() for k in range(len(ru))])
    mu0 = np.concatenate([[0], np.cumsum(0.5 * (du[1:] + du[:-1]) * np.diff(ru))])
    return ru, mu0


def anchor(rg, mu, ref):
    o = np.argsort(rg); rg, mu = np.array(rg)[o], np.array(mu)[o]
    mua = mu + (np.interp(np.median(sc), sc, ref) - np.interp(np.median(sc), rg, mu))
    return mua, np.sqrt(np.mean((np.interp(sc, rg, mua) - ref) ** 2))


sf = oz.read_ave_time_blocks('cube100u4_sf.out')
print('S_ij blocks=%d' % len(sf))
rg, mu0 = mu0_curve(np.mean(sf, 0))
mua_p, rms_p = anchor(rg, mu0, mp)
mua_t, rms_t = anchor(rg, mu0, mt)
print('k-space compressibility route:  mu0 RMS vs PeTS=%.4f  vs Thol=%.4f' % (rms_p, rms_t))

# block bootstrap
GRID = np.linspace(0.10, 0.62, 80)
boot = []
for _ in range(200):
    idx = rng.integers(0, len(sf), len(sf))
    try:
        r, m = mu0_curve(np.mean([sf[k] for k in idx], 0))
        ma, _ = anchor(r, m, mp)
        boot.append(np.interp(GRID, r, ma, left=np.nan, right=np.nan))
    except Exception:
        pass
boot = np.array(boot)
with np.errstate(all='ignore'):
    band = np.nanstd(boot, 0)
band_on = np.interp(GRID, rg, mua_p, left=np.nan, right=np.nan)
print('block-bootstrap median sigma=%.4f' % np.nanmedian(band))

plt.figure(figsize=(8.5, 6))
rr = np.linspace(0.06, 0.63, 250)
plt.plot(rr, [pmu(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS')
plt.plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2.0, label='Thol 2015 EOS')
plt.plot(rg, mua_p, 'o-', ms=4, color='tab:purple',
         label='k-space compressibility (PeTS %.3f / Thol %.3f)' % (rms_p, rms_t))
m = np.isfinite(band) & np.isfinite(band_on)
plt.fill_between(GRID[m], band_on[m] - band[m], band_on[m] + band[m], color='tab:purple', alpha=0.22,
                 label='block-bootstrap $\\pm1\\sigma$')
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored)')
plt.title(r'Route 2: exact density-response / lateral-$S(0)$ compressibility ($\Delta U=4$)')
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('cube100_response_route.png', dpi=140)
print('wrote cube100_response_route.png')
