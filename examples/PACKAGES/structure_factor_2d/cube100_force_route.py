#!/usr/bin/env python3
"""Route 1: the exact force / mechanical route, evaluated directly from the 60k-frame
configuration dump (no LAMMPS stress compute, no gradient expansion).

Two exact statements are checked and used:

(a) First-YBG / hydrostatic force balance (EXACT):
        <F_z^conf>(z) = kT rho'(z)/rho(z) + U'(z),
    the mean configurational force per particle balances the entropic + external
    force.  We measure <F_z^conf>(z) from the pair forces and verify it.

(b) Exact pressure tensor from the pair virial, Harasima localization:
        P_ab(z) = rho(z) kT delta_ab + (1/Vbin) sum_pairs (assigned to z) w_ab,
        w_ab = -u'(r) r_a r_b / r,
    then the gradient-theory homogeneous pressure P0 = 3/2 P_T - 1/2 P_N and
    mu0(rho) by Gibbs-Duhem.  This is the SAME contour as the stress/atom compute,
    recomputed from raw configs -- a from-scratch confirmation that the contour
    route is the exact-stress route (the residual is the irreducible 2nd-order
    gradient gauge, not a measurement artifact).  Compared to PeTS and Thol 2015.
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz
import pets_eos as pets, thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.198; L = 6.8582414181223398941; Lz = L; area = L * L; rc = 2.5
NB = 50; vbin = area * (Lz / NB); dumax = 4.0
STRIDE = 12
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T * np.log(r) + thol.properties(T, r)['mu_res']
sc = np.linspace(0.14, 0.55, 40)
mp = np.array([pmu(x) for x in sc]); mt = np.array([tmu(x) for x in sc])
rng = np.random.default_rng(0)


def du(r):                                    # LJ derivative (shift irrelevant)
    return -48.0 * r ** -13 + 24.0 * r ** -7


def parse_xyz(path, stride):
    lines = open(path).read().split('\n'); fr = []; i = 0; n = len(lines); fi = 0
    while i < n:
        if lines[i].startswith('ITEM: TIMESTEP'):
            na = int(lines[i + 3])
            if fi % stride == 0:
                fr.append(np.array([[float(v) for v in lines[i + 9 + j].split()] for j in range(na)]))
            fi += 1; i += 9 + na
        else:
            i += 1
    return fr


frames = parse_xyz('cube100u4_xyz.lammpstrj', STRIDE)
print('frames used=%d (stride %d)' % (len(frames), STRIDE))

# per-frame accumulators (block by frame for bootstrap)
edges = np.linspace(0, Lz, NB + 1)
nfr = len(frames)
Wzz = np.zeros((nfr, NB)); Wxy = np.zeros((nfr, NB))    # Harasima config virial (zz, (xx+yy)/2)
Fz = np.zeros((nfr, NB)); cnt = np.zeros((nfr, NB))      # mean conf force, atom counts

for f, xyz in enumerate(frames):
    z = xyz[:, 2] % Lz
    bin_a = np.clip((z / Lz * NB).astype(int), 0, NB - 1)
    np.add.at(cnt[f], bin_a, 1.0)
    d = xyz[:, None, :] - xyz[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(r, 1e9)
    iu, ju = np.where((r < rc) & (np.triu(np.ones_like(r), 1) > 0))
    rr = r[iu, ju]; dv = du(rr)
    rij = d[iu, ju]                                       # r_i - r_j
    wzz = -dv * rij[:, 2] ** 2 / rr
    wxy = -dv * 0.5 * (rij[:, 0] ** 2 + rij[:, 1] ** 2) / rr
    bi = bin_a[iu]; bj = bin_a[ju]
    # Harasima: half the pair virial to each atom's bin
    np.add.at(Wzz[f], bi, 0.5 * wzz); np.add.at(Wzz[f], bj, 0.5 * wzz)
    np.add.at(Wxy[f], bi, 0.5 * wxy); np.add.at(Wxy[f], bj, 0.5 * wxy)
    # mean configurational force per atom, F_z on i from j = -u'(r) (z_i-z_j)/r
    fz_pair = -dv * rij[:, 2] / rr
    np.add.at(Fz[f], bi, fz_pair); np.add.at(Fz[f], bj, -fz_pair)


def profiles(sel):
    cn = cnt[sel].mean(0); rho = cn / vbin
    PN = rho * T + Wzz[sel].mean(0) / vbin
    PT = rho * T + Wxy[sel].mean(0) / vbin
    Fpp = np.where(cn > 0, Fz[sel].mean(0) / np.clip(cn, 1e-9, None), 0.0)   # force per particle
    return rho, oz.fourier_cosine_smooth(PN, 6), oz.fourier_cosine_smooth(PT, 6), Fpp


def p0_to_mu0(rho, P0):
    o = np.argsort(rho); rs, ps = rho[o], P0[o]
    m = (rs > 0.11) & (rs < 0.58); rs, ps = rs[m], ps[m]
    cf = np.polyfit(rs, ps, 4); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(max(rs.min(), 0.12), min(rs.max(), 0.57), 120)
    return rg, np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])


def anchor(rg, mu, ref):
    mua = mu + (np.interp(np.median(sc), sc, ref) - np.interp(np.median(sc), rg, mu))
    return mua, np.sqrt(np.mean((np.interp(sc, rg, mua) - ref) ** 2))


allsel = np.arange(nfr)
rho, PN, PT, Fpp = profiles(allsel)

# (a) verify the YBG force balance:  <F_z>(z) =?= kT rho'/rho + U'
rho_s = oz.fourier_cosine_smooth(rho, 6)
drho = oz.fourier_cosine_deriv(rho_s, 6, Lz)
z = (np.arange(NB) + 0.5) * Lz / NB
Uprime = -0.5 * dumax * (2 * np.pi / Lz) * np.sin(2 * np.pi * z / Lz)
ybg_rhs = T * drho / rho_s + Uprime              # <F_conf,z> = kT dlnrho/dz + dU/dz
mask = rho_s > 0.05
ybg_err = np.sqrt(np.mean((Fpp[mask] - ybg_rhs[mask]) ** 2))
print('(a) YBG mean-force balance: RMS|<F_z> - (kT dlnrho/dz + dU/dz)| = %.4f  (exact identity -> ~0)' % ybg_err)

# (b) Harasima P0 -> mu0 from configs
P0 = 1.5 * PT - 0.5 * PN
rg, mu0 = p0_to_mu0(rho, P0)
mua_p, rms_p = anchor(rg, mu0, mp); mua_t, rms_t = anchor(rg, mu0, mt)
print('(b) config Harasima P0 -> mu0:  RMS vs PeTS=%.4f  vs Thol=%.4f  (LAMMPS H compute was ~0.037)'
      % (rms_p, rms_t))

# bootstrap over frames
GRID = np.linspace(0.12, 0.57, 80); boot = []
for _ in range(150):
    sel = rng.integers(0, nfr, nfr)
    rh, pn, pt, _ = profiles(sel)
    try:
        r, m = p0_to_mu0(rh, 1.5 * pt - 0.5 * pn); ma, _ = anchor(r, m, mp)
        boot.append(np.interp(GRID, r, ma, left=np.nan, right=np.nan))
    except Exception:
        pass
boot = np.array(boot)
with np.errstate(all='ignore'):
    band = np.nanstd(boot, 0)
band_on = np.interp(GRID, rg, mua_p, left=np.nan, right=np.nan)
print('block-bootstrap median sigma=%.4f' % np.nanmedian(band))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
rr = np.linspace(0.06, 0.62, 250)
ax[0].plot(rr, [pmu(x) for x in rr], 'k-', lw=2.6, label='PeTS EOS')
ax[0].plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2.0, label='Thol 2015 EOS')
ax[0].plot(rg, mua_p, 'o-', ms=4, color='tab:green',
           label='config force route, Harasima (PeTS %.3f / Thol %.3f)' % (rms_p, rms_t))
m = np.isfinite(band) & np.isfinite(band_on)
ax[0].fill_between(GRID[m], band_on[m] - band[m], band_on[m] + band[m], color='tab:green', alpha=0.22,
                   label='block-bootstrap $\\pm1\\sigma$')
ax[0].set_xlabel(r'$\rho^*$'); ax[0].set_ylabel(r'$\mu_0^*$ (anchored)')
ax[0].set_title('Route 1: exact force/virial from 60k configs'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(z, Fpp, 'o', ms=3, color='tab:red', label=r'measured $\langle F_z\rangle(z)$')
ax[1].plot(z, ybg_rhs, '-', color='k', lw=2, label=r'$kT\,\partial_z\ln\rho + \partial_z U$  (YBG)')
ax[1].set_xlabel(r'$z$'); ax[1].set_ylabel(r'mean force / particle')
ax[1].set_title('Exact YBG force balance (RMS %.4f)' % ybg_err); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('cube100_force_route.png', dpi=140)
print('wrote cube100_force_route.png')
