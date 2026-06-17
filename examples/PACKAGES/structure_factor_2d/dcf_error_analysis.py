#!/usr/bin/env python3
"""Error analysis of the DCF-charging mu0(rho) at Tc: combine the independent runs,
block-bootstrap the statistical error, and scan the analysis knobs to expose the
systematic error.  Identifies the dominant error sources."""
import sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz
import pets_eos as pets

T = 1.089; nbins = 40
lx = 10.244851881402800231; lz = 30.734555644208398917
dumax = 2.0; dz = lz/nbins; area = lx*lx

# combine the blocks of the two independent Tc runs (500k + 1M = 1.5M, 30 blocks)
blocks = oz.read_ave_time_blocks('dcf_sf.out') + oz.read_ave_time_blocks('dcf_sf_b.out')
print('combined blocks: %d (1.5M steps of sampling)' % len(blocks))

grid = np.linspace(0.10, 0.66, 40)


def pets_mu0(r):
    return T*np.log(r) + np.array([pets.properties(T, x)['mu_res'] for x in r])
mu_pets = pets_mu0(grid)


def mu0_dft(sf, kfit=2.5, ridge=1e-4, tail=1.5, smooth=6, smooth_c0=3):
    """DFT/Evans mu0(rho) on the fixed grid (anchor-free shape; anchored at compare)."""
    qs, Smats, rho = oz.assemble_matrices(sf, nbins)
    Smats, rho = oz.mirror_symmetrize(Smats, rho)
    rho_s = oz.fourier_cosine_smooth(rho, smooth)
    active = np.where(rho > 0.05)[0]
    Carr = {q: oz.invert_oz(Smats[q], rho, dz, area, active=active, ridge=ridge)[0] for q in qs}
    C0 = oz.intercept_matrix(qs, Carr, active, kfit, dz=dz, beta=1.0/T, smax=lx/2, tail_rsplit=tail)
    if smooth_c0:
        C0 = oz.smooth_intercept(C0, active, rho_s, deg=smooth_c0)
    mu_ih = oz.dft_mu_ih(C0, rho_s[active], dz, T)
    z = (active+0.5)*dz
    mu0 = -0.5*dumax*np.cos(2.0*np.pi*z/lz) - mu_ih
    ra = rho_s[active]; o = np.argsort(ra)
    return np.interp(grid, ra[o], mu0[o])


def rms_vs_pets(mu0_grid):
    # anchor at median density (binodal/shape comparison is anchor-free)
    m0 = np.interp(np.median(grid), grid, mu_pets)
    a = mu0_grid + (m0 - np.interp(np.median(grid), grid, mu0_grid))
    return np.sqrt(np.mean((a - mu_pets)**2))


# --- statistical error: block bootstrap over the 30 blocks ---
def analysis(sf): return mu0_dft(sf)
mean_mu, std_mu = oz.bootstrap_blocks(blocks, analysis, nresample=200, seed=1)
print('\nstatistical (bootstrap) error on mu0: mean +/- %.4f (median over rho grid)'
      % np.median(std_mu))
print('RMS(mu0 - PeTS), full 1.5M = %.4f' % rms_vs_pets(mean_mu))

# 500k only vs 1.5M -> does more sampling help?
half = oz.read_ave_time_blocks('dcf_sf.out')
m500 = mu0_dft(np.mean(half, axis=0))
print('RMS(mu0 - PeTS), 500k only = %.4f' % rms_vs_pets(m500))

# --- systematic error: scan the analysis knobs on the full data ---
sf_all = np.mean(blocks, axis=0)
print('\nsystematic sensitivity (RMS vs PeTS) on the full 1.5M data:')
base = dict(kfit=2.5, ridge=1e-4, tail=1.5, smooth=6, smooth_c0=3)
print('  baseline                         RMS=%.4f' % rms_vs_pets(mu0_dft(sf_all, **base)))
for name, vals in [('kfit', [1.5, 2.0, 3.0]), ('ridge', [1e-5, 1e-3, 1e-2]),
                   ('tail', [0.0, 1.0, 2.0]), ('smooth', [4, 8, 10]),
                   ('smooth_c0', [0, 2, 5])]:
    for v in vals:
        p = dict(base); p[name] = v
        print('  %-8s = %-6s                 RMS=%.4f' % (name, v, rms_vs_pets(mu0_dft(sf_all, **p))))
