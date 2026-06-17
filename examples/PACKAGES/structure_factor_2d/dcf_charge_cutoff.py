#!/usr/bin/env python3
"""DCF charging at Tc with a density cutoff: discard the dilute-vapor bins (rho <
rho_min) from the OZ inversion, where h ~ 1/rho^2 is ill-conditioned and dominates
the error.  Scan rho_min, score mu0 vs PeTS on a COMMON mid-range (so the comparison
isolates the conditioning gain, not the shrinking range), and plot the best cutoff
with block-bootstrap error bands.  Uses the corrected LJTS mean-field tail."""
import sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz
import pets_eos as pets
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.089; nbins = 40
lx = 10.244851881402800231; lz = 30.734555644208398917
dumax = 2.0; dz = lz/nbins; area = lx*lx

blocks = oz.read_ave_time_blocks('dcf_sf.out') + oz.read_ave_time_blocks('dcf_sf_b.out')
sf_all = np.mean(blocks, axis=0)
print('combined: %d blocks (1.5M steps)' % len(blocks))

# common scoring range (inside every cutoff's retained range)
score = np.linspace(0.20, 0.60, 40)
def pets_mu0(r): return T*np.log(r) + np.array([pets.properties(T, x)['mu_res'] for x in r])
mu_pets_score = pets_mu0(score)


def mu0_dft(sf, rho_min, return_curve=False):
    qs, Smats, rho = oz.assemble_matrices(sf, nbins)
    Smats, rho = oz.mirror_symmetrize(Smats, rho)
    rho_s = oz.fourier_cosine_smooth(rho, 6)
    active = np.where(rho > rho_min)[0]
    Carr = {q: oz.invert_oz(Smats[q], rho, dz, area, active=active, ridge=1e-4)[0] for q in qs}
    C0 = oz.intercept_matrix(qs, Carr, active, 2.5, dz=dz, beta=1.0/T, smax=lx/2,
                             tail_rsplit=1.5)            # rcut=2.5 LJTS by default
    C0 = oz.smooth_intercept(C0, active, rho_s, deg=3)
    mu_ih = oz.dft_mu_ih(C0, rho_s[active], dz, T)
    z = (active+0.5)*dz
    mu0 = -0.5*dumax*np.cos(2.0*np.pi*z/lz) - mu_ih
    ra = rho_s[active]; o = np.argsort(ra)
    if return_curve:
        return ra[o], mu0[o]
    return np.interp(score, ra[o], mu0[o])


def rms_score(mu_on_score):
    m0 = np.interp(np.median(score), score, mu_pets_score)
    a = mu_on_score + (m0 - np.interp(np.median(score), score, mu_on_score))
    return np.sqrt(np.mean((a - mu_pets_score)**2))


print('\nrho_min  active_bins  rho_range        RMS(mu0-PeTS) on [0.20,0.60]')
best = None
for rmin in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    ra, mu = mu0_dft(sf_all, rmin, return_curve=True)
    r = rms_score(np.interp(score, ra, mu))
    print('  %.2f      %3d        %.3f-%.3f      %.4f'
          % (rmin, len(ra), ra.min(), ra.max(), r))
    if best is None or r < best[1]:
        best = (rmin, r)
rmin_best = best[0]
print('\nbest cutoff: rho_min = %.2f  (RMS %.4f)' % (rmin_best, best[1]))

# block-bootstrap mu0 on a curve grid at the best cutoff
grid = np.linspace(rmin_best + 0.02, 0.64, 50)
def analysis(sf):
    ra, mu = mu0_dft(sf, rmin_best, return_curve=True)
    return np.interp(grid, ra, mu)
mean_mu, std_mu = oz.bootstrap_blocks(blocks, analysis, nresample=150, seed=2)
mu_pets_grid = pets_mu0(grid)
m0 = np.interp(np.median(grid), grid, mu_pets_grid)
anchor = m0 - np.interp(np.median(grid), grid, mean_mu)
mean_a = mean_mu + anchor

# also the discarded (rho_min=0.05) curve for contrast
ra0, mu0_0 = mu0_dft(sf_all, 0.05, return_curve=True)
mu0_0a = mu0_0 + (m0 - np.interp(np.median(grid), ra0, mu0_0))

plt.figure(figsize=(8, 5.5))
rr = np.linspace(0.05, 0.66, 200)
plt.plot(rr, pets_mu0(rr), 'k-', lw=2.5, label='PeTS EOS')
plt.plot(ra0, mu0_0a, ':', color='gray', lw=1.2, label=r'all bins ($\rho_{min}=0.05$)')
plt.fill_between(grid, mean_a-std_mu, mean_a+std_mu, color='tab:red', alpha=0.25)
plt.plot(grid, mean_a, 'o-', color='tab:red', ms=3, lw=1.4,
         label=r'DCF charging, $\rho_{min}=%.2f$ (bootstrap)' % rmin_best)
plt.axvspan(0.05, rmin_best, color='gray', alpha=0.08)
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored to PeTS)')
plt.title(r'DCF charging at $T^*=1.089$: discarding dilute bins $\rho<%.2f$' % rmin_best)
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('dcf_mu0_cutoff.png', dpi=140)
print('wrote dcf_mu0_cutoff.png')
