#!/usr/bin/env python3
"""Throw every mu0(rho)-recovery method at the N=100 cubic CPP torture run and
compare to the PeTS EOS at T*=1.198.  Routes:
  FC-gradient / FC-kernel : field-coupling from rho(z) + the external field
  OZ-DFT / OZ-KB          : DCF charging from the structure factor S_ij(k)
  IK / H contour          : pressure tensor -> P0(z) -> mu0 (Gibbs-Duhem)
All anchored to PeTS at the mean density; RMS over a common density window.
"""
import sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, '/home/user/lammps/ljts_eos')
import field_coupling as fc, kernel_fit as kf, contour_pressure as cp, oz_invert as oz
import pets_eos as pets
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.198
L = 6.8582414181223398941
Lx = Ly = Lz = L
dumax = 2.0
NBIN = 50
SFBIN = 16
tag = 'cube100'
smooth = 6


def pets_mu0(r):
    return T*np.log(r) + np.array([pets.properties(T, x)['mu_res'] for x in r])


curves = {}   # label -> (rho, mu0)

# ---- field-coupling (gradient + nonlocal kernel), single field strength ----
try:
    rho_z = oz.fourier_cosine_smooth(fc._read_density('%s_dens.out' % tag), smooth)
    amps = np.array([0.0, dumax/2.0])
    profiles = np.array([np.full_like(rho_z, rho_z.mean()), rho_z])
    eg = fc.local_eos(amps, profiles, T, Lz, deg=6, smooth=smooth, grad_spec={2: 0, 4: 0})
    curves['FC-gradient'] = (eg['rho'], eg['mu0'])
    ek = kf.kernel_eos(amps, profiles, T, Lz, deg=6, smax=2.5, nmodes=3, ridge=1e-3, smooth=smooth)
    curves['FC-kernel'] = (ek['rho'], ek['mu0'])
except Exception as e:
    print('FC failed:', e)

# ---- DCF charging from S_ij(k): OZ-DFT (Evans) and OZ-KB (compressibility) ----
try:
    dz = Lz/SFBIN; area = Lx*Lx
    sf = oz.read_ave_time_vector('%s_sf.out' % tag)
    qs, Smats, rho = oz.assemble_matrices(sf, SFBIN)
    Smats, rho = oz.mirror_symmetrize(Smats, rho)
    rho_s = oz.fourier_cosine_smooth(rho, smooth)
    active = np.where(rho > 0.10)[0]                  # discard ill-conditioned vapor bins
    Carr = {q: oz.invert_oz(Smats[q], rho, dz, area, active=active, ridge=1e-3)[0] for q in qs}
    C0 = oz.intercept_matrix(qs, Carr, active, 2.5, dz=dz, beta=1.0/T, smax=Lx/2, tail_rsplit=1.5)
    C0 = oz.smooth_intercept(C0, active, rho_s, deg=3)
    mu_ih = oz.dft_mu_ih(C0, rho_s[active], dz, T)
    z = (active+0.5)*dz
    mu0 = -0.5*dumax*np.cos(2.0*np.pi*z/Lz) - mu_ih
    ra = rho_s[active]; o = np.argsort(ra)
    curves['OZ-DFT'] = (ra[o], mu0[o])
    chat0 = oz.local_chat0(qs, Carr, active, SFBIN, 2.5, dz)
    rgk, muk = oz.kb_chemical_potential(rho, chat0, T)
    curves['OZ-KB'] = (rgk, muk)
except Exception as e:
    print('OZ failed:', e)

# ---- pressure-tensor contours -> P0(z) -> mu0 ----
for name, fn in [('IK-contour', 'ik'), ('H-contour', 'h')]:
    try:
        if fn == 'ik':
            r, PN, PT = cp.ik_profile('%s_ikstress.out' % tag, Lz, smooth)
        else:
            r, PN, PT = cp.h_profile('%s_hstress.out' % tag, '%s_dens.out' % tag,
                                     Lz, Lx*Lx, T, smooth)
        r2, mu, P02 = cp.mu0_from_p0(r, cp.p0_ik(PN, PT), Lz, smooth)
        curves[name] = (r2, mu)
    except Exception as e:
        print('%s failed: %s' % (name, e))

# ---- compare to PeTS ----
allr = np.concatenate([c[0] for c in curves.values()])
rlo, rhi = max(0.12, allr.min()), min(0.60, allr.max())
score = np.linspace(rlo, rhi, 40)
mp_score = pets_mu0(score)
ranch = np.median(score); m0 = np.interp(ranch, score, mp_score)
print('# method        RMS(mu0-PeTS) on [%.2f,%.2f]   rho-range' % (rlo, rhi))
fig, ax = plt.subplots(figsize=(8.5, 6))
rr = np.linspace(max(0.04, allr.min()), allr.max(), 200)
ax.plot(rr, pets_mu0(rr), 'k-', lw=2.5, label='PeTS EOS')
colors = dict(zip(curves, ['tab:orange', 'tab:red', 'tab:blue', 'tab:cyan',
                           'tab:green', 'tab:purple']))
for lab, (r, mu) in curves.items():
    o = np.argsort(r); r, mu = np.array(r)[o], np.array(mu)[o]
    mu_a = mu + (m0 - np.interp(ranch, r, mu))
    rms = np.sqrt(np.mean((np.interp(score, r, mu_a) - mp_score)**2))
    print('  %-12s  %.4f                       %.3f-%.3f' % (lab, rms, r.min(), r.max()))
    ax.plot(r, mu_a, 'o-', ms=2.5, lw=1.2, color=colors.get(lab), label='%s (RMS %.3f)' % (lab, rms))
ax.set_xlabel(r'$\rho^*$'); ax.set_ylabel(r'$\mu_0^*$ (anchored to PeTS)')
ax.set_title(r'N=100 cubic, $T^*=1.198$, $\Delta U=2$: every method vs PeTS')
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig('cube100_allmethods.png', dpi=140)
print('wrote cube100_allmethods.png')
