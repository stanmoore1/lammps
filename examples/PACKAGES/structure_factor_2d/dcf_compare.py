#!/usr/bin/env python3
"""DCF (direct-correlation-function) charging proof of concept at Tc.

OZ-invert the bin-resolved planar structure factor S_ij(k) from a single CPP
strong-field run into the direct correlation function, then charge it to the
homogeneous chemical potential mu0(rho) by two independent routes:

  DFT/Evans       : nonlocal mu_IH from the k=0 intercept matrix C_ij(0)
  KB/compressibility: beta dmu0/drho = 1/rho - c_hat(0;rho)

Compare to the PeTS EOS (and report the Maxwell binodal of the DFT route).
This is the DCF analogue of the field-coupling ('field charging') method, from
ONE run instead of a field ladder.
"""
import sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz
import pets_eos as pets
import phase_diagram as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.089
nbins = 40
lx = 10.244851881402800231
lz = 30.734555644208398917
dumax = 2.0
rho_min = 0.05
kfit = 2.5
dz = lz/nbins
area = lx*lx

sf = oz.read_ave_time_vector('dcf_sf.out')
qs, Smats, rho = oz.assemble_matrices(sf, nbins)
Smats, rho = oz.mirror_symmetrize(Smats, rho)
rho_s = oz.fourier_cosine_smooth(rho, 6)
active = np.where(rho > rho_min)[0]
print('density range (active bins): %.4f .. %.4f  (%d/%d bins)'
      % (rho_s[active].min(), rho_s[active].max(), len(active), nbins))

Carr = {q: oz.invert_oz(Smats[q], rho, dz, area, active=active, ridge=1e-4)[0] for q in qs}

# ---- DFT / Evans route: nonlocal mu_IH from C_ij(0) ----
C0 = oz.intercept_matrix(qs, Carr, active, kfit, dz=dz, beta=1.0/T,
                         smax=lx/2, tail_rsplit=1.5)
C0 = oz.smooth_intercept(C0, active, rho_s, deg=3)
mu_ih = oz.dft_mu_ih(C0, rho_s[active], dz, T)
z = (active+0.5)*dz
Uext = 0.5*dumax*np.cos(2.0*np.pi*z/lz)
mu0_dft = -Uext - mu_ih                      # up to an additive constant
rho_dft = rho_s[active]
o = np.argsort(rho_dft)
rho_dft, mu0_dft = rho_dft[o], mu0_dft[o]

# ---- KB / compressibility route ----
chat0 = oz.local_chat0(qs, Carr, active, nbins, kfit, dz)
rg_kb, mu0_kb = oz.kb_chemical_potential(rho, chat0, T)
o2 = np.argsort(rg_kb)
rg_kb, mu0_kb = rg_kb[o2], mu0_kb[o2]

# ---- PeTS reference ----
def pets_mu0(r): return T*np.log(r) + np.array([pets.properties(T, x)['mu_res'] for x in r])
rr = np.linspace(max(0.04, rho_dft.min()), rho_dft.max(), 200)
mu_pets = pets_mu0(rr)
ranchor = np.median(rho_dft)
m0 = np.interp(ranchor, rr, mu_pets)

def anchor(rg, mu):
    return mu + (m0 - np.interp(ranchor, rg, mu))
mu0_dft_a = anchor(rho_dft, mu0_dft)
mu0_kb_a = anchor(rg_kb, mu0_kb)

# ---- Maxwell binodal of the DFT route (anchor-invariant) ----
rv, rl, ps = pets.vle(T)
print('PeTS vle: rho_v=%.3f rho_l=%.3f' % (rv, rl))
f0 = np.concatenate([[0.0], np.cumsum(0.5*(mu0_dft_a[1:]+mu0_dft_a[:-1])*np.diff(rho_dft))])
P0 = rho_dft*mu0_dft_a - f0
bi = pd.binodal(rho_dft, P0, mu0_dft_a)
if bi:
    print('DCF (DFT route) binodal: rho_v=%.3f rho_l=%.3f' % (bi['rho_v'], bi['rho_l']))
else:
    print('DCF (DFT route) binodal: no loop (near-critical / flattened)')

# ---- plot ----
plt.figure(figsize=(8, 5.5))
plt.plot(rr, mu_pets, 'k-', lw=2.5, label='PeTS EOS')
plt.plot(rho_dft, mu0_dft_a, 'o-', color='tab:red', ms=3, lw=1.3,
         label='DCF charging (DFT / Evans)')
plt.plot(rg_kb, mu0_kb_a, 's-', color='tab:blue', ms=3, lw=1.3,
         label='DCF charging (KB / compressibility)')
plt.axvline(ranchor, color='gray', ls=':', lw=0.7)
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored to PeTS at $\rho_{med}$)')
plt.title(r'DCF charging at $T^*=1.089$ ($T_c$): $\mu_0(\rho)$ from one CPP run vs PeTS')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('dcf_mu0.png', dpi=140)
print('wrote dcf_mu0.png')
