#!/usr/bin/env python3
"""Idea 1: inverse-DFT forward fit.  Parametrize mu0(rho)=T ln rho + sum c_m rho^m and
a nonlocal free-energy kernel, then solve the Euler-Lagrange equation FORWARD,
  mu0(rho(z)) + mu_IH[rho](z) + U(z) = mu_tot,
to predict rho(z) and optimize the parameters to match the measured profile.  This
uses only rho(z)+the field (no stress, no S_ij), avoids differentiating noisy data in
the fit, and uses a FULL nonlocal mu_IH (not a 2nd-order gradient truncation).
Compared with the field-coupling kernel (linear fit) and the contour/DCF benchmarks."""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, field_coupling as fc, kernel_fit as kf
import pets_eos as pets, thol2015_ljts_eos as thol
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = 'cube100u4'; dumax = 4.0
T = 1.198; L = 6.8582414181223398941; Lz = L
rho_meas = oz.fourier_cosine_smooth(fc._read_density('%s_dens.out' % tag), 8)
nb = len(rho_meas); z = (np.arange(nb)+0.5)*Lz/nb; dz = Lz/nb
U = 0.5*dumax*np.cos(2*np.pi*z/Lz)
kk = 2*np.pi*np.fft.rfftfreq(nb, d=dz)
ms = np.arange(1, 6)
pmu = lambda r: T*np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T*np.log(r) + thol.properties(T, r)['mu_res']
sc = np.linspace(0.14, 0.55, 40); mp = np.array([pmu(x) for x in sc])
rgrid = np.linspace(0.004, 0.95, 6000)


def mu0_curve(c):
    return T*np.log(rgrid) + sum(ci*m*rgrid**(m-1) for ci, m in zip(c, ms))


def rms(rg, mu):
    o = np.argsort(rg); rg, mu = np.array(rg)[o], np.array(mu)[o]
    mua = mu + (np.interp(np.median(sc), sc, mp) - np.interp(np.median(sc), rg, mu))
    return np.sqrt(np.mean((np.interp(sc, rg, mua)-mp)**2))


def forward(c, kernel, mu_tot, iters=250):
    """Solve mu0(rho) + mu_IH[rho] + U = mu_tot for rho(z) by damped Picard.
    kernel = array of K(s) on the periodic lag grid (mu_IH = conv(K, rho) - rho*sum K dz)."""
    rho = rho_meas.copy()
    Khat = np.fft.rfft(kernel)
    for _ in range(iters):
        muih = dz*(np.fft.irfft(Khat*np.fft.rfft(rho), n=nb) - rho*kernel.sum())
        target = mu_tot - U - muih
        g = mu0_curve(c); o = np.argsort(g)
        rho = 0.6*rho + 0.4*np.interp(target, g[o], rgrid[o])
    return rho


# data-driven init: LDA gives mu0(rho(z)) ~ mu_tot - U(z); fit mu_ex poly to it
mu_tot0 = pmu(rho_meas.mean()) + U[np.argmin(np.abs(rho_meas-rho_meas.mean()))]
mu_ex_lda = (mu_tot0 - U) - T*np.log(rho_meas)
A = np.vstack([m*rho_meas**(m-1) for m in ms]).T
c0 = np.linalg.lstsq(A, mu_ex_lda, rcond=None)[0]

# kernel basis: even Hann-cosine lags (zero-integral), few modes
lag = (np.arange(nb)+nb//2) % nb - nb//2
s = lag*dz; smax = 2.5
hann = np.where(np.abs(s) <= smax, 0.5*(1+np.cos(np.pi*s/smax)), 0.0)
basis = []
for mmode in range(3):
    chi = hann*np.cos(mmode*np.pi*s/smax)*(np.abs(s) <= smax)
    chi = np.where(np.abs(s) <= smax, chi - chi[np.abs(s) <= smax].mean(), 0.0)
    basis.append(np.roll(chi, -(nb//2)))
basis = np.array(basis)

npar = len(ms)


def resid(p):
    c = p[:npar]; a = p[npar:npar+3]; mu_tot = p[-1]
    kernel = (a[:, None]*basis).sum(0)
    rp = forward(c, kernel, mu_tot)
    return np.concatenate([(rp-rho_meas)*50, 1e-2*c, 1e-1*a])


p0 = np.concatenate([c0, [0.0, 0.0, 0.0], [mu_tot0]])
sol = least_squares(resid, p0, method='trf', max_nfev=300, x_scale='jac')
c = sol.x[:npar]
rg = np.linspace(rho_meas.min()+0.01, rho_meas.max()-0.01, 120)
mu0_idft = T*np.log(rg) + sum(ci*m*rg**(m-1) for ci, m in zip(c, ms))
rfit = np.sqrt(np.mean(resid(sol.x)[:nb]**2))/50
print('inverse-DFT forward fit:  rho-fit RMS=%.4f   mu0 RMS vs PeTS=%.4f' % (rfit, rms(rg, mu0_idft)))

# field-coupling kernel (linear nonlocal inverse-DFT) for comparison
ek = kf.kernel_eos(np.array([0.0, dumax/2]), np.array([np.full_like(rho_meas, rho_meas.mean()), rho_meas]),
                   T, Lz, deg=6, smax=2.5, nmodes=3, ridge=1e-3, smooth=8)
print('FC-kernel (linear nonlocal):                 mu0 RMS vs PeTS=%.4f' % rms(ek['rho'], ek['mu0']))

plt.figure(figsize=(8, 5.5))
rr = np.linspace(0.05, 0.62, 200)
plt.plot(rr, [pmu(x) for x in rr], 'k-', lw=2.5, label='PeTS EOS')
plt.plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2, label='Thol 2015 EOS')
plt.plot(rg, mu0_idft + (np.interp(np.median(sc), sc, mp)-np.interp(np.median(sc), rg, mu0_idft)),
         'o-', ms=3, color='tab:red', label='inverse-DFT forward fit (RMS %.3f)' % rms(rg, mu0_idft))
o = np.argsort(ek['rho']); ekr, ekm = np.array(ek['rho'])[o], np.array(ek['mu0'])[o]
plt.plot(ekr, ekm + (np.interp(np.median(sc), sc, mp)-np.interp(np.median(sc), ekr, ekm)),
         's-', ms=2, color='tab:purple', alpha=0.6, label='FC-kernel (RMS %.3f)' % rms(ek['rho'], ek['mu0']))
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored)')
plt.title(r'Inverse-DFT forward fit ($\Delta U=4$): non-stress, nonlocal')
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('cube100_inverse_dft.png', dpi=140)
print('wrote cube100_inverse_dft.png')
