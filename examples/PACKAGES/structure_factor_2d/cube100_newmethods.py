#!/usr/bin/env python3
"""New mu0(rho)-recovery ideas for the N=100 cubic torture run (T*=1.198, dUmax=2),
beyond the standard FC / DCF / contour routes.  Goal: invert the inhomogeneous
profile back to the homogeneous EOS.  All compared to PeTS.

  LDA          : mu0(rho(z)) = mu_tot - U(z)        (no inhomogeneity correction)
  contour-rho  : P0(z)=3/2 PT-1/2 PN binned vs rho(z), then mu0 = INT dP0/rho
  force-balance: PN(z) from the EXACT momentum balance dPN/dz = rho*f_ext (smooth,
                 from rho(z)+the known field) replacing the noisy measured PN
  fwd-EL fit   : parametrize mu0(rho)+a gradient kappa, solve the Euler-Lagrange
                 equation FORWARD to predict rho(z) and fit it (no noisy rho'')
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import contour_pressure as cp, oz_invert as oz, field_coupling as fc
import pets_eos as pets
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.198; L = 6.8582414181223398941; Lz = L; AF = 1.0
NBIN = 50; tag = 'cube100'; smooth = 6
trapz = np.trapezoid


def pets_mu0(r): return T*np.log(r) + np.array([pets.properties(T, x)['mu_res'] for x in np.atleast_1d(r)])


# measured, time-averaged density profile rho(z) (smoothed)
rho_raw = fc._read_density('%s_dens.out' % tag)
rho_z = oz.fourier_cosine_smooth(rho_raw, smooth)
nb = len(rho_z)
z = (np.arange(nb) + 0.5) * (Lz/nb)
U = AF*np.cos(2.0*np.pi*z/Lz)              # external potential
fext = 2.0*np.pi*AF*np.sin(2.0*np.pi*z/Lz)/Lz   # = -dU/dz (the addforce)
dz = Lz/nb

curves = {}

# 1) LDA: mu0(rho(z)) = -U(z) + const
curves['LDA'] = (rho_z.copy(), -U.copy())

# 2) force-balance PN + measured PT -> P0 -> mu0
try:
    rik, PNik, PTik = cp.ik_profile('%s_ikstress.out' % tag, Lz, smooth)
    nbk = len(rik); zk = (np.arange(nbk)+0.5)*(Lz/nbk)
    fext_k = 2.0*np.pi*AF*np.sin(2.0*np.pi*zk/Lz)/Lz
    # exact momentum balance dPN/dz = rho*f_ext, integrated from the (clean) IK density
    integrand = rik*fext_k
    PN_fb = np.concatenate([[0.0], np.cumsum(0.5*(integrand[1:]+integrand[:-1])*(Lz/nbk))])
    PN_fb -= PN_fb.mean()                          # gauge (cancels in dP0/dz)
    P0_fb = 1.5*PTik - 0.5*PN_fb
    r2, mu, _ = cp.mu0_from_p0(rik, P0_fb, Lz, smooth)
    curves['force-balance'] = (r2, mu)
    # IK and H contours, and their average
    r3, mu3, _ = cp.mu0_from_p0(rik, cp.p0_ik(PNik, PTik), Lz, smooth)
    curves['IK-contour'] = (r3, mu3)
    rh, PNh, PTh = cp.h_profile('%s_hstress.out' % tag, '%s_dens.out' % tag, Lz, L*L, T, smooth)
    rH, muH, _ = cp.mu0_from_p0(rh, cp.p0_ik(PNh, PTh), Lz, smooth)
    curves['H-contour'] = (rH, muH)
    # IK+H average P0 (the two contours bracket the truth; average cancels opposite bias)
    P0avg = 0.5*(cp.p0_ik(PNik, PTik) + cp.p0_ik(PNh, PTh))
    ra, mua, _ = cp.mu0_from_p0(rik, P0avg, Lz, smooth)
    curves['IK+H avg'] = (ra, mua)
except Exception as e:
    import traceback; traceback.print_exc()
    print('contour/force-balance failed:', e)

# 3) forward Euler-Lagrange fit: mu0(rho)=T ln rho + sum_m c_m m rho^(m-1),
#    EL: mu0(rho(z)) - kappa rho''(z) + U(z) = mu_tot ; solve forward for rho(z),
#    fit (c_m, kappa, mu_tot) to the measured rho(z).  Avoids differentiating data.
def solve_forward(cm, kappa, mu_tot, U, dz, rho_init, iters=400):
    ms = np.arange(1, len(cm)+1)
    rho = rho_init.copy()
    k = 2.0*np.pi*np.fft.rfftfreq(len(U), d=dz)
    for _ in range(iters):
        rpp = np.fft.irfft(-(k**2)*np.fft.rfft(rho), n=len(rho))   # rho'' (spectral)
        target = mu_tot - U + kappa*rpp                            # = mu0(rho) wanted
        # invert mu0(rho)=target for rho (mu0 = T ln rho + sum c_m m rho^(m-1))
        rgrid = np.linspace(0.005, 0.95, 4000)
        mu0g = T*np.log(rgrid) + sum(c*m*rgrid**(m-1) for c, m in zip(cm, ms))
        order = np.argsort(mu0g)
        rho_new = np.interp(target, mu0g[order], rgrid[order])
        rho = 0.5*rho + 0.5*rho_new                                # damped Picard
    return rho

try:
    deg = 5
    rho_meas = rho_z
    def resid(p):
        cm = p[:deg]; kappa = p[deg]; mu_tot = p[deg+1]
        rp = solve_forward(cm, kappa, mu_tot, U, dz, rho_meas)
        return np.concatenate([(rp-rho_meas), 1e-3*np.array(cm)])     # tiny ridge
    p0 = np.concatenate([np.zeros(deg), [1.0, pets_mu0(rho_meas.mean())[0]]])
    sol = least_squares(resid, p0, method='lm', max_nfev=200)
    cm = sol.x[:deg]; ms = np.arange(1, deg+1)
    rg = np.linspace(rho_z.min()+0.01, rho_z.max()-0.01, 120)
    mu0_fwd = T*np.log(rg) + sum(c*m*rg**(m-1) for c, m in zip(cm, ms))
    curves['fwd-EL fit'] = (rg, mu0_fwd)
    print('fwd-EL fit: kappa=%.3f  rms(rho fit)=%.4f' % (sol.x[deg], np.sqrt(np.mean(resid(sol.x)[:nb]**2))))
except Exception as e:
    print('fwd-EL fit failed:', e)

# ---- compare to PeTS ----
allr = np.concatenate([np.array(c[0]) for c in curves.values()])
rlo, rhi = max(0.12, allr.min()), min(0.58, allr.max())
score = np.linspace(rlo, rhi, 40); mp = pets_mu0(score)
ranch = np.median(score); m0 = np.interp(ranch, score, mp)
print('\n# method         RMS(mu0-PeTS) on [%.2f,%.2f]' % (rlo, rhi))
fig, ax = plt.subplots(figsize=(8.5, 6))
rr = np.linspace(max(0.04, allr.min()), allr.max(), 200)
ax.plot(rr, pets_mu0(rr), 'k-', lw=2.5, label='PeTS EOS')
for lab, (r, mu) in curves.items():
    r = np.array(r); mu = np.array(mu); o = np.argsort(r); r, mu = r[o], mu[o]
    mu_a = mu + (m0 - np.interp(ranch, r, mu))
    rms = np.sqrt(np.mean((np.interp(score, r, mu_a)-mp)**2))
    print('  %-14s %.4f' % (lab, rms))
    ax.plot(r, mu_a, 'o-', ms=2.5, lw=1.2, label='%s (RMS %.3f)' % (lab, rms))
ax.set_xlabel(r'$\rho^*$'); ax.set_ylabel(r'$\mu_0^*$ (anchored)')
ax.set_title(r'N=100 cubic $T^*=1.198$: new methods vs PeTS')
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig('cube100_newmethods.png', dpi=140)
print('wrote cube100_newmethods.png')
