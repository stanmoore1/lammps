#!/usr/bin/env python3
"""More EOS-recovery attempts on the cubic N=100 runs:
 (A) optimal linear combination  P0 = a*P0_IK + (1-a)*P0_H  (the two contour gauges
     bracket the truth; a chosen to cancel the opposite gradient bias)
 (C) empirical vdW square-gradient fit (mu0(rho)+kappa*rho'' EL condition, single field)
 (D) two-field mini-ladder: pool the dUmax=2 and dUmax=4 density profiles into the
     field-coupling fit (more states -> break the single-field degeneracy)
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, field_coupling as fc, kernel_fit as kf, contour_pressure as cp
import pets_eos as pets
T = 1.198; L = 6.8582414181223398941; Lz = L
pmu = lambda r: T*np.log(r) + pets.properties(T, r)['mu_res']
sc = np.linspace(0.14, 0.55, 40); mp = np.array([pmu(x) for x in sc])


def rms(rg, mu):
    o = np.argsort(rg); rg, mu = np.array(rg)[o], np.array(mu)[o]
    mua = mu + (np.interp(np.median(sc), sc, mp) - np.interp(np.median(sc), rg, mu))
    return np.sqrt(np.mean((np.interp(sc, rg, mua)-mp)**2))


def p0_to_mu0(rho, P0):
    o = np.argsort(rho); rs, ps = np.array(rho)[o], np.array(P0)[o]
    m = (rs > 0.10) & (rs < 0.59); rs, ps = rs[m], ps[m]
    cf = np.polyfit(rs, ps, 4); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(rs.min(), rs.max(), 120)
    return rg, np.concatenate([[0], np.cumsum(0.5*(dP(rg[1:])/rg[1:]+dP(rg[:-1])/rg[:-1])*np.diff(rg))])


# ---------- (A) linear combination of IK and H contours, dUmax=4 ----------
rik, PNik, PTik = cp.ik_profile('cube100u4_ikstress.out', Lz, 6)
rh, PNh, PTh = cp.h_profile('cube100u4_hstress.out', 'cube100u4_dens.out', Lz, L*L, T, 6)
P0ik, P0h = cp.p0_ik(PNik, PTik), cp.p0_ik(PNh, PTh)
rcom = np.linspace(0.10, 0.60, 60)
oik, oh = np.argsort(rik), np.argsort(rh)
P0ik_c = np.interp(rcom, rik[oik], P0ik[oik]); P0h_c = np.interp(rcom, rh[oh], P0h[oh])
print('(A) IK/H linear combination  P0 = a*IK + (1-a)*H:')
best = None
for a in np.linspace(-0.5, 1.5, 41):
    rg, mu = p0_to_mu0(rcom, a*P0ik_c + (1-a)*P0h_c)
    r = rms(rg, mu)
    if best is None or r < best[1]:
        best = (a, r)
print('    pure IK (a=1):   RMS=%.4f' % rms(*p0_to_mu0(rcom, P0ik_c)))
print('    pure H  (a=0):   RMS=%.4f' % rms(*p0_to_mu0(rcom, P0h_c)))
print('    optimal a=%.2f:  RMS=%.4f   <- a in [0,1] would mean H/IK bracket the EOS'
      % (best[0], best[1]))

# ---------- (C) empirical vdW square-gradient fit (single field, dUmax=4) ----------
rho4 = oz.fourier_cosine_smooth(fc._read_density('cube100u4_dens.out'), 6)
prof4 = np.array([np.full_like(rho4, rho4.mean()), rho4])
eg = fc.local_eos(np.array([0.0, 2.0]), prof4, T, Lz, deg=6, smooth=6, grad_spec={2: 0})
print('\n(C) vdW square-gradient fit (mu0+kappa rho'', single dUmax=4):  RMS=%.4f  kappa=%.3f'
      % (rms(eg['rho'], eg['mu0']), eg['kappa2'](0.3)))

# ---------- (D) two-field mini-ladder: dUmax=2 + dUmax=4 ----------
rho2 = oz.fourier_cosine_smooth(fc._read_density('cube100_dens.out'), 6)
amps = np.array([0.0, 1.0, 2.0])                       # AF = dUmax/2 for 0,2,4
prof = np.array([np.full_like(rho2, rho2.mean()), rho2, rho4])
egl = fc.local_eos(amps, prof, T, Lz, deg=6, smooth=6, grad_spec={2: 0, 4: 0})
ekl = kf.kernel_eos(amps, prof, T, Lz, deg=6, smax=2.5, nmodes=3, ridge=1e-3, smooth=6)
print('\n(D) two-field ladder (dUmax=2 + dUmax=4):')
print('    field-coupling gradient: RMS=%.4f   (single-field FC-gradient was ~0.33)'
      % rms(egl['rho'], egl['mu0']))
print('    field-coupling kernel:   RMS=%.4f   (single-field FC-kernel was ~0.58)'
      % rms(ekl['rho'], ekl['mu0']))
