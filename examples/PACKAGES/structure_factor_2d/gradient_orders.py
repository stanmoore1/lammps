#!/usr/bin/env python3
"""Quantify the contribution of the 4th- and higher-order density-gradient terms
relative to the 2nd-order term in the interfacial chemical potential

    mu_IH(z) = -INT ds C(s)[rho(z+s)-rho(z)]
             = -(m2/2) rho'' - (m4/24) rho'''' - (m6/720) rho^(6) - ...   (C even)

m_2k = INT s^2k C(s) ds are the even moments of the data-driven nonlocal kernel
(field ladder, hybrid fit).  Two measures, per field strength (interface sharpness):
  (a) RMS size of each order's term on the real profile, as a % of the 2nd-order term;
  (b) fraction of the variance of the FULL nonlocal response mu_IH(z) captured by the
      2nd-order and by the (2nd+4th)-order truncations."""
import sys
import numpy as np
sys.path.insert(0, '/home/user/lammps/ljts_eos')
import field_coupling as fc, kernel_fit as kf
import oz_invert as oz

T = 0.980; rho_avg = 0.31; sm = 10


def box_dims(rho, N=1000, aspect=3.0):
    V = N / rho; Lx = (V / aspect) ** (1.0 / 3.0); return Lx, aspect * Lx


def load(Tstr, smooth=sm):
    rows = [l.split(',') for l in open('ladder_T%s.csv' % Tstr) if l.strip() and not l.startswith('#')]
    dus = [float(r[0]) for r in rows]; files = [r[2].strip() for r in rows]
    profs = [oz.fourier_cosine_smooth(fc._read_density(f), smooth) for f in files]
    amps = np.array([0.0] + [d / 2.0 for d in dus])
    return amps, dus, np.array([np.full_like(profs[0], np.mean(profs[0]))] + profs)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


Lx, Lz = box_dims(rho_avg)
amps, dus, profiles = load('0.980')

# one pooled hybrid kernel fit -> the kernel shape C(s) and its even moments
ek = kf.kernel_eos(amps, profiles, T, Lz, deg=6, smax=2.5, nmodes=3, ridge=1e-3, smooth=sm)
sg, C = ek['s'], ek['C']
m2 = np.trapezoid(sg ** 2 * C, sg)
m4 = np.trapezoid(sg ** 4 * C, sg)
m6 = np.trapezoid(sg ** 6 * C, sg)
kap2 = ek['kappa2_eff']                 # = kap2_loc + m2/2 (local backbone + kernel)
c2, c4, c6 = kap2, m4 / 24.0, m6 / 720.0  # |coef| of rho'', rho'''', rho^(6) in mu_IH
print('data-driven kernel moments:  m2=%.3f m4=%.3f m6=%.3f' % (m2, m4, m6))
print('mu_IH order coefficients:    kappa2_eff=%.3f  kappa4=m4/24=%.4f  kappa6=m6/720=%.5f'
      % (c2, c4, c6))
print('(kappa2 split: local backbone %.3f + kernel %.3f)\n'
      % (ek['kappa2_loc'], m2 / 2.0))

print('per field: term sizes in mu_IH and truncation variance explained')
print('  dU   max|rho\'|   RMS(2nd)   4th/2nd   6th/2nd   var%%(2nd)  var%%(2+4)')
for k in range(1, len(profiles)):
    p = profiles[k]
    d2 = fc.fderiv(p, sm, Lz, 2)
    d4 = fc.fderiv(p, sm, Lz, 4)
    d6 = fc.fderiv(p, sm, Lz, 6)
    d1 = fc.fderiv(p, sm, Lz, 1)
    t2 = -c2 * d2
    t4 = -c4 * d4
    t6 = -c6 * d6
    # full nonlocal response actually used by the fit (backbone + kernel modes)
    full = -ek['kappa2_loc'] * d2 + kf._kernel_columns(p, Lz / len(p), Lz, 2.5, 3, True) @ ek['a']
    mu2 = t2
    mu24 = t2 + t4
    v = np.var(full)
    ve2 = 100.0 * (1.0 - np.var(full - mu2) / v)
    ve24 = 100.0 * (1.0 - np.var(full - mu24) / v)
    print('  %.1f   %7.4f   %8.4f   %6.1f%%   %6.1f%%   %7.1f%%   %8.1f%%'
          % (dus[k - 1], rms(d1), rms(t2),
             100 * rms(t4) / rms(t2), 100 * rms(t6) / rms(t2), ve2, ve24))
