#!/usr/bin/env python3
"""Local pressure-tensor / contour analysis for the second-order homogeneous
pressure and the IK-vs-Harasima accuracy diagnostic (paper thesis point 2).

Gradient theory expands the local pressure tensor of a planar interface to second
order in density gradients.  Along the Irving-Kirkwood (IK) contour,
    PN(z) = P0(z) + kappa [1/2 rho'^2 - rho'' rho] + O(grad^4),
    PT^IK(z) = P0(z) + (1/3) kappa [1/2 rho'^2 - rho'' rho] + O(grad^4),
so the influence parameter cancels in
    P0(z) = 3/2 PT^IK(z) - 1/2 PN(z) + O(grad^4).                      (IK)
The NORMAL pressure PN(z) is contour-independent (it is mechanically defined across
a plane); only the TANGENTIAL pressure differs between contours.  The Harasima (H)
tangential profile has a different gradient coefficient, so the linear combination
that recovers P0 from (PN, PT^H) carries a different weight,
    P0(z) = PN(z) - (1/c_H)[PN(z) - PT^H(z)],                          (H)
where c_H must be obtained from the H-contour gradient expansion.  ** c_H is left as
a parameter (default = the IK value 2/3, i.e. weight 3/2) and MUST be validated
against the synthetic/real H profiles before use -- this is the one derivation the
drafts flag as uncertain. **  The agreement of P0^IK and P0^H is the parameter-free
gauge of the second-order truncation: small where the interface is wide (near Tc),
growing deep in the dome.

For many-body / ML potentials the per-pair IK contour is undefined; only the
Harasima profile (per-atom virial binned at atom positions, from compute
stress/atom) is available, so the IK-H gauge is lost and the exact field-coupling
method is required.

Self-test: python3 contour_pressure.py --selftest  (synthetic IK gradient-theory
profiles; 3/2 PT - 1/2 PN recovers P0 exactly).
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import oz_invert as oz


def read_chunk(fn):
    """Block-averaged fix ave/chunk array (rows = bins, cols as written)."""
    blocks, cur = [], None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 3:
            cur = []; blocks.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return np.mean([np.array(b) for b in blocks], axis=0)


def stress_profile(stress_file, density_file, Lz, area, smooth=10):
    """z, rho, PN(z), PT(z) from a fix ave/chunk `compute stress/atom ... norm none`
    file (Harasima localization) and a density file.  P_aa(z) = -sum(stress_aa)/Vbin;
    PN=Pzz, PT=(Pxx+Pyy)/2.  Components are cosine-smoothed."""
    st = read_chunk(stress_file); de = read_chunk(density_file)
    nb = len(st); Vbin = area * (Lz / nb)
    rho = oz.fourier_cosine_smooth(de[:, 3], smooth)
    PN = oz.fourier_cosine_smooth(-st[:, 5] / Vbin, smooth)
    PT = oz.fourier_cosine_smooth(-0.5 * (st[:, 3] + st[:, 4]) / Vbin, smooth)
    return rho, PN, PT


def p0_ik(PN, PT):
    """Second-order homogeneous pressure, IK contour: P0 = 3/2 PT - 1/2 PN."""
    return 1.5 * PT - 0.5 * PN


def p0_contour(PN, PT, c=2.0 / 3.0):
    """General contour combination P0 = PN - (PN - PT)/c.  c = 2/3 reproduces the IK
    weight (3/2, -1/2).  For the Harasima contour pass the H-contour coefficient
    (TО BE DERIVED/validated -- see module docstring)."""
    return PN - (PN - PT) / c


def diagnostic(rho, PN_ik, PT_ik, PN_h, PT_h, c_h=2.0 / 3.0):
    """P0 from the IK and Harasima combinations and their difference vs rho -- the
    second-order accuracy gauge.  (PN is shared; pass the same array twice if only
    the tangential profiles differ.)"""
    P0i = p0_ik(PN_ik, PT_ik)
    P0h = p0_contour(PN_h, PT_h, c_h)
    return dict(rho=rho, P0_ik=P0i, P0_h=P0h, diff=P0i - P0h)


def mu_ih_from_pressure(rho, PN, PT, Lz, smooth=10):
    """Inhomogeneous chemical-potential correction from the pressure-tensor route
    (dissertation Eq. 3.23): mu_IH(z) = -3/2 INT (1/rho) d(PN-PT)/dz dz, using the
    analytic (sine-series) derivative of the cosine-fitted PN-PT."""
    nb = len(rho); dz = Lz / nb
    dPdz = oz.fourier_cosine_deriv(PN - PT, smooth, Lz)
    return -1.5 * np.cumsum(dPdz / rho) * dz


# ----------------------------------------------------------------------------

def _selftest():
    Lz = 26.0; nb = 200
    z = (np.arange(nb) + 0.5) / nb * Lz
    rho = 0.40 + 0.20 * np.cos(2 * np.pi * z / Lz)
    P0 = rho - 1.5 * rho ** 2 + 0.8 * rho ** 3            # arbitrary smooth EOS
    rp = oz.fourier_cosine_deriv(rho, 10, Lz)
    # second derivative via cosine series
    coef, zc = oz.fourier_cosine_coef(rho, 10); k = np.arange(11)
    rpp = (-np.cos(2 * np.pi * np.outer(zc, k)) * (2 * np.pi * k / Lz) ** 2) @ coef
    kap = 2.0
    g = 0.5 * rp ** 2 - rpp * rho                          # gradient combination
    PN = P0 + kap * g
    PT_ik = P0 + (1.0 / 3.0) * kap * g
    rec = p0_ik(PN, PT_ik)
    err = np.max(np.abs(rec - P0))
    # the same IK weight applied to a Harasima-type profile (different coefficient,
    # say PT^H with ratio 1/2 instead of 1/3) does NOT recover P0 -> motivates c_H
    PT_h = P0 + 0.5 * kap * g
    err_wrong = np.max(np.abs(p0_ik(PN, PT_h) - P0))
    print("IK: max|3/2 PT - 1/2 PN - P0| = %.2e (should be ~0)" % err)
    print("applying the IK weight to a Harasima profile mis-estimates P0 by %.3e "
          "-> the H coefficient c_H must be derived" % err_wrong)
    print("  -> IK contour P0 recovery validated"
          if err < 1e-10 else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stress'); ap.add_argument('--density')
    ap.add_argument('--lx', type=float); ap.add_argument('--lz', type=float)
    ap.add_argument('--smooth', type=int, default=10)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    rho, PN, PT = stress_profile(args.stress, args.density, args.lz,
                                 args.lx ** 2, args.smooth)
    P0 = p0_ik(PN, PT)
    m = (np.arange(len(rho)) + 0.5) * (args.lz / len(rho)) <= args.lz / 2
    o = np.argsort(rho[m])
    print('# rho      P0(3/2PT-1/2PN, IK weight)')
    for r, p in zip(rho[m][o][::4], P0[m][o][::4]):
        print('  %6.3f  % .4f' % (r, p))


if __name__ == '__main__':
    main()
