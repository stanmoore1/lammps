#!/usr/bin/env python3
"""
Analyze CPP simulation 1 (dissertation Table 3.1) and compute the induced
surface tension three ways, for comparison with the dissertation:

    gamma_IK   = Lz * <P_zz - (P_xx+P_yy)/2>           (pressure tensor, ~0.311)
    gamma_VdW  = c * INT rho'(z)^2 dz,  c* = 4.4        (van der Waals,    ~0.319)
    gamma_TZ   = 2 * INT psi_IH(z) dz                   (Triezenberg-Zwanzig, ~0.357)

gamma_IK and gamma_VdW are computed here from cpp1_gamma.out and cpp1_density.out.
gamma_TZ uses the OZ-inverted direct correlation function (oz_invert.py, slab mode)
and is wired in once the structure-factor run is validated.
"""

import argparse
import numpy as np


def read_ave_chunk(path):
    """Read a 'fix ave/chunk' file, averaging over all output blocks.
    Returns (coord1, ncount, value) arrays for a single-value chunk file."""
    blocks = []
    cur = None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) == 3:                 # timestep nchunks totalcount
                cur = []
                blocks.append(cur)
            elif cur is not None:
                cur.append([float(x) for x in p])
    arr = np.mean([np.array(b) for b in blocks], axis=0)
    return arr                              # columns: chunk, coord1, ncount, value...


def read_ave_time(path):
    """Read a 'fix ave/time' scalar file -> dict of column means over all rows."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split()])
    a = np.array(rows)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--density', default='cpp1_density.out')
    ap.add_argument('--gamma', default='cpp1_gamma.out')
    ap.add_argument('--lz', type=float, required=True, help='box length in z')
    ap.add_argument('--cvdw', type=float, default=4.4, help='VdW influence parameter c*')
    args = ap.parse_args()

    # gamma_IK from the global pressure tensor (column: step, pN, pT, gam)
    g = read_ave_time(args.gamma)
    pN, pT, gam = g[:, 1].mean(), g[:, 2].mean(), g[:, 3].mean()
    gam_err = g[:, 3].std() / np.sqrt(len(g))
    print(f"gamma_IK  = {gam:.4f} +/- {gam_err:.4f}   "
          f"(<pN>={pN:.4f} <pT>={pT:.4f}, target ~0.311)")

    # density profile rho(z): columns chunk, coord1(reduced), ncount, density
    d = read_ave_chunk(args.density)
    z = d[:, 1] * args.lz
    rho = d[:, 3]
    dz = z[1] - z[0]
    # periodic derivative
    rprime = np.gradient(rho, dz)
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    int_rp2 = trapz(rprime ** 2, dx=dz)
    gam_vdw = args.cvdw * int_rp2
    print(f"gamma_VdW = {gam_vdw:.4f}            "
          f"(c*={args.cvdw}, INT rho'^2 dz={int_rp2:.4f}, target ~0.319)")

    print(f"\nrho(z): min={rho.min():.4f} max={rho.max():.4f} avg={rho.mean():.4f}"
          f"  (target min 0.043, max 0.749, avg 0.437)")


if __name__ == '__main__':
    main()
