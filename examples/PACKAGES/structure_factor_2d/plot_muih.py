#!/usr/bin/env python3
"""Inhomogeneous chemical potential mu_IH(rho) for a CPP simulation, from the
pressure-tensor and TZ (OZ-inversion) methods, optionally compared to a
reference (e.g. LJ EOS) curve.  Recreates dissertation Fig 3.4(b).

Sign convention: the published figure (JCP 134, 114514, Fig. 3b supplemental
data) plots the NEGATIVE of the dissertation's Eq. 3.6/3.7 mu_IH, i.e. the
correction mu0 - mu_int that is ADDED to Widom's mu_int to obtain mu0.  (Check:
Eq. 3.7 gives mu_IH(rho_min) = -c*rho'' < 0 at the density minimum, while the
spreadsheet VdW value there is +0.236.)  This script plots the published
convention:

    pressure tensor:  mu_plot(z) = -(3/2) INT (1/rho) d(PN-PT)/dz dz
    TZ:               psi_IH(z)  = (pi kT/4) rho'(z) INT dz2 rho'(z2) INT ds s^3 C
                      mu_plot(z) = +d psi_IH / d rho
                      (for constant c these reduce to +c*rho'' = -mu_IH^Eq3.7)

Both curves carry one additive constant (the mu_tot reference); they are shifted
to a common value at rho_avg, as in the paper (mu_tot fixed by matching mu0 to
the reference at rho_avg).  Inputs are the fix ave/chunk stress/density files and
the fix ave/time structure-factor file written by in.cpp1_sc.

Example:
    plot_muih.py --lx 13.1777 --lz 26.3554 --temp 1.5 --rho-avg 0.437 \
        --stress cpp1_stress.out --density cpp1_density.out --sf cpp1_sf.out \
        --nbins-sf 40 --ref eos_muih.csv --out muih.png
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import oz_invert as oz


def read_chunk(fn):
    blocks = []; cur = None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 3:
            cur = []; blocks.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return np.mean([np.array(b) for b in blocks], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lx', type=float, required=True)
    ap.add_argument('--lz', type=float, required=True)
    ap.add_argument('--temp', type=float, required=True)
    ap.add_argument('--rho-avg', type=float, required=True)
    ap.add_argument('--stress', default='cpp1_stress.out')
    ap.add_argument('--density', default='cpp1_density.out')
    ap.add_argument('--sf', default='cpp1_sf.out')
    ap.add_argument('--nbins-sf', type=int, default=40)
    ap.add_argument('--kfit', type=float, default=1.5)
    ap.add_argument('--ridge', type=float, default=1e-3)
    ap.add_argument('--ref', help='reference csv: rho,mu_IH (e.g. LJ EOS)')
    ap.add_argument('--out', default='muih.png')
    args = ap.parse_args()
    A = args.lx * args.lx
    kT = args.temp

    mu_ref_avg = 0.0
    if args.ref:
        ref = np.loadtxt(args.ref, delimiter=',', skiprows=1)
        mu_ref_avg = np.interp(args.rho_avg, ref[:, 0], ref[:, 1])

    # ----- pressure-tensor method -----
    st = read_chunk(args.stress)
    de = read_chunk(args.density)
    nb = len(st); dz = args.lz / nb
    zc = (np.arange(nb) + 0.5) * dz
    Vbin = A * dz
    rho_pt = oz.fourier_cosine_smooth(de[:, 3], 10)
    # smooth the normal and tangential pressure components separately
    PN = oz.fourier_cosine_smooth(-st[:, 5] / Vbin, 10)
    PT = oz.fourier_cosine_smooth(-0.5 * (st[:, 3] + st[:, 4]) / Vbin, 10)
    mu_pt_z = -1.5 * np.cumsum(np.gradient(PN - PT, dz) / rho_pt) * dz
    m = zc <= args.lz / 2
    o = np.argsort(rho_pt[m]); r_pt, mu_pt = rho_pt[m][o], mu_pt_z[m][o]
    mu_pt += mu_ref_avg - np.interp(args.rho_avg, r_pt, mu_pt)

    # ----- TZ / OZ-inversion method -----
    sf = oz.read_ave_time_vector(args.sf); nbs = args.nbins_sf
    qs, Smats, rho_s = oz.assemble_matrices(sf, nbs)
    dzs = args.lz / nbs
    rho_tz = oz.fourier_cosine_smooth(rho_s, 8)
    rp = np.gradient(rho_tz, dzs)
    rp[0] = (rho_tz[1] - rho_tz[-1]) / (2 * dzs)
    rp[-1] = (rho_tz[0] - rho_tz[-2]) / (2 * dzs)
    active = np.where(rho_s > 0.1)[0]
    Carr = {q: oz.invert_oz(Smats[q], rho_s, dzs, A, active=active, ridge=args.ridge)[0]
            for q in qs}
    ksmall = qs[qs < args.kfit]; k2 = ksmall ** 2
    deg = min(2, len(ksmall) - 1)         # quadratic in k^2 reduces small-k fit bias
    M2 = np.zeros((nbs, nbs))
    for ia, a in enumerate(active):
        for ib, b in enumerate(active):
            y = np.array([Carr[q][ia, ib] for q in ksmall])
            M2[a, b] = -(2.0 / np.pi) * np.polyfit(k2, y, deg)[-2]
    psi = (np.pi * kT / 4.0) * rp * (dzs * (M2 @ rp))
    rmin, rmax = rho_tz.min(), rho_tz.max()
    keep = (rho_tz > rmin + 0.05) & (rho_tz < rmax - 0.05)
    o = np.argsort(rho_tz[keep])
    rk, pk = rho_tz[keep][o], psi[keep][o]
    dp = np.polyder(np.polyfit(rk, pk, 4))
    r_tz = np.linspace(rk.min(), rk.max(), 100)
    mu_tz = np.polyval(dp, r_tz)
    mu_tz += mu_ref_avg - np.interp(args.rho_avg, r_tz, mu_tz)

    # ----- plot -----
    plt.figure(figsize=(7, 5))
    if args.ref:
        plt.plot(ref[:, 0], ref[:, 1], 'k-', lw=2, label='reference (LJ EOS)')
    plt.plot(r_pt, mu_pt, 'o-', ms=3, color='tab:blue', label='pressure tensor')
    plt.plot(r_tz, mu_tz, 's-', ms=4, color='tab:red', label='TZ / OZ inversion')
    plt.axhline(0, color='gray', lw=0.5)
    plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_{IH}^*$')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
