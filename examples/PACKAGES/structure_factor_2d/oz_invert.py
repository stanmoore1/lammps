#!/usr/bin/env python3
"""
Ornstein-Zernike inversion of the bin-resolved planar structure factor produced
by LAMMPS "compute structure/factor/2d" (proof of concept).

Pipeline (per in-plane wavevector magnitude k):

    G_ij(k) = <rho_hat_i(k) conj(rho_hat_j(k))> / A          (time-avg from fix ave/time)
    h_ij(k) = (G_ij(k) - rho_i*dz*delta_ij) / (rho_i*rho_j*dz^2)
    C(k)    = h(k) @ inv(I + D @ h(k)),   D = diag(rho_l*dz)

C_ij(k) is the in-plane Fourier transform of the (inhomogeneous) direct
correlation function C(z_i, z_j, s).

bulk mode : homogeneous fluid -- check that C_ij is translationally invariant in
            z and that its z-Fourier transform reproduces the bulk c(k); optional
            cross-check against (1 - 1/S(q))/rho from an rdf file.
slab mode : liquid-vapor slab -- extract the TZ second moment from the small-k
            slope of C_ij(k), then psi_IH(z) (Eq. 3.33) and the surface tension
            (Eq. 3.32) of the dissertation.

Time-averaging is done by fix ave/time in the LAMMPS input; this script does only
the normalization and linear algebra.
"""

import argparse
import sys
import numpy as np


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------

def read_ave_time_vector(path):
    """Read a 'fix ave/time mode vector' file and average over ALL blocks.

    Each Nfreq window is written as its own block; averaging the blocks uses the
    whole production run.  Returns a 2D array (nrows, ncols) of the averaged
    column values (the leading per-row index column is stripped)."""
    blocks = []
    cur = None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:           # "timestep nrows" -> start of a block
                cur = []
                blocks.append(cur)
            elif cur is not None:
                cur.append([float(x) for x in parts[1:]])   # drop row index
    if not blocks:
        sys.exit(f"no data blocks found in {path}")
    return np.mean([np.array(b, dtype=float) for b in blocks], axis=0)


def assemble_matrices(sf, nbins):
    """Group the [q, ibin, jbin, S_ij, density] rows into S(q) matrices.

    Returns (qs, Smats, rho) where qs is the sorted unique |k| list (excluding
    q=0), Smats[q] is the nbins x nbins matrix, and rho[i] is the per-bin density
    taken from the q==0 rows."""
    q, ib, jb, S, dens = sf[:, 0], sf[:, 1], sf[:, 2], sf[:, 3], sf[:, 4]
    ib = ib.astype(int)
    jb = jb.astype(int)

    rho = np.zeros(nbins)
    qvals = sorted({round(qq, 10) for qq in q if qq > 0})
    Smats = {qq: np.zeros((nbins, nbins)) for qq in qvals}

    for row in range(sf.shape[0]):
        i, j = ib[row], jb[row]
        if q[row] <= 0.0:
            rho[i] = dens[row]          # density stored on q==0 rows
        else:
            Smats[round(q[row], 10)][i, j] = S[row]
    return np.array(qvals), Smats, rho


# ----------------------------------------------------------------------------
# OZ inversion
# ----------------------------------------------------------------------------

def invert_oz(Smat, rho, dz, area, active=None, ridge=0.0):
    """Return C_ij(k) from one S_ij(k) matrix via the inhomogeneous OZ equation.
    `ridge` adds Tikhonov regularization for ill-conditioned (low-density) bins."""
    nb = Smat.shape[0]
    if active is None:
        active = np.arange(nb)
    S = Smat[np.ix_(active, active)]
    r = rho[active]

    G = S / area
    rr = np.outer(r, r)
    h = (G - np.diag(r * dz)) / (rr * dz * dz)
    h = 0.5 * (h + h.T)                       # symmetrize
    D = np.diag(r * dz)
    M = np.eye(len(active)) + D @ h           # C = h (I + D h)^-1
    cond = np.linalg.cond(M)
    if ridge > 0.0:
        # regularized right-inverse: C = h (M^T M + ridge I)^-1 M^T  ->  h M^-1
        C = h @ np.linalg.solve(M.T @ M + ridge * np.eye(len(active)), M.T)
    else:
        try:
            C = np.linalg.solve(M.T, h.T).T
        except np.linalg.LinAlgError:         # ill-conditioned -> pseudo-inverse
            C = h @ np.linalg.pinv(M)
    return C, cond


def fourier_cosine_smooth(y, nmodes):
    """Least-squares fit of a profile to a Fourier cosine series
    sum_k a_k cos(2 pi k z / L), k = 0..nmodes (the dissertation's smoothing).
    The cosine basis is even about the box center, so the fit automatically
    symmetrizes the data about z = L/2 (averaging the two symmetric halves) in
    addition to low-pass smoothing."""
    n = len(y)
    z = (np.arange(n) + 0.5) / n                  # bin centers in units of L
    B = np.cos(2.0 * np.pi * np.outer(z, np.arange(nmodes + 1)))
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return B @ coef


# ----------------------------------------------------------------------------
# bulk validation
# ----------------------------------------------------------------------------

def circulant_deviation(C):
    """Max deviation of C_ij from depending only on (i-j) mod nbins, relative to
    the overall scale max|C| (so near-zero off-diagonals don't dominate)."""
    nb = C.shape[0]
    scale = np.abs(C).max() + 1e-12
    dev = 0.0
    for m in range(nb):
        diag = np.array([C[i, (i + m) % nb] for i in range(nb)])
        dev = max(dev, (diag.max() - diag.min()) / scale)
    return dev


def s_of_q_from_rdf(rdf, rho):
    """Bulk S(q) from g(r): compute rdf array has columns [bin, r, g(r), coord].
    S(q) = 1 + 4 pi rho INT [g(r)-1] sin(qr)/(qr) r^2 dr."""
    trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    r = rdf[:, 1]
    g = rdf[:, 2]
    mask = r > 0.0
    r, g = r[mask], g[mask]
    dr = r[1] - r[0]
    qs = np.linspace(0.5, 8.0, 40)
    S = np.empty_like(qs)
    for n, q in enumerate(qs):
        integ = (g - 1.0) * np.sin(q * r) / (q * r) * r * r
        S[n] = 1.0 + 4.0 * np.pi * rho * trapz(integ, dx=dr)
    return qs, S


def run_bulk(args):
    sf = read_ave_time_vector(args.sf_file)
    qs, Smats, rho = assemble_matrices(sf, args.nbins)
    dz = args.lz / args.nbins
    area = args.lx * args.lx
    print(f"# bulk OZ inversion: nbins={args.nbins} dz={dz:.4f} "
          f"rho_mean={rho.mean():.4f} (spread {rho.max()-rho.min():.2e})")

    Cbar = {}                               # circulant-averaged C_m(q)
    for q in qs:
        C, cond = invert_oz(Smats[q], rho, dz, area)
        dev = circulant_deviation(C)
        nb = args.nbins
        Cm = np.array([np.mean([C[i, (i + m) % nb] for i in range(nb)])
                       for m in range(nb)])
        Cbar[q] = Cm
        print(f"  k={q:6.3f}  cond(I+Dh)={cond:8.2f}  "
              f"circulant_dev={dev:7.3f}  C_0={Cm[0]: .4e}")

    # z-Fourier transform at the smallest k to recover bulk c(k_par, k_z)
    q0 = qs[0]
    nb = args.nbins
    kz = 2.0 * np.pi * np.arange(nb) / args.lz
    chat = dz * np.array([np.sum(Cbar[q0] * np.exp(-1j * kz_n * np.arange(nb) * dz))
                          for kz_n in kz]).real
    kmag = np.sqrt(q0 * q0 + kz * kz)
    print("\n# recovered bulk c_hat(k) along k_par = %.3f (z-FT):" % q0)
    for k, c in sorted(zip(kmag, chat)):
        print(f"    k={k:6.3f}   c_hat={c: .4e}")

    if args.rdf_file:
        rdf = read_ave_time_vector(args.rdf_file)
        qref, Sref = s_of_q_from_rdf(rdf, rho.mean())
        cref = (1.0 - 1.0 / Sref) / rho.mean()
        print("\n# reference (1-1/S(q))/rho from rdf:")
        for q, s, c in zip(qref, Sref, cref):
            print(f"    q={q:6.3f}   S={s:6.3f}  c_ref={c: .4e}")
        if args.plot:
            _plot_bulk(kmag, chat, qref, cref)


def _plot_bulk(kmag, chat, qref, cref):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("# matplotlib not available, skipping plot")
        return
    plt.plot(qref, cref, '-', label='(1-1/S)/rho (rdf)')
    plt.plot(kmag, chat, 'o', label='OZ inversion (z-FT)')
    plt.xlabel('k'); plt.ylabel('c_hat(k)'); plt.legend()
    plt.savefig('bulk_c_compare.png', dpi=120)
    print("# wrote bulk_c_compare.png")


# ----------------------------------------------------------------------------
# slab: TZ correction
# ----------------------------------------------------------------------------

def run_slab(args):
    sf = read_ave_time_vector(args.sf_file)
    qs, Smats, rho = assemble_matrices(sf, args.nbins)
    dz = args.lz / args.nbins
    area = args.lx * args.lx
    kT = args.temp

    # Invert only on bins with appreciable density (h ~ 1/rho^2 blows up in vacuum);
    # vacuum bins get c = 0, which is fine because rho'(z) ~ 0 there so they carry no
    # weight in the TZ integral.  rho'(z) itself uses the full smoothed profile.
    rho_s = fourier_cosine_smooth(rho, args.smooth)
    na = args.nbins
    active = np.where(rho > args.rho_min)[0]
    print(f"# slab OZ inversion: {len(active)}/{na} active bins (rho>{args.rho_min}), "
          f"ridge={args.ridge}, rho_smooth {rho_s.min():.3f}-{rho_s.max():.3f}")

    Carr = {q: invert_oz(Smats[q], rho, dz, area, active=active, ridge=args.ridge)[0]
            for q in qs}

    # second moment from the small-k behavior: C_ij(k) = C_ij(0) - (pi/2) k^2 M2_ij
    # + O(k^4).  CAUTION: the smallest accessible k = 2*pi/Lx is often NOT in the
    # asymptotic k^2 regime (J0(ks) deviates from 1-(ks)^2/4 for ks ~ 1, and the
    # s^3-weighted c(s) extends to several sigma), which biases a linear fit LOW.
    # A polynomial in k^2 of order --fit-order partially corrects this; the real
    # fix is a larger in-plane box (smaller k_min).
    ksmall = qs[qs < args.kfit]
    deg = min(args.fit_order, len(ksmall) - 1)
    if deg < 1:
        sys.exit("not enough small-k points to fit the second moment; raise --kfit")
    k2 = ksmall ** 2
    M2 = np.zeros((na, na))                            # full grid, zeros in vacuum
    for ia, a in enumerate(active):
        for ib, b in enumerate(active):
            y = np.array([Carr[q][ia, ib] for q in ksmall])
            coef = np.polyfit(k2, y, deg)
            M2[a, b] = -(2.0 / np.pi) * coef[-2]      # k^2 coefficient -> INT ds s^3 C

    # density gradient from the smoothed, periodic profile
    z = (np.arange(na) + 0.5) * dz
    rprime = np.gradient(rho_s, dz)
    rprime[0] = (rho_s[1] - rho_s[-1]) / (2 * dz)     # periodic ends
    rprime[-1] = (rho_s[0] - rho_s[-2]) / (2 * dz)
    rho = rho_s

    # psi_IH(z_i) = (pi kT/4) rho'_i sum_j dz rho'_j INT ds s^3 C_ij   (Eq. 3.33,
    # consistent with the surface tension Eq. 3.32 via gamma = 2 INT psi_IH dz).
    # rho' is in the numerator, so psi_IH -> 0 at the density extrema as it should.
    psi = (np.pi * kT / 4.0) * rprime * (dz * (M2 @ rprime))

    # surface tension (Eq. 3.32): gamma = (pi/2) kT sum_i sum_j dz^2 rho'_i rho'_j M2_ij
    gamma_tz = 0.5 * np.pi * kT * dz * dz * (rprime @ M2 @ rprime)

    print("\n# z        rho        rho'       psi_IH")
    for a in range(na):
        print(f"  {z[a]:7.3f}  {rho[a]:8.4f}  {rprime[a]: .4e}  {psi[a]: .4e}")
    print(f"\n# TZ surface tension (Eq. 3.32), z-integrals over the WHOLE cell:")
    print(f"#   gamma_full_cell    = {gamma_tz:.4f}   (compare to Lz*<Pzz-0.5(Pxx+Pyy)>)")
    print(f"#   gamma_per_interface = {0.5*gamma_tz:.4f}  (full cell / 2: two gradient regions)")
    print("# NOTE: the small-k second-moment fit is biased low when 2*pi/Lx is not")
    print("# asymptotically small; converge with larger Lx and check --fit-order/--kfit.")


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['bulk', 'slab'], required=True)
    p.add_argument('--sf-file', required=True, help='fix ave/time output of c_sf[*]')
    p.add_argument('--nbins', type=int, required=True)
    p.add_argument('--lx', type=float, required=True, help='box length in x (= y)')
    p.add_argument('--lz', type=float, required=True, help='box length in z')
    p.add_argument('--temp', type=float, default=1.0, help='reduced temperature kT')
    p.add_argument('--rdf-file', help='[bulk] rdf ave/time file for the S(q) cross-check')
    p.add_argument('--rho-min', type=float, default=0.05,
                   help='[slab] drop bins with density below this (vapor)')
    p.add_argument('--kfit', type=float, default=2.0,
                   help='[slab] fit C_ij(k) vs k^2 for k below this')
    p.add_argument('--fit-order', type=int, default=2,
                   help='[slab] polynomial order in k^2 for the second-moment fit')
    p.add_argument('--smooth', type=int, default=6,
                   help='[slab] number of cosine modes to smooth rho(z)')
    p.add_argument('--ridge', type=float, default=1e-4,
                   help='[slab] Tikhonov regularization for the OZ inversion')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()

    if args.mode == 'bulk':
        run_bulk(args)
    else:
        run_slab(args)


if __name__ == '__main__':
    main()
