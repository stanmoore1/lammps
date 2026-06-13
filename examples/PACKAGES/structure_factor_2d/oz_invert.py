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
slab mode : liquid-vapor / CPP slab -- extract the TZ second moment from the
            small-k slope of C_ij(k), then psi_IH(z) (Eq. 3.33) and the surface
            tension (Eq. 3.32) of the dissertation.
kb mode   : Kirkwood-Buff / compressibility route to the homogeneous chemical
            potential mu0(rho), from the k->0 INTERCEPT of the same C_ij(k)
            (c_hat(0; rho) = dz sum_j C_ij(0); beta dmu0/drho = 1/rho - c_hat(0)).
            An independent third route to mu0 alongside the pressure-tensor and
            TZ methods (Nichols, Moore & Wheeler, Phys. Rev. E 80, 051203 (2009)).

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


def fourier_cosine_coef(y, nmodes):
    """Least-squares coefficients a_k of the Fourier cosine series
    sum_k a_k cos(2 pi k z / L), k = 0..nmodes, fit at the bin centers."""
    n = len(y)
    z = (np.arange(n) + 0.5) / n                  # bin centers in units of L
    B = np.cos(2.0 * np.pi * np.outer(z, np.arange(nmodes + 1)))
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return coef, z


def fourier_cosine_smooth(y, nmodes):
    """Fourier cosine-series fit of a profile (the dissertation's smoothing).
    The cosine basis is even about the box center, so the fit automatically
    symmetrizes the data about z = L/2 (averaging the two symmetric halves) in
    addition to low-pass smoothing."""
    coef, z = fourier_cosine_coef(y, nmodes)
    B = np.cos(2.0 * np.pi * np.outer(z, np.arange(nmodes + 1)))
    return B @ coef


def fourier_cosine_deriv(y, nmodes, length):
    """Analytic z-derivative of the cosine-series fit, evaluated at the bin
    centers.  The derivative of a cosine series is a SINE series:
        d/dz sum_k a_k cos(2 pi k z/L) = -sum_k a_k (2 pi k/L) sin(2 pi k z/L),
    which is exactly periodic and antisymmetric about z = L/2 (no
    finite-difference error, no special handling of the periodic ends)."""
    coef, z = fourier_cosine_coef(y, nmodes)
    karr = np.arange(nmodes + 1)
    Bd = -np.sin(2.0 * np.pi * np.outer(z, karr)) * (2.0 * np.pi * karr / length)
    return Bd @ coef


def _lj_u(r):
    """LJ pair potential, eps = sigma = 1."""
    inv6 = r ** -6
    return 4.0 * (inv6 * inv6 - inv6)


def mean_field_tail(dz, kvals, beta, r_split, smax, ns=4000):
    """Analytic mean-field (RPA) tail of the in-plane direct correlation function
    between two z-planes separated by dz:

        C_tail(s) = -beta u(sqrt(dz^2 + s^2))  for sqrt(dz^2+s^2) > r_split, else 0

    Returns (M2_tail, Chat_tail(kvals)) with
        M2_tail   = INT_0^inf  s^3 C_tail(s) ds          (the s^3 second moment)
        Chat_tail = 2 pi INT_0^inf s J0(ks) C_tail(s) ds (its in-plane transform).

    The s^3-weighted second moment of c is DOMINATED by this attractive tail, which
    is known exactly from the potential; computing it analytically (instead of from
    a biased, noisy small-k slope of the MD c) is what makes the TZ moment robust to
    noise and to a large k_min = 2 pi / Lx in small systems."""
    from scipy.special import j0
    s = np.linspace(1e-3, smax, ns)
    r = np.sqrt(dz * dz + s * s)
    Ct = np.where(r > r_split, -beta * _lj_u(r), 0.0)
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    M2 = trapz(s ** 3 * Ct, s)
    Chat = np.array([2.0 * np.pi * trapz(s * j0(k * s) * Ct, s) for k in kvals])
    return M2, Chat


def mirror_symmetrize(Smats, rho):
    """Average S_ij(k) and rho_i with their mirror images about the box center
    (z -> Lz - z), which is an exact symmetry when U_ext is even about z = Lz/2.
    Halves the noise in the matrices BEFORE the (nonlinear) OZ inversion and makes
    the two interface peaks exactly equivalent."""
    Sm = {q: 0.5 * (S + S[::-1, ::-1].copy()) for q, S in Smats.items()}
    return Sm, 0.5 * (rho + rho[::-1])


def second_moment(qs, Carr, active, nbins, kfit, dzs, temp, lx,
                  fit_order=2, tail_rsplit=0.0):
    """M2_ij = INT ds s^3 C_ij(s) for every bin pair, from the small-k behaviour of
    the inverted C_ij(k): C_ij(k) = C_ij(0) - (pi/2) k^2 M2_ij + O(k^4).

    With tail_rsplit > 0 the long-range (r > tail_rsplit) part of c is replaced by
    the analytic mean-field tail and only the SHORT-RANGE residual is fit from the
    data; the residual is smooth in k (short-ranged in s), so a low-order fit from
    a coarse k-grid is unbiased and noise-robust."""
    ks = qs[qs < kfit]
    k2 = ks ** 2
    deg = min(fit_order, len(ks) - 1)
    if deg < 1:
        sys.exit("not enough small-k points; raise --kfit")
    M2 = np.zeros((nbins, nbins))
    tail = {}
    if tail_rsplit > 0.0:
        beta = 1.0 / temp
        smax = lx / 2.0                       # in-plane half-box
        for m in range(nbins):
            dz = (((m + nbins // 2) % nbins) - nbins // 2) * dzs   # min-image
            tail[m] = mean_field_tail(abs(dz), ks, beta, tail_rsplit, smax)
    for ia, a in enumerate(active):
        for ib, b in enumerate(active):
            y = np.array([Carr[q][ia, ib] for q in ks])
            if tail_rsplit > 0.0:
                M2t, Ct = tail[abs(a - b)]
                M2[a, b] = M2t - (2.0 / np.pi) * np.polyfit(k2, y - Ct, deg)[-2]
            else:
                M2[a, b] = -(2.0 / np.pi) * np.polyfit(k2, y, deg)[-2]
    return M2


def local_chat0(qs, Carr, active, nbins, kfit, dz):
    """Local 3D direct correlation at zero wavevector for each z-bin,
        c_hat(0; rho_i) = dz * sum_j C_ij(k->0),
    i.e. the row sum of the k=0 intercepts of the inverted in-plane direct
    correlation matrix (the z-integral of c at the local density).  Combined with
    the compressibility equation beta dP/drho = 1 - rho c_hat(0) this gives the
    local beta dmu0/drho = 1/rho - c_hat(0) -- the Kirkwood-Buff / fluctuation
    route to the homogeneous chemical potential (Nichols, Moore & Wheeler,
    Phys. Rev. E 80, 051203 (2009)).  The intercept needs no tail correction:
    unlike the s^3 second moment it is set by the short-range core, not the tail."""
    ks = qs[qs < kfit]
    k2 = ks ** 2
    aidx = {a: i for i, a in enumerate(active)}
    chat0 = np.full(nbins, np.nan)
    for a in active:
        tot = 0.0
        for b in active:
            y = np.array([Carr[q][aidx[a], aidx[b]] for q in ks])
            tot += np.polyfit(k2, y, 2)[-1]          # constant term = C_ab(0)
        chat0[a] = dz * tot
    return chat0


def kb_chemical_potential(rho_bins, chat0, temp, h_star=0.183, poly=5):
    """mu0(rho) from c_hat(0; rho): integrate  beta dmu_ex/drho = -c_hat(0)  and add
    the ideal-gas part mu_id = T ln[(h*^2/(2 pi T))^(3/2) rho].  Returns (rho_grid,
    mu0) up to one additive constant (mu_tot), fixed by matching to a reference at
    one density."""
    m = ~np.isnan(chat0)
    r, c = rho_bins[m], chat0[m]
    o = np.argsort(r)
    cfit = np.poly1d(np.polyfit(r[o], c[o], poly))
    rg = np.linspace(r.min(), r.max(), 300)
    bmu_ex = -np.concatenate([[0.0], np.cumsum(0.5 * (cfit(rg[1:]) + cfit(rg[:-1]))
                                               * np.diff(rg))])
    mu_id = temp * np.log((h_star ** 2 / (2.0 * np.pi * temp)) ** 1.5 * rg)
    return rg, mu_id + temp * bmu_ex


def intercept_matrix(qs, Carr, active, kfit):
    """Full matrix of k->0 intercepts C_ij(0) = lim_{k->0} C_ij(k) (in-plane
    integral of the direct correlation function between bins i and j)."""
    ks = qs[qs < kfit]
    k2 = ks ** 2
    na = len(active)
    C0 = np.empty((na, na))
    for ia in range(na):
        for ib in range(na):
            y = np.array([Carr[q][ia, ib] for q in ks])
            C0[ia, ib] = np.polyfit(k2, y, 2)[-1]      # constant term
    return C0


def dft_mu_ih(C0, rho_active, dz, temp):
    """Inhomogeneous chemical-potential correction from the EXACT nonlocal DFT
    relation (Evans 1979), one-shot (lambda=1) approximation:

        beta mu_IH(z_i) = - sum_j dz C_ij(0) [rho(z_j) - rho(z_i)]

    Unlike the TZ second moment this uses the FULL z,z' structure of C_ij(0)
    (not just its s^2 moment) and the density DIFFERENCE rather than rho'(z)rho'(z'),
    so it is exact in the gradient to all orders -- agreeing with TZ only at second
    order.  It uses the robust k=0 intercept (no tail correction).  The one-shot
    form is exact for weak inhomogeneity; for a strong density swing do a
    thermodynamic integration over the field strength (the lambda path)."""
    one = np.ones(len(rho_active))
    return -temp * dz * (C0 @ rho_active - rho_active * (C0 @ one))


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
    if not args.no_mirror:
        Smats, rho = mirror_symmetrize(Smats, rho)
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

    # second moment INT ds s^3 C_ij from the small-k behaviour of C_ij(k).  With
    # --tail-rsplit the s^3-dominant long-range part is taken analytically from the
    # mean-field tail -beta u(r) and only the short-range residual is fit -- this
    # removes the small-k (large k_min) bias and is robust to noise / small boxes.
    M2 = second_moment(qs, Carr, active, na, args.kfit, dz, kT, args.lx,
                       fit_order=args.fit_order, tail_rsplit=args.tail_rsplit)

    # density gradient: analytic derivative of the cosine fit (a sine series)
    z = (np.arange(na) + 0.5) * dz
    rprime = fourier_cosine_deriv(rho, args.smooth, args.lz)
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
# slab: KB / compressibility route to the homogeneous chemical potential mu0(rho)
# ----------------------------------------------------------------------------

def run_kb(args):
    sf = read_ave_time_vector(args.sf_file)
    qs, Smats, rho = assemble_matrices(sf, args.nbins)
    if not args.no_mirror:
        Smats, rho = mirror_symmetrize(Smats, rho)
    dz = args.lz / args.nbins
    area = args.lx * args.lx
    active = np.where(rho > args.rho_min)[0]
    Carr = {q: invert_oz(Smats[q], rho, dz, area, active=active, ridge=args.ridge)[0]
            for q in qs}

    # local c_hat(0; rho) = dz sum_j C_ij(0) and the implied beta dmu0/drho
    chat0 = local_chat0(qs, Carr, active, args.nbins, args.kfit, dz)
    rg, mu0 = kb_chemical_potential(rho, chat0, args.temp, h_star=args.hstar)

    print("# KB / compressibility route (PRE 80, 051203):")
    print("# rho     c_hat(0)   beta*dmu0/drho")
    for a in active:
        if np.isnan(chat0[a]):
            continue
        print(f"  {rho[a]:6.3f}  {chat0[a]: .3f}   {1.0/rho[a]-chat0[a]: .3f}")
    print("\n# mu0(rho) (one additive constant = mu_tot, fix by matching a reference):")
    for k in range(0, len(rg), max(1, len(rg) // 20)):
        print(f"  rho={rg[k]:6.3f}  mu0={mu0[k]: .4f}")


# ----------------------------------------------------------------------------
# slab: nonlocal DFT route to mu0 / P0 (the van der Waals loop), gradient-exact
# ----------------------------------------------------------------------------

def run_dft(args):
    sf = read_ave_time_vector(args.sf_file)
    qs, Smats, rho = assemble_matrices(sf, args.nbins)
    if not args.no_mirror:
        Smats, rho = mirror_symmetrize(Smats, rho)
    dz = args.lz / args.nbins
    area = args.lx * args.lx
    active = np.where(rho > args.rho_min)[0]
    z = (active + 0.5) * dz
    rho_s = fourier_cosine_smooth(rho, args.smooth)
    Carr = {q: invert_oz(Smats[q], rho, dz, area, active=active, ridge=args.ridge)[0]
            for q in qs}

    C0 = intercept_matrix(qs, Carr, active, args.kfit)
    mu_ih = dft_mu_ih(C0, rho_s[active], dz, args.temp)

    # mu0(z) = mu_int(z) - mu_IH(z) = (mu_tot - U_ext(z)) - mu_IH(z); mu_tot is the
    # additive constant, fixed downstream by matching mu0 to a reference at one rho.
    Uext = 0.5 * args.dumax * np.cos(2.0 * np.pi * z / args.lz)
    mu0 = -Uext - mu_ih

    print("# nonlocal DFT route (gradient-exact mu0; Evans 1979):")
    print("# rho      U_ext     mu_IH     mu0 (up to +mu_tot)")
    order = np.argsort(rho_s[active])
    for k in order:
        print(f"  {rho_s[active][k]:6.3f}  {Uext[k]: .4f}  {mu_ih[k]: .4f}  {mu0[k]: .4f}")
    print("# add mu_tot (match a reference at one rho); P0(rho) = rho*mu0 - INT mu0 drho")


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['bulk', 'slab', 'kb', 'dft'], required=True)
    p.add_argument('--sf-file', required=True, help='fix ave/time output of c_sf[*]')
    p.add_argument('--nbins', type=int, required=True)
    p.add_argument('--lx', type=float, required=True, help='box length in x (= y)')
    p.add_argument('--lz', type=float, required=True, help='box length in z')
    p.add_argument('--temp', type=float, default=1.0, help='reduced temperature kT')
    p.add_argument('--rdf-file', help='[bulk] rdf ave/time file for the S(q) cross-check')
    p.add_argument('--rho-min', type=float, default=0.05,
                   help='[slab] drop bins with density below this (vapor)')
    p.add_argument('--kfit', type=float, default=2.5,
                   help='[slab] fit C_ij(k) vs k^2 for k below this (with the '
                        'tail correction the residual is smooth to k ~ 2.5)')
    p.add_argument('--no-mirror', action='store_true',
                   help='[slab] disable mirror symmetrization of S_ij about Lz/2')
    p.add_argument('--fit-order', type=int, default=2,
                   help='[slab] polynomial order in k^2 for the second-moment fit')
    p.add_argument('--tail-rsplit', type=float, default=1.5,
                   help='[slab] use the analytic mean-field tail of c for '
                        'r > this (sigma) and fit only the short-range residual; '
                        '0 disables (much noisier). Requires scipy.')
    p.add_argument('--smooth', type=int, default=6,
                   help='[slab] number of cosine modes to smooth rho(z)')
    p.add_argument('--ridge', type=float, default=1e-4,
                   help='[slab] Tikhonov regularization for the OZ inversion')
    p.add_argument('--hstar', type=float, default=0.183,
                   help='[kb] reduced thermal wavelength h* for the ideal-gas mu')
    p.add_argument('--dumax', type=float, default=5.0,
                   help='[dft] CPP external-field amplitude, U_ext=(dumax/2)cos(2 pi z/Lz)')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()

    if args.mode == 'bulk':
        run_bulk(args)
    elif args.mode == 'kb':
        run_kb(args)
    elif args.mode == 'dft':
        run_dft(args)
    else:
        run_slab(args)


if __name__ == '__main__':
    main()
