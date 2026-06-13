#!/usr/bin/env python3
"""Field-coupling (Kirkwood) route to the homogeneous EOS and interfacial free
energy for CPP, from a LADDER of CPP runs at increasing external-field amplitude.

The external field is U_ext(z) = A cos(2 pi z / Lz)  (A = dUmax/2).  Each rung k
provides a measured density profile rho_k(z).  Two independent quantities are
returned, each from the part of the data it is best suited to:

1. mu0(rho), P0(rho) -- the homogeneous EOS / van der Waals loop -- from the
   LOCAL Euler-Lagrange (CPP "direct") condition.  Minimising the square-gradient
   functional  F[rho] = INT [ f0(rho) + (kappa/2) rho'(z)^2 ] dz  at fixed average
   density gives, at every z,

        mu0(rho_k(z)) - kappa rho_k''(z) + U_ext(z) = mu_tot(k).             (*)

   So each profile traces out mu0 POINTWISE in rho_k(z): we pool the points
   (rho_k(z), U_ext, rho_k'') over all z and all rungs and solve the LINEAR system

        mu_ex(rho_i) - kappa rho_i'' - mu_tot(k_i) = U_ext,i - T ln rho_i

   for the excess-EOS polynomial mu_ex(rho), the square-gradient coefficient
   kappa, and one reference constant mu_tot(k) per rung (the ideal part T ln rho
   is taken analytically).  Unlike the integrated-G moment fit, rho'' and U_ext
   vary WITHIN each profile, so this is well conditioned even for a perturbative
   field ladder.  mu0 = T ln rho + mu_ex; f0 = f_ideal + INT mu_ex drho;
   P0 = rho mu0 - f0.

2. INT psi_IH dz and gamma -- the interfacial (gradient) free energy -- from the
   EXACT Hellmann-Feynman coupling integral, which needs no pressure tensor and
   no contour choice:

        A_tot(A) - A_tot(0) = INT_0^A dA' Area INT dz rho_A'(z) cos(2 pi z/Lz),
        G(A) = F[rho_A]/Area - F_unif/Area = A_tot(A)/Area - A INT rho_A cos dz,
        INT psi_IH dz = G(A) - ( INT f0(rho_A) dz - f0(rho_avg) Lz ),
        gamma = 2 INT psi_IH dz.

   As a built-in consistency check this is compared with the square-gradient value
   (kappa/2) INT rho_A'^2 dz from the fitted kappa.

Number of field strengths needed
---------------------------------
* mu0(rho), P0(rho) need only ONE strong-field profile: it traces mu0 pointwise
  along z.  But from a single (symmetric) profile rho''(z) is itself a function
  of rho, so the square-gradient kappa is NOT separable from mu_ex(rho) -- pass a
  fixed --kappa (e.g. from the TZ/OZ inversion route) for the gradient-corrected
  EOS, or kappa=0 for the bare local-density estimate.  Two or more rungs (varying
  rho'' at the same rho) make kappa identifiable on its own.
* INT psi_IH dz and gamma come from the coupling integral, which needs the LADDER
  (>= 2 field strengths) to do the dA' charging integral.

Reference: coupling-constant (charging) integration, e.g. Hansen & McDonald,
"Theory of Simple Liquids"; R. Evans, Adv. Phys. 28, 143 (1979).
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import oz_invert as oz

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    def cumtrapz(y, x, initial=0.0):
        out = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
        return out + initial


def f_ideal(rho, temp):
    """Ideal-gas Helmholtz free-energy density T rho (ln rho - 1) (reduced; the
    thermal-wavelength constant only shifts mu0 by a constant, absorbed into the
    reference)."""
    r = np.clip(rho, 1e-12, None)
    return temp * r * (np.log(r) - 1.0)


def fourier_deriv2(y, nmodes, length):
    """Analytic second z-derivative of the cosine-series fit of y (a cosine
    series): d2/dz2 sum a_k cos(2 pi k z/L) = -sum a_k (2 pi k/L)^2 cos(...)."""
    coef, z = oz.fourier_cosine_coef(y, nmodes)
    k = np.arange(nmodes + 1)
    return (-np.cos(2.0 * np.pi * np.outer(z, k)) * (2.0 * np.pi * k / length) ** 2) @ coef


def coupling_G(amps, profiles, Lz):
    """G_k = (F[rho_k]-F_unif)/Area from the exact coupling integral.
    amps[0] must be 0 (uniform reference); profiles[k] = rho_k(z) on a common grid."""
    nb = profiles.shape[1]
    z = (np.arange(nb) + 0.5) / nb
    cosz = np.cos(2.0 * np.pi * z)
    dz = Lz / nb
    I = (profiles * cosz).sum(axis=1) * dz            # INT dz rho_k cos
    cum = cumtrapz(I, amps, initial=0.0)              # INT_0^A I dA'
    return cum - amps * I                             # (F[rho_k]-F_unif)/Area


def local_eos(amps, profiles, temp, Lz, deg=4, smooth=10, kappa=None):
    """mu0(rho), P0(rho), and the square-gradient kappa from the local
    Euler-Lagrange condition (*).  Pools (rho, rho'', U_ext) over all z and rungs
    and solves the linear system for the excess-EOS polynomial mu_ex(rho), kappa,
    and a per-rung reference constant mu_tot(k).  If kappa is given it is held
    fixed (and moved to the RHS) instead of fitted."""
    nb = profiles.shape[1]
    dz = Lz / nb
    z = (np.arange(nb) + 0.5) / nb
    rho_avg = profiles[0].mean()
    nr = len(amps)
    # build the pooled point list, skipping near-empty bins where ln rho blows up
    rows, rhs = [], []
    ms = list(range(1, deg + 1))                      # excess-EOS powers rho^1..deg
    for k in range(nr):
        rho = profiles[k]
        d2 = fourier_deriv2(rho, smooth, Lz)
        U = amps[k] * np.cos(2.0 * np.pi * z)
        good = rho > 1e-3
        for i in np.where(good)[0]:
            # columns: [mu_ex powers m*rho^(m-1)] , [-rho''(if fitting kappa)] ,
            #          [-1 in this rung's mu_tot slot]
            mucol = [m * rho[i] ** (m - 1) for m in ms]
            rungcol = [0.0] * nr
            rungcol[k] = -1.0
            row = mucol + ([-d2[i]] if kappa is None else []) + rungcol
            r = -U[i] - temp * np.log(rho[i]) + (kappa * d2[i] if kappa is not None else 0.0)
            rows.append(row); rhs.append(r)
    M = np.array(rows); b = np.array(rhs)
    # gauge: the overall constant is split between mu_ex's reference and mu_tot;
    # pin mu_tot(rung 0, the uniform reference) = 0 by dropping its column
    drop = len(ms) + (0 if kappa is not None else 1)      # index of mu_tot(0) col
    keep = [j for j in range(M.shape[1]) if j != drop]
    coef, *_ = np.linalg.lstsq(M[:, keep], b, rcond=None)
    cm = coef[:len(ms)]
    if kappa is None:
        kappa = coef[len(ms)]
    rg = np.linspace(profiles.min() + 1e-3, profiles.max() - 1e-3, 200)
    mu_ex = sum(c * m * rg ** (m - 1) for c, m in zip(cm, ms))
    f_ex = sum(c * rg ** m for c, m in zip(cm, ms))
    mu0 = temp * np.log(rg) + mu_ex
    f0 = f_ideal(rg, temp) + f_ex
    P0 = rg * mu0 - f0
    return dict(rho=rg, mu0=mu0, P0=P0, f0=f0, kappa=float(kappa),
                mu_ex=mu_ex, cm=cm, ms=ms, rho_avg=rho_avg)


def interfacial(amps, profiles, eos, temp, Lz, smooth=10):
    """INT psi_IH dz and gamma=2 INT psi_IH per rung, from the coupling integral G
    minus the local free energy INT f0(rho) dz; cross-checked against the
    square-gradient value (kappa/2) INT rho'^2 dz."""
    nb = profiles.shape[1]; dz = Lz / nb
    G = coupling_G(amps, profiles, Lz)
    f0_of = lambda r: f_ideal(r, temp) + sum(c * r ** m for c, m in zip(eos['cm'], eos['ms']))
    floc = f0_of(profiles).sum(axis=1) * dz - f0_of(eos['rho_avg']) * Lz
    psi_int = G - floc                                # INT psi_IH dz
    rp = np.array([oz.fourier_cosine_deriv(p, smooth, Lz) for p in profiles])
    psi_sg = 0.5 * eos['kappa'] * (rp ** 2).sum(axis=1) * dz   # square-gradient check
    return dict(G=G, psi_int=psi_int, gamma=2.0 * psi_int, psi_sg=psi_sg)


# ----------------------------------------------------------------------------

def _solve_el(mu0_fn, kappa, A, temp, rho_avg, Lz, nb, iters=6000, dt=0.02):
    """Equilibrium square-gradient profile for U_ext = A cos(2 pi z/Lz), by a
    spectral semi-implicit relaxation of  d rho/dt = -(mu0(rho) - kappa rho'' +
    U_ext - mu_tot).  The stiff -kappa rho'' term is treated implicitly (Fourier
    factor 1 + dt kappa k^2), so the iteration is unconditionally stable; mu_tot
    is the Lagrange multiplier fixing <rho> = rho_avg."""
    z = (np.arange(nb) + 0.5) / nb * Lz
    U = A * np.cos(2.0 * np.pi * z / Lz)
    k = 2.0 * np.pi * np.fft.rfftfreq(nb, d=Lz / nb)
    denom = 1.0 + dt * kappa * k ** 2
    rho = rho_avg + 0.01 * np.cos(2.0 * np.pi * z / Lz)
    for _ in range(iters):
        expl = mu0_fn(np.clip(rho, 1e-6, None)) + U
        expl -= expl.mean()                           # remove mu_tot (mean) -> conserve mass
        rho = np.fft.irfft(np.fft.rfft(rho - dt * expl) / denom, n=nb)
        rho *= rho_avg / rho.mean()
    return rho


def _selftest():
    """Validate mu0(rho)/kappa recovery on EQUILIBRIUM profiles: prescribe a known
    EOS mu0 = T ln rho + sum a_m rho^m derivative and square-gradient kappa, solve
    the EL at a ladder of field strengths, then check local_eos recovers them and
    that the coupling-integral interfacial energy matches the square-gradient one."""
    Lz = 30.0; T = 1.0; nb = 200; rho_avg = 0.45
    a = {2: 2.5, 3: -1.0, 4: 0.6}                     # f_ex = sum a_m rho^m
    mu_ex = lambda r: sum(a[m] * m * r ** (m - 1) for m in a)
    mu0 = lambda r: T * np.log(r) + mu_ex(r)
    kappa_true = 1.5
    # the coupling integral uses a trapezoidal A-quadrature whose error is O(dA^2);
    # 13 rungs bring the coupling-vs-square-gradient cross-check under a few %
    amps = np.linspace(0.0, 2.4, 13)
    profiles = [np.full(nb, rho_avg)]
    for A in amps[1:]:
        profiles.append(_solve_el(mu0, kappa_true, A, T, rho_avg, Lz, nb))
    profiles = np.array(profiles)
    eos = local_eos(amps, profiles, T, Lz, deg=4, smooth=12)
    rg = eos['rho']
    mt = mu_ex(rg)
    de = (eos['mu_ex'] - eos['mu_ex'].mean()) - (mt - mt.mean())
    itf = interfacial(amps, profiles, eos, T, Lz, smooth=12)
    rel_grad = np.max(np.abs(itf['psi_int'] - itf['psi_sg'])
                      / (np.abs(itf['psi_sg']) + 1e-9))
    print("rho range %.3f-%.3f" % (profiles.min(), profiles.max()))
    print("kappa  true=%.3f  fit=%.4f" % (kappa_true, eos['kappa']))
    print("max|mu_ex - ref| over sampled rho = %.2e" % np.max(np.abs(de)))
    print("max rel diff (coupling INT psi vs square-gradient) = %.2e" % rel_grad)
    ok = (abs(eos['kappa'] - kappa_true) < 0.05
          and np.max(np.abs(de)) < 2e-2 and rel_grad < 0.05)
    print("  -> mu0(rho), kappa, and interfacial energy recovered"
          if ok else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ladder', help='CSV manifest: dumax,density_file per line '
                                      '(uniform rung A=0 is prepended automatically)')
    ap.add_argument('--lz', type=float)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--deg', type=int, default=4, help='excess-EOS polynomial degree')
    ap.add_argument('--kappa', type=float, default=None,
                    help='fix the square-gradient coefficient instead of fitting it')
    ap.add_argument('--smooth', type=int, default=10, help='cosine modes for profiles')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    rows = [l.split(',') for l in open(args.ladder) if l.strip() and not l.startswith('#')]
    amps, grids = [0.0], []
    for du, fn in rows:
        grids.append(oz.fourier_cosine_smooth(_read_density(fn.strip()), args.smooth))
        amps.append(float(du) / 2.0)
    profiles = np.array([np.full_like(grids[0], np.mean(grids[0]))] + grids)
    amps = np.array(amps)
    eos = local_eos(amps, profiles, args.temp, args.lz, deg=args.deg,
                    smooth=args.smooth, kappa=args.kappa)
    itf = interfacial(amps, profiles, eos, args.temp, args.lz, smooth=args.smooth)
    print("# field-coupling EOS:  kappa=%.4f" % eos['kappa'])
    print("# INT psi_IH (coupling) strongest rung = %.4f ; gamma = %.4f"
          % (itf['psi_int'][-1], itf['gamma'][-1]))
    print("# (square-gradient cross-check INT psi_IH = %.4f)" % itf['psi_sg'][-1])
    print("# rho      mu0        P0")
    for r, m, p in zip(eos['rho'][::8], eos['mu0'][::8], eos['P0'][::8]):
        print(f"  {r:6.3f}  {m: .4f}  {p: .4f}")


def _read_density(fn):
    blocks = []; cur = None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 3:
            cur = []; blocks.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return np.mean([np.array(b) for b in blocks], axis=0)[:, 3]


if __name__ == '__main__':
    main()
