#!/usr/bin/env python3
"""Field-coupling (Kirkwood) route to the homogeneous EOS and interfacial free
energy for CPP, from a LADDER of CPP runs at increasing external-field amplitude.

The external field is U_ext(z) = A cos(2 pi z / Lz)  (A = dUmax/2).  Each rung k
provides a measured density profile rho_k(z).  Two quantities are returned, each
from the part of the data it is best suited to:

1. mu0(rho), P0(rho) -- the homogeneous EOS / van der Waals loop -- from the LOCAL
   Euler-Lagrange (CPP "direct") condition.  At every z the intrinsic chemical
   potential equals mu_tot minus the field:

        mu0(rho_k(z)) + mu_IH_k(z) + U_ext(z) = mu_tot(k).                   (*)

   mu_IH is the inhomogeneous (gradient) correction.  Instead of ASSUMING the
   second-order square-gradient closure mu_IH = -kappa rho'' (the dissertation's
   VdW form, which fails at a sharp interface), we expand it in a gradient series

        mu_IH(z) = sum_g b_g  d^g rho / dz^g     (even g = 2, 4, ...)

   and FIT the b_g from the ladder: pooling (*) over all z and all rungs,

        mu_ex(rho_i) + sum_g b_g d^g rho_i - mu_tot(k_i) = -U_ext,i - T ln rho_i,

   a single linear system for the excess-EOS polynomial mu_ex(rho), the gradient
   coefficients b_g, and one reference constant mu_tot(k) per rung.  Because a
   LADDER samples a given rho at DIFFERENT gradients (different rungs), the b_g
   are identifiable, so the gradient correction is data-determined rather than
   truncated at second order.  grad_orders=(2,) reproduces the old square-gradient
   closure; (2,4) goes to fourth order; etc.  mu0 = T ln rho + mu_ex;
   f0 = f_ideal + INT mu_ex; P0 = rho mu0 - f0.

2. INT psi_IH dz and gamma -- the interfacial (gradient) free energy -- from the
   EXACT Hellmann-Feynman coupling integral (no pressure tensor, no contour, NO
   gradient expansion):

        A_tot(A) - A_tot(0) = INT_0^A dA' Area INT dz rho_A'(z) cos(2 pi z/Lz),
        G(A) = A_tot(A)/Area - A INT rho_A cos dz,
        INT psi_IH dz = G(A) - ( INT f0(rho_A) dz - f0(rho_avg) Lz ),
        gamma = 2 INT psi_IH dz.

   This is gradient-exact.  As a cross-check it is compared with the gradient-
   expansion energy sum_g (c_g/2) INT (d^(g/2) rho)^2, c_g = (-1)^(g/2) b_g.

Number of field strengths needed
---------------------------------
* gamma, INT psi_IH need the LADDER (>= 2 strengths) for the dA' charging integral.
* The b_g (gradient correction) also need the ladder: from ONE symmetric profile
  d^g rho is itself a function of rho, so b_g cannot be separated from mu_ex.  With
  one strength, pass grad_orders=() and supply the gradient correction externally
  (e.g. kappa from the OZ/TZ route) or accept the bare local-density mu0.

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


def fderiv(y, nmodes, length, n):
    """n-th z-derivative of the cosine-series fit of y, evaluated at the bin
    centers.  Even n give a cosine series, odd n a sine series; both are exactly
    periodic (no finite-difference error)."""
    coef, z = oz.fourier_cosine_coef(y, nmodes)
    k = np.arange(nmodes + 1)
    fac = (2.0 * np.pi * k / length) ** n
    if n % 2 == 0:
        basis = np.cos(2.0 * np.pi * np.outer(z, k)) * ((-1) ** (n // 2)) * fac
    else:
        basis = -np.sin(2.0 * np.pi * np.outer(z, k)) * ((-1) ** (n // 2)) * fac
    return basis @ coef


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


def local_eos(amps, profiles, temp, Lz, deg=4, smooth=10, grad_orders=(2, 4)):
    """mu0(rho), P0(rho) and the gradient (mu_IH) coefficients from the local
    Euler-Lagrange condition (*), pooled pointwise over z and all rungs.  The
    gradient correction mu_IH = sum_g b_g d^g rho/dz^g (even g) is FIT from the
    ladder rather than assuming the second-order b_2 rho'' closure.  grad_orders
    lists the even derivative orders kept; () drops the gradient fit entirely
    (bare local-density mu0).  Returns rho grid, mu0, P0, f0, mu_ex, the b_g, and
    the equivalent square-gradient kappa = -b_2."""
    nb = profiles.shape[1]
    dz = Lz / nb
    z = (np.arange(nb) + 0.5) / nb
    rho_avg = profiles[0].mean()
    nr = len(amps)
    ms = list(range(1, deg + 1))
    go = list(grad_orders)
    gderiv = {g: np.array([fderiv(p, smooth, Lz, g) for p in profiles]) for g in go}
    rows, rhs = [], []
    for k in range(nr):
        rho = profiles[k]
        U = amps[k] * np.cos(2.0 * np.pi * z)
        for i in np.where(rho > 1e-3)[0]:
            mucol = [m * rho[i] ** (m - 1) for m in ms]
            gradcol = [gderiv[g][k, i] for g in go]
            rungcol = [0.0] * nr; rungcol[k] = -1.0
            rows.append(mucol + gradcol + rungcol)
            rhs.append(-U[i] - temp * np.log(rho[i]))
    M = np.array(rows); b = np.array(rhs)
    drop = len(ms) + len(go)                          # mu_tot(rung 0) col -> gauge fix
    keep = [j for j in range(M.shape[1]) if j != drop]
    coef, *_ = np.linalg.lstsq(M[:, keep], b, rcond=None)
    cm = coef[:len(ms)]
    bg = dict(zip(go, coef[len(ms):len(ms) + len(go)]))
    rg = np.linspace(profiles.min() + 1e-3, profiles.max() - 1e-3, 200)
    mu_ex = sum(c * m * rg ** (m - 1) for c, m in zip(cm, ms))
    f_ex = sum(c * rg ** m for c, m in zip(cm, ms))
    mu0 = temp * np.log(rg) + mu_ex
    f0 = f_ideal(rg, temp) + f_ex
    P0 = rg * mu0 - f0
    return dict(rho=rg, mu0=mu0, P0=P0, f0=f0, mu_ex=mu_ex, cm=cm, ms=ms,
                bg=bg, kappa=float(-bg.get(2, 0.0)), rho_avg=rho_avg)


def interfacial(amps, profiles, eos, temp, Lz, smooth=10):
    """INT psi_IH dz and gamma=2 INT psi_IH per rung, from the coupling integral G
    minus the local free energy INT f0(rho) dz (gradient-exact); cross-checked
    against the fitted gradient-expansion energy sum_g (c_g/2) INT (d^(g/2) rho)^2,
    c_g = (-1)^(g/2) b_g."""
    nb = profiles.shape[1]; dz = Lz / nb
    G = coupling_G(amps, profiles, Lz)
    f0_of = lambda r: f_ideal(r, temp) + sum(c * r ** m for c, m in zip(eos['cm'], eos['ms']))
    floc = f0_of(profiles).sum(axis=1) * dz - f0_of(eos['rho_avg']) * Lz
    psi_int = G - floc                                # gradient-exact INT psi_IH
    psi_sg = np.zeros(len(amps))
    for g, bgv in eos['bg'].items():
        cg = ((-1) ** (g // 2)) * bgv
        d = np.array([fderiv(p, smooth, Lz, g // 2) for p in profiles])
        psi_sg += 0.5 * cg * (d ** 2).sum(axis=1) * dz
    return dict(G=G, psi_int=psi_int, gamma=2.0 * psi_int, psi_sg=psi_sg)


# ----------------------------------------------------------------------------

def _solve_el(mu0_fn, grad, A, temp, rho_avg, Lz, nb, iters=8000, dt=0.02):
    """Equilibrium profile for the functional F = INT[f0(rho) + sum_g (c_g/2)
    (d^(g/2) rho)^2] dz under U_ext = A cos(2 pi z/Lz), by spectral semi-implicit
    relaxation; grad = {g: c_g} (e.g. {2: kappa2, 4: kappa4}).  The stiff gradient
    operator (Fourier factor sum_g c_g k^g) is treated implicitly."""
    z = (np.arange(nb) + 0.5) / nb * Lz
    U = A * np.cos(2.0 * np.pi * z / Lz)
    k = 2.0 * np.pi * np.fft.rfftfreq(nb, d=Lz / nb)
    denom = 1.0 + dt * sum(c * k ** g for g, c in grad.items())
    rho = rho_avg + 0.01 * np.cos(2.0 * np.pi * z / Lz)
    for _ in range(iters):
        expl = mu0_fn(np.clip(rho, 1e-6, None)) + U
        expl -= expl.mean()
        rho = np.fft.irfft(np.fft.rfft(rho - dt * expl) / denom, n=nb)
        rho *= rho_avg / rho.mean()
    return rho


def _selftest():
    """Validate the ladder-fit gradient extraction BEYOND second order: prescribe a
    known EOS and a gradient functional with BOTH a square-gradient (kappa2) and a
    fourth-order (kappa4) term, solve the EL at a ladder of fields, and check
    local_eos recovers mu_ex, kappa2 AND kappa4, plus the gradient-exact INT psi_IH."""
    Lz = 30.0; T = 1.0; nb = 200; rho_avg = 0.45
    a = {2: 2.5, 3: -1.0, 4: 0.6}
    mu_ex = lambda r: sum(a[m] * m * r ** (m - 1) for m in a)
    mu0 = lambda r: T * np.log(r) + mu_ex(r)
    grad = {2: 1.5, 4: 0.8}                           # (k2/2)(rho')^2 + (k4/2)(rho'')^2
    amps = np.linspace(0.0, 2.4, 13)
    profiles = [np.full(nb, rho_avg)]
    for A in amps[1:]:
        profiles.append(_solve_el(mu0, grad, A, T, rho_avg, Lz, nb))
    profiles = np.array(profiles)
    eos = local_eos(amps, profiles, T, Lz, deg=4, smooth=12, grad_orders=(2, 4))
    rg = eos['rho']; mt = mu_ex(rg)
    de = (eos['mu_ex'] - eos['mu_ex'].mean()) - (mt - mt.mean())
    itf = interfacial(amps, profiles, eos, T, Lz, smooth=12)
    rel = np.max(np.abs(itf['psi_int'] - itf['psi_sg']) / (np.abs(itf['psi_sg']) + 1e-9))
    k2, k4 = -eos['bg'][2], eos['bg'][4]
    print("rho range %.3f-%.3f" % (profiles.min(), profiles.max()))
    print("kappa2 true=%.3f fit=%.4f ; kappa4 true=%.3f fit=%.4f" % (grad[2], k2, grad[4], k4))
    print("max|mu_ex - ref| = %.2e ; coupling vs grad-expansion rel = %.2e"
          % (np.max(np.abs(de)), rel))
    ok = (abs(k2 - grad[2]) < 0.05 and abs(k4 - grad[4]) < 0.05
          and np.max(np.abs(de)) < 2e-2 and rel < 0.05)
    print("  -> mu_ex, kappa2, kappa4 and interfacial energy recovered (beyond 2nd order)"
          if ok else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ladder', help='CSV manifest: dumax,density_file per line '
                                      '(uniform rung A=0 is prepended automatically)')
    ap.add_argument('--lz', type=float)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--deg', type=int, default=4, help='excess-EOS polynomial degree')
    ap.add_argument('--grad-orders', default='2,4',
                    help='comma list of even gradient orders for mu_IH '
                         '(e.g. "2" = square-gradient/VdW, "2,4" = 4th order, "" = none)')
    ap.add_argument('--smooth', type=int, default=10, help='cosine modes for profiles')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    go = tuple(int(x) for x in args.grad_orders.split(',') if x.strip())
    rows = [l.split(',') for l in open(args.ladder) if l.strip() and not l.startswith('#')]
    amps, grids = [0.0], []
    for du, fn in rows:
        grids.append(oz.fourier_cosine_smooth(_read_density(fn.strip()), args.smooth))
        amps.append(float(du) / 2.0)
    profiles = np.array([np.full_like(grids[0], np.mean(grids[0]))] + grids)
    amps = np.array(amps)
    eos = local_eos(amps, profiles, args.temp, args.lz, deg=args.deg,
                    smooth=args.smooth, grad_orders=go)
    itf = interfacial(amps, profiles, eos, args.temp, args.lz, smooth=args.smooth)
    print("# field-coupling EOS: gradient coeffs b_g = %s (kappa=-b_2=%.4f)"
          % ({g: round(v, 4) for g, v in eos['bg'].items()}, eos['kappa']))
    print("# INT psi_IH (coupling, exact) strongest rung = %.4f ; gamma = %.4f"
          % (itf['psi_int'][-1], itf['gamma'][-1]))
    print("# (gradient-expansion cross-check INT psi_IH = %.4f)" % itf['psi_sg'][-1])
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
