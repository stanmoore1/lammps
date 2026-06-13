#!/usr/bin/env python3
"""Field-coupling (Kirkwood) route to the homogeneous EOS and interfacial free
energy for CPP, from a LADDER of CPP runs at increasing external-field amplitude
(U_ext(z) = A cos(2 pi z / Lz), A = dUmax/2), each providing rho_k(z).

Returns, each from the data it is best suited to:

1. mu0(rho), P0(rho) (the van der Waals loop) from the LOCAL Euler-Lagrange
   condition  mu0(rho_k(z)) + mu_IH(z) + U_ext(z) = mu_tot(k).  The gradient
   correction mu_IH is the variational derivative of a square/4th-gradient free
   energy with possibly DENSITY-DEPENDENT influence parameters,

        F_grad[rho] = INT sum_g (kappa_g(rho)/2) (d^(g/2) rho/dz^(g/2))^2 dz,
        kappa_g(rho) = sum_p a_{g,p} rho^p   (even g = 2, 4),

   so mu_IH = dF_grad/drho is, per basis element kappa_g = rho^p,

        g=2:  -(rho^p rho'' + (p/2) rho^(p-1) (rho')^2)
        g=4:   rho^p rho'''' + (3/2) p rho^(p-1) (rho'')^2
               + 2 p rho^(p-1) rho' rho''' + p(p-1) rho^(p-2) (rho')^2 rho''.

   These are LINEAR in the unknowns a_{g,p}, so a single least-squares solve over
   all (z, rung) points yields mu_ex(rho), the influence-parameter polynomials
   kappa_g(rho), and one mu_tot(k) per rung.  Density dependence (curvature of
   kappa) and higher order are added by widening grad_spec; the LADDER (and the
   rho_avg axis) makes the extra coefficients identifiable.  Constant-kappa
   (grad_spec={2:0,4:0}) reproduces the square-gradient/4th-gradient closure.

2. INT psi_IH dz, gamma = 2 INT psi_IH from the EXACT Hellmann-Feynman coupling
   integral (gradient-exact), cross-checked against F_grad evaluated with the
   fitted kappa_g(rho).

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
    """Ideal-gas Helmholtz free-energy density T rho (ln rho - 1)."""
    r = np.clip(rho, 1e-12, None)
    return temp * r * (np.log(r) - 1.0)


def fderiv(y, nmodes, length, n):
    """n-th z-derivative of the cosine-series fit of y (low-pass smoothed)."""
    coef, z = oz.fourier_cosine_coef(y, nmodes)
    k = np.arange(nmodes + 1)
    fac = (2.0 * np.pi * k / length) ** n
    if n % 2 == 0:
        basis = np.cos(2.0 * np.pi * np.outer(z, k)) * ((-1) ** (n // 2)) * fac
    else:
        basis = -np.sin(2.0 * np.pi * np.outer(z, k)) * ((-1) ** (n // 2)) * fac
    return basis @ coef


def _spec_deriv(rho, length, n):
    """n-th derivative of a real periodic profile by full FFT (no smoothing)."""
    k = 2.0 * np.pi * np.fft.rfftfreq(len(rho), d=length / len(rho))
    return np.fft.irfft(((1j * k) ** n) * np.fft.rfft(rho), n=len(rho))


def parse_grad_spec(s):
    """'2:1,4:0' -> {2:1, 4:0} (order -> max density degree of kappa_g(rho))."""
    spec = {}
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        g, d = tok.split(':')
        spec[int(g)] = int(d)
    return spec


def grad_terms(derivs, rho, grad_spec):
    """For each free-energy basis element kappa_g = rho^p, return its contribution
    to mu_IH (the variational derivative) and to the free-energy density
    rho^p (d^(g/2) rho)^2 / 2.  derivs = (rho', rho'', rho''', rho'''').
    Returns cols, ecols, labels (label = (g, p))."""
    d1, d2, d3, d4 = derivs
    cols, ecols, labels = [], [], []
    for g, dmax in grad_spec.items():
        for p in range(dmax + 1):
            rp = rho ** p
            rpm1 = rho ** (p - 1) if p >= 1 else np.zeros_like(rho)
            rpm2 = rho ** (p - 2) if p >= 2 else np.zeros_like(rho)
            if g == 2:
                col = -(rp * d2 + (p / 2.0) * rpm1 * d1 ** 2)
                ec = 0.5 * rp * d1 ** 2
            elif g == 4:
                col = (rp * d4 + 1.5 * p * rpm1 * d2 ** 2
                       + 2.0 * p * rpm1 * d1 * d3
                       + p * (p - 1) * rpm2 * d1 ** 2 * d2)
                ec = 0.5 * rp * d2 ** 2
            else:
                raise ValueError("grad orders 2 and 4 only")
            cols.append(col); ecols.append(ec); labels.append((g, p))
    return cols, ecols, labels


def coupling_G(amps, profiles, Lz):
    """G_k = (F[rho_k]-F_unif)/Area from the exact coupling integral.  amps[0]=0."""
    nb = profiles.shape[1]
    z = (np.arange(nb) + 0.5) / nb
    dz = Lz / nb
    I = (profiles * np.cos(2.0 * np.pi * z)).sum(axis=1) * dz
    return cumtrapz(I, amps, initial=0.0) - amps * I


def local_eos(amps, profiles, temp, Lz, deg=4, smooth=10, grad_spec=None):
    """mu0(rho), P0(rho) and the (density-dependent) gradient influence parameters
    from the pooled local Euler-Lagrange condition.  grad_spec = {order: density
    degree}; default {2:0, 4:0} (constant kappa2, kappa4)."""
    if grad_spec is None:
        grad_spec = {2: 0, 4: 0}
    nb = profiles.shape[1]; dz = Lz / nb
    z = (np.arange(nb) + 0.5) / nb
    rho_avg = profiles[0].mean()
    nr = len(amps)
    ms = list(range(1, deg + 1))
    # gradient columns per rung
    labels = None
    grad_cols = []
    for p in profiles:
        derivs = tuple(fderiv(p, smooth, Lz, n) for n in (1, 2, 3, 4))
        cols, _, labels = grad_terms(derivs, p, grad_spec)
        grad_cols.append(cols)
    ng = len(labels)
    rows, rhs = [], []
    for k in range(nr):
        rho = profiles[k]; U = amps[k] * np.cos(2.0 * np.pi * z)
        for i in np.where(rho > 1e-3)[0]:
            mucol = [m * rho[i] ** (m - 1) for m in ms]
            gcol = [grad_cols[k][j][i] for j in range(ng)]
            rungcol = [0.0] * nr; rungcol[k] = -1.0
            rows.append(mucol + gcol + rungcol)
            rhs.append(-U[i] - temp * np.log(rho[i]))
    M = np.array(rows); b = np.array(rhs)
    drop = len(ms) + ng                               # mu_tot(rung 0) -> gauge fix
    keep = [j for j in range(M.shape[1]) if j != drop]
    coef, *_ = np.linalg.lstsq(M[:, keep], b, rcond=None)
    cm = coef[:len(ms)]
    acoef = dict(zip(labels, coef[len(ms):len(ms) + ng]))
    kap = lambda g, r: sum(acoef[(g, p)] * r ** p for (gg, p) in labels if gg == g)
    rg = np.linspace(profiles.min() + 1e-3, profiles.max() - 1e-3, 200)
    mu_ex = sum(c * m * rg ** (m - 1) for c, m in zip(cm, ms))
    f_ex = sum(c * rg ** m for c, m in zip(cm, ms))
    mu0 = temp * np.log(rg) + mu_ex
    f0 = f_ideal(rg, temp) + f_ex
    P0 = rg * mu0 - f0
    return dict(rho=rg, mu0=mu0, P0=P0, f0=f0, mu_ex=mu_ex, cm=cm, ms=ms,
                acoef=acoef, labels=labels, grad_spec=grad_spec, rho_avg=rho_avg,
                kappa2=lambda r: kap(2, r), kappa4=lambda r: kap(4, r))


def interfacial(amps, profiles, eos, temp, Lz, smooth=10):
    """INT psi_IH dz and gamma from the coupling integral (gradient-exact),
    cross-checked against F_grad with the fitted kappa_g(rho)."""
    nb = profiles.shape[1]; dz = Lz / nb
    G = coupling_G(amps, profiles, Lz)
    f0_of = lambda r: f_ideal(r, temp) + sum(c * r ** m for c, m in zip(eos['cm'], eos['ms']))
    ref_rho = profiles[0].mean()                      # uniform reference of THIS ladder
    floc = f0_of(profiles).sum(axis=1) * dz - f0_of(ref_rho) * Lz
    psi_int = G - floc
    psi_sg = np.zeros(len(amps))
    for k, p in enumerate(profiles):
        derivs = tuple(fderiv(p, smooth, Lz, n) for n in (1, 2, 3, 4))
        _, ecols, labels = grad_terms(derivs, p, eos['grad_spec'])
        e = sum(eos['acoef'][lab] * ec for lab, ec in zip(labels, ecols))
        psi_sg[k] = e.sum() * dz
    return dict(G=G, psi_int=psi_int, gamma=2.0 * psi_int, psi_sg=psi_sg)


# ----------------------------------------------------------------------------

def _solve_el(mu0_fn, acoef, grad_spec, A, temp, rho_avg, Lz, nb, iters=15000, dt=0.008):
    """Equilibrium profile for F = INT[f0 + sum_g (kappa_g(rho)/2)(d^(g/2)rho)^2]
    under U_ext = A cos, by spectral semi-implicit relaxation.  The stiff part uses
    REFERENCE constants kappa_g(rho_avg) (implicit); the density-dependent remainder
    of mu_IH is explicit."""
    z = (np.arange(nb) + 0.5) / nb * Lz
    U = A * np.cos(2.0 * np.pi * z / Lz)
    k = 2.0 * np.pi * np.fft.rfftfreq(nb, d=Lz / nb)
    kbar = {g: sum(acoef[(g, p)] * rho_avg ** p for (gg, p) in acoef if gg == g)
            for g in grad_spec}
    denom = 1.0 + dt * (kbar.get(2, 0.0) * k ** 2 + kbar.get(4, 0.0) * k ** 4)
    rho = rho_avg + 0.01 * np.cos(2.0 * np.pi * z / Lz)
    labels = [(g, p) for g, d in grad_spec.items() for p in range(d + 1)]
    for _ in range(iters):
        derivs = tuple(_spec_deriv(rho, Lz, n) for n in (1, 2, 3, 4))
        cols, _, labs = grad_terms(derivs, np.clip(rho, 1e-6, None), grad_spec)
        muih = sum(acoef[lab] * c for lab, c in zip(labs, cols))
        Lref = -kbar.get(2, 0.0) * derivs[1] + kbar.get(4, 0.0) * derivs[3]
        N = mu0_fn(np.clip(rho, 1e-6, None)) + U + (muih - Lref)
        N -= N.mean()
        rho = np.fft.irfft(np.fft.rfft(rho - dt * N) / denom, n=nb)
        rho *= rho_avg / rho.mean()
    return rho


def _selftest():
    """Validate the DENSITY-DEPENDENT, consistent gradient fit: prescribe an EOS and
    influence parameters kappa2(rho)=k20+k21 rho, kappa4=k40, solve the EL on a 2D
    (rho_avg x field) grid, and check local_eos recovers mu_ex AND the kappa_g(rho)
    polynomials; then check the coupling-integral INT psi_IH matches F_grad."""
    Lz = 30.0; T = 1.0; nb = 200
    a = {2: 2.5, 3: -1.0, 4: 0.6}
    mu_ex = lambda r: sum(a[m] * m * r ** (m - 1) for m in a)
    mu0 = lambda r: T * np.log(r) + mu_ex(r)
    grad_spec = {2: 1, 4: 0}
    acoef_true = {(2, 0): 1.5, (2, 1): 0.9, (4, 0): 0.8}   # kappa2(rho)=1.5+0.9 rho
    # 2D grid: rho_avg axis (coverage + curvature) x field axis (gradient spread)
    rho_avgs = [0.40, 0.45, 0.50]
    amps_grid = [0.0, 0.8, 1.6, 2.4]
    amps, profiles = [], []
    for ra in rho_avgs:
        for A in amps_grid:
            amps.append(A)
            profiles.append(_solve_el(mu0, acoef_true, grad_spec, A, T, ra, Lz, nb))
    amps = np.array(amps); profiles = np.array(profiles)
    eos = local_eos(amps, profiles, T, Lz, deg=4, smooth=12, grad_spec=grad_spec)
    rg = eos['rho']; mt = mu_ex(rg)
    de = (eos['mu_ex'] - eos['mu_ex'].mean()) - (mt - mt.mean())
    k2_true = lambda r: 1.5 + 0.9 * r
    k2err = np.max(np.abs(eos['kappa2'](rg) - k2_true(rg)))
    k4err = abs(eos['acoef'][(4, 0)] - 0.8)
    # interfacial consistency on a DENSE fixed-rho_avg ladder (coupling integral's
    # A-quadrature is O(dA^2), so it needs more rungs than the EOS fit)
    da = np.linspace(0.0, 2.4, 13)
    dp = np.array([_solve_el(mu0, acoef_true, grad_spec, A, T, 0.45, Lz, nb) for A in da])
    itf = interfacial(da, dp, eos, T, Lz, smooth=12)
    rel = np.max(np.abs(itf['psi_int'] - itf['psi_sg']) / (np.abs(itf['psi_sg']) + 1e-9))
    print("kappa2(rho)=1.5+0.9rho : max err %.2e ; kappa4 true 0.8 fit %.4f"
          % (k2err, eos['acoef'][(4, 0)]))
    print("max|mu_ex - ref| = %.2e ; coupling vs F_grad rel = %.2e"
          % (np.max(np.abs(de)), rel))
    ok = k2err < 0.05 and k4err < 0.05 and np.max(np.abs(de)) < 3e-2 and rel < 0.06
    print("  -> mu_ex and density-dependent kappa_g(rho) recovered (consistent)"
          if ok else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ladder', help='CSV manifest: dumax,density_file per line')
    ap.add_argument('--lz', type=float)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--deg', type=int, default=4, help='excess-EOS polynomial degree')
    ap.add_argument('--grad-spec', default='2:0,4:0',
                    help='order:density-degree list for kappa_g(rho), e.g. '
                         '"2:0,4:0" (constant), "2:1" (kappa2 linear in rho), "2:2,4:1"')
    ap.add_argument('--smooth', type=int, default=10)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    gspec = parse_grad_spec(args.grad_spec)
    rows = [l.split(',') for l in open(args.ladder) if l.strip() and not l.startswith('#')]
    amps, grids = [0.0], []
    for du, fn in rows:
        grids.append(oz.fourier_cosine_smooth(_read_density(fn.strip()), args.smooth))
        amps.append(float(du) / 2.0)
    profiles = np.array([np.full_like(grids[0], np.mean(grids[0]))] + grids)
    amps = np.array(amps)
    eos = local_eos(amps, profiles, args.temp, args.lz, deg=args.deg,
                    smooth=args.smooth, grad_spec=gspec)
    itf = interfacial(amps, profiles, eos, args.temp, args.lz, smooth=args.smooth)
    print("# field-coupling EOS: grad_spec=%s" % gspec)
    print("# kappa2(rho_avg)=%.4f  kappa4(rho_avg)=%.4f"
          % (eos['kappa2'](eos['rho_avg']), eos['kappa4'](eos['rho_avg'])))
    print("# INT psi_IH (coupling, exact) strongest = %.4f ; gamma = %.4f ; (F_grad %.4f)"
          % (itf['psi_int'][-1], itf['gamma'][-1], itf['psi_sg'][-1]))
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
