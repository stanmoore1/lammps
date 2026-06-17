#!/usr/bin/env python3
"""Data-driven NONLOCAL KERNEL field coupling.

Extends the field-coupling EOS extraction (field_coupling.local_eos) from a LOCAL
gradient correction (a 2nd-moment / influence-parameter kernel) to a data-driven
NONLOCAL kernel C(s), parametrized in a small even basis and fit by demanding
mu0(rho) be single-valued across a ladder of inhomogeneous runs at different field
strengths.  Motivated by the Evans (1979) relation
    mu_IH(z) = - INT ds C(s) [rho(z+s) - rho(z)]    (C even),
discretized (periodic, dz=Lz/nb) with C(s)=sum_m a_m chi_m(s):
    mu_IH(z_i) = sum_m a_m Kcol_m(z_i),
    Kcol_m(z_i) = dz sum_j chi_m(z_i - z_j) [rho_j - rho_i]   (circular convolution).
The per-(rung k, bin i) equation, pooled and solved by one ridge lstsq:
    mu_ex(rho_i) + sum_m a_m Kcol_m(z_i) - mu_tot(k) = -U_ext(z_i) - T ln rho_i.
The difference form makes a uniform profile give mu_IH=0 automatically.  This goes
BEYOND a 2nd-moment kernel: the fitted C(s) shape carries arbitrary moments; we map
it back to the old kappa via kappa2_eff = 1/2 INT s^2 C, kappa4_eff = -1/24 INT s^4 C.

Self-test: python3 kernel_fit.py --selftest  (recover a known non-cosine kernel +
EOS from EL-solved profiles, and beat a 2nd-moment-only fit).
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import field_coupling as fc
import oz_invert as oz


def kernel_basis(s, smax, nmodes, zero_integral=True):
    """(len(s), nmodes) matrix of even Hann-tapered cosine modes
    chi_m(s) = 0.5(1+cos(pi s/smax)) cos(m pi s/smax) on |s|<=smax (0 outside).
    If zero_integral, subtract the support-mean so INT chi_m ds ~ 0 (removes the
    mode's degeneracy with the linear mu_ex term)."""
    s = np.asarray(s, float)
    inside = np.abs(s) <= smax
    hann = np.where(inside, 0.5 * (1.0 + np.cos(np.pi * s / smax)), 0.0)
    B = np.zeros((len(s), nmodes))
    for m in range(nmodes):
        chi = hann * np.cos(m * np.pi * s / smax) * inside
        if zero_integral and inside.any():
            chi = np.where(inside, chi - chi[inside].mean(), 0.0)
        B[:, m] = chi
    return B


def _kernel_columns(rho, dz, Lz, smax, nmodes, zero_integral=True):
    """For one profile, return the (nb, nmodes) matrix of kernel columns
    Kcol_m(z_i) = dz sum_j chi_m(z_i-z_j)[rho_j - rho_i] (periodic convolution)."""
    nb = len(rho)
    lag = (np.arange(nb) + nb // 2) % nb - nb // 2          # signed lags in bins
    s = lag * dz
    Bvec = kernel_basis(s, smax, nmodes, zero_integral)     # (nb, nmodes), chi at each lag
    rho_hat = np.fft.rfft(rho)
    cols = np.empty((nb, nmodes))
    for m in range(nmodes):
        Cv = np.roll(Bvec[:, m], -(nb // 2))                # lag 0 at index 0
        conv = np.fft.irfft(np.fft.rfft(Cv) * rho_hat, n=nb)
        cols[:, m] = dz * (conv - rho * Cv.sum())
    return cols


def kernel_eos(amps, profiles, temp, Lz, deg=4, smax=2.5, nmodes=3, ridge=1e-4,
               smooth=12, zero_integral=True, ridge_curv=True, local_backbone=True):
    """Pooled nonlocal-kernel fit.  Returns rho-grid, mu0, P0, f0, mu_ex, the kernel
    coefficients a_m and C(s), kappa2_eff/kappa4_eff, per-rung mu_tot, cond, resid.

    local_backbone (default True): also fit a LOCAL 2nd-moment term kappa2_loc*(-rho''),
    so the nonlocal kernel C(s) carries only the higher-moment SHAPE.  Without it the
    nonlocal modes and the bulk EOS polynomial are degenerate over the narrow liquid
    slab: the fit then under-estimates the 2nd moment and the recovered mu0(rho) bends
    the wrong way on the liquid branch (the Maxwell liquid density comes out far too
    low).  The robust local backbone breaks that degeneracy; the data still set the
    nonlocal shape on top of it.  Set False to fit a purely nonlocal kernel (used by
    the self-test, where the synthetic ladder makes the kernel fully identifiable)."""
    profiles = np.asarray(profiles, float)
    profiles = np.array([oz.fourier_cosine_smooth(p, smooth) for p in profiles])
    nb = profiles.shape[1]; dz = Lz / nb
    z = (np.arange(nb) + 0.5) / nb
    nr = len(amps)
    ms = list(range(1, deg + 1))
    Kc = [_kernel_columns(p, dz, Lz, smax, nmodes, zero_integral) for p in profiles]
    d2 = [fc.fderiv(p, smooth, Lz, 2) for p in profiles] if local_backbone else None
    nloc = 1 if local_backbone else 0
    rows, rhs = [], []
    for k in range(nr):
        rho = profiles[k]; U = amps[k] * np.cos(2.0 * np.pi * z)
        for i in np.where(rho > 1e-3)[0]:
            mucol = [m * rho[i] ** (m - 1) for m in ms]
            loccol = [-d2[k][i]] if local_backbone else []   # local kappa2 backbone
            kcol = list(Kc[k][i])
            rungcol = [0.0] * nr; rungcol[k] = -1.0
            rows.append(mucol + loccol + kcol + rungcol)
            rhs.append(-U[i] - temp * np.log(rho[i]))
    M = np.array(rows); b = np.array(rhs)
    drop = len(ms) + nloc + nmodes                          # mu_tot(rung 0) gauge
    keep = [j for j in range(M.shape[1]) if j != drop]
    Mk = M[:, keep]
    # ridge on the a_m block only (m^2-weighted to damp oscillatory modes)
    aidx = list(range(len(ms) + nloc, len(ms) + nloc + nmodes))
    pen = np.zeros((nmodes, Mk.shape[1]))
    for jj, col in enumerate(aidx):
        w = (jj + 1) ** 2 if ridge_curv else 1.0
        pen[jj, col] = np.sqrt(ridge * w)
    Maug = np.vstack([Mk, pen]); baug = np.concatenate([b, np.zeros(nmodes)])
    coef, *_ = np.linalg.lstsq(Maug, baug, rcond=None)
    cond = np.linalg.cond(Mk)
    resid = np.sqrt(np.mean((Mk @ coef - b) ** 2))
    cm = coef[:len(ms)]
    kap2_loc = float(coef[len(ms)]) if local_backbone else 0.0
    a = coef[len(ms) + nloc:len(ms) + nloc + nmodes]
    # kernel C(s) and its moments; the 2nd moment includes the local backbone
    sg = np.linspace(-smax, smax, 400)
    Cs = kernel_basis(sg, smax, nmodes, zero_integral) @ a
    kap2 = kap2_loc + 0.5 * np.trapezoid(sg ** 2 * Cs, sg)
    kap4 = -(1.0 / 24.0) * np.trapezoid(sg ** 4 * Cs, sg)
    rg = np.linspace(profiles.min() + 1e-3, profiles.max() - 1e-3, 200)
    mu_ex = sum(c * m * rg ** (m - 1) for c, m in zip(cm, ms))
    f_ex = sum(c * rg ** m for c, m in zip(cm, ms))
    mu0 = temp * np.log(rg) + mu_ex
    f0 = fc.f_ideal(rg, temp) + f_ex
    P0 = rg * mu0 - f0
    return dict(rho=rg, mu0=mu0, P0=P0, f0=f0, mu_ex=mu_ex, cm=cm, ms=ms,
                a=a, s=sg, C=Cs, kappa2_eff=float(kap2), kappa4_eff=float(kap4),
                kappa2_loc=kap2_loc,
                cond=float(cond), resid=float(resid), nmodes=nmodes,
                rho_avg=profiles[0].mean())


# ----------------------------------------------------------------------------

def _apply_kernel(Cv, rho, dz):
    conv = np.fft.irfft(np.fft.rfft(Cv) * np.fft.rfft(rho), n=len(rho))
    return dz * (conv - rho * Cv.sum())


def _solve_el_kernel(mu0_fn, Cv, A, temp, rho_avg, Lz, nb, iters=20000, dt=0.005):
    """Equilibrium profile for mu0(rho) + L_K[rho] + U_ext = mu_tot, U_ext=A cos.
    The linear kernel operator L_K is treated implicitly via its Fourier symbol
    (which vanishes at k=0, so the mean is preserved); mu0+U is explicit."""
    z = (np.arange(nb) + 0.5) / nb * Lz; dz = Lz / nb
    U = A * np.cos(2.0 * np.pi * z / Lz)
    Chat = np.fft.rfft(Cv).real
    hatLK = dz * (Chat - Chat[0])                           # symbol of L_K (hatLK[0]=0)
    denom = 1.0 + dt * hatLK
    denom = np.where(denom < 0.1, 0.1, denom)
    rho = rho_avg + 0.05 * np.cos(2.0 * np.pi * z / Lz)
    for _ in range(iters):
        N = mu0_fn(np.clip(rho, 1e-4, None)) + U
        N -= N.mean()
        rho = np.fft.irfft(np.fft.rfft(rho - dt * N) / denom, n=nb)
        rho *= rho_avg / rho.mean()
    return rho


def _selftest():
    Lz = 30.0; T = 1.0; nb = 256; rho_avg = 0.45
    a = {2: 2.5, 3: -1.0, 4: 0.6}
    mu_ex = lambda r: sum(a[m] * m * r ** (m - 1) for m in a)
    mu0 = lambda r: T * np.log(r) + mu_ex(r)
    smax_t = 2.5
    # KNOWN non-cosine kernel with a sign change (NOT in the fit basis): a damped
    # oscillation so a pure 2nd-moment model cannot match its shape.
    lag = (np.arange(nb) + nb // 2) % nb - nb // 2
    s = lag * (Lz / nb)
    w = 0.9                                                 # Ricker (Mexican-hat): even,
    C_true = np.where(np.abs(s) <= smax_t,                  # ZERO integral, sign-changing
                      0.8 * (1.0 - 2.0 * (s / w) ** 2) * np.exp(-(s / w) ** 2), 0.0)
    Cv = np.roll(C_true, -(nb // 2))
    # sharp-gradient ladder: the nonlocal kernel beyond the 2nd moment is only
    # identifiable when the gradients are sharp (strong fields)
    amps = np.array([0.0, 1.5, 3.0, 4.5])
    profiles = [np.full(nb, rho_avg)]
    for A in amps[1:]:
        profiles.append(_solve_el_kernel(mu0, Cv, A, T, rho_avg, Lz, nb))
    profiles = np.array(profiles)
    eos = kernel_eos(amps, profiles, T, Lz, deg=4, smax=smax_t, nmodes=3,
                     ridge=1e-6, smooth=14, local_backbone=False)
    eos1 = kernel_eos(amps, profiles, T, Lz, deg=4, smax=smax_t, nmodes=1,
                      ridge=1e-6, smooth=14, local_backbone=False)  # 2nd-moment-ish
    rg = eos['rho']; mt = mu_ex(rg)
    de = (eos['mu_ex'] - eos['mu_ex'].mean()) - (mt - mt.mean())
    # compare recovered kernel shape to the truth on the s-grid
    Ctrue_on = np.interp(eos['s'], s[np.argsort(s)], C_true[np.argsort(s)])
    shape_corr = np.corrcoef(eos['C'], Ctrue_on)[0, 1]
    print("rho range %.3f-%.3f ; cond %.1e ; nmodes=3 resid %.2e ; nmodes=1 resid %.2e"
          % (profiles.min(), profiles.max(), eos['cond'], eos['resid'], eos1['resid']))
    print("max|mu_ex-ref| %.2e ; kernel shape corr %.3f ; kappa2_eff %.3f"
          % (np.max(np.abs(de)), shape_corr, eos['kappa2_eff']))
    ok = (np.max(np.abs(de)) < 3e-2 and shape_corr > 0.95
          and eos['resid'] < 0.6 * eos1['resid'])
    print("  -> EOS + nonlocal kernel recovered, beats 2nd-moment fit"
          if ok else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ladder', help='CSV: dumax,density_file (single rho_avg)')
    ap.add_argument('--lz', type=float); ap.add_argument('--temp', type=float)
    ap.add_argument('--deg', type=int, default=4); ap.add_argument('--smax', type=float, default=2.5)
    ap.add_argument('--nmodes', type=int, default=3); ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--smooth', type=int, default=12)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    rows = [l.split(',') for l in open(args.ladder) if l.strip() and not l.startswith('#')]
    amps, grids = [0.0], []
    for du, fn in rows:
        grids.append(fc._read_density(fn.strip())); amps.append(float(du) / 2.0)
    profiles = np.array([np.full_like(grids[0], np.mean(grids[0]))] + grids)
    eos = kernel_eos(np.array(amps), profiles, args.temp, args.lz, deg=args.deg,
                     smax=args.smax, nmodes=args.nmodes, ridge=args.ridge, smooth=args.smooth)
    print("# kernel: a=%s kappa2_eff=%.4f kappa4_eff=%.4f cond=%.1e resid=%.3e"
          % (np.round(eos['a'], 4), eos['kappa2_eff'], eos['kappa4_eff'], eos['cond'], eos['resid']))
    print("# rho      mu0        P0")
    for r, m, p in zip(eos['rho'][::8], eos['mu0'][::8], eos['P0'][::8]):
        print(f"  {r:6.3f}  {m: .4f}  {p: .4f}")


if __name__ == '__main__':
    main()
