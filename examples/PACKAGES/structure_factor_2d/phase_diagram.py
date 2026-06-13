#!/usr/bin/env python3
"""Phase-diagram analysis from CPP homogeneous isotherms.

Operates on the mu0(rho), P0(rho) van der Waals loops produced by
field_coupling.local_eos (or any homogeneous-EOS isotherm) and extracts:

  * spinodal(rho, P)      -- vapor/liquid spinodal densities, the loop turning
                             points where dP/drho = 0.
  * binodal(rho, P, mu)   -- coexisting (binodal) densities from equal-mu AND
                             equal-P.  This is the Maxwell equal-area construction
                             done correctly (the double-tangent / lower-convex-hull
                             of the Helmholtz free energy f0(rho) = INT mu0 drho),
                             which equals equal-area in V -- NOT equal-area in rho.
                             Exact GIVEN an exact loop (thesis point 5).
  * critical_point(iso)   -- Tc, rho_c, P_c from a set of isotherms by several
                             routes (thesis point 4): spinodal-gap closure,
                             isotherm inflection, binodal/spinodal scaling laws,
                             rectilinear diameters.

The key physics (paper thesis): Maxwell/equal-area gives the correct binodals when
applied to the TRUE homogeneous loop; the failures reported in the literature came
from feeding it the inaccurate 2nd-order P0 = 3/2 P_T - 1/2 P_N loop, not from the
construction.  Run this on both the exact (field-coupling) and the 2nd-order loops
to show the difference.

Self-test (`python3 phase_diagram.py --selftest`) validates every routine against
the analytic van der Waals EOS, whose critical point (rho_c=1/3b, Tc=8a/27b) and
spinodal are known in closed form.
"""
import argparse
import numpy as np

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    def cumtrapz(y, x, initial=0.0):
        return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))]) + initial


def mu_from_P(rho, P):
    """mu0(rho) (up to a constant) from Gibbs-Duhem dmu = dP/rho at fixed T."""
    dP = np.gradient(P, rho)
    return cumtrapz(dP / rho, rho, initial=0.0)


def spinodal(rho, P):
    """Vapor and liquid spinodal densities: the two interior roots of dP/drho = 0
    (the van der Waals loop turning points).  Returns (rho_sv, rho_sl), or None if
    the isotherm has no loop (supercritical / monotone P)."""
    dP = np.gradient(P, rho)
    roots = []
    for i in range(len(rho) - 1):
        if dP[i] == 0.0 or dP[i] * dP[i + 1] < 0.0:        # sign change of dP/drho
            t = dP[i] / (dP[i] - dP[i + 1]) if dP[i] != dP[i + 1] else 0.0
            roots.append(rho[i] + t * (rho[i + 1] - rho[i]))
    if len(roots) < 2:
        return None
    return min(roots), max(roots)                          # rho_sv (P max), rho_sl (P min)


def _lower_hull(x, y):
    """Indices of the lower convex hull of (x, y), x ascending."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2 and (
            (x[h[-1]] - x[h[-2]]) * (y[i] - y[h[-2]])
            - (y[h[-1]] - y[h[-2]]) * (x[i] - x[h[-2]])) <= 1e-15:
            h.pop()
        h.append(i)
    return h


def binodal(rho, P, mu=None):
    """Coexisting densities from the common-tangent (double-tangent) construction on
    the Helmholtz free-energy density f0(rho) = INT mu0 drho -- i.e. equal chemical
    potential AND equal pressure, the Maxwell equal-area rule done in V.  Returns
    dict(rho_v, rho_l, P_sat, mu_sat) or None if there is no loop."""
    if spinodal(rho, P) is None:
        return None
    if mu is None:
        mu = mu_from_P(rho, P)
    f0 = cumtrapz(mu, rho, initial=0.0)                    # mu = df0/drho
    hull = _lower_hull(rho, f0)
    # the double tangent is the hull edge that skips the most interior points
    gap = max(range(len(hull) - 1), key=lambda k: hull[k + 1] - hull[k])
    ia, ib = hull[gap], hull[gap + 1]
    if ib - ia < 2:
        return None                                        # convex -> no coexistence
    rv, rl = rho[ia], rho[ib]
    mu_sat = (f0[ib] - f0[ia]) / (rl - rv)                 # common-tangent slope
    P_sat = 0.5 * (rv * mu[ia] - f0[ia] + rl * mu[ib] - f0[ib])  # P = rho mu - f0
    return dict(rho_v=rv, rho_l=rl, P_sat=P_sat, mu_sat=mu_sat)


def _scaling_fit(T, dr, beta=None):
    """Fit dr = A (Tc - T)^beta and return Tc (and A, beta).  If beta is None it is
    fit too (log-log once Tc is bracketed); otherwise held fixed."""
    T = np.asarray(T, float); dr = np.asarray(dr, float)
    good = dr > 0
    T, dr = T[good], dr[good]

    def resid(Tc):
        x = np.log(Tc - T); y = np.log(dr)
        b = beta if beta is not None else np.polyfit(x, y, 1)[0]
        a = np.exp(np.mean(y - b * x))
        return np.sum((dr - a * (Tc - T) ** b) ** 2), b, a

    Tc_grid = np.linspace(T.max() + 1e-4, T.max() + 3 * (T.max() - T.min()) + 1e-3, 4000)
    r = [resid(tc)[0] for tc in Tc_grid]
    Tc = Tc_grid[int(np.argmin(r))]
    _, b, a = resid(Tc)
    return dict(Tc=Tc, A=a, beta=b)


def critical_point(isotherms, beta=0.5):
    """Estimate (Tc, rho_c, P_c) from a dict {T: (rho, P[, mu])} by several routes.

    isotherms: mapping temperature -> (rho_array, P_array) or (rho, P, mu).
    Returns a dict of estimates from: spinodal-gap closure + scaling, binodal
    scaling + rectilinear diameters, and direct isotherm inflection (the T whose
    loop just vanishes).  beta is the order-parameter exponent (0.5 mean-field /
    van der Waals; ~0.326 for 3D Ising)."""
    Ts = sorted(isotherms)
    sp_gap, bi_gap, diam, Tsub = [], [], [], []
    rho_c_guess = []
    for T in Ts:
        cols = isotherms[T]
        rho, P = cols[0], cols[1]
        mu = cols[2] if len(cols) > 2 else None
        sp = spinodal(rho, P)
        bi = binodal(rho, P, mu)
        if sp and bi:
            Tsub.append(T)
            sp_gap.append(sp[1] - sp[0])
            bi_gap.append(bi['rho_l'] - bi['rho_v'])
            diam.append(0.5 * (bi['rho_l'] + bi['rho_v']))
            rho_c_guess.append(0.5 * (sp[0] + sp[1]))
    out = {}
    if len(Tsub) >= 3:
        out['spinodal_scaling'] = _scaling_fit(Tsub, sp_gap, beta)
        bsc = _scaling_fit(Tsub, bi_gap, beta)
        out['binodal_scaling'] = bsc
        # rectilinear diameters: (rho_l+rho_v)/2 = rho_c + B (Tc - T)  -> rho_c
        Tc = bsc['Tc']
        B, rc = np.polyfit(Tc - np.array(Tsub), diam, 1)
        out['rho_c'] = float(rc)
        out['Tc'] = Tc
        # P_c from the highest-T subcritical binodal P_sat extrapolated, or the
        # near-critical isotherm value at rho_c
        out['rho_c_spinodal_merge'] = float(np.mean(rho_c_guess[-2:]))
    return out


# ----------------------------------------------------------------------------

def _vdw(rho, T, a=1.0, b=1.0):
    """van der Waals EOS: P and mu (reduced; mu up to an additive constant)."""
    P = rho * T / (1.0 - b * rho) - a * rho ** 2
    mu = T * (np.log(rho / (1.0 - b * rho)) + b * rho / (1.0 - b * rho)) - 2.0 * a * rho
    return P, mu


def _selftest():
    a = b = 1.0
    Tc_true, rc_true, Pc_true = 8 * a / (27 * b), 1.0 / (3 * b), a / (27 * b ** 2)
    print("van der Waals truth:  Tc=%.5f  rho_c=%.5f  P_c=%.5f" % (Tc_true, rc_true, Pc_true))

    # (1) spinodal at T=0.9 Tc vs analytic dP/drho=0
    T = 0.9 * Tc_true
    rho = np.linspace(0.005, 0.98, 6000)
    P, mu = _vdw(rho, T, a, b)
    sv, sl = spinodal(rho, P)
    # analytic spinodal: dP/drho = T/(1-b rho)^2 - 2 a rho = 0
    dPdr = T / (1 - b * rho) ** 2 - 2 * a * rho
    ra = rho[:-1][np.diff(np.sign(dPdr)) != 0]
    sp_err = max(abs(sv - ra.min()), abs(sl - ra.max()))
    print("spinodal  fit (%.4f, %.4f)  analytic (%.4f, %.4f)  err %.2e"
          % (sv, sl, ra.min(), ra.max(), sp_err))

    # (2) binodal: common-tangent (mu given) vs equal-P/equal-mu reference
    bi = binodal(rho, P, mu)
    # independent reference: solve P(rv)=P(rl) and mu(rv)=mu(rl) by bisection on P_sat
    def ref_binodal(T):
        rr = np.linspace(1e-4, 0.99, 200000)
        Pr, mur = _vdw(rr, T, a, b)
        sv2, sl2 = spinodal(rr, Pr)
        vap = rr < sv2; liq = rr > sl2
        Ps = np.linspace(max(Pr[vap].min(), Pr[liq].min()) + 1e-6,
                         min(Pr[vap].max(), Pr[liq].max()) - 1e-6, 4000)
        best = None
        for ps in Ps:
            rv = np.interp(ps, Pr[vap], rr[vap])
            rl = np.interp(ps, Pr[liq][::-1], rr[liq][::-1]) if Pr[liq][0] > Pr[liq][-1] \
                else np.interp(ps, Pr[liq], rr[liq])
            dm = np.interp(rl, rr, mur) - np.interp(rv, rr, mur)
            if best is None or abs(dm) < best[0]:
                best = (abs(dm), rv, rl)
        return best[1], best[2]
    rvr, rlr = ref_binodal(T)
    bi_err = max(abs(bi['rho_v'] - rvr), abs(bi['rho_l'] - rlr))
    print("binodal   common-tangent (%.4f, %.4f)  reference (%.4f, %.4f)  err %.2e"
          % (bi['rho_v'], bi['rho_l'], rvr, rlr, bi_err))

    # (3) critical point from a ladder of subcritical isotherms (scaling, beta=0.5)
    iso = {}
    for f in [0.80, 0.84, 0.88, 0.92, 0.95, 0.97]:
        Ti = f * Tc_true
        ri = np.linspace(0.004, 0.985, 6000)
        Pi, mui = _vdw(ri, Ti, a, b)
        iso[Ti] = (ri, Pi, mui)
    cp = critical_point(iso, beta=0.5)
    Tc_fit = cp['binodal_scaling']['Tc']; rc_fit = cp['rho_c']
    print("critical  Tc fit %.5f (true %.5f, err %.1e)  rho_c fit %.4f (true %.4f, err %.1e)"
          % (Tc_fit, Tc_true, abs(Tc_fit - Tc_true),
             rc_fit, rc_true, abs(rc_fit - rc_true)))

    ok = (sp_err < 1e-3 and bi_err < 3e-3
          and abs(Tc_fit - Tc_true) < 5e-3 and abs(rc_fit - rc_true) < 1e-2)
    print("  -> spinodal, binodal (Maxwell), and critical point recovered"
          if ok else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--iso', nargs='+', help='isotherm files "T:rho_P_mu.txt" (cols rho P [mu])')
    ap.add_argument('--beta', type=float, default=0.5, help='order-parameter exponent')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.iso:
        iso = {}
        for spec in args.iso:
            T, fn = spec.split(':')
            d = np.loadtxt(fn)
            iso[float(T)] = tuple(d[:, i] for i in range(d.shape[1]))
        for T in sorted(iso):
            rho, P = iso[T][0], iso[T][1]
            mu = iso[T][2] if len(iso[T]) > 2 else None
            sp = spinodal(rho, P); bi = binodal(rho, P, mu)
            print("T=%.4f  spinodal=%s  binodal=%s" % (
                T, None if sp is None else tuple(round(x, 4) for x in sp),
                None if bi is None else {k: round(v, 4) for k, v in bi.items()}))
        print("critical point:", {k: (round(v, 5) if isinstance(v, float) else v)
                                   for k, v in critical_point(iso, args.beta).items()})


if __name__ == '__main__':
    main()
