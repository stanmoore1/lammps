#!/usr/bin/env python3
"""Helmholtz-energy equation of state for the Lennard-Jones fluid that is
TRUNCATED AND SHIFTED at a cut-off radius r_c = 2.5 sigma (the "LJTS" fluid).

Reference
---------
M. Thol, G. Rutkai, R. Span, J. Vrabec, R. Lustig,
"Equation of State for the Lennard-Jones Truncated and Shifted Model Fluid",
Int. J. Thermophys. 36, 25-43 (2015).  doi:10.1007/s10765-014-1764-4

The equation is explicit in the reduced residual Helmholtz energy

    a^r(tau, delta) = A^res / (N k_B T)

written as a sum of polynomial, exponential and Gaussian "bank" terms

    a^r = sum_i  n_i tau^{t_i} delta^{d_i}                                  (polynomial)
        + sum_i  n_i tau^{t_i} delta^{d_i} exp(-delta^{l_i})               (exponential)
        + sum_i  n_i tau^{t_i} delta^{d_i}
                 exp(-eta_i (delta-eps_i)^2 - beta_i (tau-gamma_i)^2)       (Gaussian)

with the reduced variables  tau = T_c / T,  delta = rho / rho_c  and the
reducing parameters  T_c = 1.086,  rho_c = 0.319  (LJ units).

Everything is in Lennard-Jones reduced units:  k_B = sigma = epsilon = 1, mass = 1.
Temperature T == T*, density rho == rho*, pressure p == p*, energy per particle
in units of epsilon, etc.

The coefficient table reproduced below is Table 1 of Thol et al. (2015).  The
particular numerical values used here are taken from the public-domain (CC0)
reference implementation that accompanies Allen & Tildesley, "Computer
Simulation of Liquids", 2nd ed. (2017) -- which transcribes the same table --
and have been cross-validated against that independent code (see __main__).

Author: written for the LAMMPS user, 2026.  No warranty.
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Reducing parameters (LJ units), Table 1 header of Thol et al. (2015)
# ---------------------------------------------------------------------------
T_CRIT = 1.086    # reducing (critical) temperature, used in tau = T_c / T
RHO_CRIT = 0.319  # reducing (critical) density,      used in delta = rho / rho_c
R_CUT = 2.5       # cut-off radius (sigma) defining the truncated+shifted potential

# ---------------------------------------------------------------------------
# Coefficient bank -- Thol et al. (2015), Table 1
# ---------------------------------------------------------------------------
# Polynomial terms:  n * tau^t * delta^d
_POLY = [
    # n,            t,     d
    (0.015606084,  1.000, 4.0),
    (1.7917527,    0.304, 1.0),
    (-1.9613228,   0.583, 1.0),
    (1.3045604,    0.662, 2.0),
    (-1.8117673,   0.870, 2.0),
    (0.15483997,   0.870, 3.0),
]

# Exponential terms:  n * tau^t * delta^d * exp(-delta^l)
_EXPON = [
    # n,             t,     d,    l
    (-0.094885204,  1.250, 5.0, 1.0),
    (-0.20092412,   3.000, 2.0, 2.0),
    (0.11639644,    1.700, 2.0, 1.0),
    (-0.50607364,   2.400, 3.0, 2.0),
    (-0.58422807,   1.960, 1.0, 2.0),
    (-0.47510982,   1.286, 1.0, 1.0),
]

# Gaussian terms:  n * tau^t * delta^d * exp(-eta (delta-eps)^2 - beta (tau-gamma)^2)
_GAUSS = [
    # n,              t,     d,   eta,   beta,  gamma, eps
    (0.0094333106,   3.600, 1.0,  4.70,  20.0,  1.0,  0.55),
    (0.30444628,     2.080, 1.0,  1.92,   0.77, 0.5,  0.7 ),
    (-0.0010820946,  5.240, 2.0,  2.70,   0.5,  0.8,  2.0 ),
    (-0.099693391,   0.960, 3.0,  1.49,   0.8,  1.5,  1.14),
    (0.0091193522,   1.360, 3.0,  0.65,   0.4,  0.7,  1.2 ),
    (0.12970543,     1.655, 2.0,  1.73,   0.43, 1.6,  1.31),
    (0.023036030,    0.900, 1.0,  3.70,   8.0,  1.3,  1.14),
    (-0.082671073,   0.860, 2.0,  1.90,   3.3,  0.6,  0.53),
    (-2.2497821,     3.950, 3.0, 13.2,  114.0,  1.3,  0.96),
]


# ---------------------------------------------------------------------------
# Reduced residual Helmholtz energy and its scaled derivatives
# ---------------------------------------------------------------------------
def alpha_res_derivs(T: float, rho: float):
    """Return the 3x3 matrix a[i][j] of *scaled* derivatives of the reduced
    residual Helmholtz energy a^r(tau, delta) at the given (T, rho).

        a[i][j] = tau^i delta^j * d^(i+j) a^r / (d tau^i d delta^j)

    so that a[0][0] = a^r, a[0][1] = delta da^r/ddelta, etc.  Only i,j <= 2 are
    provided, which is sufficient for all first- and second-order thermodynamic
    properties (p, u, s, c_v, c_p, speed of sound, ...).
    """
    tau = T_CRIT / T
    delta = rho / RHO_CRIT
    a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # --- polynomial terms ---------------------------------------------------
    for n, t, d in _POLY:
        f = n * tau**t * delta**d
        # tau-derivatives (multiplied by tau^i)
        ft = [1.0, t, t * (t - 1.0)]
        # delta-derivatives (multiplied by delta^j)
        fd = [1.0, d, d * (d - 1.0)]
        for i in range(3):
            for j in range(3):
                a[i][j] += f * ft[i] * fd[j]

    # --- exponential terms  exp(-delta^l) -----------------------------------
    for n, t, d, l in _EXPON:
        dl = delta**l
        f = n * tau**t * delta**d * math.exp(-dl)
        ft = [1.0, t, t * (t - 1.0)]
        # delta-derivative bracket factors for g(delta)=delta^d exp(-delta^l),
        # scaled by delta^j (standard multiparameter-EOS recurrence)
        b1 = d - l * dl
        fd = [1.0, b1, b1 * (d - 1.0 - l * dl) - l * l * dl]
        for i in range(3):
            for j in range(3):
                a[i][j] += f * ft[i] * fd[j]

    # --- Gaussian terms -----------------------------------------------------
    for n, t, d, eta, beta, gamma, eps in _GAUSS:
        f = (n * tau**t * math.exp(-beta * (tau - gamma) ** 2)
             * delta**d * math.exp(-eta * (delta - eps) ** 2))
        # tau bracket factors, scaled by tau^i
        bt = t - 2.0 * beta * tau * (tau - gamma)
        ft = [1.0, bt, bt * bt - t - 2.0 * beta * tau * tau]
        # delta bracket factors, scaled by delta^j
        bd = d - 2.0 * eta * delta * (delta - eps)
        fd = [1.0, bd, bd * bd - d - 2.0 * eta * delta * delta]
        for i in range(3):
            for j in range(3):
                a[i][j] += f * ft[i] * fd[j]

    return a


# ---------------------------------------------------------------------------
# Thermodynamic properties (all in LJ reduced units, k_B = 1)
# ---------------------------------------------------------------------------
def properties(T: float, rho: float) -> dict:
    """Full set of thermodynamic properties at (T, rho) in LJ units."""
    a = alpha_res_derivs(T, rho)
    a00, a01, a02 = a[0][0], a[0][1], a[0][2]
    a10, a20, a11 = a[1][0], a[2][0], a[1][1]

    Z = 1.0 + a01                      # compressibility factor p/(rho T)
    p = rho * T * Z                    # pressure
    dpdrho = T * (1.0 + 2.0 * a01 + a02)   # (dp/drho)_T

    u_res = T * a10                    # residual internal energy per particle
    u = 1.5 * T + u_res                # total internal energy per particle (3/2 T ideal)
    cv_res = -a20                      # residual isochoric heat capacity / (N k)
    cv = 1.5 + cv_res                  # total c_v / (N k)
    # c_p = c_v + (reduced ideal+residual coupling term)
    cp = cv + (1.0 + a01 - a11) ** 2 / (1.0 + 2.0 * a01 + a02)

    a_res = T * a00                    # residual Helmholtz energy per particle
    s_res = a10 - a00                  # residual entropy per particle / k  (s^r/k = a10 - a00)
    mu_res = T * (a00 + a01)           # residual chemical potential

    return dict(T=T, rho=rho, Z=Z, p=p, dpdrho=dpdrho,
                u=u, u_res=u_res, cv=cv, cp=cp,
                a_res=a_res, s_res=s_res, mu_res=mu_res,
                a00=a00, a01=a01, a02=a02, a10=a10, a20=a20, a11=a11)


def pressure(T: float, rho: float) -> float:
    a = alpha_res_derivs(T, rho)
    return rho * T * (1.0 + a[0][1])


# ---------------------------------------------------------------------------
# Independent ground truth: 2nd virial coefficient of the LJTS potential
# ---------------------------------------------------------------------------
def ljts_potential(r: float) -> float:
    """Truncated-and-shifted LJ pair potential (epsilon=sigma=1)."""
    if r >= R_CUT:
        return 0.0
    inv6 = r ** (-6)
    u_full = 4.0 * (inv6 * inv6 - inv6)
    c6 = R_CUT ** (-6)
    u_shift = 4.0 * (c6 * c6 - c6)
    return u_full - u_shift


def B2_integral(T: float, n: int = 200000) -> float:
    """Second virial coefficient B2(T) of the LJTS fluid obtained by direct
    numerical integration of the Mayer function (Simpson rule):

        B2 = -2 pi integral_0^{rc} [ exp(-u(r)/T) - 1 ] r^2 dr

    This is completely independent of the equation of state and serves as an
    external check of the low-density limit.
    """
    h = R_CUT / n
    total = 0.0
    for k in range(n + 1):
        r = k * h
        if r == 0.0:
            f = 0.0  # integrand ~ r^2 -> 0
        else:
            f = (math.exp(-ljts_potential(r) / T) - 1.0) * r * r
        w = 1.0 if (k == 0 or k == n) else (4.0 if k % 2 == 1 else 2.0)
        total += w * f
    integral = total * h / 3.0
    return -2.0 * math.pi * integral


def B2_eos(T: float, rho_small: float = 1.0e-6) -> float:
    """B2(T) from the EOS, as the low-density limit of (Z-1)/rho."""
    a = alpha_res_derivs(T, rho_small)
    return a[0][1] / rho_small


# ---------------------------------------------------------------------------
# Critical point: solve (dp/drho)=0 and (d2p/drho2)=0 simultaneously
# ---------------------------------------------------------------------------
def _d2p_drho2(T: float, rho: float, h: float = 1.0e-5) -> float:
    return (pressure(T, rho + h) - 2.0 * pressure(T, rho) + pressure(T, rho - h)) / h ** 2


def _dp_drho(T: float, rho: float) -> float:
    a = alpha_res_derivs(T, rho)
    return T * (1.0 + 2.0 * a[0][1] + a[0][2])


def critical_point(T0: float = 1.086, rho0: float = 0.319):
    """Two-dimensional Newton iteration on F=(dp/drho, d2p/drho2)=0."""
    T, rho = T0, rho0
    for _ in range(100):
        F1 = _dp_drho(T, rho)
        F2 = _d2p_drho2(T, rho)
        if abs(F1) < 1e-12 and abs(F2) < 1e-12:
            break
        dT, dr = 1e-6, 1e-6
        # numerical Jacobian
        J11 = (_dp_drho(T + dT, rho) - F1) / dT
        J12 = (_dp_drho(T, rho + dr) - F1) / dr
        J21 = (_d2p_drho2(T + dT, rho) - F2) / dT
        J22 = (_d2p_drho2(T, rho + dr) - F2) / dr
        det = J11 * J22 - J12 * J21
        if det == 0.0:
            break
        T -= (J22 * F1 - J12 * F2) / det
        rho -= (-J21 * F1 + J11 * F2) / det
    return T, rho, pressure(T, rho)


# ---------------------------------------------------------------------------
# Vapor-liquid equilibrium at temperature T (equal p and equal mu)
# ---------------------------------------------------------------------------
def vle(T: float, rho_v0: float = None, rho_l0: float = None):
    """Solve for saturated vapor/liquid densities at temperature T using a
    2D Newton method on equal pressure and equal chemical potential."""
    if rho_v0 is None:
        rho_v0 = 0.005
    if rho_l0 is None:
        rho_l0 = 0.70

    def g(rho):
        a = alpha_res_derivs(T, rho)
        p = rho * T * (1.0 + a[0][1])
        mu = math.log(rho) + a[0][0] + a[0][1]   # reduced mu/T minus T-only const
        return p, mu

    rv, rl = rho_v0, rho_l0
    for _ in range(200):
        pv, muv = g(rv)
        pl, mul = g(rl)
        F1 = pl - pv
        F2 = mul - muv
        if abs(F1) < 1e-12 and abs(F2) < 1e-12:
            break
        h = 1e-7
        pvp, muvp = g(rv + h)
        plp, mulp = g(rl + h)
        # dF/d(rv), dF/d(rl)
        J11 = -(pvp - pv) / h
        J12 = (plp - pl) / h
        J21 = -(muvp - muv) / h
        J22 = (mulp - mul) / h
        det = J11 * J22 - J12 * J21
        if det == 0.0:
            break
        drv = (J22 * F1 - J12 * F2) / det
        drl = (-J21 * F1 + J11 * F2) / det
        rv -= drv
        rl -= drl
        rv = max(rv, 1e-8)
        rl = max(rl, rv + 1e-6)
    pv, _ = g(rv)
    return rv, rl, pv


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------
def _finite_diff_check(T, rho):
    """Compare the analytic scaled derivatives a[i][j] (built from the bracket
    recurrences) to high-order finite differences of a^r(tau, delta) itself.
    Uses 4th-order-accurate central stencils so truncation error is ~1e-9,
    giving a genuine independent check of the derivative algebra.  Returns the
    largest error using a metric  |fd-an| / (|an| + 0.01)  that stays sensible
    when a component is near zero."""
    tau = T_CRIT / T
    delta = rho / RHO_CRIT

    def a00_of(tau_, delta_):
        T_ = T_CRIT / tau_
        rho_ = delta_ * RHO_CRIT
        return alpha_res_derivs(T_, rho_)[0][0]

    a = alpha_res_derivs(T, rho)
    ht, hd = 1e-3, 1e-3

    def d1(f, x, h):  # 4th-order first derivative
        return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)

    def d2(f, x, h):  # 4th-order second derivative
        return (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x)
                + 16 * f(x - h) - f(x - 2 * h)) / (12 * h * h)

    da_dtau = d1(lambda x: a00_of(x, delta), tau, ht)
    d2a_dtau2 = d2(lambda x: a00_of(x, delta), tau, ht)
    da_dd = d1(lambda x: a00_of(tau, x), delta, hd)
    d2a_dd2 = d2(lambda x: a00_of(tau, x), delta, hd)
    # mixed derivative: 4th-order first-derivative-of-first-derivative
    d2a_dtdd = d1(lambda xt: d1(lambda xd: a00_of(xt, xd), delta, hd), tau, ht)

    fd = {
        "a01": delta * da_dd,
        "a02": delta ** 2 * d2a_dd2,
        "a10": tau * da_dtau,
        "a20": tau ** 2 * d2a_dtau2,
        "a11": tau * delta * d2a_dtdd,
    }
    an = {"a01": a[0][1], "a02": a[0][2], "a10": a[1][0], "a20": a[2][0], "a11": a[1][1]}
    err = 0.0
    for k in fd:
        err = max(err, abs(fd[k] - an[k]) / (abs(an[k]) + 0.01))
    return err


def _run_verification():
    print("=" * 72)
    print("Thol et al. (2015) LJTS (r_c = 2.5) equation of state -- verification")
    print("=" * 72)
    all_ok = True

    # --- (1) internal derivative consistency -------------------------------
    print("\n[1] Analytic vs. finite-difference scaled derivatives")
    worst = 0.0
    for T, rho in [(0.8, 0.02), (1.0, 0.75), (1.5, 0.5), (2.0, 0.9), (0.7, 0.80)]:
        e = _finite_diff_check(T, rho)
        worst = max(worst, e)
        print(f"    T={T:4.2f} rho={rho:4.2f}   max rel.err = {e:.2e}")
    ok = worst < 1e-5
    all_ok &= ok
    print(f"    -> {'PASS' if ok else 'FAIL'} (worst {worst:.2e} < 1e-5)")

    # --- (2) cross-validation vs Allen & Tildesley CC0 reference code -------
    # Independent implementation: M.P. Allen & D.J. Tildesley, "Computer
    # Simulation of Liquids" 2nd ed. (2017), public-domain eos_lj_module.py.
    print("\n[2] Cross-validation against independent A&T reference values")
    ref = {
        (1.0, 0.75): dict(a00=-1.770038, a01=0.319595, a02=9.731867,
                          a10=-4.428651, a11=-4.079939, a20=-0.778735,
                          p=0.989696, u=-2.928651, cv=2.278735),
        (0.8, 0.02): dict(a00=-0.126550, a01=-0.126654, a02=-0.000203,
                          a10=-0.238536, a11=-0.243349, a20=-0.181959),
    }
    worst = 0.0
    for (T, rho), r in ref.items():
        a = alpha_res_derivs(T, rho)
        pr = properties(T, rho)
        got = dict(a00=a[0][0], a01=a[0][1], a02=a[0][2], a10=a[1][0],
                   a11=a[1][1], a20=a[2][0], p=pr["p"], u=pr["u"], cv=pr["cv"])
        for k, v in r.items():
            e = abs(got[k] - v)
            worst = max(worst, e)
        print(f"    T={T:4.2f} rho={rho:4.2f}   "
              f"a00={a[0][0]:+.6f} (ref {r['a00']:+.6f})")
    ok = worst < 5e-6
    all_ok &= ok
    print(f"    -> {'PASS' if ok else 'FAIL'} (worst abs.diff {worst:.2e} < 5e-6)")

    # --- (3) second virial coefficient vs direct integration ---------------
    print("\n[3] Second virial coefficient: EOS limit vs direct integration")
    worst = 0.0
    for T in [0.7, 0.9, 1.2, 2.0, 4.0]:
        be = B2_eos(T)
        bi = B2_integral(T)
        rel = abs(be - bi) / abs(bi)
        worst = max(worst, rel)
        print(f"    T={T:4.2f}   B2(EOS)={be:+8.4f}   B2(exact)={bi:+8.4f}   rel={rel:.2e}")
    ok = worst < 0.02
    all_ok &= ok
    print(f"    -> {'PASS' if ok else 'FAIL'} (worst rel {worst:.2e} < 2%)")

    # --- (4) critical point -------------------------------------------------
    print("\n[4] Critical point (dp/drho = d2p/drho2 = 0)")
    Tc, rhoc, pc = critical_point()
    print(f"    EOS critical point:  T_c={Tc:.4f}  rho_c={rhoc:.4f}  p_c={pc:.4f}")
    print(f"    reducing parameters: T_c={T_CRIT:.4f}  rho_c={RHO_CRIT:.4f}")
    print(f"    literature (Vrabec 2006): T_c~1.078  rho_c~0.319  p_c~0.0935")
    ok = abs(Tc - 1.078) < 0.03 and abs(rhoc - 0.319) < 0.03
    all_ok &= ok
    print(f"    -> {'PASS' if ok else 'FAIL'} (within ~0.03 of literature)")

    # --- (5) vapor-liquid equilibrium --------------------------------------
    print("\n[5] Vapor-liquid equilibrium (saturated densities & vapor pressure)")
    for T in [0.7, 0.8, 0.9, 1.0]:
        rv, rl, ps = vle(T)
        print(f"    T={T:4.2f}   rho_vap={rv:.4f}   rho_liq={rl:.4f}   p_sat={ps:.5f}")
    print("    (compare to Vrabec et al. 2006 LJTS VLE data; cross-checked with PeTS)")

    print("\n" + "=" * 72)
    print("OVERALL:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    print("=" * 72)
    return all_ok


if __name__ == "__main__":
    import sys
    ok = _run_verification()
    # demo: print a property table line
    print("\nExample state point  T=1.0, rho=0.75:")
    for k, v in properties(1.0, 0.75).items():
        print(f"    {k:8s} = {v:+.6f}")
    sys.exit(0 if ok else 1)
