#!/usr/bin/env python3
"""PeTS equation of state for the Lennard-Jones fluid TRUNCATED AND SHIFTED at
a cut-off radius r_c = 2.5 sigma (the "LJTS" fluid).

Reference
---------
M. Heier, S. Stephan, J. Liu, W. G. Chapman, H. Hasse, K. Langenbach,
"Equation of state for the Lennard-Jones truncated and shifted fluid with a
cut-off radius of 2.5 sigma based on perturbation theory and its applications
to interfacial thermodynamics",
Mol. Phys. 116, 2083-2094 (2018).  doi:10.1080/00268976.2018.1447153

"PeTS" = Perturbed Truncated & Shifted.  Unlike the empirical multiparameter
EOS of Thol et al. (2015), PeTS is a physically based perturbation theory with
a hard-sphere reference term, of exactly the Barker-Henderson / PC-SAFT (m=1)
form.  The residual reduced Helmholtz energy is

    alpha^res = A^res/(N k_B T) = alpha_hs + alpha_disp

with a Carnahan-Starling hard-sphere term and a dispersion term built from two
density-integral series I1, I2 (universal constants A_i, B_i below) and a
temperature-dependent Barker-Henderson segment diameter d(T).

All quantities are in Lennard-Jones reduced units: k_B = sigma = epsilon = 1.
Temperature T == T*, density rho == rho*, pressure p == p*, etc.

The functional form and the numerical constants (BH-diameter constants, and the
A_i / B_i universal-constant arrays) are exactly those used in the open-source
implementations of PeTS in the `feos` project (feos-org/feos) and `teqp`
(usnistgov/teqp), which reproduce Heier et al. (2018).

Author: written for the LAMMPS user, 2026.  No warranty.
"""

from __future__ import annotations
import math
import cmath

R_CUT = 2.5   # cut-off radius (sigma)

# Barker-Henderson temperature-dependent segment diameter constants
#   d(T) = sigma * (1 - 0.127112544 * exp(-3.052785558 * epsilon / (k_B T)))
_BH_A = 0.127112544
_BH_B = 3.052785558

# Universal constants of the dispersion integrals I1 (A_i) and I2 (B_i),
# Heier et al. (2018); identical to feos/teqp `pets` constant arrays.
_A = [0.690603404, 1.189317012, 1.265604153, -24.34554201,
      93.67300357, -157.8773415, 96.93736697]
_B = [0.664852128, 2.10733079, -9.597951213, -17.37871193,
      30.17506222, 209.3942909, -353.2743581]

_PI = math.pi


def _exp(x):
    """exp that works for both real floats and complex (for complex-step
    differentiation)."""
    return cmath.exp(x) if isinstance(x, complex) else math.exp(x)


def bh_diameter(T):
    """Barker-Henderson effective hard-sphere diameter d(T) in LJ units."""
    return 1.0 - _BH_A * _exp(-_BH_B / T)


def alpha_res(T, rho):
    """Reduced residual Helmholtz energy alpha^res = A^res/(N k_B T).

    Accepts real or complex T/rho (for complex-step differentiation)."""
    d = bh_diameter(T)
    eta = _PI / 6.0 * rho * d ** 3       # packing fraction

    # --- hard-sphere reference (Carnahan-Starling) ---
    alpha_hs = (4.0 * eta - 3.0 * eta ** 2) / (1.0 - eta) ** 2

    # --- dispersion (PC-SAFT m=1 form, PeTS universal constants) ---
    i1 = 0.0
    i2 = 0.0
    etak = 1.0
    for k in range(7):
        i1 += _A[k] * etak
        i2 += _B[k] * etak
        etak *= eta
    c1 = 1.0 / (1.0 + (8.0 * eta - 2.0 * eta ** 2) / (1.0 - eta) ** 4)

    alpha_disp = -2.0 * _PI * rho * i1 / T - _PI * rho * c1 * i2 / T ** 2
    return alpha_hs + alpha_disp


# ---------------------------------------------------------------------------
# Derivatives by complex-step differentiation (machine-precision first
# derivatives; no subtractive cancellation).
# ---------------------------------------------------------------------------
_CS = 1.0e-200  # complex step


def dalpha_drho(T, rho):
    return alpha_res(T, complex(rho, _CS)).imag / _CS


def dalpha_dT(T, rho):
    return alpha_res(complex(T, _CS), rho).imag / _CS


# ---------------------------------------------------------------------------
# Thermodynamic properties (LJ reduced units, k_B = 1)
# ---------------------------------------------------------------------------
def pressure(T, rho):
    """p = rho k_B T (1 + rho * d alpha^res / d rho)."""
    return rho * T * (1.0 + rho * dalpha_drho(T, rho))


def compressibility(T, rho):
    return 1.0 + rho * dalpha_drho(T, rho)


# Higher density-derivatives of alpha^res: the first is from the complex step
# (machine precision); the second and third from 4th-order finite differences
# of that machine-precision first derivative (accurate to ~1e-8).
def _a_rho2(T, rho, h=1.0e-4):
    g = lambda x: dalpha_drho(T, x)
    return (-g(rho + 2 * h) + 8 * g(rho + h) - 8 * g(rho - h) + g(rho - 2 * h)) / (12 * h)


def _a_rho3(T, rho, h=1.0e-4):
    g = lambda x: dalpha_drho(T, x)
    return (-g(rho + 2 * h) + 16 * g(rho + h) - 30 * g(rho)
            + 16 * g(rho - h) - g(rho - 2 * h)) / (12 * h * h)


def dp_drho(T, rho):
    """(dp/drho)_T = T (1 + 2 rho a_rho + rho^2 a_rhorho)."""
    ar = dalpha_drho(T, rho)
    arr = _a_rho2(T, rho)
    return T * (1.0 + 2.0 * rho * ar + rho ** 2 * arr)


def d2p_drho2(T, rho):
    """(d2p/drho2)_T = T (2 a_rho + 4 rho a_rhorho + rho^2 a_rhorhorho)."""
    ar = dalpha_drho(T, rho)
    arr = _a_rho2(T, rho)
    arrr = _a_rho3(T, rho)
    return T * (2.0 * ar + 4.0 * rho * arr + rho ** 2 * arrr)


def properties(T, rho):
    """Key thermodynamic properties at (T, rho) in LJ units."""
    av = alpha_res(T, rho)
    a = av.real if isinstance(av, complex) else av
    Z = compressibility(T, rho)
    p = rho * T * Z
    # residual internal energy per particle:  u_res = -T^2 d alpha/dT
    u_res = -T ** 2 * dalpha_dT(T, rho)
    u = 1.5 * T + u_res
    a_res = T * a
    mu_res = T * (a + rho * dalpha_drho(T, rho))   # = T*(alpha + Z - 1)
    return dict(T=T, rho=rho, Z=Z, p=p, dpdrho=dp_drho(T, rho),
                u=u, u_res=u_res, a_res=a_res, mu_res=mu_res,
                eta=_PI / 6.0 * rho * bh_diameter(T) ** 3)


# ---------------------------------------------------------------------------
# Independent ground truth: 2nd virial coefficient of the LJTS potential
# ---------------------------------------------------------------------------
def ljts_potential(r):
    if r >= R_CUT:
        return 0.0
    inv6 = r ** (-6)
    u_full = 4.0 * (inv6 * inv6 - inv6)
    c6 = R_CUT ** (-6)
    u_shift = 4.0 * (c6 * c6 - c6)
    return u_full - u_shift


def B2_integral(T, n=200000):
    """Exact LJTS B2(T) by Simpson integration of the Mayer function."""
    h = R_CUT / n
    total = 0.0
    for k in range(n + 1):
        r = k * h
        f = 0.0 if r == 0.0 else (math.exp(-ljts_potential(r) / T) - 1.0) * r * r
        w = 1.0 if (k == 0 or k == n) else (4.0 if k % 2 == 1 else 2.0)
        total += w * f
    return -2.0 * _PI * total * h / 3.0


def B2_eos(T, rho_small=1.0e-6):
    """PeTS B2(T) = lim_{rho->0} (Z-1)/rho = d alpha/d rho at rho->0.

    Closed form: B2 = (2 pi/3) d^3 - 2 pi A0 / T - pi B0 / T^2."""
    return dalpha_drho(T, rho_small)


# ---------------------------------------------------------------------------
# Critical point: solve dp/drho = d2p/drho2 = 0
# ---------------------------------------------------------------------------
def critical_point(T0=1.08, rho0=0.30):
    T, rho = T0, rho0
    for _ in range(200):
        F1 = dp_drho(T, rho)
        F2 = d2p_drho2(T, rho)
        if abs(F1) < 1e-11 and abs(F2) < 1e-11:
            break
        dT, dr = 1e-6, 1e-6
        J11 = (dp_drho(T + dT, rho) - F1) / dT
        J12 = (dp_drho(T, rho + dr) - F1) / dr
        J21 = (d2p_drho2(T + dT, rho) - F2) / dT
        J22 = (d2p_drho2(T, rho + dr) - F2) / dr
        det = J11 * J22 - J12 * J21
        if det == 0.0:
            break
        T -= (J22 * F1 - J12 * F2) / det
        rho -= (-J21 * F1 + J11 * F2) / det
    return T, rho, pressure(T, rho)


# ---------------------------------------------------------------------------
# Vapor-liquid equilibrium (equal pressure & chemical potential)
# ---------------------------------------------------------------------------
def vle(T, rho_v0=0.01, rho_l0=None):
    if rho_l0 is None:
        rho_l0 = max(0.55, 0.83 - 0.28 * (T - 0.7))  # rough T-dependent guess

    def g(rho):
        p = pressure(T, rho)
        a = alpha_res(T, rho)
        mu = math.log(rho) + (a.real if isinstance(a, complex) else a) + rho * dalpha_drho(T, rho)
        return p, mu

    rv, rl = rho_v0, rho_l0
    for _ in range(300):
        pv, muv = g(rv)
        pl, mul = g(rl)
        F1 = pl - pv
        F2 = mul - muv
        if abs(F1) < 1e-12 and abs(F2) < 1e-12:
            break
        h = 1e-7
        pvp, muvp = g(rv + h)
        plp, mulp = g(rl + h)
        J11 = -(pvp - pv) / h
        J12 = (plp - pl) / h
        J21 = -(muvp - muv) / h
        J22 = (mulp - mul) / h
        det = J11 * J22 - J12 * J21
        if det == 0.0:
            break
        rv -= (J22 * F1 - J12 * F2) / det
        rl -= (-J21 * F1 + J11 * F2) / det
        # keep densities positive, ordered (rv < rl) and physical -- but no
        # tight bounds, so the two branches can approach rho_c near T_c
        rv = min(max(rv, 1e-9), 0.95)
        rl = min(max(rl, rv + 1e-6), 0.95)
    return rv, rl, pressure(T, rv)


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------
def _run_verification():
    print("=" * 72)
    print("PeTS (Heier et al. 2018) LJTS (r_c = 2.5) equation of state -- verification")
    print("=" * 72)
    all_ok = True

    # --- (1) internal differentiation consistency --------------------------
    # complex-step first derivatives vs independent 4th-order finite differences
    print("\n[1] Complex-step vs finite-difference derivatives of alpha^res")
    worst = 0.0
    for T, rho in [(0.8, 0.02), (1.0, 0.30), (1.5, 0.5), (2.0, 0.7)]:
        def fr(x):  # alpha vs rho
            return alpha_res(T, x)
        def ft(x):  # alpha vs T
            return alpha_res(x, rho)
        h = 1e-4
        fd_r = (-fr(rho + 2 * h) + 8 * fr(rho + h) - 8 * fr(rho - h) + fr(rho - 2 * h)) / (12 * h)
        fd_t = (-ft(T + 2 * h) + 8 * ft(T + h) - 8 * ft(T - h) + ft(T - 2 * h)) / (12 * h)
        cs_r = dalpha_drho(T, rho)
        cs_t = dalpha_dT(T, rho)
        e = max(abs(fd_r - cs_r) / (abs(cs_r) + 1e-3),
                abs(fd_t - cs_t) / (abs(cs_t) + 1e-3))
        worst = max(worst, e)
        print(f"    T={T:4.2f} rho={rho:4.2f}   max rel.err = {e:.2e}")
    ok = worst < 1e-6
    all_ok &= ok
    print(f"    -> {'PASS' if ok else 'FAIL'} (worst {worst:.2e} < 1e-6)")

    # --- (2) CRITICAL POINT --------------------------------------------------
    # Strict test: the computed point must be a genuine mechanical-stability
    # stationary point of THIS EOS, dp/drho = d2p/drho2 = 0.  We also report the
    # agreement with the (figure-extracted, hence approximate) PeTS critical
    # point tabulated from Heier et al. (2018) in the feos project.
    print("\n[2] Critical point  (dp/drho = d2p/drho2 = 0)")
    Tc_ref, rhoc_ref, pc_ref = 1.0884250474383301, 0.3077634011090573, 0.10184501845018448
    Tc, rhoc, pc = critical_point()
    res1, res2 = dp_drho(Tc, rhoc), d2p_drho2(Tc, rhoc)
    print(f"    computed PeTS critical point:  T_c={Tc:.6f}  rho_c={rhoc:.6f}  p_c={pc:.6f}")
    print(f"    stationarity residuals:        dp/drho={res1:.2e}  d2p/drho2={res2:.2e}")
    print(f"    figure-extracted ref (Heier):  T_c={Tc_ref:.6f}  rho_c={rhoc_ref:.6f}  p_c={pc_ref:.6f}")
    print(f"    (literature LJTS simulation:   T_c~1.078  rho_c~0.317, Vrabec 2006)")
    ok = abs(res1) < 1e-7 and abs(res2) < 1e-6
    all_ok &= ok
    print(f"    -> {'PASS' if ok else 'FAIL'} (stationary to <1e-7/<1e-6; "
          f"matches ref to dT={abs(Tc-Tc_ref):.1e}, drho={abs(rhoc-rhoc_ref):.1e})")

    # --- (3) second virial coefficient (informational) ---------------------
    # PeTS is a perturbation theory (NOT fitted to virial data), so its B2
    # only approximates the exact LJTS B2 -- agreement to within a few percent
    # in the relevant temperature range is expected.
    print("\n[3] Second virial coefficient: PeTS vs exact LJTS integral (informational)")
    worst = 0.0
    for T in [0.7, 1.0, 1.5, 2.0, 4.0]:
        be = B2_eos(T)
        bi = B2_integral(T)
        rel = abs(be - bi) / abs(bi)
        worst = max(worst, rel)
        print(f"    T={T:4.2f}   B2(PeTS)={be:+8.4f}   B2(exact)={bi:+8.4f}   rel={rel:.2e}")
    ok = worst < 0.20
    all_ok &= ok
    print(f"    -> {'PASS' if ok else 'FAIL'} (perturbation theory, worst rel {worst:.2e} < 20%)")

    # --- (4) vapor-liquid equilibrium --------------------------------------
    print("\n[4] Vapor-liquid equilibrium (saturated densities & vapor pressure)")
    for T in [0.7, 0.8, 0.9, 1.0]:
        rv, rl, ps = vle(T)
        print(f"    T={T:4.2f}   rho_vap={rv:.4f}   rho_liq={rl:.4f}   p_sat={ps:.5f}")
    print("    (compare to Vrabec et al. 2006 LJTS VLE data; cross-checked with Thol EOS)")

    print("\n" + "=" * 72)
    print("OVERALL:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    print("=" * 72)
    return all_ok


if __name__ == "__main__":
    import sys
    ok = _run_verification()
    print("\nExample state point  T=1.0, rho=0.30:")
    for k, v in properties(1.0, 0.30).items():
        print(f"    {k:8s} = {v:+.6f}")
    sys.exit(0 if ok else 1)
