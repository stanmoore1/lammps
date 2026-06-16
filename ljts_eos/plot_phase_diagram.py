#!/usr/bin/env python3
"""Plot the critical point, binodal (vapour-liquid coexistence) and spinodal
(limit of mechanical stability, dp/drho = 0) curves for both LJTS equations of
state -- Thol et al. (2015) and PeTS (Heier et al. 2018) -- on the same axes.

Everything is in Lennard-Jones reduced units (k_B = sigma = epsilon = 1).
Produces  ljts_phase_diagram.png  in this directory.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thol2015_ljts_eos as thol
import pets_eos as pets


# ---------------------------------------------------------------------------
# Uniform accessors (the two modules use slightly different private names)
# ---------------------------------------------------------------------------
def dpdrho(mod, T, rho):
    return mod.dp_drho(T, rho) if hasattr(mod, "dp_drho") else mod._dp_drho(T, rho)


# ---------------------------------------------------------------------------
# Binodal: march in temperature using the previous solution as the initial
# guess (numerical continuation), which keeps the VLE solver well-behaved
# right up to the critical point.
# ---------------------------------------------------------------------------
def binodal(mod, Tc, rhoc, T_low=0.60):
    Tv, rho_v, rho_l = [], [], []
    rv, rl = 0.0015, 0.82           # low-T initial guess (dilute vapour / dense liquid)
    T = T_low
    while T < Tc - 1.5e-3:
        rv_s, rl_s, _ = mod.vle(T, rv, rl)
        if not (1e-6 < rv_s < rl_s) or (rl_s - rv_s) < 3e-3:
            break                   # branches have merged -> stop below T_c
        Tv.append(T); rho_v.append(rv_s); rho_l.append(rl_s)
        rv, rl = rv_s, rl_s         # continuation
        # finer steps close to the critical point where the curve turns over
        T += 0.02 if (Tc - T) > 0.06 else 0.005
    return Tv, rho_v, rho_l


# ---------------------------------------------------------------------------
# Spinodal: for each T < T_c, dp/drho = 0 has two roots -- the vapour-side and
# liquid-side limits of mechanical stability.  Found by scanning rho for sign
# changes of dp/drho, then refining by bisection.
# ---------------------------------------------------------------------------
def _bisect_root(f, a, b, tol=1e-9):
    fa = f(a)
    for _ in range(100):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if (fa > 0) == (fm > 0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def spinodal(mod, Tc, rhoc, T_low=0.60):
    Tv, rho_v, rho_l = [], [], []
    T = T_low
    while T < Tc - 1.0e-3:
        # scan dp/drho over a density grid and collect sign-change brackets
        grid = [0.0008 + i * 0.0015 for i in range(int(0.84 / 0.0015))]
        vals = [(r, dpdrho(mod, T, r)) for r in grid]
        roots = []
        for (r0, f0), (r1, f1) in zip(vals[:-1], vals[1:]):
            if (f0 > 0) != (f1 > 0):
                roots.append(_bisect_root(lambda r: dpdrho(mod, T, r), r0, r1))
        if len(roots) >= 2:
            Tv.append(T); rho_v.append(roots[0]); rho_l.append(roots[-1])
        T += 0.02 if (Tc - T) > 0.06 else 0.004
    return Tv, rho_v, rho_l


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------
def main():
    models = [
        ("Thol et al. (2015)", thol, "tab:blue"),
        ("PeTS (Heier et al. 2018)", pets, "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    for label, mod, color in models:
        Tc, rhoc, pc = mod.critical_point()
        print(f"{label:28s}  T_c={Tc:.4f}  rho_c={rhoc:.4f}  p_c={pc:.4f}")

        # binodal
        Tb, rv, rl = binodal(mod, Tc, rhoc)
        rho_bin = rv[::-1] + [rhoc] + rl   # vapour branch + apex + liquid branch
        T_bin = Tb[::-1] + [Tc] + Tb
        ax.plot(rho_bin, T_bin, "-", color=color, lw=2.0,
                label=f"{label} — binodal")

        # spinodal
        Ts, sv, sl = spinodal(mod, Tc, rhoc)
        rho_spin = sv[::-1] + [rhoc] + sl
        T_spin = Ts[::-1] + [Tc] + Ts
        ax.plot(rho_spin, T_spin, "--", color=color, lw=1.6,
                label=f"{label} — spinodal")

        # critical point
        ax.plot([rhoc], [Tc], "o", color=color, ms=9, mfc="white",
                mec=color, mew=2.0, zorder=5,
                label=f"{label} — critical point")

    ax.set_xlabel(r"density  $\rho^{*} = \rho\,\sigma^{3}$", fontsize=12)
    ax.set_ylabel(r"temperature  $T^{*} = k_{\mathrm{B}}T/\varepsilon$", fontsize=12)
    ax.set_title("LJTS fluid ($r_c = 2.5\\,\\sigma$): binodal, spinodal and critical point",
                 fontsize=12.5)
    ax.set_xlim(0.0, 0.85)
    ax.set_ylim(0.60, 1.12)
    ax.grid(True, alpha=0.3)

    # annotate regions
    ax.text(0.43, 1.05, "supercritical / single phase",
            ha="center", fontsize=9, style="italic", color="0.35")
    ax.text(0.43, 0.665, "two-phase  (vapour + liquid)",
            ha="center", fontsize=9, style="italic", color="0.35")

    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ljts_phase_diagram.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
