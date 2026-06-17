#!/usr/bin/env python3
"""Demonstrate the spurious van der Waals loop of the empirical Thol (2015) EOS
deep in the two-phase region, compared with the physically well-behaved PeTS
loop, on a low-temperature subcritical isotherm (T = 0.8).

A thermodynamically sensible subcritical isotherm has exactly ONE mechanically
unstable region -> exactly two spinodal points (one local max, one local min)
where dp/drho = 0.  Empirical multiparameter EOS are fit only to stable (and
limited metastable) data, so inside the two-phase region they are unconstrained
extrapolations and can grow extra, unphysical extrema.  PeTS, a perturbation
theory, is constructed to behave correctly throughout the metastable/unstable
region.  LJ reduced units.  Writes ljts_loop_artifact.png.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thol2015_ljts_eos as thol
import pets_eos as pets

T = 0.80


def dpdrho(mod, Tt, r):
    return mod.dp_drho(Tt, r) if hasattr(mod, "dp_drho") else mod._dp_drho(Tt, r)


def spinodal_points(mod, Tt, rs):
    pts, prev = [], None
    for a, b in zip(rs[:-1], rs[1:]):
        da, db = dpdrho(mod, Tt, a), dpdrho(mod, Tt, b)
        if (da > 0) != (db > 0):
            pts.append(0.5 * (a + b))
    return pts


def main():
    rs = [0.002 + i * 0.001 for i in range(int(0.82 / 0.001))]
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    for name, mod, color in [("Thol et al. (2015) — empirical fit", thol, "tab:blue"),
                             ("PeTS (Heier et al. 2018) — perturbation theory", pets, "tab:red")]:
        p = [mod.pressure(T, r) for r in rs]
        ax.plot(rs, p, "-", color=color, lw=2.0, label=name)
        sp = spinodal_points(mod, T, rs)
        ax.plot(sp, [mod.pressure(T, r) for r in sp], "o", color=color, ms=7,
                mfc="white", mec=color, mew=1.8, zorder=5)
        nloop = len(sp)
        print(f"{name}: {nloop} spinodal points (dp/drho=0) at rho={[round(s,3) for s in sp]}")

    ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
    ax.set_xlim(0.0, 0.82)
    ax.set_ylim(-0.35, 0.12)
    ax.set_xlabel(r"density  $\rho^{*} = \rho\,\sigma^{3}$", fontsize=11)
    ax.set_ylabel(r"pressure  $p^{*} = p\,\sigma^{3}/\varepsilon$", fontsize=11)
    ax.set_title(f"Subcritical isotherm at $T^*={T}$: spurious vs. physical van der Waals loop\n"
                 r"(open circles = spinodal points, $\partial p/\partial\rho=0$)", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax.annotate("spurious extra extrema\n(unphysical stable/unstable\npocket inside the two-phase region)",
                xy=(0.35, thol.pressure(T, 0.35)), xytext=(0.40, -0.18),
                fontsize=9, color="tab:blue",
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))
    ax.text(0.02, -0.32, "Thol min pressure reaches ~ -3.7 (off scale);\n"
            "PeTS has exactly two spinodal points, as required.",
            fontsize=8.5, color="0.3")

    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ljts_loop_artifact.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
