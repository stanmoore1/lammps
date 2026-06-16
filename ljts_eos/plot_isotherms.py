#!/usr/bin/env python3
"""Plot pressure-density isotherms for the two LJTS equations of state at three
temperatures bracketing the critical point: one slightly below T_c (showing the
van der Waals loop / mechanical instability), one exactly at T_c (the critical
inflection, dp/drho = d2p/drho2 = 0 at rho_c), and one slightly above T_c
(monotonic, single phase).

LJ reduced units (k_B = sigma = epsilon = 1).  Writes ljts_isotherms.png.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thol2015_ljts_eos as thol
import pets_eos as pets

DT = 0.05   # temperature offset above/below T_c


def main():
    models = [("Thol et al. (2015)", thol), ("PeTS (Heier et al. 2018)", pets)]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharey=True)

    rhos = [0.002 + i * 0.002 for i in range(int(0.85 / 0.002))]

    for ax, (label, mod) in zip(axes, models):
        Tc, rhoc, pc = mod.critical_point()
        temps = [(Tc - DT, "tab:blue", f"$T = T_c - {DT:g} = {Tc-DT:.3f}$  (subcritical)"),
                 (Tc,      "k",        f"$T = T_c = {Tc:.3f}$  (critical)"),
                 (Tc + DT, "tab:red",  f"$T = T_c + {DT:g} = {Tc+DT:.3f}$  (supercritical)")]

        for T, color, leg in temps:
            p = [mod.pressure(T, r) for r in rhos]
            lw = 2.4 if abs(T - Tc) < 1e-9 else 1.8
            ax.plot(rhos, p, "-", color=color, lw=lw, label=leg)

        # critical point + guide lines
        ax.plot([rhoc], [pc], "o", color="k", ms=8, mfc="white", mec="k",
                mew=1.8, zorder=6, label=f"critical point ({rhoc:.3f}, {pc:.3f})")
        ax.axhline(pc, color="0.6", lw=0.8, ls=":")
        ax.axvline(rhoc, color="0.6", lw=0.8, ls=":")

        # Maxwell coexistence pressure on the subcritical isotherm
        rv, rl, ps = mod.vle(Tc - DT)
        ax.plot([rv, rl], [ps, ps], "-", color="tab:blue", lw=1.0, alpha=0.6)
        ax.plot([rv, rl], [ps, ps], "|", color="tab:blue", ms=9, mew=1.4)
        ax.annotate(f"coexistence\n$p_{{sat}}={ps:.3f}$",
                    xy=(0.5 * (rv + rl), ps), xytext=(0.55, ps - 0.018),
                    fontsize=8, color="tab:blue",
                    ha="left", va="top")

        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"density  $\rho^{*} = \rho\,\sigma^{3}$", fontsize=11)
        ax.set_xlim(0.0, 0.85)
        ax.set_ylim(0.0, 0.20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.3, loc="upper left", framealpha=0.95)

    axes[0].set_ylabel(r"pressure  $p^{*} = p\,\sigma^{3}/\varepsilon$", fontsize=11)
    fig.suptitle(r"LJTS fluid ($r_c = 2.5\,\sigma$): $p$-$\rho$ isotherms around the critical point",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ljts_isotherms.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
    for label, mod in models:
        Tc, rhoc, pc = mod.critical_point()
        print(f"{label:28s}  T_c={Tc:.4f}  rho_c={rhoc:.4f}  p_c={pc:.4f}")


if __name__ == "__main__":
    main()
