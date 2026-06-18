#!/usr/bin/env python3
"""
Reproduce Figs. 4 and 5 of Nichols, Moore, Wheeler, Phys. Rev. E 80, 051203 (2009).

Fourier (new-KB) partial structure factors S_ij(q) are obtained from LAMMPS
compute structure/factor run separately on species 1, species 2, and all atoms.
From S_ij(q) = (A^-1)_ij we derive partial molar volumes, isothermal
compressibility, the activity-coefficient correction Q11, and the direct
correlation functions rho*C_ij(q) = (Y^-1 - A)_ij.
"""
import numpy as np
from scipy.optimize import curve_fit, fsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- physical constants / state point -----
kB = 1.380649e-23            # J/K
NA = 6.02214076e23
T  = 120.0                   # K
y1, y2 = 0.4, 0.6
yv = np.array([y1, y2])
Ymat = np.diag(yv)
Yinv = np.diag(1.0/yv)

SYS = {
    "N1200": dict(N=1200, L=44.626, marker="o", color="C0", label="N = 1200"),
    "N4000": dict(N=4000, L=66.662, marker="^", color="C3", label="N = 4000"),
}

def read_sf(path):
    """Read a LAMMPS fix ave/time mode vector file -> (q[A^-1], S, norms)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) == 2:        # "timestep nrows" separator
                continue
            rows.append([float(p[1]), float(p[2]), float(p[3])])
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]

def partial_S(tag):
    """Return q[nm^-1] and partial structure factors S11,S22,S12 (matrix S=A^-1)."""
    sysd = SYS[tag]
    N = sysd["N"]
    N1, N2 = round(y1*N), round(y2*N)
    q, s1, _ = read_sf(f"sf1_{tag}.txt")   # per-group: <|sum_1 e^{iqr}|^2>/N1
    _, s2, _ = read_sf(f"sf2_{tag}.txt")
    _, sa, _ = read_sf(f"sfa_{tag}.txt")
    # raw shell-averaged |sum|^2 = out * Ngroup
    raw1 = s1 * N1
    raw2 = s2 * N2
    rawa = sa * N
    raw12 = 0.5*(rawa - raw1 - raw2)
    # paper normalization S_ij = (1/N) <sum_i sum_j*>
    S11 = raw1 / N
    S22 = raw2 / N
    S12 = raw12 / N
    return q*10.0, S11, S22, S12   # q in nm^-1

def properties(tag):
    """Per-q thermodynamic properties derived from S_ij(q)."""
    sysd = SYS[tag]
    N, L = sysd["N"], sysd["L"]
    V = L**3                      # A^3
    rho = N / V                   # A^-3
    kT_ig_GPa = 1.0/(rho*1e30 * kB * T) * 1e9   # GPa^-1
    q, S11, S22, S12 = partial_S(tag)
    out = dict(q=q, S11=S11, S22=S22, S12=S12,
               V1=[], V2=[], kT=[], Q11=[], rhoC11=[], rhoC12=[], rhoC22=[])
    for i in range(len(q)):
        S = np.array([[S11[i], S12[i]], [S12[i], S22[i]]])
        A = np.linalg.inv(S)
        Ay = A @ yv
        D = yv @ A @ yv
        Vi = Ay / D / rho * NA * 1e-24        # cm^3/mol
        kT = kT_ig_GPa / D                    # GPa^-1
        B = A - np.outer(Ay, Ay)/D
        M = Ymat @ B
        Q11 = M[0, 0] - M[0, 1]
        rhoC = Yinv - A
        out["V1"].append(Vi[0]); out["V2"].append(Vi[1])
        out["kT"].append(kT); out["Q11"].append(Q11)
        out["rhoC11"].append(rhoC[0, 0]); out["rhoC12"].append(rhoC[0, 1])
        out["rhoC22"].append(rhoC[1, 1])
    for k in list(out):
        out[k] = np.asarray(out[k])
    out["rho"] = rho; out["kT_ig"] = kT_ig_GPa
    return out

# ---------------- empirical extrapolation fits (paper Sec. IV E) -----------
def fit_props(p):
    q = p["q"]
    fV1 = lambda q, a, b: a + b*q**2
    fKi = lambda q, a, b, c: a + b*q + c*q**3        # inverse kT (bulk modulus)
    fQ  = lambda q, a, b: a + b*q
    cV1, _ = curve_fit(fV1, q, p["V1"])
    cKi, _ = curve_fit(fKi, q, 1.0/p["kT"])
    cQ,  _ = curve_fit(fQ,  q, p["Q11"])
    return dict(fV1=fV1, cV1=cV1, fKi=fKi, cKi=cKi, fQ=fQ, cQ=cQ)

def reconstruct_S(p, fits, tag):
    """On a fine q-grid, invert fitted (V1,kT,Q11) back to A and S=A^-1."""
    sysd = SYS[tag]; N, L = sysd["N"], sysd["L"]
    rho = N/L**3
    kT_ig = 1.0/(rho*1e30*kB*T)*1e9
    qf = np.linspace(0.0, p["q"].max()*1.02, 200)
    V1f = fits["fV1"](qf, *fits["cV1"])
    kTf = 1.0/fits["fKi"](qf, *fits["cKi"])
    Q11f = fits["fQ"](qf, *fits["cQ"])
    S11f, S22f, S12f = [], [], []
    V2f = []
    guess = None
    for i in range(len(qf)):
        D = kT_ig/kTf[i]
        Ay1 = V1f[i] * rho / (NA*1e-24) * D       # (Ay)_1
        def eqs(x):
            a, b, c = x
            Dc = y1*y1*a + 2*y1*y2*b + y2*y2*c
            Ay1c = a*y1 + b*y2
            Ay2c = b*y1 + c*y2
            Q11c = y1*((a-b) - Ay1c*(Ay1c-Ay2c)/Dc)
            return [Dc - D, Ay1c - Ay1, Q11c - Q11f[i]]
        if guess is None:
            S = np.array([[p["S11"][0], p["S12"][0]],[p["S12"][0], p["S22"][0]]])
            guess = np.linalg.inv(S).flatten()[[0,1,3]]
        sol = fsolve(eqs, guess, full_output=False)
        guess = sol
        a, b, c = sol
        A = np.array([[a, b],[b, c]])
        S = np.linalg.inv(A)
        S11f.append(S[0,0]); S22f.append(S[1,1]); S12f.append(S[0,1])
        Ay = A@yv; Dd = yv@A@yv
        V2f.append((Ay[1]/Dd/rho*NA*1e-24))
    return qf, np.array(S11f), np.array(S22f), np.array(S12f), np.array(V2f)

# ----------------------------- plotting -----------------------------------
def main():
    data = {t: properties(t) for t in SYS if __import__("os").path.exists(f"sf1_{t}.txt")}
    tags = list(data)
    print("Loaded:", tags)
    for t in tags:
        p = data[t]
        print(f"\n=== {t} ===  kT_ig={p['kT_ig']:.2f} GPa^-1")
        for i in range(len(p["q"])):
            print(f" q={p['q'][i]:5.2f}  S11={p['S11'][i]:7.3f} S22={p['S22'][i]:7.3f} "
                  f"S12={p['S12'][i]:7.3f}  V1={p['V1'][i]:6.1f} V2={p['V2'][i]:6.1f} "
                  f"kT={p['kT'][i]:6.2f} Q11={p['Q11'][i]:6.3f}")

    fits = fit_props(data["N1200"]) if "N1200" in data else None

    # ---------------- FIGURE 4 ----------------
    fig, ax = plt.subplots(4, 1, figsize=(5.0, 10.5), sharex=True)
    for t in tags:
        p, s = data[t], SYS[t]
        mk = dict(marker=s["marker"], mfc="none", mec=s["color"], ls="none", ms=5, label=s["label"])
        ax[0].plot(p["q"], p["S11"], **mk)
        ax[0].plot(p["q"], p["S22"], marker=s["marker"], mfc="none", mec=s["color"], ls="none", ms=5)
        ax[0].plot(p["q"], p["S12"], marker=s["marker"], mfc="none", mec=s["color"], ls="none", ms=5)
        ax[1].plot(p["q"], p["V1"], **mk); ax[1].plot(p["q"], p["V2"], marker=s["marker"], mfc="none", mec=s["color"], ls="none", ms=5)
        ax[2].plot(p["q"], p["kT"], **mk)
        ax[3].plot(p["q"], p["Q11"], **mk)
    if fits is not None:
        qf, S11f, S22f, S12f, V2f = reconstruct_S(data["N1200"], fits, "N1200")
        ax[0].plot(qf, S11f, "c-", lw=1.2, label="empir fit"); ax[0].plot(qf, S22f, "c-", lw=1.2); ax[0].plot(qf, S12f, "c-", lw=1.2)
        ax[1].plot(qf, fits["fV1"](qf, *fits["cV1"]), "c-", lw=1.2); ax[1].plot(qf, V2f, "c-", lw=1.2)
        ax[2].plot(qf, 1.0/fits["fKi"](qf, *fits["cKi"]), "c-", lw=1.2)
        ax[3].plot(qf, fits["fQ"](qf, *fits["cQ"]), "c-", lw=1.2)
        print("\nq->0 extrapolations (new KB):")
        print(f"  V1   = {fits['cV1'][0]:.2f} cm3/mol")
        print(f"  kT   = {1.0/fits['cKi'][0]:.3f} GPa^-1")
        print(f"  Q11  = {fits['cQ'][0]:.3f}")
        print(f"  V2(0)= {V2f[0]:.2f} cm3/mol")

    ax[0].set_ylabel(r"$S_{ij}$"); ax[0].set_ylim(-2, 2.5)
    ax[0].text(0.97, 0.92, r"$S_{22}$", transform=ax[0].transAxes, ha="right")
    ax[0].text(0.97, 0.62, r"$S_{11}$", transform=ax[0].transAxes, ha="right")
    ax[0].text(0.97, 0.12, r"$S_{12}$", transform=ax[0].transAxes, ha="right")
    ax[1].set_ylabel(r"vol (cm$^3$/mol)"); ax[1].set_ylim(30, 60)
    ax[1].text(0.5, 0.85, r"$\bar V_1$", transform=ax[1].transAxes)
    ax[1].text(0.5, 0.18, r"$\bar V_2$", transform=ax[1].transAxes)
    ax[2].set_ylabel(r"$\kappa_T$ (GPa$^{-1}$)"); ax[2].set_ylim(2.0, 3.2)
    ax[2].text(0.6, 0.3, r"$\kappa_T$", transform=ax[2].transAxes)
    ax[3].set_ylabel(r"$Q_{11}$"); ax[3].set_ylim(0, 1.0)
    ax[3].text(0.6, 0.3, r"$Q_{11}$", transform=ax[3].transAxes)
    ax[3].set_xlabel(r"$q$ (nm$^{-1}$)"); ax[3].set_xlim(0, 6)
    ax[0].legend(loc="lower right", fontsize=8, frameon=False)
    for i, lab in enumerate("abcd"):
        ax[i].text(0.02, 0.9, f"({lab})", transform=ax[i].transAxes, fontweight="bold")
    fig.suptitle("FIG. 4  Properties from Fourier analysis, $y_1=0.4$", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.98])
    fig.savefig("figure4.png", dpi=150)
    print("\nwrote figure4.png")

    # ---------------- FIGURE 5 ----------------
    fig2, ax2 = plt.subplots(figsize=(5.2, 4.2))
    for t in tags:
        p, s = data[t], SYS[t]
        for key in ("rhoC11", "rhoC12", "rhoC22"):
            ax2.plot(p["q"], p[key], marker=s["marker"], mfc="none", mec=s["color"],
                     ls="none", ms=5, label=s["label"] if key == "rhoC11" else None)
    if fits is not None:
        # empirical rho*C from reconstructed A on fine grid
        qf, S11f, S22f, S12f, _ = reconstruct_S(data["N1200"], fits, "N1200")
        c11, c12, c22 = [], [], []
        for i in range(len(qf)):
            S = np.array([[S11f[i], S12f[i]],[S12f[i], S22f[i]]])
            rhoC = Yinv - np.linalg.inv(S)
            c11.append(rhoC[0,0]); c12.append(rhoC[0,1]); c22.append(rhoC[1,1])
        ax2.plot(qf, c11, "c-", lw=1.2, label="empir fit")
        ax2.plot(qf, c12, "c-", lw=1.2); ax2.plot(qf, c22, "c-", lw=1.2)
    ax2.set_xlabel(r"$q$ (nm$^{-1}$)"); ax2.set_ylabel(r"$\rho C_{ij}$")
    ax2.set_xlim(0, 6); ax2.set_ylim(-26, -8)
    ax2.text(0.5, 0.8, r"$\rho C_{22}$", transform=ax2.transAxes)
    ax2.text(0.5, 0.5, r"$\rho C_{12}$", transform=ax2.transAxes)
    ax2.text(0.5, 0.2, r"$\rho C_{11}$", transform=ax2.transAxes)
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    fig2.suptitle("FIG. 5  Direct correlation functions, $y_1=0.4$", fontsize=10)
    fig2.tight_layout()
    fig2.savefig("figure5.png", dpi=150)
    print("wrote figure5.png")

if __name__ == "__main__":
    main()
