#!/usr/bin/env python3
"""
Post-process the Irving-Kirkwood (IK) vs Harasima (H) local pressure profiles
for the two-phase LJ slab and the long-cutoff convergence check.

Inputs (LAMMPS outputs from in.runA / in.rerunA / in.rerunB):
  ik_profile.dat   fix ave/time mode vector of compute stress/cartesian ... ke pair kspace
                   cols per row: z, dens, Pkxx,Pkyy,Pkzz, Pcxx,Pcyy,Pczz   (true pressure, +virial/V)
  har_profile.dat  fix ave/chunk norm none of compute stress/atom ... ke pair kspace
                   cols: chunk, z, Ncount, Sxx,Syy,Szz   (stress*vol = -virial; P=-S/Vchunk)
  press_global.dat fix ave/time of v_pxx v_pyy v_pzz v_gamma (global thermo cross-check)
  ik_rerunA.dat    rerun, stress/cartesian ke pair kspace  (short cutoff + dispersion kspace)
  ik_longcutB.dat  rerun, stress/cartesian ke pair         (plain lj/cut 8.0, no kspace)

Outputs: PNG plots + results.txt summary.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DZ = 0.1
AREA = 100.0            # 10 x 10 lateral
VCHUNK = AREA * DZ      # 10.0
LZ = 36.0


# ----------------------------------------------------------------------
def parse_ave_time_vector(fname):
    """fix ave/time mode vector: returns list of (timestep, array[nrows, ncols-1]).
    Each block: '<step> <nrows>' then nrows lines 'rowidx v1 v2 ...'. Drops rowidx."""
    blocks = []
    with open(fname) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 2 and parts[1].isdigit() and float(parts[0]).is_integer():
            step = int(parts[0]); nrows = int(parts[1]); i += 1
            rows = []
            for _ in range(nrows):
                vals = [float(x) for x in lines[i].split()]
                rows.append(vals[1:])   # drop leading row index
                i += 1
            blocks.append((step, np.array(rows)))
        else:
            i += 1
    return blocks


def parse_ave_chunk(fname):
    """fix ave/chunk: returns list of (timestep, array[nchunks, ncols-1]).
    Each block: '<step> <nchunks> <totalcount>' then nchunks lines
    'chunk coord Ncount v1 v2 ...'. Drops chunk index, keeps coord onward."""
    blocks = []
    with open(fname) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 3 and all(p.lstrip("-").replace(".", "").isdigit() for p in parts):
            step = int(parts[0]); nch = int(parts[1]); i += 1
            rows = []
            for _ in range(nch):
                vals = [float(x) for x in lines[i].split()]
                rows.append(vals[1:])   # drop chunk index -> [coord, Ncount, v1..]
                i += 1
            blocks.append((step, np.array(rows)))
        else:
            i += 1
    return blocks


def periodic_shift_bins(dens):
    """integer bin shift to move the periodic density COM to the box centre."""
    n = len(dens)
    th = 2 * np.pi * (np.arange(n) + 0.5) / n
    xb = np.sum(dens * np.cos(th)); yb = np.sum(dens * np.sin(th))
    ang = np.arctan2(yb, xb) % (2 * np.pi)
    zc = ang / (2 * np.pi) * n           # COM bin
    return int(round(n / 2.0 - zc))


# ----------------------------------------------------------------------
def load_IK(fname, recenter=True):
    """returns dict with z, dens, PN(z), PT(z) mean and standard error over blocks."""
    blocks = parse_ave_time_vector(fname)
    arrs = [b[1] for b in blocks]
    z = arrs[0][:, 0].copy()
    PN, PT, DN = [], [], []
    for a in arrs:
        dens = a[:, 1]
        PN_b = a[:, 4] + a[:, 7]                          # Pkzz + Pczz
        PT_b = 0.5 * ((a[:, 2] + a[:, 5]) + (a[:, 3] + a[:, 6]))  # 0.5(Pxx+Pyy)
        if recenter:
            s = periodic_shift_bins(dens)
            dens = np.roll(dens, s); PN_b = np.roll(PN_b, s); PT_b = np.roll(PT_b, s)
        DN.append(dens); PN.append(PN_b); PT.append(PT_b)
    return _stats(z, DN, PN, PT, len(arrs))


def load_H(fname, recenter=True):
    blocks = parse_ave_chunk(fname)
    arrs = [b[1] for b in blocks]
    z = arrs[0][:, 0].copy()
    PN, PT, DN = [], [], []
    for a in arrs:
        ncount = a[:, 1]
        Sxx, Syy, Szz = a[:, 2], a[:, 3], a[:, 4]
        dens = ncount / VCHUNK
        PN_b = -Szz / VCHUNK                              # P = -stress/Vchunk
        PT_b = -0.5 * (Sxx + Syy) / VCHUNK
        if recenter:
            s = periodic_shift_bins(dens)
            dens = np.roll(dens, s); PN_b = np.roll(PN_b, s); PT_b = np.roll(PT_b, s)
        DN.append(dens); PN.append(PN_b); PT.append(PT_b)
    return _stats(z, DN, PN, PT, len(arrs))


def _stats(z, DN, PN, PT, nb):
    DN = np.array(DN); PN = np.array(PN); PT = np.array(PT)
    se = lambda X: (X.std(0, ddof=1) / np.sqrt(nb)) if nb > 1 else np.zeros(X.shape[1])
    d = dict(z=z, nblocks=nb,
             dens=DN.mean(0), dens_se=se(DN),
             PN=PN.mean(0), PN_se=se(PN),
             PT=PT.mean(0), PT_se=se(PT))
    d["g"] = d["PN"] - d["PT"]
    d["gamma_cum"] = DZ * np.cumsum(d["g"])
    d["gamma_total"] = 0.5 * DZ * np.sum(d["g"])
    return d


def load_rerun_last(fname):
    """last block (final running average) of a rerun stress/cartesian ave/time file."""
    blocks = parse_ave_time_vector(fname)
    a = blocks[-1][1]
    z = a[:, 0].copy()
    dens = a[:, 1]
    PN = a[:, 4] + a[:, 7]
    PT = 0.5 * ((a[:, 2] + a[:, 5]) + (a[:, 3] + a[:, 6]))
    s = periodic_shift_bins(dens)
    z2 = z
    dens = np.roll(dens, s); PN = np.roll(PN, s); PT = np.roll(PT, s)
    g = PN - PT
    return dict(z=z2, dens=dens, PN=PN, PT=PT, g=g,
                gamma_cum=DZ * np.cumsum(g), gamma_total=0.5 * DZ * np.sum(g))


def pn_flatness(d):
    """constant-fit chi2/ndf of PN(z), weighted by per-bin SE (bulk+interface)."""
    PN, se = d["PN"], d["PN_se"]
    w = se > 0
    if w.sum() < 2:
        return float("nan"), PN.mean()
    pbar = np.average(PN[w], weights=1.0 / se[w] ** 2)
    chi2 = np.sum(((PN[w] - pbar) / se[w]) ** 2) / (w.sum() - 1)
    return chi2, pbar


# ----------------------------------------------------------------------
def main():
    out = []
    def log(s=""):
        print(s); out.append(s)

    ik = load_IK("ik_profile.dat")
    h = load_H("har_profile.dat")
    z = ik["z"]

    # global pressure / gamma cross-check
    pg = None
    if os.path.exists("press_global.dat"):
        g = np.loadtxt("press_global.dat", comments="#")
        if g.ndim == 1: g = g[None, :]
        pg = dict(pxx=g[:, 1].mean(), pyy=g[:, 2].mean(),
                  pzz=g[:, 3].mean(), gamma=g[:, 4].mean(),
                  gamma_se=g[:, 4].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0)

    chi2_ik, pn_ik = pn_flatness(ik)
    chi2_h, pn_h = pn_flatness(h)

    log("=" * 70)
    log("IK vs Harasima local pressure verification  (LJ liquid-vapor slab, T=0.85)")
    log("=" * 70)
    log(f"blocks: IK={ik['nblocks']}  H={h['nblocks']}   dz={DZ}  nbins={len(z)}")
    log("")
    log(f"PN flatness (constant-fit chi2/ndf):  IK={chi2_ik:.2f}   H={chi2_h:.2f}")
    log(f"mean normal pressure  PN_bar:         IK={pn_ik:+.4f}   H={pn_h:+.4f}")
    log("")
    log(f"surface tension gamma_total (0.5*dz*sum(PN-PT)):")
    log(f"   IK contour        = {ik['gamma_total']:+.4f}")
    log(f"   Harasima contour  = {h['gamma_total']:+.4f}")
    if pg:
        log(f"   thermo 0.5*Lz*(Pzz-0.5(Pxx+Pyy)) = {pg['gamma']:+.4f} +/- {pg['gamma_se']:.4f}")
        log(f"   thermo <Pxx,Pyy,Pzz> = {pg['pxx']:+.4f} {pg['pyy']:+.4f} {pg['pzz']:+.4f}")
    log(f"   IK box-avg <Pzz>={np.mean(ik['PN']):+.4f}  <PT>={np.mean(ik['PT']):+.4f}")
    log(f"    H box-avg <Pzz>={np.mean(h['PN']):+.4f}  <PT>={np.mean(h['PT']):+.4f}")

    # long-cutoff convergence
    A = B = None
    if os.path.exists("ik_rerunA.dat") and os.path.exists("ik_longcutB.dat"):
        A = load_rerun_last("ik_rerunA.dat")
        B = load_rerun_last("ik_longcutB.dat")
        log("")
        log("Long-cutoff convergence (frame-identical rerun):")
        log(f"   A short+kspace : gamma={A['gamma_total']:+.4f}  <PN>={np.mean(A['PN']):+.4f}")
        log(f"   B lj/cut 8.0   : gamma={B['gamma_total']:+.4f}  <PN>={np.mean(B['PN']):+.4f}")
        dPT = A["PT"] - B["PT"]
        log(f"   max|PT_A-PT_B| = {np.max(np.abs(dPT)):.4f}   rms = {np.sqrt(np.mean(dPT**2)):.4f}")

    with open("results.txt", "w") as f:
        f.write("\n".join(out) + "\n")

    # ---------------- plots ----------------
    # (a) density
    plt.figure(figsize=(7, 4))
    plt.plot(z, ik["dens"], label="density (IK compute)")
    plt.plot(z, h["dens"], "--", label="density (Harasima count)")
    plt.xlabel("z"); plt.ylabel(r"$\rho(z)$"); plt.legend(); plt.title("Density profile")
    plt.tight_layout(); plt.savefig("fig_density.png", dpi=130); plt.close()

    # (b) PN
    plt.figure(figsize=(7, 4))
    plt.errorbar(z, ik["PN"], yerr=ik["PN_se"], fmt="-", ms=2, lw=1, label="IK  $P_N=P_{zz}$", alpha=0.8)
    plt.errorbar(z, h["PN"], yerr=h["PN_se"], fmt="--", ms=2, lw=1, label="Harasima $P_N$", alpha=0.8)
    plt.axhline(pn_ik, color="k", ls=":", lw=0.8, label=f"$\\bar P_N$={pn_ik:.3f}")
    plt.xlabel("z"); plt.ylabel(r"$P_N(z)$")
    plt.title(f"Normal pressure (must be flat) — chi2/ndf IK={chi2_ik:.2f}")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_PN.png", dpi=130); plt.close()

    # (c) PT
    plt.figure(figsize=(7, 4))
    plt.plot(z, ik["PT"], "-", lw=1, label="IK  $P_T$")
    plt.plot(z, h["PT"], "--", lw=1, label="Harasima $P_T$")
    plt.plot(z, ik["PN"], "k:", lw=0.8, label="IK $P_N$ (ref)")
    plt.xlabel("z"); plt.ylabel(r"$P_T(z)=\frac{1}{2}(P_{xx}+P_{yy})$")
    plt.title("Tangential pressure — bulk matches $P_N$, interface wells differ by contour")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_PT.png", dpi=130); plt.close()

    # (d) gamma_cum
    plt.figure(figsize=(7, 4))
    plt.plot(z, ik["gamma_cum"], "-", lw=1.2, label=f"IK  $\\gamma_{{tot}}$={ik['gamma_total']:.3f}")
    plt.plot(z, h["gamma_cum"], "--", lw=1.2, label=f"H   $\\gamma_{{tot}}$={h['gamma_total']:.3f}")
    plt.xlabel("z"); plt.ylabel(r"$\gamma_{cum}(z)=\int (P_N-P_T)\,dz'$")
    plt.title("Cumulative surface tension (two steps = two interfaces)")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_gamma.png", dpi=130); plt.close()

    # (e) long-cutoff overlay
    if A and B:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(A["z"], A["PN"], "-", lw=1, label="A short+kspace")
        ax[0].plot(B["z"], B["PN"], "--", lw=1, label="B lj/cut 8.0")
        ax[0].set_title("$P_N(z)$: short+kspace vs long cutoff"); ax[0].set_xlabel("z"); ax[0].legend()
        ax[1].plot(A["z"], A["PT"], "-", lw=1, label="A short+kspace")
        ax[1].plot(B["z"], B["PT"], "--", lw=1, label="B lj/cut 8.0")
        ax[1].set_title("$P_T(z)$: short+kspace vs long cutoff"); ax[1].set_xlabel("z"); ax[1].legend()
        plt.tight_layout(); plt.savefig("fig_longcut.png", dpi=130); plt.close()

    log("")
    log("wrote: fig_density.png fig_PN.png fig_PT.png fig_gamma.png " +
        ("fig_longcut.png" if (A and B) else "") + "  results.txt")


if __name__ == "__main__":
    main()
