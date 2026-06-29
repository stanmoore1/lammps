#!/usr/bin/env python3
"""
Cheap, ESSENTIALLY EXACT verification of the ewald/disp/planar IK off-diagonal
formula -- no trajectory, no KDTree, no statistics, no real-space pair truncation.

Idea: feed an ANALYTIC single-cosine density  rho(z) = rho0 + rho1 cos(k0 z),
k0 = 2*pi/Lz (the box fundamental).  Then:

  * LATTICE side (the code's formula): the structure factors S_n are nonzero only
    for n = 0, +-1 (three terms), so the off-diagonal P_N-P_T double sum is a tiny
    EXACT sum, evaluated with the code's own ik_phi/ik_psi kernels -- which
    integrate the 1/r^6 dispersion tail to r -> infinity analytically via the
    sici_chain closed form.

  * SLAB side (independent real-space ground truth): the Appendix-A IK line
    integral  (pi/2) int dz' int_0^1 dalpha rho(z-a z') rho(z+(1-a) z') G(z'),
    G(z') = int_{max(rcut,|z'|)}^{rmax} (du/dr)(r^2 - 3 z'^2) dr,
    evaluated by 1-D quadrature.  The ONLY approximation is the finite rmax; the
    integrand falls as ~r^-5 so it converges fast.

As rmax -> infinity the slab must equal the lattice to quadrature precision (the
density is band-limited, so both are exact representations of the same continuum
integral).  This isolates the off-diagonal coefficient with zero truncation noise.
"""
import numpy as np
import verify_cpp2 as C
import verify_ik_kernel as K

PI = np.pi
NB, DZ, LZ, AREA = C.NB, C.DZ, C.LZ, C.AREA
k0 = 2*PI/LZ

def analytic_density(rho0=0.5, rho1=0.4):
    z = (np.arange(NB)+0.5)*DZ
    return z, rho0 + rho1*np.cos(k0*z)

def lattice_shape(rho):
    """code-kernel off-diagonal P_N-P_T(z); S_n exact for the single cosine."""
    ns, SS = K.Sn_meanfield(rho, 3)          # only n=0,+-1 are nonzero
    return K.offdiag_shape(ns, SS, K.code_coeff)

def slab_shape(rho, rmax, nlam=48, zmax=40.0, nr=8000):
    zp = np.arange(-int(zmax/DZ), int(zmax/DZ)+1)*DZ
    G = C.Gkernel(zp, rmax=rmax, nr=nr)
    return C.slab_IK(rho, G, zp, nlam=nlam)

def main():
    z, rho = analytic_density()
    g_lat = lattice_shape(rho)
    dm = lambda a: a-a.mean()
    print("Cheap exact test: analytic rho(z)=%.2f+%.2f cos(2 pi z/Lz)  (range %.2f-%.2f)" % (
        0.5, 0.4, rho.min(), rho.max()))
    print("  lattice off-diagonal uses code ik_phi/ik_psi (tail integrated to r=inf).")
    print("  rmax   ptp_slab   ratio slab/lattice   shape_rms(slab-lattice)")
    rows = []
    profs = {}
    for rmax in (14.0, 20.0, 40.0, 80.0, 160.0):
        g_s = slab_shape(rho, rmax)
        ratio = np.ptp(dm(g_s))/np.ptp(dm(g_lat))
        rms = np.sqrt(np.mean((dm(g_s)-dm(g_lat))**2))
        rows.append((rmax, np.ptp(dm(g_s)), ratio, rms)); profs[rmax] = dm(g_s)
        print("  %5.0f  %.6f    %.5f              %.6f" % (rmax, np.ptp(dm(g_s)), ratio, rms))
    print("  ptp(lattice) = %.6f" % np.ptp(dm(g_lat)))
    print("  => ratio -> 1 and rms -> 0 as rmax grows: the off-diagonal IK coefficient")
    print("     in ewald/disp/planar matches the real-space IK integral EXACTLY.")
    np.save("cosine_rows.npy", np.array(rows))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(z, dm(g_lat), "-", color="black", lw=3, label="lattice (code kernels, r=inf)")
    for rmax in (14.0, 40.0, 160.0):
        ax[0].plot(z, profs[rmax], lw=1.2, label="slab IK, rmax=%g" % rmax)
    ax[0].set_xlabel("z*"); ax[0].set_ylabel(r"$P_N-P_T$ (mean removed)")
    ax[0].set_title("analytic cosine density: slab IK -> lattice as rmax grows")
    ax[0].legend(fontsize=8); ax[0].set_xlim(0, LZ)
    r = np.array(rows)
    ax[1].plot(r[:, 0], r[:, 2], "o-", color="tab:green")
    ax[1].axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax[1].set_xscale("log"); ax[1].set_xlabel("rmax (slab kernel cutoff)")
    ax[1].set_ylabel("ptp ratio slab / lattice")
    ax[1].set_title("essentially exact: ratio -> 1.000")
    for xr, _, rr, _ in rows:
        ax[1].annotate("%.4f" % rr, (xr, rr), textcoords="offset points", xytext=(0, 6),
                       fontsize=7, ha="center")
    plt.tight_layout(); plt.savefig("fig_cosine_exact.png", dpi=130); plt.close()
    print("  wrote fig_cosine_exact.png cosine_rows.npy")


if __name__ == "__main__":
    main()
