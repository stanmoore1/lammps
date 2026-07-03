#!/usr/bin/env python3
"""
Stronger version of verify_cosine_exact.py: ESSENTIALLY EXACT verification of the
ewald/disp/planar IK off-diagonal formula with a MULTI-MODE analytic density
(several Fourier terms, not just the fundamental cosine).

  rho(z) = rho0 + sum_k a_k cos(k k0 z + phi_k),   k0 = 2 pi / Lz, k = 1..M

A single cosine excites only S_n at n = 0, +-1, so the off-diagonal coefficient
C(h_n,h_m) = (pi/H)[J(h_n)+J(h_m)] is tested at just p = n+m = +-1, +-2.  An M-mode
density excites S_n for |n| <= M, so the lattice double sum exercises ALL (n,m)
pairs with |n|,|m| <= M -- p = n+m from -2M..2M and many distinct H = h_n+h_m
arguments -- a far broader test of the closed-form coefficient.

Both sides are exact representations of the same continuum integral (the density is
band-limited), so as the slab kernel cutoff rmax -> infinity the real-space slab
must equal the code-kernel lattice to quadrature precision.  Reuses the verified
verify_cosine_exact helpers (lattice = code ik_phi/ik_psi to r=inf; slab = 1-D
quadrature).
"""
import numpy as np
import verify_cpp2 as C
import verify_ik_kernel as K
import verify_cosine_exact as CE

PI = np.pi
NB, DZ, LZ = C.NB, C.DZ, C.LZ
k0 = 2*PI/LZ

# (k, amplitude, phase): four modes beyond the mean, phases break the symmetry so
# the test is not accidentally even.  Amplitudes keep rho(z) > 0 everywhere.
RHO0 = 0.5
MODES = [(1, 0.22, 0.0), (2, 0.11, 0.7), (3, 0.07, 1.9), (4, 0.04, 2.6)]
MMAX = max(k for k, _, _ in MODES)

def multimode_density():
    z = (np.arange(NB)+0.5)*DZ
    rho = np.full(NB, RHO0)
    for k, a, ph in MODES:
        rho += a*np.cos(k*k0*z + ph)
    return z, rho

def lattice_shape(rho):
    """code-kernel off-diagonal P_N-P_T(z); S_n exact for |n| <= MMAX."""
    ns, SS = K.Sn_meanfield(rho, MMAX)
    return K.offdiag_shape(ns, SS, K.code_coeff)

def H_analytic_multi(z):
    """analytic Harasima P_N-P_T (mean removed): (pi/2) rho(z) (G*rho)(z), with
    (G*rho)(z) = sum_k a_k Ghat(k k0) cos(k k0 z + phi_k) (Ghat(0)=0), Ghat to r=inf."""
    rho = np.full(NB, RHO0)
    Gconv = np.zeros(NB)
    for k, a, ph in MODES:
        rho += a*np.cos(k*k0*z + ph)
        Gconv += a*CE.Ghat_exact(k*k0)*np.cos(k*k0*z + ph)
    g = (PI/2.0)*rho*Gconv
    return g - g.mean()

def main():
    z, rho = multimode_density()
    dm = lambda a: a-a.mean()
    print("Multi-mode exact test: rho(z) = %.2f + " % RHO0 +
          " + ".join("%.2f cos(%d k0 z+%.1f)" % (a, k, ph) for k, a, ph in MODES))
    print("  rho range %.3f .. %.3f ;  structure factors S_n nonzero for |n| <= %d" % (
        rho.min(), rho.max(), MMAX))

    # ---- IK: code-kernel lattice (tail to r=inf) vs real-space slab (rmax sweep) ----
    g_lat = lattice_shape(rho)
    print("")
    print("[IK] lattice (code ik_phi/ik_psi, r=inf) vs slab (real-space line integral):")
    print("  rmax   ptp_slab   ratio slab/lattice   shape_rms")
    for rmax in (14.0, 20.0, 40.0, 80.0, 160.0):
        g_s = CE.slab_shape(rho, rmax)
        ratio = np.ptp(dm(g_s))/np.ptp(dm(g_lat))
        rms = np.sqrt(np.mean((dm(g_s)-dm(g_lat))**2))
        print("  %5.0f  %.6f    %.5f              %.6f" % (rmax, np.ptp(dm(g_s)), ratio, rms))
    print("  ptp(lattice) = %.6f" % np.ptp(dm(g_lat)))
    print("  => ratio -> 1, rms -> 0: off-diagonal IK coefficient correct across all")
    print("     (n,m) pairs with |n|,|m| <= %d (p = n+m up to %d)." % (MMAX, 2*MMAX))

    # ---- H: real-space slab vs analytic Fourier-Harasima (r=inf) ----
    g_H_ref = H_analytic_multi(z)
    print("")
    print("[H] Harasima slab (Eq 4.18 H) vs analytic Fourier-Harasima (Ghat to r=inf):")
    print("  rmax   ptp_slabH   ratio slabH/analytic   shape_rms")
    for rmax in (14.0, 20.0, 40.0, 80.0, 160.0):
        g_h = dm(CE.slab_H_shape(rho, rmax))
        ratio = np.ptp(g_h)/np.ptp(g_H_ref)
        rms = np.sqrt(np.mean((g_h-g_H_ref)**2))
        print("  %5.0f  %.6f    %.5f               %.6f" % (rmax, np.ptp(g_h), ratio, rms))
    print("  ptp(analytic H) = %.6f" % np.ptp(g_H_ref))

    # ---- plot ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g_s = CE.slab_shape(rho, 160.0)
    g_h = dm(CE.slab_H_shape(rho, 160.0))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(z, rho, "-", color="tab:green", lw=1.5)
    ax[0].set_title("multi-mode density rho(z)"); ax[0].set_xlabel("z*"); ax[0].set_ylabel(r"$\rho$")
    ax[0].set_xlim(0, LZ)
    ax[1].plot(z, dm(g_lat), "-", color="black", lw=3, label="lattice (code kernels, r=inf)")
    ax[1].plot(z, dm(g_s), "--", color="tab:red", lw=1.4, label="slab IK, rmax=160")
    ax[1].set_title(r"IK $P_N-P_T$: lattice = slab"); ax[1].set_xlabel("z*"); ax[1].set_xlim(0, LZ)
    ax[1].set_ylabel(r"$P_N-P_T$ (mean removed)"); ax[1].legend(fontsize=8)
    ax[2].plot(z, g_H_ref, "-", color="black", lw=3, label="analytic Harasima (r=inf)")
    ax[2].plot(z, g_h, "--", color="tab:blue", lw=1.4, label="slab H, rmax=160")
    ax[2].set_title(r"H $P_N-P_T$: slab = analytic"); ax[2].set_xlabel("z*"); ax[2].set_xlim(0, LZ)
    ax[2].set_ylabel(r"$P_N-P_T$ (mean removed)"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_multimode_exact.png", dpi=130); plt.close()
    print("  wrote fig_multimode_exact.png")


if __name__ == "__main__":
    main()
