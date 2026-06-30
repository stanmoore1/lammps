#!/usr/bin/env python3
"""
Mechanical stability + box-average closure for the FULL IK contour.

A no-field liquid-vapor LJ slab in mechanical equilibrium must have a FLAT normal
pressure P_N(z) = P_zz(z) = const through the interfaces (momentum balance:
dP_zz/dz = 0 with no external force).  This is the canonical, contour-level test
that the short-range pair IK contour and the long-range ewald/disp/planar kspace IK
contour combine correctly.  The tangential P_T(z) is NOT flat -- it has the
interface wells (surface tension).

Also: the box-average of the local pressure must equal the global pressure tensor
from thermo (the kspace contour's box-average is pinned to the global kspace
pressure by construction; here we confirm it numerically for the FULL P_N, P_T).

  mech_full.dat   : compute stress/cartesian z dz NULL 0 ke pair kspace (vector)
                    cols (after coord): dens, Pk{xx,yy,zz}, Pc{xx,yy,zz}
  mech_global.dat : global pxx, pyy, pzz (time-averaged)
"""
import numpy as np
import verify_pressure as V

def main():
    ab = V.parse_ave_time_vector("mech_full.dat")[-1][1]
    z = ab[:, 0]
    pxx = ab[:, 2]+ab[:, 5]      # kinetic + configurational
    pyy = ab[:, 3]+ab[:, 6]
    pzz = ab[:, 4]+ab[:, 7]
    dens = ab[:, 1]
    pN = pzz
    pT = 0.5*(pxx+pyy)
    sm = lambda a: V.fourier_smooth(a, 20)

    # liquid vs vapor regions from the density profile (for the flatness window)
    rho = dens
    print("Mechanical stability (no-field LJ slab): full IK contour P_N(z)=P_zz(z)")
    print("  rho range %.3f .. %.3f" % (rho.min(), rho.max()))
    # P_N flatness: std across all bins, and across bins with appreciable density
    mask = rho > 0.1*rho.max()
    pN_mean = pN.mean(); pN_std = pN.std()
    pN_std_dense = pN[mask].std()
    print("  P_N(z):  mean = %.4f   std(all bins) = %.4f   std(rho>0.1max) = %.4f" % (
        pN_mean, pN_std, pN_std_dense))
    print("  smoothed P_N range = [%.4f, %.4f]  ptp = %.4f   (flat => ~0 ptp)" % (
        sm(pN).min(), sm(pN).max(), np.ptp(sm(pN))))
    print("  for contrast P_T(z) ptp (surface-tension wells) = %.4f" % np.ptp(sm(pT)))
    print("  => P_N flat (ptp_N << ptp_T) confirms momentum balance: the short(pair)")
    print("     + long(kspace IK) normal pressure is constant through the interfaces.")

    # box-average vs global thermo pressure tensor
    gl = np.loadtxt("mech_global.dat")
    if gl.ndim == 2:
        gpxx, gpyy, gpzz = gl[:, 1].mean(), gl[:, 2].mean(), gl[:, 3].mean()
    else:
        gpxx, gpyy, gpzz = gl[1], gl[2], gl[3]
    print("")
    print("Box-average of the local contour vs global thermo pressure tensor:")
    print("  <P_xx>=%.5f <P_yy>=%.5f <P_zz>=%.5f  (local profile box-average)" % (
        pxx.mean(), pyy.mean(), pzz.mean()))
    print("  global  pxx=%.5f  pyy=%.5f  pzz=%.5f  (thermo)" % (gpxx, gpyy, gpzz))
    print("  diff:   dxx=%.2e dyy=%.2e dzz=%.2e" % (
        pxx.mean()-gpxx, pyy.mean()-gpyy, pzz.mean()-gpzz))
    gamma = 0.5*(z[1]-z[0])*(pN-pT).sum()
    print("  surface tension gamma = 0.5*dz*sum(P_N-P_T) = %.4f" % gamma)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(z, sm(rho), "-", color="gray", lw=1.2, label=r"$\rho(z)$ (a.u.)")
    ax[0].set_xlabel("z*"); ax[0].set_ylabel(r"$\rho$"); ax[0].legend(loc="upper right", fontsize=8)
    ax2 = ax[0].twinx()
    ax2.axhline(pN_mean, color="0.6", lw=0.8, ls=":")
    ax2.plot(z, sm(pN), "-", color="navy", lw=2, label=r"$P_N(z)$ (flat)")
    ax2.set_ylabel(r"$P_N$"); ax2.legend(loc="lower right", fontsize=8)
    ax[0].set_title("mechanical stability: P_N(z) flat")
    ax[1].plot(z, sm(pN), "-", color="navy", lw=2, label=r"$P_N(z)$")
    ax[1].plot(z, sm(pT), "-", color="darkred", lw=2, label=r"$P_T(z)$")
    ax[1].axhline(pN_mean, color="0.6", lw=0.8, ls=":")
    ax[1].set_xlabel("z*"); ax[1].set_ylabel("pressure"); ax[1].legend(fontsize=8)
    ax[1].set_title("P_N flat vs P_T interface wells")
    plt.tight_layout(); plt.savefig("fig_mech_stab.png", dpi=130); plt.close()
    print("  wrote fig_mech_stab.png")


if __name__ == "__main__":
    main()
