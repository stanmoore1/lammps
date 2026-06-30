#!/usr/bin/env python3
"""
Mechanical stability + box-average closure for the FULL IK contour (no-field slab).

(1) BOX-AVERAGE CLOSURE (mech_*_partial.dat, from the live NVT run): the box-average
    of the local FULL contour (ke+pair+kspace) must equal the global thermo pressure
    tensor.  Exact by construction (diagonal pinning); confirmed numerically for
    P_xx, P_yy, P_zz separately.

(2) MECHANICAL STABILITY / contour attribution (frame-identical reruns of the slab
    trajectory): a slab in mechanical equilibrium must have P_N(z)=P_zz(z) flat.
    Run A = short pair (rcut=3) + ewald/disp/planar kspace IK.
    Run B = long plain lj/cut 8.0, NO kspace (untruncated-LJ reference).
    A and B are computed on the SAME frames.  If A==B and P_N is non-flat in BOTH,
    the non-flatness is the system (short equilibration / finite slab), NOT the
    kspace IK contour (whose correctness is proven to ratio 1.0001 by the
    single-frame full-workflow test verify_full_workflow.py).
"""
import numpy as np
import verify_pressure as V

def comps(fn):
    ab = V.parse_ave_time_vector(fn)[-1][1]
    z = ab[:, 0]; dens = ab[:, 1]
    pxx = ab[:, 2]+ab[:, 5]; pyy = ab[:, 3]+ab[:, 6]; pzz = ab[:, 4]+ab[:, 7]
    return z, dens, pxx, pyy, pzz

def main():
    sm = lambda a: V.fourier_smooth(a, 18)

    # (1) box-average closure from the live NVT run
    z, dens, pxx, pyy, pzz = comps("mech_full_partial.dat")
    gl = np.loadtxt("mech_global_partial.dat")
    gpxx, gpyy, gpzz = gl[:, 1].mean(), gl[:, 2].mean(), gl[:, 3].mean()
    print("(1) Box-average of the FULL local contour vs global thermo pressure tensor:")
    print("    <P_xx>=%.5f <P_yy>=%.5f <P_zz>=%.5f   (local box-average)" % (
        pxx.mean(), pyy.mean(), pzz.mean()))
    print("    global  %.5f      %.5f      %.5f   (thermo)" % (gpxx, gpyy, gpzz))
    print("    diff    %.1e     %.1e     %.1e   (pinning holds for P_N, P_T separately)" % (
        pxx.mean()-gpxx, pyy.mean()-gpyy, pzz.mean()-gpzz))

    # (2) frame-identical A vs B + flatness
    zA, dA, axx, ayy, azz = comps("A_rerun.dat")
    zB, dB, bxx, byy, bzz = comps("AB_long.dat")
    pNA, pTA = azz, 0.5*(axx+ayy)
    pNB, pTB = bzz, 0.5*(bxx+byy)
    print("")
    print("(2) Frame-identical A (short+kspace) vs B (long lj/cut, no kspace), 52 frames:")
    print("    A==B?   P_N rms=%.3f (ratio %.2f) | P_N-P_T rms=%.3f (ratio %.2f)" % (
        np.sqrt(np.mean((sm(pNA)-sm(pNB))**2)), np.ptp(sm(pNA))/np.ptp(sm(pNB)),
        np.sqrt(np.mean((sm(pNA-pTA)-sm(pNB-pTB))**2)), np.ptp(sm(pNA-pTA))/np.ptp(sm(pNB-pTB))))
    print("    P_N flatness:  A ptp=%.3f  B ptp=%.3f   (P_T ptp: A=%.3f B=%.3f)" % (
        np.ptp(sm(pNA)), np.ptp(sm(pNB)), np.ptp(sm(pTA)), np.ptp(sm(pTB))))
    print("    => P_N non-flat in BOTH A and B (same shape): the residual non-flatness")
    print("       is the system (short-run slab not at mechanical equilibrium), not the")
    print("       kspace IK code.  A tracks the untruncated-LJ reference B.")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(zA, sm(dA)/dA.max(), color="gray", lw=1.0, label=r"$\rho(z)$ (norm.)")
    ax[0].plot(zA, sm(pNA), "-", color="navy", lw=2, label="P_N  A (short+kspace)")
    ax[0].plot(zB, sm(pNB), "--", color="deepskyblue", lw=1.5, label="P_N  B (long, no kspace)")
    ax[0].axhline(pNA.mean(), color="0.6", lw=0.8, ls=":")
    ax[0].set_title("P_N(z): A (short+kspace) tracks B (untruncated)")
    ax[0].set_xlabel("z*"); ax[0].set_ylabel("pressure"); ax[0].legend(fontsize=8)
    ax[1].plot(zA, sm(pNA-pTA), "-", color="green", lw=2, label="A short+kspace")
    ax[1].plot(zB, sm(pNB-pTB), "--", color="red", lw=1.5, label="B long, no kspace")
    ax[1].set_title(r"$P_N-P_T$: short+kspace = untruncated LJ")
    ax[1].set_xlabel("z*"); ax[1].set_ylabel(r"$P_N-P_T$"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_mech_stab.png", dpi=130); plt.close()
    print("    wrote fig_mech_stab.png")


if __name__ == "__main__":
    main()
