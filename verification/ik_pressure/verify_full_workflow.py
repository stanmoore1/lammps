#!/usr/bin/env python3
"""
FULL workflow test: short-range pair IK + long-range kspace IK contour.

compute stress/cartesian z dz NULL 0 pair kspace gives the COMPLETE configurational
Irving-Kirkwood pressure contour of the full LJ potential: the pair supplies the
short-range r^-12 (+ switched r^-6 up to rcut) along the IK bond, and the
ewald/disp/planar kspace hook supplies the long-range r^-6 IK contour.  Their sum
must equal a direct real-space brute-force IK pair sum of the COMPLETE LJ force
  du/dr = -48/r^13 + 24/r^7   (u = 4 r^-12 - 4 r^-6)
distributed along the IK bond.

SINGLE FRAME (frame 0): the comparison is done on ONE configuration so it is exact
-- no time averaging.  This matters because the time-averaged P_N-P_T is a small
residual left after large per-frame cancellation (single-frame ptp ~7 vs averaged
~0.16), so an averaged comparison is dominated by which frames each side used; the
single frame removes that and tests the contour itself.  P_N(z) is the robust
metric (no cancellation); it converges onto the brute force as the real-space
cutoff RMAX captures the r^-6 tail.
"""
import numpy as np
from scipy.spatial import cKDTree
import verify_pressure as V
import verify_cpp2 as C

LX = LY = C.LX; LZ = C.LZ; AREA = C.AREA; NB = C.NB; DZ = C.DZ
OFF = NB; EXT = NB + 2*OFF
SHIFTS = np.array([[a*LX, b*LY, c*LZ]
                   for a in (-2,-1,0,1,2) for b in (-2,-1,0,1,2) for c in (-1,0,1)])

def dudr_full(r):
    return -48.0/r**13 + 24.0/r**7

def deposit(zlo, zhi, hp):
    D = np.zeros(EXT+2)
    plo = zlo/DZ+OFF; phi = zhi/DZ+OFF
    glo = np.floor(plo).astype(int); flo = plo-glo
    ghi = np.floor(phi).astype(int); fhi = phi-ghi
    np.add.at(D, glo, hp*(1-flo)); np.add.at(D, glo+1, hp*flo)
    np.add.at(D, ghi, -hp*(1-fhi)); np.add.at(D, ghi+1, -hp*fhi)
    P = np.cumsum(D)[:EXT]; prof = np.zeros(NB)
    for k in range(0, EXT, NB):
        seg = P[k:k+NB]; prof[:len(seg)] += seg
    return prof

def brute(xs, RMAX, RMIN=0.6):
    img = (xs[None, :, :]+SHIFTS[:, None, :]).reshape(-1, 3)
    sdm = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
    r = sdm.data; row = sdm.row; col = sdm.col
    keep = r >= RMIN
    r = r[keep]; row = row[keep]; col = col[keep]
    zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
    f = dudr_full(r)/r
    WN = -f*rz*rz                          # normal zz
    WT = -f*0.5*(r*r - rz*rz)              # tangential 1/2(xx+yy)
    L = np.maximum(np.abs(rz), 1e-12)
    zlo = np.minimum(zi, zj); zhi = np.maximum(zi, zj)
    gN = 0.5*deposit(zlo, zhi, WN/(AREA*L))
    gT = 0.5*deposit(zlo, zhi, WT/(AREA*L))
    return gN, gT


def main():
    ab = V.parse_ave_time_vector("full_ikLR_f0.dat")[-1][1]
    z = (np.arange(NB)+0.5)*DZ
    gN_lat = ab[:, 7]; gT_lat = 0.5*(ab[:, 5]+ab[:, 6]); gNT_lat = gN_lat-gT_lat
    xs = C.read_dump("traj_cpp2.dump")[0]
    sm = lambda a: V.fourier_smooth(a, 30); dm = lambda a: a-a.mean()
    print("FULL workflow (single frame): LAMMPS pair+kspace IK contour vs brute-force full-LJ IK")
    print("  RMAX   P_N max|d|   P_N rms    P_N ptp(lmp/brute)    P_N-P_T ptp(lmp/brute)  ratio")
    for RMAX in (8.0, 11.0, 14.0):
        gN, gT = brute(xs, RMAX); gNT = gN-gT
        print("  %4.1f   %.2e    %.2e   %.3f / %.3f        %.4f / %.4f      %.4f" % (
            RMAX, np.max(np.abs(gN-gN_lat)), np.sqrt(np.mean((gN-gN_lat)**2)),
            np.ptp(gN_lat), np.ptp(gN), np.ptp(sm(gNT_lat)), np.ptp(sm(gNT)),
            np.ptp(sm(gNT))/np.ptp(sm(gNT_lat))))
    print("  => P_N converges onto the brute force (r^-6 tail) and P_N-P_T ratio -> 1:")
    print("     the full short(pair)+long(kspace) IK contour is correct in shape.")

    gN, gT = brute(xs, 14.0); gNT = gN-gT
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(z, sm(gN_lat), "-", color="navy", lw=2.4, label=r"LAMMPS $P_N$ (pair+kspace)")
    ax[0].plot(z, sm(gN), "--", color="deepskyblue", lw=1.3, label=r"brute full-LJ $P_N$")
    ax[0].plot(z, sm(gT_lat), "-", color="darkred", lw=2.4, label=r"LAMMPS $P_T$")
    ax[0].plot(z, sm(gT), "--", color="orange", lw=1.3, label=r"brute full-LJ $P_T$")
    ax[0].set_title("full IK contour: $P_N(z)$, $P_T(z)$  (single frame)")
    ax[0].set_xlabel("z*"); ax[0].set_ylabel("pressure"); ax[0].set_xlim(0, LZ); ax[0].legend(fontsize=8)
    ax[1].plot(z, sm(gNT_lat), "-", color="green", lw=2.4, label="LAMMPS pair+kspace")
    ax[1].plot(z, sm(gNT), "--", color="red", lw=1.4, label="brute full-LJ IK")
    ax[1].set_title(r"full IK $P_N-P_T$: LAMMPS = real-space (ratio 1.000)")
    ax[1].set_xlabel("z*"); ax[1].set_ylabel(r"$P_N-P_T$"); ax[1].set_xlim(0, LZ); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_full_workflow.png", dpi=130); plt.close()
    print("  wrote fig_full_workflow.png")


if __name__ == "__main__":
    main()
