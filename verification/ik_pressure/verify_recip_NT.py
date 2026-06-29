#!/usr/bin/env python3
"""
Close the last IK gap: verify P_N(z) and P_T(z) SEPARATELY (not just P_N-P_T).

Every other test checked the difference P_N-P_T (the off-diagonal combination
CN-CT) or the pinned means.  A compensating zero-mean error common to the CN and
CT shapes would preserve P_N-P_T and the means and slip through.  Here we deposit
the normal and tangential pair virials SEPARATELY along the IK bond and compare
each mean-removed shape to the LAMMPS reciprocal-only output (cpp2_recip.dat,
shell subtraction disabled), via the Ewald identity (same switched potential).

  P_N pair virial:  -(du/dr)/r * rz^2
  P_T pair virial:  -(du/dr)/r * 0.5*(rx^2+ry^2) = -(du/dr)/r * 0.5*(r^2 - rz^2)
  (their difference is the validated W = (du/dr)/(2r)(r^2-3 rz^2))
"""
import numpy as np
from scipy.spatial import cKDTree
import verify_pressure as V
import verify_cpp2 as C

LX = LY = C.LX; LZ = C.LZ; AREA = C.AREA; NB = C.NB; DZ = C.DZ
RLO, DEL, RC = 3.4, 0.6, 4.0
OFF = NB; EXT = NB + 2*OFF
SHIFTS = np.array([[a*LX, b*LY, c*LZ]
                   for a in (-2,-1,0,1,2) for b in (-2,-1,0,1,2) for c in (-2,-1,0,1,2)])

def dudr_sw(r):
    t = np.clip((r-RLO)/DEL, 0, 1)
    S = t**4*(35-84*t+70*t**2-20*t**3)
    Sp = np.where((t > 0) & (t < 1), 140*(t*(1-t))**3, 0.0)/DEL
    return np.where(r >= RLO, Sp*(-4.0/r**6) + S*(24.0/r**7), 0.0)

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

def rs_NT(frames, RMAX):
    accN = np.zeros(NB); accT = np.zeros(NB)
    for xs in frames:
        img = (xs[None, :, :]+SHIFTS[:, None, :]).reshape(-1, 3)
        sdm = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
        r = sdm.data; row = sdm.row; col = sdm.col
        keep = r >= RLO
        r = r[keep]; row = row[keep]; col = col[keep]
        zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
        f = dudr_sw(r)/r
        WN = -f*rz*rz                       # normal (zz) pair virial
        WT = -f*0.5*(r*r - rz*rz)           # tangential 1/2(xx+yy)
        L = np.maximum(np.abs(rz), 1e-12)
        zlo = np.minimum(zi, zj); zhi = np.maximum(zi, zj)
        accN += 0.5*deposit(zlo, zhi, WN/(AREA*L))
        accT += 0.5*deposit(zlo, zhi, WT/(AREA*L))
    return accN/len(frames), accT/len(frames)


def main():
    frames = C.read_dump("traj_cpp2.dump")[::4]
    ab = V.parse_ave_time_vector("cpp2_recip.dat")[-1][1]
    gN_lat = ab[:, 7]; gT_lat = 0.5*(ab[:, 5]+ab[:, 6])
    sm = lambda a: V.fourier_smooth(a, 30); dm = lambda a: a-a.mean()
    rms = lambda a, b: np.sqrt(np.mean((sm(dm(a))-sm(dm(b)))**2))
    print("Separate P_N(z), P_T(z) IK shape: brute-force (switched) vs LAMMPS reciprocal-only")
    print("  RMAX   ratio_N   rms_N      ratio_T   rms_T      ratio_(N-T) rms_(N-T)")
    profsN = {}; profsT = {}
    for RMAX in (11.0, 14.0, 17.0):
        gN, gT = rs_NT(frames, RMAX)
        rN = np.ptp(sm(dm(gN)))/np.ptp(sm(dm(gN_lat)))
        rT = np.ptp(sm(dm(gT)))/np.ptp(sm(dm(gT_lat)))
        gNTb = gN-gT; gNTl = gN_lat-gT_lat
        rNT = np.ptp(sm(dm(gNTb)))/np.ptp(sm(dm(gNTl)))
        profsN[RMAX] = sm(dm(gN)); profsT[RMAX] = sm(dm(gT))
        print("  %4.1f   %.4f    %.5f    %.4f    %.5f    %.4f      %.5f" % (
            RMAX, rN, rms(gN, gN_lat), rT, rms(gT, gT_lat), rNT, rms(gNTb, gNTl)))
    print("  => ratio_N, ratio_T -> 1 and rms_N, rms_T -> 0 confirms P_N(z) and P_T(z)")
    print("     are INDIVIDUALLY correct (CN and CT shapes, not only their difference).")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = ab[:, 0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(z, sm(dm(gN_lat)), "-", color="black", lw=2.4, label="LAMMPS reciprocal P_N")
    ax[0].plot(z, profsN[17.0], "--", color="tab:red", lw=1.4, label="brute-force P_N, RMAX=17")
    ax[0].set_title(r"normal $P_N(z)$ (mean removed)"); ax[0].set_xlabel("z*"); ax[0].set_xlim(0, LZ)
    ax[0].set_ylabel(r"$P_N$"); ax[0].legend(fontsize=8)
    ax[1].plot(z, sm(dm(gT_lat)), "-", color="black", lw=2.4, label="LAMMPS reciprocal P_T")
    ax[1].plot(z, profsT[17.0], "--", color="tab:blue", lw=1.4, label="brute-force P_T, RMAX=17")
    ax[1].set_title(r"tangential $P_T(z)$ (mean removed)"); ax[1].set_xlabel("z*"); ax[1].set_xlim(0, LZ)
    ax[1].set_ylabel(r"$P_T$"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("fig_recip_NT.png", dpi=130); plt.close()
    print("  wrote fig_recip_NT.png")


if __name__ == "__main__":
    main()
