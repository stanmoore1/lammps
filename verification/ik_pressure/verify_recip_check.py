#!/usr/bin/env python3
"""
Decisive Ewald-identity test of the ewald/disp/planar IK RECIPROCAL formula.

The reciprocal sum is, by the Ewald identity, exactly equal to the real-space sum
of the SAME (switched) potential S(r)*u_disp(r).  So the LAMMPS reciprocal-only IK
profile (run with the shell subtraction disabled) MUST equal a direct brute-force
IK pair sum of S(r)*u_disp(r) over the SAME frames -- no mean-field, no
sharp-vs-switched, no shell.  Any difference is a genuine reciprocal-formula bug.

  cpp2_recip.dat : compute stress/cartesian z .. kspace, with the shell subtraction
                   commented out in pressure_profile_long (reciprocal only).
  RS_switched    : brute-force IK sum of d[S u]/dr over r in [3.4, rmax], this script.
"""
import numpy as np
from scipy.spatial import cKDTree
import verify_pressure as V
import verify_cpp2 as C

LX = LY = C.LX; LZ = C.LZ; AREA = C.AREA; NB = C.NB; DZ = C.DZ
RLO, DEL, RC = 3.4, 0.6, 4.0          # CPP2 switch [3.4,4.0]; S=1 beyond
RMAX = 11.0
OFF = NB; EXT = NB + 2*OFF
SHIFTS = np.array([[a*LX, b*LY, c*LZ] for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)])

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

def rs_switched(frames):
    # validated IK P_N-P_T pair virial (same as verify_cpp2.realspace_IK):
    #   W_NT = (du/dr)/(2r) (r^2 - 3 rz^2)
    acc = np.zeros(NB)
    for xs in frames:
        img = (xs[None, :, :]+SHIFTS[:, None, :]).reshape(-1, 3)
        sdm = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
        r = sdm.data; row = sdm.row; col = sdm.col
        keep = r >= RLO
        r = r[keep]; row = row[keep]; col = col[keep]
        zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
        W = dudr_sw(r)/(2.0*r)*(r*r - 3.0*rz*rz)
        L = np.maximum(np.abs(rz), 1e-12)
        zlo = np.minimum(zi, zj); zhi = np.maximum(zi, zj)
        acc += 0.5*deposit(zlo, zhi, W/(AREA*L))
    return acc/len(frames)


def main():
    frames = C.read_dump("traj_cpp2.dump")[::2]
    g_rs = rs_switched(frames)
    ab = V.parse_ave_time_vector("cpp2_recip.dat")[-1][1]
    g_recip = ab[:, 7]-0.5*(ab[:, 5]+ab[:, 6])
    z = ab[:, 0]
    sm = lambda a: V.fourier_smooth(a, 30); dm = lambda a: a-a.mean()
    rms = lambda a, b: np.sqrt(np.mean((sm(dm(a))-sm(dm(b)))**2))
    print("Ewald-identity test: LAMMPS reciprocal-only IK vs brute-force IK (same switched potential)")
    print("  P_N-P_T shape rms = %.5f   ptp recip=%.4f  ptp brute=%.4f  ratio=%.4f" % (
        rms(g_recip, g_rs), np.ptp(sm(g_recip)), np.ptp(sm(g_rs)), np.ptp(sm(g_rs))/np.ptp(sm(g_recip))))
    print("  => ~0 means the reciprocal IK formula is correct (Ewald identity holds);")
    print("     a systematic gap means a genuine reciprocal-formula bug.")
    np.save("g_recip.npy", g_recip); np.save("g_rs_switched.npy", g_rs)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    plt.plot(z, sm(dm(g_recip)), "-", color="green", lw=2, label="LAMMPS reciprocal-only IK")
    plt.plot(z, sm(dm(g_rs)), "--", color="red", lw=1.3, label="brute-force IK (switched S*u)")
    plt.xlabel("z*"); plt.ylabel(r"$P_N-P_T$ (mean removed)"); plt.legend()
    plt.title("Ewald-identity check of the reciprocal IK formula")
    plt.tight_layout(); plt.savefig("fig_recip_check.png", dpi=130); plt.close()
    print("  wrote fig_recip_check.png")


if __name__ == "__main__":
    main()
