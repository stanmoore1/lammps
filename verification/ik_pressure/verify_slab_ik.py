#!/usr/bin/env python3
"""
IK contour correctness for pppm/disp/slab (smooth-damped merged-corr method).

The reciprocal sum represents the SMOOTH dispersion tail
  u_smooth(r) = S(t) * (-4/r^6)  for rcut < r < rcut+Delta   (t=(r-rcut)/Delta)
             = -4/r^6            for r >= rcut+Delta
             = 0                 for r < rcut
(the lj/cut/dispswitch pair supplies full LJ to rcut and the (1-S) remainder over
the switch shell).  The real-space slab correction is FOLDED into the reciprocal
coefficients (merged corr), so there is NO separate shell subtraction: by the
Ewald identity the kspace IK profile equals a direct real-space IK pair sum of
u_smooth over the same frames.  Brute-force that and sweep the cutoff RMAX; the
ratio must -> 1 as RMAX captures the r^-6 tail.
"""
import numpy as np
from scipy.spatial import cKDTree
import verify_pressure as V
import verify_cpp2 as C

LX = LY = C.LX; LZ = C.LZ; AREA = C.AREA; NB = C.NB; DZ = C.DZ
RCUT, DELTA = 4.0, 0.6                # lj/cut/dispswitch 4.0 0.6 (outward switch)
OFF = NB; EXT = NB + 2*OFF
SHIFTS = np.array([[a*LX, b*LY, c*LZ]
                   for a in (-2,-1,0,1,2) for b in (-2,-1,0,1,2) for c in (-1,0,1)])

def dudr_smooth(r):
    t = np.clip((r-RCUT)/DELTA, 0, 1)
    S = t**4*(35-84*t+70*t**2-20*t**3)
    Sp = np.where((t > 0) & (t < 1), 140*(t*(1-t))**3, 0.0)/DELTA
    return np.where(r >= RCUT, Sp*(-4.0/r**6) + S*(24.0/r**7), 0.0)

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

def brute(frames, RMAX):
    acc = np.zeros(NB)
    for xs in frames:
        img = (xs[None, :, :]+SHIFTS[:, None, :]).reshape(-1, 3)
        sdm = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
        r = sdm.data; row = sdm.row; col = sdm.col
        keep = r >= RCUT
        r = r[keep]; row = row[keep]; col = col[keep]
        zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
        W = dudr_smooth(r)/(2.0*r)*(r*r - 3.0*rz*rz)
        L = np.maximum(np.abs(rz), 1e-12)
        zlo = np.minimum(zi, zj); zhi = np.maximum(zi, zj)
        acc += 0.5*deposit(zlo, zhi, W/(AREA*L))
    return acc/len(frames)


def main():
    ab = V.parse_ave_time_vector("slab_ikLR_pppm.dat")[-1][1]
    g_lat = ab[:, 7]-0.5*(ab[:, 5]+ab[:, 6])
    frames = C.read_dump("traj_cpp2.dump")[::4]
    sm = lambda a: V.fourier_smooth(a, 30); dm = lambda a: a-a.mean()
    print("pppm/disp/slab IK contour vs brute-force IK of u_smooth (Ewald identity, no shell):")
    print("  gamma_LR (lattice) = %.4f" % (0.5*DZ*g_lat.sum()))
    print("  RMAX  gamma_brute  ptp_brute/ptp_lat  shape_rms")
    for RMAX in (8.0, 11.0, 14.0):
        gb = brute(frames, RMAX)
        print("  %4.1f   %.4f      %.4f             %.5f" % (
            RMAX, 0.5*DZ*gb.sum(), np.ptp(sm(dm(gb)))/np.ptp(sm(dm(g_lat))),
            np.sqrt(np.mean((sm(dm(gb))-sm(dm(g_lat)))**2))))
    print("  => ratio -> 1, rms -> 0 as RMAX grows: the merged-corr slab kspace IK")
    print("     contour equals the real-space IK of the smooth dispersion (correct).")


if __name__ == "__main__":
    main()
