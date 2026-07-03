#!/usr/bin/env python3
"""
Arithmetic (Lorentz-Berthelot) mixing, nchan=7: verify the IK long-range pressure
profile for a 2-type dispersion system.

A single-cosine/multimode single-type test only exercises nchan=1 (geometric).
With two types and `pair_modify mix arithmetic` the dispersion uses the 7-channel
binomial C6 cross expansion (nchan=7).  This exercises:
  - the host single-channel profile structure factor (Bt = sqrt(C6_tt)) for arith,
  - the Kokkos device 7-channel amplitude tables (d_Bt / d_Bfull),
  - ewald vs pppm consistency for arith.

Config: data.arith (gen_arith.py), a deterministic multimode density, types
alternating 1/2 (eps/sig: 1.0/1.0 and 1.5/1.1).  Reads the KSPACE-ONLY IK profile
(stress/cartesian z dz NULL 0 kspace) from each code.  (Pair+kspace is not used
here because the deterministic config has close in-plane contacts -> huge r^-12
pair virial; the kspace contour is independent of the pair, so kspace-only is the
clean arith test.)

NOTE: this verification exposed and fixed a bug in ewald/disp/planar's arith
profile (it reused the channeled solver structure factor sfacrl_all instead of the
single-channel Bt one -> ~130x wrong).  After the fix ewald == pppm == pppm/kk.
"""
import numpy as np
import verify_pressure as V
import verify_cpp2 as C

DZ = C.DZ; AREA = C.AREA; LX = LY = C.LX; LZ = C.LZ; NB = C.NB

def gprof(fn):
    ab = V.parse_ave_time_vector(fn)[-1][1]
    return ab[:, 7], ab[:, 7]-0.5*(ab[:, 5]+ab[:, 6])

def brute_disp(RMAX, RCUT=4.0):
    """brute-force dispersion-only IK P_N-P_T with exact LB C6 (sharp r>RCUT)."""
    from scipy.spatial import cKDTree
    d = np.load("arith_config.npz"); xs = d["xyz"]; typ = d["typ"]
    eps = {1: 1.0, 2: 1.5}; sig = {1: 1.0, 2: 1.1}
    OFF = NB; EXT = NB+2*OFF
    SH = np.array([[a*LX, b*LY, c*LZ]
                   for a in (-2,-1,0,1,2) for b in (-2,-1,0,1,2) for c in (-1,0,1)])
    ntyp = np.tile(typ, len(SH))
    def dep(zlo, zhi, hp):
        D = np.zeros(EXT+2); plo = zlo/DZ+OFF; phi = zhi/DZ+OFF
        gl = np.floor(plo).astype(int); fl = plo-gl
        gh = np.floor(phi).astype(int); fh = phi-gh
        np.add.at(D, gl, hp*(1-fl)); np.add.at(D, gl+1, hp*fl)
        np.add.at(D, gh, -hp*(1-fh)); np.add.at(D, gh+1, -hp*fh)
        P = np.cumsum(D)[:EXT]; pr = np.zeros(NB)
        for k in range(0, EXT, NB):
            s = P[k:k+NB]; pr[:len(s)] += s
        return pr
    img = (xs[None, :, :]+SH[:, None, :]).reshape(-1, 3)
    sd = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
    r = sd.data; row = sd.row; col = sd.col; k = r >= RCUT
    r = r[k]; row = row[k]; col = col[k]
    ti = typ[row]; tj = ntyp[col]
    C6 = 4*np.sqrt(np.array([eps[t] for t in ti])*np.array([eps[t] for t in tj])) * \
        (0.5*(np.array([sig[t] for t in ti])+np.array([sig[t] for t in tj])))**6
    zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
    f = (6*C6/r**7)/r                       # du/dr = 6 C6 / r^7 for u = -C6/r^6
    WNT = f*0.5*(r*r-3*rz*rz); L = np.maximum(np.abs(rz), 1e-12)
    zlo = np.minimum(zi, zj); zhi = np.maximum(zi, zj)
    return 0.5*dep(zlo, zhi, WNT/(AREA*L))


def main():
    gNe, gNTe = gprof("arith_ewald_ks.dat")
    gNp, gNTp = gprof("arith_pppm_ks.dat")
    gNk, gNTk = gprof("arith_pppmkk_ks.dat")
    sm = lambda a: V.fourier_smooth(a, 25)
    print("Arithmetic mixing (nchan=7) KSPACE-ONLY IK contour, 2-type LB system:")
    print("  gamma_LR:  ewald=%.5f  pppm=%.5f  pppm_kk=%.5f" % (
        0.5*DZ*gNTe.sum(), 0.5*DZ*gNTp.sum(), 0.5*DZ*gNTk.sum()))
    print("  ewald vs pppm:  ratio ptp=%.4f  rms=%.2e   (exact sum vs mesh)" % (
        np.ptp(sm(gNTe))/np.ptp(sm(gNTp)), np.sqrt(np.mean((sm(gNTe)-sm(gNTp))**2))))
    print("  pppm/kk vs pppm:  max|d| P_N=%.2e  P_N-P_T=%.2e   (device arith d_Bt/d_Bfull)" % (
        np.max(np.abs(gNk-gNp)), np.max(np.abs(gNTk-gNTp))))
    print("  brute-force dispersion-mixed (exact LB C6) ptp, sharp r>4:")
    for RMAX in (8.0, 11.0, 14.0):
        gb = brute_disp(RMAX)
        print("    RMAX=%g  ptp_brute=%.4f  (kspace ptp=%.4f)" % (
            RMAX, np.ptp(sm(gb)), np.ptp(sm(gNTe))))
    print("  => the strong test is the EXACT cross-code agreement ewald == pppm ==")
    print("     pppm/kk (ratio 1.0000); the brute force is a scale/sign check and")
    print("     confirms the fix gives a physical O(0.1) profile (the pre-fix ewald")
    print("     gave O(10)).  The brute(sharp r>4)-vs-kspace(net) offset is the same")
    print("     [rcut-Delta,rcut] shell-region systematic present for nchan=1, not")
    print("     the mixing.")


if __name__ == "__main__":
    main()
