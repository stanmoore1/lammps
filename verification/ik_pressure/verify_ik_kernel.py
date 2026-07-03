#!/usr/bin/env python3
"""
Pinpoint the ewald/disp/planar IK off-diagonal profile bug by reconstructing the
reciprocal double-sum in Python from the trajectory structure factors S_n, using
the EXACT code kernels, and comparing its z-shape to:
  - the LAMMPS lattice output cpp2_ikLR.dat   (validates the replication)
  - the real-space / slab ground truth (verify_cpp2 slab_IK)

For P_N-P_T(z) the diagonal (p=n+m=0) terms are z-independent (a constant), so the
entire z-SHAPE comes from the off-diagonal p!=0 terms:
   g(z) = sum_{n,m : n+m!=0} S_n S_m (CN-CT)_{n,m} e^{i(h_n+h_m)z}
with the code coefficients
   CT = -6  pi / H (ik_phi(hm)+ik_phi(hn))
   CN = -12 pi / H (ik_psi(hm)+ik_psi(hn)),   H = h_n+h_m.
We also evaluate the paper's Appendix A coefficients (factor and n=0 handling) to
see which matches the real-space ground truth.
"""
import numpy as np
import verify_pressure as V
import verify_cpp2 as C

PI = np.pi
LX = LY = (2000.0/(2*0.598))**(1.0/3.0); LZ = 2*LX; AREA = LX*LY; VOL = AREA*LZ
CUT = 4.0                       # outer cutoff (sharp; prof_shell=0, c=rcut)
UNITK = 2*PI/LZ

# ---------- sici_chain (replicate ewald_disp_planar.cpp), returns A[0..7],B[0..7] ----------
from verify_math import sici_chain

def ik_psi(h):
    if abs(h) < 1e-300: return 0.0
    ah = abs(h); A, B = sici_chain(ah*CUT)
    psi = PI/288.0 - A[7] + B[6]                  # prof_shell=0 (sharp)
    return (1.0 if h >= 0 else -1.0)*ah**4*psi
def ik_phi(h):
    if abs(h) < 1e-300: return 0.0
    ah = abs(h); A, B = sici_chain(ah*CUT)
    sii5 = A[5]/4.0 - A[1]/(4.0*(ah*CUT)**4)
    phi = PI/576.0 - sii5 + A[7] - B[6]
    return (1.0 if h >= 0 else -1.0)*ah**4*phi

def Sn_from_traj(fn, K):
    """frame-averaged outer product <S_n S_m>, n,m in [-K,K]; S_n=(1/V)sum_j B e^{-i h_n z_j}, B=2."""
    frames = C.read_dump(fn)
    ns = np.arange(-K, K+1)
    SS = np.zeros((2*K+1, 2*K+1), complex)
    for xs in frames:
        z = xs[:, 2]
        S = (2.0/VOL)*np.sum(np.exp(-1j*UNITK*np.outer(ns, z)), axis=1)   # S_n
        SS += np.outer(S, S)
    return ns, SS/len(frames)

def Sn_meanfield(rho, K):
    """mean-field S_n from the continuum density: S_n=(2/Lz) int rho(z) e^{-i h_n z} dz.
    Removes correlations & statistics -> clean Fourier-vs-real-space formula test."""
    z = (np.arange(C.NB)+0.5)*C.DZ
    ns = np.arange(-K, K+1)
    rho_hat = np.sum(rho[None, :]*np.exp(-1j*UNITK*np.outer(ns, z)), axis=1)*C.DZ
    S = (2.0/LZ)*rho_hat
    return ns, np.outer(S, S)

def offdiag_shape(ns, SS, coeff):
    """g(z) on the NB grid from off-diagonal p=n+m!=0 terms; coeff(hn,hm)->(CN-CT)."""
    K = (len(ns)-1)//2
    z = (np.arange(C.NB)+0.5)*C.DZ
    # accumulate amplitude per p=n+m
    P = 2*K
    amp = np.zeros(2*P+1, complex)   # index p+P
    for i, n in enumerate(ns):
        hn = n*UNITK
        for j, m in enumerate(ns):
            p = n+m
            if p == 0: continue
            hm = m*UNITK
            amp[p+P] += SS[i, j]*coeff(hn, hm)
    g = np.zeros(C.NB)
    for p in range(-P, P+1):
        if p == 0: continue
        g += np.real(amp[p+P]*np.exp(1j*p*UNITK*z))
    return g

def code_coeff(hn, hm):
    H = hn+hm
    CT = -6.0*PI/H*(ik_phi(hm)+ik_phi(hn))
    CN = -12.0*PI/H*(ik_psi(hm)+ik_psi(hn))
    return CN-CT


def paper_coeff(hn, hm):
    """paper Appendix A off-diagonal: N^IK=-96pi/H sum(ik_psi), tangential=(K^IK-N^IK).
    K^IK=-96pi/H sum(sgn|h|^4 (pi/288 - Sii5)).  (CN-CT) = N^IK - (K^IK-N^IK) = 2N^IK-K^IK."""
    def Kik(h):
        if abs(h) < 1e-300: return 0.0
        ah = abs(h); A, B = sici_chain(ah*CUT)
        sii5 = A[5]/4.0 - A[1]/(4.0*(ah*CUT)**4)
        return (1.0 if h >= 0 else -1.0)*ah**4*(PI/288.0 - sii5)
    H = hn+hm
    Nik = -96.0*PI/H*(ik_psi(hn)+ik_psi(hm))
    Kik_nm = -96.0*PI/H*(Kik(hn)+Kik(hm))
    return Nik - (Kik_nm - Nik)            # CN - CT

def main():
    z = (np.arange(C.NB)+0.5)*C.DZ
    hb = V.parse_ave_chunk("cpp2_hLR.dat")[-1][1]
    rho = np.interp(z, hb[:, 0], hb[:, 1]/(AREA*0.06), period=LZ)
    zp = np.arange(-int(14/C.DZ), int(14/C.DZ)+1)*C.DZ
    G = C.Gkernel(zp)
    g_slab = C.slab_IK(rho, G, zp)            # real-space ground truth (mean-field, sharp r>4)

    K = 120
    sm = lambda a: V.fourier_smooth(a, 25)
    dm = lambda a: a - a.mean()
    rms = lambda a, b: np.sqrt(np.mean((sm(dm(a))-sm(dm(b)))**2))

    # CLEAN formula test: mean-field S_n (FFT of the SAME rho) -> lattice double sum,
    # both sharp mean-field, so lattice-formula and slab must be IDENTICAL if the
    # closed-form == the real-space integral.
    ns, SS = Sn_meanfield(rho, K)
    g_code  = offdiag_shape(ns, SS, code_coeff)
    g_paper = offdiag_shape(ns, SS, paper_coeff)
    print("CLEAN mean-field formula test (sharp r>4, K=%d), z-shape of P_N-P_T:" % K)
    print("  lattice(code kernels)  vs slab(real-space): rms=%.5f  ptp_code=%.4f ptp_slab=%.4f" % (
        rms(g_code, g_slab), np.ptp(sm(g_code)), np.ptp(sm(g_slab))))
    print("  lattice(paper kernels) vs slab(real-space): rms=%.5f  ptp_paper=%.4f" % (
        rms(g_paper, g_slab), np.ptp(sm(g_paper))))
    print("  ratio ptp slab/code = %.4f   slab/paper = %.4f" % (
        np.ptp(sm(g_slab))/np.ptp(sm(g_code)), np.ptp(sm(g_slab))/np.ptp(sm(g_paper))))
    np.save("g_code.npy", g_code); np.save("g_paper.npy", g_paper); np.save("g_slab_shape.npy", g_slab)

    # ---- SHELL-CONTOUR test ----------------------------------------------------
    # cpp2_ikLR (NET) = reciprocal_IK(switched, r>3.4) - shell.  The slab ground
    # truth = sharp r>4.0 IK = reciprocal_IK(switched) - shell_IK([3.4,4.0]).  So
    #   NET - slab = shell_IK - shell_H   (if the code subtracts an H-distributed
    # shell instead of the IK-distributed one).  Compute both shells over the
    # switch region [3.4,4.0] with the switched force d(S u)/dr and compare.
    RLO, DEL, RC = 3.4, 0.6, 4.0
    def dudr_sw(r):
        t = np.clip((r-RLO)/DEL, 0, 1)
        S = t**4*(35-84*t+70*t**2-20*t**3)
        Sp = np.where((t > 0) & (t < 1), 140*(t*(1-t))**3, 0.0)/DEL
        return np.where((r >= RLO) & (r <= RC), Sp*(-4.0/r**6) + S*(24.0/r**7), 0.0)
    def Gshell(zp):
        G = np.zeros_like(zp)
        for i, zv in enumerate(zp):
            a = max(RLO, abs(zv))
            if a >= RC: continue
            r = np.linspace(a, RC, 2000)
            G[i] = np.trapezoid(dudr_sw(r)*(r**2-3*zv**2), r)
        return G
    Gs = Gshell(zp)
    shell_H = C.slab_H(rho, Gs, zp)
    shell_IK = C.slab_IK(rho, Gs, zp)
    ab = V.parse_ave_time_vector("cpp2_ikLR.dat")[-1][1]
    g_lat = ab[:, 7]-0.5*(ab[:, 5]+ab[:, 6])
    net_minus_slab = g_lat - g_slab
    shellIK_minus_H = shell_IK - shell_H
    print("")
    print("SHELL-CONTOUR test (does NET-slab == shell_IK - shell_H?):")
    print("  rms(NET-slab)           = %.5f" % np.sqrt(np.mean(dm(sm(net_minus_slab))**2)))
    print("  rms(shell_IK - shell_H) = %.5f" % np.sqrt(np.mean(dm(sm(shellIK_minus_H))**2)))
    print("  rms( (NET-slab) - (shell_IK-shell_H) ) = %.5f  <- ~0 confirms shell-contour bug" %
          rms(net_minus_slab, shellIK_minus_H))
    np.save("net_minus_slab.npy", net_minus_slab); np.save("shellIK_minus_H.npy", shellIK_minus_H)


if __name__ == "__main__":
    main()
