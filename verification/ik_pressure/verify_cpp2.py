#!/usr/bin/env python3
"""
CPP 2 reproduction (dissertation Table 4.1): supercritical LJ, N=2000, T*=1.5,
cosine external field, rcut=4.0 Ewald dispersion.  Reproduces Fig 4.7 (local
long-range surface tension P_N^LR-P_T^LR) and overlays the digitized dissertation
curve.  Compares:
  - ewald/disp/planar lattice sum  (IK: stress/cartesian, H: stress/atom)
  - slab method Eq 4.18            (H form, and IK analogue from Appendix A)
  - direct real-space IK pair sum  (r>4.0)
All profiles Fourier-cosine smoothed (as the dissertation does).
"""
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import verify_pressure as V
from verify_pressure import fourier_smooth

# CPP 2 box
LX = LY = (2000.0/(2*0.598))**(1.0/3.0)      # 11.872
LZ = 2*LX                                     # 23.744
AREA = LX*LY
DZ = 0.06
NB = int(LZ/DZ)             # LAMMPS floors: nbins = int((boxhi-boxlo)/bin_width) = 395
DZ = LZ/NB
VCHUNK = AREA*DZ
RCUT = 4.0; RMAX = 12.0; NMODES = 25

# ---- digitized dissertation Fig 4.7 (H contour, r>4.0), z* in [0,23.74] ----
DISS_Z = np.array([0,1,2,2.5,3,4,5,6,7,7.5,8,9,10,11,12,13,14,15,16,16.5,17,18,19,20,21,21.5,22,23,23.74])
DISS_G = np.array([-0.005,-0.005,-0.007,-0.008,-0.006,-0.001,0.008,0.018,0.023,0.0243,0.0245,
                   0.023,0.0205,0.0185,0.018,0.0185,0.0205,0.023,0.0245,0.0243,0.0235,0.019,
                   0.010,0.001,-0.006,-0.0075,-0.007,-0.005,-0.005])

# ---------- dispersion tail kernel for the slab (sharp r>RCUT) ----------
def dudr_sharp(r):
    return np.where(r >= RCUT, 24.0/r**7, 0.0)
def Gkernel(zp, rmax=14.0, nr=4000):
    G = np.zeros_like(zp)
    for i, z in enumerate(zp):
        a = max(RCUT, abs(z))
        if a >= rmax: continue
        r = np.linspace(a, rmax, nr)
        G[i] = np.trapezoid(dudr_sharp(r)*(r**2 - 3.0*z**2), r)
    return G
def roll_frac(a, s):
    """periodic shift of a by s bins (fractional), linear interp: result[i]=a(i-s)."""
    n = len(a); idx = (np.arange(n) - s) % n
    i0 = np.floor(idx).astype(int) % n; f = idx - np.floor(idx)
    return a[i0]*(1.0-f) + a[(i0+1) % n]*f
def slab_H(rho, G, zp):
    out = np.zeros(NB)
    for k, sf in enumerate(zp/DZ): out += G[k]*roll_frac(rho, -sf)   # rho(z+z')
    return (np.pi/2.0)*rho*out*DZ
def slab_IK(rho, G, zp, nlam=24):
    out = np.zeros(NB); lam = (np.arange(nlam)+0.5)/nlam
    for k, sf in enumerate(zp/DZ):
        acc = np.zeros(NB)
        for l in lam:                                                # rho(z-l z') rho(z+(1-l) z')
            acc += roll_frac(rho, l*sf)*roll_frac(rho, -(1.0-l)*sf)
        out += G[k]*(acc/nlam)
    return (np.pi/2.0)*out*DZ

# ---------- direct real-space IK (exact fractional deposit) ----------
SHIFTS = np.array([[a*LX,b*LY,c*LZ] for a in(-1,0,1) for b in(-1,0,1) for c in(-1,0,1)])
OFF = NB; EXT = NB+2*OFF
def deposit(zlo, zhi, hp):
    D = np.zeros(EXT+2)
    plo = zlo/DZ+OFF; phi = zhi/DZ+OFF
    glo = np.floor(plo).astype(int); flo = plo-glo
    ghi = np.floor(phi).astype(int); fhi = phi-ghi
    np.add.at(D,glo,hp*(1-flo)); np.add.at(D,glo+1,hp*flo)
    np.add.at(D,ghi,-hp*(1-fhi)); np.add.at(D,ghi+1,-hp*fhi)
    P = np.cumsum(D)[:EXT]; prof = np.zeros(NB)
    for k in range(0,EXT,NB):
        seg = P[k:k+NB]; prof[:len(seg)] += seg
    return prof
def read_dump(fn):
    frames=[]; L=open(fn).readlines(); i=0
    while i<len(L):
        if L[i].startswith("ITEM: TIMESTEP"):
            n=int(L[i+3]); h=L[i+8].split()[2:]; ix,iy,iz=h.index("x"),h.index("y"),h.index("z")
            xs=np.array([[float(p[ix]),float(p[iy]),float(p[iz])] for p in (L[i+9+a].split() for a in range(n))])
            xs[:,0]%=LX; xs[:,1]%=LY; xs[:,2]%=LZ; frames.append(xs); i+=9+n
        else: i+=1
    return frames
def realspace_IK(frames):
    acc=np.zeros(NB)
    for xs in frames:
        img=(xs[None,:,:]+SHIFTS[:,None,:]).reshape(-1,3)
        sdm=cKDTree(xs).sparse_distance_matrix(cKDTree(img),RMAX,output_type="coo_matrix")
        r=sdm.data; row=sdm.row; col=sdm.col; keep=r>=RCUT
        r=r[keep]; row=row[keep]; col=col[keep]
        zi=xs[row,2]; zj=img[col,2]; rz=zj-zi
        W=(24.0/r**7)/(2*r)*(r**2-3*rz**2); L_=np.maximum(np.abs(rz),1e-12)
        acc+=0.5*deposit(np.minimum(zi,zj),np.maximum(zi,zj),W/(AREA*L_))
    return acc/len(frames)


def main():
    z = (np.arange(NB)+0.5)*DZ
    # IK lattice from cpp2_ikLR.dat (395-bin master grid)
    ab = V.parse_ave_time_vector("cpp2_ikLR.dat")[-1][1]
    g_IK_lat = ab[:,7]-0.5*(ab[:,5]+ab[:,6])
    # density and H lattice from cpp2_hLR.dat (chunk grid, 396 bins, width 0.06) ->
    # interpolate onto the 395-bin master grid (periodic)
    hb = V.parse_ave_chunk("cpp2_hLR.dat")[-1][1]
    zh = hb[:,0]; VCH = AREA*0.06
    rho = np.interp(z, zh, hb[:,1]/VCH, period=LZ)
    g_H_lat = np.interp(z, zh, (-hb[:,4] + 0.5*(hb[:,2]+hb[:,3]))/VCH, period=LZ)
    # slab
    zp = np.arange(-int(14/DZ), int(14/DZ)+1)*DZ
    G = Gkernel(zp)
    g_H_slab = slab_H(rho, G, zp); g_IK_slab = slab_IK(rho, G, zp)
    # real-space IK
    g_IK_real = realspace_IK(read_dump("traj_cpp2.dump"))

    sm = lambda a: fourier_smooth(a, NMODES)
    print("CPP 2 reproduction (T*=1.5, rcut=4.0).  rho range %.3f - %.3f (avg %.3f)"%(
        rho.min(), rho.max(), rho.mean()))
    print("gamma_LR: H_lat=%.4f H_slab=%.4f | IK_lat=%.4f IK_slab=%.4f IK_real=%.4f"%(
        0.5*DZ*g_H_lat.sum(),0.5*DZ*g_H_slab.sum(),0.5*DZ*g_IK_lat.sum(),
        0.5*DZ*g_IK_slab.sum(),0.5*DZ*g_IK_real.sum()))
    print("peaks: diss=%.4f  H_lat=%.4f  IK_lat=%.4f  IK_real=%.4f"%(
        DISS_G.max(), sm(g_H_lat).max(), sm(g_IK_lat).max(), sm(g_IK_real).max()))
    rms = lambda a,b: np.sqrt(np.mean((sm(a)-sm(b))**2))
    print("rms(smoothed):  H lat-vs-slab=%.5f | IK lat-vs-slab=%.5f  IK lat-vs-real=%.5f  IK slab-vs-real=%.5f"%(
        rms(g_H_lat,g_H_slab), rms(g_IK_lat,g_IK_slab), rms(g_IK_lat,g_IK_real), rms(g_IK_slab,g_IK_real)))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # left: H contour (reproduces dissertation Fig 4.7) + digitized diss
    ax[0].plot(z, sm(g_H_lat), "-", color="green", lw=2, label="Lattice Sum (ewald/disp/planar, H)")
    ax[0].plot(z, sm(g_H_slab), "-.", color="black", lw=1.4, label="Slab Method (Eq 4.18, H)")
    ax[0].plot(DISS_Z, DISS_G, "o", color="red", ms=4, label="Dissertation Fig 4.7 (digitized)")
    ax[0].set_title("H contour (CPP 2) vs dissertation Fig 4.7")
    # right: IK contour (new code) verified 3 ways
    ax[1].plot(z, sm(g_IK_lat), "-", color="green", lw=2, label="Lattice Sum (ewald/disp/planar, IK)")
    ax[1].plot(z, sm(g_IK_slab), "-.", color="black", lw=1.4, label="Slab Method (Eq 4.18 IK, Appendix A)")
    ax[1].plot(z, sm(g_IK_real), "--", color="purple", lw=1.3, label="Direct real-space IK")
    ax[1].plot(DISS_Z, DISS_G, "o", color="red", ms=3, alpha=0.5, label="Dissertation Fig 4.7 (H, ref)")
    ax[1].set_title("IK contour (CPP 2): lattice = slab = real-space")
    for a in ax:
        a.set_xlabel("z*"); a.set_ylabel(r"$P_N^{LR}-P_T^{LR}$"); a.axhline(0,color="0.7",lw=0.6)
        a.legend(fontsize=8); a.set_xlim(0, LZ)
    plt.tight_layout(); plt.savefig("fig_cpp2_fig47.png", dpi=130); plt.close()

    # density profile vs dissertation range
    plt.figure(figsize=(7,4))
    plt.plot(z, sm(rho), "-", lw=1.5)
    plt.axhline(0.053, color="r", ls=":", label="diss rho_min=0.053")
    plt.axhline(0.898, color="b", ls=":", label="diss rho_max=0.898")
    plt.xlabel("z*"); plt.ylabel(r"$\rho(z)$"); plt.title("CPP 2 density profile (cosine field)")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_cpp2_density.png", dpi=130); plt.close()
    print("wrote fig_cpp2_fig47.png fig_cpp2_density.png")


if __name__ == "__main__":
    main()
