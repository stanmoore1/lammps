#!/usr/bin/env python3
"""
Reproduce Figure 4.7 of the dissertation (Cribb, "Chemical Potential Perturbation")
for the ewald/disp/planar long-range dispersion: the local long-range contribution
to the surface tension P_N^LR(z) - P_T^LR(z), computed by

  (i)  the LATTICE SUM  = ewald/disp/planar kspace contribution to compute
       stress/cartesian (IK contour) and compute stress/atom (H contour), and
  (ii) the SLAB METHOD = the virial-type density-profile integral, Eq. 4.18:

   P_N^LR - P_T^LR = (pi/2) rho(z) \int_rcut^inf dr du/dr \int_{-r}^{r} dz'
                         [r^2 - 3 z'^2] rho(z+z')          (Harasima contour)

   The Irving-Kirkwood version replaces rho(z) rho(z+z') by the bond line average
   \int_0^1 dlambda rho(z-lambda z') rho(z+(1-lambda) z').

Our kspace represents the SWITCHED dispersion S(r) u_disp(r) (septic switch on
[rcut_lo, rcut_lo+Delta] = [2.4, 3.0]); the kspace NET (reciprocal - shell) is the
r>3.0 tail (the real-space pair covers r<3.0). So the matching slab potential is
the sharp dispersion tail u_disp(r) = -4/r^6 for r > 3.0; we also show the full
switched S*u_disp for reference.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import verify_pressure as V

DZ = 0.1; AREA = 100.0; VCHUNK = AREA*DZ; LZ = 36.0
RLO = 2.4; DELTA = 0.6; RCUT = 3.0          # switch [2.4,3.0]; sharp tail at 3.0

def switch_S(t):
    t = np.clip(t, 0.0, 1.0)
    return t**4*(35.0 - 84.0*t + 70.0*t**2 - 20.0*t**3)
def switch_dSdt(t):
    tu = t*(1.0-t)
    return np.where((t>0)&(t<1), 140.0*tu**3, 0.0)

def dudr_switched(r):           # d/dr [S(r) * (-4/r^6)]
    t = (r-RLO)/DELTA
    S = switch_S(t); Sp = switch_dSdt(t)/DELTA
    u = -4.0/r**6; dudr = 24.0/r**7
    return Sp*u + S*dudr
def dudr_sharp(r):              # d/dr [-4/r^6] for r>RCUT, else 0
    return np.where(r>=RCUT, 24.0/r**7, 0.0)

def Gkernel(zp, dudr, rmax=12.0, nr=4000):
    """G(z') = \int_{max(rlo,|z'|)}^inf (du/dr)(r^2 - 3 z'^2) dr  for each z'."""
    G = np.zeros_like(zp)
    for i, z in enumerate(zp):
        a = max(RLO if dudr is dudr_switched else RCUT, abs(z))
        if a >= rmax:
            continue
        r = np.linspace(a, rmax, nr)
        G[i] = np.trapezoid(dudr(r)*(r**2 - 3.0*z**2), r)
    return G

def slab_H(rho, z, G, zp):
    """(pi/2) rho(z) sum_z' G(z') rho(z+z') dz'  (periodic)."""
    n = len(z); out = np.zeros(n)
    shifts = np.round(zp/DZ).astype(int)
    for k, s in enumerate(shifts):
        out += G[k]*np.roll(rho, -s)
    return (np.pi/2.0)*rho*out*DZ

def slab_IK(rho, z, G, zp, nlam=12):
    """(pi/2) sum_z' G(z') [\int_0^1 dl rho(z-l z') rho(z+(1-l) z')] dz' (periodic)."""
    n = len(z); out = np.zeros(n)
    shifts = zp/DZ
    lam = (np.arange(nlam)+0.5)/nlam
    for k, sfull in enumerate(shifts):
        acc = np.zeros(n)
        for l in lam:
            sa = int(round(-l*sfull)); sb = int(round((1.0-l)*sfull))
            acc += np.roll(rho, -sa)*np.roll(rho, -sb)
        out += G[k]*(acc/nlam)
    return (np.pi/2.0)*out*DZ


def main():
    # density rho(z) and H long-range from hLR.dat (lab frame, same frames as lattice)
    hb = V.parse_ave_chunk("hLR.dat")[-1][1]   # last running-avg block: [coord,Ncount,Sxx,Syy,Szz]
    z = hb[:,0].copy(); rho = hb[:,1]/VCHUNK
    PN_H = -hb[:,4]/VCHUNK; PT_H = -0.5*(hb[:,2]+hb[:,3])/VCHUNK
    g_H_lat = PN_H - PT_H

    # IK long-range from ikLR.dat (kspace-only stress/cartesian: pcxx=pT, pczz=pN)
    ab = V.parse_ave_time_vector("ikLR.dat")[-1][1]
    PN_IK = ab[:,7]; PT_IK = 0.5*(ab[:,5]+ab[:,6])
    g_IK_lat = PN_IK - PT_IK

    # slab kernels
    zp = np.arange(-90, 91)*DZ                 # z' in [-9,9]
    G_sharp = Gkernel(zp, dudr_sharp)
    G_switch = Gkernel(zp, dudr_switched)
    g_H_slab   = slab_H(rho, z, G_sharp, zp)
    g_IK_slab  = slab_IK(rho, z, G_sharp, zp)
    g_H_slab_s = slab_H(rho, z, G_switch, zp)

    def rms(a,b): return np.sqrt(np.mean((a-b)**2))
    def mx(a,b): return np.max(np.abs(a-b))
    print("Fig 4.7 reproduction: local long-range P_N^LR - P_T^LR(z)")
    print("="*64)
    print("gamma_LR (0.5*dz*sum):  IK_lat=%.4f  IK_slab=%.4f | H_lat=%.4f  H_slab=%.4f"%(
        0.5*DZ*g_IK_lat.sum(),0.5*DZ*g_IK_slab.sum(),0.5*DZ*g_H_lat.sum(),0.5*DZ*g_H_slab.sum()))
    print("IK: lattice vs slab : max|diff|=%.5f rms=%.5f"%(mx(g_IK_lat,g_IK_slab),rms(g_IK_lat,g_IK_slab)))
    print("H : lattice vs slab : max|diff|=%.5f rms=%.5f"%(mx(g_H_lat,g_H_slab),rms(g_H_lat,g_H_slab)))
    print("H : lattice vs slab(switched) : max|diff|=%.5f"%mx(g_H_lat,g_H_slab_s))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(z, g_IK_lat, "-", color="green", lw=2, label="Lattice Sum (ewald/disp/planar IK)")
    ax[0].plot(z, g_IK_slab, "-.", color="black", lw=1.4, label="Slab Method (Eq 4.18, IK form)")
    ax[0].set_title("IK contour: lattice sum vs slab (Eq 4.18 IK)")
    ax[1].plot(z, g_H_lat, "-", color="green", lw=2, label="Lattice Sum (stress/atom H)")
    ax[1].plot(z, g_H_slab, "-.", color="black", lw=1.4, label="Slab Method (Eq 4.18 H)")
    ax[1].set_title("H contour: lattice sum vs slab (Eq 4.18)")
    for a in ax:
        a.set_xlabel("z*"); a.set_ylabel(r"$P_N^{LR}-P_T^{LR}$"); a.legend(); a.axhline(0,color="0.7",lw=0.6)
    plt.tight_layout(); plt.savefig("fig47_reproduction.png", dpi=130); plt.close()
    print("wrote fig47_reproduction.png")


if __name__ == "__main__":
    main()
