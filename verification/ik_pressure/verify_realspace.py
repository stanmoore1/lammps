#!/usr/bin/env python3
"""
DIRECT real-space verification of the ewald/disp/planar long-range IK contour.

For every snapshot in traj.dump we sum the Irving-Kirkwood pair contribution to
P_N(z)-P_T(z) by brute force over ALL pairs (incl. periodic images):

  pair virial of P_N-P_T:  W_NT = (du/dr)/(2r) (r^2 - 3 r_z^2)
  IK contour: spread W_NT uniformly in z over the bond [z_i, z_j], EXACTLY
  (fractional-bin deposit), so a bond crossing bin g gets W_NT*(overlap_g/L)/(area*dz).

Two long-range potentials are compared to the lattice sum (ikLR.dat):
  SHARP   : u = -4/r^6 for r > rcut=3.0          (Appendix A; matches kspace NET,
            since the shell correction removes the [2.4,3.0] switch region)
  SWITCHED: u = S(r)*(-4/r^6), r > 2.4           (the actual kspace functional;
            equals the reciprocal sum, i.e. NET + shell, so slightly higher)
The Ewald identity makes reciprocal == real-space for the same potential, so the
real-space sum reproduces the kspace IK contour.  Profiles are Fourier-smoothed
(matched band-limit) for the overlay.
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
import verify_pressure as V
from verify_pressure import fourier_smooth

DZ = 0.1; NB = 360; LZ = 36.0; LX = LY = 10.0; AREA = 100.0
RMAX = 12.0
RLO, DELTA, RCUT = 2.4, 0.6, 3.0
NMODES = 45

def switch_S(t):
    t = np.clip(t, 0.0, 1.0)
    return t**4*(35.0 - 84.0*t + 70.0*t**2 - 20.0*t**3)
def switch_dSdt(t):
    tu = t*(1.0-t)
    return np.where((t > 0) & (t < 1), 140.0*tu**3, 0.0)

def dudr(r, mode):
    if mode == "sharp":
        return np.where(r >= RCUT, 24.0/r**7, 0.0)
    t = (r-RLO)/DELTA                       # switched: d/dr[S(r)*(-4/r^6)]
    S = switch_S(t); Sp = switch_dSdt(t)/DELTA
    return np.where(r >= RLO, Sp*(-4.0/r**6) + S*(24.0/r**7), 0.0)

def read_dump(fn):
    frames = []
    with open(fn) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            natoms = int(lines[i+3]); hdr = lines[i+8].split()[2:]
            ix, iy, iz = hdr.index("x"), hdr.index("y"), hdr.index("z")
            xs = np.array([[float(p[ix]), float(p[iy]), float(p[iz])]
                           for p in (lines[i+9+a].split() for a in range(natoms))])
            xs[:, 0] %= LX; xs[:, 1] %= LY; xs[:, 2] %= LZ
            frames.append(xs); i += 9 + natoms
        else:
            i += 1
    return frames

SHIFTS = np.array([[a*LX, b*LY, c*LZ] for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)])
OFF = NB; EXT = NB + 2*OFF

def deposit(zlo, zhi, hp):
    """exact fractional-bin box deposit of per-bond pressure rate hp=W_NT/(L*area)
    over [zlo,zhi], folded periodically into NB bins."""
    D = np.zeros(EXT + 2)
    plo = zlo/DZ + OFF; phi = zhi/DZ + OFF
    glo = np.floor(plo).astype(int); flo = plo - glo
    ghi = np.floor(phi).astype(int); fhi = phi - ghi
    np.add.at(D, glo,     hp*(1.0-flo)); np.add.at(D, glo+1,  hp*flo)
    np.add.at(D, ghi,    -hp*(1.0-fhi)); np.add.at(D, ghi+1, -hp*fhi)
    P = np.cumsum(D)[:EXT]
    prof = np.zeros(NB)
    for k in range(0, EXT, NB):
        seg = P[k:k+NB]; prof[:len(seg)] += seg
    return prof

def frame_profile(xs, mode):
    images = (xs[None, :, :] + SHIFTS[:, None, :]).reshape(-1, 3)
    sdm = cKDTree(xs).sparse_distance_matrix(cKDTree(images), RMAX, output_type="coo_matrix")
    row, col, r = sdm.row, sdm.col, sdm.data
    rcut_lo = RCUT if mode == "sharp" else RLO
    keep = r >= rcut_lo
    row, col, r = row[keep], col[keep], r[keep]
    zi = xs[row, 2]; zj = images[col, 2]; rz = zj - zi
    W_NT = dudr(r, mode)/(2.0*r) * (r**2 - 3.0*rz**2)
    L = np.maximum(np.abs(rz), 1e-12)
    hp = W_NT/(AREA*L)
    zlo = np.minimum(zi, zj); zhi = np.maximum(zi, zj)
    return 0.5*deposit(zlo, zhi, hp)             # 0.5: ordered pairs double counted


def main():
    frames = read_dump("traj.dump")
    print(f"frames={len(frames)} atoms={len(frames[0])} rmax={RMAX} (exact deposit, {NMODES}-mode smoothing)")
    out = {}
    for mode in ("sharp", "switched"):
        acc = np.zeros(NB)
        for xs in frames:
            acc += frame_profile(xs, mode)
        out[mode] = acc/len(frames)

    ab = V.parse_ave_time_vector("ikLR.dat")[-1][1]
    g_lat = ab[:, 7] - 0.5*(ab[:, 5] + ab[:, 6]); z = ab[:, 0]
    g_lat_s = fourier_smooth(g_lat, NMODES)
    np.save("g_realspace.npy", out["sharp"])

    rms = lambda a,b: np.sqrt(np.mean((a-b)**2)); mx = lambda a,b: np.max(np.abs(a-b))
    print("gamma_LR (0.5*dz*sum):  lattice=%.4f  sharp=%.4f  switched=%.4f"%(
        0.5*DZ*g_lat.sum(), 0.5*DZ*out["sharp"].sum(), 0.5*DZ*out["switched"].sum()))
    for m in ("sharp", "switched"):
        gs = fourier_smooth(out[m], NMODES)
        print("real-space %-9s vs lattice (smoothed): max|diff|=%.5f rms=%.5f  peak=%.4f (lat peak=%.4f)"%(
            m, mx(gs, g_lat_s), rms(gs, g_lat_s), gs.max(), g_lat_s.max()))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(z, g_lat_s, "-", color="green", lw=2.4, label="ewald/disp/planar lattice sum (kspace IK)")
    plt.plot(z, fourier_smooth(out["sharp"], NMODES), "--", color="red", lw=1.5,
             label="direct real-space IK, sharp r>3.0 (= NET)")
    plt.plot(z, fourier_smooth(out["switched"], NMODES), ":", color="purple", lw=1.7,
             label="direct real-space IK, switched S*u (= reciprocal)")
    plt.axhline(0, color="0.7", lw=0.6)
    plt.xlabel("z*"); plt.ylabel(r"$P_N^{LR}-P_T^{LR}$")
    plt.title("Long-range IK contour: ewald/disp/planar vs direct real-space (smoothed)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("fig_realspace_IK.png", dpi=130); plt.close()
    print("wrote fig_realspace_IK.png g_realspace.npy")


if __name__ == "__main__":
    main()
