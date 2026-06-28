#!/usr/bin/env python3
"""
DIRECT real-space verification of the ewald/disp/planar long-range IK contour.

For every snapshot in traj.dump we sum the Irving-Kirkwood pair contribution to
P_N(z)-P_T(z) by brute force over ALL pairs (incl. periodic images) with the
sharp dispersion tail u(r) = -4/r^6 for r in [rcut, rmax]:

  pair virial of P_N-P_T:  W_NT = (du/dr)/(2r) (r^2 - 3 r_z^2),   du/dr = 24/r^7
  IK contour: spread W_NT uniformly in z over the segment [z_i, z_j], so each bin
  the bond crosses gets  W_NT / (area * |z_j - z_i|)  (true pressure).

This is the SAME quantity the reciprocal sum (ewald/disp/planar) evaluates: the
Ewald identity makes reciprocal == real-space for the same potential.  The kspace
NET (reciprocal - shell) corresponds to the sharp r>rcut tail because the shell
correction removes the switch-region [2.4,3.0] mean field (Appendix A has no
switch).  Averaged over snapshots, this must match the lattice sum (ikLR.dat).
"""
import numpy as np
from scipy.spatial import cKDTree
import verify_pressure as V

DZ = 0.1; NB = 360; LZ = 36.0; LX = LY = 10.0; AREA = 100.0
RCUT = 3.0; RMAX = 12.0     # tail beyond 12 negligible (~12^-6); 3x3x3 images cover it

def read_dump(fn):
    """yield per-frame Nx3 positions (x,y,z), wrapped into the box."""
    frames = []
    with open(fn) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            natoms = int(lines[i+3])
            # box bounds lines i+5,6,7 ; atom header i+8 ; atoms i+9..
            hdr = lines[i+8].split()[2:]          # column names
            ix, iy, iz = hdr.index("x"), hdr.index("y"), hdr.index("z")
            xs = np.empty((natoms, 3))
            for a in range(natoms):
                p = lines[i+9+a].split()
                xs[a] = [float(p[ix]), float(p[iy]), float(p[iz])]
            xs[:, 0] %= LX; xs[:, 1] %= LY; xs[:, 2] %= LZ
            frames.append(xs)
            i += 9 + natoms
        else:
            i += 1
    return frames

# 3x3x3 periodic images (lateral fully, z for edge atoms)
SHIFTS = np.array([[a*LX, b*LY, c*LZ] for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)])

def frame_profile(xs):
    n = len(xs)
    images = (xs[None, :, :] + SHIFTS[:, None, :]).reshape(-1, 3)   # 27n x 3
    tC = cKDTree(xs); tI = cKDTree(images)
    sdm = tC.sparse_distance_matrix(tI, RMAX, output_type="coo_matrix")
    row, col, r = sdm.row, sdm.col, sdm.data
    keep = r >= RCUT
    row, col, r = row[keep], col[keep], r[keep]
    zi = xs[row, 2]; zj = images[col, 2]
    rz = zj - zi
    dudr = 24.0 / r**7
    W_NT = dudr/(2.0*r) * (r**2 - 3.0*rz**2)        # pair virial of P_N-P_T
    # bond z-extent; a bond shorter than one bin deposits into a single bin with
    # pressure W_NT/(area*dz), so cap the effective length at dz (the IK line
    # integral of a sub-bin bond is just W_NT in that one bin).
    L = np.maximum(np.abs(rz), DZ)
    rate = W_NT / (AREA * L)                          # per-bin pressure for crossed bins
    # deposit over [zi, zj] onto an extended bin grid, then fold mod NB
    zlo = np.minimum(zi, zj); zhi = np.maximum(zi, zj)
    OFF = NB                                          # offset = multiple of NB -> fold is shift-free
    EXT = NB + 2*OFF
    g0 = np.round(zlo/DZ).astype(int) + OFF
    g1 = np.round(zhi/DZ).astype(int) + OFF
    g1 = np.maximum(g1, g0+1)                         # at least one bin
    diff = np.zeros(EXT+1)
    np.add.at(diff, g0, rate)
    np.add.at(diff, g1, -rate)
    dens_ext = np.cumsum(diff)[:EXT]
    prof = np.zeros(NB)                               # fold periodic
    for k in range(0, EXT, NB):
        seg = dens_ext[k:k+NB]
        prof[:len(seg)] += seg
    return 0.5 * prof                                 # 0.5: ordered pairs double counted


def main():
    frames = read_dump("traj.dump")
    print(f"frames={len(frames)}  atoms={len(frames[0])}  rcut={RCUT} rmax={RMAX}")
    acc = np.zeros(NB)
    for k, xs in enumerate(frames):
        acc += frame_profile(xs)
    g_real = acc/len(frames)                           # P_N-P_T(z), direct real space

    # lattice sum (ewald/disp/planar kspace IK) and slab IK, same frames
    ab = V.parse_ave_time_vector("ikLR.dat")[-1][1]
    g_lat = ab[:, 7] - 0.5*(ab[:, 5] + ab[:, 6])
    z = ab[:, 0]
    np.save("g_realspace.npy", g_real)

    rms = lambda a,b: np.sqrt(np.mean((a-b)**2)); mx = lambda a,b: np.max(np.abs(a-b))
    print("gamma_LR (0.5*dz*sum):  real-space=%.4f   lattice(kspace)=%.4f"%(
        0.5*DZ*g_real.sum(), 0.5*DZ*g_lat.sum()))
    print("real-space vs lattice : max|diff|=%.5f  rms=%.5f"%(mx(g_real,g_lat), rms(g_real,g_lat)))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4.5))
    plt.plot(z, g_lat, "-", color="green", lw=2, label="Lattice sum (ewald/disp/planar IK)")
    plt.plot(z, g_real, "--", color="red", lw=1.3, label="Direct real-space IK integration")
    plt.axhline(0, color="0.7", lw=0.6)
    plt.xlabel("z*"); plt.ylabel(r"$P_N^{LR}-P_T^{LR}$")
    plt.title("Long-range IK contour: ewald/disp/planar vs direct real-space sum")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_realspace_IK.png", dpi=130); plt.close()
    print("wrote fig_realspace_IK.png  g_realspace.npy")


if __name__ == "__main__":
    main()
