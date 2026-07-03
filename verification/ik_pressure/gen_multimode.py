#!/usr/bin/env python3
"""
Generate a single-frame LAMMPS dump whose z-density EXACTLY realizes the multi-mode
profile rho(z) = rho0 + sum_k a_k cos(k k0 z + phi_k), for a controlled test of the
compiled ewald/pppm/kokkos IK pressure code on a non-trivial density.

Atoms are placed DETERMINISTICALLY: z_j = invCDF((j+0.5)/N) (so the empirical
z-density = rho(z), no shot noise -> the config's structure factors equal the
analytic ones, i.e. mean-field, no ideal-gas self term), and xy on a decoupled
grid so no two atoms overlap (the planar IK profile depends only on z).
"""
import numpy as np

# box from traj_cpp2.dump (sc 0.598, 10x10x20)
LX = LY = 1.1869514044823584e+01
LZ = 2.3739028089647167e+01
N = 2000
k0 = 2*np.pi/LZ
RHO0 = 0.5
MODES = [(1, 0.22, 0.0), (2, 0.11, 0.7), (3, 0.07, 1.9), (4, 0.04, 2.6)]

def rho(z):
    r = np.full_like(z, RHO0)
    for k, a, ph in MODES:
        r += a*np.cos(k*k0*z + ph)
    return r

# inverse-CDF sampling on a fine grid
zg = np.linspace(0.0, LZ, 200001)
pdf = rho(zg); pdf /= np.trapezoid(pdf, zg)
cdf = np.concatenate([[0.0], np.cumsum(0.5*(pdf[1:]+pdf[:-1])*np.diff(zg))])
cdf /= cdf[-1]
q = (np.arange(N)+0.5)/N
zj = np.interp(q, cdf, zg)               # z realizes rho(z)

# decoupled xy grid (45x45 cells, distinct cell per atom -> min xy ~0.26, no overlap)
ng = 45
ix = np.arange(N) % ng
iy = (np.arange(N)//ng) % ng
xj = (ix + 0.5)*(LX/ng)
yj = (iy + 0.5)*(LY/ng)

with open("traj_multimode.dump", "w") as f:
    f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n" % N)
    f.write("ITEM: BOX BOUNDS pp pp pp\n")
    f.write("0.0000000000000000e+00 %.16e\n" % LX)
    f.write("0.0000000000000000e+00 %.16e\n" % LY)
    f.write("0.0000000000000000e+00 %.16e\n" % LZ)
    f.write("ITEM: ATOMS id type x y z vx vy vz\n")
    for j in range(N):
        f.write("%d 1 %.8f %.8f %.8f 0 0 0\n" % (j+1, xj[j], yj[j], zj[j]))
print("wrote traj_multimode.dump  N=%d  rho range %.3f..%.3f" % (N, rho(zj).min(), rho(zj).max()))
