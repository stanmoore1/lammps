#!/usr/bin/env python3
# 2-type deterministic multimode config (Lorentz-Berthelot dispersion path).
# Writes a LAMMPS data file (for read_data + run 0) and an npz (for the brute force).
import numpy as np
LX = LY = 1.1869514044823584e+01; LZ = 2.3739028089647167e+01
N = 2000; k0 = 2*np.pi/LZ; RHO0 = 0.5
MODES = [(1,0.22,0.0),(2,0.11,0.7),(3,0.07,1.9),(4,0.04,2.6)]
def rho(z):
    r = np.full_like(z, RHO0)
    for k,a,ph in MODES: r += a*np.cos(k*k0*z+ph)
    return r
zg = np.linspace(0,LZ,200001); pdf = rho(zg); pdf/=np.trapezoid(pdf,zg)
cdf = np.concatenate([[0],np.cumsum(0.5*(pdf[1:]+pdf[:-1])*np.diff(zg))]); cdf/=cdf[-1]
zj = np.interp((np.arange(N)+0.5)/N, cdf, zg)
ng=45; ix=np.arange(N)%ng; iy=(np.arange(N)//ng)%ng
xj=(ix+0.5)*(LX/ng); yj=(iy+0.5)*(LY/ng)
typ = 1 + (np.arange(N)%2)
with open("data.arith","w") as f:
    f.write("2-type multimode config\n\n%d atoms\n2 atom types\n\n"%N)
    f.write("0 %.16e xlo xhi\n0 %.16e ylo yhi\n0 %.16e zlo zhi\n\n"%(LX,LY,LZ))
    f.write("Masses\n\n1 1.0\n2 1.0\n\nAtoms\n\n")
    for j in range(N):
        f.write("%d %d %.8f %.8f %.8f\n"%(j+1,typ[j],xj[j],yj[j],zj[j]))
np.savez("arith_config.npz", xyz=np.column_stack([xj,yj,zj]), typ=typ)
print("wrote data.arith + arith_config.npz  types:", np.bincount(typ)[1:])
