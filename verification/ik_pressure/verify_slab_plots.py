#!/usr/bin/env python3
"""Generate the verification plots for the smooth-damped slab dispersion method
(ewald/disp/slab, pppm/disp/slab, pppm/disp/slab/kk) after the two IK-hook fixes."""
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import verify_pressure as V
import verify_cpp2 as C

DZ, AREA, LZ, NB, LX, LY = C.DZ, C.AREA, C.LZ, C.NB, C.LX, C.LY
z = (np.arange(NB)+0.5)*DZ
sm = lambda a: V.fourier_smooth(a, 25); dm = lambda a: a-a.mean()
def gIK(fn):
    ab = V.parse_ave_time_vector(fn)[-1][1]; return ab[:, 7]-0.5*(ab[:, 5]+ab[:, 6])

# ---- brute-force IK of u_smooth (geometric + LB), RMAX=14 ----
RCUT, DELTA = 4.0, 0.6
OFF = NB; EXT = NB+2*OFF
SH = np.array([[a*LX, b*LY, c*LZ] for a in (-2,-1,0,1,2) for b in (-2,-1,0,1,2) for c in (-1,0,1)])
def deposit(zlo, zhi, hp):
    D = np.zeros(EXT+2); plo = zlo/DZ+OFF; phi = zhi/DZ+OFF
    gl = np.floor(plo).astype(int); fl = plo-gl; gh = np.floor(phi).astype(int); fh = phi-gh
    np.add.at(D, gl, hp*(1-fl)); np.add.at(D, gl+1, hp*fl)
    np.add.at(D, gh, -hp*(1-fh)); np.add.at(D, gh+1, -hp*fh)
    P = np.cumsum(D)[:EXT]; pr = np.zeros(NB)
    for k in range(0, EXT, NB):
        s = P[k:k+NB]; pr[:len(s)] += s
    return pr
def brute_geo(frames, RMAX=14.0):
    acc = np.zeros(NB)
    for xs in frames:
        img = (xs[None, :, :]+SH[:, None, :]).reshape(-1, 3)
        sd = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
        r = sd.data; row = sd.row; col = sd.col; k = r >= RCUT; r = r[k]; row = row[k]; col = col[k]
        t = np.clip((r-RCUT)/DELTA, 0, 1); S = t**4*(35-84*t+70*t**2-20*t**3)
        Sp = np.where((t > 0) & (t < 1), 140*(t*(1-t))**3, 0.0)/DELTA
        dudr = Sp*(-4.0/r**6) + S*(24.0/r**7)
        zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
        W = dudr/(2*r)*(r*r-3*rz*rz); L = np.maximum(np.abs(rz), 1e-12)
        acc += 0.5*deposit(np.minimum(zi, zj), np.maximum(zi, zj), W/(AREA*L))
    return acc/len(frames)

frames = C.read_dump("traj_cpp2.dump")[::4]
ge = gIK("slab_ikLR_ewald.dat"); gp = gIK("slab_ikLR_pppm.dat"); gk = gIK("slab_ikLR_kk.dat")
gb = brute_geo(frames)
hb = V.parse_ave_chunk("slab_hLR_pppm.dat")[-1][1]; rho = np.interp(z, hb[:, 0], hb[:, 1]/(AREA*0.06), period=LZ)

# === Figure 1: three-way cross-code IK + brute + density ===
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(z, sm(rho), color="gray", lw=1.4)
ax[0].set_xlabel("z*"); ax[0].set_ylabel(r"$\rho(z)$"); ax[0].set_title("CPP2 density"); ax[0].set_xlim(0, LZ)
ax[1].plot(z, sm(ge), "-", color="green", lw=3, label="ewald/disp/slab (Fix 1)")
ax[1].plot(z, sm(gp), "--", color="black", lw=1.6, label="pppm/disp/slab")
ax[1].plot(z, sm(gk), ":", color="red", lw=1.8, label="pppm/disp/slab/kk (Fix 2)")
ax[1].plot(z, sm(gb), "-.", color="purple", lw=1.2, label="brute IK of u_smooth (RMAX=14)")
ax[1].set_xlabel("z*"); ax[1].set_ylabel(r"$P_N^{LR}-P_T^{LR}$"); ax[1].set_xlim(0, LZ)
ax[1].set_title("IK contour: ewald = pppm = kk = real-space"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_slab_crosscode.png", dpi=130); plt.close()

# === Figure 2: LB (arithmetic) mixing IK vs brute ===
gl = gIK("slab_arithfluid_pppm.dat")
d = np.load("arith_config.npz"); typ2 = np.array([((i+1) % 2)+1 for i in range(frames[0].shape[0])])
eps = {1: 1.0, 2: 1.5}; sig = {1: 1.0, 2: 1.1}; ntyp = np.tile(typ2, len(SH))
def brute_lb(RMAX=14.0):
    acc = np.zeros(NB)
    for xs in frames:
        img = (xs[None, :, :]+SH[:, None, :]).reshape(-1, 3)
        sd = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
        r = sd.data; row = sd.row; col = sd.col; k = r >= RCUT; r = r[k]; row = row[k]; col = col[k]
        ti = typ2[row]; tj = ntyp[col]
        C6 = 4*np.sqrt(np.array([eps[t] for t in ti])*np.array([eps[t] for t in tj])) * \
            (0.5*(np.array([sig[t] for t in ti])+np.array([sig[t] for t in tj])))**6
        t = np.clip((r-RCUT)/DELTA, 0, 1); S = t**4*(35-84*t+70*t**2-20*t**3)
        Sp = np.where((t > 0) & (t < 1), 140*(t*(1-t))**3, 0.0)/DELTA
        dudr = Sp*(-C6/r**6) + S*(6*C6/r**7)
        zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
        W = dudr/(2*r)*(r*r-3*rz*rz); L = np.maximum(np.abs(rz), 1e-12)
        acc += 0.5*deposit(np.minimum(zi, zj), np.maximum(zi, zj), W/(AREA*L))
    return acc/len(frames)
glb = brute_lb()
plt.figure(figsize=(7.5, 4.6))
plt.plot(z, sm(gl), "-", color="darkgreen", lw=2.6, label="pppm/disp/slab IK (LB, nchan=7)")
plt.plot(z, sm(glb), "--", color="red", lw=1.4, label="brute IK of mixed u_smooth (RMAX=14)")
plt.xlabel("z*"); plt.ylabel(r"$P_N^{LR}-P_T^{LR}$"); plt.xlim(0, LZ)
plt.title("Lorentz-Berthelot mixing: IK contour vs real-space"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_slab_lb.png", dpi=130); plt.close()
print("wrote fig_slab_crosscode.png fig_slab_lb.png")
print("  IK gamma ewald/pppm/kk = %.4f/%.4f/%.4f ; brute(RMAX14)=%.4f" % (
    0.5*DZ*ge.sum(), 0.5*DZ*gp.sum(), 0.5*DZ*gk.sum(), 0.5*DZ*gb.sum()))
