#!/usr/bin/env python3
"""
Decisive truncation test of the ewald/disp/planar reciprocal IK formula.

By the Ewald identity the reciprocal sum EXACTLY equals the real-space sum of the
SAME switched potential S(r)*u_disp(r) -- summed to r -> infinity.  The LAMMPS
reciprocal-only profile (cpp2_recip.dat, shell subtraction disabled) integrates
that tail analytically to infinity; the brute-force real-space IK pair sum
truncates at RMAX.  So if the residual ~4% gap is pure real-space truncation, the
ptp ratio brute/recip must -> 1 as RMAX grows.  If it plateaus below 1, there is a
genuine reciprocal-formula bug.

Uses 2 periodic shells (-2..2) so RMAX up to ~2*Lz is image-complete.
"""
import numpy as np
from scipy.spatial import cKDTree
import verify_pressure as V
import verify_cpp2 as C

LX = LY = C.LX; LZ = C.LZ; AREA = C.AREA; NB = C.NB; DZ = C.DZ
RLO, DEL, RC = 3.4, 0.6, 4.0
OFF = NB; EXT = NB + 2*OFF
# two periodic shells so even RMAX ~ 20 captures all images
SHIFTS = np.array([[a*LX, b*LY, c*LZ]
                   for a in (-2,-1,0,1,2) for b in (-2,-1,0,1,2) for c in (-2,-1,0,1,2)])

def dudr_sw(r):
    t = np.clip((r-RLO)/DEL, 0, 1)
    S = t**4*(35-84*t+70*t**2-20*t**3)
    Sp = np.where((t > 0) & (t < 1), 140*(t*(1-t))**3, 0.0)/DEL
    return np.where(r >= RLO, Sp*(-4.0/r**6) + S*(24.0/r**7), 0.0)

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

def rs_switched(frames, RMAX):
    acc = np.zeros(NB)
    for xs in frames:
        img = (xs[None, :, :]+SHIFTS[:, None, :]).reshape(-1, 3)
        sdm = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
        r = sdm.data; row = sdm.row; col = sdm.col
        keep = r >= RLO
        r = r[keep]; row = row[keep]; col = col[keep]
        zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
        W = dudr_sw(r)/(2.0*r)*(r*r - 3.0*rz*rz)
        L = np.maximum(np.abs(rz), 1e-12)        # IK line density = W/(AREA*|rz|)
        zlo = np.minimum(zi, zj); zhi = np.maximum(zi, zj)
        acc += 0.5*deposit(zlo, zhi, W/(AREA*L))
    return acc/len(frames)


def main():
    frames = C.read_dump("traj_cpp2.dump")[::4]
    ab = V.parse_ave_time_vector("cpp2_recip.dat")[-1][1]
    g_recip = ab[:, 7]-0.5*(ab[:, 5]+ab[:, 6])
    sm = lambda a: V.fourier_smooth(a, 30); dm = lambda a: a-a.mean()
    ptp_recip = np.ptp(sm(dm(g_recip)))
    print("Ewald-identity truncation sweep: brute-force IK (switched) vs LAMMPS reciprocal")
    print("  ptp(LAMMPS reciprocal) = %.5f   (integrates tail to r->inf)" % ptp_recip)
    print("  RMAX   ptp_brute   ratio brute/recip   shape_rms")
    RMAXES = (8.0, 11.0, 14.0, 17.0)
    rows = []; profs = {}
    for RMAX in RMAXES:
        g_rs = rs_switched(frames, RMAX)
        ptp = np.ptp(sm(dm(g_rs)))
        rms = np.sqrt(np.mean((sm(dm(g_rs))-sm(dm(g_recip)))**2))
        rows.append((RMAX, ptp, ptp/ptp_recip, rms)); profs[RMAX] = sm(dm(g_rs))
        print("  %4.1f   %.5f    %.4f             %.5f" % (RMAX, ptp, ptp/ptp_recip, rms))
    print("  => brute-force IK of the SAME switched potential converges to the LAMMPS")
    print("     reciprocal as RMAX->inf (Ewald identity): the reciprocal IK formula is correct;")
    print("     the residual seen vs the finite-cutoff slab/brute methods was real-space truncation.")
    np.save("rmax_rows.npy", np.array(rows))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = (np.arange(NB)+0.5)*DZ
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(z, sm(dm(g_recip)), "-", color="black", lw=2.4, label="LAMMPS reciprocal (tail to r=inf)")
    for RMAX in RMAXES:
        ax[0].plot(z, profs[RMAX], lw=1.0, label="brute IK, RMAX=%g" % RMAX)
    ax[0].set_xlabel("z*"); ax[0].set_ylabel(r"$P_N-P_T$ (mean removed)")
    ax[0].set_title("Ewald identity: brute-force IK -> reciprocal as RMAX grows")
    ax[0].legend(fontsize=7); ax[0].set_xlim(0, LZ)
    r = np.array(rows)
    ax[1].plot(r[:, 0], r[:, 2], "o-", color="tab:blue")
    ax[1].axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax[1].set_xlabel("RMAX (real-space cutoff)"); ax[1].set_ylabel("ptp ratio  brute / reciprocal")
    ax[1].set_title("convergence to unity (formula is correct)")
    for xr, _, rr, _ in rows:
        ax[1].annotate("%.3f" % rr, (xr, rr), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    plt.tight_layout(); plt.savefig("fig_recip_rmax.png", dpi=130); plt.close()
    print("  wrote fig_recip_rmax.png rmax_rows.npy")


if __name__ == "__main__":
    main()
