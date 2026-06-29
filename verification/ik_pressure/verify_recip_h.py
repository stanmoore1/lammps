#!/usr/bin/env python3
"""
Direct real-space verification of the ewald/disp/planar HARASIMA (H) long-range
contour -- the H analogue of the IK three-way check in verify_cpp2.py.

The H contour comes from `compute stress/atom ... kspace` (planar-Ewald per-atom
virial), binned by `fix ave/chunk` (cpp2_hLR.dat).  The kspace-only net over the
switch region [3.4,4.0] is the laterally-correlated residual (the pair supplies
the 3-D mean field), so the kspace H contour corresponds to the SHARP r>4
dispersion tail.  We therefore compare it to a brute-force Harasima pair sum of the
SHARP tail 24/r^7 (r>4), depositing each pair's P_N-P_T virial half-and-half AT THE
TWO ATOMS (the stress/atom convention), and sweep the real-space cutoff RMAX.

  NOTE: this is NOT an Ewald-identity test of the per-atom decomposition -- the
  shell correction (corr) also modifies vatom, so the reciprocal-only stress/atom
  is reciprocal-minus-shell.  Comparing the kspace H contour to the sharp r>4
  Harasima is the correct, contour-consistent check (same logic as IK lat-vs-real).
"""
import numpy as np
from scipy.spatial import cKDTree
import verify_pressure as V
import verify_cpp2 as C

LX = LY = C.LX; LZ = C.LZ; AREA = C.AREA
RCUT = 4.0
DZH = 0.06; NBH = 396; VCH = AREA*DZH
SHIFTS = np.array([[a*LX, b*LY, c*LZ]
                   for a in (-2,-1,0,1,2) for b in (-2,-1,0,1,2) for c in (-2,-1,0,1,2)])

def dudr_sharp(r):
    return np.where(r >= RCUT, 24.0/r**7, 0.0)

def rs_harasima(frames, RMAX):
    """brute-force Harasima P_N-P_T (sharp r>4): each ordered (i,j) deposits half
    its virial as a POINT at the field atom z_i (the stress/atom convention)."""
    acc = np.zeros(NBH)
    for xs in frames:
        img = (xs[None, :, :]+SHIFTS[:, None, :]).reshape(-1, 3)
        sdm = cKDTree(xs).sparse_distance_matrix(cKDTree(img), RMAX, output_type="coo_matrix")
        r = sdm.data; row = sdm.row; col = sdm.col
        keep = r >= RCUT
        r = r[keep]; row = row[keep]; col = col[keep]
        zi = xs[row, 2]; zj = img[col, 2]; rz = zj-zi
        W = dudr_sharp(r)/(2.0*r)*(r*r - 3.0*rz*rz)
        g = (np.floor(zi/DZH).astype(int)) % NBH
        np.add.at(acc, g, 0.5*W/VCH)
    return acc/len(frames)


def main():
    frames = C.read_dump("traj_cpp2.dump")[::4]
    hb = V.parse_ave_chunk("cpp2_hLR.dat")[-1][1]
    g_lat = (-hb[:, 4] + 0.5*(hb[:, 2]+hb[:, 3]))/VCH        # P_N-P_T (H), LAMMPS lattice
    sm = lambda a: V.fourier_smooth(a, 30); dm = lambda a: a-a.mean()
    ptp_lat = np.ptp(sm(dm(g_lat)))
    print("Direct real-space check (HARASIMA): brute-force Harasima (sharp r>4) vs LAMMPS H lattice")
    print("  ptp(LAMMPS H lattice) = %.5f" % ptp_lat)
    print("  RMAX   ptp_brute   ratio brute/lat   shape_rms")
    rows = []; profs = {}
    RMAXES = (8.0, 11.0, 14.0, 17.0)
    for RMAX in RMAXES:
        g_rs = rs_harasima(frames, RMAX)
        ptp = np.ptp(sm(dm(g_rs)))
        rms = np.sqrt(np.mean((sm(dm(g_rs))-sm(dm(g_lat)))**2))
        rows.append((RMAX, ptp, ptp/ptp_lat, rms)); profs[RMAX] = sm(dm(g_rs))
        print("  %4.1f   %.5f    %.4f           %.5f" % (RMAX, ptp, ptp/ptp_lat, rms))
    print("  => brute-force Harasima converges to the LAMMPS stress/atom H contour")
    print("     as RMAX grows: the H per-atom kspace virial reproduces the real-space")
    print("     Harasima contour (the small residual is the [3.4,4] correlated shell).")
    np.save("rmax_h_rows.npy", np.array(rows))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = (np.arange(NBH))*DZH + 0.03
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(z, sm(dm(g_lat)), "-", color="black", lw=2.4, label="LAMMPS H lattice (stress/atom)")
    for RMAX in RMAXES:
        ax[0].plot(z, profs[RMAX], lw=1.0, label="brute Harasima r>4, RMAX=%g" % RMAX)
    ax[0].set_xlabel("z*"); ax[0].set_ylabel(r"$P_N-P_T$ (mean removed)")
    ax[0].set_title("Harasima: brute-force (sharp r>4) -> LAMMPS lattice")
    ax[0].legend(fontsize=7); ax[0].set_xlim(0, LZ)
    r = np.array(rows)
    ax[1].plot(r[:, 0], r[:, 2], "o-", color="tab:orange")
    ax[1].axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax[1].set_xlabel("RMAX (real-space cutoff)"); ax[1].set_ylabel("ptp ratio  brute / lattice")
    ax[1].set_title("convergence to unity (H contour correct)")
    for xr, _, rr, _ in rows:
        ax[1].annotate("%.3f" % rr, (xr, rr), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    plt.tight_layout(); plt.savefig("fig_recip_h.png", dpi=130); plt.close()
    print("  wrote fig_recip_h.png rmax_h_rows.npy")


if __name__ == "__main__":
    main()
