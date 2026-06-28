#!/usr/bin/env python3
"""
Reproduce dissertation Fig 4.5: the local mechanical-stability residual.
Eq 3.16 reads dP_N(z)/dz = rho(z) f_ext(z); our slab has NO external field
(f_ext=0), so the condition is simply dP_N/dz = 0.  We plot dP_N/dz for the IK
contour (compute stress/cartesian) and the H contour (compute stress/atom).
The IK contour satisfies local mechanical stability (dP_N/dz ~ 0); the H contour
violates it (systematic oscillations at the interfaces).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import verify_pressure as V

DZ = 0.1

def smooth(a, w=7):
    k = np.ones(w)/w
    return np.convolve(np.r_[a[-(w//2):], a, a[:w//2]], k, mode="valid")[:len(a)]

ik = V.load_IK("ik_profile.dat"); h = V.load_H("har_profile.dat")
z = ik["z"]
# full normal pressure P_N(z), lightly smoothed (same filter for both contours)
PN_ik = smooth(ik["PN"]); PN_h = smooth(h["PN"])
# periodic derivative
dPN_ik = (np.roll(PN_ik, -1) - np.roll(PN_ik, 1)) / (2*DZ)
dPN_h  = (np.roll(PN_h,  -1) - np.roll(PN_h,  1)) / (2*DZ)

# restrict the max-error metric to the interface+liquid region (density-supported)
d = ik["dens"]; supp = d > 0.05*d.max()
print("Fig 4.5: mechanical-stability residual dP_N/dz  (f_ext = 0)")
print("  max|dP_N/dz|  IK = %.4f   H = %.4f   (H/IK = %.1fx)"%(
    np.max(np.abs(dPN_ik[supp])), np.max(np.abs(dPN_h[supp])),
    np.max(np.abs(dPN_h[supp]))/max(np.max(np.abs(dPN_ik[supp])),1e-9)))

plt.figure(figsize=(8,4.5))
plt.plot(z, dPN_ik, "-", color="red", lw=1.4, label="IK")
plt.plot(z, dPN_h, "--", color="blue", lw=1.1, label="H")
plt.axhline(0, color="0.6", lw=0.6)
plt.xlabel("z*"); plt.ylabel(r"Error in $dP_N^*/dz^*$")
plt.title("Mechanical stability (Eq 3.16, $f_{ext}=0$): IK vs H contour")
plt.legend(); plt.tight_layout(); plt.savefig("fig45_reproduction.png", dpi=130); plt.close()
print("wrote fig45_reproduction.png")
