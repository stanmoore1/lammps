# ewald/disp/planar IK long-range profile: diagnosis, root cause, and fix

## Symptom
On the CPP 2 system (rcut=4.0) the long-range local surface tension
P_N^LR(z)-P_T^LR(z) for the **IK contour** disagreed between three methods:

| method | what it is |
|---|---|
| lattice | `ewald/disp/planar` `pressure_profile_long` (the new code) |
| slab    | Eq 4.18 IK analogue (Appendix A) on the measured rho(z) |
| real    | direct brute-force IK pair sum over the trajectory (r>4) |

slab and real agreed (rms 0.0006); the **lattice was the outlier** (rms 0.0022,
~25% high in the bulk dip), while the H contour (`compute stress/atom`) matched its
slab to 0.0005.

## Root cause (found by numerical reconstruction — `verify_ik_kernel.py`)
Two independent effects, separated by reconstructing the reciprocal double-sum in
Python from the structure factors S_n:

1. **(dominant) Harasima-distributed shell subtracted from the IK reciprocal.**
   `pressure_profile_long` builds the IK profile as `reciprocal - shell`.  The
   reciprocal Phi/Psi double-sum is IK-distributed, but `shell_profile_virial`
   localized the compact-switch shell virial **at the field point** (Harasima:
   `shellT[g]=dens[g]*sum_gp dens[gp] w(|g-gp|)`), not along the IK bond.  A direct
   test confirmed `NET - slab == shell_IK - shell_H` (rms of the shell-contour
   difference 0.00136 of the 0.00216 gap).  Subtracting an H-shaped shell from an
   IK-shaped reciprocal pulls the IK profile toward the H profile — exactly the
   observed signature.  The shell's **z-integral is contour-independent**, so this
   only distorts the SHAPE; the box-average (gamma) stays correctly pinned.

2. **(secondary, ~4%) off-diagonal amplitude.** After accounting for (1) a residual
   ~0.001 remains: the code's off-diagonal Phi/Psi double-sum has ~4% more
   z-amplitude than the real-space IK (ptp ratio slab/code = 0.96), **independent
   of kmax (K=40..200 identical)**, so it is not truncation.  This is a genuine but
   small discrepancy in the off-diagonal kernel/assembly — left for follow-up (see
   below); it does not affect gamma or the box average.

## Fix (implemented)
`src/KSPACE/ewald_disp_planar.cpp`, `shell_profile_virial` (both `corr bin` and
`corr raw`): spread each shell pair (g,gp)/(i,j) virial **uniformly in z along the
bond** connecting the two bins/atoms (IK contour) instead of localizing it at the
field bin.  The z-integral `sum_g shellT[g]` is unchanged, so
box-average(profile) == box pressure still holds (gamma unchanged).  Only
`shell_profile_virial` was touched, and it is called only from
`pressure_profile_long`, so forces, energy, the box pressure tensor, and the H
contour are bit-identical.

## Verification (CPP 2, `verify_cpp2.py`)
| IK lattice vs slab | rms |
|---|---|
| before (H-shell) | 0.00216 |
| after (IK-shell, corr bin) | 0.00102 |
| after (IK-shell, corr raw) | 0.00107 |

gamma_LR unchanged (0.1194), IK peak 0.0254 -> 0.0238 (toward slab/dissertation
0.0245); the IK panel of `fig_cpp2_fig47.png` now overlays slab + real-space.  The
remaining 0.001 is effect (2).

## Follow-up (not fixed here)
The residual ~4% off-diagonal amplitude (effect 2) is K-independent, so it is a
property of the closed-form coefficients
`CT=-6pi/H(ik_phi(hm)+ik_phi(hn))`, `CN=-12pi/H(ik_psi(hm)+ik_psi(hn))`
vs the real-space IK line integral.  Pinning it down requires re-deriving the
off-diagonal Fourier coefficient from the IK contour and comparing term-by-term
(the paper's published `N^IK_{n,m}=-96pi/H[...]` differs from the code by a
constant that I could only partly reconcile because of the S_n=sfac/V convention).
Recommended for the author; reproduce with `verify_ik_kernel.py`.
