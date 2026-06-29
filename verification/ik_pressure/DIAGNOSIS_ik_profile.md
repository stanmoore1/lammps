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

2. **(secondary, ~4%) off-diagonal amplitude — later shown to be a verification
   artifact, NOT a code bug.** After accounting for (1) a residual ~0.001 remained:
   the code's off-diagonal Phi/Psi double-sum had ~4% more z-amplitude than the
   *finite-cutoff* real-space IK (ptp ratio slab/code = 0.96).  It is k-independent
   in the LATTICE (K=40..200 identical) because it is not a lattice truncation — it
   is the **real-space** side that was truncated: the slab `Gkernel` integrates the
   `1/r^6` tail only to rmax=14 and the brute-force sum only to RMAX~11-12, while
   the reciprocal integrates to infinity.  Extending the real-space cutoff closes
   the gap to zero (see "Follow-up — RESOLVED" below).  It never affected gamma or
   the box average.

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

## Follow-up — RESOLVED: the off-diagonal coefficient is correct; the residual was truncation

The residual ~4% (effect 2) was chased down and is **not** a code bug.  Two
independent checks settle it.

### (a) Analytic — off-diagonal coefficient re-derived from scratch
Re-deriving the off-diagonal `P_N-P_T` Fourier coefficient directly from the IK
contour gives
`C^{N-T}_{n,m} = (pi/H) [ J(h_n) + J(h_m) ]`, `J(h) = 24*ik_phi(h) - 48*ik_psi(h)`,
which is exactly the code's assembly
`CN-CT = -12pi/H(ik_psi(hm)+ik_psi(hn)) + 6pi/H(ik_phi(hm)+ik_phi(hn))`
(the factor 4 = B^2).  Term-by-term ratio `J / (24 ik_phi - 48 ik_psi) = 0.99998`.
A single-cosine density test confirmed the Python reconstruction of the code's
double-sum (`verify_ik_kernel.offdiag_shape`) reproduces the hand-derived analytic
amplitudes EXACTLY at every mode, while the finite-cutoff slab (`slab_IK`,
`Gkernel` rmax=14) sat a *uniform* 0.924x below — a constant scaling, i.e. a
property of the verification slab, not a per-mode formula error.

### (b) Numerical — Ewald-identity truncation sweep (`verify_recip_rmax.py`)
The decisive test.  By the Ewald identity the reciprocal sum equals the
real-space sum of the SAME switched potential `S(r) u_disp(r)` summed to
r -> infinity.  Running the LAMMPS reciprocal-only IK (shell subtraction
disabled, `cpp2_recip.dat`) against a brute-force IK pair sum of the identical
switched potential, sweeping the real-space cutoff RMAX:

| RMAX | ptp ratio brute/recip | shape rms |
|---|---|---|
| 8  | 0.739  | 0.00353 |
| 11 | 0.909  | 0.00193 |
| 14 | 0.973  | 0.00109 |
| 17 | 1.0001 | 0.00066 |
| 20 | 1.013  | 0.00048 |

The brute-force IK converges to the LAMMPS reciprocal as RMAX grows (ratio -> 1,
shape rms -> 0).  The reciprocal integrates the `1/r^6` tail analytically to
infinity; the slab (rmax=14) and the earlier brute-force (RMAX=11-12) truncated
it, which is exactly the ~0.92-0.97 undershoot previously attributed to the code.
At RMAX=14 the ratio is 0.973 — matching the slab's residual — and it closes to
<0.0007 rms by RMAX=17.  See `fig_recip_rmax.png`.

### (c) Cheap, essentially-exact confirmation (`verify_cosine_exact.py`)
The trajectory brute force is expensive and never reaches exact agreement because
of the finite real-space cutoff.  A purely analytic single-cosine density
`rho(z) = 0.5 + 0.4 cos(2 pi z/Lz)` removes statistics, the trajectory, and the
KDTree entirely: the structure factors are nonzero only for `n = 0, +-1`, so the
lattice off-diagonal double sum is a tiny EXACT sum evaluated with the code's own
`ik_phi/ik_psi` (tail to r=inf), while the slab side is a 1-D quadrature whose only
approximation is the kernel cutoff `rmax`:

| rmax | ptp ratio slab/lattice | shape rms |
|---|---|---|
| 14  | 0.9241 | 8.5e-4 |
| 20  | 0.9799 | 2.2e-4 |
| 40  | 0.9992 | 9e-6 |
| 80  | 1.0002 | 2e-6 |
| 160 | 1.0001 | 1e-6 |

As `rmax -> inf` the slab matches the LAMMPS lattice off-diagonal kernel to ~6
significant figures (rms ~1e-6).  The 0.924 at rmax=14 is exactly the same
truncation seen in the slab `Gkernel`.  This is the decisive, cheap proof that the
off-diagonal IK coefficient in `ewald/disp/planar` equals the real-space IK
integral exactly.  See `fig_cosine_exact.png`.

### Conclusion
After the shell-contour fix above, the `ewald/disp/planar` IK long-range pressure
profile is **correct**: diagonal pinned to the box pressure, off-diagonal kernel
verified analytically and by the Ewald identity, shell now IK-distributed.  The
remaining lattice-vs-slab rms (~0.001) is the verification slab's finite r-cutoff,
not the LAMMPS code.  Reproduce with `verify_recip_rmax.py` (sweep) and
`verify_ik_kernel.py` (analytic single-cosine check).
