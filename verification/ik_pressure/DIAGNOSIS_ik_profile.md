# Diagnosis: ewald/disp/planar IK long-range profile disagrees with independent IK methods

## Symptom
On the CPP 2 system (rcut=4.0), the long-range local surface tension
P_N^LR(z)−P_T^LR(z) computed three ways for the **IK contour**:

| method | what it is | γ_LR |
|---|---|---|
| **lattice** | `ewald/disp/planar` `pressure_profile_long` (Φ/Ψ double sum) | 0.1194 |
| **slab** | Eq 4.18 IK analogue (Appendix A), evaluated on ρ(z) | 0.1167 |
| **real-space** | direct brute-force IK pair sum over the trajectory (r>4) | 0.1133 |

Pairwise rms of the (Fourier-smoothed) profiles:

```
H  lattice-vs-slab  = 0.00046     <- H contour is fine
IK lattice-vs-slab  = 0.00217     <- lattice IK is the outlier
IK lattice-vs-real  = 0.00271
IK slab-vs-real     = 0.00063     <- the two independent IK methods AGREE
```

The lattice IK sits ~25% high in the bulk dip (lattice 0.0115 vs slab/real 0.0092)
and is visibly pulled toward the H profile (H dip ≈ 0.018). See
`fig_cpp2_fig47.png` (right panel).

## What it is NOT
- **Not statistics** — profiles are block-averaged and Fourier-smoothed; the gap is
  smooth and systematic.
- **Not correlations** — the mean-field slab (g=1) and the correlated brute-force
  real-space agree to 0.0006, so correlations are negligible for the IK shape; the
  lattice (which also uses the actual structure factors) should therefore match
  them, but doesn't.
- **Not the switch/shell** — rerunning the same trajectory at Δ=0.4 vs Δ=0.6 leaves
  the lattice IK essentially unchanged (γ 0.1194 both; bulk dip 0.0115 vs 0.0117).
  So the compact-switch shell correction is not the cause.
- **Not the box-average** — ⟨P_N−P_T⟩ (γ) is pinned correctly; the H and IK
  box-averages match the global pressure.
- **Not the H path** — `compute stress/atom` (per-atom kspace virial, H contour)
  matches its slab to 0.0005.

## Where it is localized
The discrepancy is in the **off-diagonal (p=n+m≠0) Φ/Ψ shape terms** of
`EwaldDispPlanar::pressure_profile_long` — the p=0 (box-average) terms are pinned
and correct, the H per-atom path is correct, and both P_N (Ψ) and P_T (Φ)
off-diagonal components carry the error (~0.0025 each).

## Suspected cause (for the author to confirm)
The off-diagonal coefficients (`ewald_disp_planar.cpp` ~1707):
```
CT = -6.0*MY_PI/H * (ik_phi(hm) + ik_phi(hn));
CN = -12.0*MY_PI/H * (ik_psi(hm) + ik_psi(hn));
```
have **no explicit `volume` factor**, whereas the diagonal coefficients on the
adjacent lines do:
```
CT = CN = volume * GT[0];                 // (0,0)
CT = 0.5*volume*GT[kk]; CN = 0.5*volume*GN[kk];   // n=-m diagonal
```
With the structure factors normalized as `S_n = sfac/volume`, the diagonal terms
scale like 1/V while the bare off-diagonal terms scale like 1/V² — an apparent
normalization asymmetry between the diagonal and off-diagonal blocks.  Separately,
matching the paper's published `N^IK_{n,m} = -96π/(h_n+h_m)[...]` to the code's
`CN = -12π/H[ik_psi(hm)+ik_psi(hn)]` leaves a constant factor (the code is 1/8 of
the paper bracket) that I could not reconcile with the S_n=sfac/V convention.

Because the *box-average* (diagonal) is pinned independently, a wrong off-diagonal
normalization would distort only the **shape** while preserving γ — exactly the
observed signature.

**Caveat:** the measured discrepancy is only ~10–25% of the local signal, *not*
the factor-8 or factor-V that a literal missing factor would produce — so if a
normalization factor is involved it must be largely compensated elsewhere, and the
true cause may instead be a subtler shape error (a single term, sign, or argument
in the off-diagonal Φ/Ψ assembly, or the diagonal↔off-diagonal continuity).  The
empirical localization (off-diagonal shape terms; not box-average / correlations /
switch / H-path) is solid; the specific root cause above is a lead, not a
conclusion.  Recommended concrete check: the m→−n limit of the off-diagonal CT/CN
should reduce continuously to the diagonal coefficient (0.5·volume·GT[k],
0.5·volume·GN[k]); a mismatch there would pinpoint the inconsistency.

## How to reproduce
```
python3 verify_cpp2.py     # prints the rms table above; fig_cpp2_fig47.png
```
(uses cpp2_ikLR.dat = lattice IK, cpp2_hLR.dat = lattice H + density, traj_cpp2.dump)
