# Review: pppm/dipole charge–dipole support (branch `pppm-charge-dipole`)

Scope reviewed: commits `001fde9c`, `d5e848f0`, `055b29e5`, `2a41d90a` on top of
`16f8568c`, i.e. `src/KSPACE/pppm_dipole.{cpp,h}`, the new example
`examples/dipole/in.charge_dipole`, and the doc note.

Reference for the **charge–dipole** methodology is `ewald/disp`
(`src/KSPACE/ewald_disp.cpp`), since `ewald/dipole` explicitly forbids charges
(`ewald_dipole.cpp:76` "Cannot (yet) use charges with Kspace style
EwaldDipole"). For the **dipole–dipole** terms the reference is the
pre-existing `ewald/dipole` / original `pppm/dipole`.

## Verdict

The new charge–dipole methodology is **correct and matches `ewald/disp`**, both
analytically (term-by-term) and numerically. Per-atom energy and virial for
dipole–dipole **and** charge–dipole are correct (sum exactly to the global
values, which are themselves FD-verified). The solver converges to the
requested RMS accuracy. No correctness bugs found; minor notes at the end.

---

## 1. Methodology comparison vs `ewald/disp`

### 1.1 Structure factors / FFT sign convention
- `ewald/disp` (`eik_dot_r`) builds, with `e^{+ik·r}`:
  - charge:   `S_q(k)  = Σ_j q_j e^{ik·r_j}`
  - dipole:   `S_μ(k)  = Σ_j (μ_j·k) e^{ik·r_j}`   (`muk = mui·hvec`, `hvec = k`)
- `pppm/dipole` spreads `q` and the dipole vector `μ` onto grids
  (`make_rho_dipole`) and FFTs them. LAMMPS `FFT3d::FORWARD` uses `e^{-ik·r}`,
  so the PPPM structure factors are the **complex conjugates** of the Ewald
  ones: `Q = conj(S_q)`, `P = conj(S_μ)` (real parts equal, imaginary parts
  negated). `D ≡ k·P` is the PPPM dipole structure factor.

This conjugate convention is the key to checking every cross-term sign below.

### 1.2 Reciprocal energy (poisson_ik_dipole vs compute_energy)
PPPM per-k energy density (`pppm_dipole.cpp:1473-1476, 1505-1507`):
```
eng = gqq·|Q|²  +  gdd·|D|²  +  2·gqm·(Qr·Di − Qi·Dr)
```
`ewald/disp` (`ewald_disp.cpp:1015-1040`):
```
E ∝ kqq·|S_q|² + kdd·|S_μ|² + 2·kdd·(S_μ.re·S_q.im − S_μ.im·S_q.re)
```
The cross numerators are **identical** after the conjugate convention:
```
(Qr·Di − Qi·Dr) = (S_q.im·S_μ.re − S_q.re·S_μ.im)
                = (S_μ.re·S_q.im − S_μ.im·S_q.re)   ✓  (same sign)
```
i.e. `2·gqm·Im(conj(Q)·D) == 2·k·Im(conj(S_μ)·S_q)`. The `+` sign is correct.

### 1.3 The three Green's functions (compute_gf_dipole)
A single aliasing sweep builds three influence functions differing only by the
power `p` of `(k·k_n)` in the numerator and `1/k²` in the denominator
(`pppm_dipole.cpp:1138-1148`):
- `greensfn_qq`  (p=1) — charge–charge. This is **exactly** the base-PPPM
  ik-optimal influence function `Σ (k·k_n)(4π S/k_n²)W² / [k²(ΣW²)²]`
  (cf. base `pppm.cpp` `compute_gf_ik`).
- `greensfn_qmu` (p=2) — charge–dipole cross.
- `greensfn`     (p=3) — dipole–dipole (matches the original dipole-only code).

Each additional dipole multiplies the structure factor by one extra `(k·μ)`,
hence one extra `(k·k_n)` / `1/k²`. In the continuum limit (`W→1`, no aliasing,
`k_n→k`) all three reduce to the common Ewald kernel `4π e^{-k²/4g²}/k²`, so the
charge channel is bit-for-bit the standard PPPM kernel and the cross/dipole
channels are its PPPM-optimized generalizations. Consistent with `ewald/disp`,
which uses the same `kenergy = 2(2π/V)e^{-k²/4g²}/k²` for all dipole channels.

### 1.4 Fields / forces (poisson_ik_dipole + fieldforce_ik_dipole)
- charge E-field `vd{xyz}_brick`:   `u_a = i k_a gqq Q + k_a gqm D`
- dipole E-field `u{xyz}_brick_dipole`: `u_a = k_a gdd D + i k_a gqm Q`
- dipole field-gradient `vd{ab}_brick_dipole`: `i k_a k_b gdd D − k_a k_b gqm Q`

`fieldforce_ik_dipole` then applies `f = q·E_charge + (∇E)·μ`, torque `μ×E`.
This is the exact analog of `ewald/disp::compute_force`, where the charge feels
the dipole field (`ncoul` cross block) and the dipole feels the charge field.
Charge feels `−i gqm D`, dipole feels `+i gqm Q` — the antisymmetric pairing
required for Newton's third law and a single shared `gqm`. ✓

### 1.5 Self-energy and net-charge correction (compute(), :467-510)
```
E -= musqsum·2g³/(3√π)   (dipole self)
E -= g·qsqsum/√π          (charge self)
E -= (π/2)·qsum²/(g²V)     (net-charge / k=0)
```
These match `ewald/disp` (`energy_self[DIPOLE]` at :673-674) and base
`ewald`/`pppm` (charge self + net charge). There is **no** charge–dipole self
term — correct: the on-site `q_i–μ_i` interaction is odd in `r` and vanishes,
and neither code subtracts one.

### 1.6 Slab correction (slabcorr, commit 055b29e5)
Combined first moment `M_z = Σ(q_i z_i + μ_iz)` with
`E = (2π/V)[M_z² − qsum·R2 − qsum²Lz²/12]` (Yeh–Berkowitz + Ballenegger
non-neutral form). The previous dipole-only `μ²/12` self term is removed (the
comment notes it was inconsistent with the `−4π/V` field); the combined moment
with no `1/12` on `M_z²` is consistent with the field felt by both species, the
`−4π/V·q(M_z − qsum·z)` force on charges, and the `±ffact·M_z·μ` torque on
dipoles. Reviewed analytically; matches the standard EW3DC generalization.
(Not exercised by the example, which is 3-D periodic.)

---

## 2. Per-atom energy & virial

### 2.1 Per-atom energy (fieldforce_peratom_dipole, poisson_peratom_dipole)
`eatom_i = μ_i·E_dipole + q_i·φ_q`, with the charge potential built in
`poisson_peratom_dipole`: `φ_q = gqq Q − i gqm D` (`:1876-1877`) and the dipole
"potential" reusing `u{xyz}_brick_dipole = gdd D + i gqm Q`. The charge picks up
`−i gqm D`, the dipole `+i gqm Q`; summed they reproduce the factor-2 cross term
of the global energy. Mirrors `ewald/disp::compute_energy_peratom` (the `ncoul`
cross block, :1109-1114).

### 2.2 Per-atom virial
Charge virial bricks `v{0..5}_brick` and dipole virial bricks
`v{0..5}{x,y,z}_brick_dipole` carry the `vg`·potential term plus the
structure-factor "strain" terms (`2 gdd k_b P_a` and `2 gqm k_b P_a`,
`:1927-1928, 1959-1960`). The `Pw[6]={w1,w2,w3,w1,w1,w2}` / `kb` mapping
correctly selects `(P_a, k_b)` for each Voigt component `xx,yy,zz,xy,xz,yz`.
This is the PPPM analog of `ewald/disp::compute_virial` +
`compute_virial_dipole` + `compute_virial_peratom`.

Note: dipole–dipole per-atom energy/virial were **disabled** in the old code;
they are newly enabled here and validated below.

---

## 3. Numerical validation (built serial, KISS FFT)

System: 216-particle sc lattice, alternating ±0.5 charges + random unit
dipoles, `lj/cut/dipole/long 4.0`. Cross-checked against `ewald/disp`
(`lj/long/dipole/long cut long 4.0`) and `ewald/dipole`.

**A. Cross-check elong (single point, tight 1e-6):**
- charge+dipole: `pppm/dipole` elong = −0.127587, `ewald/disp` = −0.125799
  (≈1.4 %, consistent with the different real-space pair formulations).
- dipole-only: `pppm/dipole` = −0.0581764 vs `ewald/dipole`/`ewald/disp`
  = −0.0580709 (≈0.2 %).

**B. Forces by central finite-difference of total PE** (`norm no`):
| atom/comp | reported f | −dE/dx (FD) |
|-----------|-----------:|------------:|
| 1 fx | −0.02926504 | −0.02926623 |
| 1 fy | −0.70741089 | −0.70741016 |
Match to FD truncation error. (The naïve mismatch I first saw was the
`units lj` default `thermo_modify norm yes`, which reports PE/elong **per
atom**, i.e. ÷N=216.)

**C. Total virial by affine-strain FD of total PE** (pinned g_ewald):
trace(virial) FD = −1704.666 vs reported −1704.664 — match to 5+ digits,
including all charge–dipole cross contributions.

**D. Per-atom ↔ global consistency** (combined system, `norm no`):
```
elong = Σ eatom_i = −27.55886                       (energy)
global vir (xx,yy,zz) = (−63.470, 3.550, 7.831)
Σ vatom_i  (xx,yy,zz) = (−63.470, 3.550, 7.831)     (all 6 components match)
```
Same exact consistency holds for charge-only and dipole-only; dipole-only
per-atom virial also matches `ewald/dipole`.

**E. Convergence — measured RMS force vs requested accuracy** (vs 1e-6 ref):
| requested rel. acc. | printed est. abs RMS | measured RMS | max|Δf| |
|--------------------:|---------------------:|-------------:|--------:|
| 1e-3 | 9.0e-4 | 3.3e-4 | 1.1e-3 |
| 1e-4 | 1.3e-4 | 5.0e-5 | 1.6e-4 |
| 1e-5 | 9.8e-6 | 1.8e-6 | 5.7e-6 |
Measured error is below the estimate and scales ~10× per decade — the solver
converges to (and slightly inside) the requested RMS relative accuracy.
(`two_charge_force ≈ 1` in these lj units, so abs ≈ rel.)

---

## 4. Minor notes (not blocking)

1. `find_gewald_dipole` / `newton_raphson_f_dipole` build the g_ewald
   *initial estimate* from the **dipole** real-space error only (no charge
   term in the objective), whereas `newton_raphson_f()` (used by
   `adjust_gewald`) and `compute_df_kspace_dipole` do add the charge error in
   quadrature. For strongly charge-dominated systems the initial g_ewald guess
   may be slightly off, but `adjust_gewald` + grid search recover it
   (convergence test in §3E passed). Consider adding the charge term to the
   initial objective for consistency.
2. `musum_musq()` raises a hard error when `mu2 == 0`, so `pppm/dipole` cannot
   run a pure-charge system (intended — use `pppm`), but the message
   "Using kspace solver PPPMDipole on system with no dipoles" is an
   `error->all` rather than the more usual guard; fine, just noting.
3. The slab path is analytically reviewed but not covered by a regression
   example; a slab (2-D) charge+dipole test would be worth adding.
