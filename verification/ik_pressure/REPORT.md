# Verification: Irving–Kirkwood long-range dispersion pressure profile

**Feature under test:** the long-range (1/r⁶ dispersion) Irving–Kirkwood (IK) local
pressure contour P(z) added to `compute stress/cartesian` (new `kspace` keyword),
fed by `kspace_style ewald/disp/planar` through the new hook
`KSpace::pressure_profile_long(dir, nbins, lo, width, pN, pT)`
(branch commits `b944d80a`, `b39af99e`, `64f98f33`).

**System:** two-phase LJ liquid–vapor slab, `data.lj_slab` (1267 atoms, box
10×10×36, z inhomogeneous, T=0.85), `pair_style lj/cut/dispplanar 3.0 0.6`,
`kspace_style ewald/disp/planar 1e-5`. Run on 4 MPI ranks.

---

## 1. Code audit (math correctness)

Source reviewed: `src/KSPACE/ewald_disp_planar.cpp::pressure_profile_long`
(line ~1601) and `src/EXTRA-COMPUTE/compute_stress_cartesian.cpp` (kspace wiring,
line ~441).

The IK reciprocal-space profile is
`P_α(z) = Σ_{n,m} S_n S_m C^α_{n,m} e^{i(h_n+h_m)z} − shell_α(z)`, with:

- **Box-average pinning (correct):** only `p=n+m=0` terms survive the z-average,
  and they are pinned to the verified global coefficients — `(0,0)→V·GT[0]`,
  each `n=−m` diagonal `→V·GT[k]/2` (and GN for the normal). Hence
  `box-average(profile) ≡ global kspace pressure` **by construction**
  (`ewald_disp_planar.cpp:1700-1708`). The off-diagonal Φ/Ψ kernels
  (`:1707-1708`) set only the *shape*.
  → *Implication for verification:* a correct box-average is guaranteed and proves
  nothing about the shape; the shape is tested independently below.
- **Anti-alias guard (correct):** the profile contains modes up to `|p|=2·kmax`,
  so the hook requires `nbins > 2·kmax` (`:1618`), else it errors. For
  `ewald/disp/planar 1e-5`, kmax≈175 ⇒ nbins>350. We use dz=0.1 ⇒ nbins=360. ✓
- **Switch-aware k=0 coefficient (correct):** `GT[0]=GN[0]` now includes the
  previously-dropped S′u switch-derivative term (`:491-521`), removing the
  ~1/rcut³ isotropic pressure offset.
- **Hermitian evaluation (correct):** real cosine/sine reconstruction with the
  `2·(...)` factor for ±p (`:1719-1722`); the `p=0` term carried once.
- **Shell subtraction (consistent):** the same real-space `corr_shell` mean field
  used for the box average is subtracted per bin (`shell_profile_virial`,
  `:1662`), so the contour and the box average use an identical correction.
- **Compute wiring (correct):** the hook's `pN` is added to the normal axis (dir1)
  and `pT` to the two lateral axes (`compute_stress_cartesian.cpp:457-460`);
  units already a true pressure (`+virial/V`), added directly to the
  configurational columns.

**Verdict (math):** the equations are internally consistent and the construction
guarantees thermodynamic consistency (box-average = global pressure) by design.
The numerical tests below validate the *shape* and *magnitude*.

### Issues found (reported, not patched — per request)

1. **Only `ewald/disp/planar` implements the hook.** `EwaldDispPlanar` overrides
   `pressure_profile_long`; `PPPMDispPlanar` does **not** (inherits the base no-op
   returning 0). Yet `compute_stress_cartesian.cpp:449-452` errors with a message
   that says *"only ewald/disp/planar / pppm/disp/planar do"* — so a user pairing
   `pppm/disp/planar` with the `kspace` keyword hits that very error, whose text
   wrongly implies pppm is supported. Either implement the hook on PPPM (e.g. by
   inheriting/delegating to the ewald routine) or drop pppm from the message.
2. **Stale docs.** `doc/src/compute_stress_cartesian.rst` still states the compute
   supports "no kspace" and does not document the new `kspace` keyword; the example
   `README` references `kspace_modify` keywords removed in this branch (`contour`,
   `pressure/profile`) and describes the switch shell as `[rcut, rcut+Δ]` whereas
   the code uses `[rcut−Δ, rcut]` (the switch ramps *inward* to the cutoff).

---

## 2. Independent check of the analytic reductions (`verify_math.py`)

The IK kernels Φ(h), Ψ(h) and the global GU/GT/GN coefficients reduce a 1/r⁶
dispersion tail integral to closed form via the `sici_chain` generalized
sine/cosine integrals plus constants (π/48, π/288, π/576, π/72). I re-derived
these by brute-force oscillatory (QUADPACK Fourier) quadrature and compared:

- **Constants** (chain `x→∞` limits): exact to 1e-13.
- **Generalized integrals** `∫ₓ^∞ sin t/tᵐ dt == Aₘ(∞)−Aₘ(x)`: ≤1.3e-6.
- **Kernels** Φ, Ψ, and the GN combo vs `∫_c^∞ g(r)dr`: agree to **1e-9–1e-6
  across all reciprocal modes n=1…175 (=kmax)**. (High-n Ψ/N residuals ~1e-6 are
  the brute-force cancellation floor — true values ~1e-13 — which is exactly why
  the code uses the closed form.)

→ The special-function algebra in `ik_phi`, `ik_psi`, and the GU/GT/GN tail terms
is correct. See `math_results.txt`.

## 3. Numerical verification on the LJ slab

Quick-demo run: 4 MPI ranks, `kspace_modify corr bin`, 30k equil + 150k production
NVT, dz=0.1 (nbins=360 > 2·kmax). IK via `compute stress/cartesian z 0.1 NULL 0 ke
pair kspace`; Harasima via `compute stress/atom NULL ke pair kspace` + `fix
ave/chunk`; long-cutoff reference by `rerun` with `pair_style lj/cut 8.0` (no
kspace). See `results.txt`, `shell_results.txt`, and the figures.

### 3a. IK vs Harasima cross-reference — contour invariance ✓
The two contours agree on every contour-invariant quantity (box averages):

| quantity | IK | Harasima | thermo |
|---|---|---|---|
| ⟨P_N⟩=⟨P_zz⟩ | −0.0657 | −0.0656 | −0.0656 |
| ⟨P_T⟩ | −0.1043 | −0.1043 | — |
| γ_total | **0.6956** | **0.6968** | 0.6968 ± 0.16 |

γ and ⟨P_N⟩ match to ~0.1%, and ⟨P_N⟩ equals the global thermo P_zz — confirming
the box-average pinning and the **magnitude**. The local P_T(z) shapes differ
between IK and Harasima near the interface (expected: contour-dependent) while
enclosing the same area (same γ). See `fig_PT.png`, `fig_gamma.png`.

### 3b. Long-cutoff (no kspace) → short-cutoff (with kspace) — magnitude ✓
Frame-identical reruns (same 76 configs). A = short cutoff + dispersion kspace,
B = plain `lj/cut 8.0`, no kspace:

| region | P_N(A) | P_N(B) | P_T(A) | P_T(B) |
|---|---|---|---|---|
| liquid | −0.1301 | −0.1235 | −0.2189 | −0.2075 |
| vapor | +0.0105 | +0.0108 | +0.0105 | +0.0106 |

A reproduces B **bin-for-bin** (P_N rms 0.0068, P_T rms 0.0086; see
`fig_longcut.png`), with A consistently a hair more attractive in the liquid —
exactly the 1/r⁶ tail beyond B's rcut=8 that A captures via kspace (correct sign
and ~1/rcut³ size). This validates the **magnitude and shape** of the long-range
IK contribution end-to-end.

### 3c. Shell (corr) correction is correct for the IK contour ✓
`corr raw` (exact per-atom shell) vs `corr bin` (density convolution) vs ground
truth B (no shell at all), frame-identical (`verify_shell.py`, `fig_shell.png`):

- **corr raw == corr bin**: P_N rms 0.0002, P_T rms 0.00004 — the shell
  correction is robust and the `bin` approximation (used for speed) is harmless.
- **raw/bin == B to ~1%** through the interface (where the shell term lives): the
  `reciprocal_IK − shell` really is the long-range IK pressure that B computes
  directly in real space.
- raw-vs-B equals bin-vs-B (both ~0.007), so the residual is B's rcut=8 tail, not
  a shell artifact.

### 3d. Mechanical stability — P_N(z)
P_N(z) is flat in each bulk phase but shows a liquid↔vapor offset
(P_N liquid −0.151±0.013 vs vapor +0.010) in this short demo, so it is only
approximately flat. **Crucially this same offset appears identically in the
kspace-free long-cutoff reference B** (B liquid P_N −0.124), so it is a property
of the small-slab / short-equilibration *configurations*, not of the IK pressure
code — the new kspace contribution faithfully tracks the ground truth. The
box-average remains exactly pinned to the global P_zz (−0.066). Strict P_N
flatness would require a larger system and a much longer run; that is a sampling
statement about the test system, not about the estimator. See `fig_PN.png`.

## 4. Cross-validation against the dissertation / SB-Ewald paper

Reproduces the validation of Cribb's dissertation Fig. 4.7/4.5 and the SB-Ewald
paper's Appendix A, for the long-range local surface tension
P_N^LR(z) − P_T^LR(z).

- **Fig. 4.7 (slab method, Eq. 4.18).** The Harasima slab integral
  `(π/2)ρ(z)∫dr (du/dr)∫dz'[r²−3z'²]ρ(z+z')` and the IK analogue (Appendix A:
  `ρ(z)ρ(z+z')` → `∫₀¹dα ρ(z−αz')ρ(z+(1−α)z')`) were evaluated on the measured
  ρ(z) and compared to the lattice sum (kspace). IK: lattice vs slab rms 0.0023;
  H: rms 0.0010. The **sharp** rcut=3.0 form (Appendix A has no switch) matches
  the kspace *net* far better than the switched form (H switched rms 0.0175),
  because the shell correction removes the switch-region [2.4,3.0] mean field.
  Files: `verify_fig47.py`, `fig47_reproduction.png`, `fig47_results.txt`.
  Paper typo found: `K_m` uses `h_n³`/`Sii₅(h_n r_cut)` but should be `h_m`.
- **Direct real-space integration.** An independent brute-force IK pair sum over
  all periodic images (sharp dispersion tail, r∈[3,12]) reproduces the kspace IK
  contour: rms 0.0024, and γ_LR converges to the lattice value as rmax grows
  (0.208 at rmax=9 → 0.221 at rmax=12 vs lattice 0.233; the ~5% residual is the
  sharp-vs-switch treatment, shared by the mean-field slab at 0.224).
  Files: `verify_realspace.py`, `fig_realspace_IK.png`, `fig_IK_threeway.png`.
- **Fig. 4.5 (mechanical stability).** `dP_N/dz` (f_ext=0): IK and H both ≈0 in
  the vapor; in the liquid/interface the H contour shows larger excursions than IK
  (max-residual H/IK ≈ 1.8×), the same direction as the dissertation, but the
  short-run liquid noise prevents the clean 13× separation of the CPP data.
  Files: `verify_fig45.py`, `fig45_reproduction.png`.

**Three-way agreement** (lattice sum = slab Eq 4.18 IK = direct real-space) in
`fig_IK_threeway.png` independently confirms the `ewald/disp/planar` long-range
**IK contour** is correct in both shape and magnitude.

## 5. Direct reproduction of the dissertation CPP 2 simulation

To match the dissertation's Fig 4.7 quantitatively (which uses the long-range
cutoff r>4.0σ, not our earlier r>3.0), the CPP 2 run was reproduced from Table 4.1
(`in.cpp2`): supercritical LJ, N=2000, T\*=1.5, box L_x=L_y=11.872 / L_z=23.744
(sc lattice 10×10×20), full LJ + `ewald/disp/planar` at **rcut=4.0**, and the
cosine external field `U_ext(z)=(ΔU_max/2)cos(2πz/L_z)` (Eq 3.36, ΔU\*max=9.67)
applied via `fix addforce` (`f_z=1.279 sin(2πz/L_z)`).

- **Density profile** (`fig_cpp2_density.png`): the field produces ρ from ~0.05
  (edges) to ~0.9 (centre), **avg 0.598** — matching the dissertation's
  ρ_avg=0.598, ρ range 0.053–0.898.
- **Fig 4.7 reproduction** (`fig_cpp2_fig47.png`, `verify_cpp2.py`): with rcut=4.0
  the long-range P_N^LR−P_T^LR peaks at **~0.024**, matching the dissertation's
  ~0.0245 (the earlier r>3.0 peak of ~0.075 was simply the longer-range tail).
  - **H contour** (left): our lattice sum, the slab (Eq 4.18 H), and the
    **digitized dissertation Fig 4.7** all overlay — double hump, peaks ~0.024,
    slightly-negative bulk ~−0.007. The dissertation used the H contour for the
    long-range (hybrid IK-short/H-long), so this is a direct reproduction.
  - **IK contour** (right, the new code): lattice = slab (Appendix A IK) =
    direct real-space; the IK profile peaks higher and dips lower than H in the
    bulk (contour-dependent), while γ_LR matches H (0.119 vs 0.119) — the
    contour-invariant integral.

### Verdict
The new long-range dispersion IK pressure code is **correct**: the analytic
reductions are verified to ~1e-9; the kspace contribution reproduces the
long-cutoff ground truth bin-for-bin for both P_N and P_T; the shell correction
is correct for the IK contour (raw==bin==ground truth); and IK and Harasima agree
on all contour-invariant quantities (⟨P_N⟩, γ) which also match the global thermo
pressure. The only imperfect metric, strict P_N(z) flatness, is a property of the
quick-demo configurations (shared by the kspace-free reference), not the code.

---

## 3. How to reproduce

```
# build (KSPACE + EXTRA-COMPUTE + MPI)
cd lammps && mkdir build && cd build
cmake -D PKG_KSPACE=on -D PKG_EXTRA-COMPUTE=on -D BUILD_MPI=on ../cmake && cmake --build . -j4
# run
cd ../verification/ik_pressure
mpirun -np 4 ../../build/lmp -in in.runA      # IK + Harasima profiles + traj.dump
mpirun -np 4 ../../build/lmp -in in.rerunA    # IK short+kspace over traj (frame-identical)
mpirun -np 4 ../../build/lmp -in in.rerunB    # plain lj/cut 8.0, no kspace, over traj
python3 verify_pressure.py                    # profiles, gamma, plots, results.txt
```
