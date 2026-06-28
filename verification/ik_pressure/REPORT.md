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

## 2. Numerical verification

_(results filled in after the run — see results.txt and the figures)_

### 2a. Mechanical stability — PN(z) flat
<!-- RESULTS -->

### 2b. IK vs Harasima cross-reference
<!-- RESULTS -->

### 2c. Long-cutoff (no kspace) → short-cutoff (with kspace) convergence
<!-- RESULTS -->

### 2d. Surface tension γ(z)
<!-- RESULTS -->

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
