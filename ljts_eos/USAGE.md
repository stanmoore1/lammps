# Using the LJTS equations of state (agent quick-reference)

Two interchangeable Python modules implement equations of state (EOS) for the
**Lennard-Jones fluid truncated and shifted at r_c = 2.5 σ** (LAMMPS
`pair_style lj/cut 2.5` with the energy shifted to zero at the cutoff):

| Module | EOS | Nature |
|--------|-----|--------|
| `thol2015_ljts_eos.py` | Thol et al. (2015), *Int. J. Thermophys.* **36**, 25 | Empirical 21-term Helmholtz-energy fit — most accurate for **stable/bulk** properties |
| `pets_eos.py` | PeTS, Heier et al. (2018), *Mol. Phys.* **116**, 2083 | Perturbation theory — physically well-behaved in the **metastable & unstable** region |

- **Pure Python, stdlib only** (`math`, `cmath`). No numpy/scipy needed to *use* them.
  (matplotlib is only needed for the optional `plot_*.py` scripts.)
- **All quantities are in Lennard-Jones reduced units: `k_B = σ = ε = m = 1`.**
  So `T` is `T* = k_B T/ε`, `rho` is `ρ* = ρ σ³`, `p` is `p* = p σ³/ε`, energies
  are in units of `ε`, etc.

## Quick start

```python
import sys; sys.path.insert(0, "/home/user/lammps/ljts_eos")
import thol2015_ljts_eos as thol
import pets_eos as pets

# pressure at (T*, rho*)
p = thol.pressure(1.0, 0.70)          # -> p*

# full property set
props = pets.properties(1.0, 0.70)    # -> dict, see schema below
print(props["p"], props["Z"], props["u"])

# critical point  (solves dp/drho = d2p/drho2 = 0)
Tc, rhoc, pc = thol.critical_point()  # Thol: (1.086, 0.319, 0.101)
Tc, rhoc, pc = pets.critical_point()  # PeTS: (1.089, 0.309, 0.102)

# vapour-liquid equilibrium at temperature T (Maxwell construction)
rho_vap, rho_liq, p_sat = pets.vle(0.80)
```

## Common API (identical names in BOTH modules)

| Call | Returns | Meaning |
|------|---------|---------|
| `pressure(T, rho)` | float | pressure `p*` |
| `properties(T, rho)` | dict | full thermodynamic state (schema below) |
| `critical_point(T0=…, rho0=…)` | `(Tc, rhoc, pc)` | critical point by 2-D Newton |
| `vle(T, rho_v0=…, rho_l0=…)` | `(rho_vap, rho_liq, p_sat)` | coexistence densities + saturation pressure |
| `B2_eos(T)` | float | 2nd virial coeff. from the EOS (low-density limit) |
| `B2_integral(T)` | float | 2nd virial coeff. by direct integration of the LJTS Mayer function (EOS-independent ground truth) |
| `ljts_potential(r)` | float | the pair potential `u*(r)` |

**Portable derivative:** to get `(∂p/∂ρ)_T` in a way that works for *both*
modules, use `properties(T, rho)["dpdrho"]`.
(`pets_eos` also exposes `dp_drho`, `d2p_drho2`, `dalpha_drho`, `dalpha_dT`,
`compressibility`, `alpha_res`, `bh_diameter`; in `thol2015_ljts_eos` the
analogous helpers are `alpha_res_derivs` and the private `_dp_drho`.)

### `properties()` return schema

Both modules return these keys (LJ units, per particle where extensive):

`T, rho, Z, p, dpdrho, u, u_res, a_res, mu_res`

- `Z` = compressibility factor `p/(ρ k_B T)`
- `p` = pressure, `dpdrho` = `(∂p/∂ρ)_T`
- `u` = total internal energy per particle (`= 1.5 T + u_res`), `u_res` = residual part
- `a_res` = residual Helmholtz energy per particle
- `mu_res` = residual chemical potential

**Thol only** additionally returns: `cv`, `cp` (isochoric/isobaric heat
capacity per particle, `/k_B`), `s_res` (residual entropy), and the raw scaled
Helmholtz derivatives `a00, a01, a02, a10, a20, a11`.
**PeTS only** additionally returns `eta` (packing fraction).
If you need `cv`/`cp`, use the Thol module.

## Which EOS to use

- **Bulk / single-phase / saturation properties, max accuracy:** either; they
  agree closely. Thol is the reference-quality fit.
- **Heat capacities (`cv`, `cp`):** use **Thol** (`properties()` provides them).
- **Metastable or unstable region** (nucleation, spinodal decomposition,
  density-gradient theory / DFT interfaces, cavitation, negative pressures):
  use **PeTS**. See caveat below.

## Important caveat — Thol's van der Waals loop

Below **T* ≈ 1.031** the empirical Thol EOS develops **spurious extra extrema**
inside the two-phase region (four `dp/dρ = 0` points instead of two) and large
unphysical negative-pressure excursions. It is an extrapolation artifact of the
fit. **PeTS keeps exactly two spinodal points at all temperatures.** Do not
trust Thol inside the binodal below T*≈1.03; use PeTS there.
(Demonstration: `plot_loop_artifact.py`.)

## Validity ranges

- **Thol (2015):** ~`0.6 < T*/T_c < 10`, `p*/p_c < 70` for stable states.
- **PeTS (2018):** built to be usable across the whole metastable/unstable
  region as well; less accurate than Thol for some bulk properties.

## Verification & plots

```bash
python3 thol2015_ljts_eos.py     # runs built-in verification suite (exit 0 = pass)
python3 pets_eos.py              # ditto
python3 plot_phase_diagram.py    # binodal + spinodal + critical point (both EOS)
python3 plot_isotherms.py        # p-rho isotherms around T_c (both EOS)
python3 plot_loop_artifact.py    # demonstrates the Thol loop artifact
```

Each EOS module's `_run_verification()` checks: analytic-vs-numerical derivative
consistency; the 2nd virial coefficient vs direct Mayer-function integration;
the critical point; (Thol) cross-validation against the independent CC0
Allen-Tildesley reference code; and mutual Thol↔PeTS agreement at VLE.

## Gotchas for automation

- Pass arguments in **reduced units**, not SI. To map to a real substance,
  scale by that substance's `ε`, `σ` (e.g. argon `ε/k_B ≈ 119.8 K`,
  `σ ≈ 0.3405 nm`).
- `vle(T)` and `critical_point()` are iterative; for `vle` far from the default
  guesses (very low T, or near T_c) pass good `rho_v0`/`rho_l0` (e.g. via
  temperature continuation — see `plot_phase_diagram.py:binodal`).
- These EOS are for **r_c = 2.5 σ truncated-AND-shifted** only. They are *not*
  the full-LJ EOS and need no long-range/tail correction. For full LJ or other
  cutoffs, a different EOS is required.
