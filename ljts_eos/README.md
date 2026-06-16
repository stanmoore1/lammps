# Equations of state for the Lennard-Jones truncated-and-shifted (LJTS) fluid

Two self-contained Python implementations of published equations of state (EOS)
for the Lennard-Jones fluid **truncated and shifted at a cut-off radius
`r_c = 2.5 σ`** — i.e. the potential you get from `pair_style lj/cut 2.5` in
LAMMPS with the energy shifted to zero at the cut-off.

| Script | EOS | Type | Reference |
|--------|-----|------|-----------|
| `thol2015_ljts_eos.py` | Thol et al. (2015) | Empirical multiparameter Helmholtz-energy fit (21 terms) | M. Thol, G. Rutkai, R. Span, J. Vrabec, R. Lustig, *Int. J. Thermophys.* **36**, 25 (2015), doi:10.1007/s10765-014-1764-4 |
| `pets_eos.py` | PeTS — Heier et al. (2018) | Physically-based perturbation theory (Barker-Henderson hard-sphere reference + PC-SAFT-type dispersion) | M. Heier, S. Stephan, J. Liu, W. G. Chapman, H. Hasse, K. Langenbach, *Mol. Phys.* **116**, 2083 (2018), doi:10.1080/00268976.2018.1447153 |

Both are written in **pure Python (stdlib only — `math`, `cmath`)**, no
third-party dependencies. All quantities are in **Lennard-Jones reduced units**
(`k_B = σ = ε = m = 1`).

## Usage

```bash
python3 thol2015_ljts_eos.py    # runs the verification suite + a demo state point
python3 pets_eos.py
```

Both modules expose a `properties(T, rho)` function returning pressure,
compressibility factor `Z`, internal energy, residual Helmholtz energy,
chemical potential, etc., plus helpers `pressure(T, rho)`, `B2_integral(T)`,
`critical_point()` and `vle(T)`.

```python
import pets_eos as eos
print(eos.pressure(1.0, 0.3))      # -> 0.0546...
print(eos.critical_point())        # -> (1.0890, 0.3092, 0.1020)
```

## How correctness is verified

Each script ends with a `_run_verification()` suite (run automatically as
`__main__`, exit code 0 = all passed):

1. **Internal differentiation consistency.** The analytic derivatives used for
   the thermodynamic properties are checked against high-order (4th-order)
   finite differences / complex-step differentiation of the residual Helmholtz
   energy. Agreement to `~1e-9` (Thol) / `~1e-11` (PeTS).

2. **Independent ground truth — 2nd virial coefficient.** The low-density limit
   of the EOS, `B2 = lim_{ρ→0}(Z-1)/ρ`, is compared with `B2(T)` obtained by
   *direct numerical integration of the Mayer function* of the LJTS potential
   (Simpson rule) — completely independent of the EOS. Thol: matches to <1%
   (it is fitted to virial data). PeTS: matches to ~5% (a perturbation theory,
   not fitted to virial data — expected).

3. **Critical point.** Solved from `dp/dρ = d²p/dρ² = 0`.
   - Thol: `T_c=1.086, ρ_c=0.319, p_c=0.101` (its reducing parameters).
   - PeTS: `T_c=1.0890, ρ_c=0.3092, p_c=0.1020`, verified to be a genuine
     stationary point. (The figure-extracted reference in the `feos` project,
     1.0884/0.3078, is *not* exactly stationary — as `feos`'s own comment
     warns — so it is reported only informationally.)

4. **Cross-validation Thol ↔ independent reference code.** The Thol residual
   Helmholtz derivatives and properties are checked against the public-domain
   (CC0) reference implementation accompanying Allen & Tildesley, *Computer
   Simulation of Liquids*, 2nd ed. (2017), at two state points (agreement to
   `5e-7`).

5. **Cross-validation Thol ↔ PeTS.** Vapor-liquid equilibria computed
   independently from the two EOS agree closely (e.g. at `T=0.7`, saturated
   liquid density `ρ_liq = 0.7869` (Thol) vs `0.7870` (PeTS)), and both are
   consistent with the published LJTS simulation data of Vrabec et al. (2006).

## Notes / provenance of coefficients

- The **Thol (2015)** coefficient bank reproduced in `thol2015_ljts_eos.py` is
  Table 1 of the paper; the specific numerical values were taken from the
  CC0/public-domain reference code of Allen & Tildesley (which transcribes the
  same table) and cross-validated against it.
- The **PeTS** functional form and numerical constants (Barker-Henderson
  diameter constants, and the `A_i`/`B_i` dispersion-integral universal
  constants) match the open-source PeTS implementations in `feos`
  (feos-org/feos) and `teqp` (usnistgov/teqp), which reproduce Heier et al.
  (2018).
