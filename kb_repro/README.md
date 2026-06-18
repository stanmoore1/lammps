# Reproducing Figs. 4 & 5 of Nichols, Moore & Wheeler (Phys. Rev. E 80, 051203, 2009)

"Improved implementation of Kirkwood–Buff solution theory in periodic molecular
simulations" — the **Fourier ("new KB")** method. This directory reproduces
**Figure 4** (thermodynamic properties derived from the Fourier partial
structure factors vs. wavevector `q`) and **Figure 5** (Fourier transforms of
the direct correlation functions `rho*C_ij(q)`) at composition `y1 = 0.4`,
`T = 120 K`.

## System

Binary Lennard-Jones mixture imitating **CF4 (species 1)** + **CH4 (species 2)**
(Table I of the paper):

| pair | σ (Å) | ε/k_B (K) |
|------|-------|-----------|
| 1-1  | 4.150 | 175.0 |
| 2-2  | 3.728 | 149.0 |
| 1-2  | 3.939 | 142.1 (modified Berthelot, k12=0.12) |

State point (Table II, `y1=0.4`, `T=120 K`, molar volume 44.6 cm³/mol):

| N    | L (Å)  | N1 (CF4) | N2 (CH4) |
|------|--------|----------|----------|
| 1200 | 44.626 | 480      | 720      |
| 4000 | 66.662 | 1600     | 2400     |

`units real`, `pair_style lj/cut 13.4` (≈0.3L / 0.2L, three neighbor shells),
`pair_modify tail yes`, Nosé–Hoover NVT at 120 K, `dt = 4 fs`.

## New code

`src/EXTRA-COMPUTE/compute_structure_factor.{cpp,h}` (Stan Moore) — an
Ewald-style compute that evaluates the per-group structure factor

    S_group(q) = < |Σ_{a∈group} exp(i q·r_a)|² > / N_group

averaged over the `|q|`-shells with `m1²+m2²+m3² ≤ 17` (the paper's cutoff
`q_c = 2π L⁻¹ √17`). Output array columns: `[q, S(q), shell multiplicity]`.

## Method

`compute structure/factor` is run on three groups simultaneously — species 1,
species 2, and all atoms — and time-averaged with `fix ave/time`. The paper's
partial structure factors `S_ij = A⁻¹_ij` (Eq. 18) are reconstructed as

    raw_i  = S_i(q) · N_i                       (|Σ_i|²)
    raw_12 = (raw_all − raw_1 − raw_2)/2        (cross term)
    S_11 = raw_1/N,  S_22 = raw_2/N,  S_12 = raw_12/N

From `S(q)` (matrix `A = S⁻¹`) the analysis script derives, at each q:

* partial molar volumes  `V̄_i = (1/ρ)(Ay)_i/(yᵀAy)`        (Eq. 6)
* isothermal compressibility `κ_T = κ_T^ig/(yᵀAy)`           (Eq. 7)
* activity-coefficient correction `Q11`                       (Eq. 10)
* direct correlation functions `ρC = Y⁻¹ − A`                 (Eq. 26)

The empirical `q→0` extrapolation of the paper (Sec. IV E) is then applied to the
N=1200 data: `V̄1 = V1ᴬ+V1ᴮq²`, `κ_T⁻¹ = KTᴬ+KTᴮq+KTᶜq³`, `Q11 = Q11ᴬ+Q11ᴮq`,
and the fitted properties are inverted back to `S_ij(q)` for the Fig. 4(a) curves.

## Running

```
../build_kb/lmp_kb -var L 44.626 -var N1 480 -var N2 720 -var tag N1200 \
    -var nequil 100000 -var nprod 600000 -in in.kb
../build_kb/lmp_kb -var L 66.662 -var N1 1600 -var N2 2400 -var tag N4000 \
    -var nequil 80000 -var nprod 240000 -in in.kb
python3 analyze.py     # -> figure4.png, figure5.png
```
