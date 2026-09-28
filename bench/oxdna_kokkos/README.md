# oxDNA-Kokkos

A portable, GPU-ready standalone implementation of the [oxDNA](https://github.com/lorenzo-rovigatti/oxdna)
coarse-grained DNA model, written with [Kokkos](https://github.com/kokkos/kokkos).
It is intended as a compact benchmark and reference port: the force field
faithfully reproduces the standalone oxDNA **oxDNA1**, **oxDNA2** and
(sequence-dependent) **oxDNA3** models (validated term-by-term, see
[Validation](#validation)), while the data layout and kernels are structured
for performance on CPUs (Serial/OpenMP) and GPUs (CUDA) through a single Kokkos
code base, mirroring the standalone oxDNA CUDA kernels.

The oxDNA3 port tracks upstream oxDNA commit **c2c74cc0 (2026-09-21)**
(`src/CUDA/Interactions/CUDA_DNA3.cuh`, `src/Interactions/DNA3Interaction.cpp`).

Each nucleotide is a rigid body (center of mass + orientation quaternion) with
three interaction sites (backbone, base, stacking) derived from its orientation.

## What it computes

Molecular dynamics in the NVE or NVT (Brownian thermostat) ensemble, with the
full oxDNA interaction set:

| Interaction | Type | oxDNA1 | oxDNA2 | oxDNA3 |
|---|---|:---:|:---:|:---:|
| FENE backbone bond | bonded | yes | yes | yes (tetramer r0, Delta) |
| Bonded excluded volume (base-base, base-back, back-base) | bonded | yes | yes | yes (tetramer sigma, r*) |
| Stacking (F1 radial, F4 angles, F5 dihedrals) | bonded | yes | yes | yes (tetramer tables) |
| Nonbonded excluded volume (4 site pairs) | nonbonded | yes | yes | yes (per base pair) |
| Hydrogen bonding (F1, 6 angular terms, Watson-Crick) | nonbonded | yes | yes | yes (per base pair) |
| Cross-stacking (F2, 6 angular terms) | nonbonded | yes | yes | yes (asymmetric 3'3' + 5'5', tetramer) |
| Coaxial stacking | nonbonded | yes (+cosphi3) | yes (harmonic theta1) | yes (oxDNA2 angles, SD K) |
| Debye-Hueckel electrostatics | nonbonded | - | yes | yes |
| Grooved backbone site (major/minor grooving) | geometry | - | yes | yes |
| Per-base site offsets (purine/pyrimidine base site) | geometry | - | - | yes |

oxDNA2 additionally uses its own well depths (`HYDR_EPS`, stacking eps), the
`FENE_R0_OXDNA2` bond length, and a salt-dependent Debye length.

oxDNA3 is sequence dependent through tetramer tables `T(n3_2, n3_1, n5_1, n5_2)`
(the bonded pair plus its two flanking neighbours, `NO_TYPE` at strand ends)
read from `oxDNA3_sequence_dependent_parameters.txt` (a copy ships in
`params/`); see [oxDNA3](#oxdna3) below.

All model constants are taken directly from the standalone oxDNA `src/model.h`
(lj/reduced units), so results are directly comparable.

### Components

| File | Purpose |
|---|---|
| `src/main.cpp` | CLI entry point |
| `src/simulation.h` | MD driver: I/O, force evaluation, time loop |
| `src/integrator.h` | Velocity-Verlet + quaternion (lab-frame) orientation update |
| `src/thermostat.h` | Brownian ("John") thermostat (optional, NVT) |
| `src/neighbor_list.h` | Cell list + flat Verlet edge list (one thread per pair) |
| `src/particles.h`, `src/types.h` | SoA particle storage, quaternion / box types |
| `src/forces/params.h` | Force-field parameters (`make_oxdna1_params`, `make_oxdna2_params`) |
| `src/forces/params_dna3.h` | oxDNA3 parameters: sequence-dependence file parser, tetramer tables (host build, one device View), `make_oxdna3_params` |
| `src/forces/dna3_forces.h` | oxDNA3 kernels: edge nonbonded (`DNA3_forces_edge_nonbonded`) + bonded gather (`DNA3_forces_edge_bonded`) ports of `CUDA_DNA3.cuh` |
| `params/oxDNA3_sequence_dependent_parameters.txt` | Copy of the upstream oxDNA3 parameter file (default `seq_dep_file`) |
| `src/forces/mf_oxdna.h` | Modulation functions F1–F6 and derivatives |
| `src/forces/bonded.h` | Bonded gather kernel: FENE + bonded excluded volume + stacking (one thread per particle, reads its n3/n5 neighbours, no atomics — mirrors oxDNA's `dna_forces_edge_bonded`) |
| `src/forces/dna_forces.h` | Nonbonded kernel: excv + H-bond + cross + coaxial + Debye–Hückel |
| `src/forces/orient.h` | Quaternion → body-axis vectors |
| `src/io/topology_reader.h`, `src/io/config_reader.h` | oxDNA `.top` (old 3'->5' and new `5->3` formats) / `.conf` readers |

## Building

Requires CMake ≥ 3.20 and a C++20 compiler (Kokkos 5.0 requires C++20). Kokkos
is taken from the bundled LAMMPS tree (`lib/kokkos`) by default, or any
installed/standalone Kokkos via `-DKOKKOS_SOURCE_DIR=<path>` (or `find_package`).

```bash
cd bench/oxdna_kokkos

# CPU, single-threaded (debug / portable)
cmake -B build -DKokkos_ENABLE_SERIAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CPU, multi-threaded
cmake -B build -DKokkos_ENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# NVIDIA GPU (e.g. Ampere SM80)
cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
      -DCMAKE_CXX_COMPILER=$(pwd)/../../lib/kokkos/bin/nvcc_wrapper \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Useful CMake options:

- `-DOXDNA_SINGLE_PRECISION=ON` — use `float` instead of `double`.
- `-DOXDNA_BUILD_TESTS=ON` — also build the validation tools (`fd_test`, `xcheck`).

### Matching the reference GPU performance

The single biggest performance lever is **precision**. The standalone oxDNA GPU
benchmarks run with `backend_precision = mixed`, which means the **force kernels
execute in FP32** (only the integrator uses FP64). This code's force kernels run
at the compile-time `c_number` precision, so the **default `double` build runs
the force kernels in FP64** — roughly 2× the compute and 2× the memory traffic
of the reference on most GPUs, which accounts for most of the observed slowdown.
For an apples-to-apples GPU comparison build single precision:

```bash
cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
      -DCMAKE_CXX_COMPILER=$(pwd)/../../lib/kokkos/bin/nvcc_wrapper \
      -DOXDNA_SINGLE_PRECISION=ON -DCMAKE_BUILD_TYPE=Release
```

Structurally the code now mirrors the reference GPU layout: one thread-per-edge
nonbonded kernel with atomic accumulation (oxDNA `use_edge`, `edge_n_forces=1`),
and one thread-per-particle **gather** bonded kernel (FENE + bonded excluded
volume + stacking) that writes only its own particle with no atomics. The
Verlet-list rebuild check is **fused into the first-step kernel** (it flags a
rebuild via a single device int while integrating), so no separate full-N
reduction runs each step — matching oxDNA's `_d_are_lists_old` flag. The Verlet
list uses oxDNA's convention (radius `rcut + 2·skin`, rebuild when a particle
moves `> skin`). Per-particle arrays are stored **AoS** (`Kokkos::LayoutRight`,
i.e. each particle's `x,y,z,w` contiguous) to match oxDNA's `c_number4`/`float4`
layout, giving one coalesced transaction per particle for the scattered reads in
the edge kernel.

## Running

```bash
./build/oxdna_kokkos <input_file>
```

The program is driven by a **standalone-oxDNA-style input file** (`key = value`),
so the *same* input that drives the reference oxDNA drives this code —
unrecognized keys (`backend`, `CUDA_list`, `trajectory_file`, `ensemble`,
`data_output_*` blocks, `${...}` expressions, ...) are ignored. Recognized keys:

| Key | Default | Description |
|---|---|---|
| `topology`           | — | Topology `.top` (mandatory) |
| `conf_file`          | — | Configuration `.conf`/`.dat` (mandatory) |
| `energy_file`        | (none) | If set, write oxDNA-style `time U K total` (per nucleotide) |
| `interaction_type`   | DNA | `DNA`/`DNA1` -> oxDNA1, `DNA2` -> oxDNA2, `DNA3`/`DNA3_nomesh` -> oxDNA3 |
| `salt_concentration` | 0.5 | mol/L (oxDNA2, oxDNA3) |
| `seq_dep_file`       | `params/oxDNA3_sequence_dependent_parameters.txt` | oxDNA3 parameter file (path relative to the working directory, as upstream); default: the copy in `params/` (absolute path baked in at configure time, CMake cache variable `OXDNA_DEFAULT_SEQ_DEP_FILE`), then `params/...` or `./oxDNA3_sequence_dependent_parameters.txt` |
| `use_average_seq`    | 0 (oxDNA3) | oxDNA3 only: `1` keeps the average (non-SD) tables; see note below |
| `dh_half_charged_ends` | 1 | oxDNA3: halve the Debye-Hueckel charge of strand-end nucleotides |
| `dh_lambda`, `dh_strength`, `debye_huckel_rhigh` | 0.3616455, 0.0543, 3 lambda | oxDNA3 Debye-Hueckel parameters (upstream keys) |
| `dna3_consistent_gamma` | 0 | oxDNA3, bench-only: exact stacking-dihedral gradient (see [oxDNA3](#oxdna3)) |
| `T`                  | 0.1 | `20C`, `300K`, or a number in oxDNA units (1 unit ≈ 3000 K) |
| `dt`                 | 0.001 | Timestep |
| `steps`              | 10000 | Number of MD steps (accepts `1e7`) |
| `verlet_skin`        | 0.3 | Verlet skin |
| `print_energy_every` | 1000 | Energy print frequency |
| `seed`               | 12345 | RNG seed (velocity refresh + thermostat) |
| `refresh_vel`        | 0 | `1` → draw fresh Maxwell-Boltzmann velocities at T on startup (required for velocity-less confs) |
| `thermostat`         | (none) | `brownian`/`john` → NVT; otherwise NVE |
| `newtonian_steps`    | 0 | Brownian thermostat period in steps |
| `diff_coeff`         | 2.5 | Translational diffusion coefficient |
| `pt`                 | 0 | Refresh probability; overrides `diff_coeff` if `> 0` |
| `timing`             | 0 | `1` → per-kernel timing breakdown (adds fences; `0` for production) |

Paths are resolved relative to the working directory (run from the case
directory, as with the reference oxDNA). oxDNA value expressions are supported:
`$(key)` substitutes another key's value and `${ ... }` evaluates `+ - * / ()`
arithmetic, e.g. `print_energy_every = ${$(steps) / 100}`. The `.top` lists
`<N> <N_strands>` then one `<strand_id> <base> <n3> <n5>` line per nucleotide
(or, in the new upstream format, `<N> <N_strands> 5->3` then one 5'->3'
sequence line per strand, optionally with `circular=true`);
the `.conf` has `t = …`, `b = Lx Ly Lz`, `E = …`, then one line per nucleotide
with position, `a1`, `a3`, and optionally velocity and angular momentum.

Example (the bundled oxDNA2 cases each ship an `input` file, and an oxDNA3
`input_dna3` that also runs unchanged with the standalone oxDNA):

```bash
cd tests/N8 && ../../build/oxdna_kokkos input        # oxDNA2
cd tests/N8 && ../../build/oxdna_kokkos input_dna3   # oxDNA3
```

Note on `use_average_seq`: the standalone oxDNA defaults to
`use_average_seq = true`, which for `DNA3` skips the parameter file and runs
with average tables (oxDNA1 H-bond depth, and a stacking depth computed from
the not-yet-initialised member `_T` in the DNA3Interaction constructor, i.e.
undefined). A real oxDNA3 run therefore needs `use_average_seq = false` and
`seq_dep_file = ...` upstream; here oxDNA3 is sequence dependent by default and
`use_average_seq = 1` uses the average tables with the stacking depth at the
run temperature. The `input_dna3` files set both keys explicitly.

### Energy output

Energies are reported **per nucleotide** in oxDNA units, exactly like the
reference oxDNA (the reference divides the total energy by N). stdout has the
columns `step  time  U  K  total` (with `time = step * dt`), and if
`energy_file` is set it is written in oxDNA's `time U K total` format so it can
be compared directly with the reference `energy_file`:

```
#       step           time              U              K          total
           0       0.000000      -1.354229       0.293346      -1.060883
```

(The earlier builds printed *total* extensive energies; multiply by N to
convert old output, or just use the per-nucleotide values now emitted.)

## Performance output

At the end of a run the code prints a LAMMPS-style loop-time / performance
summary. By default (production) it reports only the loop time and performance
with no per-section fences, so the loop time is the true throughput:

```
Loop time of 3.92618 on 1 procs (Serial x 1) for 2000 steps with 1024 atoms

Performance: 132036.667 tau/day, 509.401 timesteps/s, 0.522 Matom-step/s
(set 'timing = 1' in the input file for the per-kernel breakdown)
```

Set `timing = 1` in the input to also get the per-kernel breakdown. This fences
at each section boundary (a no-op on CPU; one sync per section on GPU, like
LAMMPS `timer full`), so prefer `timing = 0` for production throughput numbers:

```
Loop time of 4.8036 on 1 procs (Serial x 1) for 2000 steps with 1024 atoms

Performance: 107918.974 tau/day, 416.354 timesteps/s, 0.426 Matom-step/s

Kernel timing breakdown:
Section                |   time (s) |  %loop |    us/step
------------------------------------------------------------
Neigh                  |     0.9686 |  20.16 |    484.318
Bond (FENE+bond-excv)  |     0.2581 |   5.37 |    129.073
Pair: stacking         |     0.4989 |  10.39 |    249.444
Pair: nonbonded        |     2.9464 |  61.34 |   1473.225
Modify (integ+thermo)  |     0.1311 |   2.73 |     65.550
Output                 |     0.0003 |   0.01 |      0.141
Other                  |     0.0001 |   0.00 |      0.052
------------------------------------------------------------
Total (loop)           |     4.8036 | 100.00 |   2401.802
```

The sections map onto LAMMPS' timing breakdown for a CG-DNA run as follows, so
the two can be compared side by side:

| This code | LAMMPS section(s) |
|---|---|
| `Neigh` | `Neigh` |
| `Bond (FENE+bond-excv)` | `Bond` (FENE) + part of `Pair` (bonded excluded volume) |
| `Pair: stacking` | `Pair` (`oxdna/stk`) |
| `Pair: nonbonded` | `Pair` (`oxdna2/excv`, `oxdna/hbond`, `oxdna/xstk`, `oxdna2/coaxstk`, `oxdna2/dh`) |
| `Modify (integ+thermo)` | `Modify` (`nve/dotc/langevin` + thermostat) |
| `Output` | `Output` |

`timesteps/s` is the most directly comparable metric to LAMMPS' `Performance:`
line. On CPU backends `Kokkos::fence()` is a no-op, so the section times are
exact; on the CUDA backend each section boundary fences (like LAMMPS `timer
full`), which slightly inflates the loop time versus an untimed run.

## Validation

Build with `-DOXDNA_BUILD_TESTS=ON` and run from this directory.

- **`./build/fd_test`** -- self-checking suite for all three models: analytic
  forces & torques vs. central finite differences of the energy (every term;
  for oxDNA3 each of the 8 upstream terms separately, on the relaxed duplex, on
  a nicked + perturbed duplex in which every term is active, and on a
  configuration with active stacking cos(phi1/phi2) modulation), NVE energy
  conservation, and thermostat temperature (equipartition). Exits non-zero on
  failure. This is what CI runs.
- **`./build/xcheck <model> <T> <salt> <top> <conf> [ft_out] [seq_dep_file]
  [--only=<term>] [--consistent-gamma] [--average-seq]`** -- prints the
  potential energy (total and per group; for oxDNA3 also every term, in the
  column order of `potential_energy split = true`) and optionally dumps
  per-particle force/torque (lab frame), for direct comparison against the
  standalone oxDNA `potential_energy split = true` and `force_and_torque`
  (`particle = -1`, `lab_frame = true`) observables. `--only=<term>`
  (`fene bexc stck nexc hb crst cxst dh`) restricts the dump to one term, like
  the upstream `DNA_enable_<term>` switches.

Cross-checked against the compiled standalone oxDNA on an 8bp duplex
(average-sequence, T = 0.1; oxDNA2 at salt = 0.5):

| Quantity | oxDNA1 | oxDNA2 |
|---|---|---|
| Total potential energy (per particle) | matches to ~1e-5 | matches to ~1e-5 |
| Per-particle force / torque vectors | ~5e-5 / ~4e-5 (rel.) | ~6e-5 / ~1e-4 (rel.) |
| FD force/torque = -grad E | ~1e-8 (rel.) | ~1e-8 (rel.) |
| NVE total-energy drift | ~1e-6 | ~1e-7 |

### oxDNA3 validation

Reference: standalone oxDNA c2c74cc0, CPU build (double), `interaction_type =
DNA3_nomesh` (the analytic-f4 CPU variant, i.e. the same functional form as the
CUDA kernels; the default `DNA3` uses interpolation meshes for f4),
`use_average_seq = false`, the shipped `seq_dep_file`. This code: double build.
The standalone observables print 6 significant digits, which limits the
agreement below to ~1e-6 relative.

| Case (all 8 split terms compared) | Energy per term / nt | max \|dF\| / max \|F\| | max \|dT\| / max \|T\| |
|---|---|---|---|
| 8bp duplex, `test_dna3.conf`, T = 0.1, salt 0.5 | identical (6 decimals) | 7.2e-7 | 8.5e-7 |
| 8bp nicked (`test_dna3_nicked.top`), coax + nonbonded excv active | identical | 6.7e-7 | 1.0e-6 |
| 8bp nicked + perturbed (`test_dna3_pert.conf`), bonded excv active | identical | 1.9e-6 | 3.2e-6 |
| N8 (128 nt), T = 20C, salt 1.0 | identical | 3.5e-6 | 9.7e-7 |
| N64 (1024 nt), T = 20C, salt 1.0 | identical | 4.1e-6 | 1.0e-6 |
| 5 thermalised oxDNA3 N8 frames (upstream MD) | identical | <= 3.5e-6 | <= 3.6e-6 |
| vs. upstream `DNA3` (mesh), N8 / 8bp | HB differs by 2e-6 / nt | 6.3e-5 / 4.5e-5 | 6.1e-5 / 4.5e-5 |

Total energy / nt: 8bp -1.156022, N8 and N64 -1.185172 (both codes).
Single-precision build (`-DOXDNA_SINGLE_PRECISION=ON`): N8 energies identical to
6 decimals, forces/torques 7e-5 / 4e-5 relative, `fd_test` passes.

`fd_test` (double): every oxDNA3 term separately and combined matches the
finite-difference gradient to <= 4e-8 (relative to the largest component), NVE
drift 7.8e-4 over 3000 steps (see the cross-stacking note below), equipartition
within 5% (8bp, short run). With the stacking cos(phi) modulation active, the upstream-faithful
forces deviate from -grad E by 9.7e-5 (force) / 5.3e-4 (torque), the
`dna3_consistent_gamma = 1` variant by 3e-9 / 6e-10 (see below).

MD: `tests/{8bp_duplex,N8,N64}/input_dna3` are stable (10000 steps, dt = 0.003,
Brownian thermostat). N8, 100000 steps, dt = 0.003, `pt = 0.1`, 3 seeds each:
<U>/nt = -1.3486 +- 0.0013 (this code) vs -1.3427 +- 0.0061 (standalone oxDNA,
`DNA3`), <K>/nt = 0.2918 vs 0.2926 (3T = 0.2932). oxDNA1/oxDNA2 results are
bit-for-bit unchanged by the oxDNA3 addition (`tests/8bp_duplex/input`,
`tests/N8/input`, `xcheck` force dumps). CPU cost (Serial, N64, 3000 steps):
oxDNA2 5.3 s, oxDNA3 6.5 s.

Continuous integration (`.github/workflows/oxdna-kokkos.yml`) builds with the
Serial backend and runs `fd_test` on every change under `bench/oxdna_kokkos/`.

> Note: from a cold start (zero velocities) oxDNA2 needs a smaller timestep
> (`-dt 1e-4`) than oxDNA1 because the Debye–Hückel + grooved backbone make the
> potential stiffer near close approaches; a thermostat or smaller `dt` keeps it
> stable. This does not affect force-evaluation throughput.

## oxDNA3

### Kernel structure (mirrors `CUDA_DNA3.cuh`)

- `DNA3NonbondedFunctor` = `DNA3_forces_edge_nonbonded`: one thread per Verlet
  edge. As in the CUDA edge list, `p` (CUDA `from`) is the larger and `q` (`to`)
  the smaller index, `r = q - p` (minimum image). Per edge it reads both
  positions and quaternions, both `LR_bonds`, the `uint8` particle-type array
  for `p`, `q` and their n3/n5 neighbours (tetramer flanks, `NO_TYPE` = 5 at
  strand ends) and the tables; `_DNA3_particle_particle_DNA_interaction`
  (excluded volume, Debye-Hueckel with half-charged ends, H-bonding, cross
  stacking 3'3' + 5'5', coaxial stacking) accumulates the force `F` and torque
  `T` on `p`, then atomics add `T -> p`, `F -> p`, `-F -> q`,
  `-T + r x F -> q` (skipped when zero, as CUDA).
- `DNA3BondedFunctor` = `DNA3_forces_edge_bonded`: one thread per particle,
  `_DNA3_bonded_part<true>` for its n3 bond and `<false>` for its n5 bond
  (FENE + bonded excluded volume + stacking), writes only its own
  force/torque, no atomics, runs after the nonbonded kernel.
- Differences from CUDA: torques stay in the lab frame (this code's integrator
  is lab-frame, so the final body-frame transform is not needed); bonded
  separations use the minimum image (positions are kept wrapped); the energy
  is reduced separately (CUDA stores it in `.w`); the functors take a
  compile-time term mask (`dna3::ALL` in production, single terms for the
  validation tools). oxDNA1/oxDNA2 keep their own kernels (the model is
  selected once per step on the host, no per-pair runtime branch).

### Parameters and tables

`make_oxdna3_params` (`params_dna3.h`) ports the `DNA3Interaction` constructor
and `init()`: model.h defaults (float literals, as upstream), independent
parameters from the sequence-dependence file (parsed like `getInputFloat`:
`atof`, rounded to float), strand-end (`NO_TYPE`) entries as averages over the
missing flank, the symmetric coaxial K, and the enslaved smoothing parameters
(F1 `RLOW/RHIGH/RC = R0 - 0.06/+0.3/+0.35` (HB), `-0.08/+0.35/+0.5` (stacking),
F2 ranges, F3 `RC/B` from `sigma, r*`, F4 `TS = sqrt(0.81225/A)`, `TC`, `B`,
F5 `XC`, `B`). All 214 tables of `6 x 5 x 5 x 6` entries (FENE r0/Delta^2,
excluded volume sigma/r*/b/rc (7), F1 (2), F2 incl. symmetric K (4), F4 (21),
F5 (4)) live in ONE device View (`c_number`: float in single precision), the
analogue of the CUDA `__constant__`/`__device__` `MD_*_SD` arrays, indexed as
`sd[table * 900 + ((i*5 + j)*5 + k)*6 + l]`. For a bonded pair
(p, q = p.n3) the index is `(type(q.n3), type(q), type(p), type(p.n5))`;
H-bonding and coaxial-stacking ranges use `(0, type(q), type(p), 0)`; nonbonded
excluded volume `(NO_TYPE, type(q), type(p), NO_TYPE)`; cross stacking
`(type(q.n3), type(q), type(p), type(p.n3))` (3'3') and the n5 analogue (5'5').
Types use the oxDNA order A=0, G=1, C=2, T=3 (separate `ptype` array); sites:
backbone `-0.34 a1 + 0.3408 a2`, stacking `0.37 a1`, base `0.43 a1` (A, G) /
`0.37 a1` (C, T).

### Upstream quirks found (c2c74cc0)

- **Stacking dihedral gradient (CPU and CUDA).** The cos(phi1)/cos(phi2)
  derivatives are written in terms of the stacking-site separation with
  `GAMMA = POS_STACK - POS_BACK = 0.74` (oxDNA1/2 stacking site 0.34), but the
  oxDNA3 stacking site is at 0.37, so the force/torque is not the exact
  gradient of the energy whenever `cos(phi) < 0` (f5 active, e.g. frayed ends):
  errors of ~1e-4 (force) to ~5e-4 (torque) of the largest stacking
  force/torque in `fd_test`; on thermal N8 frames the two gamma values give
  per-particle stacking torques differing by up to ~1e-2 (relative). This code reproduces upstream by default (bit-level agreement with
  the standalone forces); `dna3_consistent_gamma = 1` uses gamma = 0.77, which
  makes the forces exact (FD error 3e-9) but no longer identical to upstream.
- **Cross-stacking energy is discontinuous.** The 3'3' / 5'5' branches are
  gated on `cos(theta7) > 0 && cos(theta8) > 0` resp. both `< 0`; when one of
  the two changes sign inside the radial range the energy jumps (a single
  ~1e-2 jump dominates the NVE drift in `fd_test`).
- `use_average_seq` defaults to true upstream; for DNA3 the averaged stacking
  depth is computed in the constructor from the uninitialised `_T` (observed:
  `_T = 0`, i.e. eps = 1.3448 at any T) and the H-bond depth is the oxDNA1 value
  1.077.
- Parameters read from the file but overwritten by the enslaved-parameter step
  (e.g. every `*_TS`, `STCK_RLOW/RHIGH`, `CRST_RLOW/RHIGH`); `EXCL_B*`,
  `EXCL_RC*`, `*_TC`, `*_B`, `FENE_DELTA2` and `*_BLOW/BHIGH` are
  commented out in the reader. `HYDR_THETA3`/`THETA8` alias `THETA2`/`THETA7`
  (same table index), so the later key wins.
- The CUDA kernels (DNA2 and DNA3) accumulate the Debye-Hueckel energy into
  `F.w` with a negative sign (`F -= Ftmp` with `Ftmp.w = +E`); forces are
  unaffected.
- `DNA3Interaction::init()` computes the neighbour cutoff from `POS_MM_BACK1`
  for the base site offset (0.34 instead of 0.43); harmless because the
  Debye-Hueckel range dominates.
- The upstream CPU `DNA3` interaction uses f4 interpolation meshes (~5e-5
  relative force differences vs. the analytic CUDA form); `DNA3_nomesh` is the
  analytic variant (its coaxial terms still use the oxDNA2 meshes).
- `max_backbone_force`, `major_minor_grooving` and `hb_multiplier` are not
  supported here for oxDNA3 (a warning is printed). Dummy bases (btype 4) and
  integer base types in the topology are not supported.
