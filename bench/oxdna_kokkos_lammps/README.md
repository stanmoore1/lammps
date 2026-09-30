# oxDNA-Kokkos (LAMMPS-faithful variant)

> **This is the `oxdna_kokkos_lammps` benchmark — the LAMMPS-faithful sibling of
> `bench/oxdna_kokkos`.** It reads the *same* input files and produces the *same*
> oxDNA energy output (`step time U K total`, per nucleotide) as `bench/oxdna_kokkos`,
> but its internal force-kernel *structure* mirrors the **LAMMPS KOKKOS** oxDNA
> implementation instead of the original CUDA standalone. The physics functions are
> reused verbatim, so energies agree; only *which kernel calls them and how data is
> read* changes. Use it to benchmark the LAMMPS kernel fragmentation head-to-head
> against the CUDA-faithful `bench/oxdna_kokkos`.

All three models are available: oxDNA1, oxDNA2 and the sequence-dependent
**oxDNA3** (upstream `DNA3_nomesh` physics, identical to the sibling's oxDNA3;
kernel structure of the LAMMPS `oxdna3/*` KOKKOS styles). See
[oxDNA3](#oxdna3) for its kernels, inputs and validation.

## How this differs from `bench/oxdna_kokkos` (CUDA-faithful)

The kernel structure tracks the LAMMPS KOKKOS oxDNA styles on LAMMPS branch
**`oxdna3KK-kk-fixes` (2afff57fb5, 2026-09-30)**. With `lammps_overhead = 1`
the bench also reproduces the framework around the kernels (per-type
coefficient tables, ghost atoms and the per-step communication, the binned
half/newton neighbor build with the per-style trimmed lists, the rebuild
decision, the Verlet sequence); see
[Fidelity to LAMMPS KOKKOS](#fidelity-to-lammps-kokkos) for the complete list,
what is not mirrored and why, and a kernel-by-kernel comparison against a
real LAMMPS run.

The table below describes the oxDNA1/2 kernels; the oxDNA3 kernels follow the
same pattern (see [oxDNA3 kernel structure](#oxdna3-kernel-structure-mirrors-lammps-oxdna3)).

| Aspect | `bench/oxdna_kokkos` (CUDA-faithful) | `bench/oxdna_kokkos_lammps` (this, LAMMPS-faithful) |
|---|---|---|
| Body frames (a1,a2,a3) | recomputed from the quaternion inside every force kernel | **LRF precompute pass** (`compute_lrf`, mirrors `fix OXDNA/LRF`): one thread/atom stores a1/a2/a3 in `nx,ny,nz`; every force kernel *reads* these |
| Nonbonded operator | one *fused* edge kernel computing excv + hbond + xstk + coaxstk + dh per pair | **one kernel per LAMMPS pair style**: `excv`, `hbond`, `xstk`, `coaxstk`, `dh` |
| Neighbor list | flat edge list (one thread per pair), bonded pairs excluded | half list per atom (`d_num_neigh`/`d_neigh_matrix`, HALFTHREAD) that **keeps bonded 1-2 pairs** with the LAMMPS special bit set (LAMMPS uses `special_flag = 2` for oxDNA). Every kernel decodes it (`ox_sbmask`/`OX_NEIGHMASK`) and applies `special_lj = 0` |
| excv | nonbonded pairs only; bonded excluded volume in the bonded kernel | per atom over the half list, **including bonded pairs**. Backbone-backbone is multiplied by `special_lj` (0 for bonded), and the other three site pairs *are* the bonded excluded volume, as in LAMMPS. Term order bkbk, bk-bs, bs-bk, bs-bs. Before bs-bs the NEW per-pair tetramer topology test runs (`bonds(b).n3 == a`, ...). Atom a accumulates in registers and is flushed once; atom b gets atomics per active term |
| hbond / xstk / oxDNA2 coaxstk | inside the fused edge kernel | one thread per **screened pair** (`fix OXDNA/NPAIR`): packed `uint64` `(a << 32) \| b_raw` entries (special bits kept; those pairs exit early) whose COM distance is within the largest hbond / xstk / coaxstk pair cutoff (a COM range: site cutoff + site offsets) **+ the full LAMMPS skin** (`2 * verlet_skin`), screened from the master list and rebuilt only when the neighbor lists rebuild; all updates through `ScatterAtomic` access |
| oxDNA1 coaxstk | inside the fused edge kernel | per atom over its own trimmed half list (LAMMPS `pair oxdna/coaxstk` does not use the screen) |
| dh | inside the fused edge kernel | per atom over the half list. Special pairs are skipped first, then `rsq` is tested against the cutoff, then `rinv = rsqrt(rsq)`. A per-atom `qeff` (0.5 at strand ends with half-charged ends) is hoisted for a and loaded for b after the cutoff. Atom a is accumulated in registers |
| Bonded | one fused per-particle gather kernel (FENE + bonded excv + stacking) | two styles: `stk` (`pair oxdna/stk`) and **FENE only** (`bond oxdna/fene`); the bonded excluded volume is in excv |
| Launch policy | tunable LaunchBounds on the force kernels | as LAMMPS: `OxdnaRangePolicy` (`LaunchBounds<64,1>` CUDA / `<128,1>` HIP) on **excv and dh only**; every other kernel uses a plain `RangePolicy` |
| Precision | `c_number` everywhere | `c_number` compute type (`KK_FLOAT`) plus a separate `c_acc` accumulation type (`KK_ACC_FLOAT`) for forces, torques, the atomics and the register accumulators; `-DOXDNA_MIXED_PRECISION=ON` = LAMMPS `SINGLE_DOUBLE` |

**Per-step kernel sequence** (LAMMPS `pair_style hybrid/overlay` order, then the
bond style):

    Pair:  LRF -> excv -> stk -> hbond -> xstk -> coaxstk -> dh
    Bond:  fene

(`lammps_overhead`: `[prime_neighs_pair] excv`, `[stk's prime_neighs_bond] stk`,
`[fene's prime_neighs_bond] fene` on rebuild steps; with `lammps_ghosts` the LRF
and the screen rebuild run as `pre_force` between the force clear and the pair
styles, see the full per-step sequence below.)

The screened pair list is rebuilt only on neighbor-list rebuild steps. Energy
reductions (`parallel_reduce`) only run on output steps, as LAMMPS only reduces
when `eflag` is set.

The physics helpers are the validated standalone-oxDNA ports shared with
`bench/oxdna_kokkos`:
- `add_excv_contrib`, `hbond_pair`, `crst_pair`, `cxst_pair`, `dh_pair` and the
  F3 excluded-volume math in `forces/dna_forces.h`;
- `bonded_fene`, `bonded_stk` in `forces/bonded.h`.

So the printed oxDNA energy matches `bench/oxdna_kokkos` to FP round-off. It is
exact on the oxDNA1 8bp duplex. On the oxDNA2 N8 case it is identical through
thousands of steps, then drifts only at the level of floating-point
operation-reordering chaos.

### LAMMPS-only physics option: terminal coaxial stacking

LAMMPS `pair oxdna2/coaxstk` (CPU and KOKKOS, since 3c86796749 "Introduced
terminal criterion" and 78d6dcb303 "3'3'/5'5' blunt end stacking") departs from
the standalone oxDNA2 (and oxDNA3) coaxial stacking in two ways:
- it only acts between two **strand-terminal** nucleotides;
- theta4 gets a second, mirrored lobe.

`lammps_coaxstk_terminal = 1` enables both, for oxDNA2 and for oxDNA3 (LAMMPS
`oxdna3/coaxstk` inherits the oxDNA2 kernel) and, since `oxdna3KK-kk-fixes`, for
oxDNA1 (`pair oxdna/coaxstk/kk`: the terminal test of atom a returns before any
load of its frame, that of b skips the neighbour, and theta4 gets the same
mirrored lobe; `pair_oxdna_coaxstk_kokkos.cpp:254, 300, 364`). The default (0)
keeps standalone physics.

On intact duplexes coaxial stacking is zero, so the option does not change
energies; on the nicked duplex it gives the same energy (the nick ends are
terminal, and the mirrored lobe vanishes for normal stacking). Its main effect
is on cost: in LAMMPS almost every screened coaxstk pair exits after 4 integer
loads.

---

A portable, GPU-ready standalone implementation of the [oxDNA](https://github.com/lorenzo-rovigatti/oxdna)
coarse-grained DNA model, written with [Kokkos](https://github.com/kokkos/kokkos).
It is intended as a compact benchmark and reference port: the force field
faithfully reproduces the standalone oxDNA **oxDNA1** and **oxDNA2** models
(validated term-by-term, see [Validation](#validation)) and, through the
sibling's port, **oxDNA3** (see [oxDNA3](#oxdna3)), while the data layout
and kernels are structured for performance on CPUs (Serial/OpenMP) and GPUs
(CUDA) through a single Kokkos code base.

Each nucleotide is a rigid body (center of mass + orientation quaternion) with
three interaction sites (backbone, base, stacking) derived from its orientation.

## What it computes

Molecular dynamics in the NVE or NVT (Brownian thermostat) ensemble, with the
full oxDNA interaction set:

| Interaction | Type | oxDNA1 | oxDNA2 | oxDNA3 |
|---|---|:---:|:---:|:---:|
| FENE backbone bond | bonded | yes | yes | yes (sequence-dependent r0, Delta) |
| Bonded excluded volume (base-base, base-back, back-base) | bonded | yes | yes | yes (tetramer tables) |
| Stacking (F1 radial x F4 angles x F5 dihedrals) | bonded | yes | yes | yes (tetramer tables) |
| Nonbonded excluded volume (4 site pairs) | nonbonded | yes | yes | yes (type-pair tables) |
| Hydrogen bonding (F1 x 6 angular terms, Watson-Crick) | nonbonded | yes | yes | yes (pair dependent) |
| Cross-stacking (F2 x 6 angular terms) | nonbonded | yes | yes | yes (3'3' + 5'5' channels, tetramer) |
| Coaxial stacking | nonbonded | yes (+cosphi3) | yes (harmonic theta1) | yes (sequence-dependent K) |
| Debye-Hueckel electrostatics | nonbonded | - | yes | yes |
| Grooved backbone site (major/minor grooving) | geometry | - | yes | yes |
| Purine / pyrimidine base sites (0.43 / 0.37) | geometry | - | - | yes |

oxDNA2 additionally uses its own well depths (`HYDR_EPS`, stacking ε), the
`FENE_R0_OXDNA2` bond length, and a salt-dependent Debye length.

All model constants are taken directly from the standalone oxDNA `src/model.h`
(lj/reduced units), so results are directly comparable.

### Components

| File | Purpose |
|---|---|
| `src/main.cpp` | CLI entry point |
| `src/simulation.h` | MD driver: I/O, force evaluation, time loop |
| `src/integrator.h` | Velocity-Verlet + quaternion (lab-frame) orientation update |
| `src/thermostat.h` | Brownian ("John") thermostat (optional, NVT) |
| `src/neighbor_list.h` | Lean cell-list neighbor matrix (minimum image) + the `fix OXDNA/NPAIR` screened pair list |
| `src/lammps_comm.h` | `lammps_ghosts`: single-rank LAMMPS `CommKokkos` (pbc, exchange, borders, map, fused forward / reverse self communication) |
| `src/lammps_neigh.h` | `lammps_ghosts`: LAMMPS `NBinKokkos` / half stencil / `NPairKokkos` half/bin/newton build, `check_distance`, `NeighBond` `bond_all` |
| `src/lammps_framework.h` | `lammps_ghosts`: the `VerletKokkos` pieces around the force kernels (rebuild, forward / reverse comm, force clear, virial, decide) |
| `src/forces/tables.h` | `lammps_tables`: per-type (5x5) and tetramer (5^4) coefficient tables of the oxDNA1/2 kernels |
| `src/particles.h`, `src/types.h` | SoA particle storage, quaternion / box types |
| `src/forces/params.h` | Force-field parameters (`make_oxdna1_params`, `make_oxdna2_params`) |
| `src/forces/mf_oxdna.h` | Modulation functions F1–F6 and derivatives |
| `src/forces/bonded.h` | Bonded gather kernel: FENE + bonded excluded volume + stacking (one thread per particle, reads its n3/n5 neighbours, no atomics — mirrors oxDNA's `dna_forces_edge_bonded`) |
| `src/forces/dna_forces.h` | Nonbonded kernel: excv + H-bond + cross + coaxial + Debye–Hückel |
| `src/forces/params_dna3.h` | oxDNA3 parameters: host port of upstream `DNA3Interaction` setup (tetramer tables), screen range (copied from `bench/oxdna_kokkos`) |
| `src/forces/dna3_forces.h` | oxDNA3 physics functions (`dna3::bonded_part`, `particle_particle_interaction`, ...; copied from `bench/oxdna_kokkos`) plus `bonded_pair` (one bond, torques on both ends) and the optional LAMMPS blunt theta4 lobe |
| `src/forces/dna3_kernels.h` | oxDNA3 LAMMPS-structured kernels and per-step drivers |
| `src/forces/orient.h` | Quaternion → body-axis vectors |
| `src/io/topology_reader.h`, `src/io/config_reader.h` | oxDNA `.top` / `.conf` readers |

## Building

Requires CMake ≥ 3.20 and a C++20 compiler (Kokkos 5.0 requires C++20). Kokkos
is taken from the bundled LAMMPS tree (`lib/kokkos`) by default, or any
installed/standalone Kokkos via `-DKOKKOS_SOURCE_DIR=<path>` (or `find_package`).

```bash
cd bench/oxdna_kokkos_lammps

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

The executable is `build/oxdna_kokkos_lammps`. Useful CMake options:

- `-DOXDNA_MIXED_PRECISION=ON` — compute in `float`, accumulate forces / torques /
  energies in `double` (LAMMPS `SINGLE_DOUBLE`, `-DKOKKOS_PREC=MIXED`).
- `-DOXDNA_SINGLE_PRECISION=ON` — `float` everywhere (LAMMPS `SINGLE_SINGLE`).
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
./build/oxdna_kokkos_lammps <input_file>
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
| `seq_dep_file`       | see [oxDNA3](#oxdna3) | oxDNA3 sequence-dependence file |
| `use_average_seq`    | 0 | oxDNA3: `1` -> averaged (sequence-independent) tables |
| `dh_half_charged_ends`, `dh_lambda`, `dh_strength`, `debye_huckel_rhigh` | 1, 0.3616455, 0.0543, 3 lambda | oxDNA3 Debye-Hueckel options (upstream keys) |
| `dna3_consistent_gamma` | 0 | oxDNA3, bench-only: exact stacking cos(phi) gradient (see [oxDNA3](#oxdna3)) |
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
| `lammps_overhead`    | 0 | `1` -> add the LAMMPS framework costs (see [below](#lammps_overhead-toggle-isolating-the-framework-cost)) |
| `fuse_hbond_xstk`    | 0 | `1` -> bench-only fused hbond+xstk screened-pair kernel (no LAMMPS equivalent) |
| `lammps_coaxstk_terminal` | 0 | `1` -> oxDNA1/2/3 LAMMPS-only terminal-nucleotide coaxial stacking + blunt-end theta4 lobe (changes energies) |
| `lammps_tables` | = `lammps_overhead` | `1` -> oxDNA1/2 kernels read their coefficients from per-type device tables (LAMMPS layout; same values) |
| `lammps_ghosts` | = `lammps_overhead` | `1` -> ghost atoms + per-step communication + LAMMPS neighbor build + Verlet sequence (needs `lammps_overhead = 1`; same physics) |
| `lammps_cutoff` | 0 | lean mode (no ghosts): `1` -> neighbor-list radius = LAMMPS `cutforce` (the largest pair style cutoff, a COM range since kk-fixes e1a8c85f05) `+ 2 * verlet_skin` instead of `max(cutoff, exact COM range)`; same physics. The ghost mode always uses the LAMMPS cutoffs (per-style trimmed lists) |
| `comm_cutoff` | 0 | ghost cutoff (LAMMPS `comm_modify cutoff`); 0 = the list radius (LAMMPS default) |
| `neigh_every`, `neigh_check` | 1, 1 | `lammps_ghosts`: LAMMPS `neigh_modify every N check yes/no` (`delay 0`) |
| `lammps_integrator` | 0 | `1` -> `fix nve/asphere/kk` (Richardson quaternion update, bonus data; `lammps_ghosts` only; changes the dynamics) |
| `lammps_mass`, `lammps_shape` | 1, 1.5811 | rmass and ellipsoid radii for `lammps_integrator` (defaults give unit mass and unit inertia, the bench convention) |

Paths are resolved relative to the working directory (run from the case
directory, as with the reference oxDNA). oxDNA value expressions are supported:
`$(key)` substitutes another key's value and `${ ... }` evaluates `+ - * / ()`
arithmetic, e.g. `print_energy_every = ${$(steps) / 100}`. The `.top` lists
`<N> <N_strands>` then one `<strand_id> <base> <n3> <n5>` line per nucleotide;
the `.conf` has `t = …`, `b = Lx Ly Lz`, `E = …`, then one line per nucleotide
with position, `a1`, `a3`, and optionally velocity and angular momentum.

Example (the bundled oxDNA2 cases each ship an `input` file):

```bash
cd tests/N8 && ../../build/oxdna_kokkos_lammps input
```

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
LAMMPS `timer full`), so prefer `timing = 0` for production throughput numbers.
The sections are LAMMPS' own timer sections:

```
Kernel timing breakdown:
Section                |   time (s) |  %loop |    us/step
------------------------------------------------------------
Pair                   |    13.0877 |  91.08 |  13087.702
Bond                   |     0.7786 |   5.42 |    778.607
Neigh                  |     0.0207 |   0.14 |     20.733
Modify (integ+thermo)  |     0.4826 |   3.36 |    482.639
...
```

| This code | LAMMPS section(s) |
|---|---|
| `Pair` | `Pair`: `OXDNA/LRF` pre_force + `oxdna*/excv`, `oxdna*/stk`, `oxdna*/hbond`, `oxdna*/xstk`, `oxdna*/coaxstk`, `oxdna2/dh` |
| `Bond` | `Bond` (`oxdna*/fene`) |
| `Neigh` | `Neigh` (incl. the `OXDNA/NPAIR` screen rebuild; with `lammps_ghosts` the decide check, bins, list and bond topology) |
| `Comm` (`lammps_ghosts` only) | `Comm` (pbc, exchange, borders, forward and reverse communication) |
| `Modify (integ+thermo)` | `Modify` (`nve/asphere` + thermostat; with `lammps_ghosts` also the LRF and the NPAIR screen, which LAMMPS runs in `pre_force`) |
| `Output` | `Output` |

(Without ghosts the LRF pass is timed under `Pair`; LAMMPS runs it in
`pre_force`, which it times under `Modify`, as the ghost mode does.)

`timesteps/s` is the most directly comparable metric to LAMMPS' `Performance:`
line. On CPU backends `Kokkos::fence()` is a no-op, so the section times are
exact; on the CUDA backend each section boundary fences (like LAMMPS `timer
full`), which slightly inflates the loop time versus an untimed run.

## Validation

Build with `-DOXDNA_BUILD_TESTS=ON` and run from this directory.

- **`./build/fd_test`** -- self-checking suite for all three models: analytic
  forces & torques vs. central finite differences of the energy (every term
  group; for oxDNA3 every kernel separately, in lean and `lammps_overhead`
  mode, see [oxDNA3 validation](#oxdna3-validation)), NVE energy conservation,
  and thermostat temperature (equipartition). Exits non-zero on failure. This
  is what CI runs.
- **`./build/xcheck <model> <T> <salt> <top> <conf> [ft_out|-] [overhead]
  [--overhead] [--only=<term>] [--consistent-gamma] [--average-seq]
  [--seq=<file>] [--terminal]`** -- prints the potential energy (total and per
  group; for oxDNA3 also every term, in the column order of the standalone
  `potential_energy split = true` observable and of the sibling's `xcheck`)
  and optionally dumps per-particle force/torque (lab frame, 17 significant
  digits), for direct comparison against the standalone oxDNA and
  `bench/oxdna_kokkos`. `--only=<term>` (oxDNA3: `fene bexc stck nexc hb crst
  cxst dh excv`) restricts the dump to one term. `--tables=0|1`
  (`lammps_tables`, default = overhead), `--ghosts` (the `lammps_ghosts` path:
  ghosts, binned list, bond topology, reverse communication before the dump;
  needs `--overhead`), `--comm-cutoff=<x>`, `--shift=<dx,dy,dz>` (translate and
  fold into [0, L) so that strands cross the periodic boundaries);
  `--terminal` now works for all models.

`fd_test` also FD-checks a nicked 8bp duplex (`tests/8bp_nicked`, coaxial
stacking across the nick) for oxDNA1, oxDNA2, oxDNA1 / oxDNA2 with the LAMMPS
terminal coaxial stacking, lean and `lammps_overhead` (with tables), and the
**ghost-atom path**: the nicked duplex folded into [0, L) so that it crosses
the periodic boundaries (oxDNA1, oxDNA2, oxDNA2 terminal; oxDNA3 nicked +
perturbed, every kernel separately), each FD evaluation forward-communicating
the displaced positions / quaternions and reverse-communicating the ghost
forces / torques (FD errors ~7e-9 / ~2e-9, as without ghosts). The tight FD
tolerances need a double-precision build (single precision uses h = 2e-3 and
passes with the loose tolerances). The FD reference forces are now taken with
`create_mirror` (a `create_mirror_view` aliased the device force views on host
backends, so the displaced evaluations overwrote the reference); with this fix
the oxDNA1/2 FD errors are ~7e-9 (force) / ~2e-9 (torque) instead of ~1e-4.

**This revision (kk-fixes sync, tables, ghosts) against the previous one
(`oxdna3KK` sync), double precision, Serial.** `xcheck` on the 8bp duplex and
the nicked duplex (oxDNA1, oxDNA2), N8 and N512 (oxDNA2), the oxDNA3 8bp,
nicked + perturbed, N8 and N64 cases:
- lean and `lammps_overhead` without ghosts (tables on): total energy and every
  17-digit force / torque component **bitwise identical**, except N512 lean /
  overhead, where the previous revision was wrong (see below);
- `lammps_overhead` with ghosts: |dE|/|E| <= 1.5e-14, max|dF|/max|F| <=
  5e-14, max|dT|/max|T| <= 5e-14 (the largest on the perturbed oxDNA3 duplex,
  energies up to 7.8e3; 1e-15 elsewhere): summation order only (neighbor order
  of the binned build, ghost forces added by the reverse communication);
- configurations shifted across the periodic boundaries (`--shift`: nicked
  duplexes, N8, N512, N64; 36-1384 ghost atoms), lean vs ghosts: |dE|/|E| <=
  6.6e-16, dF <= 2.6e-15, dT <= 1.6e-15.

MD (`tests/*/input`, `input_dna3`; Brownian thermostat, energy every 100
steps): lean and `lammps_overhead = 1, lammps_ghosts = 0` are identical to the
previous revision to print precision over the whole runs (8bp 5000, N8 / N64
10000, N512 3000 steps). With ghosts the output is identical up to step 5900
(N8 oxDNA2), 4200 (N8 oxDNA3), 2400 (N64 oxDNA3) and over the whole run (8bp
oxDNA1, N512 3000 steps), then diverges by floating-point reordering chaos:
the summation order differs (binned list, reverse communication) and the
positions are folded only on rebuild steps (LAMMPS `domain->pbc()`), so a
crossing atom's coordinate is rounded differently than with the per-step fold.

Single / mixed precision builds compile, pass `fd_test` (257 checks, loose
tolerances) and run MD in ghost mode; `xcheck --overhead --ghosts` against
double: |dE|/|E| <= 1.1e-5 (N512 single; <= 3.5e-7 on N8 / N64), forces /
torques <= 3.4e-4 / 2.8e-4 (N512), 6.4e-5 / 2.9e-5 (oxDNA3 N8), as before.

**Bug fixed in this revision (lean cell list, also in the previous revision).**
The lean cell list had a fixed `max_per_cell = max(20, 4N/Ncells + 8)` and
silently dropped the particles of a fuller cell; `tests/N512` has 28 particles
in one cell (xcheck, `verlet_skin = 1`), so interactions were missing (total
energy -11091.655 instead of -11093.841, i.e. -1.353962 instead of the N8
value -1.354229 per nucleotide of the tiled system). The build now detects the
overflow and re-bins with enough room; N512 now equals the ghost-atom result
and the per-nucleotide energy of N8. (The MD inputs, with `verlet_skin =
0.5`, did not overflow.) The same code is in `bench/oxdna_kokkos`, which this
revision does not touch. Also fixed: `NeighborList::needs_rebuild` (used by
`fd_test`'s MD) tested a `Max` reduction result against 0, which is `INT_MIN`
when nothing moved, so it rebuilt on every step (harmless, now correct).

Cross-checked against the compiled standalone oxDNA on an 8bp duplex
(average-sequence, T = 0.1; oxDNA2 at salt = 0.5):

| Quantity | oxDNA1 | oxDNA2 |
|---|---|---|
| Total potential energy (per particle) | matches to ~1e-5 | matches to ~1e-5 |
| Per-particle force / torque vectors | ~5e-5 / ~4e-5 (rel.) | ~6e-5 / ~1e-4 (rel.) |
| FD force/torque = -grad E | ~7e-9 / ~2e-9 (rel.) | ~7e-9 / ~2e-9 (rel.) |
| NVE total-energy drift | ~1e-6 | ~1e-6 |

Continuous integration (`.github/workflows/oxdna-kokkos.yml`) builds both
benches with the Serial backend and runs their `fd_test` on every change under
`bench/oxdna_kokkos/` or `bench/oxdna_kokkos_lammps/`.

> Note: from a cold start (zero velocities) oxDNA2 needs a smaller timestep
> (`-dt 1e-4`) than oxDNA1 because the Debye–Hückel + grooved backbone make the
> potential stiffer near close approaches; a thermostat or smaller `dt` keeps it
> stable. This does not affect force-evaluation throughput.

## `lammps_overhead` toggle (isolating the framework cost)

The lean standalone above is *faster* than in-tree LAMMPS-KOKKOS because it omits
the LAMMPS framework costs. `lammps_overhead = 1` adds them back; with the
defaults of its sub-switches this is the **most faithful mode** (the physics /
energy output is unchanged, see [Validation](#validation)):

- **Faithful per-bond bonded kernels**: stk / fene switch from the lean
  per-particle gather to LAMMPS' per-**bond** atomic scatter. Each kernel reads
  its atoms and 3'/5' context only from its own copy of the
  `fix OXDNA/PRIME_NEIGHS` bond table (`prime_bond`: a, b, a3p, b5p), then the
  4 type reads and the tetramer coefficients; stk returns before any atomics
  for a non-stacking bond, fene raises the device overstretch flag.
- **Rebuild-step precomputes** (`neighbor->ncalls` gated, as in LAMMPS
  kk-fixes): `prime_neighs_pair` before excv (an `(nrows, maxneigh, 4)` table,
  nrows = rows of the neighbor list), `prime_neighs_bond` before stk **and**
  again before fene (each style keeps its own copy; the fix itself has no
  `pre_force` any more), `prime_neighs_oxdna3_xstk` before oxdna3/xstk. All
  resolve the 3'/5' partners as LAMMPS does: direction test
  `tag(b) != id5p(a)`, flank tags through the atom map.
- **excv**: LAMMPS' tag-based topology test (`tag(a) == id3p(b) && tag(b) ==
  id5p(a)`), `d_ilist` indirection, tetramer branch for bonded base-base pairs.
- **FENE overstretch flag host copy** on energy steps only, device reset when raised.
- `lammps_tables` (default on): per-type coefficient tables (below).
- `lammps_ghosts` (default on): ghost atoms, communication, LAMMPS neighbor
  build and the VerletKokkos step sequence (below).

Always on (both modes, because LAMMPS does it the same way): one ScatterView per
kernel (non-duplicated = atomics; the screened-pair kernels use `ScatterAtomic`
access on every backend), special pairs in every list skipped via
`special_lj`, `qeff` in dh.

## Fidelity to LAMMPS KOKKOS

Reference: LAMMPS `origin/oxdna3KK-kk-fixes` (2afff57fb5), `-k on -sf kk -pk
kokkos neigh half newton on` with the GPU code paths (`comm device`, the
screened-pair kernels; on a CPU build: `comm device sort device atom/map
device` and the verification-only `OXDNA_FORCE_GPU=1` patch). Unit mapping:
**LAMMPS skin = 2 x `verlet_skin`** (list radius = cutoff + skin, rebuild when
an atom moved more than skin/2 = `verlet_skin`, exactly the bench's criterion).

### Per-step sequence with `lammps_overhead = 1` (ghosts on)

    [initial_integrate]                      (unless fused into the previous step)
    neighbor->decide()                       every N; check: REDUCE check_distance (x - xhold > skin/2)
    rebuild:  DomainPBC | map clear | exchange (3 x BuildExchangeList) |
              borders (6 x BuildBorderList, Pack/Unpack Border + Bonus) | copy_swap_info |
              map_set (2) | [xhold] | bins of the custom and the master list (2 x MemsetZero,
              BinAtoms) | half/bin/newton build of the custom list | NPairTrim chain (4) |
              half/bin/newton build of the master list | NeighBond bond_all (+ bond list to host)
    else:     forward comm: PackCommSelfFused (x) + PackCommSelfFusedBonus (quat)
    force clear (f, torque over nlocal + nghost)
    pre_force: fix OXDNA/LRF (nlocal + nghost, via the bonus index)
               [fix OXDNA/NPAIR screen: count, scan, count to host, fill]
    Pair:  [prime_pair] excv, [prime_bond] stk, hbond, [prime_xstk (oxDNA3)] xstk, coaxstk, dh
           [energy steps: REDUCE variants + PairVirialFDotRCompute]
    Bond:  [prime_bond] fene [energy steps: overstretch flag to host]
    reverse comm: PackReverseSelfFused (f + torque, atomics)
    final_integrate, fused with the next initial_integrate (one kernel) unless
    this is an output / thermostat / last step

### Aspect by aspect

| Aspect | Status | Notes (LAMMPS evidence: `src/KOKKOS/` of 2afff57fb5) |
|---|---|---|
| Kernel per style, LAMMPS order, launch policies | mirrored | `OxdnaRangePolicy` on excv / dh only |
| Screened pair list (`fix OXDNA/NPAIR`) | mirrored | kk-fixes cutoff: the largest cutoff registered by hbond, xstk (incl. **oxdna3/xstk**) and coaxstk (`request_screen_cutoff(cutone)`, their `init_one()` COM ranges since e1a8c85f05) + the **full** skin (`fix_oxdna_npair_kokkos.cpp` `update_screen_cutsq`); screens a copy of the master list; built in `pre_force` after the LRF (ghost mode); rebuilt when `neighbor->ncalls` changes; packed list sized by the count + 20%. Never shorter than the bench's exact range + 2 x `verlet_skin`, so no pair is dropped |
| `ScatterAtomic` in the pairwise kernels | mirrored | `pair_oxdna_hbond_kokkos.cpp:1126`, `pair_oxdna_xstk_kokkos.cpp:1113`, `pair_oxdna2_coaxstk_kokkos.cpp:983`, oxdna3/xstk; also the oxDNA3 hbond / xstk / coaxstk kernels |
| Bond prime tables: stk and fene each their own copy per rebuild, no fix `pre_force` | mirrored | `pair_oxdna_stk_kokkos.cpp:102`, `bond_oxdna_fene_kokkos.cpp:143`, `fix_oxdna_prime_neighs_kokkos.cpp:73` |
| `prime_neighs_pair` rows = neighbor-list rows | mirrored | `fix_oxdna_prime_neighs_kokkos.cpp:137` |
| oxDNA1 coaxstk terminal filter + blunt lobe | mirrored, under `lammps_coaxstk_terminal` | `pair_oxdna_coaxstk_kokkos.cpp:254,300,364` (physics option, off by default) |
| Per-type coefficient tables | mirrored (`lammps_tables`) | oxDNA1/2: 2D 5x5 tables for excv (3 site pairs), hbond (incl. `epsilon_hb` = 0 gate), xstk, coaxstk, stk (eps, a, b_lo/hi, theta4_0, theta5/6, phi1/2), dh; 4D 5^4 tables for stk (cutoffs, shift, theta4), fene (Delta, r0; k per bond type), bonded base-base excv. One struct per entry (LAMMPS: one View per coefficient); uniform values = same physics. oxDNA3 reads the upstream sequence-dependent 4D tables per pair already |
| Ghost atoms, borders, map | mirrored (`lammps_ghosts`) | `comm_kokkos.cpp` `borders_device` (TeamPolicy + `team_scan` build list, pack/unpack border + ellipsoid bonus, border data x, tag, type, mask, molecule, id3p, id5p, qeff, rmass, ellipsoid flag + shape, quat), `copy_swap_info` (`:1545`), `map_set_device`; ghost cutoff `comm_cutoff` (LAMMPS `comm_modify cutoff`) |
| Per-step forward / reverse communication | mirrored | fused self comm (`atom_vec_kokkos.cpp` `PackCommSelfFused`, `PackReverseSelfFused` with atomics `:1523`, `atom_vec_ellipsoid_kokkos.cpp` `PackCommSelfFusedBonus`; `comm_kokkos.cpp:315`) |
| pbc / exchange on rebuild steps | mirrored | `DomainPBCFunctor`, 3 x `BuildExchangeListFunctor` (nothing leaves on one rank; kk-fixes skips the bonus host work when no bonus data leaves) |
| No minimum image with ghosts | mirrored | positions unwrapped between rebuilds; pairs / bonds reference the interacting image (`SimBox::min_image = false`) |
| Kernels over the LAMMPS ranges | mirrored | LRF over nlocal + nghost (`fix_oxdna_lrf_kokkos.cpp:139`), force clear over nall (`verlet_kokkos.cpp:639`), pair styles over the list (nlocal, `d_ilist`), stk / fene over the bond list |
| Binned half/bin/newton list | mirrored (flat variant) | `NBinKokkos` (binsize = cutneighmax/2, ghost extent, overflow -> regrow), `NStencilBin<half,3d>`, `build_Item` own-bin / ghost-coordinate rule, `special_flag = 2`, `rsq <= cutneighsq(itype,jtype)` table read, maxneighs x 1.2 regrow, scalars to / from the device. **Not mirrored:** the GPU shared-memory variant `build_ItemGPU` (TeamPolicy, 2 bins per team); it cannot be exercised on the Serial backend (team size 1) |
| Bond topology per rebuild | mirrored | `NeighBond::bond_all` with `closest_image()` over the `sametag` chain, `k_bondlist.sync_host()` (`neigh_bond_kokkos.cpp:250`) |
| Rebuild decision | mirrored | `neigh_every` / `neigh_check`; `TagNeighborCheckDistance` REDUCE over nlocal (`neighbor_kokkos.cpp:244`), `TagNeighborXhold` on builds with check |
| Pair style cutoffs | mirrored | kk-fixes e1a8c85f05 / 1a43ca2e27: `init_one()` returns the COM range, site cutoff + the offsets of the two sites of the selected model (`forces/lmp_cuts.h`; oxDNA1/2 from the LAMMPS coefficients the bench uses, oxDNA3 from the bench's tables and sites). N8 check: excv 1.674676, stk 0.956 / 1.024365, hbond 1.583775 / 1.643775, xstk 1.5 / 1.614251, coaxstk 1.302222 / 1.332233, dh 3.264295 (oxDNA2 / oxDNA3), identical to LAMMPS' values. These are the exact ranges of the bench physics, so no option drops a pair |
| Per-style trimmed lists (`pair_modify neigh/trim`, default since e2f233566c) | mirrored (ghost mode) | `Neighbor::morph_copy_trim()`: the style with the largest cutoff owns the master list (default bins, cutforce + skin; fix OXDNA/NPAIR copies it); the next-largest gets its own half/bin/newton build with its own bins and stencil (`NBin` / `NStencil` `cutoff_custom`: a custom-cutoff list above `cutneighmin` may not trim from the master list); every smaller one is an `NPairTrimKokkos` of the next-larger one. oxDNA2/3: dh master, excv built, hbond <- excv, xstk <- hbond, coaxstk <- xstk, stk <- coaxstk; oxDNA1: hbond master, excv built, xstk <- excv, coaxstk <- xstk, stk <- coaxstk. excv, dh and oxDNA1 coaxstk read their own list; the stk / hbond / xstk / oxDNA2-3 coaxstk trims are built but unused (those kernels use the bond list or the screen), as in LAMMPS. Build order: bins (custom list first), custom build, trims, master build (oxDNA1: LAMMPS runs the master build after the first trim; same kernels). `LammpsNeigh` prints the lists at startup |
| Verlet fuse_integrate / force clear | mirrored | final + initial integrate in one kernel except before output / thermostat / last steps (`verlet_kokkos.cpp:785`); two `Zero` kernels per step |
| Energy / virial steps | mirrored | per-kernel REDUCE, `PairVirialFDotRCompute` REDUCE over nall (virial not printed), FENE flag to host |
| `fix nve/asphere/kk` | optional (`lammps_integrator`) | Richardson iteration through `bonus(ellipsoid(i))`, inertia 0.2 m (s1^2 + s2^2) (`fix_nve_asphere_kokkos.cpp:22`); changes the dynamics, off by default; with `lammps_mass = 1`, `lammps_shape = sqrt(2.5)` the inertia is 1 as in the bench |
| Atom sorting | not mirrored | the reference inputs use `atom_modify sort 0` (LAMMPS default: every 1000 steps) |
| Thermo output | differs | the bench prints K (a `kinetic_energy` REDUCE) on output steps; the LAMMPS reference thermo prints pe only (host sums of the style energies) |
| `bond_style hybrid` | not mirrored | the xcheck inputs use it (2 lambda kernels + 2 scalar copies per rebuild); the LAMMPS examples use plain `bond_style oxdna2/fene` |
| Brownian thermostat | bench only | standalone oxDNA physics, no LAMMPS equivalent (applied between final and next initial integrate; those steps are not fused) |

### kk-fixes diff (`git diff origin/oxdna3KK origin/oxdna3KK-kk-fixes -- src/KOKKOS src/CG-DNA`)

| Change | Status |
|---|---|
| NPAIR screen: full skin, `request_screen_cutoff(cutone)` (COM ranges since e1a8c85f05, `max_site_offset()` removed), oxdna3/xstk registers, `ncalls` trigger, packed list sized by count | mirrored |
| e1a8c85f05 / 1a43ca2e27: pair cutoffs include the interaction-site offsets (per selected model in the KOKKOS styles) | mirrored (`lmp_cuts.h`) |
| e2f233566c: `trim_flag` removed, sub-style lists trimmed to their own cutoffs | mirrored (ghost mode, see above) |
| 5956b882c3: `Kokkos::fma` instead of the unqualified `fma` helper (LRF, NPAIR screen) | n/a (the bench already calls `Kokkos::fma`) |
| 2afff57fb5: no silent fp32/fp64 conversions (float math on `KK_FLOAT`, literals cast, accumulators `KK_ACC_FLOAT` widened explicitly) | mirrored: the bench's force kernels compute in `c_number` and widen to `c_acc` only at accumulation; the last `double` literals in the kernel helpers (F2 / F6 `0.5`, the stk `1e-12` guards, the box half length) are now `c_number`. `clang -Wdouble-promotion` on the single build: no promotion left in the force kernels (only in host setup / I/O / the thermostat); mixed build: only the float -> double widening of accumulations |
| b16e6ca38f: example skins halved (lj 1.0, real 8.518 A) | n/a (inputs; the bench's `verlet_skin` is half the LAMMPS skin: 0.5 = LAMMPS 1.0) |
| `prime_neighs_bond` owned by stk and fene (own copies), fix without `pre_force`; `ncalls` triggers in excv / stk / xstk3 / fene | mirrored |
| `prime_neighs_pair` table rows = `d_neighbors.extent(0)` | mirrored |
| `ScatterAtomic` in hbond / xstk / oxDNA2 coaxstk / oxDNA3 xstk pair kernels (and `ev_tally_xyz<...,PAIRWISE>`) | mirrored (per-atom tallies: n/a) |
| oxDNA1 `coaxstk`: terminal filter + theta4 blunt lobe | mirrored under `lammps_coaxstk_terminal` |
| oxdna3/xstk: theta from `atan2(|cross|, cos)`, no early exit at sin(theta) = 0; hbond / coaxstk: no early exit at sin = 0 | not mirrored: physics (the bench uses the standalone oxDNA angle functions) |
| `sqrtf` / `expf` -> `Kokkos::sqrt` / `exp` (fene, hbond, xstk, coaxstk, stk, `F1_KK`, `F3_KK`, dh) | n/a: the bench always used the `c_number` functions; precision handling as before (`c_number` / `c_acc`) |
| FENE overcompressed energy formula, restart / coeff read, `ntypes == 4` check for oxDNA3 | n/a (FENE overstretch physics: bench keeps the standalone clamp; input handling) |
| `CommKokkos` reverse fused self comm: atomics, skipped with legacy forward comm | mirrored (atomics) |
| exchange: no bonus host work when no bonus atom leaves | mirrored (nothing leaves) |
| `nve/asphere/kk`: no `ELLIPSOID_MASK` modified (fewer DualView syncs), style lookup in `init()` | n/a (no DualViews); integrator optional |
| per-atom energy / virial ScatterViews, `/kk/host` errors, fix deletion / reuse, idc (unique base pairing) refresh, LRF respa error, hybrid atom / bond style fixes, Install.sh, bonus comm offsets, comm_style tiled | n/a (not used by the benchmark) |

### Kernel sequence vs. a real LAMMPS run

N8 (128 nt, box 40, 75 ghost atoms), 300 steps of NVE, `dt = 0.003`, T = 0.1,
salt 0.5, rebuild every 10 steps (`neigh_modify every 10 delay 0 check no` /
`neigh_every = 10`, `neigh_check = 0`), skin 1.0 (`verlet_skin = 0.5`), `comm_modify
cutoff 5.8` (`comm_cutoff = 5.8`), the LAMMPS cutoffs (master list 960 pairs,
excv 912, hbond / xstk 896, coaxstk 816 / 840, stk 736 / 776 for oxDNA2 / oxDNA3),
energy every 100 steps; LAMMPS with plain `bond_style oxdna2/fene` /
`oxdna3/fene`, both logged with the Kokkos Tools kernel / deep-copy logger. On
the Serial backend a DualView sync is no copy, so neither code logs the
host <-> device copies that a GPU run adds for DualViews (the bench skips the
same copies through `dual_sync()`); the explicit `deep_copy`s are logged on
both sides.

| LAMMPS kernel / copy | bench label | regular | rebuild | notes |
|---|---|:-:|:-:|---|
| `FixNVEAsphereKokkosFusedIntegrateFunctor` | `fused_integrate` | 1 | 1 | `Initial` / `Final` / `initial_integrate` / `second_step` around output steps |
| `TagNeighborCheckDistance` (with `check yes`) | `TagNeighborCheckDistance` | 1 | 1 | |
| `AtomVecKokkos_PackCommSelfFused` + `...EllipsoidKokkos_PackCommSelfFusedBonus` | same | 2 | - | |
| `DomainPBCFunctor` | same | - | 1 | |
| copy + `ViewFill` of `atom:map_array` | same | - | 2 | map clear |
| 3 x (copy `k_count`, `BuildExchangeListFunctor`, copy + `ViewFill` copylist_bonus) | same | - | 12 | |
| 6 x (`BuildBorderListFunctor`, `PackBorder`, `PackBorderBonus`, `UnpackBorder`, `UnpackBorderBonus`) | same | - | 30 | |
| 2 x `map_set_device` + 2 error-flag copies | `map_set_device (ghosts / local)` | - | 4 | |
| 2 x (resize copy, `MemsetZeroFunctor`, `NPairKokkosBinAtomsFunctor`, resize copy) | same | - | 8 | bins of the excv list (own bins) and of the master list |
| scalars copy, `NPairKokkosBuildFunctor`, scalars copy | same (half/newton, flat) | - | 3 | excv list |
| 4 x `NPairTrimKokkos` | `NPairTrimKokkos` | - | 4 | hbond <- excv, xstk <- hbond, coaxstk <- xstk, stk <- coaxstk |
| scalars copy, `NPairKokkosBuildFunctor`, scalars copy | same (half/newton, flat) | - | 3 | master (dh) list; fix OXDNA/NPAIR copies it (no kernel) |
| scalars copy, REDUCE `TagNeighBondBondAll`, scalars copy | same | - | 3 | |
| 2 x `Zero` | `VerletKokkos::force_clear (f / torque)` | 2 | 2 | |
| `TagFixOxdnaLRFComputeQuatToXYZ` | `oxdna_lrf` | 1 | 1 | |
| `TagFixOxdnaNpairNeighScreen`, SCAN, count copy, `TagFixOxdnaNpairFill` | `count_screened`, `scan_screened`, copy, `fill_screened` | - | 4 | |
| `...PrecomputePrimeNeighsPair` | `oxdna_prime_neighs_pair` | - | 1 | |
| `PairOxdna(3)ExcvKokkos` | `oxdna(3)_excv` | 1 | 1 | |
| `...PrecomputePrimeNeighsBond` (stk) | `oxdna_stk_prime_neighs_bond` | - | 1 | |
| `PairOxdnaStkKokkos` | `oxdna(3)_stk` | 1 | 1 | |
| `PairOxdnaHbondKokkos` (GPUPair) | `oxdna(3)_hbond` | 1 | 1 | |
| `...PrecomputePrimeNeighsOxdna3Xstk` (oxDNA3) | `oxdna3_prime_neighs_xstk` | - | 1 | |
| `PairOxdna(3)XstkKokkos` (GPUPair / Npair) | `oxdna(3)_xstk` | 1 | 1 | |
| `PairOxdna2CoaxstkKokkos` (GPUPair) | `oxdna2_coaxstk` / `oxdna3_coaxstk` | 1 | 1 | |
| `PairOxdna2DhKokkos` | `oxdna2_dh` | 1 | 1 | |
| `...PrecomputePrimeNeighsBond` (fene) | `oxdna_fene_prime_neighs_bond` | - | 1 | |
| `BondOxdnaFENEKokkos` | `oxdna(3)_fene` | 1 | 1 | |
| `AtomVecKokkos_PackReverseSelfFused` | same | 1 | 1 | |
| energy steps: REDUCE variants, `PairVirialFDotRCompute`, `bond:flag` copy | same | | | |

Totals per step (oxDNA2 / oxDNA3, LAMMPS = bench): regular 14 / 14 launches, no
logged copy (15 with `check yes`); rebuild 89 / 90 events incl. 20 logged
copies (oxDNA3: 90 or 91, on the same steps in both codes); rebuild + energy step 106 events incl. 21 copies (bench +1: its
`kinetic_energy` REDUCE). Before neigh/trim (kk-fixes 392462c401) a rebuild
was 78 / 79 events: the trimmed lists add a second bin pass (4), a second
half/bin/newton build (3) and the 4 trims. Remaining
mismatches, all explained: (1) the bench's output step adds a
`kinetic_energy` REDUCE (it prints K; the reference thermo prints pe only);
(2) names. Sequence, order and counts of every kernel and logged copy are
otherwise identical over all 300 steps (270 regular, 27 rebuild, 3 rebuild +
energy steps; and 297 + 3 steps with `check yes`).

### Serial CPU timings (1 core, NOT a GPU measurement)

Same N8 runs, 10000 steps (1000 rebuilds), `timing = 1` / LAMMPS `timer`,
seconds:

| Section | LAMMPS oxDNA2 | bench oxDNA2 (faithful) | LAMMPS oxDNA3 | bench oxDNA3 (faithful) |
|---|---:|---:|---:|---:|
| Pair | 1.951 | 2.414 | 2.485 | 3.081 |
| Bond | 0.048 | 0.054 | 0.056 | 0.071 |
| Neigh | 0.162 | 0.190 | 0.197 | 0.242 |
| Comm | 0.083 | 0.103 | 0.107 | 0.119 |
| Modify (integrate + LRF + screen) | 0.299 | 0.170 | 0.331 | 0.184 |
| Loop | 2.558 | 2.941 | 3.194 | 3.708 |

(LAMMPS 2afff57fb5 with the GPU code paths forced, `OXDNA_FORCE_GPU=1`; one
run each, so +-5%.) The trimmed per-style lists (neigh/trim) more than double
Neigh in both codes (0.072 -> 0.162 s in LAMMPS, 0.063 -> 0.190 s in the
bench, oxDNA2): every rebuild now bins twice, runs two half/bin/newton builds
and four trims, while the excv / dh kernels read lists cut at their own
cutoffs (excv 912 instead of 960 pairs here; kk-fixes e2f233566c reports
1.2x - 1.7x faster LAMMPS runs overall with trimming). The previous revision's numbers (single list, 392462c401): LAMMPS /
bench Loop 2.585 / 2.896 s (oxDNA2), 2.893 / 3.298 s (oxDNA3).

Lean / `lammps_overhead` without ghosts on the same input: 2.835 / 2.727 s
(oxDNA2), 3.258 / 3.019 s (oxDNA3). The pair kernels evaluate the standalone
oxDNA physics (more expensive than LAMMPS' own functions on a CPU); the
framework sections (Neigh, Comm) are of the same size as LAMMPS'. LAMMPS'
Modify is larger because its Richardson integrator costs more than the
bench's exact rotation.

### Remaining deliberate physics differences (the bench follows standalone oxDNA)

- oxDNA3 hbond / xstk angular smoothing widths (upstream `TS = sqrt(0.81225 / A)`
  vs LAMMPS' oxDNA2 widths) and the upstream cross-stacking sign gate on
  cos(theta7) / cos(theta8) (LAMMPS evaluates both channels unconditionally).
- oxdna3/xstk small-angle handling (`atan2`, no early exit) and the sin = 0
  handling of hbond / coaxstk in kk-fixes: the bench uses the standalone
  functions.
- Coaxial stacking terminal criterion + blunt theta4 lobe: LAMMPS-only, off by
  default (`lammps_coaxstk_terminal`).
- Debye-Hueckel: the bench's `dh_strength` 0.0543 (and the oxDNA3 upstream
  convention) vs LAMMPS' `qeff_dh_pf` 0.815 in the examples; half-charged
  strand ends on by default.
- FENE: the standalone clamp (oxDNA1/2) / NaN (oxDNA3, as upstream) outside
  the FENE range vs LAMMPS' capped-force extension.
- The integrator (unless `lammps_integrator`), see above.

## oxDNA3

`interaction_type = DNA3` (or `DNA3_nomesh`) selects the sequence-dependent
oxDNA3 model. Its **physics is identical to the oxDNA3 of the CUDA-faithful
sibling `bench/oxdna_kokkos`** (a port of upstream oxDNA c2c74cc0, verified
against the standalone `DNA3_nomesh` interaction, see the sibling README): the
parameter setup (`params_dna3.h`) and the physics functions (`dna3_forces.h`,
namespace `dna3`) are copies of the sibling's files. Only the **kernel
structure** differs; it mirrors the LAMMPS KOKKOS `oxdna3/*` styles on LAMMPS
branch `oxdna3KK-kk-fixes` (2afff57fb5), in the same way as the oxDNA1/2 kernels above.
The oxDNA1/2 code paths are untouched: the model is dispatched once per step on
the host (`Simulation::pair_step` / `bond_step`), with no per-pair model
branch.

### Inputs

All oxDNA3 keys of `bench/oxdna_kokkos` are accepted with the same meaning:
`seq_dep_file`, `use_average_seq` (default 0 here, i.e. sequence dependent),
`salt_concentration`, `dh_half_charged_ends`, `dh_lambda`, `dh_strength`,
`debye_huckel_rhigh`, `dna3_consistent_gamma`. The topology reader also reads
the new upstream 5'->3' format (`<N> <N_strands> 5->3`, one sequence per
strand). `lammps_overhead` and `lammps_coaxstk_terminal` work as for oxDNA2;
`fuse_hbond_xstk` has no oxDNA3 variant and is ignored (with a warning).

This bench does not ship its own copy of the 15k-line upstream parameter file:
without a `seq_dep_file` key it uses
`bench/oxdna_kokkos/params/oxDNA3_sequence_dependent_parameters.txt`, whose
absolute path is baked in at configure time (CMake cache variable
`OXDNA_DEFAULT_SEQ_DEP_FILE`), falling back to the paths
`../oxdna_kokkos/params/...` (run from this directory) and
`../../../oxdna_kokkos/params/...` (run from `tests/<case>/`); an error is
raised if none exists. An explicit `seq_dep_file` is used as given, relative to
the working directory. `tests/{8bp_duplex,N8,N64}/input_dna3` mirror the
sibling's inputs (with `seq_dep_file` pointing at the sibling's copy);
`tests/8bp_duplex/` also has the sibling's `test_dna3.conf` (the duplex relaxed
with oxDNA3), `test_dna3_nicked.top` and `test_dna3_pert.conf`.

### oxDNA3 kernel structure (mirrors LAMMPS oxdna3)

Per step (GPU, HALFTHREAD, newton on), in LAMMPS order:

    pre_force: LRF [+ NPAIR screen]                 ([..]: lammps_overhead, rebuild steps only)
    Pair:      [prime_neighs_pair] excv -> [stk's prime_neighs_bond] stk -> hbond ->
               [prime_neighs_oxdna3_xstk] xstk -> coaxstk -> dh
    Bond:      [fene's prime_neighs_bond] fene (+ overstretch flag host copy on
               energy steps, lammps_overhead)

(with `lammps_ghosts` inside the full Verlet sequence of
[Fidelity to LAMMPS KOKKOS](#fidelity-to-lammps-kokkos); the per-type tables
of `lammps_tables` are not needed for oxDNA3, whose physics already reads the
sequence-dependent tetramer tables per pair / bond.)

For a pair `a < b` from the half list or the screened list, the physics is
evaluated with `p = b`, `q = a` (`r = x_a - x_b`), the orientation of the
sibling's CUDA edge list, so the pair-asymmetric tables are used identically.

| Kernel | LAMMPS style | Threads / list | Reads | Scatter | Policy |
|---|---|---|---|---|---|
| excv | `oxdna3/excv` | one per atom a, half list incl. special (1-2) pairs | a in registers; per b: x, a1/a2, oxDNA type, `bonds(b)`; bonded pairs: 2 flank indices (lean: `bonds()`; overhead: the `prime_neighs_pair` table) + 2 flank types | a: registers, flushed once; b: atomics per active site pair | `OxdnaRangePolicy` |
| stk | `oxdna3/stk` | lean: one per particle (gather both bonds, each bond twice); overhead: one per bond | overhead: (a, b, a3p, b5p) from `prime_bond` + 4 type reads | lean: none (own particle); overhead: atomics to a and b, early return before any atomic when not stacking | plain |
| hbond | `oxdna3/hbond` | one per screened pair | special exit, 2 types (complementary test), frames | 12 atomics | plain |
| xstk | `oxdna3/xstk` | one per screened pair | special exit; 4 flank indices (overhead: per-screened-pair `prime_neighs_oxdna3_xstk` table (3'a, 5'a, 3'b, 5'b), built on rebuild steps by its own per-pair kernel; lean: `bonds()`) + 4 flank types | overhead: two rounds as LAMMPS (force + site torques, then the pure torques: 18 atomics); lean: one round of 12 | plain |
| coaxstk | `oxdna3/coaxstk` | one per screened pair | special exit, `bonds()` of both (terminal filter, K selection) + 4 flank types | 12 atomics | plain |
| dh | `oxdna3/dh` (= `oxdna2/dh`) | one per atom, half list | the oxDNA2 `DHFunctor` unchanged: special skip, `rsq` cutoff before `rsqrt`, per-atom `qeff` (0.5 at strand ends, = `dh_half_charged_ends`) | a in registers, b atomics | `OxdnaRangePolicy` |
| fene | `oxdna3/fene` | as stk | as stk | overhead: atomics + device overstretch flag | plain |

Notes on the individual kernels:

- **excv.** Term order as LAMMPS: bkbk (x `special_lj`, i.e. zero for bonded
  pairs), bk(a)-bs(b), bs(a)-bk(b), bs-bs. Nonbonded pairs use the upstream
  nonbonded excluded-volume tables at `(NO_TYPE, type(a), type(b), NO_TYPE)`;
  bonded pairs (topology test `b.n3 == a && a.n5 == b` or the mirrored one) use
  the tetramer tables `(type(3' flank), type(n3 end), type(n5 end), type(5'
  flank))` for all three site pairs, which together are exactly upstream's
  bonded excluded volume (`bonded_part<.., BONDED_EXCLUDED_VOLUME>`; verified
  term by term, see below). Difference to LAMMPS: LAMMPS' `oxdna3/excv` has
  tetramer parameters only for the bonded base-base pair and 2D type-pair
  tables otherwise, and runs its topology test just before bs-bs; here the test
  runs before the first site pair because upstream's bonded bk-bs / bs-bk terms
  are tetramer dependent too.
- **stk / fene.** Lean mode calls `dna3::bonded_part<qIsN3, STACKING or
  BACKBONE>` exactly like the sibling's bonded gather. The per-bond kernels
  use `dna3::bonded_pair<>`, which evaluates the bond once and returns the
  force on the n5 end plus the torques on both ends with the per-side
  arithmetic of `bonded_part<true>` / `bonded_part<false>` (the torque on the
  n3 end cannot be taken from momentum conservation, because upstream's
  stacking cos(phi) derivative is not the exact gradient, see below). FENE is
  FENE only (the bonded excluded volume is in excv) with sequence-dependent
  r0 / Delta; outside the FENE range the energy is NaN as in the sibling and
  upstream CUDA (no clamp); in lammps_overhead mode the device overstretch flag
  is raised when `1 - (r - r0)^2 / Delta^2 < 0.2` (or NaN) and copied to the
  host on energy steps, as for oxDNA1/2.
- **hbond / xstk / coaxstk** call `dna3::particle_particle_interaction<TERM>`
  and scatter `F -> p`, `-F -> q`, `T -> p`, `-T + r x F -> q`. In the two-round
  xstk scatter the site torque on p is `(pos_base(p) a1_p) x F`; the pure torque
  is `T` minus that.
- **coaxstk, `lammps_coaxstk_terminal = 1`.** As for oxDNA2, both parts of the
  LAMMPS-only variant are available for oxDNA3: the terminal filter (both
  nucleotides must be strand ends; checked in the functor before the frames are
  loaded) and the mirrored theta4 lobe `f4(theta4; pi - theta4_0)`
  (`DNA3Params::cxst_t4_blunt`, added to the copied coaxial term together with
  its derivative). Default off = upstream physics.
- **Screened-pair cutoff.** As for oxDNA1/2, the LAMMPS kk-fixes formula: the
  largest cutoff registered by hbond, xstk (`oxdna3/xstk` registers too) and
  coaxstk, i.e. their COM ranges (`DNA3Params::lmp`: site cutoffs from the DNA3
  tables over all flanks that can occur + the base / stacking site offsets of
  the two types), + the full skin (`2 * verlet_skin`), and never less than the
  exact range derived from the
  tables (`screen_cutsq`: the outer radial cutoff plus the COM->site offsets of
  the two nucleotide types, **1.614** with the upstream sequence-dependent
  tables (purine-purine cross stacking, 0.754 + 2 x 0.43), 1.584 with
  `use_average_seq = 1`) + `2 * verlet_skin`. The previous LAMMPS
  discrepancies (xstk not registered, a fixed 0.8 site margin shorter than
  2 x 0.43, a half-skin margin, then a fixed 2 x 0.43 margin on the site
  cutoffs) are fixed in kk-fixes. `xcheck` builds its
  oxDNA3 screen with no skin margin at all (the exact range), so a too short
  screen would show up as a difference to the sibling.

### Known LAMMPS-vs-upstream oxDNA3 physics differences

The bench follows **upstream** physics (like the sibling). When comparing
against LAMMPS runs, be aware of:

- LAMMPS' `oxdna3_lj.cgdna` keeps the oxDNA2 hbond/xstk angular smoothing
  widths, while upstream recomputes every F4 smoothing width as
  `TS = sqrt(0.81225 / A)` from the (sequence-dependent) `A`.
- LAMMPS evaluates both cross-stacking channels (3'3' and 5'5')
  unconditionally, while upstream gates them on the sign of cos(theta7) and
  cos(theta8) (3'3' needs both > 0, 5'5' both < 0), which makes the upstream
  cross-stacking energy discontinuous (see the sibling README).
- Upstream's stacking cos(phi1)/cos(phi2) derivative uses
  `GAMMA = 0.74` (the oxDNA1/2 stacking-site offset) although the oxDNA3
  stacking site is at 0.37; the forces are then not the exact gradient when
  cos(phi) < 0. `dna3_consistent_gamma = 1` (bench-only) uses 0.77 and makes
  them exact.
- The LAMMPS excluded volume differs as described under excv above
  (2D tables except for the bonded base-base pair), and the LAMMPS
  coaxial-stacking terminal filter / blunt lobe are not upstream physics.

### oxDNA3 validation

All numbers from a double-precision Serial build.

**Physics identity with `bench/oxdna_kokkos`.** `xcheck` (this bench, lean and
`lammps_overhead`) against the sibling's `xcheck` (17-digit dumps). Energies:
all 8 split terms and the total; forces/torques: every per-particle component
(max abs difference relative to the largest component; "per-comp" is the
largest relative difference of any component above 1e-6 of the maximum).

| Case | total E / nt | \|dE_tot\| / \|E\| | max \|dF\| / max \|F\| | max \|dT\| / max \|T\| | per-comp |
|---|---|---|---|---|---|
| 8bp `test_dna3.conf`, T = 0.1, salt 0.5 | -1.156022 | 1.9e-16 | 1.8e-16 | 1.9e-16 | 4e-15 |
| 8bp nicked (`test_dna3_nicked.top`) | -1.094713 | 0 | 2.0e-16 | 1.4e-16 | 5e-15 |
| 8bp nicked + perturbed (`test_dna3_pert.conf`, every term active) | -0.672222 | 0 | 2.5e-16 | 1.3e-16 | 4e-15 |
| N8 (128 nt), T = 20C, salt 1.0 | -1.185172 | 3.7e-16 | 2.9e-16 | 3.4e-16 | 3e-14 |
| N64 (1024 nt), T = 20C, salt 1.0 | -1.185172 | 1.9e-16 | 4.3e-16 | 3.4e-16 | 9e-14 |
| N8, `use_average_seq = 1` | -1.371307 | 1.6e-16 | 3.1e-16 | 2.9e-16 | 3e-13 |
| 8bp, stacking cos(phi) active, upstream and `--consistent-gamma` | (FENE NaN in both) | -- | 1.6e-16 | 1.7e-16 | 4e-15 |

Lean and `lammps_overhead` agree with each other to the same level. The bonded
and nonbonded halves of the excv kernel reproduce the sibling's `BEXC` / `NEXC`
split terms separately. Single / mixed precision builds (`-DOXDNA_SINGLE_PRECISION=ON`,
`-DOXDNA_MIXED_PRECISION=ON`): N8 split energies identical to 6 decimals,
forces / torques 6.4e-5 / 2.9e-5 relative to the double-precision sibling.

**`fd_test`** (double): every oxDNA3 kernel separately (fene, excv bonded,
excv nonbonded, stk, hbond, xstk, coaxstk, dh) and the pair / bond / total
groups, in lean and `lammps_overhead` mode, on the relaxed duplex, on the
nicked + perturbed duplex (every term checked to be nonzero), with
`lammps_coaxstk_terminal` (terminal filter; plus a test that turns the nick's
5' end by pi about a1 so that only the mirrored theta4 lobe can stack it: the
coaxial energy is reproduced to 3e-15 and its FD gradient checked), and with
active stacking cos(phi) modulation: FD error <= 3.8e-8 (force) / 5.6e-9
(torque) relative to the largest component (tolerance 1e-6); upstream gamma
with active cos(phi): 9.7e-5 / 5.3e-4 (the upstream quirk, tolerance 5e-3),
consistent gamma 3.1e-9 / 5.8e-10. NVE drift over 3000 steps 7.8e-4 (lean and
overhead; the cross-stacking discontinuity, as in the sibling), equipartition
within 5%. oxDNA1/2 FD errors: ~7e-9 / ~2e-9.

**MD.** `tests/{8bp_duplex,N8,N64}/input_dna3` (10000 steps for N8 / N64, dt =
0.003, Brownian thermostat) are stable in lean and `lammps_overhead` mode. The
energy output equals the sibling's to all printed digits up to step 4000 (N8)
and 3000 (N64; 1e-6 at step 4000), then diverges through floating-point
reordering chaos (different summation order), as for oxDNA2; the 8bp run is
identical over all 5000 steps. Single and mixed precision runs are stable too.

**oxDNA1/2 unchanged.** `xcheck` models 1 and 2 on `tests/8bp_duplex`,
`tests/8bp_nicked`, `tests/N8` and `tests/N512` (lean and overhead; total
energies and 17-digit force/torque dumps) and the MD energy output of
`tests/8bp_duplex/input` and `tests/N8/input` (lean and overhead) are
bit-identical to the build before the oxDNA3 addition.

**Timing (Serial CPU, 1 core -- NOT a GPU measurement).** N64 (1024 nt),
`input_dna3`, 10000 steps, loop time; on a CPU the per-bond scatter is cheaper
than the lean gather (which evaluates every bond twice), the opposite of what
is expected on a GPU:

| Code | loop time (s) | timesteps/s |
|---|---|---|
| `bench/oxdna_kokkos` (CUDA-faithful, oxDNA3) | 21.96 | 455 |
| this bench, lean | 24.44 | 409 |
| this bench, `lammps_overhead = 1` | 21.20 | 472 |

(3000-step `timing = 1` breakdown, us/step: lean Pair 2304 / Bond 117;
overhead Pair 2103 / Bond 44; the same system with oxDNA2: lean Pair 1645 /
Bond 99.)
