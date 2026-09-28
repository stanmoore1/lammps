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
**`oxdna3KK` (ec131639, 2026-09-25)**.

The table below describes the oxDNA1/2 kernels; the oxDNA3 kernels follow the
same pattern (see [oxDNA3 kernel structure](#oxdna3-kernel-structure-mirrors-lammps-oxdna3)).

| Aspect | `bench/oxdna_kokkos` (CUDA-faithful) | `bench/oxdna_kokkos_lammps` (this, LAMMPS-faithful) |
|---|---|---|
| Body frames (a1,a2,a3) | recomputed from the quaternion inside every force kernel | **LRF precompute pass** (`compute_lrf`, mirrors `fix OXDNA/LRF`): one thread/atom stores a1/a2/a3 in `nx,ny,nz`; every force kernel *reads* these |
| Nonbonded operator | one *fused* edge kernel computing excv + hbond + xstk + coaxstk + dh per pair | **one kernel per LAMMPS pair style**: `excv`, `hbond`, `xstk`, `coaxstk`, `dh` |
| Neighbor list | flat edge list (one thread per pair), bonded pairs excluded | half list per atom (`d_num_neigh`/`d_neigh_matrix`, HALFTHREAD) that **keeps bonded 1-2 pairs** with the LAMMPS special bit set (LAMMPS uses `special_flag = 2` for oxDNA). Every kernel decodes it (`ox_sbmask`/`OX_NEIGHMASK`) and applies `special_lj = 0` |
| excv | nonbonded pairs only; bonded excluded volume in the bonded kernel | per atom over the half list, **including bonded pairs**. Backbone-backbone is multiplied by `special_lj` (0 for bonded), and the other three site pairs *are* the bonded excluded volume, as in LAMMPS. Term order bkbk, bk-bs, bs-bk, bs-bs. Before bs-bs the NEW per-pair tetramer topology test runs (`bonds(b).n3 == a`, ...). Atom a accumulates in registers and is flushed once; atom b gets atomics per active term |
| hbond / xstk / oxDNA2 coaxstk | inside the fused edge kernel | one thread per **screened pair** (`fix OXDNA/NPAIR`): packed `uint64` `(a << 32) \| b_raw` entries (special bits kept; those pairs exit early) whose COM distance is within `max(cut_hc) + 0.8` **+ the skin margin**, rebuilt only when the neighbor list rebuilds |
| oxDNA1 coaxstk | inside the fused edge kernel | per atom over the half list (LAMMPS `pair oxdna/coaxstk` does not use the screen) |
| dh | inside the fused edge kernel | per atom over the half list. Special pairs are skipped first, then `rsq` is tested against the cutoff, then `rinv = rsqrt(rsq)`. A per-atom `qeff` (0.5 at strand ends with half-charged ends) is hoisted for a and loaded for b after the cutoff. Atom a is accumulated in registers |
| Bonded | one fused per-particle gather kernel (FENE + bonded excv + stacking) | two styles: `stk` (`pair oxdna/stk`) and **FENE only** (`bond oxdna/fene`); the bonded excluded volume is in excv |
| Launch policy | tunable LaunchBounds on the force kernels | as LAMMPS: `OxdnaRangePolicy` (`LaunchBounds<64,1>` CUDA / `<128,1>` HIP) on **excv and dh only**; every other kernel uses a plain `RangePolicy` |
| Precision | `c_number` everywhere | `c_number` compute type (`KK_FLOAT`) plus a separate `c_acc` accumulation type (`KK_ACC_FLOAT`) for forces, torques, the atomics and the register accumulators; `-DOXDNA_MIXED_PRECISION=ON` = LAMMPS `SINGLE_DOUBLE` |

**Per-step kernel sequence** (LAMMPS `pair_style hybrid/overlay` order, then the
bond style):

    Pair:  LRF -> excv -> stk -> hbond -> xstk -> coaxstk -> dh
    Bond:  fene

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
`oxdna3/coaxstk` inherits the oxDNA2 kernel). The default (0) keeps standalone
physics.

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
| `src/neighbor_list.h` | Cell list + flat Verlet edge list (one thread per pair) |
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
| `lammps_coaxstk_terminal` | 0 | `1` -> oxDNA2/oxDNA3 LAMMPS-only terminal-nucleotide coaxial stacking + blunt-end theta4 lobe |

Paths are resolved relative to the working directory (run from the case
directory, as with the reference oxDNA). oxDNA value expressions are supported:
`$(key)` substitutes another key's value and `${ ... }` evaluates `+ - * / ()`
arithmetic, e.g. `print_energy_every = ${$(steps) / 100}`. The `.top` lists
`<N> <N_strands>` then one `<strand_id> <base> <n3> <n5>` line per nucleotide;
the `.conf` has `t = …`, `b = Lx Ly Lz`, `E = …`, then one line per nucleotide
with position, `a1`, `a3`, and optionally velocity and angular momentum.

Example (the bundled oxDNA2 cases each ship an `input` file):

```bash
cd tests/N8 && ../../build/oxdna_kokkos input
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
| `Neigh` | `Neigh` (incl. the `OXDNA/NPAIR` screen rebuild) |
| `Modify (integ+thermo)` | `Modify` (`nve/asphere` + thermostat) |
| `Output` | `Output` |

(LAMMPS runs the LRF fix in `pre_force`, which it times under `Modify`; here it
is part of `Pair`.)

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
  cxst dh excv`) restricts the dump to one term.

`fd_test` also FD-checks a nicked 8bp duplex (`tests/8bp_nicked`, coaxial
stacking across the nick) for oxDNA1, oxDNA2, oxDNA2 with the LAMMPS terminal
coaxial stacking, and the latter in `lammps_overhead` mode. The tight FD
tolerances need a double-precision build (single precision uses h = 2e-3 and
passes with the loose tolerances). The FD reference forces are now taken with
`create_mirror` (a `create_mirror_view` aliased the device force views on host
backends, so the displaced evaluations overwrote the reference); with this fix
the oxDNA1/2 FD errors are ~7e-9 (force) / ~2e-9 (torque) instead of ~1e-4.

Synced to LAMMPS `oxdna3KK` (this revision), with `xcheck` checked against
the previous (`oxdna-framework-overhead`) structure. On the 8bp duplex
(oxDNA1), N8 and N512 (oxDNA2), total energy and every per-particle
force/torque component agree to all 10 printed digits, in both lean and
`lammps_overhead` mode. MD energy output is identical to print precision
until floating-point reordering chaos sets in (N8: after ~7000 steps).

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
several real LAMMPS framework costs. Setting `lammps_overhead = 1` in the input
adds them back (physics/energy output is unchanged -- verified identical on/off):

- **Faithful per-bond bonded kernels**: stk/fene switch from the lean
  per-particle gather to LAMMPS's per-**bond** atomic scatter.
  - The lean gather has no atomics and computes each bond twice; the LAMMPS
    scatter computes each bond once and scatters it to both endpoints.
  - Each kernel reads its atoms and 3'/5' context only from the
    `fix OXDNA/PRIME_NEIGHS` bond table (`prime_bond`: a, b, a3p, b5p).
  - It then does the 4D ("tetramer") coefficient indexing: 4 type reads and
    uniform table lookups per bond.
  - stk returns before any atomics for a non-stacking bond.
  - fene raises the device overstretch flag.
- **Rebuild-step precomputes** (`neighbor->lastcall` gated, as in LAMMPS):
  - the `prime_neighs_bond` per-bond kernel launched **twice**: by the fix's
    `pre_force` and again by `pair oxdna/stk`, which keeps its own counter;
  - the new excv `prime_neighs_pair` kernel, which writes an
    (N, max_neigh, 4) table;
  - the device-to-host copy of the rebuilt bond list done by
    `neigh_bond build_topology_kk`.
- **excv tetramer branch**: bonded base-base pairs read the prime-pair table,
  two base types and a 4D (5^4) table entry.
- **FENE overstretch flag host copy** on energy steps only (LAMMPS: `eflag ||
  vflag`), with the device reset when raised.

Always on (in both modes, because LAMMPS does it the same way):
- each kernel makes its own ScatterView (non-duplicated = atomics);
- the per-pair topology reads in excv;
- special pairs in every list, skipped via `special_lj`;
- `qeff` in dh.

What it deliberately does NOT model is **ghost atoms + per-step communication**:
the standalone uses minimum-image PBC (`box.wrap`) and processes exactly N atoms,
whereas LAMMPS replicates the boundary shell as ghosts, forward-communicates
positions every step, and runs the LRF fix / neighbour list / pair styles over
`nlocal+nghost`. So:

    (LAMMPS time) - (standalone with lammps_overhead=1) ~= ghost/comm cost,

isolating the fundamental (domain-decomposition) floor from the optimizable
per-step overheads (bond precompute, per-style setup, flag copies).

Also not modelled:
- LAMMPS' `nve/asphere` Richardson integrator: the bench integrates the lab-frame
  angular momentum exactly with unit inertia;
- LAMMPS' FENE overstretch extension: the bench keeps the standalone clamp;
- the fp32 `sqrtf`/`expf` calls that many LAMMPS oxDNA kernels make even in a
  double build (fene, hbond, xstk, coaxstk, stk, `F1_KK`/`F3_KK`, dh).

## oxDNA3

`interaction_type = DNA3` (or `DNA3_nomesh`) selects the sequence-dependent
oxDNA3 model. Its **physics is identical to the oxDNA3 of the CUDA-faithful
sibling `bench/oxdna_kokkos`** (a port of upstream oxDNA c2c74cc0, verified
against the standalone `DNA3_nomesh` interaction, see the sibling README): the
parameter setup (`params_dna3.h`) and the physics functions (`dna3_forces.h`,
namespace `dna3`) are copies of the sibling's files. Only the **kernel
structure** differs; it mirrors the LAMMPS KOKKOS `oxdna3/*` styles on LAMMPS
branch `oxdna3KK` (ec131639), in the same way as the oxDNA1/2 kernels above.
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

    pre_force: LRF | [prime_neighs_bond]            ([..]: lammps_overhead, rebuild steps only)
    Pair:      excv [prime_neighs_pair] -> [stk's prime_neighs_bond] stk -> hbond ->
               [prime_neighs_oxdna3_xstk] xstk -> coaxstk -> dh
    Bond:      fene (+ overstretch flag host copy on energy steps, lammps_overhead)

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
- **Screened-pair cutoff.** The screen range is derived from the DNA3 tables
  (`make_oxdna3_params`): the maximum over hbond (complementary pairs only),
  xstk (3'3' and 5'5') and coaxstk of the outer radial cutoff (over all flanks
  that can occur, incl. `NO_TYPE`) plus the COM->site offsets of the two
  nucleotide types involved (base sites for hbond/xstk, stacking sites for
  coaxstk), then the existing skin margin (`+ verlet_skin`, as for oxDNA1/2).
  With the upstream sequence-dependent tables this is **1.614** (purine-purine
  cross stacking, 0.754 + 2 x 0.43), with `use_average_seq = 1` 1.584 (hbond).
  **LAMMPS discrepancy:** LAMMPS' `fix OXDNA/NPAIR` screen only registers the
  hbond (`cut_hb_hc + 0.8`) and coaxstk (`cut_cxst_hc + 0.8`) ranges;
  `oxdna3/xstk` does not register its cutoff, and the fixed 0.8 site margin
  (0.43 + 0.37) is too small for purine-purine cross stacking (2 x 0.43), so
  its screen (max 1.584 before the skin margin) is shorter than the
  cross-stacking range (1.614 with the upstream tables; about 1.631 with the
  xstk cutoff of LAMMPS' own oxDNA3 parameters). Cross-stacking pairs in that
  gap are only kept by LAMMPS' half-skin margin. The bench uses the correct
  range; `xcheck` builds its oxDNA3 screen with no skin margin at all, so a too
  short screen would show up as a difference to the sibling.
- **Skin margin (oxDNA1/2/3, LAMMPS and bench).** The screen is only rebuilt
  with the neighbour list, and its margin covers one atom's drift (LAMMPS: half
  skin, rebuilding at a drift of half skin; bench: `verlet_skin`, rebuilding at
  a drift of `verlet_skin`), while a pair distance can change by twice that
  between rebuilds (LAMMPS' own comment calls the half-skin margin "a little
  cheeky"). Not an issue in the tests below, noted for completeness.

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
