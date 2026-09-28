# oxDNA-Kokkos (LAMMPS-faithful variant)

> **This is the `oxdna_kokkos_lammps` benchmark — the LAMMPS-faithful sibling of
> `bench/oxdna_kokkos`.** It reads the *same* input files and produces the *same*
> oxDNA energy output (`step time U K total`, per nucleotide) as `bench/oxdna_kokkos`,
> but its internal force-kernel *structure* mirrors the **LAMMPS KOKKOS** oxDNA
> implementation instead of the original CUDA standalone. The physics functions are
> reused verbatim, so energies agree; only *which kernel calls them and how data is
> read* changes. Use it to benchmark the LAMMPS kernel fragmentation head-to-head
> against the CUDA-faithful `bench/oxdna_kokkos`.

## How this differs from `bench/oxdna_kokkos` (CUDA-faithful)

The kernel structure tracks the LAMMPS KOKKOS oxDNA styles on LAMMPS branch
**`oxdna3KK` (ec131639, 2026-09-25)**.

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

`lammps_coaxstk_terminal = 1` enables both. The default (0) keeps standalone
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
(validated term-by-term, see [Validation](#validation)), while the data layout
and kernels are structured for performance on CPUs (Serial/OpenMP) and GPUs
(CUDA) through a single Kokkos code base.

Each nucleotide is a rigid body (center of mass + orientation quaternion) with
three interaction sites (backbone, base, stacking) derived from its orientation.

## What it computes

Molecular dynamics in the NVE or NVT (Brownian thermostat) ensemble, with the
full oxDNA interaction set:

| Interaction | Type | oxDNA1 | oxDNA2 |
|---|---|:---:|:---:|
| FENE backbone bond | bonded | ✅ | ✅ |
| Bonded excluded volume (base–base, base–back, back–base) | bonded | ✅ | ✅ |
| Stacking (F1 radial · F4 angles · F5 dihedrals) | bonded | ✅ | ✅ |
| Nonbonded excluded volume (4 site pairs) | nonbonded | ✅ | ✅ |
| Hydrogen bonding (F1 · 6 angular terms, Watson–Crick) | nonbonded | ✅ | ✅ |
| Cross-stacking (F2 · 6 angular terms) | nonbonded | ✅ | ✅ |
| Coaxial stacking | nonbonded | ✅ (+cosphi3) | ✅ (harmonic θ1) |
| Debye–Hückel electrostatics | nonbonded | — | ✅ |
| Grooved backbone site (major/minor grooving) | geometry | — | ✅ |

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
| `interaction_type`   | DNA | `DNA`/`DNA1` → oxDNA1, `DNA2` → oxDNA2 |
| `salt_concentration` | 0.5 | mol/L (oxDNA2) |
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
| `lammps_coaxstk_terminal` | 0 | `1` -> oxDNA2 LAMMPS-only terminal-nucleotide coaxial stacking + blunt-end theta4 lobe |

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

- **`./build/fd_test`** — self-checking suite for both models: analytic forces &
  torques vs. central finite differences of the energy (every term), NVE energy
  conservation, and thermostat temperature (equipartition). Exits non-zero on
  failure. This is what CI runs.
- **`./build/xcheck <model> <T> <salt> <top> <conf> [ft_out]`** — prints the
  potential energy (total and per group) and optionally dumps per-particle
  force/torque, for direct comparison against the standalone oxDNA
  `potential_energy split = 1` and `particle_force_torque` (`lab_frame = 1`)
  observables.

`fd_test` also FD-checks a nicked 8bp duplex (`tests/8bp_nicked`, coaxial
stacking across the nick) for oxDNA1, oxDNA2, oxDNA2 with the LAMMPS terminal
coaxial stacking, and the latter in `lammps_overhead` mode. The FD checks need a
double-precision build (h = 1e-5 is below float resolution).

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
| FD force/torque = −∇E | ~1e-4 (rel.) | ~1e-4 (rel.) |
| NVE total-energy drift | ~1e-6 | ~1e-6 |

Continuous integration (`.github/workflows/oxdna-kokkos.yml`) builds with the
Serial backend and runs `fd_test` on every change under `bench/oxdna_kokkos/`.

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
