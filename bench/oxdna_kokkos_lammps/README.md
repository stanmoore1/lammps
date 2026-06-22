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

| Aspect | `bench/oxdna_kokkos` (CUDA-faithful) | `bench/oxdna_kokkos_lammps` (this, LAMMPS-faithful) |
|---|---|---|
| Body frames (a1,a2,a3) | recomputed from the quaternion inside every force kernel | **LRF precompute pass** (`compute_lrf`, mirrors `fix oxdna/lrf`): one thread/atom computes a1/a2/a3 and stores them in per-particle arrays `nx,ny,nz`; every force kernel *reads* these |
| Nonbonded operator | one *fused* edge kernel computing excv + hbond + xstk + coaxstk + dh per pair | **one separate kernel per interaction term**: `excv`, `hbond`, `xstk`, `coaxstk`, `dh` (LAMMPS `pair oxdna/*`) |
| Neighbor handling | flat edge list (one thread per pair) for everything | **excv & dh** iterate *per-atom* over the half neighbor matrix (`d_num_neigh`/`d_neigh_matrix`, each pair once — LAMMPS `neigh half` HALFTHREAD); **hbond, xstk, coaxstk** iterate over a **screened flat pair list** (`fix oxdna/npair`): pairs whose center-of-mass distance is within `rsq < 4.0` (r < 2.0), rebuilt only when the neighbor list rebuilds |
| Bonded operator | one *fused* per-particle gather kernel (FENE + bonded-excv + stacking) | **two separate kernels**: `fene` (FENE + 3 bonded-excv terms, LAMMPS `bond oxdna/fene`) and `stk` (stacking, LAMMPS `pair oxdna/stk`) |

**Per-step kernel sequence** (mirroring LAMMPS): LRF precompute → excv → hbond →
xstk → coaxstk → dh → stk → fene (8 force kernels + the bonded LRF guard). Each
term-kernel does its own atomic scatter into the shared force/torque arrays, so
the order is interchangeable; the energies are summed into `epot_` exactly as
before. (The screened pair list is rebuilt only on neighbor-list rebuild steps.)

The physics helpers — `add_excv_contrib`, `hbond_pair`, `crst_pair`,
`cxst_pair`, `dh_pair` (`forces/dna_forces.h`), and the split `bonded_fene_excv`
/ `bonded_stk` (`forces/bonded.h`, identical math to the original fused
`bonded_pair`) — are unchanged, so the printed oxDNA energy matches
`bench/oxdna_kokkos` to FP round-off (exact on the oxDNA1 8bp duplex; identical
through thousands of steps on the oxDNA2 N8 case, then drifting only at the level
of floating-point operation-reordering chaos).

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

- **`./build/fd_test`** — self-checking suite for both models: analytic forces &
  torques vs. central finite differences of the energy (every term), NVE energy
  conservation, and thermostat temperature (equipartition). Exits non-zero on
  failure. This is what CI runs.
- **`./build/xcheck <model> <T> <salt> <top> <conf> [ft_out]`** — prints the
  potential energy (total and per group) and optionally dumps per-particle
  force/torque, for direct comparison against the standalone oxDNA
  `potential_energy split = 1` and `particle_force_torque` (`lab_frame = 1`)
  observables.

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
several real LAMMPS per-step framework overheads. Setting `lammps_overhead = 1`
in the input adds them back (physics/energy output is unchanged — verified
identical on/off):

- **Per-step bond-prime-neigh precompute**: two extra per-bond kernels (stk, fene)
  that re-derive the 3'/5' bonded-neighbour table every step, mirroring
  `TagPairOxdnaStkPrecomputeBondPrimeNeighs` (the lean code stores `bonds.n3/n5`
  directly and skips this).
- **Per-kernel ScatterView**: each nonbonded kernel creates its own ScatterView
  instead of sharing one (mirrors LAMMPS creating `dup_f/dup_torque` per pair
  style; cheap on GPU where HALFTHREAD uses non-duplicated atomics).
- **Per-step host flag copy**: a device->host `deep_copy` each step, mirroring
  the FENE bond-overstretch flag check.

What it deliberately does NOT model is **ghost atoms + per-step communication**:
the standalone uses minimum-image PBC (`box.wrap`) and processes exactly N atoms,
whereas LAMMPS replicates the boundary shell as ghosts, forward-communicates
positions every step, and runs the LRF fix / neighbour list / pair styles over
`nlocal+nghost`. So:

    (LAMMPS time) - (standalone with lammps_overhead=1) ~= ghost/comm cost,

isolating the fundamental (domain-decomposition) floor from the optimizable
per-step overheads (bond precompute, per-style setup, flag copies).
