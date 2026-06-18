# oxDNA-Kokkos

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
| `src/forces/backbone.h` | FENE + bonded excluded volume |
| `src/forces/stacking.h` | Bonded stacking |
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
| `seed`               | 12345 | Thermostat RNG seed |
| `thermostat`         | (none) | `brownian`/`john` → NVT; otherwise NVE |
| `newtonian_steps`    | 0 | Brownian thermostat period in steps |
| `diff_coeff`         | 2.5 | Translational diffusion coefficient |
| `pt`                 | 0 | Refresh probability; overrides `diff_coeff` if `> 0` |
| `timing`             | 0 | `1` → per-kernel timing breakdown (adds fences; `0` for production) |

Paths are resolved relative to the working directory (run from the case
directory, as with the reference oxDNA). The `.top` lists `<N> <N_strands>`
then one `<strand_id> <base> <n3> <n5>` line per nucleotide; the `.conf` has
`t = …`, `b = Lx Ly Lz`, `E = …`, then one line per nucleotide with position,
`a1`, `a3`, and optionally velocity and angular momentum.

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
