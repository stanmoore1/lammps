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

Molecular dynamics in the NVE or NVT (Andersen thermostat) ensemble, with the
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
| `src/thermostat.h` | Andersen ("John") thermostat (optional, NVT) |
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

Requires CMake ≥ 3.20 and a C++17 compiler. Kokkos is taken from the bundled
LAMMPS tree (`lib/kokkos`) by default, or any installed/standalone Kokkos via
`-DKOKKOS_SOURCE_DIR=<path>` (or `find_package`).

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
./build/oxdna_kokkos -top <file.top> -conf <file.conf> [options]
```

| Option | Default | Description |
|---|---|---|
| `-top <file>`   | — | Topology (`.top`) |
| `-conf <file>`  | — | Configuration (`.conf`/`.dat`) |
| `-steps <N>`    | 10000 | Number of MD steps |
| `-dt <dt>`      | 0.001 | Timestep (reduced units) |
| `-T <T>`        | 0.1 | Temperature (reduced units; 1 unit ≈ 3000 K) |
| `-model <1\|2>` | 1 | oxDNA1 or oxDNA2 |
| `-salt <c>`     | 0.5 | Salt concentration [mol/L] (oxDNA2 only) |
| `-cut <r>`      | 2.5 | Minimum nonbonded cutoff (grown automatically for Debye–Hückel) |
| `-skin <r>`     | 0.3 | Verlet skin |
| `-newt <N>`     | 0 | Andersen thermostat period in steps (`0` = NVE) |
| `-pt <p>` / `-pr <p>` | 0.1 | Translational / rotational refresh probability |
| `-seed <N>`     | 12345 | Thermostat RNG seed |
| `-freq <N>`     | 1000 | Energy print frequency |

Input files use the standard oxDNA formats: the `.top` lists
`<N> <N_strands>` then one `<strand_id> <base> <n3> <n5>` line per nucleotide;
the `.conf` has `t = …`, `b = Lx Ly Lz`, `E = …`, then one line per nucleotide
with position, `a1`, `a3`, velocity, and angular momentum. Coordinates should be
folded into the box.

Example (oxDNA2, NVE, 8bp duplex bundled under `tests/`):

```bash
./build/oxdna_kokkos -top tests/8bp_duplex/test.top -conf tests/8bp_duplex/test.conf \
                     -model 2 -salt 0.5 -T 0.1 -dt 1e-4 -steps 10000
```

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
