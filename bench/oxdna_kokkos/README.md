# oxDNA-Kokkos

A portable, GPU-ready standalone implementation of the [oxDNA](https://github.com/lorenzo-rovigatti/oxdna)
coarse-grained DNA model, written with [Kokkos](https://github.com/kokkos/kokkos).
It is intended as a compact benchmark and reference port: the force field
faithfully reproduces the standalone oxDNA **oxDNA1**, **oxDNA2** and
(sequence-dependent) **oxDNA3** models (validated term-by-term, see
[Validation](#validation)), while the data layout and kernels are structured
for performance on CPUs (Serial/OpenMP) and GPUs (CUDA) through a single Kokkos
code base, mirroring what a default standalone oxDNA CUDA MD run does per step
(kernels, data layout, precision modes, Verlet list, synchronisations,
integrator and thermostat; see [Fidelity to oxDNA CUDA](#fidelity-to-oxdna-cuda)),
so that timing it against LAMMPS KOKKOS on a GPU isolates LAMMPS' own costs.

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
| `src/simulation.h` | MD driver: SimManager main loop + `MD_CUDABackend::sim_step` (output, fix_diffusion, first step, sorting, lists, forces, second step, thermostat) |
| `src/integrator.h` | `first_step` / `second_step` (CUDA_MD.cuh) and `first_step_mixed` / `second_step_mixed` (CUDA_mixed.cuh): body-frame angular momentum, right-multiplied quaternion update |
| `src/thermostat.h` | CUDA Brownian ("John") thermostat: per-particle XORWOW states, Box-Muller refresh; host `refresh_vel` (drand48) |
| `src/neighbor_list.h` | `CUDASimpleVerletList`: cells, full neighbour matrix, optional edge list (`use_edge`) |
| `src/sort.h` | Hilbert-curve particle sorting (`CUDA_sort_every`, CUDA_sort.cu) |
| `src/particles.h`, `src/types.h` | AoS particle arrays (float4-like), mixed-precision double copies and conversion kernels, `set_external_forces`, box, precision / launch configuration |
| `src/forces/params.h` | Force-field parameters (`make_oxdna1_params`, `make_oxdna2_params`) |
| `src/forces/params_dna3.h` | oxDNA3 parameters: sequence-dependence file parser, tetramer tables (host build, one device View), `make_oxdna3_params` |
| `src/forces/dna_forces.h` | oxDNA1/2 kernels: per-particle `dna_forces` (default) and `dna_forces_edge_nonbonded` (`use_edge = 1`) |
| `src/forces/bonded.h` | oxDNA1/2 bonded part (`_bonded_part`) and `dna_forces_edge_bonded` (`use_edge = 1`) |
| `src/forces/dna3_forces.h` | oxDNA3 kernels: per-particle `DNA3_forces` (default), `DNA3_forces_edge_nonbonded` + `DNA3_forces_edge_bonded` (`use_edge = 1`), ports of `CUDA_DNA3.cuh` |
| `params/oxDNA3_sequence_dependent_parameters.txt` | Copy of the upstream oxDNA3 parameter file (default `seq_dep_file`) |
| `src/forces/mf_oxdna.h` | Modulation functions F1-F6 and derivatives |
| `src/forces/orient.h` | Quaternion -> body-axis vectors |
| `src/io/topology_reader.h`, `src/io/config_reader.h`, `src/io/input_reader.h` | oxDNA `.top` (old 3'->5' and new `5->3` formats) / `.conf` readers, input file |

## Building

Requires CMake >= 3.20 and a C++20 compiler (Kokkos 5.0 requires C++20). Kokkos
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

# NVIDIA GPU (e.g. Ampere SM80), mixed precision = the upstream CUDA default
cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
      -DCMAKE_CXX_COMPILER=$(pwd)/../../lib/kokkos/bin/nvcc_wrapper \
      -DOXDNA_MIXED_PRECISION=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake options:

| Option | Default | Meaning |
|---|---|---|
| `OXDNA_MIXED_PRECISION` | OFF | oxDNA `backend_precision = mixed` (`CUDAMixedBackend`), see [precision](#1-precision-backend_precision-mirrored) |
| `OXDNA_SINGLE_PRECISION` | OFF | oxDNA `backend_precision = float` (`MD_CUDABackend`, float) |
| (neither) | | oxDNA `backend_precision = double` (`MD_CUDABackend`, `CUDA_DOUBLE=ON` build) |
| `OXDNA_THREADS_PER_BLOCK` | 0 (= 64 on CUDA, 128 on HIP, i.e. 2 x warp size) | upstream input key `threads_per_block`, compile-time here |
| `OXDNA_BUILD_TESTS` | OFF | also build the validation tools (`fd_test`, `xcheck`) |
| `OXDNA_DEFAULT_SEQ_DEP_FILE` | `params/...` | default oxDNA3 parameter file |

Preprocessor overrides (tuning only, not upstream behaviour): `OXDNA_NB_MAXT` /
`OXDNA_NB_MINB` (force kernel, default `OXDNA_THREADS_PER_BLOCK` / 0) and
`OXDNA_BOND_MAXT` / `OXDNA_BOND_MINB` (bonded edge kernel) set the Kokkos
`LaunchBounds`; a non-zero MINB caps registers, which upstream does not do.

### Matching the reference GPU performance

Build with `-DOXDNA_MIXED_PRECISION=ON`: a CUDA build of oxDNA defaults to
`backend_precision = mixed` (`BackendFactory.cpp`: `backend_prec = "mixed"`
when the key is absent), whose force kernels, lists and thermostat run in
FP32 with only the integrator in FP64. The default (double) build of this code
mirrors the `CUDA_DOUBLE` build of oxDNA and runs every kernel in FP64. Leave
`use_edge` unset (the upstream default, per-particle force kernels), leave
`CUDA_sort_every` unset (0), and keep `timer_sync = 1` (upstream synchronises
the device after every timed section).

## Fidelity to oxDNA CUDA

Reference: upstream oxDNA master c2c74cc0, a default CUDA MD run
(`backend = CUDA`, `sim_type = MD`, Brownian thermostat, no external forces,
no barostat, default `CUDA_list = verlet`, `use_edge = false`,
`CUDA_sort_every = 0`, `threads_per_block` = 2 x warp size,
`backend_precision = mixed` in a default build). File:line references are to
`src/CUDA/...` unless noted.

Summary:

| # | Item | Status |
|---|---|---|
| 1 | Precision (double / float / mixed) | mirrored (compile-time), incl. every conversion kernel |
| 2 | Particle sorting (`CUDA_sort_every`, default 0) | mirrored behind the same key (off by default) |
| 3 | Launch configuration (`threads_per_block`) | mirrored as far as Kokkos allows (compile-time block size cap) |
| 4 | Verlet list (`CUDASimpleVerletList`) | mirrored (kernels, layout, cutoff, host syncs) |
| 5 | Per-step host work / syncs | mirrored except the CPU-side observable evaluation |
| 6 | Integrator + thermostat | mirrored (body frame, right multiplication, float dt, per-particle curand-style RNG) |
| 7a | Default force kernels: `use_edge = false` | mirrored (per-particle kernels are now the default) |
| 7b | `set_external_forces` every step | mirrored |
| 7c | Box / positions / configuration reading / `fix_diffusion` | mirrored |
| 7d | Energy computation | per-particle energies in `F.w` as upstream; the printed value is summed on the host |
| 7e | Compiler flags (`-use_fast_math`, `-ffast-math`) | mirrored (`OXDNA_FAST_MATH`, default ON) |

### 1. Precision (`backend_precision`): mirrored

| oxDNA | This code | Force kernels, lists, thermostat (`c_number`) | Integrator arrays |
|---|---|---|---|
| `double` (`CUDA_DOUBLE=ON`, `MD_CUDABackend`) | default build | double | double (same arrays) |
| `float` (`MD_CUDABackend`) | `-DOXDNA_SINGLE_PRECISION=ON` | float | float (same arrays) |
| `mixed` (`CUDAMixedBackend`, upstream default) | `-DOXDNA_MIXED_PRECISION=ON` | float | double copies `possd`, `velsd`, `Lsd`, `orientationsd` |

Mixed precision follows `Backends/MD_CUDAMixedBackend.cu` and
`Backends/CUDA_mixed.cuh`:

- Float arrays (`c_number4` = float4): positions, orientations, forces,
  torques, velocities, angular momenta, list positions. Double arrays
  (`LR_double4`): positions, velocities, angular momenta, orientations,
  used only by the integrator (`MD_CUDAMixedBackend.h:24-27`).
- `first_step_mixed` (`CUDA_mixed.cuh:23-69`): `v += F * dt * 0.5f` with the
  product evaluated in float and added to the double velocity, `r += v * dt`
  in double, the float copy `poss = (float) r` written for the force kernels,
  `L += T * dt * 0.5f` (float product), the orientation update in double
  (`_get_updated_orientation`, `CUDA_mixed.cuh:8-21`), the float copy of the
  quaternion, and the Verlet check on the float position.
  `second_step_mixed` (`CUDA_mixed.cuh:71-93`): float products added to the
  double v and L; no `.w` kinetic energies.
- Conversion kernels (`float4_to_LR_double4` / `LR_double4_to_float4`,
  `CUDA_mixed.cuh:95-123`, one thread per particle) run where upstream runs
  them: after the initial upload (`init`, 4 x float->double,
  `MD_CUDAMixedBackend.cu:37-40`); before every energy output
  (`apply_simulation_data_changes`, 4 x double->float, `:115-125`); around every
  thermostat application (velocities and angular momenta double->float, the
  thermostat on the float arrays, float->double, `:140-148`, i.e. every
  thermostat step rounds all velocities to float, as upstream); around every
  Hilbert sort (4 x each way, `:93-103`); after `fix_diffusion`
  (`apply_changes_to_simulation_data`, 4 x float->double, `:127-138`).
- In every precision mode the time step and the squared Verlet skin are
  single-precision constants (`__constant__ float MD_dt`, `MD_sqr_verlet_skin`,
  `CUDA_MD.cuh:2,5`, `CUDA_mixed.cuh:1-2`), the list radius too
  (`verlet_sqr_rverlet`, `Lists/CUDA_simple_verlet.cuh:8`); `pt`, `diff_coeff`
  and `dt` of the thermostat are read as float (`getInputFloat`,
  `src/Backends/Thermostats/BrownianThermostat.cpp`).
- Not mirrored: in the double build upstream's `_get_updated_orientation`
  calls `sqrtf` / `fmaxf` (`CUDA_MD.cuh:12,19`, `_module` in
  `cuda_utils/CUDA_lr_common.cuh:178`), i.e. rounds `|L|` and the rotation
  quaternion to float; this code uses the working precision (same cost).

Validation (Serial, this machine): the float kernels of the single and mixed
builds are bit-identical (`xcheck` dumps); both agree with the double build to
<= 1.3e-4 (force) / 1.1e-4 (torque) relative to the largest component on
8bp/N8/N64 for oxDNA1/2/3, and the energy per nucleotide (per-particle `F.w`
summed in double) to <= 7e-7 relative. NVE total-energy drift in `fd_test`
(8bp duplex from rest, 3000 steps), double / mixed / single: oxDNA1
1.5e-6 / 1.7e-6 / 2.9e-4, oxDNA2 6.6e-8 / 9.4e-7 / 4.1e-5, oxDNA3
4.5e-5 / 4.6e-5 / 5.1e-4. N8 NVE (6000 steps, dt = 0.0029296875): max
|E_tot(t) - E_tot(0)| per nucleotide double / mixed / single = 6.0e-4 /
6.5e-4 / 1.3e-3 (oxDNA2) and 7.7e-4 / 6.7e-4 / 3.3e-3 (oxDNA3); after 1000
steps U(mixed) - U(double) = 0 / 3e-6 per nucleotide, U(single) - U(double) =
3e-4 / 5e-5 (oxDNA2 / oxDNA3). `fd_test` passes in all three builds (the FD
checks are only meaningful in double).

### 2. Particle sorting (`CUDA_sort_every`): mirrored, off by default

Upstream default `CUDA_sort_every = 0` (`Backends/CUDABaseBackend.cu:25,119`),
so a default run never sorts. With `CUDA_sort_every = n > 0` this code, like
`MD_CUDABackend::sim_step` (`Backends/MD_CUDABackend.cu:575-577`), re-orders
the particles along a 3D Hilbert curve whenever the lists are old and
`N_updates % n == 0`, right before the list update: `reset_sorted_hindex`,
`hilbert_curve` (depth 8, cubic box side Lx, `CUDA_sort.cu:106-144`),
`sort_by_key` (Kokkos calls thrust on CUDA, as upstream,
`CUDABaseBackend.cu:263-281`), `get_inverted_sorted_hindex`,
`permute_particles` (positions, velocities, angular momenta, orientations,
bonds remapped through the inverse permutation, strand ids,
`CUDA_sort.cu:146-166`) and device-to-device copies back
(`MD_CUDABackend.cu:535-550`); forces, list positions and thermostat states
are not permuted (as upstream). Beyond upstream the base / particle-type
arrays are permuted too (upstream keeps btype in `poss.w`, but does not permute
`CUDADNA3Interaction::_d_particle_types`, so DNA3 with sorting is broken
upstream). Check (double): N8 NVE with `CUDA_sort_every = 1` reproduces the
unsorted energies to 6 decimals for 5000 (oxDNA3) / 6000 (oxDNA2, the whole
run) steps; in mixed precision every sort rounds the double state to float
(as upstream), so the trajectories separate after a few hundred steps.

### 3. Launch configuration: mirrored as far as Kokkos allows

Upstream launches every particle kernel (first/second step, forces,
`set_external_forces`, conversions, sorting) with `threads_per_block`
threads (default 2 x warpSize = 64 on NVIDIA,
`CUDABaseBackend.cu:244-261`) and ceil(N / tpb) blocks, the edge kernels with
ceil(N_edges / tpb) blocks (`Interactions/CUDADNAInteraction.cu:177`), the
list kernels with 64 threads (`Lists/CUDASimpleVerletList.cu:192`) and the
thermostat with 64 (`Thermostats/CUDABaseThermostat.cu:110-113`); no kernel
has `__launch_bounds__`. Kokkos chooses the block size of a `RangePolicy`
itself; every kernel here uses `Kokkos::LaunchBounds<OXDNA_THREADS_PER_BLOCK, 0>`
(default 64), which makes Kokkos' occupancy search start at 64 threads (it only
picks 32 if that gave strictly more resident threads, which it does not) and
emits `__launch_bounds__(64)`, which does not restrict registers below
upstream's unbounded compile. The grid is ceil(n / 64), as upstream.
Not mirrored: `threads_per_block` is a compile-time value
(`-DOXDNA_THREADS_PER_BLOCK`; the input key only triggers a warning if it
differs); `cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)`
(`CUDABaseBackend.cu:188`) has no Kokkos equivalent (the kernels use no shared
memory, so on Volta+ the driver's per-kernel carveout already favours L1); the
block size of Kokkos reductions (only on energy-output steps with
`use_edge = 1`, and with `CUDA_print_energy = 1`) is Kokkos' choice (upstream
uses thrust there).

### 4. Verlet list (`CUDA_list = verlet`, `CUDASimpleVerletList`): mirrored

- Cutoff: rverlet = rcut + 2 x `verlet_skin` with rcut the interaction cutoff
  of the upstream interaction class (`DNAInteraction::init`,
  `DNA2Interaction::init`, `DNA3Interaction::init`: max of backbone excluded
  volume, H-bonding and Debye-Hueckel ranges, e.g. 1.5838 for oxDNA1 and
  2.5715 for oxDNA2/3 at T = 20C, salt 1 M); the previous version used
  max(2.5, force range), i.e. a much larger list for oxDNA1. Rebuild when a
  particle moved more than the skin (plain difference, float constant).
- Cells (`Lists/CUDASimpleVerletList.cu:96-163`): floor(L / rverlet) per
  side, at least 3, capped by `cells_auto_optimisation` (default true) at
  ceil((2N/V)^(1/3) L); `max_N_per_cell` = round(`max_density_multiplier` (3)
  x the largest initial occupancy) (count kernel + device max + one host read,
  once), clamped to [5, N+1]; `max_neigh` = min(4/3 pi max_N_per_cell, N-1);
  cell index `(z Ny + y) Nx + x` with upstream's fractional-coordinate formula
  (`cuda_utils/CUDABox.h`).
- Update (`CUDASimpleVerletList.cu:231-289`, `Lists/CUDA_simple_verlet.cuh`):
  memset of the counters, `simple_fill_cells` (atomic slot, host-pinned
  overflow flag), device synchronisation + host check of the flag, then one `simple_update_neigh_list` kernel that builds the FULL
  neighbour matrix (both i->j and j->i, bonded neighbours excluded, the cell
  itself followed by the 26 neighbours in upstream's order), stored
  column-major `matrix[j * N + i]` (`Kokkos::LayoutLeft`), and writes the
  neighbour count and `list_poss` (no separate copy). With `use_edge = 1`:
  `edge_update_neigh_list` (also counts the neighbours with a smaller index), a
  host write of `offsets[N] = 0`, a device exclusive scan, a host read of
  `N_edges`, and `compress_matrix_neighs` (edges `from = i > to`), into an
  edge array allocated once with N x max_neigh entries.
- The first-step kernel writes the "lists old" flag directly into host-pinned
  memory (`Kokkos::SharedHostPinnedSpace`, upstream `cudaMallocHost`,
  `CUDABaseBackend.cu:210`), read by the host after the first-step sync and
  reset by the host after the update (`MD_CUDABackend.cu:581-592`).
- Beyond upstream: on a cell overflow upstream aborts; this code doubles
  `max_N_per_cell` and re-bins (with a warning), and a neighbour-matrix
  overflow (upstream writes out of bounds) is detected and grows the matrix.
  Neither costs anything unless it happens. The previous version of this code
  silently dropped particles beyond 20 per cell: `xcheck` on N512 (list radius
  2.5 + 2) gave -11091.655618 instead of -11093.841834 (= 8192 x the N8 value
  -1.354229 per nucleotide), which this version reproduces; the production
  inputs did not overflow. The counters are read through a `RandomAccess`
  view (upstream: a texture object).

### 5. Per-step host work and synchronisations: mirrored except CPU observables

- Upstream constructs every timer with `enable_sync()`
  (`MD_CUDABackend.cu:48`), and `Timer::pause()` calls
  `cudaDeviceSynchronize()` (`src/Utilities/Timings.cpp:53-56`), so every
  step synchronises after the first step, the sorting, the lists, the forces,
  the thermostat and the observable timers. This code fences at the same
  section ends (`timer_sync = 1`, default; `timer_sync = 0` removes them, not
  upstream). The fence after the first step is always kept, since the
  host-pinned flag is read right after it (upstream relies on the timer sync
  for that too).
- Energy output (`SimBackend::print_observables`, `src/Backends/SimBackend.cpp:728-760`):
  on steps with `curr_step % print_energy_every == 0` (absolute step, so a
  configuration with t = 20618 first prints at 21000, as upstream) the full
  device-to-host copy of `apply_simulation_data_changes` is done
  (`MD_CUDABackend.cu:99-106,315-394`: positions, bonds, orientations, strand
  ids, velocities, angular momenta, forces, torques; preceded by the 4
  conversion kernels in mixed precision), then K and U are summed on the host
  in double. Not mirrored: upstream then rebuilds the CPU neighbour lists
  (`_lists->global_update(true)`) and evaluates the potential energy with the
  CPU interaction; this code has no CPU force field and sums the per-particle
  energies the force kernel already stored in `F.w` (default path; identical
  to round-off in double, FP32 kernel energies in float/mixed where upstream
  evaluates the float coordinates in double). With `use_edge = 1` the
  energy is reduced inside the force kernels of the step before an output
  step (upstream's edge kernels do not accumulate `.w`).
- `CUDA_print_energy = 1` (default 0, `MD_CUDABackend.cu:603-606,636`):
  device sum of `F.w` in double and a host read every step, printed as an
  extra column.
- `fix_diffusion` every `fix_diffusion_every` (100000) steps
  (`src/Managers/SimManager.cpp:167`, `SimBackend.cpp:788-840`): full
  device-to-host copy, strands shifted back into the box, orientations
  re-normalised, host-to-device copy (+ float->double in mixed). The CPU energy
  checks around it are not mirrored.
- `update_observables_data` (every step) does nothing without observables
  that need updating, as here.

### 6. Integrator and thermostat: mirrored

- `first_step` / `second_step` (`Backends/CUDA_MD.cuh:26-59,556-578`) line by
  line: same association of the products (`F * (dt * 0.5)` in the first,
  `F * dt * 0.5` in the second step), no position folding, `second_step`
  stores v^2/2 and L^2/2 in `.w`.
- Angular momenta and torques are in the BODY frame: the force kernels rotate
  the accumulated lab-frame torque into the body frame at their end
  (`_vectors_transpose_c_number4_product`, `Interactions/CUDA_DNA.cuh:822,896`,
  `CUDA_DNA3.cuh:992,1069`), and the orientation is updated by
  right-multiplication q <- q * R(L_body dt) with upstream's formula
  (sincos of dt |L|, `qw = 0.5 sqrt(max(0, 2 + 2 cos))`, no re-normalisation,
  `CUDA_MD.cuh:11-24`). The previous version integrated lab-frame L with a
  left-multiplied rotation and re-normalised the quaternion (the same dynamics
  mathematically, but configuration files store body-frame L upstream).
- `|L| = 0` skips the rotation (upstream: division by zero; upstream refuses
  such initial configurations unless `refresh_vel = true`, and so does this
  code).
- Thermostat (`Thermostats/CUDABrownianThermostat.cu:15-51,74-84`,
  `CUDABaseThermostat.cu:94-123`): one persistent 48-byte XORWOW state per
  particle (curandState layout), set up once by a setup kernel, loaded and
  stored by the thermostat kernel on steps with
  `curr_step % newtonian_steps == 0`; per particle a uniform draw against pt,
  then v from two Box-Muller pairs ((x, y), (z, trash)), then the same for L
  against pr; refreshed vectors store their kinetic energy in `.w`. The
  generator, `curand_uniform` and `gaussian` (`cuda_utils/CUDA_lr_common.cuh:73-87`,
  fast `__sinf/__cosf` on CUDA in float) are bit-level ports. Not mirrored:
  `curand_init(seed, IND, 0)` places particle IND on subsequence IND (skip-ahead
  with curand's precomputed jump matrices); here each state is initialised
  with curand's seed scrambling of a per-particle hash of (seed, IND): same
  generator and memory traffic, independent streams, not bit-identical. The
  seed is `lrand48()` after `srand48(seed)`, as upstream.
- `refresh_vel` (`src/Backends/MDBackend.cpp:140-165`) runs on the host with
  `drand48` and the Marsaglia polar method of `Utils::gaussian()`, in the
  upstream order (vx, vy, vz, Lx, Ly, Lz per particle): with the same `seed`
  the initial kinetic energy equals the standalone oxDNA's (N8: 0.283532 per
  nucleotide in both).

### 7. Other per-step work

- **Default force kernels (`use_edge = false`)**: upstream's default
  (`MD_CUDABackend.cu:625`, `Interactions/CUDABaseInteraction.cu:87`) is the
  per-particle kernel `dna_forces` / `DNA3_forces`
  (`Interactions/CUDA_DNA.cuh:832-908`, `CUDA_DNA3.cuh:1002-1077`): one thread
  per particle reads its F and T, adds its n3 and n5 bonds and every
  non-bonded neighbour of its full matrix row (every pair is evaluated twice,
  once from each side, no atomics), rotates T into the body frame and writes
  F, T once; `F.w` holds the particle's energy. This is now the default here.
  The edge kernels (`use_edge = 1`, `edge_n_forces = 1`: atomic edge kernel +
  bonded gather kernel that rotates the total torque) are kept as an option
  (upstream refuses `use_edge` in double precision; here it only warns).
  `edge_n_forces > 1` is not supported.
- **`set_external_forces`** (`CUDA_MD.cuh:97-104`, `MD_CUDABackend.cu:519`):
  every step one kernel zeroes F and T (all four components) before the force
  kernels (previously two memsets).
- **Box and positions**: positions are not folded into the box during MD;
  non-bonded separations use the minimum image with `rint`
  (`cuda_utils/CUDABox.h:42-54`), bonded separations the plain difference
  (`r = qpos - ppos`), the rebuild check the plain displacement.
  Configurations are read like `SimBackend::read_next_configuration`
  (`src/Backends/SimBackend.cpp:600-700`): a1 and a3 are orthonormalised and
  every strand is shifted by whole box vectors so that its centre of mass lies
  in [0, L) (`fix_diffusion = true`, the default); `restart_step_counter`
  selects t = 0 or the configuration's t (thermostat and output steps use the
  absolute step), `reset_initial_com_momentum` is supported.
- **Compiler flags**: upstream compiles CUDA device code with
  `-use_fast_math` (flush-to-zero, approximate float division and square
  root, fast float intrinsics; `src/CMakeLists.txt:309-313`) and all host code
  with `-ffast-math` (`CMakeLists.txt:96`, every non-Debug build). CMake option
  `OXDNA_FAST_MATH` (default ON, not in Debug builds) adds `-ffast-math` and,
  with `Kokkos_ENABLE_CUDA`, `-use_fast_math` (nvcc_wrapper forwards it).
  Without it the float per-particle oxDNA3 kernel ran 1.5x slower on the CPU
  than the double one because of denormals (flush-to-zero removes that).
  Physics is unchanged (`xcheck` identical to all printed digits, `fd_test`
  identical to 4 digits).
- **Base type with the position**: the oxDNA1/2 kernels read the base type
  from `poss.w`, loaded with the position, like upstream's
  `get_particle_type(ppos)` (upstream bit-packs `btype << 22 | index` into the
  float; that bit pattern is a denormal, destroyed by flush-to-zero on the
  host, so the type is stored as a plain number and the index is dropped); the
  oxDNA2 edge kernel reads the per-particle strand-end flag array only with
  Debye-Hueckel (`init_DNA_strand_ends`, `CUDA_DNA.cuh:739,746`), the
  per-particle kernels derive it from the bonds they load anyway (as
  `dna_forces`).
- **Not mirrored** (off in a default run, or no Kokkos/bench counterpart):
  external forces, barostat, stress tensor (`CUDA_update_stress_tensor_every`),
  `CUDA_avoid_cpu_calculations` (no CPU observables here), trajectory /
  configuration output, `max_backbone_force`, other lists
  (`CUDA_list = no | bin_verlet`), other thermostats.

## Running

```bash
./build/oxdna_kokkos <input_file>
```

The program is driven by a **standalone-oxDNA-style input file** (`key = value`),
so the *same* input that drives the reference oxDNA drives this code --
unrecognized keys (`backend`, `trajectory_file`, `ensemble`, `data_output_*`
blocks, `${...}` expressions, ...) are ignored. Recognized keys:

| Key | Default | Description |
|---|---|---|
| `topology`           | -- | Topology `.top` (mandatory) |
| `conf_file`          | -- | Configuration `.conf`/`.dat` (mandatory); L is the body-frame angular momentum, as upstream |
| `energy_file`        | (none) | If set, write oxDNA-style `time U K total` (per nucleotide) |
| `interaction_type`   | DNA | `DNA`/`DNA1` -> oxDNA1, `DNA2` -> oxDNA2, `DNA3`/`DNA3_nomesh` -> oxDNA3 |
| `salt_concentration` | 0.5 | mol/L (oxDNA2, oxDNA3) |
| `seq_dep_file`       | `params/oxDNA3_sequence_dependent_parameters.txt` | oxDNA3 parameter file (path relative to the working directory, as upstream); default: the copy in `params/` (absolute path baked in at configure time, CMake cache variable `OXDNA_DEFAULT_SEQ_DEP_FILE`), then `params/...` or `./oxDNA3_sequence_dependent_parameters.txt` |
| `use_average_seq`    | 0 (oxDNA3) | oxDNA3 only: `1` keeps the average (non-SD) tables; see note below |
| `dh_half_charged_ends` | 1 | oxDNA3: halve the Debye-Hueckel charge of strand-end nucleotides |
| `dh_lambda`, `dh_strength`, `debye_huckel_rhigh` | 0.3616455, 0.0543, 3 lambda | oxDNA3 Debye-Hueckel parameters (upstream keys) |
| `dna3_consistent_gamma` | 0 | oxDNA3, bench-only: exact stacking-dihedral gradient (see [oxDNA3](#oxdna3)) |
| `T`                  | 0.1 | `20C`, `300K`, or a number in oxDNA units (1 unit ~ 3000 K) |
| `dt`                 | 0.001 | Timestep (used as a float constant by the kernels, as upstream) |
| `steps`              | 10000 | Number of MD steps (accepts `1e7`) |
| `verlet_skin`        | 0.3 | Verlet skin |
| `print_energy_every` | 1000 | Energy output on steps that are multiples of it |
| `seed`               | 12345 | `srand48` seed (refresh_vel, thermostat seed) |
| `refresh_vel`        | 0 | `1` -> draw fresh Maxwell-Boltzmann velocities at T on startup (required for confs without angular momenta, as upstream) |
| `restart_step_counter` | 0 | `1` -> start at step 0 instead of the configuration's t |
| `reset_initial_com_momentum` | 0 | remove the centre-of-mass velocity at startup |
| `fix_diffusion`, `fix_diffusion_every` | 1, 100000 | strand COMs back into the box (at read time and every n steps) |
| `thermostat`         | (none) | `brownian`/`john` -> NVT; otherwise NVE |
| `newtonian_steps`    | 0 | Brownian thermostat period in steps |
| `diff_coeff`         | 2.5 | Translational diffusion coefficient |
| `pt`                 | 0 | Refresh probability; overrides `diff_coeff` if `> 0` |
| `use_edge`           | 0 | upstream key: `1` -> edge-based force kernels (atomics) instead of the per-particle kernels |
| `CUDA_sort_every`    | 0 | upstream key: Hilbert-sort the particles every n list updates |
| `CUDA_print_energy`  | 0 | upstream key: device energy sum every step (extra output column) |
| `max_density_multiplier`, `cells_auto_optimisation` | 3, 1 | upstream Verlet-list cell keys |
| `backend_precision`, `threads_per_block`, `CUDA_list`, `edge_n_forces` | | checked against the compiled configuration (warning on mismatch) |
| `timer_sync`         | 1 | bench: `1` -> device sync after every timed section (upstream behaviour), `0` -> none |
| `timing`             | 0 | bench: `1` -> print the per-section timing breakdown |

Paths are resolved relative to the working directory (run from the case
directory, as with the reference oxDNA). oxDNA value expressions are supported:
`$(key)` substitutes another key's value and `${ ... }` evaluates `+ - * / ()`
arithmetic, e.g. `print_energy_every = ${$(steps) / 100}`. The `.top` lists
`<N> <N_strands>` then one `<strand_id> <base> <n3> <n5>` line per nucleotide
(or, in the new upstream format, `<N> <N_strands> 5->3` then one 5'->3'
sequence line per strand, optionally with `circular=true`);
the `.conf` has `t = ...`, `b = Lx Ly Lz`, `E = ...`, then one line per nucleotide
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
reference oxDNA (the reference divides the total energy by N), on steps that
are multiples of `print_energy_every`. stdout has the columns
`step  time  U  K  total` (with `time = step * dt`), and if `energy_file` is set
it is written in oxDNA's `time U K total` format so it can be compared directly
with the reference `energy_file`:

```
#       step           time              U              K          total
           0       0.000000      -1.354229       0.283532      -1.070697
```

## Performance output

At the end of a run the code prints a LAMMPS-style loop-time / performance
summary and the number of Verlet-list updates:

```
Loop time of 2.91377 on 1 procs (Serial x 1) for 10000 steps with 128 atoms

Performance: 889568.862 tau/day, 3431.979 timesteps/s, 0.439 Matom-step/s
Verlet-list updates: 29
```

Set `timing = 1` to also print the time of each section, named after the
upstream timers (the sections are synchronised anyway with `timer_sync = 1`):

```
Section timing breakdown (oxDNA timers):
Section                |   time (s) |  %loop |    us/step
------------------------------------------------------------
First Step             |     0.0335 |   1.27 |      5.585
Hilbert sorting        |     0.0007 |   0.03 |      0.112
Lists                  |     0.0019 |   0.07 |      0.315
Forces                 |     2.5965 |  98.54 |    432.751
Thermostat             |     0.0008 |   0.03 |      0.141
Output                 |     0.0014 |   0.05 |      0.230
Other                  |     0.0003 |   0.01 |      0.044
------------------------------------------------------------
Total (loop)           |     2.6351 | 100.00 |    439.178
```

| This code | LAMMPS section(s) |
|---|---|
| `Lists` | `Neigh` |
| `Forces` | `Pair` + `Bond` (all oxDNA terms; `set_external_forces` and the second step are in this upstream timer too) |
| `First Step`, `Thermostat` | `Modify` (`nve/dotc/langevin` or integrator + thermostat fix) |
| `Output` | `Output` |

`timesteps/s` is the most directly comparable metric to LAMMPS' `Performance:`
line.

Serial CPU timings (this machine, 1 thread, shared with other jobs, best of 3,
ms per step; `tests/N512/input` with 1000 steps and `tests/N64/input_dna3`
with 2000 steps, both Brownian thermostat, `verlet_skin = 0.5`):

| Serial CPU | N512 oxDNA2 (8192 nt) | N64 oxDNA3 (1024 nt) |
|---|---|---|
| previous version (double, edge kernels, no syncs) | 15.1 | 2.46 |
| this version, double, default (per-particle kernels) | 20.4 | 3.46 |
| this version, double, `use_edge = 1` | 14.1 | 2.21 |
| this version, mixed, default | 18.5 | 3.24 |
| this version, mixed, `use_edge = 1` | 13.3 | 2.01 |
| this version, float, default | 18.7 | 3.14 |

On the CPU the default per-particle kernels are slower than the edge kernels
because every pair is evaluated twice (upstream's choice for its default,
which trades the extra arithmetic for no atomics on the GPU); the numbers only
document the CPU cost, the fidelity target is the GPU.

## Validation

Build with `-DOXDNA_BUILD_TESTS=ON` and run from this directory.

- **`./build/fd_test`** -- self-checking suite for all three models: analytic
  forces & torques vs. central finite differences of the energy through both
  force paths (per-particle and edge kernels; every term; for oxDNA3 each of
  the 8 upstream terms separately, on the relaxed duplex, on a nicked +
  perturbed duplex in which every term is active, and on a configuration with
  active stacking cos(phi1/phi2) modulation), NVE energy conservation with the
  production MD pipeline (both paths), and thermostat temperature
  (equipartition). Exits non-zero on failure. This is what CI runs (double and
  mixed precision).
- **`./build/xcheck <model> <T> <salt> <top> <conf> [ft_out] [seq_dep_file]
  [--only=<term>] [--consistent-gamma] [--average-seq] [--edge]`** -- prints
  the potential energy (total and per group; for oxDNA3 also every term, in
  the column order of `potential_energy split = true`) and optionally dumps
  per-particle force/torque (lab frame; default per-particle kernels, `--edge`
  for the edge kernels), for direct comparison against the standalone oxDNA
  `potential_energy split = true` and `force_and_torque` (`particle = -1`,
  `lab_frame = true`) observables. `--only=<term>`
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

The CUDA-fidelity changes (per-particle kernels, body-frame torques, new
Verlet list, configuration reading) leave the physics unchanged: `xcheck`
energies (all groups and oxDNA3 terms) of the 8bp, N8 and N64 cases for
oxDNA1/2/3 are identical to the previous version to all printed digits, and the
per-particle forces/torques (both kernel paths) agree to <= 2e-10 relative
(the 10 printed digits); N512 differs because the previous version dropped
particles from overfull cells (see [Verlet list](#4-verlet-list-cuda_list--verlet-cudasimpleverletlist-mirrored)). NVE MD of N8 (dt = 0.0029296875, exactly
representable in float, L converted to the body frame for the new code)
reproduces the previous version's energies to 6 decimals for 6000 steps
(oxDNA2) and 2800 steps (oxDNA3, whose cross-stacking discontinuities amplify
round-off faster). Thermostatted runs differ from the previous version from
the first step on, because `refresh_vel` and the thermostat now use the
upstream RNG layout (statistics below).

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

`fd_test` (double): every oxDNA3 term separately and combined matches the
finite-difference gradient to <= 4e-8 (relative to the largest component),
equipartition within 5% (8bp, short run). With the stacking cos(phi)
modulation active, the upstream-faithful forces deviate from -grad E by 9.7e-5
(force) / 5.3e-4 (torque), the `dna3_consistent_gamma = 1` variant by 3e-9 /
6e-10 (see below). NVE drift over 3000 steps: 4.5e-5 (the previous version
reported 7.8e-4 from a single cross-stacking energy jump that this trajectory
does not hit).

Thermostatted MD statistics (N8, T = 20C, salt 1 M, dt = 0.003,
`newtonian_steps = 103`, `diff_coeff = 2.5`, `refresh_vel = 1`, 50000 steps,
averages over steps >= 10000, mean +- standard error over seeds; standalone
oxDNA: CPU build, `DNA2` with the default average sequence, `DNA3` with
`use_average_seq = false`):

| | oxDNA2 \<U\>/nt | oxDNA2 \<K\>/nt | oxDNA3 \<U\>/nt | oxDNA3 \<K\>/nt |
|---|---|---|---|---|
| standalone oxDNA | -1.3812 +- 0.0029 (9 seeds) | 0.2993 +- 0.0023 | -1.3012 +- 0.0082 (3) | 0.3146 +- 0.0053 |
| this code, double | -1.3872 +- 0.0038 (9) | 0.2945 +- 0.0023 | -1.3095 +- 0.0143 (3) | 0.3103 +- 0.0020 |
| this code, mixed | -1.3729 +- 0.0073 (3) | 0.2999 +- 0.0021 | -1.3108 +- 0.0101 (3) | 0.3147 +- 0.0064 |
| previous version (double) | -1.3753 (seeds 1, 2; seed 3 blew up at step ~46000) | 0.3037 | -1.2949 +- 0.0071 (3) | 0.3234 +- 0.0027 |

3T = 0.2932; the oxDNA3 excess is heating by the integration error at
dt = 0.003 with the weak default coupling (pt ~ 0.012), present upstream too.
The initial kinetic energy (`refresh_vel`) equals the standalone oxDNA's
for every seed tested (e.g. 0.284732 / 0.300627 / 0.293522 for seeds 1 / 2 / 3).


Continuous integration (`.github/workflows/oxdna-kokkos.yml`) builds with the
Serial backend and runs `fd_test` (double and mixed precision) on every change
under `bench/oxdna_kokkos/`.

> Note: from a cold start (zero velocities) oxDNA2 needs a smaller timestep
> (`-dt 1e-4`) than oxDNA1 because the Debye-Hueckel + grooved backbone make the
> potential stiffer near close approaches; a thermostat or smaller `dt` keeps it
> stable. This does not affect force-evaluation throughput.

## oxDNA3

### Kernel structure (mirrors `CUDA_DNA3.cuh`)

- `DNA3PerParticleFunctor` = `DNA3_forces` (default, `use_edge = 0`): one
  thread per particle; `_DNA3_bonded_part<true>` for its n3 bond, `<false>` for
  its n5 bond, then `_DNA3_particle_particle_DNA_interaction` for every
  non-bonded neighbour of its full Verlet-matrix row (each pair from both
  sides), torque rotated into the body frame, F and T written once, no
  atomics; `F.w` = the particle's energy.
- `DNA3NonbondedFunctor` = `DNA3_forces_edge_nonbonded` (`use_edge = 1`): one
  thread per Verlet edge, `p` (CUDA `from`) the larger and `q` (`to`) the
  smaller index, `r = q - p` (minimum image). Per edge it reads both
  positions and quaternions, both `LR_bonds`, the `uint8` particle-type array
  for `p`, `q` and their n3/n5 neighbours (tetramer flanks, `NO_TYPE` = 5 at
  strand ends) and the tables; the interaction accumulates the force `F` and
  torque `T` on `p`, then atomics add `T -> p`, `F -> p`, `-F -> q`,
  `-T + r x F -> q` (skipped when zero, as CUDA).
- `DNA3BondedFunctor` = `DNA3_forces_edge_bonded` (`use_edge = 1`): one
  thread per particle, FENE + bonded excluded volume + stacking for its two
  bonds on top of the edge result, total torque rotated into the body frame,
  no atomics.
- Differences from CUDA: the energy is kept with the correct Debye-Hueckel
  sign in `F.w` (CUDA packs it with the wrong sign); the functors take a
  compile-time term mask (`dna3::ALL` in production, single terms for the
  validation tools) and a `body_frame` switch (lab-frame torques for the
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
F5 (4)) live in ONE device View (`c_number`: float in single and mixed precision), the
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

- **CUDA backend (found while mirroring it).** With `CUDA_sort_every > 0`
  the particles are permuted but `CUDADNA3Interaction::_d_particle_types`
  (and, for the oxDNA2 edge kernel, `_d_is_strand_end`) are not, so oxDNA3
  (oxDNA2 with `use_edge`) sorted runs use the wrong types (strand ends); in
  the `CUDA_DOUBLE` build the orientation update rounds |L| and the rotation
  quaternion to float (`sqrtf`, `fmaxf`); `_get_updated_orientation` divides
  by |L| (NaN for L = 0); the neighbour matrix is written without a bound
  check; the edge kernels add only `.xyz` atomically, so `CUDA_print_energy`
  is meaningless with `use_edge`. None of these is reproduced here.
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
