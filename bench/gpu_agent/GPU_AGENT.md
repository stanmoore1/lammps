# Running the standalone oxDNA Kokkos benchmarks on a GPU

These are instructions for an agent working on a machine with an NVIDIA GPU. The goal is
to measure how close LAMMPS KOKKOS oxDNA comes to the raw CUDA code of standalone oxDNA,
using the two standalone Kokkos codes in this directory to explain the gap.

The four codes, all run on the same systems:

| code | what it is | source |
|---|---|---|
| **oxDNA (CUDA)** | upstream reference, raw CUDA | github.com/lorenzo-rovigatti/oxDNA @ c2c74cc0 |
| **bench/oxdna_kokkos** | Kokkos port that mirrors the oxDNA CUDA kernels | this repo |
| **bench/oxdna_kokkos_lammps** | Kokkos code that mirrors the LAMMPS KOKKOS kernel structure; `lammps_overhead = 1` adds the LAMMPS framework (ghosts, comm, binned lists, per-style trimmed lists) | this repo |
| **LAMMPS KOKKOS** | the real thing | github.com/stanmoore1/lammps, branch `oxdna3KK-kk-fixes` |

Everything here was verified on CPU (Serial backend) and compiled with nvcc 12.8 for sm_80.
**Nothing has been run on a GPU yet.** That is your job.

Use the branch `claude/magical-hamilton-6rt90z` of `stanmoore1/lammps`, which contains
`bench/`. All paths below are relative to that checkout (`$REPO`).

## 0. Prerequisites

```bash
nvidia-smi                      # GPU model -> Kokkos arch flag, see below
nvcc --version                  # CUDA >= 12 (tested: 12.8)
cmake --version                 # >= 3.20
g++ --version                   # >= 10 (C++20)
python3 -c "import numpy"       # the case tools need numpy
```

Set these once. The value of `ARCH` depends on the GPU:

| GPU | `ARCH` (Kokkos flag) | `SM` (oxDNA / CMake CUDA arch) |
|---|---|---|
| V100 | `VOLTA70` | 70 |
| A100 | `AMPERE80` | 80 |
| A10, A40, RTX 30xx | `AMPERE86` | 86 |
| L4, L40, RTX 40xx | `ADA89` | 89 |
| H100, H200 | `HOPPER90` | 90 |

```bash
export REPO=$HOME/lammps                 # checkout of claude/magical-hamilton-6rt90z
export ARCH=AMPERE80 SM=80               # from the table
export WORK=$HOME/oxdna_gpu              # builds, cases, results
mkdir -p $WORK
```

## 1. Build the two standalone Kokkos codes

Both use the Kokkos bundled in `$REPO/lib/kokkos` and compile with its `nvcc_wrapper`.
Build **mixed precision**, the apples-to-apples setting:

- **oxDNA:** a CUDA build runs `backend_precision = mixed` by default. Forces are
  computed in FP32 and the integrator runs in FP64.
- **LAMMPS:** compare against a `-D KOKKOS_PREC=mixed` build (compute in FP32,
  accumulate in FP64).

Double builds are useful as a second data point.

```bash
NVW=$REPO/lib/kokkos/bin/nvcc_wrapper
for code in oxdna_kokkos oxdna_kokkos_lammps; do
  for prec in MIXED DOUBLE; do
    opt=""; [ $prec = MIXED ] && opt="-DOXDNA_MIXED_PRECISION=ON"
    cmake -S $REPO/bench/$code -B $WORK/build_${code}_$prec \
          -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_$ARCH=ON -DCMAKE_CXX_COMPILER=$NVW \
          -DCMAKE_BUILD_TYPE=Release -DOXDNA_BUILD_TESTS=ON $opt
    cmake --build $WORK/build_${code}_$prec -j
  done
done
export OXKK=$WORK/build_oxdna_kokkos_MIXED/oxdna_kokkos
export OXKKL=$WORK/build_oxdna_kokkos_lammps_MIXED/oxdna_kokkos_lammps
```

Each build dir then has the code plus `fd_test` and `xcheck`.

## 2. Validate on the GPU (before timing anything)

Run the finite-difference suites from each source directory. They exit non-zero on
failure. Mixed/float builds automatically use looser tolerances.

```bash
(cd $REPO/bench/oxdna_kokkos        && $WORK/build_oxdna_kokkos_DOUBLE/fd_test        | tail -3)
(cd $REPO/bench/oxdna_kokkos_lammps && $WORK/build_oxdna_kokkos_lammps_DOUBLE/fd_test | tail -3)
(cd $REPO/bench/oxdna_kokkos        && $WORK/build_oxdna_kokkos_MIXED/fd_test         | tail -3)
(cd $REPO/bench/oxdna_kokkos_lammps && $WORK/build_oxdna_kokkos_lammps_MIXED/fd_test  | tail -3)
```

Expected: all checks PASS (257 for `oxdna_kokkos_lammps` on CPU). On a GPU, each of
these paths runs code that the CPU runs never exercised:

- device atomics and scatter views;
- the Cuda `LaunchBounds<64,1>` of excv/dh;
- the binned neighbor build with more than one thread per team.

A failure here is a real bug. Report it with the output.

Also compare one energy against the CPU value:

```bash
cd $REPO/bench/oxdna_kokkos_lammps/tests/N8
$WORK/build_oxdna_kokkos_lammps_DOUBLE/oxdna_kokkos_lammps input | head -4
#  step 0 U must be -1.354229 (oxDNA2 N8, T=20C, salt 1.0), as on CPU
```

## 3. Build the two references

### Standalone oxDNA (CUDA)

```bash
cd $WORK
git clone https://github.com/lorenzo-rovigatti/oxDNA.git && cd oxDNA && git checkout c2c74cc0
cmake -B build -DCUDA=ON -DCUDA_COMMON_ARCH=OFF -DCMAKE_BUILD_TYPE=Release   # autodetects the GPU
cmake --build build -j
export OXDNA=$WORK/oxDNA/build/bin/oxDNA
```

### LAMMPS KOKKOS (reference branch)

```bash
cd $WORK
git clone -b oxdna3KK-kk-fixes https://github.com/stanmoore1/lammps.git lammps-ref
cmake -S lammps-ref/cmake -B lammps-ref/build -C lammps-ref/cmake/presets/kokkos-cuda.cmake \
      -D Kokkos_ARCH_$ARCH=ON -D KOKKOS_PREC=mixed \
      -D PKG_CG-DNA=on -D PKG_ASPHERE=on -D PKG_MOLECULE=on -D CMAKE_BUILD_TYPE=Release
cmake --build lammps-ref/build -j
export LMP=$WORK/lammps-ref/build/lmp
```

Build a second LAMMPS with `-D KOKKOS_PREC=double` (e.g. into `build_double`) to pair
with the DOUBLE benches.

## 4. Make benchmark cases

`make_case.sh` tiles one of the bundled systems n x n x n and writes:

- a single oxDNA-style `input` that runs unchanged in oxDNA, `oxdna_kokkos` and
  `oxdna_kokkos_lammps` (NVE, T = 0.1, salt 0.5, dt 0.003, `verlet_skin` 0.5);
- `input_lammps_mode` (the same input + `lammps_overhead = 1`);
- `data.lmp` + `in.lammps` for LAMMPS (the same settings; LAMMPS skin 1.0 = 2 x `verlet_skin`).

```bash
G=$REPO/bench/gpu_agent
for m in 2 3; do
  $G/make_case.sh $m N512 1 $WORK/cases/dna${m}_8k    10000   #   8,192 nt
  $G/make_case.sh $m N512 2 $WORK/cases/dna${m}_65k    5000   #  65,536 nt
  $G/make_case.sh $m N512 4 $WORK/cases/dna${m}_524k   1000   # 524,288 nt
done
```

(`N8` = 128 nt and `N64` = 1024 nt bases are also available for small sizes.)

## 5. Run and collect timings

```bash
for c in $WORK/cases/*; do OXDNA=$OXDNA OXKK=$OXKK OXKKL=$OXKKL LMP=$LMP $G/run_perf.sh $c; done
```

`run_perf.sh` runs every code whose variable is set, keeps the logs as
`<case>/log_<code>.txt`, and prints:

```
case dna2_65k: 65536 nucleotides, 5000 steps
code                                              timesteps/s  U/nt (step 0)
oxDNA (standalone CUDA)                               ...        -1.353820
bench/oxdna_kokkos                                    ...        -1.353819
bench/oxdna_kokkos_lammps (lean)                      ...        -1.353819
bench/oxdna_kokkos_lammps (lammps_overhead = 1)       ...        -1.353819
LAMMPS KOKKOS                                         ...        -1.3538201
```

The step-0 energies must agree to ~1e-5, except oxDNA3 LAMMPS, which may differ by
~1e-4 (known physics differences, see `../oxdna_lammps_vs_standalone`). If they don't,
something is wrong: stop and report.

LAMMPS runs with `-k on g 1 -sf kk -pk kokkos neigh half newton on comm device`. Override
this with `LMP_ARGS=...`, for example to try `neigh full` or `newton off`, and report
which settings are fastest.

Each run is a single measurement. Repeat each case 3 times and report the best, with
nothing else running on the GPU (`nvidia-smi` should show no other process).

## 6. Where the time goes

- **oxdna_kokkos / oxdna_kokkos_lammps:** add `timing = 1` to the input for a
  per-section breakdown. For `oxdna_kokkos` the sections match the oxDNA timers
  (Forces, Lists, First Step, ...); for `oxdna_kokkos_lammps` they match the LAMMPS
  timer (Pair, Bond, Neigh, Comm, Modify). Keep `timing = 0` for the throughput numbers.
- **oxDNA:** prints its own timer table at the end of the log.
- **LAMMPS:** prints the "MPI task timing breakdown". Its sections are only
  GPU-accurate with `timer full sync` added to `in.lammps` before `run`, which costs some
  speed, so don't use it for the throughput numbers.
- **Kernel-level, all four codes:** use Nsight Systems, e.g.
  ```bash
  nsys profile -o $c/prof_lmp --stats=true $LMP -k on g 1 -sf kk -pk kokkos neigh half newton on comm device \
       -in in.lammps -var data data.lmp -var steps 500
  ```
  Compare per-kernel times between `oxdna_kokkos_lammps (lammps_overhead = 1)` and LAMMPS.
  The bench's kernel sequence was checked kernel-for-kernel against LAMMPS on CPU
  (`../oxdna_kokkos_lammps/README.md`, "Kernel sequence vs. a real LAMMPS run"), so the
  kernel names correspond.

## 7. What to report

1. GPU model, driver, CUDA version, and the commit of each code.
2. `fd_test` results for all four bench builds (pass/fail counts, any failure output).
3. For every case, the timesteps/s of all five rows plus the step-0 energies, in
   mixed and in double.
4. The per-section breakdown (`timing = 1` / LAMMPS timer) for the 65k case of each code.
5. The top 10 kernels by time from `nsys` for LAMMPS and for
   `oxdna_kokkos_lammps (lammps_overhead = 1)` on the 65k oxDNA2 case.
6. Anything that failed, verbatim.

The key ratios:

- oxDNA / LAMMPS: the gap to close.
- oxdna_kokkos / oxDNA: whether Kokkos itself costs anything.
- oxdna_kokkos_lammps lean / oxdna_kokkos: the cost of the LAMMPS kernel structure.
- oxdna_kokkos_lammps overhead / lean: the cost of the LAMMPS framework.
- LAMMPS / oxdna_kokkos_lammps overhead: whatever the bench does not capture.

## Notes and pitfalls

- Run from any directory: `run_perf.sh` changes into the case directory itself, and
  paths in `input` are relative to it.
- The oxDNA3 cases use
  `$REPO/bench/oxdna_kokkos/params/oxDNA3_sequence_dependent_parameters.txt` via an
  absolute path written by `make_case.sh`. If you move `$REPO`, regenerate the cases.
- `bench/oxdna_kokkos` warns if `backend_precision` in the input does not match its
  compiled precision. That is expected when you run the DOUBLE build on these inputs.
- Brownian thermostats differ between the codes, so the cases run NVE. LAMMPS uses its
  own integrator and masses, so its trajectory differs; only the per-step cost and the
  step-0 energy are comparable.
- If the 524k case does not fit in device memory in one of the codes, use `N512 3`
  (221,184 nt) instead, and report which code ran out.
