# N8 benchmark case

A 128-nucleotide oxDNA2 system (8 strands of 16 nt) used for performance and
correctness comparison against the standalone oxDNA. The configuration and
topology are taken from Erik Poppleton's oxDNA performance benchmark suite:
<https://github.com/ErikPoppleton/oxDNA_performance> (directory `N8`).

Model settings for this case (from the original benchmark `input`):
oxDNA2, average sequence, `T = 20C` (= 0.097717 reduced), `salt_concentration = 1.0`,
`dt = 0.003`, Brownian thermostat (`newtonian_steps = 103`, `diff_coeff = 2.5`).

## Run (Kokkos)

```bash
# from bench/oxdna_kokkos, after building (see ../../README.md)
./build/oxdna_kokkos -top tests/N8/topology_N8.top -conf tests/N8/init_conf_N8.dat \
                     -model 2 -salt 1.0 -T 0.097717 -dt 0.003 \
                     -newt 103 -diff 2.5 -steps 100000 -freq 1000
```

## Cross-check vs. standalone oxDNA

Per-particle potential energy and forces (build with `-DOXDNA_BUILD_TESTS=ON`):

```bash
./build/xcheck 2 0.097717 1.0 tests/N8/topology_N8.top tests/N8/init_conf_N8.dat ft_kok.txt
```

compared against the standalone oxDNA `potential_energy split = 1` and
`particle_force_torque` (`lab_frame = 1`) observables on the same config.

Validated agreement (standalone CPU/double vs. Kokkos double):

| Quantity | Result |
|---|---|
| Total potential energy / particle | −1.354230 (std) vs −1.354229 (Kokkos) |
| Per-group energies (backbone, stacking, nonbonded incl. Debye–Hückel) | match to ~1e-5 |
| Per-particle force / torque vectors | ~6e-5 / ~4e-4 relative |
| ⟨U/particle⟩ over 300k Brownian-NVT steps | −1.38233 (std) vs −1.37963 (Kokkos), 0.2% |
| ⟨K/particle⟩ (temperature) | ≈ 3T in both |

## CPU performance (single comparison point; both designed for GPU)

128 particles, oxDNA2, double precision, Release builds, on this host:

| Build | Threads | steps/s |
|---|---|---|
| standalone oxDNA (CPU) | 1 | ~6050 |
| Kokkos Serial | 1 | ~4400 |
| Kokkos OpenMP | 4 | ~6600 |

N8 is small, so CPU threading overhead limits scaling; the Kokkos data layout
and one-thread-per-edge kernel are intended for GPU throughput
(`-DKokkos_ENABLE_CUDA=ON`).

## Larger cases from the same suite

`N64` (1024 nt) and `N512` (8192 nt) are committed alongside this case and
verified the same way. The much larger `N4096` (65536 nt) and `N32768`
(524288 nt) configurations are not committed here (≈15 MB / ≈70 MB); fetch them
from the upstream performance repo to run. All sizes reproduce the standalone
per-particle energies to ~1e-5 and forces/torques to ~1e-4.

Note: the `N32768` configuration stores only 9 columns per line (position, a1,
a3) with no velocity/angular-momentum — the config reader handles such
velocity-less confs (missing v/L default to zero).
