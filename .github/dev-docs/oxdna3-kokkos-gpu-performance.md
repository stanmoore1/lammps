# oxDNA3 KOKKOS vs standalone oxDNA CUDA: why "oligomer" is slow, and what to try

Handoff notes for an agent that will profile on a GPU.  Written from a code
comparison of branch `oxdna3KK` (head ec131639a7) with the standalone code
`lorenzo-rovigatti/oxDNA` (head c2c74cc).  The comparison was done without a GPU;
CPU (OpenMP) Kokkos runs were used only to count kernels and neighbors.
Everything below marked **CONFIRM** is a hypothesis that must be checked
with a GPU profile before code is changed.

## 1. The performance gap (read from the benchmark plots, +/- 10%)

Time per step, RTX 4090 (benchmark repo `lrussell676/oxDNA-KOKKOS-LAMMPS-Benchmarking`, `oxDNA3/`):

| system    | nucleotides | KOKKOS  | standalone | ratio |
|-----------|-------------|---------|------------|-------|
| polybrick | 2174k       | 12.6 ms | 12.1 ms    | 1.04  |
| polybrick | 543k        | 3.1 ms  | 2.7 ms     | 1.15  |
| oligomer  | 8192        | 0.14 ms | 0.055 ms   | 2.5   |
| oligomer  | 65536       | 0.40 ms | 0.15 ms    | 2.7   |
| oligomer  | 524288      | 2.7 ms  | 1.0 ms     | 2.7   |

RTX 3050 oligomer: about 3x at every size.  RX 7900XT KOKKOS oligomer is
flat at about 0.85 ms from 8k to 65k nucleotides (standalone has no HIP port).

**Cost per nucleotide per step is the key number:**

| | polybrick (2174k) | oligomer (524k) |
|---|---|---|
| KOKKOS, 4090     | 5.8 ns | 5.2 ns |
| standalone, 4090 | 5.6 ns | 1.9 ns |

The KOKKOS cost per nucleotide hardly changes between the dense brick and the
dilute oligomer.  The standalone code gets about 3x cheaper per nucleotide for the dilute system.
So the gap is not "KOKKOS is slow at pair math in general".  KOKKOS has a
per-nucleotide cost that does not shrink when there are few neighbors.  In the
dense system that cost is hidden, because standalone also pays a lot there:
it evaluates every listed pair twice (full list) with no center-of-mass screening.

The oligomer sizes are 8^3, 16^3, 32^3 and 64^3 copies of a 16-nucleotide
octamer duplex (`examples/PACKAGES/cgdna/examples/lj_units/oxDNA3/duplex2`).
With the example's 40^3 box per duplex, the system is extremely dilute.  Each
nucleotide's neighbors are then only the other 15 nucleotides of its own duplex.
**CONFIRM** the box size and the input settings in the benchmark inputs.

## 2. What the standalone code does per step (mixed precision, default)

Source: `src/CUDA/Backends/MD_CUDABackend.cu:567-619`, `src/CUDA/Interactions/CUDA_DNA3.cuh`.

* 4 kernels per step, one thread per particle, 64 threads per block:
  `first_step_mixed` (integrates, and flags a list rebuild when any particle
  moves more than the skin), `set_external_forces` (zeroes forces),
  `DNA3_forces`, `second_step_mixed`.
* `DNA3_forces` is one fused kernel over a full list with no atomics.  Each thread:
  * computes its own two bonded neighbors (n3, n5), so each bond is computed twice;
  * loops over its nonbonded neighbors and evaluates excluded volume, Debye-Hueckel,
    hbond, cross-stacking and coaxial stacking in one pass;
  * accumulates the force and torque in registers and writes them once.
* All pair math is float and built with `-use_fast_math`.  Positions are float4
  with the type packed into .w.  Quaternions are float4 and turned into axes for
  each neighbor.  Parameter tables are read with `__ldg`.
* There is no center-of-mass screening.  Every listed pair goes through every
  term, each gated by its own distance window.
* List radius = rcut + 2 * verlet_skin (rcut includes 2*|r_back| + DH cutoff;
  about 3.26 at 0.5 M salt).  Cell count is capped at about 2N, independent of
  box volume (`CUDASimpleVerletList.cu:96-110`).
* Sorting is off by default.  There are about 6 `cudaDeviceSynchronize` per step (timers).

## 3. What KOKKOS does per step (`oxdna3KK`)

Kernel profile (Kokkos Tools simple-kernel-timer, OpenMP backend, 8192
nucleotide oligomer proxy, 2000 steps).  On a GPU the kernels that run every step are:

* 7 force kernels: `oxdna3/excv` (thread per atom, half list), `oxdna3/dh`
  (thread per atom), `oxdna3/hbond` / `oxdna3/xstk` / `oxdna3/coaxstk` (GPU path:
  thread per screened pair), `oxdna3/stk` and `oxdna3/fene` (thread per bond);
* `nve/asphere` fused integrate, `langevin` post_force and angmom (2),
  `OXDNA/LRF` quaternion to frame (1), 2 zeroing kernels for f and torque;
* the neighbor check reduction, which reads back to the host every step;
* forward and reverse comm kernels (the oligomer proxy had 0 ghosts on 1 rank).

That is about 14-16 launches plus 1 host sync per step, against 4 for standalone.
Every force kernel re-reads x (double[3]) and the nx/ny/nz frames for both atoms
and re-derives the interaction sites.  Each kernel scatters into f and torque
with atomics; these are double-precision atomics in mixed and double builds.  `oxdna3/xstk`
issues up to 18 atomic adds per pair
(`src/KOKKOS/pair_oxdna3_xstk_kokkos.cpp:815-864`).  In the screened list
(`fix_oxdna_npair_kokkos.cpp`) the pairs of one atom a are adjacent, so the
threads of a warp atomically add to the same address.

CPU time split for the oligomer proxy: xstk 22%, stk 12%, hbond 12%, dh 10%, excv 6%,
nve 6%, coaxstk 4%, fene 4%, langevin 4%.  For a dense proxy (the same duplex in a
2.8 x 2.8 x 3.72 box, 40 half-neighbors per atom): dh 17%, hbond 14%, xstk 13%,
excv 12%, coaxstk 9%, stk 6%.  So in the dilute system the bonded kernels and
xstk take a much larger share.  The host path of hbond differs from the GPU path,
so treat these splits only as a guide.

Neighbor list, measured for the oligomer proxy (0.5 M salt, cutoff set by DH, 2.30):

| skin | master cutoff | half neighbors per atom | rebuilds per 2000 steps (dt 0.003) |
|------|---------------|-------------------------|------------------------------------|
| 0.3  | 2.60          | 7.2                     | 29                                 |
| 1.0  | 3.30          | 7.5                     | 5                                  |
| 2.0  | 4.30          | 7.5                     | 1                                  |

For the oligomer, the neighbor count does not depend on the skin (the dense
proxy went from 18 to 94).  Note that the kk-fixes branch documents that the skin
must be at least about 1.0: the cutoffs are between interaction sites, but the
neighbor list uses the centers of mass (commit 713adeda23).

## 4. Hypotheses, ranked, with how to confirm and what to try

### H1. Work split into 7 passes with atomics, where standalone does 1 pass with no atomics

* In a dilute system, most of the per-nucleotide work is intra-duplex:
  * 7 half-neighbor visits per nucleotide in each of excv, dh, hbond, xstk and coaxstk;
  * about 1 bond per nucleotide in each of stk and fene.

  Each pass reloads positions and frames, recomputes the site vectors, and ends with
  6-18 double atomics.  Standalone loads each neighbor once and computes all terms
  in registers.
* **CONFIRM:** in Nsight Systems, the sum of the 7 force kernels against
  `DNA3_forces`; in Nsight Compute, the memory throughput, L2 atomic throughput
  (`lts__t_sectors_op_red.sum`, `lts__t_sectors_op_atom.sum`) and stall reasons
  of xstk/hbond/stk.
* **TRY (in this order):**
  1. Build on the earlier fusion work: branches `oxdna-hbxstk-fusion`,
     `oxdna-hbxstk-fusion-clean` (hbond+xstk fused, precomputed frames,
     packed per-type-pair coefficient structs, launch bounds) and
     `oxdna-screened-acosf-lb`.  Rebase them onto the current kk-fixes code and
     measure them.
  2. Fuse all nonbonded terms (excv + dh + hbond + xstk + coaxstk) into one
     kernel, one thread per atom, over a full list, accumulating in registers with
     no atomics, as `DNA3_forces` does.  With newton off and a full list, each pair
     is computed twice but with no atomics.  In the dilute regime (7-15 neighbors)
     this should win; in the dense brick, compare against the current screened
     half list.  Keep per-style energy and virial tallies only when `eflag`/`vflag`
     are set.
  3. Fuse `oxdna3/stk` and `oxdna3/fene`: they share the bond list and the
     precomputed 3'/5' table.  Better still, compute the bonded terms in the
     per-atom kernel from `id3p`/`id5p` mapped to local indices, as standalone
     does, and remove the bond-list atomics.
  4. If one thread per pair is kept: reduce the pairs of the same atom a within the
     warp (segmented reduction) before the atomic, or interleave the pair order so
     a warp does not hit one address 32 times.

  These cross the LAMMPS "one style = one kernel" boundary.  One option is a
  single hybrid-aware "oxdna3/kk fused" path, enabled when all six oxDNA3 pair
  sub-styles are present in `pair_style hybrid/overlay`.

### H2. Launch and sync latency at small N

* At 8192 nucleotides on the 4090, KOKKOS takes 140 us per step against 55 us.  That fits about 15
  launches plus host syncs at a few us each.  The 7900XT being flat at about 0.85 ms
  points the same way (HIP launches cost more).
* **CONFIRM:** Nsight Systems timeline of a 8k oligomer run: count kernels,
  `cudaStreamSynchronize`/`cudaMemcpy` per step, and gaps between kernels.
  Check which host syncs happen every step: the neighbor check, thermo, and any
  `DualView::sync` host round trip.
* **TRY:** the fusion in H1; fold the f/torque zeroing into the first force kernel;
  fold `OXDNA/LRF` into the end of `nve/asphere` final integrate (the ghost frames
  then need forward comm of the frames or a small ghost-only kernel); fuse the
  langevin post_force into the force pass or integrator.  Run with thermo output
  far apart (`thermo 0` or large).

### H3. Rebuild costs that scale with box volume (dilute systems)

* On GPUs, `NeighborKokkos::set_binsize_kokkos()` sets binsize = cutneighmax
  (`neighbor_kokkos.cpp:392-395`), so the number of bins is proportional to box
  volume.  `k_bins` is `mbins x atoms_per_bin(16)` ints
  (`nbin_kokkos.cpp:31,67-71`).  The GPU build uses one team per 2 bins
  (`npair_kokkos.cpp:264-323`); empty bins exit early but still cost a block.
  With a 40^3 box per duplex, there are hundreds of bins per nucleotide.
* Per rebuild the oxDNA helper fixes also:
  * run a screening pass plus a scan, then read back the pair count with a
    `deep_copy` to the host (`fix_oxdna_npair_kokkos.cpp:220`);
  * run the 3'/5' prime-neighbor precomputes (3 kernels).
* **CONFIRM:** `bins = ...` in the log, `Neighbor list builds`, and the time
  per rebuild step in Nsight Systems.  Check GPU memory use at 4.2M nucleotides.
* **TRY:**
  * For the oligomer, use a larger skin: measured neighbor counts do not change,
    and rebuilds drop from 29 to 1 per 2000 steps.
  * Set `package kokkos binsize` (or `neigh_modify binsize`) larger for dilute systems.
  * Consider capping mbins at about 2N as standalone does.

### H4. Memory layout

* x is `double[3]` LayoutRight (24 B stride); frames are 3 separate `[3]` arrays.
  Standalone uses a float4 position and a float4 quaternion.
* **TRY** after H1: a packed float4 position+type copy for the force kernels;
  one array of frames (the fusion branch already did "LayoutRight AoS").
  Alternatively, rebuild the frames from a float4 quaternion in the kernel
  (tried in `aa59e87056`, later reverted to precomputed frames in `06eb06031f`).

### H5. Things that are NOT the cause (checked)

* The prime-neighbor map lookups run only on rebuild steps.
* Ghost communication: the oligomer proxy has 0 ghosts on 1 rank.
* The neighbor-list size: KOKKOS (cutoff 3.30 at skin 1.0) is about the same as
  standalone (about 3.36).
* Build precision: the benchmark build's precision setting has been checked and
  is correct.

## 5. Caveats

* The bug-fix branch `oxdna3KK-kk-fixes` (lammps PR #57 in this fork) added
  atomics to the hbond/xstk/coaxstk GPU pair kernels, which had races with full
  lists.  The published benchmark likely predates this.  Re-baseline on the
  fixed code before measuring improvements.
* Standalone can skip some work that LAMMPS cannot (no ghosts, minimum image,
  energies always in .w, no virial).  Compare with `thermo` far apart.

## 6. Suggested GPU session plan

1. Record the build config (arch, nvcc flags) and the benchmark inputs (skin,
   salt, thermostat, box, `package kokkos` options).
2. `nsys profile --stats=true` on oligomer 65k for 1000 steps: kernel list, time
   per kernel, launches and syncs per step, rebuild spikes.
3. `ncu --set full -k regex:"Xstk|Hbond|Stk|Excv|Dh|Coaxstk|FENE"` on a few
   launches: atomics, registers and occupancy, memory throughput.
4. Then apply H1 (fusion), H2 and H3 in order of measured payoff; check each
   change against the CPU energies (`examples/PACKAGES/cgdna/examples/test_KOKKOS.sh`).

## 7. Reproducing the CPU proxy measurements

Build (CPU only): `cmake -S cmake -B build -G Ninja -D PKG_KOKKOS=on -D Kokkos_ENABLE_OPENMP=on
-D BUILD_OMP=on -D PKG_CG-DNA=on -D PKG_MOLECULE=on -D PKG_ASPHERE=on -D BUILD_MPI=off`.

Oligomer proxy: the `duplex2` input with `replicate ${n} ${n} ${n}`,
`fix nve/asphere` + `fix langevin 0.1 0.1 2.5 457145 angmom 10`, `timestep 0.003`,
`oxdna3/dh 0.1 0.5`, and `-k on t 4 -sf kk -pk kokkos neigh half newton on`.

Dense proxy: the same with the data file box changed to `-1.31 1.49`, `-2.15 0.65`,
`-0.20 3.52` (a square-lattice bundle of parallel duplexes, 0.55 nt per sigma^3).

Kernel counts: `KOKKOS_TOOLS_LIBS=.../libkp_kernel_timer.so` from kokkos-tools;
Kokkos needs `Kokkos_ENABLE_LIBDL=on`.
