# oxDNA3 KOKKOS vs standalone oxDNA CUDA: why "oligomer" is slow, and what to try

Handoff notes for an agent that will profile and optimize on a GPU.  They compare
branch `oxdna3KK` (head ec131639a7) with the standalone code
`lorenzo-rovigatti/oxDNA` (head c2c74cc).  No GPU was available, so the analysis
combines:

* a line-by-line code comparison of the force paths;
* a CUDA 12.6 compile of both codes for sm_89 (RTX 4090): registers, stack,
  kernel-parameter size and SASS instruction mix for each kernel (section 5);
* CPU (OpenMP) KOKKOS runs of an oligomer proxy and a dense proxy, with the GPU
  pair kernels forced onto the host and instrumented to count how many pairs
  pass each early-exit test (section 4), plus Kokkos Tools kernel counts.

Out of scope here: the build precision (checked by the user) and double
precision constants in the kernels (handled separately).  Everything marked
**CONFIRM** must be checked with a GPU profile before code is changed.

## 1. The performance gap (read from the benchmark plots, +/- 10%)

Time per step, RTX 4090 (benchmark repo `lrussell676/oxDNA-KOKKOS-LAMMPS-Benchmarking`, `oxDNA3/`):

| system    | nucleotides | KOKKOS  | standalone | ratio |
|-----------|-------------|---------|------------|-------|
| polybrick | 2174k       | 12.6 ms | 12.1 ms    | 1.04  |
| polybrick | 543k        | 3.1 ms  | 2.7 ms     | 1.15  |
| oligomer  | 8192        | 0.14 ms | 0.055 ms   | 2.5   |
| oligomer  | 65536       | 0.40 ms | 0.15 ms    | 2.7   |
| oligomer  | 524288      | 2.7 ms  | 1.0 ms     | 2.7   |

RTX 3050 oligomer: about 3x at every size.  RX 7900XT KOKKOS oligomer is flat
at about 0.85 ms from 8k to 65k nucleotides (there is no standalone HIP port).

Cost per nucleotide per step:

| | polybrick (2174k) | oligomer (524k) |
|---|---|---|
| KOKKOS, 4090     | 5.8 ns | 5.2 ns |
| standalone, 4090 | 5.6 ns | 1.9 ns |

The oligomer sizes are 8^3 ... 64^3 copies of the 16-nucleotide octamer duplex in
`examples/PACKAGES/cgdna/examples/lj_units/oxDNA3/duplex2`.  **CONFIRM** the box
per duplex and the other input settings in the benchmark inputs.

## 2. Main conclusions

1. **The useful work is the same in dense and dilute systems; only the rejected
   neighbor visits differ.**  In both CPU proxies the number of pairs that
   actually interact per step is identical: hbond 3,880, xstk 7,480,
   coaxstk about 400, stk 6,656 (section 4).  The dense proxy only adds neighbors
   that are rejected by cheap distance tests.
2. **Standalone scales with rejected visits; KOKKOS does not.**  Standalone
   runs one pass over a full list with no screening, so it pays for every
   rejected neighbor.  That makes it about 3x more expensive per nucleotide in
   the brick than in the oligomer.  KOKKOS screens those neighbors cheaply
   (center-of-mass pre-filter, half lists), so its cost per nucleotide hardly
   changes with density.
3. **So the oligomer shows KOKKOS's baseline cost.**  That baseline is the cost of
   the useful interactions plus the fixed per-atom, per-thread and per-launch
   overhead, and it is about 2.7x the whole standalone step.  In the brick the
   baseline is hidden, because standalone spends that much on rejected visits.
   **To speed up KOKKOS, make the useful work and the fixed overhead cheaper.
   Better screening will not help.**
4. The baseline is high because of how the work is split and laid out, not
   because the physics costs more (sections 3-5):
   * about 14 launches per step, against 4;
   * 3 separate thread-per-pair kernels (hbond, xstk, coaxstk) over the same
     screened list; 85-99% of their threads exit early, so warps run mostly
     empty (divergence);
   * about 12 separate 4-byte gathers per neighbor visit, against 2-3 16-byte
     loads;
   * 192 atomic-update sites across the force kernels, against 0;
   * a stk kernel that spills to local memory;
   * 14-16 KB of kernel parameters per force launch, against about 100 bytes.

## 3. Side-by-side structure

| | standalone oxDNA CUDA (mixed backend) | KOKKOS `oxdna3KK` |
|---|---|---|
| kernels per step (no rebuild) | 4: `first_step_mixed`, `set_external_forces`, `DNA3_forces`, `second_step_mixed` | about 14: 7 force kernels (excv, dh, hbond, xstk, coaxstk, stk, fene), nve/asphere (fused), langevin x2, OXDNA/LRF, zero f, zero torque, neighbor-check reduction (plus comm kernels when there are ghosts) |
| host syncs per step | rebuild flag read after a synced timer | `check_distance` reduction read back every step (`neighbor_kokkos.cpp:232-238`) |
| force decomposition | one fused kernel, one thread per particle, full list; all terms in one pass, in registers | 5 nonbonded + 2 bonded kernels; each re-reads positions and frames and re-derives the interaction sites |
| nonbonded thread mapping | one thread per particle, loop over about 13-15 (oligomer) full-list neighbors | excv, dh: one thread per atom over the half list; hbond, xstk, coaxstk: **one thread per screened pair** (`fix_oxdna_npair_kokkos.cpp`) |
| bonded (FENE, stacking, bonded excv) | inside `DNA3_forces`: each bond computed twice, once from each end, no atomics | `oxdna3/stk` and `oxdna3/fene`: one thread per bond, with atomics on both atoms |
| force accumulation | registers, one store per particle | atomics on f and torque (double in mixed builds); 192 atomic-update sites across the force kernels |
| per-particle data | float4 position (type packed in .w) + float4 quaternion; axes rebuilt per neighbor | x as `float[3]` LayoutRight (12 B, not vector aligned); frames as 3 separate `float*[3]` LayoutLeft views (`fix_oxdna_lrf_kokkos.h:55`), i.e. 9 scalar arrays |
| gathers per neighbor visit | 2 x 16 B (pos, quat) + 8 B bonds + 2 type bytes | 3 (x) + 3-9 (frames) scalar gathers, each in a different array, repeated in every kernel that visits the pair |
| parameter tables | `__ldg` global tables, flat `[6][5][5][6]` sequence-dependent | a separate `View` per parameter (2-D or 4-D), each gathered separately |
| thermostat | Brownian, applied every `newtonian_steps` (often about 100) | `fix langevin ... angmom`: 2 RNG kernels every step (**CONFIRM** the thermostats in the benchmark inputs) |
| list build | cells capped at about 2N; list = rcut + 2*skin; one sync for the overflow flag | bins of size cutneighmax (count proportional to box volume), then the screening pass + scan + `deep_copy` of the pair count to the host, then 3 prime-neighbor precomputes |

## 4. Measured workload: pairs per step and early-exit pass rates

Method: in a copy of `oxdna3KK`, the hbond and coaxstk GPU pair kernels were
forced onto the host (screened pair path).  xstk already uses that path on all
backends.  Every top-level early `return` in the xstk, hbond, coaxstk and stk
operators, and the neighbor and cutoff tests in dh and excv, got an atomic
counter.  Settings: 8192 nucleotides, 200 steps, skin 1.0, 0.5 M salt.
Oligomer proxy: `duplex2` replicated 8^3 in its 40^3 box.  Dense proxy: the same
duplex in a 2.8 x 2.8 x 3.72 box (parallel-helix bundle, 0.55 nt per sigma^3).

Per step, all 8192 nucleotides:

| quantity | oligomer (7.5 half neighbors/nt) | dense (39 half neighbors/nt) |
|---|---|---|
| screened pairs = threads launched in each of hbond, xstk, coaxstk | 50,688 (6.2/nt) | 77,312 (9.4/nt) |
| ... exit immediately (1-2 bonded pairs, special factor 0) | 13% | 9% |
| hbond: pass the radial test / compute fully | 9.1% / 7.7% (3,879) | 6.0% / 5.0% (3,880) |
| xstk: pass the radial test / compute fully | 16.5% / 14.8% (7,480) | 10.8% / 9.7% (7,477) |
| coaxstk: pass the terminal filter / pass all angle tests / pass the radial test | 10% / 3.9% / 0.8% (404) | 12% / 2.5% / 0.5% (403) |
| stk bonds (all pass every test) | 6,656 | 6,656 |
| dh neighbor visits / within the DH cutoff | 61,440 / 41,340 | 322,560 / 108,063 |
| excv neighbor visits | 61,440 | 322,560 |

What this means for the GPU:

* In each pair kernel, 85-99% of the threads leave early.  Because the pairs of
  one atom are adjacent in the screened list, a warp of 32 pairs almost always
  contains at least one pair that runs the full angular path.  The warp then
  executes that path with only a few active lanes, so each kernel costs about
  (screened pairs / 32) x (full path length).
* 13% of the screened pairs are 1-2 bonded pairs.  They are in the list only
  because excv needs them for the bonded excluded volume: `neighbor.cpp:560-585`
  forces `special_flag = 2` whenever an `ox*/excv` style is present.  hbond, xstk
  and coaxstk launch a thread for each of them just to return.
* coaxstk launches a thread for every screened pair.  Only 0.8% of them pass,
  and it tests the cheap radial window **last**, after 4 angle terms each with
  an acos (`pair_oxdna2_coaxstk_kokkos.cpp:1050-1066`).  Standalone tests the
  r^2 window first.
* hbond has no base-complementarity test before the site geometry, sqrt and F1
  evaluation.  Non-complementary pairs are rejected inside `hbond_radial_terms`
  through epsilon = 0.  Standalone tests `int_type == 3` and the r^2 window first.

## 5. Static GPU code metrics (sm_89, CUDA 12.6, `KOKKOS_PREC=mixed`)

KOKKOS kernels as launched in an oxDNA3 run: half list, meaning HALFTHREAD on
the GPU, newton on, no energy/virial.  The instruction counts are static SASS
counts for the whole kernel, not dynamic counts.

| kernel | regs | stack (B) | kernel params (B) | SASS instr | LDG | atomic sites | MUFU | local ld/st |
|---|---|---|---|---|---|---|---|---|
| standalone `DNA3_forces` (everything) | 153 | 0 | ~100 | 6,070 | 409 | 0 | 221 | 0 |
| standalone `first_step_mixed` | 42 | 48 | - | 695 | 15 | 0 | 10 | 8 |
| standalone `second_step_mixed` | 22 | 0 | - | 62 | 8 | 0 | 0 | 0 |
| KK xstk (`ComputeNpair`) | 74 | 0 | 3,904 | 1,227 | 81 | 18 | 19 | 0 |
| KK hbond (`ComputeGPUPair`) | 80 | 0 | 16,200 | 1,689 | 65 | 18 | 32 | 0 |
| KK coaxstk (`ComputeGPUPair`) | 72 | 0 | 14,584 | 1,466 | 63 | 18 | 21 | 0 |
| KK excv (per atom) | 94 | 0 | 15,304 | 1,382 | 80 | 42 | 23 | 0 |
| KK dh (per atom) | 72 | 0 | 4,096 | 327 | 31 | 12 | 2 | 0 |
| KK stk (per bond) | 72 | **96** | 14,784 | 3,083 | 133 | **60** | 22 | **84** |
| KK fene (per bond) | 42 | **64** | 2,392 | 933 | 33 | 24 | 8 | **33** |
| KK nve/asphere fused | 47 | 0 | 1,384 | 590 | 34 | 0 | 13 | 0 |
| KK langevin angmom + post_force | 48 / 40 | 0 | 3,704 / - | 671 / 571 | 20 / 11 | 2 / 2 | 15 / 16 | 0 |
| KK OXDNA/LRF | 34 | 0 | 1,512 | 129 | 6 | 0 | 0 | 0 |

Notes:

* The "kernel params" column is the size of constant bank 0 (`CONSTANT[0]`
  from `cuobjdump -res-usage`).  That is the kernel parameters plus about
  350 B of fixed overhead.  Kokkos passes the functor, i.e. the whole pair-style object
  (`*this`), by value.  That object holds every coefficient table as a DualView
  (host and device halves) plus all the Pair base members.  The GPU needs only
  a few device views and scalars.  Large kernel parameters make each launch
  slower (**CONFIRM**: `cudaLaunchKernel` time in Nsight Systems).
* Register counts are moderate (72-94), so occupancy is not the obvious
  limit.  Standalone runs at 153 registers and about 25% occupancy and is still
  faster.
* stk and fene use a stack frame with local-memory traffic (84 and 33 local
  load/store instructions).  Check with `-Xptxas -v` and fix these first; they
  run on every bond, every step.

## 6. Why the oligomer baseline is 2.7x: ranked contributors

These are estimates, to be confirmed by the GPU profile in section 8.

1. **Divergent thread-per-pair kernels over the same screened list (hbond, xstk,
   coaxstk).**  That is 3 x 6.2 = 18.6 pair threads per nucleotide per step,
   of which about 1.4 do useful work.  Each thread gathers 10-20 scalars and
   each warp runs the full path.  Standalone runs about 13 neighbor iterations
   per nucleotide in one kernel and loads each neighbor once.
2. **Memory-access pattern.**  About 12 separate 4-byte gathers per neighbor
   visit (x as unaligned float[3], frames as 9 separate arrays), repeated in
   up to 5 kernels, against 2 aligned float4 loads once.  In the oligomer every
   gather is for a neighbor in the same duplex, so these are cache hits, but
   they still cost load instructions and L1 bandwidth.
3. **Atomics.**  192 atomic-update sites across the force kernels.  Screened
   pairs of the same atom a are adjacent, so a warp's atomics to a's force and
   torque hit the same address and serialize.  Standalone does 0.
4. **Bonded kernels.**  stk has 3,083 SASS instructions, 60 atomic sites and a
   96 B stack; fene has a 64 B stack.  They run for every bond (0.81 per
   nucleotide) and none of their work is filtered out.
5. **Per-step fixed overhead.**  About 14 launches with 1.4-16 KB parameters,
   1 host sync, 2 zeroing kernels, a thermostat every step and a separate frame
   kernel.  This dominates at 8k nucleotides (0.14 vs 0.055 ms), and the
   7900XT's flat 0.85 ms fits it.
6. **Neighbor rebuild** (minor at skin >= 1.0: 5 rebuilds per 2000 steps in
   the proxy).  With a small skin it becomes important: 29 rebuilds per 2000
   steps at skin 0.3, and GPU bins scale with box volume (section 3).

## 7. Master list of suggestions (deduplicated, all sources)

This merges the suggestions from this analysis with those from other agents.
The other agents based theirs on a Kokkos Tools log of a 300-step oxDNA3 run
(128 nt, 31 neighbor builds, GPU code paths), OpenMP measurements and code reading.

* Line references marked (kkf) are to branch `oxdna3KK-kk-fixes`; the others
  are to `oxdna3KK`.
* Evidence: M = measured, C = code reading, S = static SASS/compile metrics
  (section 5).
* Impact is the expected impact on the GPU oligomer benchmark: H(igh), M(edium),
  L(ow), or 0 (no GPU effect).  Effort: S(mall), M(edium), L(arge).
* Not covered: double precision constants (handled separately) and build
  precision (checked).

Validate every change against the CPU energies
(`examples/PACKAGES/cgdna/examples/test_KOKKOS.sh`) and re-benchmark both systems.

### A. Kernel structure (the main lever for the oligomer baseline)

| # | suggestion | evidence | impact | effort |
|---|---|---|---|---|
| A1 | Full standalone-style force kernel: one thread per atom, full list, bonded n3/n5 terms and all nonbonded terms in one pass, accumulated in registers, one write, no atomics; energy/virial only when requested.  Could be a single hybrid-aware oxDNA3 path, enabled when all six pair sub-styles and `oxdna3/fene` are present.  Compare against A2 on both benchmarks (the brick may favor the screened half list). | C, S, M (sec. 4-6) | H | L |
| A2 | Fuse hbond + xstk + coaxstk into one screened-pair kernel that computes the shared geometry once.  Start from `oxdna-hbxstk-fusion(-clean)` (packed per-type-pair coefficient structs, launch bounds; rebase onto kk-fixes). | C, S | H | M |
| A3 | Two-phase pair evaluation: a cheap radial-window pass appends survivors to a compact list (warp-aggregated append or scan), then the heavy kernel runs only on survivors.  Measured survival: 8-15% (hbond, xstk), under 1% (coaxstk), so warps go from mostly idle to mostly full. | M (sec. 4) | H | M |
| A4 | Fewer atomics: accumulate in registers across a thread's pairs and do one atomic update per atom.  xstk does 2 rounds of atomics per active pair, 18 in total, with no register accumulation (`pair_oxdna3_xstk_kokkos.cpp:815-864`; (kkf) `:886-929`).  Warp-reduce the adjacent pairs of the same atom a before the atomic, or use one thread per atom a.  192 atomic-update sites across the 7 force kernels, against 0 in standalone. | C, S | H | M |
| A5 | Bonded terms: fuse stk + fene (same bond, sites and 3'/5' table), or compute them per atom from local id3p/id5p (each bond twice, no atomics) inside A1.  Remove the local-memory stack of stk (96 B, 84 LDL/STL) and fene (64 B, 33 LDL/STL). | S | M-H | S-M |

### B. Pair lists and early exits

| # | suggestion | evidence | impact | effort |
|---|---|---|---|---|
| B1 | Drop 1-2 bonded pairs (special factor 0) from the screened list in `TagFixOxdnaNpairFill`.  They are 9-13% of all screened-pair threads and exit at once; they are only in the list because excv forces `special_flag = 2` (`neighbor.cpp:560-585`). | M | M | S |
| B2 | Separate compacted lists built at rebuild time: coaxstk only strand-end pairs, i.e. filter at rebuild instead of per thread (the per-thread filter is at (kkf) `pair_oxdna2_coaxstk_kokkos.cpp:999`); hbond only complementary types; xstk and hbond with site-aware tighter cutoffs.  Center-of-mass screening removes little when the skin is large; in the oligomer proxy at skin 1.0, 6.2 of 7.5 half neighbors per nucleotide survive. | M, C | M | M |
| B3 | Reorder the early exits to test cheap conditions first: the coaxstk radial window before the 4 acos-based angle terms (`pair_oxdna2_coaxstk_kokkos.cpp:1050-1066`); hbond complementarity and r^2 window before the site math and F1 (standalone gates on `int_type == 3` and the r^2 window). | M, C | M | S |
| B4 | Skip the tetramer (sequence-dependent) type loads when sequence averaging is used.  For stk this is the unmerged commit `4ac94dcaa7` (`oxdna-framework-overhead`).  excv runs the tetramer topology test (`tag(a) == id3p(b) ...`) for every pair every step, even for oxDNA1/2 ((kkf) `pair_oxdna_excv_kokkos.cpp:593,636`). | C | L-M | S |

### C. Data layout and per-atom precomputation

| # | suggestion | evidence | impact | effort |
|---|---|---|---|---|
| C1 | Pack per-atom data for the force kernels: a float4 (x, y, z, type), and a float4 quaternion (axes rebuilt in the kernel, as standalone does and `aa59e87056` tried) or the frames as one aligned AoS record.  Today a neighbor visit gathers about 12 separate scalars (x as float[3] LayoutRight; nx/ny/nz as 3 LayoutLeft `[3]` views), repeated in up to 5 kernels. | C, S | M-H | M |
| C2 | Cache the per-atom interaction-site positions (backbone, stacking, base) in the LRF fix, which already runs every step over nall, so the pair kernels stop re-deriving them. | C | M | S-M |
| C3 | Pack the coefficients needed per type pair into one struct, so a pair loads one record instead of about 20 separate views (as in the fusion branches). | C | M | M |
| C4 | Make the prime-neighbor pair table per atom: map(id3p) and map(id5p), 8 B per atom, instead of 4 map lookups and 16 B per neighbor slot (anum x maxneigh x 4).  excv reads at most the 2 bonded slots per atom. | M (1 launch per rebuild), C | L (rebuild only) | S |

### D. Per-step framework overhead (dominant at small N; 8k nt: 0.14 vs 0.055 ms)

| # | suggestion | evidence | impact | effort |
|---|---|---|---|---|
| D1 | Slim the functors: launch on a small struct of device views and scalars instead of `*this`.  Kernel parameters today are 14-16 KB for hbond, excv, coaxstk and stk, against about 100 B in standalone.  Measure `cudaLaunchKernel` time first. | S | M (small N) | M |
| D2 | Fewer launches per step (about 14 against 4): fold the f/torque zeroing into the first force kernel; fold the LRF frame update into the nve/asphere final integrate (ghost frames then need forward comm); consider fusing langevin into the integrator. | M (kernel counts) | M (small N) | M |
| D3 | Remove the per-step host read-back of the `check_distance` reduction, e.g. with a pinned-memory flag as standalone does; check for any other per-step host syncs in the Nsight Systems timeline. | C | L-M (small N) | S-M |
| D4 | Re-sweep the launch bounds and block size per GPU for every oxDNA kernel.  `LaunchBounds<64,1>` is only on excv and dh; the others use the Kokkos default. | C, S | L-M | S |
| D5 | Compare thermostats like for like: `fix langevin ... angmom` runs 2 RNG kernels every step, while the usual standalone Brownian thermostat fires every `newtonian_steps` steps (**CONFIRM** in the benchmark inputs). | C | benchmark fairness | S |
| D6 | (CPU/OpenMP only) the duplicated scatter views for f/torque are created every step in each sub-style; share one duplicated f/torque per step.  On GPUs the views are not duplicated, so there is no effect. | M (OpenMP) | 0 | S |

### E. Neighbor rebuild path (matters with a small skin or frequent rebuilds)

| # | suggestion | evidence | impact | effort |
|---|---|---|---|---|
| E1 | Build the bond prime-neighbor table once per rebuild, not twice.  In `oxdna3KK`, `FixOxdnaPrimeNeighsKokkos::pre_force` and `pair_oxdna_stk_kokkos.cpp:101-104` each build it, because they track rebuilds separately; in kk-fixes stk and fene each keep their own copy ((kkf) `bond_oxdna_fene_kokkos.cpp:143`).  Measured: 62 `PrecomputePrimeNeighsBond` launches for 31 builds. | M, C | L | S |
| E2 | Avoid the device-to-host copy of the bond list on every rebuild: `k_bondlist.sync_host()` in `neigh_bond_kokkos.cpp:249` (a no-op on CPU builds, a real copy on GPUs). | C | L-M | S |
| E3 | Avoid the host read-back (`deep_copy`) of the screened pair count in `fix_oxdna_npair_kokkos.cpp:220`, e.g. by sizing from a device-side upper bound. | C | L | S |
| E4 | Use a larger skin for dilute systems: in the proxy, the neighbor count stayed the same from skin 0.3 to 2.0, while rebuilds dropped from 29 to 1 per 2000 steps.  Cap the GPU bin count for dilute boxes: binsize = cutneighmax, so the bin count scales with box volume (`neighbor_kokkos.cpp:392-395`). | M | L-M | S |
| E5 | `pair oxdna3/stk/kk` requests a pair neighbor list that it never uses (it loops over the bond list); drop the request. | C | L | S |

### F. Developer productivity

| # | suggestion | evidence | impact | effort |
|---|---|---|---|---|
| F1 | Instantiate only reachable template combinations: excv takes about 80 s and hbond about 65 s to compile.  For example, drop the host-only instantiations from device builds, and the oxDNA-model tags a style can never use. | M | 0 (build time only) | S-M |

### Suggested order

1. Quick, low-risk items to establish the gains: B1, B3, A5 (stack only), E1,
   E2, D4, then re-baseline on kk-fixes.
2. The main structural work, chosen by the GPU profile (section 8): A3 + A4 +
   C1/C2 on the existing split kernels, or go straight to A2 or A1.
3. Small-N work: D1, D2, D3.
4. The rest as the profile justifies.

## 8. GPU session plan

1. Record the inputs (box, skin, salt, thermostat, `package kokkos` options) and
   rerun oligomer 65k/524k and polybrick 543k on the kk-fixes code.  That branch
   added atomics to the hbond/xstk/coaxstk GPU kernels, so re-baseline first.
2. `nsys profile --stats=true -t cuda,nvtx` on oligomer 65k, 1000 steps:
   * time per kernel, and the sum of the 7 force kernels vs `DNA3_forces`;
   * `cudaLaunchKernel` API time per launch (functor size, D1);
   * gaps between kernels and synchronizations per step;
   * the cost of a rebuild step.
3. `ncu --set full -k regex:"Xstk|Hbond|Coaxstk|Stk|FENE|Excv|Dh"`, a few
   launches each.  Check:
   * warp execution efficiency (`smsp__thread_inst_executed_per_inst_executed.ratio`), for divergence (A3);
   * L1/L2 atomic traffic (`lts__t_sectors_op_red.sum`, `lts__t_sectors_op_atom.sum`);
   * local memory traffic of stk/fene (A5);
   * L1 hit rate and sectors per request of the frame and x gathers (C1).
4. Work through the master list in section 7 in the suggested order, with the
   profile deciding between A1, A2 and A3/A4.

## 9. Reproducing the static analysis and the counts without a GPU

* CUDA toolchain without a GPU: download the NVIDIA redist archives (`cuda_nvcc`,
  `cuda_cudart`, `cuda_cccl`, `cuda_cuobjdump`, `cuda_nvdisasm`, `libcurand`,
  version 12.6) from `developer.download.nvidia.com/compute/cuda/redist/` and
  unpack them into one prefix.
* LAMMPS: `cmake -S cmake -B build-cuda -G Ninja
  -D CMAKE_CXX_COMPILER=$PWD/lib/kokkos/bin/nvcc_wrapper -D PKG_KOKKOS=on
  -D Kokkos_ENABLE_CUDA=on -D Kokkos_ARCH_ADA89=on -D KOKKOS_PREC=mixed
  -D PKG_CG-DNA=on -D PKG_MOLECULE=on -D PKG_ASPHERE=on -D BUILD_MPI=off`.
  Then build only the object files with `ninja <obj>`, and inspect them with
  `cuobjdump -res-usage` / `cuobjdump -sass`.
* Standalone: `nvcc -std=c++17 -arch=sm_89 -O3 -use_fast_math -include string
  -I src -I src/extern -Xptxas -v -c src/CUDA/Interactions/CUDADNA3Interaction.cu`.
* Early-exit counts: CPU build (`Kokkos_ENABLE_OPENMP=on`).  In the hbond and
  coaxstk `compute()`, force `use_host_launch = false` and the screened list.
  Put a `Kokkos::atomic_inc` on a global counter after every top-level early
  `return` in the pair operators, and print the counters in the destructor only
  when `copymode == 0`.
* Proxies: the `duplex2` input with `replicate 8 8 8`, `fix nve/asphere`,
  `fix langevin 0.1 0.1 2.5 457145 angmom 10`, `timestep 0.003`,
  `oxdna3/dh 0.1 0.5`, `-k on t 4 -sf kk -pk kokkos neigh half newton on`.
  Dense proxy: data file box `-1.31 1.49`, `-2.15 0.65`, `-0.20 3.52`.
