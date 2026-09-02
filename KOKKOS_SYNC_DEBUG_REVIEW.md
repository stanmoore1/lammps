# Review of the KOKKOS split-memory sync debugging (branch `claude/lammps-kokkos-dualview-debug-t9be12`)

What the tool does, what it caught and missed when run against the bugs of the
`kk_bugfixes` and `claude/kk-bug-sweep` branches, where its time goes, and
which other GPU-only bug classes could be brought to a CPU the same way.
Section 6 has the measurements.

## 1. Mechanism

`src/KOKKOS/dual_view_kokkos.h` replaces `Kokkos::DualView` with a subclass
(`LAMMPS_NS::DualView`) that, when the host and device memory spaces are the
same (`SPLIT`, line ~190), gives the host side its own allocation `h_split`
and drives the coherence state machine itself: modification counters per side
in a `Kokkos::View<unsigned int[8]>` so that functor copies share them
(`lmp_flags`), an "authoritative side" marker that survives until a copy
really brings the two together, shadows of both sides as of the previous
coherence call (`shadow_h/shadow_d`, watch mode) and as of the last agreement
(`agreed_h/agreed_d`, stale mode).  Production builds get a plain type alias,
so no generated code changes.

Detectors, all runtime-selected through environment variables on one binary:

| mode | catches | how |
|---|---|---|
| `LMP_KOKKOS_STALE` (+`_STRICT`) | a read of the side that is not current | check at the accessor (`view<Device>()`, `view_host()`) against the counters |
| `LMP_KOKKOS_WATCH[=substr]` | a write to one side never followed by its claim | `memcmp` of each side against its shadow at every coherence call; a change with no counter increment is an unclaimed write |
| `LMP_KOKKOS_POISON` (ASan build) | any access to the stale side through any path, plain pointers and MPI included | `ASAN_POISON_MEMORY_REGION` on the non-authoritative side, unpoisoned inside `PoisonScope` accessors |
| `LMP_KOKKOS_AUDIT` | a style writing a per-atom array it did not declare in `datamask_modify` | `datamask_audit_kokkos.cpp` snapshots every per-atom array around every fix/compute call and compares bytes |
| `LMP_KOKKOS_VERIFY`, `_PARANOID` | sides that differ while the counters say they agree | byte compare on demand |
| `LMP_KOKKOS_COPYSTATS` | syncs and claims that never copy (inert calls) | a census of copies per array and direction |
| `LMP_KOKKOS_ALIAS` | control: no split at all | |

Enabling: `-D KOKKOS_DEBUG_SYNC=on` (defines `LMP_KOKKOS_DEBUG_SYNC`, links
`-rdynamic` for backtraces), `-D KOKKOS_DEBUG_SYNC_ASAN=on` for poison mode,
`-D KOKKOS_DEBUG_SYNC_SPLIT_HOST=on` to give host and device different Kokkos
backends (Serial = host, OpenMP = device) so that the `/kk/host` and
`/kk/device` instantiations of every style are distinct and the transfers
between them are real.

Two further changes make the emulation reach the code that matters.  In a
debug build without a device backend every style is routed to the `Device`
execution space (`kokkos_type.h` ~275), otherwise everything would sit on
`HostKK` and the host/device edge would never be crossed.  Styles that read
their team and tile sizes off `ExecutionSpaceFromDevice` therefore asked a
Serial backend for GPU-sized teams; the branch adds `HostBackendFromDevice`
(memory-space based) and switches 20 sites to it.  The uniform two-line edit
in ~300 files is `#ifdef LMP_KOKKOS_GPU` becoming
`#if defined(LMP_KOKKOS_GPU) || defined(LMP_KOKKOS_SPLIT_HOST)` around the
`template class X<LMPHostType>` instantiations, so that the split-host build
has both instantiations to work with.

The documentation (`.github/dev-docs/kokkos-sync-debugging.md`,
`doc/src/Build_extras.rst`) is thorough, and its central warning is the
right one: the `package kokkos` defaults of a build without a GPU route the
comm, sort, atom map and neighbor build around the device paths, so the GPU
settings have to be asked for explicitly on every run.  The branch also
carries a mutation campaign (`.github/dev-docs/kokkos-sync-sweep/`) that
comments out one sync/modify at a time and asks the detectors whether they
notice; its verdict files are a useful map of which sites are inert under
which inputs.

## 2. What it catches and what it misses, against the real bug list

Mapped onto the fixes of `kk_bugfixes` and the sweep:

| bug shape | example | caught on a CPU by | notes |
|---|---|---|---|
| host write, claim missing | temp/sphere/kk bias (sweep 1.4), `fix shake` xshake (5405f6f35), `set` command writes (a87bcc04b) | watch | confirmed in section 6: the report names the array and the routine |
| claim placed before the kernel that writes | min fire x/v (2460560dd), dpd/fdt/energy (b65be4720) | watch | the intervening sync takes the claim; the shadows see the later change with no counter increment |
| device read of a stale side | host mask read in fix nh remap (12d299199) | stale, poison | poison gives the faulting line |
| per-type/global dual views written on the host | `k_mass` (38f9554b2, sweep 1.1), coefficient tables after `coeff()` | watch, stale | **masked whenever another style syncs the same view first**: in a run with a `temp` compute in thermo, `temp/kk` syncs `k_mass` before `min quickmin/kk` reads it, and the tool is silent (section 6).  The bug is real but latent; see extension 5.3 |
| undeclared write (datamask too small) | `fix property/atom` (00d4c8c59) | audit | |
| `ALL_MASK` datamasks on a style that owns nothing | compute reaxff/atom/kk (sweep 1.3) | audit reports it explicitly | |
| host buffers not cleared / merge path (aab4d7a30, verlet `zero_host`) | | SPLIT_HOST only | needs host and device styles to coexist; the plain debug build has no `execute_on_host` path |
| comm buffer copied before `MPI_Waitall` (36aa1bebf, sweep 1.11) | | poison with `-np >= 3` | the stale side of a receive buffer is poisoned; needs real MPI, not STUBS |
| missing fence before MPI on a device buffer (sweep 1.12) | | nothing | on a CPU every kernel is synchronous |
| kernel launched in the default execution space (nbin, mliap; sweep 1.5) | | nothing | both spaces are HostSpace even under SPLIT_HOST; no fault, wrong results only on a GPU |
| host object dereferenced in a kernel (pair dpd/kk `update->ntimestep`, sweep 1.17), extended lambda in a private member (min sd/kk) | | nothing | only a device compiler sees these |
| `double` vs `KK_FLOAT` byte counts (pace, pod, eam; 0ed8a1ba7) | | nothing (double build) | a single-precision debug build would trap the over-read under ASan |
| per-atom eatom/vatom leaks and the `alloc = 1` invalid free (sweep 1.2, 1.19, 2.1) | | ASan with `detect_leaks=1` | the docs turn leak detection off because of MPI noise; under STUBS it can stay on |
| statistics/parity differences (shake counts), wrong virial (cmap) | | nothing | not coherence bugs; a plain-vs-kk comparison is the detector |

## 3. Cost and where it goes

Measured in section 6 on one binary (`LMP_KOKKOS_ALIAS=1` is the no-split
control, so the difference is the tool's own cost, not the compiler's).

Per coherence call the wrapper does, in order: the `watched()` /
`paranoid()` / `verify_filter()` tests, each of which calls
`base_type::view_device().label()` -- a `std::string` constructed by value
from the Kokkos label -- and then `find()`; for a watched view two
`deep_copy` refreshes of the shadows plus a `memcmp` of each side against its
shadow (`first_difference`, then a per-element scan only when they differ);
for the empty-sync report a `std::map<std::string,int>` lookup keyed by label
(`empty_sync_seen`).  Poison mode adds an `ASAN_(UN)POISON_MEMORY_REGION` over
the whole buffer at every accessor scope.  The audit snapshots every per-atom
array around every style call.

Concrete speedups, in order of payoff for the effort:

1. **Decide the filters once per view.**  Cache the outcome of `watched()`,
   `paranoid()` and the verify filter in the wrapper at `split()` time (the
   label never changes); today every sync, modify and accessor of an
   *unwatched* view still pays a string construction and a substring search.
   This is the whole overhead of the "detector build" for views nobody
   watches, and it is pure waste.
2. **Report accessor staleness at the launch, not at the accessor.**  The
   biggest noise class in every run (section 6) is a `view<DeviceType>()`
   taken a few lines *before* the sync that makes it current
   (`npair_kokkos.cpp:173` vs `:209`, `fix_shake_kokkos.cpp:858-871`,
   `atom_vec_*_kokkos.cpp grow_pointers()`).  Record the accessor as
   *pending* and settle it in a `Kokkos::Tools` `begin_parallel_for/reduce`
   callback: still stale at the launch, report; synced in between, drop.  The
   same callback removes a second noise class, the reads inside
   `grow()`/`resize()`.  Fewer false reports is the largest practical speedup,
   since every report costs a person's time.
3. **Compare hashes, not bytes.**  A 64-bit hash of each side stored in the
   flags view replaces `shadow_h/shadow_d` (two full extra copies per watched
   view) and turns the `memcmp` into an integer compare; the per-element scan
   runs only for a report.  Halves the memory of watch mode and removes the
   two `deep_copy` per call.
4. **Sample the audit.**  `LMP_KOKKOS_AUDIT` copies every per-atom array
   around every style call.  An `LMP_KOKKOS_AUDIT_EVERY=n` (audit one call in
   n) and a style-name filter keep the coverage while dividing the cost; the
   audit is a bytes-vs-declaration check, so sampling loses nothing but
   which step catches it.
5. **Replace `std::map<std::string>` in `empty_sync_seen` and the copy census**
   with an integer id assigned once per view; the label lookups run on every
   empty sync.

None of these change what is detected.

## 4. Correctness concerns

- **Accessor-time reporting.**  Stale mode fires at `view<DeviceType>()`,
  which in most KOKKOS styles precedes the `sync` in the same function.
  Every run in section 6 carries 5-140 such reports on correct code; the
  documentation's answer (diff against a clean run) works but makes the tool
  unusable without a baseline of the very code being tested.  Item 2 above
  is the fix.
- **`atom:map_array` under `atom/map device`.**  Every run reports the host
  side written without a claim and read stale in `map_set_device()` /
  `map_clear()`.  Either the atom map's host writes lack a claim (a real bug)
  or these are whole-array rewrites that want a `clear_sync_state()`; the
  branch's own `labels_all.txt` lists the label without a verdict.  Worth
  settling, since it appears in every input and hides other reports.
- **Masking by unrelated syncs** (the `k_mass` case).  A missing sync is
  invisible whenever another style happens to sync the same view earlier in
  the step.  This is a property of any state-based detector, not a bug in
  this one, but the documentation should say so and extension 5.3 would
  expose it.
- **`SPLIT` gate on the element type** (`std::is_trivially_copyable` and
  `_destructible`) is well reasoned and correctly leaves the two
  DualView-of-DualView custom arrays alone.
- **Thread safety.**  The counters live in a `Kokkos::View` and are updated
  without atomics; under the Serial backend that is fine, under SPLIT_HOST
  (OpenMP as device) two host threads calling `modify_host()` on one view
  would race.  Harmless for counters that only grow, but the shadow
  `deep_copy` in watch mode is not re-entrant.  Document that watch mode is
  single-threaded.
- **The per-style edit** (`#if defined(LMP_KOKKOS_GPU) ||
  defined(LMP_KOKKOS_SPLIT_HOST)`) is correct but touches ~300 files for one
  token.  A single `LMP_KOKKOS_HOST_INSTANTIATION` macro defined in
  `kokkos_type.h` from the two would make future styles (and future
  conditions) a one-line matter; the porting guide would say "wrap the host
  instantiation in `#ifdef LMP_KOKKOS_HOST_INSTANTIATION`".
- The 20 `HostBackendFromDevice` conversions are genuine fixes for any build
  where execution space and hardware disagree and should be taken regardless
  of the tooling (this review's scratch branch needed one more, in
  `pair_pace_kokkos.h`, where the merge with develop had reintroduced the
  `ExecutionSpaceFromDevice` form).

## 5. Other GPU-only bug classes testable on a CPU (ranked by value for effort)

5.1 **A device compiler pass without a GPU** (high value, low effort, no
    changes to the tool).  `nvcc` and `hipcc` compile device code without a
    device present; a CI job on a CPU runner with the CUDA toolkit installed
    that runs the KOKKOS package through `nvcc -c` (or clang `-x cuda
    --cuda-device-only -fsyntax-only` with the toolkit headers) catches every
    "only a device compiler sees it" class at once: host objects dereferenced
    in kernels (sweep 1.17), extended lambdas in private members (min sd/kk),
    host-only calls in `KOKKOS_INLINE_FUNCTION` bodies, and -- built in
    single precision -- the `double`/`KK_FLOAT` mismatches (0ed8a1ba7).  This
    would have caught three of the bugs on the two branches before they
    reached a GPU machine.

5.2 **Static check for default-execution-space launches** (high value, an
    afternoon).  `Kokkos::parallel_{for,reduce,scan}(<integer>, LAMMPS_LAMBDA
    ...)` in a class also instantiated for `LMPHostType` runs the kernel on
    the device over host memory (nbin fix in fa275ab5f, mliap in sweep 1.5).
    No runtime emulation can see it on a CPU.  A `make check-kokkos-launch`
    target (regex or `clang-query`: bare-count launch with a lambda, or with
    a functor lacking a `device_type` typedef, in a file that contains
    `template class .*<LMPHostType>`) would.  The sweep found exactly two
    hits in the package, so the check starts clean.

5.3 **Attribute syncs to their caller** (medium value, small change).  In
    watch/copystats mode record, per view, which routine performed the last
    host-to-device copy (one `backtrace` frame at sync time).  At a device
    accessor from a *different* class, report "relies on a sync by X".  That
    exposes the latent class of section 2 (`k_mass` synced by `temp/kk`,
    consumed by `quickmin/kk`; the same shape as sweep 1.7) that today only
    shows when the input happens to lack the helping style.

5.4 **Run the existing force-style and fix-timestep tests under the tool**
    (medium value, mostly wiring).  The unit tests already run every style
    with `-k on -sf kk` in-process; adding a CTest configuration that builds
    with `KOKKOS_DEBUG_SYNC=on` and fails a test on any `[watch]` report
    (after the noise of item 2 is gone) turns the tool from a session
    instrument into a regression gate.  The harness written for section 6
    (`syncrun.sh`) is that wiring in shell form.

5.5 **Leak and invalid-free detection under STUBS** (medium value, no code).
    A poison build with `ASAN_OPTIONS=detect_leaks=1` and the MPI stubs (no
    MPI noise) catches the eatom/vatom leaks and the `alloc = 1` regrow free
    (sweep 1.2, 1.19, 2.1, 2.2) with a run that grows `nlocal` mid-run
    (`create_atoms` after a first `run`).

5.6 **SPLIT_HOST runs with mixed host and device styles** (medium value, no
    code).  A run with a `/kk/host` pair or bonded style next to `/kk/device`
    ones makes `execute_on_host` true and exercises the host force buffer,
    the merge in `VerletKokkos::run()` and `force_clear()`'s host clearing
    (aab4d7a30 and the `zero_host` gating on `kk_bugfixes`) on a CPU.  Worth
    one case in the sweep's `cases.txt`.

5.7 **Multi-rank poison runs** (medium value, needs real MPI in the debug
    build).  The receive-before-wait class (grid3d 36aa1bebf, remap sweep
    1.11) only shows with three or more ranks and point-to-point comm; the
    poison side of the receive buffer would trap the early copy.  The
    fence-before-MPI class (sweep 1.12) stays out of reach on a CPU.

5.8 **Precision split** (low effort).  Building the debug binary with
    `KOKKOS_PREC=single` makes every TransformView's legacy side a separate
    `double` array even on a host backend, so the "legacy side vs KK side"
    sync calls (`sync_host()` vs `sync<LMPHostType>()`, sweep 1.22) become
    testable, and byte-count mistakes over-read under ASan.

## 6. Measurements from this session

Setup: the tooling merged onto the sweep fixes (scratch branch
`claude/kk-sync-debug-run`), gcc, Serial backend, `-D KOKKOS_DEBUG_SYNC=on`,
Release, MPI stubs, the GPU package settings from the documentation.  The
scripts and inputs are in `.github/dev-docs/kokkos-sync-harness/`.

### 6.1 Overhead (6912-atom LJ, 200 steps, one thread, best of two)

| binary and mode | loop time | relative |
|---|---|---|
| ordinary build (gcc, OpenMP backend, 1 thread) | 1.25 s | 1.0 |
| debug build, `LMP_KOKKOS_ALIAS=1` (no split) | 1.17 s | 0.93 |
| debug build, split, no detector | 1.28 s | 1.02 |
| split + `LMP_KOKKOS_STALE= _STRICT=1` | 1.20 s | 0.96 |
| split + `LMP_KOKKOS_WATCH=` (every view) | 1.71 s | 1.36 |
| split + `LMP_KOKKOS_WATCH=atom:x` | 1.34 s | 1.07 |
| split + watch every view + strict stale | 4.31 s | 3.4 |
| split + `LMP_KOKKOS_AUDIT=1` | 1.08 s | 0.86 |

The split and the stale check are free on this input; the copies they add
are the copies a GPU would make.  Watching every view costs the shadow
copies and compares (item 3.3 would remove most of it).  The combination of
watch and strict stale is the expensive one, and it is the combination the
documentation recommends as the default detector run; the audit is cheap here
because only one fix runs per step.  Bigger inputs scale these linearly.

### 6.2 Confirmations of the sweep's findings (fixed code vs one fix reverted)

| entry | tool verdict |
|---|---|
| 1.4 temp/sphere/kk bias claim | **confirmed**: `[watch] atom:v: the host side was written without a claim ...`, stale reads in `compute_scalar()` and `FixNVESphereKokkos::initial_integrate()`, biased temperature 0.837 instead of 0.738 |
| 1.1 quickmin/kk masses | **confirmed** once nothing else syncs the masses: `[stale] atom::mass: device side read while host side is newer, from MinQuickMinKokkos::iterate(int)`, minimization stops after 2 iterations with a zero force norm; silent and identical results while thermo's `temp/kk` syncs `k_mass` first |
| 1.14 shake/kk pack claim | not exercised: the host comm path needs `correct_coordinates()`, which `examples/peptide` never takes |
| 1.3 compute reaxff/atom/kk datamasks | not visible: the compute runs inside `fix ave/atom`'s audit window, and that plain fix is itself reported as "declares every array"; watch/stale see no difference and the results are identical under this input |

### 6.3 Noise on correct code

Every input produces reports on the fixed code (5 to 140 per run), all of
three shapes, none of them a bug:

- **accessor before sync**: `view<DeviceType>()` taken before the `sync` of
  the same function (`neigh:cutneighsq` in `NPairKokkos::build()`,
  `atom:x/v/f` in `FixShakeKokkos::post_force()` and
  `unconstrained_update()`, `atom:tag/type/mask` in the neighbor build of a
  minimization);
- **resize**: both sides read inside `AtomVec*Kokkos::grow()` and
  `grow_pointers()`;
- **atom map**: `atom:map_array` host side written without a claim and read
  stale in `map_clear()`/`map_set_device()` on every molecular input.  This
  one is real code, not tool noise: `AtomKokkos::map_one()`
  (`atom_map_kokkos.cpp:389-390`) syncs the host side and writes it with no
  `modify_host()`; it has no consequence because the device map is rebuilt
  from scratch before it is read, but the claim is missing, and
  `map_clear()`'s whole-array reset wants a `clear_sync_state()` so that the
  stale read of what it is about to overwrite is not reported.  Listed as
  sweep entry 1.23.

The 638 force-style and fix-timestep unit tests were also run under
`WATCH= STALE= STALE_STRICT=1` (in-process `-k on t 1 -sf kk`, host package
defaults since the test harness fixes the arguments): 638 pass, 223 produce
reports, all of the three shapes above plus one dead-view claim in
`pair lj/gromacs/kk` (sweep 2.6).  Further accessor-before-sync sites seen
there: `GroupKokkos::xcm_kk()` (`group_kokkos.h:90-102`),
`FixRigidSmallKokkos::set_xv_kokkos()` (`fix_rigid_small_kokkos.cpp:1290` vs
`:1320`), `FixNVELimitKokkos::final_integrate()`, `VerletKokkos::force_clear()`
and `Special::combine()` around `grow_kokkos`.  Item 3.2 (settle the report
at the kernel launch) would remove all of them at once.

Bottom line: on this code base the tool finds the "claim missing" and
"stale read" classes reliably and names the routine (two sweep findings
confirmed, one demonstrated to be masked by an unrelated sync); it cannot
see the execution-space, device-compiler, fence, precision and parity
classes, which together were the majority of the bugs on the two branches.
The device-compiler CI job of 5.1 and the launch check of 5.2 are the
cheapest way to cover most of that remainder.
