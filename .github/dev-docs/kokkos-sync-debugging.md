# Debugging KOKKOS host/device sync bugs

A guide for finding a missing `sync()` / `modify()` in the KOKKOS package by
running an input deck under the split-memory debug build.  The user-facing
reference for the build options and every environment variable is
`doc/src/Build_extras.rst` (search for `KOKKOS_DEBUG_SYNC`); this file is the
working procedure that goes with it.

## What the tool can find, and what it cannot

Kokkos turns its coherence state machine off whenever the host and device
memory spaces are the same, which is every CPU-only build: `sync()` and
`modify()` return immediately and both sides are one allocation.  A missing
declaration therefore has no effect on a CPU and silently corrupts results on
a GPU.  `-D KOKKOS_DEBUG_SYNC=on` gives the host side its own allocation and
drives the state machine in software, so the GPU bug reproduces on a CPU.

Two failure classes exist, and they need different detectors:

1. **Stale access** -- a read or write of the side that is not current.  Caught
   by **poison mode** at the exact faulting instruction, whatever path the
   access took: a Kokkos view, a cached view, a subview, a plain `double **`
   from `memory_kokkos.h`, or a `memcpy` inside MPI.
2. **Unclaimed write** -- a write to one side of an in-sync pair that is never
   followed by the matching `modify_*()`, so the pair silently diverges with
   clean counters.  Poison cannot catch this (an in-sync pair must stay
   writable); the **watch** and **stale** checks do it by comparing state.

Neither detector can see a bug that the input never exercises.  A run that
behaves correctly proves nothing about the code it did not execute.

## Two builds

Configure both once; keep them side by side.  Add whatever packages your input
needs.

```bash
# detector build -- watch / stale / trace, fast enough for repeated runs
cmake -S cmake -B build-sync -G Ninja \
      -D PKG_KOKKOS=on -D Kokkos_ENABLE_SERIAL=on -D BUILD_MPI=on \
      -D CMAKE_BUILD_TYPE=Release -D KOKKOS_DEBUG_SYNC=on \
      -D PKG_MOLECULE=on -D PKG_KSPACE=on -D PKG_RIGID=on
cmake --build build-sync -j 4

# poison build -- adds AddressSanitizer; slower, needed only for class 1
cmake -S cmake -B build-poison -G Ninja \
      -D PKG_KOKKOS=on -D Kokkos_ENABLE_SERIAL=on -D BUILD_MPI=on \
      -D CMAKE_BUILD_TYPE=RelWithDebInfo \
      -D KOKKOS_DEBUG_SYNC=on -D KOKKOS_DEBUG_SYNC_ASAN=on \
      -D PKG_MOLECULE=on -D PKG_KSPACE=on -D PKG_RIGID=on
cmake --build build-poison -j 4
```

Every run below must use the KOKKOS styles, or the debug build tests nothing:

```bash
./build-sync/lmp -in in.your_input -k on -sf kk       # add -pk kokkos ... as needed
```

Runs use about twice the per-atom memory and are considerably slower.

## Procedure

### 0. Establish that there is a fault, and where it shows

Run the input on a stock build and on `build-sync`, and compare thermo output.
A difference is the bug reproducing.  If the two agree, vary the conditions the
coherence paths depend on before concluding there is nothing to find:

```bash
-pk kokkos comm device / comm host      # which side the communication runs on
-pk kokkos sort device / sort no        # atom sorting
-np 1 / -np 4                           # exchange and border paths differ
```

Keep the exact command line that shows the difference; every later step reuses
it.

### 1. Poison mode first

It names the root cause rather than a symptom, and it is the only detector that
sees reads through plain pointers.

```bash
LMP_KOKKOS_POISON=1 ASAN_OPTIONS=detect_leaks=0 \
  ./build-poison/lmp -in in.your_input -k on -sf kk
```

The run stops at the first stale access with an ASan report whose top frames
name the function that used the data.  That function is where the missing
`sync()` belongs -- or, read the other way, the array it touched is the one
whose `modify_*()` is missing upstream.

To collect every fault in one run instead of stopping at the first:

```bash
ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:log_path=/tmp/poison/a
```

`detect_leaks=0` silences MPI's own leaks.  With more than one rank the
`log_path` is essential: four ranks writing one stream interleave into reports
that belong to no single process.

### 2. Watch and stale for the unclaimed-write class

If poison is silent but results still differ, the write was never claimed:

```bash
LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1 \
  ./build-sync/lmp -in in.your_input -k on -sf kk
```

An empty value means "every view"; give a substring instead to follow one array
(`LMP_KOKKOS_WATCH=atom:special`).  Add `LMP_KOKKOS_WATCH_BT=1` for a backtrace
at each report once you know which array to chase -- it is verbose.

### 3. Always diff against a clean run

This is the step that decides whether a report matters.  Some reports appear on
correct runs too: scratch buffers filled and thrown away on purpose, and pairs
left deliberately apart by `clear_sync_state()`.  Run the **unmodified** code
with the same flags, keep its reports, and compare:

```bash
labels() { sed -n 's/^\[stale\] \+\([a-zA-Z_:]*\): .*, from \(.*\)$/\1 <- \2/p' "$1" | sort -u; }
comm -13 <(labels clean.err) <(labels suspect.err)     # only what the fault added
```

Key the comparison on the array **and the routine that touched it**, not the
array alone: the noise and a real finding often share an array name, and a
bare-name diff throws the finding away with the noise.  Watch reports name an
element index as well; include it for the same reason.

## Reading the reports

```
[stale] atom:bond_atom: device side read while host side is newer,
        from LAMMPS_NS::NeighBondKokkos<Kokkos::Serial>::bond_all()
```
A missing **copy**: `bond_all()` needs `atomKK->sync(execution_space, BOND_MASK)`,
or the caller that wrote the host side owes a `sync` before this point.

```
[watch] AngleHarmonic::k: the host side was written without a claim and this
        sync_device has nothing to copy -- the device keeps stale data
        element 1 of 2 is where they part
```
A missing **claim**: something filled the host side and never called
`modify_host()`, so this sync copies nothing.  Look at whoever last wrote that
array -- typically a style's `coeff()`, or legacy code writing through a plain
LAMMPS pointer.  This is the report to expect when a kernel reads a device view
that was cached in `allocate()` long before the write, because no accessor runs
at the time of the stale read.

```
[watch] atom:v: the host side was written, never claimed, and is now lost
        the write is between modify_device and sync_host, which discards it
```
The same fault caught at the moment the data is thrown away, with the two calls
it happened between.

## Fixing what you find

Three legitimate remedies, in order of how often they are right:

1. **Add the missing call** -- `k_foo.modify_host()` after a host write, or a
   `sync` before a read.  For per-atom arrays prefer the mask form,
   `atomKK->modified(Host, V_MASK)` / `atomKK->sync(execution_space, V_MASK)`.
2. **Declare it in the style's datamask** -- if a style writes an array it does
   not list in `datamask_modify`, the copies are steered wrongly no matter how
   many calls you add.
3. **`clear_sync_state()`** -- when the code deliberately overwrites a whole
   array and the other side's contents are irrelevant.  The package already has
   many such annotations; follow them rather than inventing a fourth remedy.

Remember that a bug in one style is rarely alone: check the base style and every
suffix variant (`/omp`, `/gpu`, `/kk`) for the same shape, per the repository's
contributing rules.

## Testing a change to the tool itself

`dual_view_kokkos.h` can be exercised without building LAMMPS: compile a small
program that includes it against the Kokkos libraries from any configured build
tree, providing a stub for `LAMMPS_NS::datamask_audit_note_copy`.  A test that
constructs a view, fills the host side, omits the claim and syncs takes seconds
to run and tells you at once whether a detector change still fires.  Do this
before rebuilding LAMMPS, which takes far longer.

## Pitfalls

* **`-k on -sf kk` is mandatory.**  Without it the run uses the plain styles and
  every detector is correctly silent.  This has produced false conclusions more
  than once.
* **Multi-rank output interleaves.**  Capture per rank
  (`mpirun --output-filename DIR ...`) and concatenate afterwards, and use
  `ASAN_OPTIONS=log_path=...` for the poison build.
* **A silent run is not a clean run.**  Check the run actually finished
  (`Total wall time` in the log) before reading anything into an empty report.
* **Compare like with like.**  Clean and suspect runs must use the same binary
  options, the same environment variables and the same rank count.
* **Poison mode and the paranoid/verify modes do not mix**; poison short
  circuits the accessor checks by design.
* **A run whose results did not change proves nothing** about a detector's
  coverage.  Establish the fault first (step 0), then ask what the tools say.
