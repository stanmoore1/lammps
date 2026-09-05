# KOKKOS pre-release review: open issues for the LAMMPS developers

Companion to `REPORT.md` (what was reviewed and found) and `NON_KOKKOS_FINDINGS.md` (defects
outside `src/KOKKOS`).  This file lists only what is **not resolved** on branch
`claude/kokkos-code-review-6lcg8d`, plus the judgement calls and coverage gaps a reviewer
should know about before trusting the rest.

Nothing here is a fix waiting to be applied.  Each item is either a defect that needs a
decision, a change that needs a second opinion, or a hole in what was actually tested.

---

## 1. Defects that are NOT fixed

### 1.1 `fix property/atom` corrupts the heap under KOKKOS

**Severity: high.  Pre-existing -- upstream `develop` aborts identically.**

Reproducer (either form):

```
atom_style  atomic/kk
fix         p all property/atom d_foo        # run with: -k on t 1 -pk kokkos
```
```
suffix off
fix         p all property/atom d_foo        # run with: -k on t 1 -sf kk -pk kokkos
suffix on
```

Both abort immediately:

```
realloc(): invalid pointer
  Memory::srealloc
  FixPropertyAtom::grow_arrays
  Modify::add_fix
```

`AtomKokkos::add_custom()` grows `dvector` with `memoryKK->grow_kokkos()`, so
`atom->dvector[index]` points into a Kokkos allocation.  The base
`FixPropertyAtom::grow_arrays()` then calls `memory->srealloc()` on that pointer.  This is
normally hidden by the `/kk` suffix, because `FixPropertyAtomKokkos` overrides
`grow_arrays()`.  Reach the base class and it is heap corruption, not a wrong answer.

**Why it is not fixed here.**  `AtomKokkos::update_property_atom()` already carries a
`kokkosable` guard with a good message (this review improved the wording).  That guard cannot
catch this case: it runs from `FixPropertyAtomKokkos::post_constructor()`, and a plain
`fix property/atom` aborts inside its own constructor first.  An attempt to make the guard
reachable by calling `update_property_atom()` from `AtomKokkos::init()` was made and then
**reverted** -- `init()` also runs long after `Modify::add_fix()`, so it was dead code.

**Why it needs a maintainer.**  None of the obvious placements work:

* `AtomKokkos::add_custom()` cannot see which fix is calling it.  The fix is not yet in
  `modify->fix[]`, and `FixPropertyAtom`'s constructor runs before `FixPropertyAtomKokkos`
  sets `kokkosable`.
* A check in `Modify::add_fix()` would work but is core code, not KOKKOS.
* Alternatively `AtomKokkos::add_custom()` could hand back ordinary memory and adopt it into
  a Kokkos view later.

That is an architectural choice, so it was left to you.

### 1.2 `fix nve/sphere/kk` has no DLM integrator

**Severity: medium.  Behaviour changed by this branch.**

Upstream `fix_nve_sphere_kokkos.cpp` contains no reference to `dlm` at all.  It inherits the
keyword parsing from `FixNVESphere`, then ignores the flag and runs the plain orientation
update.  A user asking for `update dipole/dlm` silently got a different integrator.

`FixTimestep:nve_sphere_dipole_dlm` passed upstream only because the two schemes agree to
within the fixture tolerance over the few steps it runs.

This branch makes the style **refuse** the keyword, and marks the kokkos variants of
`unittest/force-styles/tests/fix-timestep-nve_sphere_dipole_dlm.yaml` as `skip_tests`.
Refusing is the safe behaviour, but the real answer is to port the DLM kernel.  That is a
genuine piece of work and was not attempted.

### 1.3 `math_special_kokkos.cpp` is never compiled by CMake

**Severity: low now, a trap later.**

`cmake/Modules/Packages/KOKKOS.cmake:178` builds `KOKKOS_PKG_SOURCES` from an **explicit
list**, and `math_special_kokkos.cpp` (479 lines) is not in it, nor added by any package
block.  The traditional make build copies and compiles every `src/KOKKOS/*.cpp`, so it *is*
compiled there.

`math_special_kokkos.h` declares two functions that only the .cpp defines:

```cpp
extern double factorial(const int n);
extern double erfcx_y100(const double y100);
```

Nothing calls them today, which is why the CMake build links.  The first KOKKOS style that
does will build under `make` and **fail to link under CMake** -- a confusing failure to
diagnose.  Either add the file to `KOKKOS_PKG_SOURCES` or remove it.

---

## 2. Judgement calls that deserve a second opinion

These were decided during the review.  Each could reasonably go the other way.

| # | Item | Decision taken | Why you might disagree |
|---|---|---|---|
| 2.1 | `pppm_kokkos.cpp` unguarded `destroy`/`create` of `k_eatom`/`k_vatom` each step (F0297) | **Not a defect.**  The pattern is the dominant idiom in the package (~187 files) and is load-bearing here: `ev_init(...,0)` means nothing else zeroes the arrays | It is still a per-step reallocation in a hot path |
| 2.2 | `fix_addforce_kokkos.cpp` "over-broad sync" | **Not a defect.**  The single `sync(execution_space, X\|F\|IMAGE\|MASK)` exactly matches what the kernels read; `x` and `image` are needed for the unwrapped virial | It differs from `fix_setforce_kokkos`, which looks like a divergence until you check why |
| 2.3 | `fix_qeq_reaxff_kokkos.cpp` copymode bracket | **Not a defect.**  No path sets `copymode=1` without clearing it, and the wide bracket is required because `allocate_array()` itself dispatches a `parallel_for` on `*this`.  Same shape in `fix_acks2_reaxff_kokkos.cpp` | The bracket is wide enough that a future early return inside it would leak |
| 2.4 | `improper_fourier_kokkos.cpp`, `angle_mm3_kokkos.cpp` | Fixed **per-timestep Kokkos reallocation**, not a correctness bug.  Physics, eflag/vflag and datamasks were checked and match the CPU base | These are the only performance-motivated changes in an otherwise correctness-only branch |
| 2.5 | `bond_hybrid_kokkos.cpp` `orig_map` write-back (F0040) | Fixed with a **host-side** write-back, because `bond_quartic_kokkos` writes its zeros into the host mirror and a device pass would force a round trip | Narrowly exercised: only `bond_style hybrid/kk quartic/kk` reaches it |
| 2.6 | Three grid computes: group filter | The `mask` view is bound through the existing explicit `sync(...)` rather than by adding `MASK_MASK` to `datamask_read` | Those classes set `datamask_read = EMPTY_MASK` and sync `X\|F\|TYPE` explicitly, so adding only `MASK_MASK` would be inconsistent -- but it also departs from the package idiom |

---

## 3. Changes that warrant extra scrutiny

### 3.1 Three defects fixed outside `src/KOKKOS`

Detailed in `NON_KOKKOS_FINDINGS.md`.  They touch `src/`, `src/EXTRA-PAIR/`, `src/ML-SNAP/`
and `src/OPENMP/`, and they change physics.  They are on this branch because the project's
own rule is to fix a style, its parent and all suffix variants together, but you may want
them split into their own pull request:

* `pair lj/expand/sphere` multiplied by `rshift` where the gradient requires a division --
  in the base, `/omp` **and** `/kk` variants, so no accelerator comparison could see it.
* `compute sna/grid` and `sna/grid/local` read `sinnerelem`/`dinnerelem` with a 0-based
  element index from arrays filled 1-based by type.
* `AtomVec::pack_comm_vel`/`pack_border_vel` tested `mask[i]`, the send-list position,
  instead of `mask[j]`, the atom.

### 3.2 The `lj/expand/sphere` reference fixture was regenerated

`unittest/force-styles/tests/atomic-pair-lj_expand_sphere.yaml` was generated from the
defective force expression, so it had to be regenerated.  **It therefore no longer
independently validates that style.**

The evidence that the change is self-consistent is that before regeneration the fixture failed
for exactly the three variants that were changed -- `plain`, `omp`, `kokkos_omp` -- and for no
others.  Independent confirmation is `tests/in.lj_expand_sphere_equiv`, which pits the style
against `lj/expand` on a configuration where the two are the same potential.

Note also that `PairStyle.single` is **skipped** for this style, so the matching fix in
`PairLJExpandSphere::single()` is not covered by the fixture at all.

### 3.3 A latent bug was exposed, not introduced, in meam

`PairMEAM::coeff()` writes only the upper triangle of `scale[][]`; the lower triangle is
uninitialised until `PairMEAM::init_one()` mirrors it.  `PairMEAMKokkos::coeff()` copied the
whole array to the device at coeff time, so the device copy held indeterminate values below
the diagonal.

That was harmless only because the kernel read the diagonal.  Correcting `meam_force` to read
`d_scale(type[i],type[j])` -- matching `src/MEAM/meam_force.cpp:92` -- started reading the
uninitialised half and **every meam force-style test failed**.  Fixed by seeding the device
copy with 1.0 and refreshing it from an `init_one()` override.

Worth a look because it is the one place in this branch where a correct change produced
grossly wrong forces until the underlying allocation bug was found.

### 3.4 The single-precision `EPSILON` fix was never run

`pair_born_coul_dsf_cs_kokkos.cpp` used the CPU's `EPSILON = 1.0e-20`.  In single precision
the Born potential's `r^-8` term gives `1e80`, which overflows to `inf`, and the zero
special-bond factor then produces NaN rather than removing the pair.  Verified directly in
float arithmetic:

```
float  EPS=1e-20: r6inv=inf   r8inv=inf   0*r8inv=-nan
double EPS=1e-20: r6inv=1e60  r8inv=1e80
float  EPS=1e-8 : r6inv=1e24  r8inv=1e32  0*r8inv=0
float 1.0f + 1e-8f == 1.0f ? yes      double 1.0 + 1e-20 == 1.0 ? yes
```

The value is now `1e-8` when `KK_FLOAT` is `float`, applied as a floor rather than an
unconditional add (equivalent in double, avoids a real perturbation in single).  **This was
reasoned and checked against float arithmetic, but never executed** -- see gap 4.2.

---

## 4. Coverage gaps: what was NOT tested

The validation build is one point in a large configuration space.  Everything below was
reviewed by reading only.

### 4.1 One backend

OpenMP.  `Kokkos_ENABLE_CUDA`, `HIP` and `SYCL` are all OFF, so **every `LMP_KOKKOS_GPU`
block and every device-specific kernel is uncompiled and unrun**.  The Serial backend's tests
also skip (`Cannot test KOKKOS/Serial with threading support enabled`).

This matters more than usual for this review, because `KokkosLMP::newton_check()` forces
newton pair off whenever `neighflag == FULL`, which makes most `FULL` paths unreachable on a
CPU build -- while **FULL is the default on GPU**.  A few were forced with
`-pk kokkos neigh full newton off`; this was not systematic.  See `reachability.md`.

### 4.2 One precision

`KOKKOS_PREC=double`.  Neither `single` nor `mixed` was ever built.  The precision sweep was
code-reading only, and the `EPSILON` fix of section 3.4 exists *specifically* for single precision.

### 4.3 One layout, one index size

`KOKKOS_LAYOUT=legacy` (LayoutRight); the `default` LayoutLeft path is untested.
`LAMMPS_SIZES=smallbig`; `bigbig` was not built, and the project's own notes identify
`bigbig` as the usual cause of a failure on a single CI job.

### 4.4 Package coverage -- initially incomplete, now closed

The first validation build had 36 packages enabled, which left **25 of 384 `src/KOKKOS`
sources uncompiled**, including five files carrying changes from this review:

| file | change |
|---|---|
| `dynamical_matrix_kokkos.cpp` | 41-line host-merge and HostKK rewrite |
| `third_order_kokkos.cpp` | 41-line host-merge and HostKK rewrite |
| `pair_lj_class2_kokkos.cpp` | `eflag` to `eflag_global` |
| `pair_lj_spica_kokkos.cpp` | `eflag` to `eflag_global` |
| `pair_coul_shield_kokkos.cpp` | `eflag` to `eflag_global` |

CLASS2, CG-SPICA, PHONON, INTERLAYER, SPIN, BROWNIAN and MOFFF were then enabled and all five
compile cleanly.  Only `fix_colvars_kokkos.cpp` (needs the external COLVARS library) and
`math_special_kokkos.cpp` (section 1.3) remain uncompiled.

**Anyone repeating this validation should check compile coverage explicitly rather than
assume a green build covers the branch.**

### 4.5 Two defect classes this method structurally cannot find

* **A KOKKOS style that faithfully copies a defective CPU parent.**  kk and cpu agree in both
  builds, so no accelerator-vs-reference comparison can see it.  `lj/expand/sphere` was found
  by deriving the gradient by hand, not by the sweep.
* **An upstream fix never mirrored into KOKKOS.**  Commit `47cea8e1ba` added the `cutsq_trim`
  fallback to four `src/npair*.cpp` files and touched no KOKKOS counterpart; both sides read
  as internally coherent and only the history reveals the gap.

One instance of each was found -- by derivation and by accident respectively, not by
systematic search.  A deliberate sweep for both classes would likely find more.  For the
second, a periodic `git log` diff of `src/npair*.cpp` against `src/KOKKOS/npair*_kokkos.cpp`
(and the same for other mirrored families) would catch it mechanically.

### 4.6 Other gaps

* **Unit tests only** -- `tools/regression-tests` was not run.
* **`fix adapt` never exercised at runtime.**  The `scale[][]` work in meam and eam added
  `reinit()` overrides specifically for the fix adapt path; correctness there rests on code
  reading, not on a test.
* **Mostly single-rank.**  The targeted regression inputs in `tests/` run on one MPI rank;
  multi-rank exchange, border and reverse-comm paths are covered only by whatever the ctest
  MPI variants reach.

---

## 5. One regression on `develop` itself

Fixed on this branch, but flagged separately because it is **recent breakage rather than a
long-standing defect**, and is worth an upstream report of its own.

`AtomKokkos::map_clear()` gained a device branch without a matching `clear_sync_state()`.  A
preceding `map_one()` leaves the host side dirty, and the `modify_device()` then trips
`dual_hash_type`'s concurrent-modification abort:

```
Kokkos::abort: Concurrent modification of host and device hashes
```

`Special::dedup()` walks exactly that sequence, so **any molecular system with special bonds
aborts when the atom map lives on the device** -- which is the default on a GPU build.
Reproduced on a CPU build with `-pk kokkos atom/map device`; input in
`tests/in.map_hash_dedup`.

---

## 6. Environmental test failures (not defects)

The full `ctest` suite leaves three failures that are properties of this container, not of the
code.  Listed so nobody re-investigates them:

| test | cause |
|---|---|
| `Platform` | `chmod 0` then asserts the file is unreadable; root reads it anyway |
| `TextFileReader` | same -- expects a no-permissions file to throw |
| `PythonFormats` | `NameError: name 'Loader' is not defined` -- PyYAML C-loader not installed |

Also note two pieces of tooling that report success while doing nothing, which cost time here:
`make html` fails in setup if **doxygen** is absent, and `make spelling` **exits 0** while
failing to load its extension if the **enchant** C library is absent.
