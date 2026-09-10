# KOKKOS host backend test sweep -- results and open items

## Test results (branch rebased onto bugfixes)

| suite | mixed | single |
|---|---|---|
| build | pass | pass |
| CI selection, 708 tests | 0 failures | 0 failures |
| the twelve KOKKOS cases | pass | pass |
| the same twelve without an accelerator | pass | pass |
| `ctest -T memcheck` | see below | not run |

## Fixed here

* `neighbor.cpp` -- `init_pair()` set `nlist` and allocated `lists` and
  `neigh_pair` without value initialization, so an error thrown between the
  allocation and the loop that fills them left the destructor deleting
  indeterminate pointers.  Turned a reported error into a segmentation fault.
* the command and format tests -- assumed double precision reference values;
  now precision aware through `kokkos_precision()` in the shared test core.
* `timer.cpp` -- `get_timeout_remain()` read `timeout_start` before
  `init_timeout()` assigned it.  Found by `ctest -T memcheck`, which reported
  two uninitialized-value defects on every run of the thermo tests; zero after.

The remaining memcheck defects are leaks, 143 with the KOKKOS styles and 149
without them, so they predate this work and are not KOKKOS specific.

## Open: pair ldd crashes with the atom map on the device

Reproduced on a CPU with the split memory debug build of branch
`claude/lammps-kokkos-dualview-debug-t9be12` (`-D KOKKOS_DEBUG_SYNC=on`,
Serial backend, double precision, one rank):

    ./test_pair_ldd            # with LAMMPS_ACCELERATOR_ARGS below

| package kokkos setting | result |
|---|---|
| `-k on -sf kk` (host defaults) | 2 tests pass, then a Kokkos teardown crash at exit |
| `-pk kokkos atom/map device`   | segmentation fault during the test |

    #0  AtomKokkos::map_set_device()
    #1  CreateAtoms::command(int, char**)
    #4  PairLDDTest_local_density_values_Test::TestBody()

`atom/map` defaults to `no` without a GPU and to `device` with one, which is
why this is invisible in a host run and always present on a GPU.
`map_set_device()` is identical on the current bugfixes branch, so this is not
already fixed there.

Candidate mechanism, NOT yet confirmed: in `atom_map_kokkos.cpp`,
`map_set_device()` sizes `d_sorted` to `atom->nmax` but then takes a subview
over `nall = nlocal + nghost`.  Confirming it needs a build with debug
information, which did not fit in the disk budget of this session.

## Open: dump grid does not reproduce here

`test_dump_grid` passes all 19 cases under the same split memory build and GPU
settings, and the watch and stale detectors report nothing.  Whatever fails on
a GPU is not visible to this tool in this configuration.

## Two defects in the debug tooling branch itself

* `KOKKOS.cmake` declared `-DLMP_KOKKOS_DEBUG_SYNC` `PRIVATE` to the `lammps`
  target.  `dual_view_kokkos.h` selects a different DualView on that macro, so
  the unit test binaries compiled a different class than the library holds --
  an ODR mismatch, and the unit test build fails outright.  Changed to `PUBLIC`
  in the worktree to get a usable build.
* that branch has no `LAMMPS_ACCELERATOR_ARGS` support in `unittest/testing/core.h`
  and no `add_kokkos_test`, both of which arrived later on the bugfixes line.
  Without them the test binaries silently ignore the accelerator settings and
  run the plain styles, so the tool reports a clean run while testing nothing.
  Check with a deliberately invalid setting, which must abort:

      LAMMPS_ACCELERATOR_ARGS="-k on -sf kk -pk kokkos bogusoption 1" ./test_dump_grid


## Sweep: the fix adapt "scale" factor across the KOKKOS pair styles

Signature: the CPU style multiplies its force and energy by `scale[i][j]` and
returns that array from `extract()`, so `fix adapt` writes to it; the `/kk`
subclass inherits both and never reads it.  The accelerated run then silently
computes the unscaled potential.  `fix adapt` calls `Pair::reinit()`, which
calls `init_one()` for every type pair, so a style that refreshes its device
copy there stays current.

| CPU style | KOKKOS variant | applies scale |
|---|---|---|
| pair coul/cut | coul/cut/kk | yes, in the per-type-pair parameters |
| pair coul/cut/global | inherits PairCoulCutKokkos | yes |
| pair coul/long | coul/long/kk | **no -- fixed here** |
| pair coul/slater/long | coul/slater/long/kk | yes |
| pair eam | eam/kk | yes |
| pair meam | meam/kk | yes |
| pair pace | pace/kk | yes |
| pair pace/extrapolation | pace/extrapolation/kk | yes |
| pair snap | snap/kk | **no -- fixed here** |
| pair ufm | ufm/kk | yes |

Both defects were confirmed by running with and without the accelerator and
comparing, and the factor was exactly the reciprocal of the requested one:

    snap, weights 0.2 and 0.8   plain -5.7803044   kokkos -11.560901  (x2)
    coul/long, scale 0.25       plain -19.166655   kokkos -76.666619  (x4)

Both now agree with the plain styles to every printed digit, and the 50 coul
and snap force style tests still pass.

The styles that expose `scale` but have no KOKKOS variant were not examined:
coul/esp, coul/msm, coul/tt, coul/streitz, thole, lj/cut/thole/long,
lj/sf/dipole/sf, kim, quip.

### Separate, not a KOKKOS problem: the soft core FEP styles

`src/FEP/pair_coul_long_soft.cpp` allocates `scale`, sets it to 1.0,
symmetrises it in `init_one()` and returns it from `extract()`, but neither
`compute()` nor `single()` ever reads it, so `fix adapt ... pair coul/long/soft
scale` is accepted and does nothing at all -- on the CPU as much as with the
accelerator.  `pair_lj_cut_coul_long_soft.cpp` and `pair_coul_cut_soft.cpp`
have the same shape.  Either apply the factor the way pair coul/long does, or
drop the array and the extract() entry so the fix reports an error instead of
ignoring the request.  Left alone here because it is not a KOKKOS defect.

## examples/nemd/in.nemd on a GPU

Two separate things, and only one of them is a defect.

### 1. The 1000 step regression failure is chaotic amplification, not a bug

The reported errors against the gold log (Temp 0.026, E_pair 0.165) are what a
round-off seed grows into over 1000 steps of this deck.  Measured on a CPU,
comparing the plain styles with the KOKKOS ones in the same binary:

    step     0    |dTemp| 0        |dE_pair| 1e-07
    step    50    |dTemp| 5.3e-06  |dE_pair| 8.2e-06
    step   300    |dTemp| 6.5e-05  |dE_pair| 7.9e-05
    step   600    |dTemp| 9.5e-03  |dE_pair| 9.6e-03
    step   900    |dTemp| 1.7e-01  |dE_pair| 1.5e-01

Six orders of magnitude in 900 steps, e-folding roughly every 86 steps: 160
atoms of two dimensional Lennard-Jones under shear is chaotic, and the deck
runs far past the point where any two bitwise-different trajectories separate.
The reported errors sit exactly on that curve.

On a GPU the seed is supplied for free: the force summation uses atomics, so
the order of the additions varies between runs of the same binary.  The test
as written therefore cannot be expected to reproduce a stored log.

The check that settles it, on the GPU: run the same binary twice and compare
the two runs with each other, not with the gold log.

* the two runs differ from each other -> nondeterminism, and this deck needs a
  shorter comparison window or a tolerance that grows with step number
* the two runs agree and both differ from the gold log -> something did change,
  and the gold log is the right thing to bisect against

The detector output does not support a coherence bug here either.  Running the
2 rank, 1000 step case under the split memory build gives 662 reports, all of
them in `CommKokkos::exchange_device` on `comm:k_count` and `comm:k_buf_send`.
The same run of a plain melt deck, which is not failing, produces the same
labels, so they are the scratch buffer noise the guide warns about.  The
difference of the two label sets is empty.

### 2. A real bug: fix deform corrupts atom:x at a box flip with KOKKOS

`fix deform ... xy erate` grows the tilt until the box flips.  For this deck
the flip is at about step 12500, so the stock 1000 step run never reaches it
and the regression test never sees this.  Running longer, the split memory
build aborts:

    LAMMPS::DualView::modify_host ERROR: concurrent modification of host and
    device views in DualView "atom:x"
      DualView<double *[3], LayoutRight, Serial>::modify_host()
      AtomVecAtomicKokkos::modified(ExecutionSpace, unsigned long)
      AtomKokkos::modified(ExecutionSpace, unsigned long)
      ModifyKokkos::pre_exchange()
      VerletKokkos::run(int)

Both sides of the coordinates claim to be newer than the other, so one set of
writes is dropped.  On a CPU build the two sides are one allocation and this is
invisible; on a GPU it silently corrupts the positions from the flip onwards.

Reproduced at every combination tried:

| configuration | result |
|---|---|
| 1 rank, KOKKOS, GPU settings | aborts at the flip |
| 2 ranks, KOKKOS, GPU settings | aborts at the flip |
| 4 ranks, KOKKOS, GPU settings | aborts at the flip |
| comm host / sort no / atom/map no | aborts at the flip |
| 2 ranks, plain styles | runs to 20000 steps cleanly |

So it is neither rank count nor any one device setting: it is the flip path
itself under KOKKOS.  That path is

    domain->image_flip(); domain->remap_all();
    domain->x2lamda(nlocal); migrate_atoms(); domain->lamda2x(nlocal);

where `x2lamda`/`lamda2x` run on the device through DomainKokkos while
`FixDeformKokkos::migrate_atoms()` brackets the host only irregular migration
with `sync(Host,ALL_MASK)` and `modified(Host,ALL_MASK)`.  The comments already
in `fix_deform_kokkos.cpp` show that the bracketing here has been adjusted
before.  ### The flip bug is already fixed on bugfixes

Chased to the end: the abort is the *old* bracketing, and the current bugfixes
branch already carries the cure.  `1b110d2531 KOKKOS: fix defects in fix and
compute styles` replaced

    void FixDeformKokkos::pre_exchange()
    {
      atomKK->sync(Host,ALL_MASK);      // claims the host for the whole call
      FixDeform::pre_exchange();        // ... but x2lamda/lamda2x run on the device
      atomKK->modified(Host,ALL_MASK);  // and this then claims host as well
    }

with a bracket around the migration alone, which is the one host-only step of
that path, and dropped the matching host bracket in `update_box()`.  It also
factored `irregular->migrate_atoms()` out of `FixDeform::pre_exchange()` into a
virtual `FixDeform::migrate_atoms()` so the KOKKOS class has something narrow
to override.

Verified rather than assumed: porting those four files back into the debug
worktree, which sits 905 commits earlier, and rebuilding makes the abort
disappear at 1, 2 and 4 ranks over the full 20000 steps, with the thermostat
holding T = 1.0 across the flip.  Without the port the same binary aborts at
step 12000 every time.

So a GPU run on current bugfixes does not have this defect.  A GPU run on a
base older than `1b110d2531` does, and it corrupts the coordinates silently
from the first flip onwards -- worth knowing when the failing job is built
from an older tree.

To reproduce:

    sed 's/^run             1000/run             20000/' examples/nemd/in.nemd > in.nemd.long
    ./lmp -in in.nemd.long -k on -sf kk -pk kokkos neigh full newton off \
          comm device sort device atom/map device gpu/aware on
