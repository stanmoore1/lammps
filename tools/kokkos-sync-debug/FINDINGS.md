# Split-memory sync debugging over the examples -- findings

Branch `claude/lammps-kokkos-dualview-debug-t9be12`.  361 example inputs, 4 MPI
ranks, GPU-equivalent package settings:

    -pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on

Stage order as asked: audit, then poison, then stale/watch.

## Stage 1 -- audit (LMP_KOKKOS_AUDIT=1)

361 inputs.  No undeclared write survived triage: every report was a style whose
`datamask_modify` is the base-class `ALL_MASK`, which the audit says out loud it
cannot check.

## Stage 2 -- poison (LMP_KOKKOS_POISON=1, build-poison)

361 inputs, 28 ASan reports over 16 distinct sites.  Fixed, with commits:

| site | defect | commit |
|---|---|---|
| `atom_map_kokkos.cpp:236,238` (11 SEGV) | `map_delete()` left `max_same` at its high-water mark, so the next `map_set()` skipped the allocation and wrote through the destroyed view; `sametag` realloc also had to move after `map_init()` for MAP_HASH | a85c76c29 |
| `fix_langevin.cpp:586` | `fix langevin/kk` called the base `omega_thermostat()`, reaching for `atom->torque/omega/radius` as host pointers.  Its own `omega_thermostat_kokkos()` was dead code | c293b7250 |
| `min_fire_kokkos.cpp:220` | views and `nlocal` bound once before the iteration loop and reused across `energy_force()`, which reneighbors and reallocates; four device writes also unclaimed | c293b7250 |
| `min_linesearch_kokkos.cpp:253,289` | `fvec`/`xvec` are unmanaged views over the raw device pointers, read after `alpha_step()` and after `max_alpha()` with no sync | c293b7250 |
| `compute_chunk_atom.cpp:1712` | `ModifyKokkos::setup()` ran the compute loop bare while both fix loops sync/claim; `compute chunk/atom` binned on stale host coordinates | c293b7250 |
| `fix_cmap_kokkos.cpp:191` | `pre_neighbor()` never synced `num_crossterm`, `crossterm_type` or `crossterm_atom1..5` to the device although the scan reads them | 91a3d94d4 |
| `fix_shake.cpp:455` | `FixShakeKokkos::init()` reached the base `init()` without syncing `k_shake_flag`/`k_shake_type` to host | 91a3d94d4 |
| `comm_tiled.cpp:916`, `atom_vec.cpp:825` | `CommTiledKokkos` left the device claim on `k_buf_send` standing while the plain `CommTiled` routines refill it through the host pointer; `CommKokkos`, the class it was cloned from, clears it | 91a3d94d4 |
| `compute_viscosity_cos.cpp:249` (SEGV) | `vbiasall` never allocated; `FixNHKokkos` batches the bias removal and so reaches `remove_bias_all()`, which the plain `FixNH` never does | 92d7e3dcc |
| `atom_vec.cpp:1623` (SEGV) | `atom_style ellipsoid superellipsoid` accepted under KOKKOS but `AtomVecEllipsoidKokkos` grows neither `radius` nor a device `bonus_super`; now rejected with an error | 92d7e3dcc |
| `kokkos.cpp:916` (SEGV) | debug-build artifact: the build routes host-backend lists to the Device space, so `neigh_count()` mirrors empty device arrays with a live `inum`.  Confirmed clean under the plain KOKKOS build.  Walk bounded so the detector build survives to the end of the run | pending |

Still open:

- `FixNumDiff::calculate_forces()` (`fix_numdiff.cpp:201`) and
  `ComputeBornMatrix::compute_numdiff()` (`compute_born_matrix.cpp:511`) displace
  atoms through plain host pointers and then re-enter the Kokkos force pipeline
  via `update_force()` / `update_virial()`.  The host writes are never claimed.
  `ModifyKokkos` forces `auto_sync` around a fix that is not Kokkos-aware, which
  is the mechanism that would pick them up; `Thermo::compute()` has no such
  wrapper at all, so the compute path is unprotected.  To be re-checked against a
  rebuilt poison binary before deciding the fix.

## Stage 3 -- stale/watch (LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1)

361 inputs, 174 with reports, 1.2M report lines but only 564 distinct
signatures over **39 distinct routines**.  Triaged by routine:

**One real finding.**  `compute born/matrix numdiff` displaces the atoms through
`atom->x`, recomputes the virial and restores them, and `Thermo::compute()`
reaches it from `Output::write()` with no wrapper.  The watch detector named it
exactly:

    [watch] atom:x: the host side was written without a claim and this
            sync_device has nothing to copy -- the device keeps stale data

So the pair recomputes from the undisplaced positions: every finite difference
is taken at zero displacement, and the closing restore is lost the same way.
`examples/ELASTIC_T/BORN_MATRIX/Argon/Numdiff/in.ljcov` has no `fix numdiff` at
all, so the compute path is reached on its own.  `auto_sync` is the mechanism
for this and `VerletKokkos::setup()` already used it for the setup output; the
run loop, both minimizer iterate loops and `MinKokkos::setup()` did not, and
`MinKokkos::setup()` explicitly turned it off.  Fixed at all five sites in
cb4caef7d, with a trailing host claim so a write made after the style's last
device sync is not dropped.

**Everything else is one benign class: binding a view before the sync that
fills it.**  `view<DeviceType>()`, `view_host()`, `.data()` and `.extent()` all
count as a read of that side, so a routine that captures the view into a functor
and syncs afterwards reports every time -- but the sync lands in the same
allocation the captured view addresses.  `NPairKokkos::build` (special,
nspecial, 12 inputs), `AtomKokkos::sort_device`, `FixShakeKokkos::dof`,
`FixEOStableRXKokkos::init`, `Special::combine` (through the `grow()` it
triggers) and every `AtomVec*Kokkos::grow`/`grow_pointers` are this.  Each was
checked for a following sync before being discarded.  Documented in
`.github/dev-docs/kokkos-sync-debugging.md` (880435a73) so the next sweep does
not re-chase it; `sort_device` had a one-line version, which was reordered.

The known-benign scratch pairs also appear as expected and were left alone:
`comm:k_count` in `exchange_device` (102 inputs), `NBinSSAKokkos::gbincount`
written on the device and then deliberately re-zeroed on the host, `atom:f` in
`force_clear` (the force array is exempt from the protocol), and the
`clear_sync_state` pairs in `MinLineSearchKokkos::alpha_step`.

## Stage 4-6 -- mixed precision

`KOKKOS_PREC=mixed` forces `PKG_ML-IAP=off` (cmake/Modules/Packages/KOKKOS.cmake:293),
so the ML-IAP examples are out of the mixed-precision sweep.

## New-input catch-up (869-input list, 2026-09-15)

Upstream moved examples/{gjf,relres,tracker} under examples/PACKAGES/ and renamed the
electrode/dielectric inputs; 55 inputs were never covered by the 845 sweeps.  Audit
catch-up over just those:

### How to read "reads stale" (corrected)

`stale` is `need_sync_device()` sampled at style entry, i.e. "this array is not in
the style's datamask_read AND the device copy is behind".  It is NOT evidence the
style read it.  Two large benign classes produce it:

- a style that legitimately never touches the array (gjf/kk declares EMPTY_MASK and
  manages its own syncs, so every array it does not use shows here);
- a style running with `execution_space == Host`, where ModifyKokkos syncs the HOST
  side, correctly leaving the device behind -- every array then reports at once.
  PACKAGES/frenkel is exactly this: all seven arrays, one event at step 101,
  repeated once per MPI rank.

So a "reads stale" line is only a lead when the style demonstrably does read that
array.  The poison, stale-watch and ASan detectors remain the load-bearing ones.

- `rigid/nve/small/kk reads stale image|mask|molecule` (13 rigid/cubes* inputs) --
  fix rigid/small/kk declares `datamask_read = X|F|V|VIRIAL|TYPE|TAG`, but base-class
  host paths it inherits read image/mask/molecule per step.  Stale-watch on
  in.rigid.cubes.nve resolves every report to the known-benign classes
  (AtomVec*Kokkos::grow_pointers binding before a following sync, and CommKokkos'
  internal k_buf_send/k_count scratch), so the narrow datamask is not *currently*
  producing a wrong read -- but it is only correct by accident of who syncs first.
  TO CONFIRM: which host path does the audit attribute; not yet root-caused.
- `gjf/kk reads stale image|v` -- NOT A BUG.  fix_gjf_kokkos declares EMPTY_MASK
  (both read and modify) and syncs per routine; it never touches image.
- `nvt/kk reads stale f|image|mask|tag|type|v|x` in PACKAGES/frenkel -- NOT A BUG,
  host-execution-space class above.
- `comm:k_count: host side written, never claimed, and is now lost` (136x):
  NOT A BUG.  comm_kokkos.cpp:1165 writes k_count.view_host()(0) purely as the
  growth loop's own sentinel and the following deep_copy(d_count,0) discards it
  on purpose.

## Mixed poison sweep, ASan reports (12 at input 760/869)

FIXED, pushed:
- `BondHarmonicKokkos::compute` line 185, atomic add into poisoned f, via
  Respa::force() (PACKAGES/filter_corotate/in.respa).  Same root cause as the
  chreg-polymer stale reads -- bonded styles did not sync.  Commit 0277fece79.
- `FixLangevinKokkos::post_force` -> `Group::count()` reading poisoned host
  atom->mask (hyper.global, hyper.local, widom.lj).  Commit 8ecf42b49c /
  9534e76efb.  Latent only for a subgroup: every example in the tree uses
  "group all", where a stale mask still gives the right count.
- `NEBSpin::~NEBSpin` MPI_Comm_free on an uninitialised `roots` (SPIN/gneb_iron)
  -- fixed earlier as c84409ae9f; this sweep's binary predates it.

STILL OPEN, need a live reproduction (the poison build is busy with the sweep;
a re-verify stage is queued to rebuild it and re-run just these):

- `FixRigidSmall::pack_reverse_comm` reading a poisoned 961032-byte Kokkos
  HostSpace buffer (= k_bodyown, int per atom), reached from
  `FixRigidSmallKokkos::dof()` -> base dof() -> CommBrick::reverse_comm.
  3 deposit/rigid-*-small inputs, plus mc/in.hmc.rigid via setup().
  The KK dof() wrapper already does k_bodyown.sync_host() under `if (setupflag)`
  and forces the host comm path, so something re-claims the device between that
  sync and the pack, or the host bytes were poisoned outside the flag protocol.
  Ruled out: clear_sync_state() -- poison_apply() unpoisons BOTH sides when both
  counters are 0, so the documented opt-out is not the source.
  Note the poison starts 128 bytes (32 ints) into the allocation, not at 0.

- `FixGroup::pack_forward_comm` / `Group::count` reading poisoned atom->mask
  under ModifyKokkos::post_force, and `FixNumDiff::calculate_forces` at
  fix_numdiff.cpp:225 likewise (mc/in.gcmc.co2, in.gcmc.h2o, numdiff/in.numdiff).
  These are non-kokkosable consumers, which ModifyKokkos already runs with
  auto_sync=1.  Hypothesis to test: auto_sync only acts through
  AtomKokkos::modified(); a claim made straight on the DualView
  (k_mask.modify_device() inside CommKokkos or NPairKokkos) bypasses it, so the
  host copy is left poisoned with auto_sync none the wiser.

## Mixed-precision stale watch, all 869 inputs -- COMPLETE, no new bugs

427 of 869 inputs reported.  Every one of the 5 non-benign-looking call sites
resolves to a documented benign class; nothing new to fix.

| site | inputs | verdict |
|---|---|---|
| `FixShakeKokkos::post_force` reads atom:f | 11 | bind-before-sync.  d_x/d_f/d_type are bound at the top, the `atomKK->sync(execution_space,X|F|...)` follows a few lines down.  ~500 hits = one per step, i.e. once per call, exactly as that class predicts. |
| `FixShakeKokkos::dof` reads atom:mask | 1 | same: binds d_mask/d_tag, then syncs MASK\|TAG before the kernel. |
| `FixRigidSmallKokkos::refresh_atom_views` / `sort_kokkos` | 7 | same -- refresh_atom_views exists to rebind views. |
| `CommKokkos::grow_swap` / `grow_list` on comm:sendlist | 9 | reallocation, 1 hit each. |
| `FixWallFlowKokkos::grow_arrays` on current_segment | 1 | 12 hits, a DIFFERENT site from the init() one fixed in 122d6024ed.  Device is claimed either side of the grow so the host content is not used; low priority but not yet read closely. |

The discriminator from the dev-doc ("look for a following sync in the same
routine") did all the work here.  Worth noting what is NOT in this list: the
bonded styles, which dominated the double-precision stale sweep before
0277fece79.  That fix removed the only real class the stale watcher was seeing.

## ASan KOKKOS double sweep -- heap-buffer-overflow in fix rigid/small + fix gcmc

`examples/mc/in.gcmc.co2`, reported by the plain AddressSanitizer build (no
sync debugging at all, so this is a real memory error, not a protocol artifact):

    READ of size 4, 388 bytes BEFORE a 3920000-byte region
    #0 FixRigidSmall::copy_arrays(int,int,int)  fix_rigid_small.cpp:2857
    #1 AtomVec::copy(int,int,int)               atom_vec.cpp:350
    #2 FixGCMC::attempt_molecule_deletion_full()  fix_gcmc.cpp:2076

Line 2857 is

    if (delflag && bodyown[j] >= 0) {
      bodyown[body[nlocal_body-1].ilocal] = bodyown[j];

and the region is `body`, allocated in the FixRigidSmall constructor.  The
address is BEFORE the allocation, not past its end: this is `body[-1]`, i.e.
`nlocal_body == 0` while `bodyown[j] >= 0` still claims atom j owns a body.
Those two cannot both be true of one consistent state.

KOKKOS-SPECIFIC -- established by A/B on the SAME binary, same input, same
4 ranks:
  - no `-sf kk`:  rc=0, 0 reports
  - with `-sf kk`: rc=1, reproduces every time

FixRigidSmallKokkos does NOT override copy_arrays, so the base runs and reads
the raw host `bodyown[]`, `body[]` and `nlocal_body`, which the KOKKOS subclass
maintains as DualViews.  When that bookkeeping is live on the device the host
copies are the stale ones, and a stale `bodyown[j]` against a current
`nlocal_body` is exactly the contradiction above.

### CORRECTION: the coherence hypothesis above is WRONG

I wrote a fix on that hypothesis -- FixRigidSmallKokkos overriding copy_arrays
and set_arrays to flush the bookkeeping device->host first, plus pre_neighbor
honouring the resulting host claim instead of clearing it -- and it does NOT
fix the bug.  Two results refute the hypothesis:

  - with the override in place (so the flush definitely ran), the overflow
    still fires, and the backtrace goes straight THROUGH it:
        #0 FixRigidSmall::copy_arrays        fix_rigid_small.cpp:2857
        #1 FixRigidSmallKokkos::copy_arrays  fix_rigid_small_kokkos.cpp:318
  - the overflow reproduces with `-pk kokkos comm host sort no atom/map no`,
    i.e. the host exchange path, where pre_exchange() already flushes and no
    device claim is outstanding at all.

So stale host bookkeeping is not the cause.  The fix was reverted; it also made
the run 100x slower (401 s timeout against 4 s), which is a second reason not
to keep it.

What still holds: the A/B is unchanged -- no `-sf kk` is clean, `-sf kk` crashes
-- so it IS KOKKOS-specific, and the immediate cause is still `nlocal_body == 0`
while `bodyown[j] >= 0`.  But since it is independent of the comm path, the
difference has to be in the KOKKOS subclass's own body bookkeeping (something
around nlocal_body maintenance -- unpack_exchange_kokkos adjusts it directly at
fix_rigid_small_kokkos.cpp:1991-2031), not in host/device coherence.
NOT YET DIAGNOSED.  Do not assume the poison finding below shares a root cause
with it; that was the assumption that just failed.

The earlier guess that this shares a root cause with the still-open poison finding
(`FixRigidSmall::pack_reverse_comm` reading a poisoned k_bodyown via
FixRigidSmallKokkos::dof).  Same array, same class: base-class host code
reached without the KOKKOS subclass flushing its bookkeeping down first.


## gcmc + rigid/small/kk: sharper diagnosis, still no working fix

A second attempt also failed.  Recording what is now established, and what
each attempt disproved, so the next person does not repeat either.

ESTABLISHED (all directly observed, not inferred):

1. `sizeof(Body)` is 392 and `offsetof(Body, ilocal)` is 4, so the faulting
   address at `base - 388` is exactly `body[-1].ilocal`.  `nlocal_body` is 0
   at the point `bodyown[j] >= 0` says atom j owns a body.  Not a large
   negative index, not a different field: exactly zero bodies.

2. Under the sync-debugging build the same input aborts EARLIER, in the
   DualView guard, with a precise chain:

       DualView::modify_host  -- "concurrent modification of host and device
                                  views in DualView rigid/small:atom2body"
       FixRigidSmallKokkos::pre_neighbor
       ModifyKokkos::pre_neighbor
       FixGCMC::energy_full
       FixGCMC::attempt_molecule_insertion_full   (and ..._deletion_full)
       FixGCMC::pre_exchange

   fix gcmc re-enters modify->pre_neighbor() from inside its own
   pre_exchange(), while the claim reset_atom2body() left on the device is
   still outstanding.  pre_neighbor's HOST-exchange branch then calls
   modify_host() on top of it.  The DEVICE branch guards against precisely
   this (clear_sync_state() before modify_device(), commented "...marking the
   device on top of that leaves both flags set and Kokkos::abort()s"); the
   host branch has no equivalent.  That asymmetry is real and is the best
   lead.

DISPROVED, attempt 1 -- "the host bookkeeping is stale":
   Overriding copy_arrays/set_arrays to flush device->host first does NOT fix
   it; the overflow still fires with the backtrace running through the
   flushing override, and it reproduces with `comm host sort no atom/map no`
   where nothing is stale.  Also made the run ~100x slower.

DISPROVED, attempt 2 -- "mirror the device branch's clear_sync_state()":
   Clearing atom2body alone moves the abort to bodyown; clearing the whole set
   (bodytag, bodyown, atom2body, xcmimage, displace, vatom, eflags...) removes
   every abort but the run then dies with "Non-numeric atom coords -
   simulation unstable".  So the device side of those arrays DOES hold data
   that matters on the host exchange path, and the reasoning that only the
   host writes them there is wrong.

WHAT THIS NEEDS: someone who knows the intended ownership of bodyown/
atom2body/xcmimage/displace across the two exchange paths.  The fix is
probably to not let the re-entrant pre_neighbor take the host branch at all
while a device claim is live, rather than to clear claims -- but that is a
design question, not a patch I can validate from here.


## Poison re-verify against a freshly built binary: 10 of 16 cleared

The sweep directory still held the FIRST poison run's reports, dated three days
earlier than the binary carrying the fixes.  They look current because the
directory is current; comparing each report's mtime against the binary's build
time is what separates them.  Checked that way, and re-run, the picture is:

CLEARED -- no report from a binary that has the fix:
  hyper.global, hyper.local, widom.lj   fix langevin/kk mask sync
  numdiff                               fix numdiff mask sync
  filter_corotate/in.respa              bonded styles sync/claim
  gneb_iron                             roots initialisation
  wall.flow                             wall/flow current_segment
  ilves/in.peptide-ilves                bonded sync -- was a heap-buffer-overflow
  ilves/in.rhodo-ilves                  at bond_harmonic_kokkos.cpp:170, INSIDE
                                        the kernel, never separately diagnosed
  pafi/in.pafi                          bonded sync -- was a heap-use-after-free
                                        in fix_property_atom_kokkos.cpp:191

The last three are the interesting ones: five ilves inputs and pafi were never
diagnosed on their own, and the bonded sync/claim fix resolved them as a side
effect.  One fix, five example directories.

STILL REPORTING -- all of them things not claimed as fixed:
  deposit/in.deposit.molecule.rigid-{small,nve-small,nvt-small}
  mc/in.hmc.rigid          FixRigidSmall::pack_reverse_comm / copy_arrays
                           -- the body[-1] bug, two failed fix attempts above
  mc/in.gcmc.co2, gcmc.h2o FixGroup::pack_forward_comm under gcmc, never
                           diagnosed


## Divergence: split memory vs regular memory, 869 inputs -- 18 REAL

The one check that does not go through the sync protocol.  Same source, same
compiler, same flags, same 4 ranks; the only difference is that build-sync
keeps the two sides of every DualView in separate allocations while build-plain
lets them alias.  With every sync correct the two are bit-identical, so any
difference means a value was taken from the side that was not current --
exactly the failure an accessor-based detector cannot see.

Getting to a trustworthy number took three filters, each of which removed
something that would have been reported as a bug:

  869 inputs
  -> 302 actually compared      (557 are rejected outright by KOKKOS with a
                                 clean ERROR and produce no thermo on either
                                 side; 10 more run but print no thermo)
  ->  51 differ
  ->  42 differ in VALUES       (9 differed only in LENGTH: the split build is
                                 slower and the sweep bounds runs with "timer
                                 timeout", so the two arms stop at different
                                 steps.  Comparing only the steps present on
                                 both sides removes these.)
  ->  18 are REPRODUCIBLE       (24 of the 42 are not deterministic at all --
                                 the same binary run twice already disagrees,
                                 e.g. fix balance redistributing on measured
                                 time.  For those, "the memory models disagree"
                                 is not a finding.)

The 18, with how much of the shared trajectory differs:

  ASPHERE/box/in.box, in.box.mp            10 of 11
  ASPHERE/dimer/in.dimer, in.dimer.mp      10 of 11
  ASPHERE/star/in.star, in.star.mp         10 of 11
  PACKAGES/drude/butane/in.butane.lang     20 of 41
  PACKAGES/drude/butane/in.butane.nh       40 of 41
  PACKAGES/drude/ethylene_glycol/...       10 of 11
  PACKAGES/drude/swm4-ndp/in.swm4-ndp.nh  100 of 101
  PACKAGES/drude/toluene/in.toluene.lang   40 of 41
  PACKAGES/drude/toluene/in.toluene.nh     40 of 41
  PACKAGES/fep/ta/in.spce.lmp               1 of 2
  PACKAGES/pafi/in.pafi                     1 of 27
  PACKAGES/relres/in.22DMH.respa           10 of 11
  VISCOSITY/in.nemd.2d                      2 of 10
  mc/in.gcmc.h2o                            4 of 17
  mc/in.hmc.rigid                           1 of 12

Shape of the divergence, the same in every one examined: step 0 is IDENTICAL on
both sides, and they part company at the first or second thermo interval and
grow apart from there.  Identical initial state rules out setup; something
during integration reads the stale side.

    ASPHERE/box   step 0: 1.4252596 = 1.4252596
                  step 100: 4.7699234 vs 4.7726984
                  step 300: 21.249889 vs 20.923297
    drude/swm4    step 0: 4568.0413 = 4568.0413
                  step 20: 3742.804 vs 3736.9936

MD is chaotic, so the SIZE of the gap says nothing -- a last-bit difference
grows into this.  What matters is that there is any difference at all, because
with correct syncs there would be none.

NOT ROOT-CAUSED.  But the two families are suggestive: ASPHERE and drude both
carry per-atom state beyond x/v/f (omega, angmom, quat, and the drude bookkeeping),
which is where datamask coverage is most likely to be incomplete.  That is a
hypothesis, not a result -- and the last two hypotheses in this file were both
wrong, so it should be tested before it is believed.


## Verification pass over the four outstanding issues

### FIXED -- fix group's forward comm packs into a device-claimed buffer
CommKokkos::forward_comm(Fix *) delegates to CommBrick, which packs through
buf_send -- the raw host pointer behind k_buf_send -- while a device claim is
outstanding.  Proved by a WRITE to poisoned memory at fix_group.cpp:406, into a
140992-byte Kokkos HostSpace allocation.  Same defect forward_comm_array() was
fixed for.  in.gcmc.h2o: 3 reports -> 0.  Commit 6b414f5e6a.

### NOT A BUG -- rigid/small/kk's narrow datamask over atom->molecule
The base reads atom->molecule in five places: three in the constructor (no
device state yet) and two in readfile()/write_restart_file().  Both of the
latter are reached only through callers that have already synced everything to
the host -- write_restart writes the atom arrays first, and the infile path
runs inside the host rebuild.  Verified by running examples/rigid/in.rigid.small.infile
(which uses infile, so restart_file is set and write_restart_file really runs,
producing ri.restart.rigid) under the UNFIXED poison build: zero reports.  A
speculative MOLECULE_MASK addition was written and then reverted as unnecessary.

### STILL OPEN -- rigid/small + gcmc, now better understood but not fixed
Fixing fix group let in.gcmc.co2 run further and exposed the real sequence:

  1. FixGCMC inserts an atom -> Modify::create_attribute -> FixRigidSmall::set_arrays
     WRITES bodyown/bodytag/atom2body/xcmimage/displace through host pointers
     while the device owns them.  The write is lost.
  2. The new atom therefore keeps whatever the device held, which can be a
     bodyown >= 0 naming a body that does not exist.
  3. FixRigidSmall::copy_arrays later reads that and indexes body[nlocal_body-1]
     with nlocal_body == 0 -- the original overflow.

Without any fix the unfixed build does not merely read out of bounds, it drives
nlocal_body negative: "Kokkos::RangePolicy bounds error: The lower bound (0) is
greater than the upper bound (-2)".  The same input completes cleanly on the
non-KOKKOS CPU build (21 thermo rows, 14 s), so the input is legitimate.

ATTEMPT 3, REVERTED.  Overriding set_arrays and copy_arrays to sync the
bookkeeping to the host, let the base write it, then claim the host side, does
take in.gcmc.co2 from 62 poison reports to 0 -- but it REGRESSES
examples/rigid/in.rigid.spheres, which goes "Non-numeric atom coords" at step
~900 where the unfixed build finishes.  Same failure mode as attempt 2.  The
host claim is evidently wrong somewhere else in the fix's sync discipline.

Keeping only the sync_host half was considered and rejected: it removes the
poison reports without saving the lost write, i.e. it silences the detector
while leaving the bug.  That is worse than leaving it visible.

### STILL OPEN -- the 18 memory-model divergences
ASPHERE/box narrowed a long way and still not root-caused:
  - first divergence is at step 1, not an accumulation
  - np=1 and np=2 agree; only np=4 differs, so it is decomposition dependent
  - per-atom x and f are bit-identical at np=1 through step 1
  - both builds are deterministic across 3 runs each at np=4, so the difference
    is real
  - removing fix adapt makes the two agree (with a constant non-zero
    coefficient, so forces are still present -- not a degenerate test)
  - k_params is a raw Kokkos::DualView, not the instrumented one, so the pair
    coefficients are not split and cannot be the stale side

ATTEMPT, REVERTED: adding the buf_send claim-clear to all 13 host-delegating
comm overloads did NOT fix it and made the stale count worse (12 vs 10), so it
was reverted.  Only the one verified site (the Fix forward overload) was kept.
