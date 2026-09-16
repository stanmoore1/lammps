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
