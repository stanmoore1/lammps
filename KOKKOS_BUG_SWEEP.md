# KOKKOS package bug sweep

Follow-up to the fixes on the `kk_bugfixes` branch: each bug fixed there was used
as a template and the rest of `src/KOKKOS/` was searched for the same shape, with
emphasis on the styles ported since April 2026.  Every entry below was verified by
reading the code (and the non-KOKKOS base class where relevant); nothing has been
changed yet.  CONFIRMED = the defect is certain from the code; PLAUSIBLE = the
shape is there, the consequence depends on usage.

Section 1 lists bugs that change results or crash.  Section 2 lists leaks and
housekeeping.  Section 3 lists what the clang `-Wall -Wextra` builds (double,
single, mixed precision) turned up.  Section 4 lists what was searched and found
clean.

## 1. Wrong results or crashes

### 1.1 `min_quickmin_kokkos.cpp:116` -- per-type masses never synced to the device (CONFIRMED)

`l_mass = atomKK->k_mass.view_device()` is divided into `dtfm` in the integration
kernel, but `k_mass` is never synced in this file.  The masses are written through
the plain host array, so on a GPU (and in single/mixed precision on any backend,
where the device copy is a separate allocation) the device masses are zero, `dtfm`
is infinite and the coordinates are NaN after the first iteration.  A minimization
has no integrator fix to push the masses across.  Same defect that `38f9554b2`
fixed in `MinFireKokkos::init()`.

Fix: `atomKK->k_mass.sync_device();` in `MinQuickMinKokkos::init()`.

### 1.2 `fix_wall_harmonic_outside_kokkos.cpp:49-63` -- per-atom virial tallied before it is sized (CONFIRMED)

The one wall/kk style that did not get the `v_setup_peratom()` override from
`95cf1400b`.  `if (vflag_atom)` is tested before `FixWall::post_force()` calls
`v_init()`, which is what sets `vflag_atom` and `maxvatom`, so on the first
per-atom-virial step `d_vatom` is still empty or sized from a stale `maxvatom`
while the kernel writes into it (out of bounds).  In addition the base
`v_init(vflag)` runs with `alloc = 1` and allocates the plain `vatom`, whose
pointer `create_kokkos` then overwrites: one orphaned allocation per growth.

Fix: copy the `v_setup_peratom(int)` override from `fix_wall_harmonic_kokkos.cpp:58-71`
(declare it in the header, call `v_init(vflag,0)`, then reallocate) and delete
the block at lines 51-55.

### 1.3 `compute_reaxff_atom_kokkos.cpp:33-38` -- ALL_MASK datamasks on a compute that owns no atom data (CONFIRMED)

`kokkosable = 1` but `execution_space` and both datamasks keep the `Compute`
defaults (`Host`, `ALL_MASK`).  `ModifyKokkos::setup()` syncs `datamask_read`
and then marks `datamask_modify` modified for every compute, so this one claims
every per-atom array on the host at every setup.  That is the mechanism of
`00d4c8c59` / `46ae8890a` (issue #5080): the next device claim aborts with
"Concurrent modification of host and device views", or host data is left dirty.
The compute reads everything from the reaxff pair style's own views; the only
atom array it touches is `atom->tag`, on the host, in `compute_local()`.

Fix: `datamask_read = TAG_MASK; datamask_modify = EMPTY_MASK;` in the
constructor, plus `atomKK->sync(Host,TAG_MASK)` before the `compute_local()`
loop, which runs outside the Modify wrappers.

### 1.4 `compute_temp_sphere_kokkos.cpp:52-56, 77-80, 118-122, 143-146` -- bias removal never claimed (CONFIRMED)

`tbias->remove_bias_all()` / `restore_bias_all()` are called with no sync of the
bias compute's `datamask_read` before and no
`atomKK->modified(tbias->execution_space, tbias->datamask_modify)` after; only
`atomKK->sync(execution_space, V_MASK)` follows.  With a non-KOKKOS bias compute
the bias is removed by writing the plain host `v` without a claim, so the sync
copies nothing and the reduction runs on velocities that still contain the bias:
a silently wrong temperature.  The restore is likewise unclaimed.  The correct
bracket is in `compute_temp_deform_kokkos.cpp:127-131` and `148-154`.

Fix: wrap both calls in that sync/modified pair.

### 1.5 `pair_mliap_kokkos.cpp:358, 384, 428, 499, 562, 630` -- kernels launched in the default execution space (CONFIRMED)

`pack_forward_comm_kokkos()` and its siblings take `view<DeviceType>()` views
and then launch `Kokkos::parallel_for(nv*nf, KOKKOS_LAMBDA ...)` with a bare
count, i.e. in the default (device) execution space.  `mliap/kk/host` is a
registered style (`PairMLIAPKokkos<LMPHostType>`), so in a GPU build its comm
packing runs a device kernel over host memory.  Same shape as the nbin fix in
`fa275ab5f`.  `mliap_unified_kokkos.cpp:272-285, 310, 381` and
`mliap_model_linear_kokkos.cpp:54` have the same launches, but over
`MLIAPDataKokkosDevice`, which only exists for the device instantiation
(PLAUSIBLE there).

Fix: `Kokkos::RangePolicy<DeviceType>(0, nv*nf)` (and likewise for the other
five launches).

### 1.6 `fix_nh_sphere_kokkos.cpp:52` -- debug output left in the constructor (CONFIRMED)

`fprintf(stderr, "flags dipole %d  dlm %d\n", ...)` prints on every rank for
every `fix nvt/nph/npt/sphere/kk`.  Delete the line.

### 1.7 `fix_spring_kokkos.cpp:115, 185`; `fix_wall_flow_kokkos.cpp:88`; `compute_ave_sphere_atom_kokkos.cpp:114` -- per-type masses read on the device without a sync (PLAUSIBLE)

`mass = atomKK->k_mass.view<DeviceType>()` with no `k_mass.sync<DeviceType>()`,
unlike the siblings ported at the same time (`fix_flow_gauss_kokkos.cpp:84`,
`fix_heat_kokkos.cpp:89`, `fix_nvk_kokkos.cpp:69`, `fix_baoab_kokkos.cpp:92`,
`fix_addtorque_group_kokkos.cpp:126`).  Works today only because `fix nve/kk`
and `fix nh/kk` sync the masses in their own `init()`; a `mass` command between
runs, or a run without such an integrator, leaves the device copy behind.

### 1.8 `compute_inertia_kokkos.cpp:58` -- host sync misses what the group routine reads (PLAUSIBLE)

`atomKK->sync(Host, MASK_MASK|RMASS_MASK|TYPE_MASK)` precedes
`group->inertia_extended()`, but that routine also reads `atom->radius`, the
`ellipsoid/line/tri/body` indices and the bonus arrays through plain host
pointers (`group.cpp:1698-1702`).  After a device integrator such as
`fix nve/asphere/kk`, which claims `BONUS_MASK` on the device, those host copies
are stale.  Fix: add `RADIUS_MASK|ELLIPSOID_MASK|LINE_MASK|TRI_MASK|BODY_MASK|BONUS_MASK`.

### 1.9 Host loops on plain pointers with no sync (PLAUSIBLE)

- `fix_nh_sphere_kokkos.cpp:63-70` walks `atom->radius` / `atom->mask` in
  `init()` without `atomKK->sync(Host, RADIUS_MASK|MASK_MASK)`
  (`fix_heat_kokkos.cpp:53` shows the idiom).
- `ComputeTempSphereKokkos` does not override `dof_compute()`, so the base host
  loop over `mask`/`radius` (`compute_temp_sphere.cpp:161-213`) runs on possibly
  stale host copies after the device kernels: wrong degrees of freedom.

### 1.10 `pair_hybrid_scaled_kokkos.cpp:224, 525, 606` -- forward comm through a null pointer (CONFIRMED)

`PairHybridKokkos` sets `execution_space = Device` but neither it nor
`PairHybridScaledKokkos` inherits `KokkosBase`.  `CommKokkos::forward_comm(Pair*)`
picks the device path on `execution_space` alone, does
`dynamic_cast<KokkosBase*>(pair)` (null here) and calls
`pairKKBase->pack_forward_comm_kokkos()` through it (`comm_kokkos.cpp:780`, and
`comm_tiled_kokkos.cpp:508` in the tiled twin).  Triggered by
`pair_style hybrid/scaled` under KOKKOS with an atom-style variable scale
factor, on any number of ranks.

Fix: derive `PairHybridKokkos` from `KokkosBase` and implement the device
pack/unpack for `atomscale`, or force the host comm path around those three
calls the way `fix_shake_kokkos.cpp` does with `forward_comm_device = 0`.

### 1.11 `remap_kokkos.cpp:186-198` -- scratch buffer copied while receives are still landing (CONFIRMED)

The same defect `36aa1bebf` fixed in grid3d.  All receives are posted into one
host buffer `plan->h_scratch` at their own offsets; the unpack loop then does
`MPI_Waitany` and, inside the loop, `Kokkos::deep_copy(d_scratch, plan->h_scratch)`
copies the whole buffer while the not yet completed receives are still writing
into it.  Reached on the `!usegpu_aware` path of the FFT remap with three or
more ranks.

Fix: `MPI_Waitall(plan->nrecv, plan->request, MPI_STATUSES_IGNORE)`, one
`deep_copy`, then unpack every `irecv` at `recv_bufloc[irecv]`.

### 1.12 `remap_kokkos.cpp:153-166` -- no fence between the pack kernel and the MPI send (CONFIRMED)

There is no `Kokkos::fence()` anywhere in the file.  With `usegpu_aware`,
`plan->pack()` launches a kernel into `d_sendbuf` and the next statement is
`MPI_Isend`/`MPI_Send` of that device pointer; `MPI_Irecv` into `d_scratch`
is likewise not fenced against the previous call's unpack kernels.  Every other
KOKKOS comm path fences (`comm_kokkos.cpp:195`, `grid3d_kokkos.cpp:739`).
Secondary: with `usenonblocking` each iteration re-copies the whole host send
buffer while the previous `MPI_Isend` may still read its stretch (same values,
so a standards violation rather than corruption).

Fix: fence after `plan->pack()` and before each MPI call; copy only the
`[send_bufloc[isend], +send_size[isend])` stretch.

### 1.13 `comm_tiled_kokkos.cpp:573` -- reverse comm size logic inverted (CONFIRMED, also in the CPU class)

`if (size) nsize = MAX(comm_reverse, comm_reverse_off); else nsize = comm_reverse;`
discards the caller's `size` and applies the newton-off size exactly backwards.
`CommKokkos::reverse_comm_device` and `CommBrick` have
`if (size) nsize = size; else nsize = MAX(comm_reverse, comm_reverse_off);`.
Copied from `CommTiled::reverse_comm(Pair*)` (`comm_tiled.cpp:1321`), which has
the same defect.  A pair with `comm_reverse_off > comm_reverse` under
`comm_style tiled` and `newton off` undersizes the buffer and the per-proc
offsets: overrun or garbage.

### 1.14 `fix_shake_kokkos.cpp:2166` -- host pack claims a write that never happens (CONFIRMED)

`pack_forward_comm()` syncs `k_xshake` to the host, calls the read-only base
pack, then `k_xshake.modify_host()`.  `correct_coordinates()` sets
`xshake = x` and forces the host path, so this claims the shake host buffer on
a pack of atom coordinates; the next `unconstrained_update()` claims the device
side, both sides stand claimed (the failure of `0a6056c42`) and the
`k_xshake.sync<DeviceType>()` added in `5405f6f35` can copy the stale host copy
over the device one.  Compare `compute_ave_sphere_atom_kokkos.cpp:284` and
`pair_eam_kokkos.cpp:495`, which sync without claiming.

Fix: delete line 2166; the claim in `unpack_forward_comm` (`:2201`) is the right one.

### 1.15 `fix_cmap_kokkos.cpp:228-247` -- no virial, no per-atom energy/virial (CONFIRMED)

`FixCMAP` declares global and per-atom energy and virial and tallies every
crossterm with `ev_tally(nlist,list,5.0,E,vcmap)` (`fix_cmap.cpp:597`).
`FixCMAPKokkos::post_force()` calls `ev_init(eflag,vflag)` and reduces only the
global energy into `ecmap`; there is no `ev_tally` in the file.  With
`fix cmap/kk` the fix's global virial is zero (wrong pressure with
`fix_modify ... virial yes`) and `compute pe/atom` / `stress/atom` silently
omit the CMAP term.  Same class as `257b8364f` (shake/kk).

Fix: as in `257b8364f`: a device `ev_tally` into `k_eatom`/`k_vatom` allocated
after `ev_init(eflag,vflag,0)`, reduce `ev.v[6]` into `virial[]`, modify/sync
the dual views.

### 1.16 `pair_uf3_kokkos.cpp:743-745` -- centroid virial sized with the wrong counter (CONFIRMED)

`create_kokkos(k_cvatom, cvatom, maxvatom, "pair:vatom")` sizes the centroid
array with `maxvatom`, but `Pair::ev_setup()` grows `maxvatom` only under
`vflag_atom` and `maxcvatom` under `cvflag_atom`.  `pair uf3/kk` with
`compute centroid/stress/atom` and no `compute stress/atom` leaves
`maxvatom == 0`, so the guarded kernel writes `a_cvatom(i,0..8)` into a
zero-extent view: out-of-bounds heap writes.

Fix: size with `maxcvatom` (and label it `"pair:cvatom"`).

### 1.17 `pair_dpd_kokkos.cpp:426` -- host object dereferenced inside a device kernel (CONFIRMED)

`auto timestep = update->ntimestep;` sits in the team-policy
`operator()(TagDPDKokkos<NEIGHFLAG,EVFLAG>, team, ev)`.  `update` is the
`Pointers` member copied into the functor; on CUDA/HIP every thread
dereferences a host-resident `Update` object.  Reached only with
`package kokkos autotuning` (the team kernel), `neighflag == HALF` and
`evflag == 0`, which is why host CI never sees it.  Related (PLAUSIBLE): that
team kernel draws randoms from `saru` while the range kernel uses
`rand_pool.normal()`, so with autotuning the random forces come from a
different stream on thermo steps.

Fix: cache the timestep in a member in `compute()` next to `dtinvsqrt` and read
that in the kernel.

### 1.18 SNAP scratch requests hard-coded to level 0 (PLAUSIBLE)

`pair_snap_kokkos_impl.h:212-216`, `compute_sna_grid_kokkos_impl.h:234-238, 274-279`,
`compute_sna_grid_local_kokkos_impl.h:218-222` request
`team_size_compute_neigh * max_neighs` ints of level-0 scratch with no
`scratch_size_max(0)` query and no level-1 fallback: the shape that aborted
pace/kk with "Requested too much scratch memory on level 0" (issue #5063).
Fix: lift the `neigh_scratch_level_select()` logic from `pair_pace_kokkos.cpp`.
Minor: `scratch_size_helper` in those files types the scratch on
`Kokkos::DefaultExecutionSpace` instead of `DeviceType`.
### 1.19 `fix_propel_self_kokkos.cpp:78` -- `v_init(vflag)` with the default `alloc = 1` on a self-managed array (CONFIRMED)

The base allocates the plain `vatom`, then line 83 replaces the pointer with a
`create_kokkos` allocation (a leak of the plain data block on every first
allocation).  Worse, on any later step where `atom->nlocal > maxvatom`
(migration, `create_atoms`), `Fix::v_setup()` calls `memory->destroy(vatom)` on
the Kokkos-owned row pointers, which does `sfree(vatom[0])` on an address
inside the Kokkos host allocation: invalid free or heap corruption, and
`d_vatom` no longer aliases `vatom`, so `compute stress/atom` reads zeros.
Same defect `95cf1400b` fixed in efield/kk, shake/kk and wall/region/kk
(`fix_wall_region_kokkos.cpp:86` has the comment explaining `alloc = 0`).

Fix: `v_init(vflag,0)`.

### 1.20 `pair_pod_kokkos.cpp:1789` -- `sizeof(double)` byte count over a `KK_FLOAT` mirror (CONFIRMED, dead code)

`savematrix2binfile()` does
`fwrite(A.data(), sizeof(double) * (nrows*ncols), 1, fp)` where `A` is the
mirror of a `View<KK_FLOAT*>`: the shape fixed in `0ed8a1ba7`.  In single or
mixed precision it reads twice the allocation and writes a file that does not
match its own `double` header.  Only reachable from `savedatafordebugging()`,
whose sole call (`:308`) is commented out.  Fix: delete the three debugging
dumpers (`pair_pod_kokkos.h:219-221`) per the review rules on commented-out
debug code, or stage through a `double` array.

### 1.21 `pair_eam_kokkos.cpp:1184` -- HIP scratch sized with `sizeof(double)` for a `KK_FLOAT` view (CONFIRMED, performance)

`Kokkos::PerTeam(MAX_CACHE_ROWS*7*sizeof(double))` feeds a
`View<KK_FLOAT*[7], scratch_memory_space, Unmanaged>` (lines 868 and 971).
Over-allocation, so no corruption, but twice the LDS per team and the matching
occupancy loss on AMD in single/mixed.  Fix: `sizeof(KK_FLOAT)`.

### 1.22 `compute_temp_profile_kokkos.cpp:333` -- wrong side of a TransformView synced (PLAUSIBLE)

`atomKK->k_mass.sync<LMPHostType>()` followed by `double *mass = atom->mass`.
On a TransformView the templated `sync<LMPHostType>()` refreshes the `KK_FLOAT`
host mirror, not the legacy `double` array `atom->mass` points to
(`kokkos_type.h:1077-1081`); they alias only in a double build.  Harmless today
because per-type masses are only ever written on the legacy side, but the only
such call in the package.  Fix: `atomKK->k_mass.sync_host()`.

## 2. Leaks and housekeeping

### 2.1 `pair_uf3_kokkos.cpp:62-75` -- centroid array never freed (CONFIRMED)

The destructor calls `destroy_kokkos` for `k_eatom`, `k_vatom` and `k_cutsq`
but only `cvatom = NULL` for the centroid array, so the row-pointer block from
`create_kokkos` leaks and `~Pair()` sees a null pointer.  Fix:
`memoryKK->destroy_kokkos(k_cvatom, cvatom)`.

### 2.2 `fix_addtorque_atom_kokkos.cpp:97`, `fix_settorque_atom_kokkos.cpp:95` -- base allocation orphaned (CONFIRMED, tiny)

The base constructors do `maxatom = 1; memory->create(storque, 1, 3, ...)`.
The first KOKKOS `post_force()` calls `memoryKK->destroy_kokkos(k_storque, storque)`
on that plain array, which frees only the row-pointer block
(`memory_kokkos.h:272-281`) and orphans the three doubles.  `fix addforce/kk`,
`setforce/kk` and `efield/kk` avoid this with a `memory->destroy()` in their
constructors before the dual view takes over; do the same here.

### 2.3 `fix_rigid_small_kokkos.cpp:802` -- `v_init(vflag)` with `alloc = 1` on a self-managed array (PLAUSIBLE, latent)

`k_vatom` is a tied dual view grown in `grow_arrays()`; currently
`maxvatom = atom->nmax >= nlocal`, so `Fix::v_setup()`'s regrow branch never
fires, but if it did `memory->destroy(vatom)` would free Kokkos-owned memory.
Fix: `v_init(vflag,0)`.

### 2.4 Stale `k_eatom` claims in the granular pair styles (cosmetic)

`pair_gran_hooke_kokkos.cpp:97-100,176-177`, `pair_gran_hooke_history_kokkos.cpp:115-117,239-240`,
`pair_gran_hertz_history_kokkos.cpp:115-117,239-240` allocate, `modify<Device>()`
and `sync_host()` `k_eatom` although nothing writes `d_eatom` (no pair energy):
the stale claim retired in `a06c0e2c3`; results are zeros as expected, only
wasted copies and a false claim under `KOKKOS_DEBUG_SYNC`.

### 2.5 Standards nits

- `MPI_Waitall(..., MPI_STATUS_IGNORE)` at `comm_tiled_kokkos.cpp:168, 528, 646`
  and `remap_kokkos.cpp:202` should be `MPI_STATUSES_IGNORE` (same value in
  OpenMPI, MPICH and the STUBS since `c34cb6e93`, so no runtime effect).
- `comm_tiled_kokkos.cpp:486, 605` capture the raw `buf_send_pair`/`buf_recv_pair`
  pointer once before the swap loop and then call `pack_forward_comm_kokkos`
  inside it; `pair_bondval_vec_kokkos.cpp` resizes the dual view inside its
  pack, which would free the captured allocation.  Not reachable today because
  `grow_buf_pair` already sizes for `nsize*sendnum`; the brick path re-reads the
  pointer after the pack (`comm_kokkos.cpp:418`) and the tiled one should too.
- `region_ellipsoid_kokkos.h:132-155`: `rotate()` declares locals `a[3], b[3],
  c[3]` that shadow the `RegEllipsoid` semi-axis members `a, b, c`.  The body
  never reads the members today, so results are right, but it is the shadowing
  shape of `8da0b4b53`; rename the locals.
- `NBinKokkos`, `NBinSSAKokkos` and `MLIAP_SO3Kokkos` launch `*this` with no
  `copymode` at all.  Safe today (their destructors free nothing they allocate),
  but one `memory->create` in an `NBin` path reproduces `f912fb14b`; a
  `copymode` guard is cheap insurance.

## 3. clang -Wall -Wextra

Serial backend (clang 18 has no libomp here), all packages of
`cmake/presets/kokkos-packages.cmake` except COLVARS, ML-PACE from a local
clone, 375 KOKKOS package objects per precision.  In addition every
`src/KOKKOS/*.cpp` was syntax-checked with g++ under
`-DLMP_KOKKOS_SINGLE_SINGLE` and `-DLMP_KOKKOS_SINGLE_DOUBLE`, and the 98
styles ported since April were fully compiled at `-O2` in single precision:
no errors, so nothing else in the package is double-only after `0ed8a1ba7`.

### 3.1 double precision: compiles, 1076 warnings, none a bug

| category | count | assessment |
|---|---|---|
| `-Wunused-parameter` | 1054 | unnamed-parameter noise from the functor interfaces |
| `-Wsign-compare` | 11 | `int` loop index against `extent()`/`size()`: `atom_vec_kokkos.cpp:2976-3023`, `comm_kokkos.cpp:1162`, `pair_uf3_kokkos.cpp:789`; benign |
| `-Wunused-variable` | 4 | see below |
| `-Wunused-const-variable` | 3 | `dihedral_table_kokkos.cpp:39-41` (`TOLERANCE`, `SMALL`, `SMALLER`) |
| `-Wunused-but-set-variable` | 3 | see below |
| `-Wmismatched-tags` | 1 | see below |

Worth cleaning up:

- `fix_shake_kokkos.cpp:522` -- `i0` is unused since the angle statistics fix
  in `fa275ab5f` dropped the central atom from the count.  Delete the line.
- `fix_qeq_reaxff_kokkos.h:339` -- `FixQEqReaxFFKokkosNeighborFunctor`
  forward-declared as `struct`, defined as `class`.  clang notes this can
  break linking under the MSVC ABI, which LAMMPS supports; make the two agree.
- `comm_kokkos.cpp:1183` (`count_bonus`), `comm_kokkos.cpp:1541` (`mlo`,
  `mhi`: assigned in the `mode != SINGLE` branch of `borders_device()`, but
  multi mode falls back to the legacy path before reaching it, so dead),
  `pair_uf3_kokkos.cpp:886` (`v_f`: the per-atom virial is tallied inside
  `ev_tally`/`ev_tally3`, which take their own scatter access), and
  `npair_kokkos.cpp:157` (`nbor_chunk_size`, consumed only in the GPU block at
  line 317).  Dead locals, no missing logic behind any of them.

### 3.2 single and mixed precision

(pending)

## 4. Searched and found clean

- `atomKK->modified()` placed before the kernel that writes the data (shape of
  `2460560dd`, `b65be4720`): every remaining occurrence in the fixes and
  computes is the legitimate bracket around a bias compute or a claim after a
  communication.
- Host loops reading `d_mask` or other device views (shape of `12d299199`): none
  left.
- Atom views cached in `init()` that reneighboring can reallocate: the two
  remaining cases (`fix_wall_flow_kokkos.cpp:61`, `fix_eos_table_rx_kokkos.cpp`)
  use the views only inside the same function.
- Bare-integer-count `parallel_for`/`parallel_reduce` with a lambda in a class
  that is also instantiated for `LMPHostType`: only the ML-IAP files (1.5).
  Functor launches are fine because every functor carries a `device_type`
  typedef that Kokkos uses to pick the execution space.
- Extended lambdas inside `private:`/`protected:` member functions (the nvcc
  error fixed in `min_sd_kokkos.h`): none left.
- Other `create_kokkos(k_eatom/k_vatom/k_cvatom)` without a preceding
  `destroy_kokkos`, destructors missing `destroy_kokkos`, `ev_init`/`v_init`
  without `alloc = 0` in styles that own the dual views (the hybrid styles
  legitimately keep `alloc = 1`): only the entries above.
- `ev_tally` bodies of the newly ported table styles (`angle_table`,
  `bond_table`, `dihedral_table`, `dihedral_table_cut`) match their harmonic/opls
  templates including the `newton_bond` branches; `bond_table_kokkos.cpp:30`
  correctly uses `LINEAR == 1` for the bond enum and 0 for angle/dihedral.
- `read_restart()` shadowing (shape of `8da0b4b53`): the remaining same-named
  locals in `pair_sw/tersoff/vashishta` (`k_map`, `k_elem3param`) and
  `bond_quartic_exp`/`dihedral_charmm` (`k_lj14_*`) have no same-named member.
- Host-only code inside `KOKKOS_INLINE_FUNCTION` bodies: only 1.17.
- `grid3d_kokkos` (brick and tiled), `fft3d_kokkos`, `pppm_kokkos`,
  `pppm_tip4p_kokkos`, the `comm_kokkos` fix/pair/compute device paths,
  `fix_qeq_reaxff_kokkos`, `fix_acks2_reaxff_kokkos`, `pair_eam/adp/meam/eam_he_kokkos`
  comm packing: request/wait pairing, pack strides and sizes agree.  None of the
  styles ported since April implement comm of their own.
- `copymode` guards: every `Pair`/`Fix`/`Compute`/`Region` class that launches
  `*this` and frees memory in its destructor is guarded.
- Region `match_kokkos`/`surface_kokkos` of all seven region styles against
  `Region::match`/`surface` including the `domain->remap` ordering: consistent.
  The five new region styles (cone, cylinder, plane, prism, ellipsoid) already
  carry the post-`26fbb5759` shape (`boxremap.capture()` in `prematch()` and
  `match_all_kokkos()`, `inverse_transform` for dynamic regions, `openflag`);
  each `k_inside()` matches the CPU `inside()` (the `c02a80bef` cone hi-boundary
  fix is in the inherited parser); none implements `surface_kokkos()`, so the
  `67c3ddbfb` contact-list race cannot recur, and `fix wall/region/kk` rejects
  every region other than block/kk and sphere/kk.
- `atom_vec_*_kokkos` constructors: every cached per-atom pointer is now in the
  initializer list.
- CPU parity of the ported fixes/computes (`fix_heat`, `flow_gauss`,
  `propel_self`, `damping_cundall`, `viscous_sphere`, `store_force`, `aveforce`,
  `oneway`, `wall_piston`, `min_quickmin`, `compute temp/region`, `temp/ramp`,
  `temp/sphere`, `gyration`, `centro/atom`, `entropy/atom`, bond/angle table
  lookups): reductions, group and region handling, unit factors and table
  interpolation match the base classes.
