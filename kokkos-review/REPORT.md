# KOKKOS package pre-release code review

Line-by-line review of the complete `src/KOKKOS` package (799 files, ~240,000 lines,
~412 style units) against the corresponding CPU base classes, carried out ahead of the
next LAMMPS stable release.

**Result: 546 findings — 176 high, 151 medium, 219 low.**  76 of them were independently
re-verified by reading the code directly (see `findings_orchestrator.json`, every entry
carries a `verified_by` field naming the exact lines compared).

## Method and coverage

The package was partitioned into 30 file groups plus 4 orthogonal package-wide rule
audits.  Each group was reviewed by a separate agent that read every assigned file end to
end and diffed it against its CPU base class, checkpointing after every style so that a
usage-limit interruption cost at most one file.  Coverage is complete: every group
reported `COVERAGE: all files read completely`.

* `group_NN.txt` — the file partition.
* `progress_NN/` — per-group checkpoints (`done.txt`, `findings.jsonl`, `notes.txt`).
  The `notes.txt` files also record candidates that were investigated and **dismissed**,
  with the reasoning — worth reading before re-litigating a pattern.
* `findings_NN.json`, `findings_rules_[A-D].json` — per-group / per-audit results.
* `findings_orchestrator.json` — the 76 independently verified findings.
* `all_findings.json` — everything merged.

Each finding carries `severity`, `confidence` (`confirmed` / `likely` / `possible`),
`evidence` (file:line on both the KOKKOS and the CPU side), `failure_scenario` and
`suggested_fix`.

**Confidence caveat.** `confirmed` means the reviewer quoted both sides of the divergence;
it does not mean the bug was reproduced in a run.  Nothing here was compiled or executed.
The 76 verified entries were re-derived from the source a second time by a different
reader; the rest should be spot-checked before acting.  Note also that one group's line
numbers were found to be offset by ~70 lines — **locate findings by symbol, not by line
number.**

---

## Tier 1 — memory safety and crashes

These write out of bounds, dereference null, or divide by zero on documented, reachable
inputs.

| Where | What |
|---|---|
| `pair_snap_kokkos_impl.h:554` | **No `j &= NEIGHMASK` anywhere in the file.** `npair_kokkos.cpp:500` stores `j ^ (which << SBBITS)`, and `neighbor.cpp:557` forces `special_flag[1..3]=2` whenever *any* kspace style is defined. So `hybrid/overlay snap/kk … coul/long/kk` on a molecular system reads `x(j,·)`/`type(j)` at `j|(which<<30)` and later writes `a_f(j,·)` there. CPU masks at `pair_snap.cpp:143`; sibling `pair_sw_kokkos.cpp:234` masks. |
| `pair_reaxff_kokkos.cpp:1596` | `TagPairReaxZero` writes `d_hb_num(n)` unconditionally, but it is allocated only under `cut_hbsq > 0` (`:1530`). `hbond_cutoff 0.0` — the documented way to disable H-bonds — gives a null-pointer write in the first kernel launch. Every other use of `d_hb_num` is guarded. |
| `pair_mliap_kokkos.cpp:100` | The per-atom virial block sizes and size-checks `k_vatom` with `maxeatom`, not `maxvatom` (and labels it `"pair:eatom"`). `compute stress/atom` without a per-atom energy compute leaves `maxeatom == 0`, the guard `0 < 0` fails, `vatom` is never allocated, and `v_tally` writes through null. |
| `pair_brownian_kokkos.cpp:114` | `ev_init(eflag,vflag,0)` with no `eatom` allocation. `Pair::ev_setup` still raises `maxeatom`, and `compute_pe_atom.cpp:132` reads `force->pair->eatom` unguarded. |
| `pair_pod_kokkos.cpp:254` | Integer division by zero in `divideInterval()` when a rank has `inum == 0` (more ranks than atoms, or transient load imbalance). |
| `pair_uf3_kokkos.cpp:1622` | `single()` runs the knot search **before** any cutoff guard, then gates on `r < d_cutsq(…)` — a distance against a *squared* cutoff. `pair_write` past the cutoff walks off the end of `d_n2b_knot`; a 5 Å cutoff admits r up to 25 Å. |
| `atom_map_kokkos.cpp:178` | `map_set_device()` reallocates `sametag` **before** the `map_init(0)` that can free it; the function's own comment states the opposite order is required. |
| `atom_map_kokkos.cpp:404` | `map_one()` sets `modify_host()` while the device path calls `modify_device()`, tripping `dual_hash_type`'s `Kokkos::abort("Concurrent modification of host and device hashes")`. |
| `fix_property_atom_kokkos.cpp:84` | The destructor calls `atomKK->update_property_atom()`, which rebuilds `fix_prop_atom[]` from `modify->get_fix_by_style(…)` — and `Modify::delete_fix` deletes the object at `modify.cpp:1075` *before* unlinking it at `:1078`. Every later `atomKK->sync()` then follows a dangling pointer. |
| `fix_wall_gran_kokkos.cpp:371` | `pack_exchange` derives the send-buffer offset from the atom's **local index** rather than the send-list slot, while the buffer is only `nsend*size_history` long: uninitialized bytes transmitted, and the write can run past the end. |
| `mliap_unified_kokkos.cpp:144` | Double `Py_DECREF` — both the Kokkos destructor and its base decrement the same reference. The sibling descriptor destructor at `:44` has exactly this call commented out, which is the intended fix. |
| `compute_sna_grid[_local]_kokkos_impl.h` | `max_neighs` hard-coded to 100 with unbounded writes into `snaKK.rij`; element views sized `nelements` but filled to `ntypes`. |
| `pair_reaxff_kokkos.cpp:4403` | `FindBondSpecies` writes `d_tmpid(i,nj)` *before* checking `nj > MAXSPECBOND`, and the error is only acted on after the whole kernel finishes — so it keeps corrupting later rows. |

## Tier 2 — silently wrong physics

Runs complete and look plausible; the numbers differ from the identical CPU run.

| Where | What |
|---|---|
| `pair_dpd_ext_tstat_kokkos.cpp:302` | **Sign error.** CPU: `fpair = -factor_dpd*gamma*wdPar²*dot*rinv`. Kokkos drops the minus, turning the thermostat's drag into an energy-pumping anti-drag. (`pair_dpd_ext_kokkos.cpp:301` is correct because it accumulates onto a conservative term first.) |
| `pair_lj_cut_dipole_cut_kokkos.cpp:401` | The total force and the whole force/torque accumulation sit **inside** the `if (rsq < cut_coulsq_ij)` block. For `cut_coul ≤ r < cut_lj` the LJ energy is tallied but the force and virial are zero. Duplicated in the team kernel. |
| `pair_vashishta_kokkos.cpp:510` | The `FullB` kernel indexes the short 3-body list with the *slot counter* `jj` instead of the neighbor atom `j` — the local is even named `j_jnum`, and both legs are measured from `j`. Reached only with `neighflag FULL`, i.e. the GPU default, which CPU-only CI never exercises. |
| `pair_reaxff_kokkos.cpp:1260` | Non-shielded vdW derivative **multiplies** by `rij` where it must divide. Confirmed three ways: the CPU (`reaxff_nonbonded.cpp:165`) divides, and so does the table generator *in the same Kokkos file* (`:669`). Energies stay right, which hides it. |
| `pair_coul_dsf_kokkos.cpp:260`, `pair_lj_cut_coul_dsf_kokkos.cpp:223` | `special_coul` folded into the prefactor instead of subtracting `(1-factor_coul)*prefactor`. With the default `special_bonds coul 0 0 0`, Kokkos contributes exactly zero for 1-2/1-3/1-4 pairs where the CPU contributes `-prefactor`. `pair_born_coul_dsf_kokkos.cpp:205` is the correct sibling. |
| `pair_coul_cut_kokkos.cpp:230`, `pair_coul_debye_kokkos.cpp:253` | `init_one()` fills only `(i,j)` of the standalone `k_cut_ljsq`/`k_cut_coulsq` views. Above 12 atom types the non-stack path tests `rsq` against 0 for every `itype>jtype` pair and drops its Coulomb interaction entirely. |
| `npair_kokkos.h:261`, `npair_ssa_kokkos.h:175` | `xprd_half/yprd_half/zprd_half` declared **`const int`**, initialized from `double domain->xprd_half`. They feed the float comparison in `minimum_image_check()`, which decides special-bond coding — so any box with a non-integer half-length makes Kokkos and CPU disagree about which pairs are special-bonded. |
| `npair_skip_kokkos.h:106` | `cutsq_custom` is an `int`: a 3.5 Å trim cutoff becomes 12 instead of 12.25, and any cutoff below 1.0 empties the list. Sibling functors declare it `double`. |
| `npair_skip_kokkos.cpp:104` | `build()` writes `d_ilist` on the device but never claims `k_ilist.modify<DeviceType>()`; all four sibling npair styles do. The `sync_host()` in `NPairCopyKokkos`/`NPairTrimKokkos`/`fix reaxff/species` is a no-op. |
| `npair_halffull_kokkos.h:221` | `halffull/newton/tri/trim/skip/kk/device` registered with the **TRI=0** class while its mask carries `NP_TRI`. The host sibling at `:225` is correct. |
| `pppm_kokkos.cpp:2418` | The per-atom forward unpack reads `d_buf[7*i]` for `u_brick` while all six virial reads add `unpack_offset`. Breaks per-atom Coulomb energy under `comm_style tiled`. |
| `pppm_kokkos.h:486` | No `reset_grid()` override, so `fix balance` drives `PPPM::reset_grid()`, which dereferences the base `gc` (always null for the Kokkos style) and calls the non-virtual `PPPM::compute_rho_coeff()` on never-allocated pointers. |
| `pair_table_rx_kokkos.cpp:671` | On non-energy steps only `F_MASK` is claimed, though the kernel writes `uCG`/`uCGnew` unconditionally. Both sibling RX styles keep `UCG_MASK\|UCGNEW_MASK` there, so the DPD internal-energy integration loses its pair contribution on every step without output. |
| `sna_kokkos_impl.h:1118` | With `bzeroflag=1, chemflag=1, wselfallflag=0`, `bzero[j]` is subtracted from **every** diagonal triple rather than only `ielem`. |
| `fix_shake_kokkos.cpp:539` | `min_post_force` applies the restraint force to a possibly-ghost index and the following `reverse_comm` is a no-op, so ghost contributions are dropped while the matching energy is kept. |
| `fix_spring_self_kokkos.cpp:105` | Unconditional `k_xoriginal.modify_host()` + `sync<DeviceType>()` declares the never-updated host mirror newer, overwriting the device copy the exchange kernels just wrote. |
| `fix_wall_gran_kokkos.cpp:424` | `unpack_exchange` writes `d_history_one(i,·)` instead of `d_history_one(index,·)`; shear history lands on the wrong atoms. |
| `fix_wall_gran_kokkos.cpp:107` | `pairstyle GRANULAR` and `wallstyle REGION` fall through the if/else chain with neither a kernel nor an error — the wall exerts **no force at all**. |
| `improper_cvff_kokkos.cpp:278` | The device kernel reads the host pointer `sign[type]`; `:247` correctly uses `d_multiplicity[type]`. |
| `bond_quartic_kokkos.cpp:176` | `k_brokenflag` written on device, never marked modified — **no bonds break on GPU**. |
| `angle/dihedral/improper_hybrid_kokkos` | All three keep the inherited `centroidstressflag = CENTROID_AVAIL` although every Kokkos sub-style sets `CENTROID_NOTAVAIL`, giving a null `cvatom` dereference with centroid stress. |
| `comm_tiled_kokkos.cpp:233` | The generic self-send branch packs but never unpacks (`n` assigned and unused); the velocity branch at `:195` does both. |
| `atom_vec_kokkos.cpp` | Five separate defects: `PackBorderVel` missing the non-deform branch (`:1848`), `PackCommVel` never initializing `_mu`/`_sp` (`:809`), `unpack_border_vel_kokkos` branching on `ncomm_vel` instead of `nborder_vel` (`:2131`), `field2size("num_improper")` using `dihedral_per_atom` (`:2956`), and no `RADIUS`/`RMASS` in any forward-comm functor although `atom_style sphere` with `radvary` requires it (`:118`). |

## Tier 3 — recurring families

These are the patterns worth a targeted sweep rather than a one-line fix, because each
recurred across styles that were created by copy-adapt.

**1. `scale[][]` dropped, truncated, or mis-indexed.**  Four independent instances:
`meam_kokkos.h:65` stores `d_scale` in an **integer** view *and* indexes it `(type[i],type[i])`
*and* omits the CPU's `dUdrij *= scaleij` block; `pair_pace_extrapolation_kokkos.cpp:1429`
uses the diagonal element for the force where the CPU reserves the diagonal for the
*energy*; `pair_eam_kokkos.cpp` never references `scale[][]` at all; `pair_coul_long[_cs]_kokkos`
likewise.  In every case `fix adapt … scale` silently does nothing (or the wrong thing)
under `-sf kk`.  Additionally `pair_meam_kokkos.cpp:299` uploads `d_scale` only from
`coeff()`, so even a correct value never reaches the device after `init_one` symmetrizes it.

**2. `int` where a `double` is required.**  Beyond the neighbor-list cases in Tier 2:
`mliap_so3_kokkos.cpp:758` (`int weight = t_wjelem[jelem]`), `fix_eos_table_rx_kokkos.h:93`
(table temperature bounds in integer views), `meam_kokkos.h:65`.  Worth a grep for
`int` locals assigned from `double` views across the package.

**3. `Pair::special_lj` is `int[4]`** (`pair.h:247`, commented "copied from
`force->special_lj` for Kokkos").  Most `/kk` styles shadow it with a `KK_FLOAT` member.
A package sweep found exactly four that do not — `pair_gran_hooke_kokkos`,
`pair_gran_hooke_history_kokkos`, `pair_gran_hertz_history_kokkos` and
**`pair_table_kokkos`** — plus `pair_table_rx_kokkos.cpp:679`, which uses a `Few<int,4>`.
`pair_table/kk` dispatches through `PairComputeFunctor`, which reads
`c.special_lj[sbmask(j)]` at `pair_kokkos.h:156`, so `special_bonds lj 0.5` becomes 0 and
those pairs are dropped entirely.  Fixing the base member's type would close all five at once.

**4. Self-energy computed on the host from `atom->q[i]`** while only the device copy was
synced: `pair_born_coul_{dsf,wolf}[_cs]_kokkos` (4 files), `pair_lj_cut_coul_wolf_kokkos.cpp:120`,
`pair_lj_cut_coul_dsf_kokkos.cpp:131`.  The intended idiom computes the self-energy inside
the kernel from the device view — see `pair_coul_dsf_kokkos.cpp:231`.

**5. `atomKK->k_mass` used on the device without a sync.**  There is no `MASS_MASK`, so it
must be synced explicitly (the idiom is `fix_nve_kokkos.cpp:45-46`).  Missing in
`fix_nve_limit`, `fix_dt_reset`, `fix_electron_stopping`, `fix_shake`, `fix_gravity`,
`fix_momentum`, `fix_shardlow` — see `findings_rules_C.json` for the full list.

**6. `eflag_global` used where `eflag_either` is meant**, so per-atom energies come out
zero on timesteps that request only per-atom energy: `pair_dpd_kokkos.cpp:370`,
`pair_lj_cut_dipole_cut_kokkos.cpp:307`, and the mirror-image
`if (eflag)` vs `if (eflag_global)` in `pair_table_kokkos.cpp:177` and
`pair_table_rx_kokkos.cpp:810`.

**7. TIP4P hydrogen-type check dropped**: `pppm_tip4p_kokkos.cpp:115` and
`pair_tip4p_kokkos.h:106` both resolve `tag+1`/`tag+2` without verifying `type == typeH`,
so a mis-ordered topology that the CPU rejects with a clear error silently builds M sites
from unrelated atoms.

**8. Compute-group filter dropped in grid computes**: `compute_sna_grid`,
`compute_sna_grid_local`, `compute_gaussian_grid_local` all include atoms outside the
compute's group; `compute_coord_atom_kokkos.cpp:279` applies the second-group filter only
in the single-column branch.

## Silently ignored user options

Each of these accepts a documented keyword and then does not implement it — arguably worse
than an error, because the run looks successful:

* `fix gravity/kk` ignores `disable` (`fix_gravity_kokkos.cpp:43`; CPU returns early at `fix_gravity.cpp:276`).
* `fix langevin/kk` wipes the user's `scale <type> <value>` values by re-defaulting `ratio[]` in its constructor (`fix_langevin_kokkos.cpp:73`).
* `fix npt/kk` drops the entire `isochoric` branch of `FixNH::remap()` (`fix_nh_kokkos.cpp:354`).
* `fix nve/sphere/kk` ignores `update dipole/dlm` — no `dlm` handling exists in the file.
* `pair morse/kk` never subtracts the offset, so `pair_modify shift yes` is a no-op (`pair_morse_kokkos.cpp:177`).
* `pair reaxff/kk` hardcodes `thb_cutsq = 1e-5`, ignoring `thb_cutoff_sq` from the control file.
* All three granular pair styles ignore `fix rigid` body masses in `meff` (CPU substitutes `mass_rigid`, `pair_gran_hooke.cpp:149-162`).
* `pair nm/cut/coul/long/kk` is the only Kokkos `coul/long` style without the `CoulLongTable` specialization, so it ignores `pair_modify table`.
* `pair sw/kk` ignores `threebody off` (correct results, but the full O(n²) loop still runs).
* `pair gauss/kk` drops the documented `occupancy` pvector output.

## Defects in non-KOKKOS code found along the way

* **`src/EXTRA-PAIR/pair_lj_expand_sphere.cpp:122` and `:396`** — the pair force is formed as
  `forcelj * rshift / r`, but with `lj1 = 48εσ¹²`, `lj2 = 24εσ⁶` and `r6inv = 1/rshift⁶`,
  `forcelj = -rshift·dE/drshift`, so the correct scalar is `forcelj / (rshift·r)`.
  `src/pair_lj_expand.cpp:115` and `:409` use exactly that correct form for the identical
  energy expression.  Every force in `lj/expand/sphere` is therefore off by `rshift²` and
  does not match the tallied energy.  The Kokkos port faithfully copies the wrong base.
  **The force-style YAML fixtures for this style were generated from the current code and
  encode the wrong forces — they must be regenerated with the fix.**
* `src/compute_temp_sphere.cpp:106` — the `temp/region` bias is detected with an exact
  `strcmp` on `tbias->style`, which stops matching once the bias compute is promoted to
  its `/kk` variant.
* `src/ML-POD/pair_pod.cpp:107` — `~PairPOD()` lacks the mandatory `if (copymode) return;`
  guard required of any base class a Kokkos style inherits from.
* `src/pair_hybrid_scaled.cpp:263-276` — `single()`/`born_matrix()` index `scaleidx`/`atomvar`
  with the loop counter instead of `map[itype][jtype][m]`; the Kokkos copy inherits it.

## Suggested triage order

1. The Tier 1 table — memory safety, all on reachable inputs.
2. `pair_dpd_ext_tstat` sign error, `pair_lj_cut_dipole_cut` dropped forces,
   `pair_vashishta` FullB indexing, `pair_reaxff` vdW derivative — wrong forces in
   styles people run.
3. The five `special_lj` styles and the `scale[][]` family — both are one-line-per-site
   fixes with a shared root cause.
4. The `int`-vs-`double` neighbor-list members (`npair_kokkos.h`, `npair_ssa_kokkos.h`,
   `npair_skip_kokkos.h`) — trivial fixes, package-wide effect on special-bond handling.
5. The silently-ignored options — each needs either an implementation or an explicit
   `error->all()` in `init_style()`.

## What was checked and deliberately *not* reported

Several package-wide patterns were investigated and dismissed with reasons recorded in the
`progress_NN/notes.txt` files.  The main ones:

* **Missing `copymode = 1` around `pair_compute<>`** (13 styles).  Benign:
  `~PairComputeFunctor` (`pair_kokkos.h:103`) sets `c.copymode = 1` on the copy before its
  destructor runs, and `pair_virial_fdotr_compute` makes no style copy.
* **`if (eflag)` instead of `if (eflag_global)` around `eng_vdwl/eng_coul +=`** in ~30 coul
  styles — `eng_coul` is not read when `eflag_global == 0` and is reset by `ev_init` on the
  next energy step.  (The two `pair_table` cases above are listed only because they were
  reported separately; the same reasoning applies.)
* **`lj/class2/coul/long/cs` EPS_EWALD_SQR handling** — relative difference ~1e-12 at normal
  separations, and the r≈0 core/shell case is removed by `factor_lj`.
