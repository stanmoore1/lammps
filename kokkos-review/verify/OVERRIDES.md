# Orchestrator overrides on verifier verdicts

Spot check of a stratified sample of CONFIRMED verdicts, re-derived from the source by the
orchestrator before any fix was applied.

## DO NOT APPLY — verifier verdict is wrong

**F0249 / F0481 — `src/KOKKOS/mliap_unified_kokkos.cpp:144` "double Py_DECREF".**
Both verifiers marked this CONFIRMED and recommend deleting the `Py_DECREF` from
`~MLIAPDummyModelKokkos`.  **That fix would introduce a leak.**  Both verifiers read only
the C++ and counted "one Py_INCREF at build_unified_kokkos:252-253".  They missed that
`mliap_unified_connect_kokkos` is Cython-generated and returns a NEW reference
(`__Pyx_INCREF` on the return path, from `mliap_unified_couple_kokkos.pyx:492-549`), which
`build_unified` never releases.  The real count is 3 increfs vs 3 decrefs — balanced by
accident.  Correct fix: keep the two explicit `Py_INCREF`s, ADD the missing `Py_DECREF`
for the value returned by `mliap_unified_connect_kokkos()`, correct the wrong "Borrowed
references" comment, and only then remove the duplicate at :144.  Establish this in one
edit or not at all; a partial fix is worse than the current state.

Lesson: a verdict that depends on evidence outside the two C++ files being diffed
(generated code, Python, build system) needs that evidence actually consulted.

## Confirmed sound on independent re-derivation

- **F0022** `atom_vec_kokkos.cpp:1845` — `DEFORM_VREMAP` branch has no `else`, so `m` is
  not advanced for atoms outside the deform group. Verified earlier by the orchestrator too.
- **F0042** `bond_fene_expand_kokkos.cpp:191` — `d_flag() = 2` / `= 1` are plain unordered
  stores to the same 0-d view; a later thread writing 1 can mask a 2, downgrading
  `error->one("Bad FENE bond")` to a warning. Post-kernel dispatch at :133-138 confirms
  the severity ordering matters.
- **F0238** `fix_wall_gran_kokkos.cpp:227` — CPU `fix_wall_gran_old.cpp:901` has
  `if (limit_damping && (ccel < 0.0)) ccel = 0.0;`; the KOKKOS kernel has no such line.
- **F0424** `pair_uf3_kokkos.cpp:1647` — CPU `pair_uf3.cpp:1857` does
  `fforce = factor_lj * force_2b;`. The KOKKOS `single()` scales only the returned energy
  (`return factor_lj * value;`) and never scales `fforce`.
- **F0091** `compute_temp_deform_kokkos.cpp:186` — KOKKOS brackets the kernel with
  `domainKK->x2lamda(nlocal)` / `lamda2x(nlocal)`, mutating `atom->x` in place for every
  local atom. The CPU style uses the per-atom two-argument form
  `domain->x2lamda(atom->x[i], lamda)` writing into a local (`compute_temp_deform.cpp:313`)
  and never touches `atom->x`. Genuine divergence: round-trip drift plus a dirtied X.

Sample result: 4 of 5 sound, 1 wrong. Fix phase must re-read the code at each site rather
than applying a verdict's `fix` string blind.

## Update after wave 1 (274 verdicts)

**F0249 / F0481 has now been CONFIRMED by three independent verifiers** (batches 00, 07
and 08).  The override still stands.  All three read only the C++ and counted the two
explicit `Py_INCREF`s in `build_unified_kokkos`; none opened the Cython-generated code for
`mliap_unified_connect_kokkos`, which carries a third `__Pyx_INCREF` on its return path
(`mliap_unified_couple_kokkos.pyx:492-549`).  Agreement among verifiers who share a blind
spot is not corroboration.  Do not delete the `Py_DECREF` at `mliap_unified_kokkos.cpp:144`
on its own.

Batch 08 adds a supporting detail that is easy to misread as confirmation: the sibling
descriptor destructor has the same call *deliberately commented out*.  That is consistent
with the balanced-by-accident reading, not with a double release.

## Checkpoint contamination (batch 12 / batch 13)

A shared helper script in the scratchpad was overwritten by a sibling verifier mid-run, so
16 batch-12 verdicts were appended to `progress_v13/` instead of `progress_v12/`.  The
batch-12 agent recovered them into its own directory and deliberately did NOT rewrite
`progress_v13/` (which was still being written by a live sibling).

Consequence: `progress_v13/verdicts.jsonl` contains 16 ids that do not belong to batch 13 —
F0124, F0125, F0126, F0370, F0373, F0525, F0178, F0294, F0514, F0097, F0235, F0371, F0087,
F0141, F0265, F0399.  They are duplicates of the recovered batch-12 entries, not lost work.

**Any aggregation over `progress_v*/verdicts.jsonl` must deduplicate by finding id.**  The
per-batch `verdicts_NN.json` files are unaffected and are the authoritative record.

## Reachability correction: pair_vashishta_kokkos FullA/FullB is DEAD CODE

Findings F0421 / F0503 (and the orchestrator's own verification) reported the FullB
3-body indexing bug — `d_numneigh_short_3body[jj]` / `d_neighbors_short_3body(jj,kk)` using
the slot counter instead of the neighbour atom `j` — as a live, high-severity
silently-wrong-forces defect "reached with neighflag FULL, the GPU default".

**That reachability claim is wrong, established by running it, not reading it.**  Two
guards cannot be satisfied at the same time:

* `KokkosLMP::newton_check()` (`src/KOKKOS/kokkos.cpp:859-860`) — `neigh full` requires
  `newton off`;
* `PairVashishta::init_style()` (`src/MANYBODY/pair_vashishta.cpp:268-269`) — errors with
  "Pair style Vashishta requires newton pair on", and
  `PairVashishtaKokkos::init_style()` calls the base first.

Observed directly:
  `-pk kokkos neigh full newton off` -> ERROR: Pair style Vashishta requires newton pair on
  `-pk kokkos neigh full`            -> ERROR: Must use 'newton off' with ... 'neigh full'

So `neighflag == FULL` is unreachable for this style and the FullA/FullB kernels are dead
code.  The indexing defect is real but **latent**, not live; severity drops from high to
low.  The fix on this branch is kept because it makes the dead code correct and documents
the intent, but it should not be described as fixing wrong forces.

Lesson: three verifiers confirmed the code defect correctly and none checked reachability
through the init_style chain.  "It's the GPU default" was assumed from
`kokkos.cpp:345-348` without checking whether the style's own init_style permits it.  Every
other finding whose severity rests on "FULL is the GPU default" needs the same check —
notably pair_brownian (F0309) and pair_multi_lucy_rx (F0370).

## Reachability sweep result — severities recalibrated package-wide

A dedicated sweep established the governing rule and applied it to every style whose
severity rested on a neighflag/newton configuration.  Full table in `reachability.md`.

**C1: `neighflag == FULL` implies `newton_pair == 0`, always.**  `package kokkos` runs
unconditionally at startup (`lammps.cpp:916`), so `newton_check()` (`kokkos.cpp:859-860`)
always fires, and `input.cpp:1729` re-runs it after any later `newton` command.  Every
`FULL && newton_pair` branch in the package is therefore dead code — a list of 15 such
sites is in `reachability.md`.

**C4 is the decisive test for a style:** does its KOKKOS `init_style()` call a CPU base
`init_style()` that errors on newton off, before the neighbor request is adjusted?  If so,
FULL is unreachable for that style even though FULL is the GPU default.

UNREACHABLE (findings in these FULL paths are latent, not live): vashishta, uf3, mliap,
tip4p variants, sw/sw-mod, all four tersoff variants, meam, snap, reaxff, pace,
pace/extrapolation, pod, all four dpd variants, all three granular styles, fix shake.

REACHABLE and the GPU default: brownian (base only WARNS at
`COLLOID/pair_brownian.cpp:445-446`), multi/lucy/rx (no `init_style` override at all), and
the plain pair family (lj/cut, coul/*, born/*, buck/*, eam*, adp, table, exp6/rx, table/rx,
dpd/fdt/energy, ylz, colloid, bondval*, lj/cut/dipole/cut).

IMPORTANT EXCEPTION: `fix qeq/reaxff/kk` and `fix acks2/reaxff/kk` read `neighflag_qeq`,
which `newton_check()` never constrains and `neigh half` does not change.  There FULL
coexists with newton pair ON, so C1 does not apply and findings in those kernels must not
be dismissed.

Also severity-lowering: every KOKKOS many-body style can only run on a GPU after an
explicit `package kokkos neigh half newton on`, so a finding in one of their kernels is
never "the GPU default path".
