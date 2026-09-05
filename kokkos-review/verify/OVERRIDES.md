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
