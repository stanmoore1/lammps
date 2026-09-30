# Subagent prompt template

This is the prompt skeleton handed to each parallel subagent (one batch of
1-4 files per agent).  It is derived verbatim from the last-phase prompts of
the original session (stanmoore1/private:lammps/kokkos-precision/history/agents/all_agent_calls.jsonl, seq 19163 and
19191), with the batch-specific parts replaced by `<PLACEHOLDERS>` and the
scratchpad paths replaced by the toolkit paths.  The "other agents" and "no
git state" paragraph was implicit in the original setup (the main thread
committed every batch through the gate script) and is spelled out here.

Fill in:
- `<REPO>`: absolute path of the LAMMPS checkout the agents edit
- `<TOOLS>`: `<REPO>/tools/kokkos-precision` (or wherever this toolkit is)
- `<FILES>`: one line per assigned file with the baseline counts from
  `survey.sh` as `(single/mixed)`
- `<DOMAIN HINTS>`: what the style does and which patterns to expect
  (RNG narrowing, base-class members needing `_kk` copies, accumulator
  views, comm buffers, table views, ...)
- `<REFERENCES>`: two or three already-cleaned files with similar code
- `<VARIANT>`: optional paragraph for special batches (see below)

## Template

````
Remove silent fp32/fp64 conversion warnings in LAMMPS KOKKOS styles, for BOTH the single and mixed precision builds.

FIRST read <TOOLS>/RECIPE.md and follow it EXACTLY.

Checker (run BOTH per file; done only when BOTH print NOTHING). Run from <REPO>:
  <TOOLS>/chk.sh single <file>
  <TOOLS>/chk.sh mixed  <file>
It prints flagged src/KOKKOS file:line:col (including the style's own .h) plus any real compile ERROR lines.

Assigned files (baseline single/mixed):
<FILES>
  e.g. - src/KOKKOS/fix_brownian_kokkos.cpp        (40/50)

<DOMAIN HINTS>

<VARIANT>

Note the single and mixed counts differ, so BOTH configurations must be checked; a fix for one must not regress the other. Decide cast direction from the declared view type in the .h (t_kkacc_* = KK_ACC_FLOAT, t_kkfloat_* = KK_FLOAT).

Useful already-cleaned references on this branch: <REFERENCES>.

Edit ONLY your assigned files. Other agents are working on other files in the same tree at the same time: do not touch, revert, or reformat their files, and ignore warnings the checker reports in files you do not own (a shared header has exactly one owner). Do not change git state: no git add, commit, stash, checkout, reset, or branch operations; the main thread verifies and commits your batch. Put any helper scripts or scratch files OUTSIDE the repository.

Only change lines the checker flags. Do NOT introduce compile errors. Every cast must be a no-op in the default double build; call out anything that is not bit-identical.

Report final single AND mixed counts per file (both must be 0) and any judgment calls.
````

## Variants used in the original session

- **Mixed-only files** (single count 0): "These are already clean in the
  SINGLE build and only warn in the MIXED build (where KK_FLOAT is float but
  KK_ACC_FLOAT is double). ... Because they are single-clean, the warnings
  are almost certainly at KK_FLOAT -> KK_ACC_FLOAT accumulator boundaries ...
  Those casts are no-ops in the single build, so they cannot regress it --
  but re-run the single checker anyway to confirm it stays at 0."
- **Headers**: list each header with the including TU to check it through,
  `<TOOLS>/chk.sh single src/KOKKOS/<including_tu>.cpp <header>.h`, and
  "Edit the HEADER (.h) files."  Assign each shared header to one agent only,
  and schedule header batches first.
- **Very large files** (hundreds of flags): one file per agent, "work in
  chunks, re-check often".

## Example 1 (verbatim, seq 19163): two new styles, both precisions

````
Remove silent fp32/fp64 conversion warnings in newly ported LAMMPS KOKKOS styles, for BOTH the single and mixed precision builds.

FIRST read /tmp/claude-0/-home-user-lammps/93546465-9b0e-5641-a871-ee99a80f840a/scratchpad/RECIPE.md and follow it EXACTLY.

Checker (run BOTH per file; done only when BOTH print NOTHING). Run from /home/user/lammps:
  bash /tmp/claude-0/-home-user-lammps/93546465-9b0e-5641-a871-ee99a80f840a/scratchpad/chk.sh single <file>
  bash /tmp/claude-0/-home-user-lammps/93546465-9b0e-5641-a871-ee99a80f840a/scratchpad/chk.sh mixed  <file>
It prints flagged src/KOKKOS file:line:col (including the style's own .h) plus any real compile ERROR lines.

Assigned files (baseline single/mixed):
- src/KOKKOS/fix_brownian_kokkos.cpp        (40/50)
- src/KOKKOS/fix_damping_cundall_kokkos.cpp (28/22)

fix_brownian is a Brownian dynamics integrator: expect random number generator calls returning double that must be narrowed to KK_FLOAT, base-class double members (dt, gamma_t, gamma_r, diffusion coefficients, temperature) needing _kk local copies, double literals, and KK_FLOAT velocity/position views meeting KK_ACC_FLOAT force/torque views. fix_damping_cundall is a granular damping fix: expect torque/omega views and damping coefficients.

Note the single and mixed counts differ, so BOTH configurations must be checked; a fix for one must not regress the other. Decide cast direction from the declared view type in the .h (t_kkacc_* = KK_ACC_FLOAT, t_kkfloat_* = KK_FLOAT).

Useful already-cleaned references on this branch: src/KOKKOS/fix_langevin_kokkos.cpp (RNG narrowing + _kk copies), src/KOKKOS/pair_brownian_kokkos.cpp, src/KOKKOS/fix_gjf_kokkos.cpp.

Only change lines the checker flags. Do NOT introduce compile errors. Every cast must be a no-op in the default double build; call out anything that is not bit-identical.

Report final single AND mixed counts per file (both must be 0) and any judgment calls.
````

## Example 2 (verbatim, seq 19191): mixed-only files

````
Remove silent fp32/fp64 conversion warnings in newly ported LAMMPS KOKKOS styles. These four are already clean in the SINGLE build and only warn in the MIXED build (where KK_FLOAT is float but KK_ACC_FLOAT is double).

FIRST read /tmp/claude-0/-home-user-lammps/93546465-9b0e-5641-a871-ee99a80f840a/scratchpad/RECIPE.md and follow it EXACTLY, especially rule 4 (accumulators).

Checker (run BOTH per file; done only when BOTH print NOTHING). Run from /home/user/lammps:
  bash /tmp/claude-0/-home-user-lammps/93546465-9b0e-5641-a871-ee99a80f840a/scratchpad/chk.sh single <file>
  bash /tmp/claude-0/-home-user-lammps/93546465-9b0e-5641-a871-ee99a80f840a/scratchpad/chk.sh mixed  <file>
It prints flagged src/KOKKOS file:line:col (including the style's own .h) plus any real compile ERROR lines.

Assigned files (baseline single/mixed):
- src/KOKKOS/fix_propel_self_kokkos.cpp  (0/6)
- src/KOKKOS/fix_flow_gauss_kokkos.cpp   (0/6)
- src/KOKKOS/fix_store_force_kokkos.cpp  (0/3)
- src/KOKKOS/bond_table_kokkos.cpp       (0/1)

Because they are single-clean, the warnings are almost certainly at KK_FLOAT -> KK_ACC_FLOAT accumulator boundaries: a KK_FLOAT value feeding a KK_ACC_FLOAT force/torque/energy view or an EV_FLOAT field, or the reverse (a KK_ACC_FLOAT force read into KK_FLOAT math). Decide the direction from the declared view type in each .h (t_kkacc_* = KK_ACC_FLOAT, t_kkfloat_* = KK_FLOAT). Those casts are no-ops in the single build, so they cannot regress it -- but re-run the single checker anyway to confirm it stays at 0.

Useful already-cleaned references on this branch: src/KOKKOS/fix_setforce_kokkos.cpp, src/KOKKOS/fix_addforce_kokkos.cpp, src/KOKKOS/fix_spring_self_kokkos.cpp.

Only change lines the checker flags. Do NOT introduce compile errors. Every cast must be a no-op in the default double build; call out anything that is not bit-identical.

Report final single AND mixed counts per file (both must be 0) and any judgment calls.
````

## Example 3 (verbatim, seq 9113): shared headers checked through including TUs

(`checkall.sh <TU> <header>` was the predecessor of `chk.sh <prec> <TU> <header>`.)

````
Remove silent fp32/fp64 conversion compiler warnings in several LAMMPS KOKKOS HEADER files. Each is verified by compiling a DIFFERENT including .cpp TU.

FIRST read /tmp/claude-0/-home-user-lammps/93546465-9b0e-5641-a871-ee99a80f840a/scratchpad/RECIPE.md (especially the "HEADER FILES" section) and follow it EXACTLY. Reference: /home/user/lammps/src/KOKKOS/meam_funcs_kokkos.h.

Edit the HEADER (.h) files. For each, verify with checkall.sh <including_TU> <header>:
- math_extra_kokkos.h    (~69) -> bash .../checkall.sh src/KOKKOS/compute_erotate_asphere_kokkos.cpp math_extra_kokkos.h
- group_kokkos.h         (~46) -> bash .../checkall.sh src/KOKKOS/compute_temp_com_kokkos.cpp group_kokkos.h
- sna_kokkos_impl.h      (~45) -> bash .../checkall.sh src/KOKKOS/pair_snap_kokkos.cpp sna_kokkos_impl.h
- pair_kokkos.h          (~7)  -> bash .../checkall.sh src/KOKKOS/fix_neigh_history_kokkos.cpp pair_kokkos.h
- pair_brownian_kokkos.h (~8)  -> bash .../checkall.sh src/KOKKOS/pair_brownian_kokkos.cpp pair_brownian_kokkos.h
- fix_langevin_kokkos.h  (~1)  -> bash .../checkall.sh src/KOKKOS/fix_langevin_kokkos.cpp fix_langevin_kokkos.h
(full checkall.sh path: /tmp/claude-0/-home-user-lammps/93546465-9b0e-5641-a871-ee99a80f840a/scratchpad/checkall.sh)

Iterate each until its checkall prints NOTHING (0 warnings AND 0 ERROR lines). Rules: bare sqrt/pow/exp/log on KK_FLOAT -> Kokkos:: (Kokkos::pow KK_FLOAT exponent); double literals -> static_cast<KK_FLOAT>; base-class double members -> _kk copies; accumulators -> KK_ACC_FLOAT else KK_FLOAT; genuine-double sub-exprs at a KK_FLOAT boundary -> wrap in static_cast<KK_FLOAT>. Only flagged lines. NO compile errors.

Report final count per header (must be 0) and judgment calls.
````
