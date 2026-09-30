# Subagent prompt template

This is the prompt skeleton handed to each parallel subagent.  It is derived
from the last-phase prompts of the original session (seq 19163 and 19191 in
`stanmoore1/private:lammps/kokkos-precision/history/agents/all_agent_calls.jsonl`),
with the batch-specific parts replaced by `<PLACEHOLDERS>`, the scratch paths
replaced by the toolkit paths, and the lessons of the whole session added:
the "work in chunks" line (originally only in the large-file prompts), the
rules on git state, report files, scratch space, and nested agents (implicit
or missing in the original, and each the cause of a real problem).

Batch sizes in the original: 1-14 files per agent, mostly 1-6.  Very large
files (300-868 flagged lines, e.g. `pair_exp6_rx`, `atom_vec`) had an agent
of their own; tail clusters of small files from one family went 8-14 to an
agent.  Group a batch by style family so that one set of domain hints and
reference files fits all of it.

Fill in:
- `<REPO>`: absolute path of the LAMMPS checkout the agents edit
- `<TOOLS>`: `<REPO>/tools/kokkos-precision` (or wherever this toolkit is)
- `<SCRATCH>`: an absolute scratch directory OUTSIDE the repository, e.g.
  `$KKP_WORK/agents`; every agent gets its own subfolder `<SCRATCH>/<NAME>`
- `<NAME>`: a short unique batch name (e.g. `dihedral-2`, `mixed-fix-a`)
- `<FILES>`: one line per assigned file with the baseline counts from
  `survey.sh` as `(single/mixed)`
- `<DOMAIN HINTS>`: what the style does and which patterns to expect
  (RNG narrowing, base-class members needing `_kk` copies, accumulator
  views, comm buffers, table views, ...)
- `<REFERENCES>`: two or three already-cleaned files with similar code
- `<VARIANT>`: optional paragraph for special batches (see below)
- `<NESTING>`: one of the two nested-agent paragraphs below

## Template

````
Remove silent fp32/fp64 conversion warnings in LAMMPS KOKKOS styles, for BOTH the single and mixed precision builds.

FIRST read <TOOLS>/RECIPE.md and follow it EXACTLY.

Checker (run BOTH per file; done only when BOTH print NOTHING). Run from <REPO>:
  <TOOLS>/chk.sh single <file>
  <TOOLS>/chk.sh mixed  <file>
It prints flagged src/KOKKOS file:line:col (including the style's own .h) plus any real compile ERROR lines. For the full warning text with the types (needed to choose the cast direction): <TOOLS>/warntext.sh <single|mixed> <file>

Assigned files (baseline single/mixed):
<FILES>
  e.g. - src/KOKKOS/fix_brownian_kokkos.cpp        (40/50)

<DOMAIN HINTS>

<VARIANT>

Note the single and mixed counts differ, so BOTH configurations must be checked; a fix for one must not regress the other. Decide cast direction from the declared view type in the .h (t_kkacc_* = KK_ACC_FLOAT, t_kkfloat_* = KK_FLOAT).

Useful already-cleaned references on this branch: <REFERENCES>.

Work in chunks: fix a group of flagged lines (about 30-40 lines in a large file), re-run the checker, confirm the count went down and no new warnings or ERROR lines appeared, then continue.

Edit ONLY your assigned files. Other agents are working on other files in the same tree at the same time: do not touch, revert, or reformat their files, and ignore warnings the checker reports in files you do not own (a shared header has exactly one owner). Do not change git state: no git add, commit, stash, checkout, reset, or branch operations; the main thread verifies and commits your batch.

Use absolute paths for every file you read, edit, or mention. Do not write report or summary files anywhere in the repository; your final message is the report. Put helper scripts and scratch files ONLY in your own folder <SCRATCH>/<NAME>/ (create it; other agents use the same parent folder, so never use a bare file name there).

<NESTING>

Only change lines the checker flags (a new helper line such as a _kk copy is allowed; list it). Do NOT introduce compile errors. Every cast must be a no-op in the default double build; call out anything that is not bit-identical.

Report final single AND mixed counts per file (both must be 0) and any judgment calls.
````

## Nested agents (`<NESTING>`)

In the original mixed-precision pass, no instruction addressed nested
agents, and two agents with 6-8 files each started one subagent per file
(11 nested agents in total).  Both parents then returned early, reporting
only that their subagents were running; the main thread had to check the
assigned files itself and commit as the nested notifications arrived.  The
nested results were good, but the parent reports were useless.  Use one of:

- Default (batches of up to about 6 files):
  "Do NOT launch subagents of your own; do all edits yourself."
- Large batches of independent files, when you want the extra parallelism:
  "You MAY launch one subagent per file for files you cannot finish yourself.
  Give each the same rules (this prompt, with only its file). You MUST wait
  for every subagent to finish, re-run both checkers on every assigned file
  yourself, and report the final counts; do not return while any subagent is
  still running."

Whether the orchestrating agent may launch agents at all is a separate
question, see "Agents and the Agent tool" in README.md.

## Variants used in the original session

- **Mixed-only files** (single count 0): "These are already clean in the
  SINGLE build and only warn in the MIXED build (where KK_FLOAT is float but
  KK_ACC_FLOAT is double). ... Because they are single-clean, the warnings
  are almost certainly at KK_FLOAT -> KK_ACC_FLOAT accumulator boundaries ...
  Those casts are no-ops in the single build, so they cannot regress it --
  but re-run the single checker anyway to confirm it stays at 0."
- **Headers**: list each header with the including TU(s) to check it through
  (find them with `hdrorigin.sh`),
  `<TOOLS>/chk.sh single src/KOKKOS/<including_tu>.cpp <header>.h`, and
  "Edit the HEADER (.h) files."  Assign each shared header to one agent only.
  For a residual header: "this header was partially cleaned via one TU, but a
  DIFFERENT including TU instantiates more code; check it through <TU2> and
  keep <TU1> at 0."  (Recommendation, not what happened: the original found
  the header gap late and cleaned most headers after the .cpp files; schedule
  shared headers first, see README.md.)
- **Very large files** (hundreds of flags): one file per agent, "This file is
  LARGE (<N> flagged lines): work in chunks of about 30-40 lines, re-running
  the checker after each chunk", plus a short "watch for" list (base-class
  members, `fm_exp`, EV accumulations, bare math).
- **Resume after a killed agent** (usage limit, tool outage): "This file was
  PARTIALLY done by a previous run that was interrupted; its edits are still
  in the tree. Do NOT revert them. Run the checker to see what is left and
  continue from there; the current counts are <single>/<mixed>."  Name files
  of the batch that are already done and committed: "NOTE: <file> is already
  done; do not edit it."
- **After a merge of upstream changes**: "These files contain code that was
  written upstream WITHOUT the cast discipline" (or "whose physics was just
  rewritten upstream"); check every including TU in both precisions.
- **Double-build scope check**: "call out anything that is not
  bit-identical" and "do not change algorithm behavior" for new ports.

## Examples from the original session

The three prompts below are the original main-thread prompts with only the
paths replaced by `<REPO>`, `<TOOLS>`, and `<SCRATCH>`.  They predate the
chunking, scratch-folder, report-file, and nesting lines of the template
above; add those when reusing them.

## Example 1 (original prompt, seq 19163): two new styles, both precisions

````
Remove silent fp32/fp64 conversion warnings in newly ported LAMMPS KOKKOS styles, for BOTH the single and mixed precision builds.

FIRST read <TOOLS>/RECIPE.md and follow it EXACTLY.

Checker (run BOTH per file; done only when BOTH print NOTHING). Run from <REPO>:
  <TOOLS>/chk.sh single <file>
  <TOOLS>/chk.sh mixed  <file>
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

## Example 2 (original prompt, seq 19191): mixed-only files

````
Remove silent fp32/fp64 conversion warnings in newly ported LAMMPS KOKKOS styles. These four are already clean in the SINGLE build and only warn in the MIXED build (where KK_FLOAT is float but KK_ACC_FLOAT is double).

FIRST read <TOOLS>/RECIPE.md and follow it EXACTLY, especially rule 4 (accumulators).

Checker (run BOTH per file; done only when BOTH print NOTHING). Run from <REPO>:
  <TOOLS>/chk.sh single <file>
  <TOOLS>/chk.sh mixed  <file>
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

## Example 3 (original prompt, seq 9113): shared headers checked through including TUs

(`checkall.sh <TU> <header>` was the predecessor of `chk.sh <prec> <TU> <header>`.)

````
Remove silent fp32/fp64 conversion compiler warnings in several LAMMPS KOKKOS HEADER files. Each is verified by compiling a DIFFERENT including .cpp TU.

FIRST read <TOOLS>/RECIPE.md (especially the "HEADER FILES" section) and follow it EXACTLY. Reference: <REPO>/src/KOKKOS/meam_funcs_kokkos.h.

Edit the HEADER (.h) files. For each, verify with checkall.sh <including_TU> <header>:
- math_extra_kokkos.h    (~69) -> bash .../checkall.sh src/KOKKOS/compute_erotate_asphere_kokkos.cpp math_extra_kokkos.h
- group_kokkos.h         (~46) -> bash .../checkall.sh src/KOKKOS/compute_temp_com_kokkos.cpp group_kokkos.h
- sna_kokkos_impl.h      (~45) -> bash .../checkall.sh src/KOKKOS/pair_snap_kokkos.cpp sna_kokkos_impl.h
- pair_kokkos.h          (~7)  -> bash .../checkall.sh src/KOKKOS/fix_neigh_history_kokkos.cpp pair_kokkos.h
- pair_brownian_kokkos.h (~8)  -> bash .../checkall.sh src/KOKKOS/pair_brownian_kokkos.cpp pair_brownian_kokkos.h
- fix_langevin_kokkos.h  (~1)  -> bash .../checkall.sh src/KOKKOS/fix_langevin_kokkos.cpp fix_langevin_kokkos.h
(checkall.sh was a script in the scratch directory of the original session)

Iterate each until its checkall prints NOTHING (0 warnings AND 0 ERROR lines). Rules: bare sqrt/pow/exp/log on KK_FLOAT -> Kokkos:: (Kokkos::pow KK_FLOAT exponent); double literals -> static_cast<KK_FLOAT>; base-class double members -> _kk copies; accumulators -> KK_ACC_FLOAT else KK_FLOAT; genuine-double sub-exprs at a KK_FLOAT boundary -> wrap in static_cast<KK_FLOAT>. Only flagged lines. NO compile errors.

Report final count per header (must be 0) and judgment calls.
````
