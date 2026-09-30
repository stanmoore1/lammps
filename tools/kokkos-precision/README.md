# Removing silent fp32/fp64 conversions from the KOKKOS package

This folder contains a toolkit and the accumulated knowledge from a large,
successful cleanup of silent single/double precision conversions in the
LAMMPS KOKKOS package, carried out with a coding agent that ran many parallel
subagents.  It is meant to be reused whenever new KOKKOS styles are ported or
existing ones change, by a person or by an agent.

| File | Purpose |
|---|---|
| `README.md` | this knowledge base: background, workflow, rules of scope, pitfalls |
| `RECIPE.md` | the canonical fixing rules; every subagent reads it first |
| `PROMPT_TEMPLATE.md` | the subagent prompt skeleton, with real examples |
| `configure.sh` | configure a clang KOKKOS Serial build in single or mixed precision with the warning flags |
| `mkcmd.sh` | capture the compile command of a configured build for the checkers |
| `chk.sh` | per-TU checker: flagged `src/KOKKOS` locations plus compile errors; empty = clean |
| `survey.sh` | baseline counts `single mixed errors basename` per TU |
| `vc.sh` | verify-and-commit gate for a finished batch (both precisions, 0 warnings, 0 errors) |
| `fullsurvey.sh` | per-file counts from a full build log (the authoritative check) |
| `kkp-env.sh` | shared settings sourced by the scripts |
| `recovery/` | decode cached session pages and extract the tooling from a transcript (the step-by-step procedure is in `stanmoore1/private:lammps/kokkos-precision/kokkos-precision-session-recovery.md`) |
| `stanmoore1/private:lammps/kokkos-precision/history/` | the verbatim recovered record of the original work (see `stanmoore1/private:lammps/kokkos-precision/history/README.md`) |

All scripts print their usage when run without arguments.  They find the
LAMMPS checkout with `git rev-parse --show-toplevel` (override with
`KKP_REPO`), keep their state in `KKP_WORK` (default
`$HOME/.cache/kk-precision`), and use `CXX_CLANG` (default: `clang++` from
the `PATH`).

## Background

### Precision modes of the KOKKOS package

Pull request [lammps/lammps#4754](https://github.com/lammps/lammps/pull/4754)
("silent fp32/fp64 conversion removal") introduced single and mixed
precision support to the KOKKOS package.  Two typedefs in
`src/KOKKOS/kokkos_type.h` carry the precision:

| `KOKKOS_PREC` (CMake) | define | `KK_FLOAT` | `KK_ACC_FLOAT` |
|---|---|---|---|
| `double` (default) | `LMP_KOKKOS_DOUBLE_DOUBLE` | double | double |
| `mixed` | `LMP_KOKKOS_SINGLE_DOUBLE` | float | double |
| `single` | `LMP_KOKKOS_SINGLE_SINGLE` | float | float |

`KK_FLOAT` is used for positions, velocities, parameters, and most math;
`KK_ACC_FLOAT` for accumulators: the energy and virial fields of `EV_FLOAT`,
per-atom energy/virial views, and force/torque views.  The view typedefs make
the choice visible: `t_kkfloat_*` views hold KK_FLOAT, `t_kkacc_*` views hold
KK_ACC_FLOAT.

In the default double build every mix of the two types is harmless.  In the
single and mixed builds an expression such as
`KK_FLOAT x = 1.0 + log(y);` silently promotes the float operand to double,
computes in double (slow on GPUs) and narrows back, which also changes the
numerics in a way nobody asked for.  #4754 found such spots by compiling with
clang and

    -Wimplicit-float-conversion   (narrowing, double -> float)
    -Wdouble-promotion            (widening, float -> double)

and fixed them with explicit casts such as `static_cast<KK_FLOAT>(...)`,
which are no-ops in the double build and fold at compile time for constants.
That cleanup was not exhaustive, and every new port brings new instances.

### The conventions in brief (details in RECIPE.md)

- Bare C library math (`sqrt`, `exp`, `pow`, ...) on KK_FLOAT calls the double
  version; `Kokkos::sqrt` etc. have float overloads.  Gotcha: `Kokkos::pow(x, 2)`
  with an int exponent promotes to double; write
  `Kokkos::pow(x, static_cast<KK_FLOAT>(2))`.
- `MathSpecialKokkos::fm_exp()` is double-only: cast the argument to double
  and the result to KK_FLOAT explicitly.
- Base-class `double` members used in KK_FLOAT kernels get a local copy with
  the `_kk` suffix: `const KK_FLOAT cut_kk = static_cast<KK_FLOAT>(cut);`.
- Values feeding accumulators are cast to KK_ACC_FLOAT, host reductions into
  base-class doubles (`eng_vdwl`, `virial[]`) to `double`.
- Header warnings are per translation unit: a header only warns when an
  including `.cpp` is compiled, and different TUs instantiate different
  templates.  Check a header through its including TUs, and trust only a
  full rebuild.

## The workflow, as it actually ran

The numbers below are from the original work: the first round took 13,695
single and 2,956 mixed warnings to zero across 246 files on a branch off
`develop`; later rounds cleaned newly ported styles on feature branches
(249/234 and 89/116 single/mixed locations, and so on).

1. **Configure two builds** (single and mixed), clang, KOKKOS Serial, all
   packages with KOKKOS styles, the warning flags, Ninja:

        tools/kokkos-precision/configure.sh single build-kk-single
        tools/kokkos-precision/configure.sh mixed  build-kk-mixed

   This is the configure line of the original plan
   (`-C cmake/presets/clang.cmake -C cmake/presets/kokkos-serial.cmake
   -C cmake/presets/kokkos-packages.cmake -D KOKKOS_PREC=<prec>
   -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wimplicit-float-conversion -Wdouble-promotion -pedantic"
   -G Ninja`) plus `-D PKG_REAXFF=off`, because REAXFF was skipped by
   instruction.  ML-IAP is double-only and the preset already leaves it out.
   The script regenerates the package list with the command documented in
   `cmake/presets/kokkos-packages.cmake` and warns about packages with
   KOKKOS styles that the preset lacks; add them as extra arguments, e.g.
   `-D PKG_CG-DNA=on`.  (On `develop` at the time of writing the preset was
   complete; on feature branches it repeatedly lagged, see the pitfalls.)
   External library downloads (e.g. for ML-PACE) may need network access.

2. **Full build** of both, keep going past errors, keep the logs:

        cmake --build build-kk-single -- -k 0 2>&1 | tee build-kk-single/build.log
        cmake --build build-kk-mixed  -- -k 0 2>&1 | tee build-kk-mixed/build.log

3. **Capture the compile commands** for the checkers (re-run after every
   re-configure):

        tools/kokkos-precision/mkcmd.sh single build-kk-single
        tools/kokkos-precision/mkcmd.sh mixed  build-kk-mixed

4. **Survey.**  `fullsurvey.sh build-kk-single` lists flagged locations per
   file from the full build log (headers counted once).  For the assigned
   files, `survey.sh` gives the per-TU baseline `single mixed errors` that goes
   into the prompts.  The file list must come from a COMPLETE build (see
   pitfalls).

5. **Batch the files**: 1-4 files per agent, grouped by family (angles,
   dihedrals, coul/long pairs, manybody pairs, integrator fixes, computes, ...)
   so that the domain hints and reference files fit the whole batch.  Very
   large files (hundreds of flags, e.g. `pair_exp6_rx`, `pair_pod`) get their
   own agent.  Shared headers (`pair_kokkos.h`, `math_extra_kokkos.h`,
   `group_kokkos.h`, `sna_kokkos_impl.h`, the MEAM headers) go FIRST, each
   with exactly ONE owner, because their warnings show up in many TUs.

6. **Launch general-purpose subagents in parallel**, one batch each, with the
   prompt from `PROMPT_TEMPLATE.md`: read `RECIPE.md` first; the two checker
   commands; the assigned files with baseline counts; domain hints; already
   cleaned reference files; edit only the assigned files, other agents work
   in the same tree, no git state changes, helper scripts outside the
   repository; change only flagged lines; no compile errors; casts must be
   no-ops in the double build; report final counts and judgment calls.  The
   original ran 82 main-thread agents (plus 11 that subagents started
   themselves), several at a time.

7. **Gate and commit each batch as soon as its agent finishes**:

        tools/kokkos-precision/vc.sh --commit "KOKKOS: remove silent fp32/fp64 conversions in <files>" <files>

   `vc.sh` recompiles every file in both precisions and commits only these
   files, and only if all counts are zero.  Committing per batch keeps the
   tree committable at all times, lets you push regularly, and survives
   container resets (the original once lost its local branch and build
   directories to a reset; everything was safe on the remote).  The original
   branch reached 813 per-batch commits before squashing.  Read the agent's
   judgment calls before committing; they are the best review aid.

8. **Full rebuild of both precisions as the authoritative check.**  Force a
   recompile of every KOKKOS TU and re-survey:

        tools/kokkos-precision/fullsurvey.sh --rebuild --touch build-kk-single
        tools/kokkos-precision/fullsurvey.sh --rebuild build-kk-mixed

   (`--touch` touches `src/KOKKOS/kokkos_type.h`, which every KOKKOS TU
   includes; an incremental build only reports the TUs it recompiles.)
   Anything left over becomes a new batch.  Also compile once with the
   default double precision to show that no new warnings or errors appear
   there.

9. **Numerical inertness check.**  Build a control binary from the same
   branch WITHOUT the cast commits, run relevant examples with both binaries
   in each precision, and compare the thermo output.  The original used
   `examples/melt` with `-sf kk`, `examples/rigid/in.rigid.small.infile`, and
   the `examples/PACKAGES/brownian` inputs.  Result: mixed precision identical
   to the last printed digit; single precision differed only in the 8th
   significant figure of the pressure (12.288905 vs 12.288904), traced to one
   `v_tally` accumulator change.  Two failures seen during this check
   (`dpdrx-shardlow` with `-sf kk`, granular `in.pour.drum`) were shown with
   the control build to be pre-existing and unrelated.

10. **Squash and open the pull request.**  Squash the cast commits into one;
    author and committer Stan Moore (the committer comes from `user.name`/
    `user.email` in the git configuration); unsigned; no trailers of any kind:

        git reset --soft <base> && git commit --no-gpg-sign \
            --author="Your Name <you@example.com>" -m "..."
        git log -1 --format='%an <%ae> / %cn <%ce> / %G?'   # expect ... / N

    Open the PR against the target branch with the repository's pull request
    template; the **AI Tools Usage** section of the template is the ONLY place
    for AI attribution (no `Co-Authored-By:`, `Claude-Session:`, or
    "Generated with" lines anywhere).

## Scope rules

- Only `src/KOKKOS/`.  Kokkos library headers under `lib/kokkos`, other
  third-party code in `lib/`, non-KOKKOS sources, and the `-Wall`/`-Wextra`/
  `-pedantic` warnings are out of scope.
- Skip REAXFF (by instruction) and ML-IAP (double-only).
- Only the two flags.  `-Wimplicit-int-float-conversion` (int/`bigint`/
  `tagint` to float or double, e.g.
  `double delta = update->ntimestep - update->beginstep;`) is out of scope,
  and so is every warning that also appears in the default double build.
  Check with `chk.sh double <TU>` (optionally with
  `KKP_WFLAGS='Wimplicit-int-float-conversion|Wimplicit-float-conversion|Wdouble-promotion'`).
- If in doubt, do not change code the compiler does not flag.
- Keep separate concerns on separate branches as the user directs.  In the
  last round the instruction was: fix only the package-specific (CG-DNA)
  warnings on the feature branch, and fix unrelated drift that had
  accumulated in other files on a new branch off `develop`.
- Latent bugs found in passing (for example coordinates unpacked through
  `static_cast<tagint>` in `fix_spring_self_kokkos.cpp`, or an `MPI_DOUBLE`
  reduction into a KK_FLOAT scalar in `min_fire_kokkos.cpp`) are reported,
  not fixed in the cast change.

## Pitfalls and lessons (all of these happened)

- **A warnings-only checker hides broken code.**  One agent emptied the
  right-hand sides in `pair_coul_long_kokkos.cpp`, leaving eight
  `h_table(i) = static_cast<KK_FLOAT>();`.  The file no longer compiled, so it
  no longer warned, and the checker reported it clean.  Checkers must print
  compile ERROR lines, and a gate must require zero errors
  (`stanmoore1/private:lammps/kokkos-precision/history/agents/failure_pair_coul_long.md`).
- **Filtering to the `.cpp` basename hides header warnings.**  The first
  checker only showed lines of the compiled file, so the MEAM headers still
  had hundreds of warnings after their TUs were "clean".  `chk.sh` prints
  all `src/KOKKOS` files reached by the TU.
- **A manifest from an incomplete build misses whole families.**  The first
  file list came from a build that had not finished; all `improper_*`,
  `fix_wall_*`, `bond_gaussian`, `bond_quartic`, and `angle_dipole` were
  missing (647 warnings).  They were found only by forcing a full recompile.
- **Wrong git baseline.**  Listing "already cleaned" files with
  `git diff develop..HEAD` was wrong because the branch base was ahead of
  `develop`; five files were reported done that were never cleaned.  The full
  build, not git history, is the source of truth.
- **`git checkout --theirs` on a conflicted file discards the auto-merged
  casts.**  In a merge of upstream changes into `fix_langevin_kokkos.cpp`
  this would have thrown away 45 casts; resolve such conflicts in place.
- **The `kokkos-packages.cmake` preset lags new ports.**  On the porting
  branches it was missing FEP, CORESHELL, BROWNIAN, and SHOCK (new styles
  failed with `fatal error: 'pair_lj_cut_soft.h' file not found`), and later
  CG-DNA.  `configure.sh` now warns; add `-D PKG_<NAME>=on`.
- **Upstream force-pushes.**  When the feature branch you are cleaning is
  rebased upstream, rebase (or cherry-pick) your cast commits onto the new
  tip instead of merging the rewritten history, which would duplicate every
  rewritten commit.
- **Commit signing.**  Do not sign with the container's SSH key: GitHub shows
  such commits as "Unverified".  Commit with `--no-gpg-sign` (`vc.sh` does
  this unless `KKP_SIGN=1`).
- **Stop-hook pressure.**  An "uncommitted changes, please commit and push"
  hook fires constantly while agents work.  The answer is the per-batch gated
  commit of finished, verified work, never a blanket commit of whatever is in
  the tree.
- **Ambiguous warning text.**  Through atomic and scatter views clang names
  the target type as plain `double` instead of `KK_ACC_FLOAT`; decide from
  the declared view type in the header.
- **Build tree and source tree must match.**  The checkers reuse the include
  paths of the build; a build configured from a different checkout makes them
  compile against the wrong headers (`mkcmd.sh` warns).

## The post-compaction failure

Late in the original session the agent's context was compacted.  The summary
kept the facts (files, counts, branch names) but lost the method: the
checker scripts lived only in a scratchpad whose contents the summary did not
carry, and the reasoning behind the rules was reduced to a list.  The agent
rebuilt the tooling from the summary and, instead of running per-file
subagents that read each flagged line and judged it, substituted regex
auto-fixers that rewrote flagged lines from the warning text
(`stanmoore1/private:lammps/kokkos-precision/history/postcompaction_attempt/`).  Observed failure modes:

- runaway nested casts, `static_cast<KK_FLOAT>(static_cast<KK_FLOAT>(...))`,
  growing with each driver iteration;
- dropped `if (eflag_global)` guards when a statement was rewritten;
- `using MathConst::MY_PI;` corrupted into a nonexistent `MY_PI_KK`;
- over-widening to `static_cast<double>`, which re-narrowed at the next
  assignment and INCREASED the warning counts;
- splicing text into the middle of existing `static_cast` tokens;
- treating continuation lines of multi-line statements as complete
  statements.

The user stopped it ("It seems like you are spinning your wheels"), the
session transcript was recovered, and the original tooling and prompts were
extracted from it; this toolkit is the result.  Lessons:

- Never replace the agent-per-file workflow with pattern rewriting.  The
  fixes look mechanical but each needs a type decision (KK_FLOAT vs
  KK_ACC_FLOAT vs double, which operand, which side of the boundary) that
  depends on declarations elsewhere.
- Keep the tooling in the repository (this folder), not in a scratchpad.
- After a compaction, recover the tooling first (see `recovery/`), then
  continue.

## Session recovery

Claude Code on the web streams the session from the server; the transcript
is not a local file you can rely on:

- The server keeps every event, available page by page from
  `/v1/code/sessions/<session id>/events`.  The browser-console script in
  `stanmoore1/private:lammps/kokkos-precision/kokkos-precision-session-recovery.md` downloads all of it as one JSON file.
- The Claude desktop app caches the pages it has displayed.  On macOS, open
  the session, scroll to the very top so that every page is loaded, quit the
  app, and collect the cache files from
  `~/Library/Application Support/Claude/Cache/Cache_Data/` (the script in
  `stanmoore1/private:lammps/kokkos-precision/kokkos-precision-session-recovery.md` copies only the transcript pages).
- The local `~/.claude/projects/*.jsonl` of a cloud container covers only
  that container's lifetime, which is not the whole session after resets.

Then decode and extract:

    recovery/recover_cache.py <dir-with-cache-pages> allev.json   # needs: pip install zstandard
    recovery/extract_tooling.py allev.json recovered/

`recover_cache.py` reports the event span and the number of gaps (zero gaps
means the whole session is present).  `extract_tooling.py` writes every
version of the named tooling files (Write calls, Bash heredocs including
`cat >>` appends, full Read snapshots), the generator commands, all Agent
prompts (clustered into families), the task-notification reports, and the
user messages, in the layout of `stanmoore1/private:lammps/kokkos-precision/history/`.

## Quick start on a new branch

    git switch -c my-kk-precision-fixes            # or the feature branch to clean
    T=tools/kokkos-precision
    $T/configure.sh single build-kk-single         # heed any "preset lacks" warning
    $T/configure.sh mixed  build-kk-mixed
    cmake --build build-kk-single -- -k 0 2>&1 | tee build-kk-single/build.log
    cmake --build build-kk-mixed  -- -k 0 2>&1 | tee build-kk-mixed/build.log
    $T/mkcmd.sh single build-kk-single
    $T/mkcmd.sh mixed  build-kk-mixed
    $T/fullsurvey.sh build-kk-single; $T/fullsurvey.sh build-kk-mixed
    for f in <flagged .cpp files>; do $T/survey.sh $f; done   # baselines
    # batch files, fill in PROMPT_TEMPLATE.md, launch agents in parallel;
    # as each agent finishes:
    $T/vc.sh --commit "KOKKOS: remove silent fp32/fp64 conversions in ..." <files>
    # when all batches are in:
    $T/fullsurvey.sh --rebuild --touch build-kk-single
    $T/fullsurvey.sh --rebuild build-kk-mixed
    # numerical check against a control build, squash, PR (see workflow above)

If the toolkit is not on your branch, keep a separate worktree of the
toolkit branch and point the scripts at your checkout with
`KKP_REPO=/path/to/checkout`.
