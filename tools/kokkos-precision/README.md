# Removing silent fp32/fp64 conversions from the KOKKOS package

This folder contains a toolkit and the accumulated knowledge from a large,
successful cleanup of silent single/double precision conversions in the
LAMMPS KOKKOS package, carried out with a coding agent that ran many parallel
subagents.  It is meant to be reused whenever new KOKKOS styles are ported or
existing ones change, by a person or by an agent.

| File | Purpose |
|---|---|
| `README.md` | this knowledge base: background, decisions, runbook, scope rules, pitfalls |
| `RECIPE.md` | the canonical fixing rules; every subagent reads it first |
| `PROMPT_TEMPLATE.md` | the subagent prompt skeleton, variants, and original examples |
| `configure.sh` | configure a clang KOKKOS Serial build in single or mixed precision with the warning flags |
| `mkcmd.sh` | capture the compile command of a configured build for the checkers |
| `chk.sh` | per-TU checker: flagged `src/KOKKOS` locations plus compile errors; empty = clean |
| `warntext.sh` | full clang warning text (types, source line, caret) for one TU, filtered to a file |
| `hdrorigin.sh` | which including TU(s) produce a header's warnings (from a build log or by compiling) |
| `survey.sh` | baseline counts `single mixed errors basename` per TU |
| `vc.sh` | verify-and-commit gate for a finished batch (both precisions, headers through several TUs) |
| `fullsurvey.sh` | per-file counts from a full build log (the authoritative check) |
| `squashverify.sh` | squash a branch to one commit and prove the tree is unchanged |
| `kkp-env.sh` | shared settings sourced by the scripts |
| `recovery/` | decode cached session pages and extract the tooling from a session transcript |

The complete record of the original work (the workflow audit with event
citations, the recovered tooling versions, all agent prompts, the pull
request bodies, and the session recovery procedure) is kept in the private
repository `stanmoore1/private` under `lammps/kokkos-precision/`
(`kokkos-precision-workflow-audit.md`, `history/`,
`kokkos-precision-session-recovery.md`).  References of the form "seqN" in
this folder are event numbers of that record.

All scripts print their usage when run without arguments.  They find the
LAMMPS checkout with `git rev-parse --show-toplevel` (override with
`KKP_REPO`), keep their state in `KKP_WORK` (default
`$HOME/.cache/kk-precision`), and use `CXX_CLANG` (default: `clang++` from
the `PATH`).  Below, `$LAMMPS` is the LAMMPS checkout and `$KKP_WORK` the
work directory; `T=$LAMMPS/tools/kokkos-precision`.

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

In the default double build every mix of the two types is harmless, and the
two warnings below never fire on KOKKOS code, so a single or mixed build is
mandatory for this work.  In the single and mixed builds an expression such
as `KK_FLOAT x = 1.0 + log(y);` silently promotes the float operand to
double, computes in double (slow on GPUs) and narrows back, which also
changes the numerics in a way nobody asked for.  #4754 found such spots by
compiling with clang and

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
- Values feeding accumulators are cast to KK_ACC_FLOAT; sums on the host
  that end in the base-class double members (`eng_vdwl`, `virial[]`) get
  `static_cast<double>`.
- Header warnings are per translation unit: a header only warns when an
  including `.cpp` is compiled, and different TUs instantiate different
  templates.  Check a header through several including TUs, and trust only
  a full rebuild.
- Change only lines the compiler flags.

## Decisions of the original work

These were settled with the user during planning or when a question came
up; reuse them unless the person you work for decides otherwise.  "Asked"
means the question was put to the user as a multiple-choice question with a
recommended option.

| Decision | How it was made | Why |
|---|---|---|
| clang, KOKKOS Serial backend, `kokkos-packages` preset, the warning flags `-Wall -Wextra -Wimplicit-float-conversion -Wdouble-promotion -pedantic` | initial request | reproduce the technique of #4754 |
| Build and clean BOTH single and mixed precision | asked (options: single, mixed, both); both chosen | mixed exposes accumulator boundaries that single hides |
| KK_ACC_FLOAT for accumulators (decide by `+=` use), KK_FLOAT otherwise; follow existing examples | plan feedback | the precision design of #4754 |
| Skip REAXFF; ML-IAP is off (double-only) | plan feedback | out of scope / cannot be built in single or mixed |
| Iterate compile, fix, compile until both builds are clean; change only lines the compiler flags | plan feedback | avoid unneeded or wrong edits |
| ML-PACE off; the resulting package list confirmed by the user | agent, then confirmed | the ML-PACE library download was blocked |
| Unqualified math on KK_FLOAT becomes `Kokkos::` math (rather than computing in double and casting the result back) | asked (three options: `Kokkos::` overloads, casting through double, inspecting first); `Kokkos::` chosen | float overloads; already used about 130 times in the package |
| New upstream TIP4P code: an O-branch force local declared KK_ACC_FLOAT next to KK_FLOAT siblings stays KK_FLOAT (a mixed-only rounding change, disclosed in the PR) | asked (options: keep KK_FLOAT, preserve exact mixed numerics, leave the lines alone); KK_FLOAT chosen | consistency with the sibling variables and branches |
| A single-only 8th-digit change from a `v_tally` accumulator cast in `fix_rigid_small` is kept and disclosed | agent, disclosed | consistency with the TIP4P decision |
| Commit trailers found on 94 commits: squash | asked (options: squash into a few clean commits, strip trailers and keep all, leave as is); squash chosen, later one commit requested | LAMMPS AI-attribution rule |
| Author and committer are the user; unsigned commits; AI attribution only in the PR template | user instruction | project rules; the container signing key shows "Unverified" |
| Where to push work for someone else's shared feature branch | asked (options: PR from a separate branch, push to the shared branch, keep local); separate branch plus PR chosen; a later round pushed directly on explicit permission | never push to a shared branch without permission |
| Package-specific warnings on the feature branch; unrelated drift in other files on a separate branch off `develop` | user instruction (later rounds) | separate concerns |
| Latent bugs found in passing (e.g. coordinates unpacked through `tagint`) are reported, not fixed | agent, left to the user | behavior change outside scope |
| Preset gaps (packages missing from `kokkos-packages.cmake`) are added on the configure line and reported; the preset file is not edited unasked | agent | the preset belongs to the branch author |

## How the original work was organized

The first round (July 2026) took 13,695 single and 2,956 mixed flagged
locations (unique; headers counted once) to zero across 246 files on a
branch off `develop`, in 94 gated commits that were later squashed into one
(merged upstream as lammps/lammps#5148).  Later rounds cleaned newly ported
styles on feature branches (249/234 and 89/116 single/mixed locations).

- **Plan mode first.**  The session started in plan mode: research (the
  #4754 pull request, the precision switch in `kokkos_type.h` and
  `cmake/Modules/Packages/KOKKOS.cmake`, the presets), a written plan, and
  multiple-choice questions only for the genuinely open decisions.  The
  user rejected the plan three times, each time with feedback; each piece
  of feedback became a standing rule (accumulator typing, skipping
  REAXFF/ML-IAP, iterate-to-zero and change-only-flagged-lines).  Treat
  plan rejections the same way: fold the feedback into the rules and the
  RECIPE, not just into the next plan revision.
- **Exemplar first.**  With 13,695 locations, the agent first cleaned the
  densest file by hand, `src/KOKKOS/meam_funcs_kokkos.h` (156 of the first
  318 warnings), to prove the recipe before fanning out.  It surfaced the
  rules that became the RECIPE: the early warning list was partial (header
  warnings from TUs not yet compiled); bare `sqrt` on KK_FLOAT promotes
  everywhere (which led to the `Kokkos::` question); `fm_exp` is
  double-only; the `_kk` copy needs its own cast; `Kokkos::pow(float, int)`
  promotes.  48 -> 5 -> 0 locations; committed alone and pushed; the
  RECIPE (v1) was written right after.
- **Validation wave.**  Five agents on a representative mix (small angles,
  pair morse/buck, core `domain_kokkos`, `fix_nve`, `dihedral_harmonic`);
  their output was verified before scaling up.
- **Scaling.**  About 8 agents live at a time, dedicated agents for the
  largest files (`pair_exp6_rx` 868 flags, `atom_vec`, dihedral class2,
  tersoff variants), family clusters alongside, tail clusters of 8-14 small
  files per agent at the end; 82 main-thread agent launches in about 17
  waves over the session, 1-14 files per agent (mostly 1-6).  Median agent
  run time 7.5 minutes (range 1.8-26.6).
- **Headers were discovered late.**  After most `.cpp` files were "clean",
  the MEAM headers still had hundreds of warnings because the first checker
  showed only the compiled file's own lines.  The checker was extended to
  print every `src/KOKKOS` file reached, the RECIPE got a HEADER FILES
  section, and dedicated header agents verified each header through one or
  more including TUs.  *Recommendation (not what happened):* survey
  headers first and give each shared header (`pair_kokkos.h`,
  `math_extra_kokkos.h`, `group_kokkos.h`, `sna_kokkos_impl.h`, the MEAM
  headers) a single owner before the per-file agents start.
- **The main thread's own hand fixes.**  Architectural cases were kept out
  of the agents: the `pair_table` interface (a double interface leaking
  into the shared `pair_kokkos.h` functor; fixed by making the
  `compute_*` functions take and return KK_FLOAT with an explicit double
  local inside, plus six `rsq < static_cast<KK_FLOAT>(cutsq)` in
  `pair_kokkos.h`, verified through `pair_table` and four other pair TUs);
  the `min_*` minimizers; `comm_kokkos` and the Binner in `kokkos_type.h`;
  and the `MY_EPSILON` constant of the DPD-REACT styles after an upstream
  merge.  Agents do well on local edits; interface changes that span files
  need one owner who sees all callers.
- **Mixed pass.**  A separate phase after single was clean: mixed build
  from scratch (2,956 unique locations in 87 files), a MIXED-BUILD PASS
  section in the RECIPE, 11 agents launched at once, and a gate that
  verifies BOTH precisions.
- **Later rounds** (August 2026, feature branches, same tooling): branch
  surveys of only the changed files, always followed by a full build (in
  one round 187 of 249 warnings were in files the branch had not modified:
  changed headers instantiated new template paths), control builds, pull
  requests from a separate branch.

## Runbook

### 1. Ground rules and planning

Start in plan mode.  Establish, and write into the plan: which precision
builds (both), the accumulator rule, the packages in and out of scope, the
math convention, "only flagged lines", "every cast is a no-op in double",
where commits go (branch, PR target, whether pushing to a shared branch is
allowed), and the commit identity.  Ask only the questions whose answers
are not already decided above.  Read `.github/copilot-instructions.md`
(AI attribution, style checks) and, for porting branches,
`.github/instructions/kokkos.instructions.md`.

### 2. Configure two builds

    cd $LAMMPS
    $T/configure.sh single build-kk-single     # heed any "preset lacks" warning
    $T/configure.sh mixed  build-kk-mixed

This is the configure line of the original work:
`-C cmake/presets/clang.cmake -C cmake/presets/kokkos-serial.cmake
-C cmake/presets/kokkos-packages.cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo
-D KOKKOS_PREC=<prec> -D BUILD_MPI=off -D FFT=KISS -D PKG_REAXFF=off
-D PKG_ML-IAP=off -D PKG_ML-PACE=off
-D CMAKE_CXX_FLAGS="-Wall -Wextra -Wimplicit-float-conversion -Wdouble-promotion -pedantic"
-G Ninja`.  Notes:

- ML-PACE: configuring it downloads an external library, which was blocked
  in the original environment, so ML-PACE was never in scope.  The script
  keeps it off by default; with network access add `-D PKG_ML-PACE=on` (or
  set `KKP_ML_PACE=on`) and clean its styles as well.
- The script regenerates the package list with the command documented in
  `kokkos-packages.cmake` and warns about packages with KOKKOS styles that
  the preset lacks.  On the porting branches the preset lacked FEP,
  CORESHELL, BROWNIAN, and SHOCK (new styles failed with
  `fatal error: '<base>.h' file not found`), and later CG-DNA.  Add them as
  extra arguments (`-D PKG_FEP=on ...`) and report the gap; do not edit the
  preset unasked.
- Always pass the full option set when re-configuring; do not rely on
  cached values (a later run re-configured with only `-D KOKKOS_PREC` and
  lost settings).  Build directories can disappear with a container reset;
  re-configure from scratch then.
- "BLAS/LAPACK not found" messages at configure time are harmless.
- The original toolchain: clang 18.1.3, cmake 3.28.3, ninja 1.11.1, 4 cores.

### 3. Full builds, in the background

    (ninja -C build-kk-single -k 0; echo "EXIT=$?") > build-kk-single/build.log 2>&1
    (ninja -C build-kk-mixed  -k 0; echo "EXIT=$?") > build-kk-mixed/build.log 2>&1

A full build took 19-29 minutes on 4 cores, so run each as a background
task (see "Tool timeouts and waiting" below) and wait for the `EXIT=`
marker.  Progress: `grep -oE "\[[0-9]+/[0-9]+\]" build.log | tail -1`.

### 4. Capture the compile commands

    $T/mkcmd.sh single build-kk-single
    $T/mkcmd.sh mixed  build-kk-mixed

Re-run after every re-configure.  The flags are stripped of `-MD -MT -MF`;
without that, clang reports "error opening ...cpp.o.d" and every agent wastes
turns explaining it (the original hit this and fixed it; a later
reconstruction hit it again).

### 5. Survey

WAIT FOR THE BUILDS TO FINISH.  The original extracted a list from an
unfinished build and missed every `improper_*`, several `fix_wall_*`,
`bond_gaussian`, `bond_quartic`, and `angle_dipole` (647 warnings), found
only much later by a forced full recompile.

    $T/fullsurvey.sh build-kk-single      # per-file counts, headers once
    $T/fullsurvey.sh build-kk-mixed
    for f in <flagged .cpp files>; do $T/survey.sh $f; done   # per-TU baselines

Raw warning lines repeat header warnings once per TU; always count unique
`file:line:col`.  On a feature branch, survey only the files the branch
added or changed (`git diff --name-status $(git merge-base HEAD <target>)
HEAD -- src/KOKKOS/`), but ALWAYS follow up with a full build of both
precisions, because changed headers make unchanged files warn.

### 6. Exemplar

Clean the densest file (preferably a header) by hand, with `chk.sh` and
`warntext.sh`, until both precisions are empty.  Update RECIPE.md with
anything it surfaces.  Commit it alone before agents start editing.

### 7. Batch and dispatch

- Launch a validation wave of about 5 agents on a representative mix and
  verify their output (step 8) before scaling up.
- Then keep about 8 agents live; a phase with many small independent files
  (the mixed pass) ran 11 at once.
- Group by style family; name one to three already-cleaned references from
  the same family in each prompt.  Very large files get their own agent
  with "work in chunks".  Headers: see "Headers were discovered late"
  above; each shared header has exactly one owner, and its prompt names the
  including TUs to check it through (`hdrorigin.sh` finds them).
- Keep architectural changes (interfaces shared by many styles) for the
  main thread.
- Prompts: `PROMPT_TEMPLATE.md`.  Every agent gets its own scratch
  subfolder outside the repository; shared scratch file names collided in
  practice.

### 8. Verify every agent (main thread)

For every completed agent, before committing anything:

1. Re-run the checker on each assigned file yourself (`survey.sh`, or
   `vc.sh` without `--commit`).  Never trust the report's counts; count
   compile errors separately.
2. INSPECT THE DIFF: `git diff --stat` and a sample of the `+`/`-` lines of
   each file; list added lines that contain no cast, which catches rewrites
   and emptied expressions:

        git diff -- src/KOKKOS/<file> | grep '^+' | grep -v '^+++' | grep -v static_cast

3. Read the agent's judgment calls.  Anything that is not bit-identical in
   single or mixed, any retyped declaration, any added helper line: check
   it, and escalate to the user when it changes numerics.
4. Gate and commit (step 9).  Hold back files that are not at 0/0; commit
   files that are, even while their agent works on another file.

### 9. Gate and commit per batch

    $T/vc.sh --style --commit "KOKKOS: remove silent fp32/fp64 conversions in <styles>" <files>

`vc.sh` recompiles every file in both precisions (headers through up to 3
including TUs, or the TUs named as `header.h@tu1.cpp,tu2.cpp`) and commits
exactly these files only if all counts are zero, with the configured
identity (`KKP_AUTHOR_NAME`/`KKP_AUTHOR_EMAIL`, default git
`user.name`/`user.email`) as author and committer, unsigned, and without
trailers.  `--style` runs `make check-whitespace` and
`make check-permissions` first.  Keep a gate call to about 4-6 `.cpp` files
in a foreground call with a 2-minute limit (each file is two clang runs of
5-20 s), or run larger batches in the background.  Push after every gate
call; the remote is the backup against container resets.

Commit messages: subject `KOKKOS: remove silent fp32/fp64 conversions in
<styles>` (mixed pass: `KOKKOS(mixed): ...`), body naming the rules applied
and "Numerics-preserving in the default double build"; say explicitly if
anything is not bit-identical.

### 10. Killed agents and resume

Agents died twice to a session usage limit and once to a tool
(classifier) outage that denied agent launches.  On resume:

    git log --oneline <base>..HEAD                   # what is committed
    git status --porcelain                           # what is in flight
    $T/survey.sh <every modified file>               # where each file stands

Commit the files that are at 0/0 through the gate; re-dispatch partial
files with the "PARTIALLY done by a previous run" variant of the prompt,
naming the files of that batch that are already done.  After a usage-limit
reset, relaunch about 4 agents first, then the rest.  Stopping an agent
that has already finished produces a harmless error.

### 11. Authoritative full rebuilds

    $T/fullsurvey.sh --rebuild --touch build-kk-single
    $T/fullsurvey.sh --rebuild --touch build-kk-mixed

`--touch` touches `src/KOKKOS/kokkos_type.h`, which every KOKKOS TU
includes, so every TU recompiles (an incremental build only reports the TUs
it recompiles).  Compare the number of recompiled KOKKOS objects with
`ninja -C <build> -t targets all | grep -cE "KOKKOS/[a-z0-9_]+\.cpp\.o:"`
and check any missing TU with `chk.sh`.  A build that ran while agents were
still editing is not authoritative; rebuild after the last commit.  The
full build, not git history, decides which files are done.  Leftovers
become a new batch.  Also compile the edited TUs once in the default double
precision (`chk.sh double <TU>`): no errors and no in-scope warnings.

### 12. Runtime smoke tests and control builds

Run the binaries, and run examples FROM INSIDE THE EXAMPLE DIRECTORY (data
and potential files are relative paths; running elsewhere fails):

    $LAMMPS/build-kk-single/lmp -h | head
    cd $LAMMPS/examples/melt && $LAMMPS/build-kk-single/lmp -k on -sf kk -in in.melt -log none
    # also: micelle/in.micelle, crack/in.crack, meam/in.meam.shear,
    #       rigid/in.rigid.small.infile, PACKAGES/brownian/2d_velocity/in.2d.velocity

Compare `/kk` against the CPU styles of the same binary (drop
`-k on -sf kk`): e.g. MEAM single -8233.07 vs CPU -8232.78 initial energy
is ordinary float rounding.

Numerical inertness is shown with a control build, not argued:

    git worktree add $KKP_WORK/ctrl <base-without-the-cast-commits> --detach
    cd $KKP_WORK/ctrl
    cmake -S cmake -B build -C cmake/presets/clang.cmake -C cmake/presets/kokkos-serial.cmake \
      -D PKG_RIGID=on -D PKG_MOLECULE=on -D KOKKOS_PREC=mixed \
      -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=off -G Ninja
    ninja -C build                      # about 8 minutes with a minimal package set
    # run the same input with both binaries, from the example directory:
    cd $LAMMPS/examples/rigid
    $KKP_WORK/ctrl/build/lmp -k on -sf kk -in in.rigid.small.infile -log none > $KKP_WORK/ctrl.out
    $LAMMPS/build-kk-mixed/lmp -k on -sf kk -in in.rigid.small.infile -log none > $KKP_WORK/new.out
    diff <(grep -E '^ +[0-9]+ ' $KKP_WORK/ctrl.out) <(grep -E '^ +[0-9]+ ' $KKP_WORK/new.out)
    cd $LAMMPS && git worktree remove $KKP_WORK/ctrl --force && git worktree prune

Enable the packages the chosen inputs need, and repeat for single.
Results of the original: over 10,000 steps the mixed thermo output matched
the control in every printed digit; in single precision only the pressure
differed, in its eighth significant digit (12.288904 in the control,
12.288905 with the casts), caused by the `v_tally` accumulator cast in
`fix_rigid_small`.

Two failures appeared during the smoke tests, with different evidence:

- `examples/PACKAGES/dpd-react/dpdrx-shardlow` with `-sf kk` fails with
  "Requested neighbor stencil method does not exist".  A control build of
  plain `develop` (KOKKOS and DPD-REACT only) reproduced it: pre-existing,
  PROVEN by the control build.
- `examples/granular/in.pour.drum` under `/kk` stops with the explicit
  error "Cannot yet use fix pour with the KOKKOS package".  That is a
  deliberate guard in the source, pre-existing by construction; no control
  build was run for it.

Also show structurally that nothing but casts changed: no files added or
removed, no `*Style(` macro lines changed, and the added lines outside
`static_cast`/`Kokkos::` categorized.

### 13. Merging upstream changes (merge-conflict procedure)

    git fetch origin develop; git rev-list --count HEAD..origin/develop
    git merge origin/develop --no-edit > $KKP_WORK/merge.log 2>&1
    git diff --name-only --diff-filter=U | tee $KKP_WORK/conflicts.txt
    for f in $(cat $KKP_WORK/conflicts.txt); do echo "$f: $(grep -c '^<<<<<<<' $f)"; done

For each conflicted file, first size both sides against the merge base
(`git diff --stat <base> origin/develop -- F`, `git diff --stat <base> HEAD
-- F`), then:

- **Upstream deleted or replaced the code that held the casts** (e.g.
  `pair_eam_alloy`/`pair_eam_fs` became short stubs): `git checkout --theirs
  F` is correct, and ONLY in this case.
- **Upstream made a small change in a file that also holds auto-merged
  casts** (e.g. two unused lines deleted in `fix_langevin_kokkos.cpp`):
  resolve IN PLACE, taking upstream's side of each conflict hunk only.
  `--theirs` on the whole file would have discarded 45 casts.  A regex that
  keeps the upstream side of every hunk and nothing else:

        re.sub(r'<<<<<<< HEAD\n.*?=======\n(.*?)>>>>>>> origin/develop\n', r'\1', s, flags=re.S)

- **Both sides changed the same lines**: write the combined text (upstream's
  logic plus the casts) by exact string replacement, with an assertion
  that the old text was found.
- **Upstream rewrote the physics**: take upstream's text, then let the
  compiler flag it again and re-clean it like new code.
- **Mechanical rename upstream** (e.g. index names): keep your side and
  apply the rename.
- NEVER resolve a whole file with `--ours` or `--theirs` without checking
  every hunk.  In a cherry-pick or rebase the meaning is reversed: "ours" is
  the upstream side.

Then `grep -rlE '^(<<<<<<<|>>>>>>>) ' src/` must be empty; `git add`,
`git commit --no-edit`, and check that both parents are ancestors
(`git merge-base --is-ancestor`).  Re-configure, rebuild both precisions,
clean the new warnings in upstream's new or changed code, full rebuild,
style checks.  A stop hook reporting hundreds of "unpushed commits" right
after the merge is counting the upstream commits that arrived through it
(813 in the first merge); push the merge.

*Learned in later rounds (Sep 2026):* when the feature branch being cleaned
is force-pushed upstream, merging the rewritten history duplicates every
rewritten commit.  Instead, re-apply only the cast commits onto the new tip
(`git cherry-pick` or `git rebase --onto`), resolving each conflict hunk by
hunk as above.  The post-compaction run that did this resolved three
conflicts with a wholesale `git checkout --ours` (in a cherry-pick, the
upstream side); it happened to be right because upstream was a superset,
but it was not checked, and it is the same hazard as a wholesale
`--theirs` in a merge.  Also, a fork's `develop` can lag the real upstream;
fetch `lammps/lammps` `develop` through a separate remote when the target
is upstream.

### 14. Squash, identity, signing, pull request, push

**Squash** the gated commits into one (or as the user asks) with
`squashverify.sh`, which records the tree hash, keeps a backup branch,
commits with the configured identity, unsigned and without trailers, and
verifies: unchanged tree hash, empty diff against the old head, exactly one
commit on the base, parent is the base, no trailer lines, author and
committer as configured, signature status N:

    git fetch origin develop
    $T/squashverify.sh origin/develop -F $KKP_WORK/squash-msg.txt
    git push --force-with-lease=<branch>:<old remote sha> origin <branch>
    git fetch origin <branch>; test "$(git rev-parse HEAD)" = "$(git rev-parse origin/<branch>)"
    git log -1 --format='%an <%ae> / %cn <%ce> / %G?'     # expect ... / N

**Attribution.**  The AI Tools Usage section of the pull request template
is the ONLY place for AI attribution: no `Co-Authored-By:`,
`Claude-Session:`, "Generated with" or similar lines in commit messages or
PR bodies.  The original's 94 commits carried such trailers, which is why
they were squashed.  Author and committer are the person the work is done
for.

**Commit signing.**  The container had `commit.gpgsign=true`,
`gpg.format=ssh`, and `user.signingkey` pointing at a container key, in
both the local and the global git configuration, so every commit was
silently signed with a key GitHub cannot verify ("Unverified" badge).  An
initial answer that nothing was signed was wrong.  Check and fix:

    git config --get commit.gpgsign; git config --get gpg.format; git config --get user.signingkey
    git log -1 --format='%G?'                        # N = unsigned
    git cat-file commit HEAD | grep -c '^gpgsig'     # 0 = unsigned
    git config commit.gpgsign false                  # (and --global, if set there)
    git commit --amend --no-gpg-sign --no-edit       # re-create an already signed tip

The scripts pass `--no-gpg-sign` and `-c commit.gpgsign=false` unless
`KKP_SIGN=1`.

**Pull request.**  Read `.github/PULL_REQUEST_TEMPLATE.md` and fill every
section.  The structure that was used:

- **Summary**: what was done, how the locations were found, file count,
  before/after unique counts per precision, "casts and local copies only;
  no-ops in double".
- **Related Issue(s)**: follow-up to lammps/lammps#4754.
- **Author(s)**, **Licensing**: as in the template.
- **Artificial Intelligence (AI) Tools Usage**: that an AI coding agent did
  the warning triage and applied the casts, working from fixed rules and
  verifying every file with the compiler in both precisions; which
  conventions and decisions were made by a human; that human review of the
  diff is expected.  Replace the template's default "no AI" text.
- **Backward Compatibility**: none affected; double build unchanged.
- **Implementation Notes**: the numbered patterns; every change that is not
  a pure cast; every change that is not bit-identical in single or mixed;
  the verification list (full rebuilds, double build, smoke tests, control
  build).
- **Post Submission Checklist**: leave unticked; those are the author's
  attestations.
- **Further Information**: scope, out-of-scope findings (preset gaps, latent
  bugs, pre-existing failures with their evidence).

After creating or updating the PR, re-read it: the PR tool appended a
"Generated with ..." footer (remove it), and GitHub strips `<KK_FLOAT>` in
plain text as an HTML tag.  Put type names in code spans
(`` `static_cast<KK_FLOAT>` ``) or write "a static_cast to KK_FLOAT".

**Push target.**  Push to your own branch; push to someone else's shared
branch only with explicit permission, and verify it is a fast-forward
(`git merge-base --is-ancestor origin/<branch> HEAD`) first.

**Push failures.**  After a token was regenerated, every push failed with
"could not read Username for 'https://github.com'".  Diagnosis: `git
ls-remote` (read) worked; any push, also to a throwaway ref, failed; a
dummy credential

    git -c credential.helper='!f(){ echo username=x-access-token; echo password=x; };f' push origin HEAD:refs/heads/tmp-probe

got "Invalid username or token" from GitHub, so the git proxy was not
injecting a credential: an environment problem, not a repository problem.
Fallback: `git format-patch <base>..HEAD -o $KKP_WORK/patch` and hand the
patch to the user.  Do not loop retries; retry once when the user reports
the credential fixed (that push succeeded).  For ordinary network errors,
a few retries with backoff are fine.

### 15. Container resets

Container resets wiped the build directories and the scratch directory
twice; after the first, the local branch was also back at `develop`.
Everything committed and pushed was safe:

    git fetch origin <branch>; git status --porcelain | wc -l; git reflog -8
    git reset --hard origin/<branch>        # only after checking there is no local work

Then re-configure (step 2), rebuild, and re-run `mkcmd.sh`.  This is why the
tooling lives in the repository and why every gated batch is pushed.

### 16. Reporting

- A short status after every wave or completion: committed and remaining
  counts.
- Tables: per file, single/mixed baseline -> final
  (`fix_brownian_kokkos.cpp | 40/50 -> 0/0`); build x errors x warnings;
  conflict x resolution; category x count x verdict.
- Always state what is authoritative (the full build), what is pre-existing
  and how that was proven (control build, or an explicit guard), what is not
  bit-identical, and which decisions need the user.
- Out-of-scope findings are reported with evidence and an offer, not fixed.

## Tool timeouts and waiting

- A foreground shell call had a 2-minute default limit and a 10-minute
  maximum.  A verify loop over too many files hit the limit; hence the
  4-6 file gate batches.
- A foreground `sleep N; cmd` was blocked by the agent harness, and a
  malformed monitor fired early.  What worked: run long jobs in the
  background with an exit marker, and wait on the task-completion
  notification or on a background loop such as
  `until grep -q "EXIT=" build.log; do sleep 20; done`.
- `chk.sh` costs about 5 s per TU, 20 s for large TUs.

## Stop hooks

The agent harness ran a stop hook at the end of turns.  Answers by type:

- **"Uncommitted changes, please commit and push"** (fires constantly while
  agents edit): run the checker over every modified file, commit only the
  files at 0/0 through the gate, and explain the rest.  Never commit
  unverified work, never blanket-commit the tree.
- **"N unpushed commits" after an upstream merge**: those are the upstream
  commits arriving with the merge; push the merge.
- **"Unverified commit ... reset the author"**: do not re-author the commits
  to the agent (that contradicts the user's authorship and the attribution
  rule), but DO check signing (step 14): the "Unverified" state came from
  the container signing key.
- **"Unpushed commit" on a shared branch**: hold, and ask where to push.

## Agents and the Agent tool

The first round was driven by general-purpose subagents launched with the
Agent tool (82 launches from the main thread; 11 more launched by two
subagents of the mixed pass, see `PROMPT_TEMPLATE.md` on nesting).  In the
later rounds (September 2026) the agent's own session instructions
restricted Agent-tool use to explicit user requests, even though the tool
was available; the agent worked alone in the main thread, and the
post-compaction run replaced the agents with pattern rewriters (below).  The
user has explicitly authorized parallel subagents for this work.  When a
session's instructions restrict the Agent tool, cite that authorization and
this workflow, or ask the user, instead of silently working alone.

## Scope rules

- Only `src/KOKKOS/`.  Kokkos library headers under `lib/kokkos`, other
  third-party code in `lib/`, non-KOKKOS sources, and the `-Wall`/`-Wextra`/
  `-pedantic` warnings are out of scope.
- Skip REAXFF (by instruction) and ML-IAP (double-only); ML-PACE was out of
  scope because its download was blocked (see step 2).
- Only the two flags.  `-Wimplicit-int-float-conversion` (int/`bigint`/
  `tagint` to float or double, e.g.
  `double delta = update->ntimestep - update->beginstep;`) is out of scope,
  and so is every warning that also appears in the default double build.
  Check with `chk.sh double <TU>` (optionally with
  `KKP_WFLAGS='Wimplicit-int-float-conversion|Wimplicit-float-conversion|Wdouble-promotion'`).
  A post-compaction run counted int-to-float warnings, reclassified them,
  and still kept 12 such fixes: scope creep against this rule.
- Change only lines the compiler flags; the only exception is a new helper
  line that a flagged fix requires (a `_kk` copy, a file-scope constant).
  When unsure whether a line needs a change, leave it alone.
- *Learned in later rounds (Sep 2026):* keep separate concerns on separate
  branches as the user directs.  In the CG-DNA round the direction was to
  fix only the package-specific warnings on the feature branch, and to fix
  unrelated drift that had accumulated in other files on a new branch off
  `develop`.
- Latent bugs found in passing (for example coordinates unpacked through
  `static_cast<tagint>` in `fix_spring_self_kokkos.cpp`, or an `MPI_DOUBLE`
  reduction into a KK_FLOAT scalar in `min_fire_kokkos.cpp`) are reported,
  not fixed in the cast change.

## Pitfalls and lessons (all of these happened)

- **A warnings-only checker hides broken code.**  One agent emptied the
  right-hand sides in `pair_coul_long_kokkos.cpp`, leaving eight
  `h_table(i) = static_cast<KK_FLOAT>();`, and reported the file clean.  The
  file no longer compiled, so it no longer warned.  The gate caught it
  (warnings 0, errors 8); the lines were restored from `git show HEAD:` by
  mapping each broken line to its table, and the checker was changed to
  print ERROR lines.  Checkers must print compile errors, and a gate must
  require zero errors.
- **Filtering to the `.cpp` basename hides header warnings.**  See
  "Headers were discovered late".  `chk.sh` prints all `src/KOKKOS` files
  reached by the TU.
- **A manifest from an incomplete build misses whole families** (step 5).
- **Wrong git baseline.**  Listing "already cleaned" files with
  `git diff develop..HEAD` was wrong because the branch had been started
  from a commit newer than the local `develop`; five files were reported done that were never cleaned.  The full
  build, not git history, is the source of truth.
- **`git checkout --theirs`/`--ours` on a whole conflicted file** discards
  the other side's changes (step 13).
- **The `kokkos-packages.cmake` preset lags new ports** (step 2).
- **Commit signing with the container key** (step 14).
- **Stop-hook pressure** (see "Stop hooks").
- **Ambiguous warning text.**  Through atomic and scatter views clang names
  the target type as plain `double` instead of `KK_ACC_FLOAT`; decide from
  the declared view type in the header.
- **Build tree and source tree must match.**  The checkers reuse the include
  paths of the build; a build configured from a different checkout makes them
  compile against the wrong headers (`mkcmd.sh` warns).
- **Nested agents that return early** (`PROMPT_TEMPLATE.md`).
- **Shared scratch names.**  Agents writing helper files with the same
  names into one shared scratch directory overwrote each other; give each
  agent its own subfolder.

## The post-compaction failure

Late in the session (September 2026) the agent's context was compacted.  The
summary kept the facts (requests, decisions, counts, PRs, the checker's
signature, the RECIPE outline) but lost the method: the scripts themselves,
the agent workflow and prompt skeleton, the per-agent verification loop and
gates, batching and the exemplar rationale, timeouts and batch sizes, the
`-MD/-MF` strip, the merge-conflict rules beyond `--theirs`, the
out-of-scope status of int-to-float warnings, and most configure flags.  It
also carried one error (that both smoke-test failures had been proven
pre-existing by control builds).

The agent rebuilt the tooling from the summary (repeating the `.o.d` error,
dropping the clang and kokkos-serial presets and the warning flags from the
configure line, and using a path-anchored grep that matched 0 of 3,099
warnings) and, instead of running per-file subagents that read each flagged
line and judged it, substituted regex auto-fixers that rewrote flagged lines
from the warning text.  Observed failure modes:

- casts that created new promotions;
- runaway nested casts, `static_cast<KK_FLOAT>(static_cast<KK_FLOAT>(...))`,
  growing with each driver iteration;
- splicing text into the middle of existing `static_cast` tokens (seven
  files no longer compiled);
- treating continuation lines of multi-line statements as complete
  statements (appending `;`);
- over-widening to `static_cast<double>`, which re-narrowed at the next
  assignment and INCREASED the warning counts about fourfold;
- a dropped `if (eflag_global)` guard;
- `using MathConst::MY_PI;` corrupted into a nonexistent `MY_PI_KK`;
- finally a stall with 4 of 106 locations fixed.

The user stopped it and asked for the original workflow to be recovered;
the session transcript was recovered from the server, and the original
tooling and prompts were extracted from it; this toolkit is the result.
Lessons:

- Never replace the agent-per-file workflow with pattern rewriting.  The
  fixes look mechanical but each needs a type decision (KK_FLOAT vs
  KK_ACC_FLOAT vs double, which operand, which side of the boundary) that
  depends on declarations elsewhere.  Scripts are fine as an edit vehicle
  for explicit per-line edit lists written after reading each line (the
  original agents used them that way, with assertions).
- Keep the tooling in the repository (this folder), not in a scratch
  directory.
- After a compaction, recover the method first (this README, RECIPE.md,
  PROMPT_TEMPLATE.md), verify the configure line against step 2, then
  continue.
- If the session's instructions seem to forbid the workflow (e.g. the Agent
  tool), say so and ask, rather than inventing a substitute.

## Session recovery

Claude Code on the web streams the session from the server; the transcript
is not a local file you can rely on:

- The server keeps every event, available page by page from
  `/v1/code/sessions/<session id>/events`.  The browser-console script in
  `stanmoore1/private:lammps/kokkos-precision/kokkos-precision-session-recovery.md`
  downloads all of it as one JSON file.
- The Claude desktop app caches the pages it has displayed.  On macOS, open
  the session, scroll to the very top so that every page is loaded, quit the
  app, and collect the cache files from
  `~/Library/Application Support/Claude/Cache/Cache_Data/` (the procedure
  in the private recovery document copies only the transcript pages).
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
user messages, in the layout of
`stanmoore1/private:lammps/kokkos-precision/history/`.  Such output is
private session material; keep it out of public repositories.

## Quick start on a new branch

    git switch -c my-kk-precision-fixes            # or the feature branch to clean
    T=tools/kokkos-precision
    $T/configure.sh single build-kk-single         # heed any "preset lacks" warning
    $T/configure.sh mixed  build-kk-mixed
    # in the background, wait for EXIT=:
    (ninja -C build-kk-single -k 0; echo "EXIT=$?") > build-kk-single/build.log 2>&1
    (ninja -C build-kk-mixed  -k 0; echo "EXIT=$?") > build-kk-mixed/build.log 2>&1
    $T/mkcmd.sh single build-kk-single
    $T/mkcmd.sh mixed  build-kk-mixed
    $T/fullsurvey.sh build-kk-single; $T/fullsurvey.sh build-kk-mixed
    for f in <flagged .cpp files>; do $T/survey.sh $f; done   # baselines
    # exemplar by hand; batch files; fill in PROMPT_TEMPLATE.md; validation
    # wave; then agents in parallel.  As each agent finishes: re-check, read
    # the diff and the judgment calls, then
    $T/vc.sh --style --commit "KOKKOS: remove silent fp32/fp64 conversions in ..." <files>
    git push origin HEAD
    # when all batches are in (background):
    $T/fullsurvey.sh --rebuild --touch build-kk-single
    $T/fullsurvey.sh --rebuild --touch build-kk-mixed
    # smoke tests and control build (step 12), squashverify.sh, PR (step 14)

If the toolkit is not on your branch, keep a separate worktree of the
toolkit branch and point the scripts at your checkout with
`KKP_REPO=/path/to/checkout`.
