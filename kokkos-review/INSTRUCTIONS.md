# KOKKOS package line-by-line review: reviewer instructions

You are reviewing part of the LAMMPS KOKKOS accelerator package (`src/KOKKOS/` in
/home/user/lammps) ahead of a stable release. The goal is to find REAL BUGS: code that
produces wrong numbers, crashes, races, leaks, or diverges from the CPU base style it
mirrors. Style nits are much lower priority; only report a style/convention issue if it
is a project rule violation (listed below) or hides a real bug.

## How to work

1. Read EVERY file in your assigned list completely (use `cat -n`/`sed -n` with line
   numbers; large files in chunks). Do not skim, do not sample. Line-by-line.
2. For each `_kokkos` style, open the CPU base class it derives from (usually
   `src/<name>.cpp` or `src/<PACKAGE>/<name>.cpp`; find with
   `find src -name '<name>.cpp' -not -path '*/KOKKOS/*'`) and compare the ported
   math/logic against the original: coefficient handling, cutoffs, special_lj/special_coul
   handling, energy/virial tallies (ev_tally, v_tally, eatom/vatom), newton_pair / half vs
   full neighbor-list handling, restart read/write, coeff/settings parsing, init_one,
   pack/unpack comm buffers, `extract()`, per-atom array grow/copy/exchange.
3. Read `.github/instructions/kokkos.instructions.md` once for the project rules. Key ones:
   - `if (copymode) return;` first line of every base-class destructor a kokkos style inherits.
   - base class must declare `virtual void allocate()` when the kokkos subclass overrides it.
   - per-atom views assigned from `atomKK->k_<field>.view<DeviceType>()` must be `typename AT::t_...`,
     never `DAT::t_...` (DAT is only OK for self-allocated dual views).
   - device kernels must use Kokkos:: math (no std::pow/exp/sqrt, no powint()).
   - datamask_read / datamask_modify must match what the style actually reads/writes;
     `atomKK->sync(...)` before reading and `atomKK->modified(...)` after writing per-atom data.
   - DualView sync/modify discipline: after writing host side of a dual view call
     `modify<LMPHostType>()`/`modify_host()` and `sync<DeviceType>()` before device use, and
     vice versa; look for stale-data bugs (writes without modify, reads without sync).
   - No alternative logical tokens (`and`/`or`/`not`), no VLAs, fmtlib not sprintf.
   - Explicit template instantiation at bottom of .cpp (`template class X<LMPDeviceType>;`
     and `LMPHostType` under `#ifdef LMP_KOKKOS_GPU`).
4. Typical KOKKOS bug shapes to hunt, from past bug fixes in this package:
   - `k_foo.template sync<DeviceType>()` missing after host update of coefficients
     (`init_one`, `coeff`, `read_restart`, `init_style`, `setup`).
   - Missing `k_foo.template modify<LMPHostType>()` after filling host mirror.
   - Wrong `eflag_atom`/`vflag_atom` handling; `k_eatom`/`k_vatom` not resized when
     `atom->nmax` grows (`maxeatom`/`maxvatom`); missing `modified()` after ev tallies.
   - `d_neighbors`/`d_ilist` used with wrong list (`list->inum` vs `nlocal`), half list
     with `newton_pair=0`, full list double counting, `NEIGHFLAG` template branches
     with asymmetric energy/virial factors (0.5 vs 1.0).
   - Atomic vs non-atomic force accumulation with `NEIGHFLAG==HALF`/`HALFTHREAD`;
     `a_f` (atomic view) vs `f` for the j-atom update; missing `Kokkos::fence()`.
   - Race conditions in team/scratch kernels; scratch size mismatches; uninitialized
     scratch/shared arrays.
   - Integer overflow: `int` used for products of nlocal*nmax, `bigint`/`tagint` truncation,
     `MAXSMALLINT`, `imageint` packing.
   - Off-by-one in `n+1` sized type arrays (`ntypes+1`), `atom->nmax` vs `nlocal+nghost`.
   - `nlocal` captured before `atomKK->sync` / grown arrays; view members assigned from
     `atomKK->k_x` before `sync` or reused after `atom->nmax` changed (stale views).
   - `copymode = 1` not set before `parallel_for` with `*this` functor, or destructor
     freeing memory when `copymode` (double free).
   - Restart: `read_restart` allocating new dual views but `d_` device views not reassigned.
   - `init_one` returning value from unsynced arrays; `cutsq` dual view stale.
   - Wrong `special_lj[]`/`special_coul[]` factor use; `factor_lj` applied twice or never.
   - Comm: `pack_forward_comm_kokkos` / `unpack_*` buffer size (`comm_forward` count)
     mismatched with kokkos version; `pack_border`/`unpack_border` fields differing from
     CPU `atom_vec`; `pack_exchange` ordering vs `unpack_exchange`.
   - fix: `post_force` respa levels (`ilevel_respa`), `min_post_force`, `setup` calling
     `post_force` with `vflag`; virial contributions (`v_init`, `virial_fdotr`, `vflag_global`).
   - MPI reductions on wrong datatype (`MPI_DOUBLE` for `bigint`), `MPI_Allreduce` on
     non-device-synced value.
   - Memory: `memoryKK->destroy_kokkos` mismatch with `create_kokkos`; leaking `k_` views
     across `allocate()` re-entry; raw `new[]` without delete.
   - Floating point: `KK_FLOAT`/`F_FLOAT` mixing causing precision loss in accumulators
     (e.g. `float` accumulator with `LMP_KOKKOS_SINGLE`), `X_FLOAT` vs `double`.
   - Wrong sign / index in derivative expressions vs CPU version (compare every formula).
   - Kernels indexed by `i < inum` but writing `i` into arrays sized `nlocal`.
   - Missing `if (!force->newton_pair)` / `neighbor->ago` / `list->ghost` handling.
   - PAIR STYLES ONLY: `pair_compute<>()` (pair_kokkos.h:984-1001) sets
     `fuse_force_clear_flag = 1` when `neighflag == FULL` (non-hybrid), and VerletKokkos then
     SKIPS force_clear() and relies on the PairComputeFunctor's ZEROFLAG branch to zero f(i)
     for each i in the neighbor list right before accumulating. Consequences to check in every
     pair style that calls `pair_compute<>`: (a) any force contribution the style adds in its
     OWN kernel BEFORE the pair_compute<> call is wiped in fused mode; (b) any per-atom array
     other than f that the style expects VerletKokkos::force_clear() to zero (torque is
     covered by the fuse guard, but e.g. a style-owned accumulator is not) is not zeroed;
     (c) a style that calls pair_compute<> more than once per step (e.g. once per sub-term)
     has the first call's forces wiped by the second. Also check styles that set neighflag
     FULL and write ghost forces (must not: newton is forced off with FULL).
5. Keep a running log of candidates. Before reporting, RE-VERIFY each one against the
   code: quote the exact lines and the base-class lines that disagree. Drop anything you
   cannot substantiate. False positives waste release-crunch time; be honest about
   confidence.

## Checkpointing (REQUIRED: the session can be killed by usage limits at any moment)

Your group id G is the number in your `group_G.txt` file name (rule audits use their
letter, e.g. `rules_A`). Your progress directory is
`/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_G/`
(create it with `mkdir -p` if missing). It holds three files:

- `done.txt`: one file name per line for every assigned file you have finished reviewing.
- `findings.jsonl`: one JSON finding object per line (same schema as the output format below).
- `notes.txt`: free-form candidates you still want to re-verify, and anything a successor
  needs to know (e.g. "base class for X is src/EXTRA-PAIR/x.cpp", "already compared
  pack_border of A and B", "scan script saved as progress_G/scan.py, results in scan.json").

Protocol:
1. FIRST THING: `cat progress_G/done.txt progress_G/notes.txt progress_G/findings.jsonl`
   if they exist. Skip every file listed in done.txt (it is already reviewed). You are
   possibly a restarted agent continuing a killed predecessor's work; trust the checkpoint.
2. Work one style (its .cpp and .h together) at a time. IMMEDIATELY after finishing each
   style, and BEFORE opening the next one: append its file names to done.txt (`echo >>`),
   append each substantiated finding as ONE line of JSON to findings.jsonl (`cat >>` with a
   heredoc, one object per line), and append open candidates to notes.txt. Never batch
   several styles before checkpointing. A kill between checkpoints must lose at most one
   style of work. Rule audits: save every script and its raw output in progress_G/ and
   record in notes.txt which rule/sub-step is finished.
3. At the end, your final JSON array is the contents of findings.jsonl (deduplicated by
   file+line), written to `findings_G.json` as described below, followed by the COVERAGE
   line. When every assigned file is in done.txt, also `touch progress_G/COMPLETE`.

## Output format

Return ONLY a JSON array (no prose before or after) of findings. Each finding:

{
  "file": "src/KOKKOS/xxx_kokkos.cpp",
  "line": 123,
  "severity": "high" | "medium" | "low",
  "category": "correctness" | "sync-modify" | "race" | "memory" | "overflow" | "rule" | "efficiency" | "cleanup",
  "confidence": "confirmed" | "likely" | "possible",
  "summary": "one sentence stating the defect",
  "evidence": "the exact lines (quoted) and the base-class lines they disagree with, with paths and line numbers",
  "failure_scenario": "concrete input/state -> wrong output/crash",
  "suggested_fix": "one or two sentences"
}

Order by severity, then confidence. Include at most 40 findings; if you have more, keep
the ones with the strongest evidence. If you found nothing substantiated, return `[]`.
Also append, after the JSON array, a single line starting with `COVERAGE:` listing any
assigned files you did NOT read completely (say `COVERAGE: all files read completely` if
you read everything).
