# KOKKOS split-memory sync debugging harness

Drives the example suite under the KOKKOS sync-debugging build options and
triages what the detectors report.  Kept here rather than in a scratch
directory because a scratch directory does not survive a container restart or a
filesystem rollback, and these scripts encode a fair amount of hard-won detail.

## Builds

| dir | options |
|---|---|
| `build-sync` | `KOKKOS_DEBUG_SYNC=on` -- audit and stale watch |
| `build-poison` | the above plus `KOKKOS_DEBUG_SYNC_ASAN=on` and `-fsanitize=address` |
| `build-sync-mixed` / `build-poison-mixed` | as above with `KOKKOS_PREC=mixed` |
| `build-plain` | plain KOKKOS, regular memory -- the divergence baseline |
| `build-asan-kk` / `build-asan-cpu` | plain AddressSanitizer, with and without KOKKOS |

## Detectors

- `LMP_KOKKOS_AUDIT=1` -- compares array contents across a style call against
  what the style declared.  Note that its "reads stale X" line means *X is not
  in this style's datamask_read and the device copy is behind*, NOT that the
  style read it; a style that never touches X reports too, and so does any
  style running with `execution_space == Host`.
- `LMP_KOKKOS_POISON=1` -- poisons whichever side is not authoritative, so any
  stale dereference traps in AddressSanitizer with a full backtrace.  This is
  the load-bearing one; it catches raw-pointer reads the accessors cannot see.
- `LMP_KOKKOS_WATCH=` / `LMP_KOKKOS_STALE=` / `LMP_KOKKOS_STALE_STRICT=1` --
  reports an access to the stale side, naming the reading function.
- `LMP_KOKKOS_TRACE=<label substring>` -- prints every coherence call on the
  matching views.  This is what identifies the *claimer* when poison mode has
  told you the reader; run it alongside poison and read the ops immediately
  before the fault.

## Running it

Launch `pipeline.sh` with the Bash tool's `run_in_background`, NOT with
`nohup ... &` from inside a foreground call -- a process detached that way is
reaped when the foreground call returns.  Then hold the foreground with
`wd.sh`, which only blocks: the container is reclaimed on inactivity, and
background work does not count as activity, so the sweep needs a foreground
call to survive.  `wd.sh` blocks on `read -t` against a FIFO opened read-write,
which is a real wait with no CPU.  Reading stdin instead spins at 100% and
steals cores from the sweep.

`pipeline.sh` is idempotent: every stage writes a marker when it finishes and
`run-sweep.sh` resumes from its own index, so a restart picks up where it
stopped rather than repeating.

## compare-builds.sh / compare-thermo.py

Compares the shared-memory build against the split-memory one over a list of
inputs.  Use these rather than diffing two logs by hand: both guards exist
because their absence produced findings that were not real.

    tools/kokkos-sync-debug/compare-builds.sh inputs.txt [outdir]

Each build runs twice, so an input that does not reproduce itself is reported
(PLAIN-NONDET / SYNC-NONDET) instead of being compared against one arbitrary run
of the other side -- examples/VISCOSITY/in.nemd.2d is nonreproducible about one
pair in four, with or without KOKKOS, and was written up as a divergence on the
strength of a single pair.  And wall-clock columns (S/CPU and friends) are
dropped before comparing, since they differ on every run -- examples/wall/in.wall.sphere
differs in nothing else.

NP and TIMEOUT override the defaults of 4 ranks and 900 s.
