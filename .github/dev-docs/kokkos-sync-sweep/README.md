# KOKKOS sync-debugging mutation sweep

The harness behind task "exhaustive mutation campaign over reachable sites".
It comments out one `sync`/`modify` call at a time in `src/KOKKOS`, screens the
result against a set of clean baselines, and -- where the fault changes the
answer -- asks the detectors described in `../kokkos-sync-debugging.md` whether
they can name it.

Everything runs out of a scratchpad directory, but the scratchpad does not
survive a container restart, so the authoritative copies live here and
`keepalive.sh` copies them out on each tick.

| Script | What it does |
|---|---|
| `keepalive.sh` | the entry point: refresh the scratchpad, restore the checkpoint, resume the poison build, launch the campaign.  Idempotent. |
| `campaign.sh` | the sweep itself: pooled screening, bisection to a single site, then diagnosis |
| `checkpoint.sh` | copy the verdicts back here and push them |
| `auditbase.sh`, `diagbase.sh` | clean-run baselines for the audit and for the watch/stale diff |
| `inject.py` | comment out one call, or put the file back |
| `runcase.sh`, `runcase_bin.sh`, `thermo.py` | run one case and reduce it to comparable thermo output |
| `report.py` | the site map |
| `cases.txt`, `cases_core.txt` | the inputs, and the cheaper subset used for screening |
| `sites_reached_filemajor.txt` | the reachable sites, ordered so a pool rebuilds few files |

`campaign.state` and `campaign.results` are the checkpoint: one line per site,
either `INERT`, `MANIFESTS-CRASH`, `MANIFESTS-DIVERGED` or `SKIP`, with the
detector verdict beside it in `campaign.results`.

## What `cases.txt` covers, and what it does not

The fourth field of a `cases.txt` line is passed to `package kokkos`, and a `-`
there means "no arguments" -- which on a build without a GPU is not neutral.  It
is the host defaults: `comm no`, `sort no`, `atom/map no`, `neigh half`,
`newton on`.  The atom exchange, the border builds, the sort and the atom map
then never touch a device view, so a site in any of them screens as `INERT`
whatever the truth is.  Only the lines carrying `gpu/aware:on:comm:device`
exercise the device communication at all, and none of the current lines asks for
`sort device`, `atom/map device`, `neigh full` or `newton off`.

Read the recorded verdicts with that in mind: `INERT` means "inert under these
cases", not "inert".  A case list meant to stand in for a GPU wants

```
... 4 neigh:full:newton:off:comm:device:sort:device:atom/map:device:gpu/aware:on
```

and `../kokkos-sync-debugging.md` explains why.

Two rules the harness enforces, both learned the hard way:

* A site is only written down once the fault was really there to be found.  If
  the injection is missing from the tree at diagnosis time, or the poison binary
  is absent, the site is left unclassified for a later pass instead of being
  recorded as something the tools missed.
* Injections are reverted before anything is compiled.  A leftover fault
  otherwise ends up inside the binary the diagnosis trusts.
