# Adversarial verification protocol

You are verifying findings from an earlier code review of the LAMMPS KOKKOS package.
Repo root is `/home/user/lammps`.  **Do NOT modify any repository file.**

Your job is **not** to agree.  Your job is to try to REFUTE each finding by reading the
actual code.  A finding that survives a genuine attempt to refute it is worth acting on; a
finding you merely failed to disprove because you did not look is worse than useless,
because someone will patch working code on the strength of it.

## For each finding

1. Open the named file and read the cited code **and its surrounding context** — the whole
   function, and the caller if the claim depends on how it is called.
   **Line numbers in the findings may be off by up to ~100 lines.  Locate the code by
   symbol name, not by line number.**  If you cannot find the described code at all, that
   is `NOT_FOUND`.
2. Open the CPU base class / sibling style the finding compares against and read the
   corresponding code there.  Most findings are "KOKKOS diverges from CPU" claims and are
   settled by reading both sides.
3. Actively look for reasons the finding is WRONG:
   - Is the divergence compensated somewhere else (a different sync, a guard in the
     caller, a base-class call, a default that makes it moot)?
   - Is the claimed-unreachable path actually unreachable, or the claimed-reachable path
     actually gated off?
   - Does the "correct sibling" it cites really do what the finding says?
   - For sync/modify claims: does some other call already mark the view modified in the
     right space?  Trace it.
   - For type-truncation claims: is the value actually ever non-integral in practice?
   - Is the base class itself doing the same thing (so it is not a port defect)?
4. Assign a verdict.

## Verdicts

- `CONFIRMED` — you read both sides and the defect is real as described. Say what the fix is.
- `CONFIRMED_ADJUSTED` — a real defect, but the description, severity, or location is
  wrong. Give the corrected version.
- `REFUTED` — the code does not do what the finding says, or it does but it is not a
  defect. **Explain what actually happens.**
- `UNCERTAIN` — you could not settle it by reading (needs a run, a build, or domain
  knowledge you do not have). Say exactly what would settle it.
- `NOT_FOUND` — the described code is not in the file under any name.

Do not use `UNCERTAIN` as a hedge for findings you did not investigate.  If you ran out of
budget, leave them out of `done.txt` so a successor picks them up.

## Output, one JSON object per finding

```json
{"id": "F0123", "file": "src/KOKKOS/foo.cpp", "verdict": "CONFIRMED",
 "real_line": 412,
 "reasoning": "What you read on each side and why it settles the question. Quote the decisive lines.",
 "severity_agreed": "high",
 "fix": "The concrete edit, precise enough to apply without re-deriving it."}
```

`severity_agreed` is your own judgement (`high`/`medium`/`low`), not necessarily the
original.  Downgrade freely: a latent issue with no reachable failure is `low`.
For `REFUTED` and `NOT_FOUND`, `fix` may be omitted.

## Checkpointing (REQUIRED — this session can be killed by a usage limit at any moment)

Your progress directory is given in your prompt.  After **every single finding**, before
starting the next one:

1. append the finding id to `done.txt`
2. append its verdict object as one line to `verdicts.jsonl`

Never batch.  First thing you do: `cat done.txt` and skip anything already listed — you may
be resuming a killed predecessor.  When your whole batch is done, `touch COMPLETE`.

At the end, also write the full verdict array to the output file named in your prompt, and
return it as your final reply.
