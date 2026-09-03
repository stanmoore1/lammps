# KOKKOS package release review (work in progress)

Snapshot of the line-by-line review of src/KOKKOS driven by Claude Code agents.

- STATE.md: orchestrator log and how to resume.
- INSTRUCTIONS.md, LAUNCH.md: reviewer protocol (checkpointing) and launch prompt.
- group_NN.txt: file partition (30 groups); progress_NN/: per-group checkpoints
  (done.txt, findings.jsonl, notes.txt, COMPLETE marker).
- findings_NN.json / findings_rules_X.json: per-group and rule-audit results
  (JSON array followed by a COVERAGE line).
- findings_orchestrator.json: findings re-verified by direct code reading.
