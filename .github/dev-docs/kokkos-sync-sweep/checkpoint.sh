#!/bin/bash
# Copy the sweep's progress out of the scratchpad and push it.
#
# The push goes through the wt-checkpoint worktree, which is the one on the
# tooling branch.  The main worktree is on a different line of development, so
# committing there and pushing HEAD to the tooling branch is a non-fast-forward
# that git refuses -- and the first version of this script printed success
# anyway, so two checkpoints looked saved when neither had left the machine.
set -u
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
WT=$SP/wt-checkpoint
BRANCH=claude/lammps-kokkos-dualview-debug-t9be12
D=$WT/.github/dev-docs/kokkos-sync-sweep

[ -d "$WT/.git" ] || [ -f "$WT/.git" ] || { echo "checkpoint: $WT is not a worktree"; exit 2; }
cd $WT || exit 2

# Take whatever the remote has first: the container restart rolls local git
# state back too, and a checkpoint that starts from a stale branch cannot be
# pushed without discarding somebody else's commits.
git fetch -q origin $BRANCH || { echo "checkpoint: fetch failed"; exit 1; }
git merge --ff-only FETCH_HEAD -q 2>/dev/null

for f in campaign.state campaign.results campaign.warnings; do
  [ -f $SP/$f ] && sort -u $SP/$f > $D/$f
done

git add $D >/dev/null
if git diff --cached --quiet $D; then echo "checkpoint unchanged"; exit 0; fi

n=$(sort -u $D/campaign.state | wc -l)
git commit -q -m "KOKKOS sweep: checkpoint at $n verdicts" -- $D || { echo "checkpoint: commit failed"; exit 1; }

for i in 1 2 3 4; do
  if git push -q origin HEAD:refs/heads/$BRANCH 2>/dev/null; then
    echo "checkpointed $n verdicts"; exit 0
  fi
  sleep $((2 ** i))
done
echo "checkpoint: PUSH FAILED at $n verdicts -- the commit is local only"
exit 1
