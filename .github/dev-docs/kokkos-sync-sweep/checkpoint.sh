#!/bin/bash
# Copy the sweep's progress out of the scratchpad and into the repository, then
# push it.  The scratchpad is rolled back by a container restart; the repository
# is not, so this is what keeps a day of verdicts from being re-derived.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
REPO=/home/user/lammps
D=$REPO/.github/dev-docs/kokkos-sync-sweep
cd $REPO || exit 2

for f in campaign.state campaign.results campaign.warnings; do
  [ -f $SP/$f ] || continue
  sort -u $SP/$f > $D/$f
done

git add $D >/dev/null
git diff --cached --quiet $D && { echo "checkpoint unchanged"; exit 0; }
git commit -q -m "KOKKOS sweep: checkpoint at $(sort -u $D/campaign.state | wc -l) verdicts" -- $D
for i in 1 2 3 4; do
  git push -q origin HEAD:refs/heads/claude/lammps-kokkos-dualview-debug-t9be12 && break
  sleep $((2 ** i))
done
echo "checkpointed $(sort -u $D/campaign.state | wc -l) verdicts"
