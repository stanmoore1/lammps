#!/bin/bash
# Idempotent keepalive for the KOKKOS sync-debugging mutation sweep.
#
# This is the copy that matters.  The scratchpad the sweep works in does not
# survive a container restart: the restart rolls it back to an earlier snapshot,
# which has cost the poison binary twice and silently reverted the harness
# itself more than once.  The repository does survive, so the harness lives here
# and is copied out to the scratchpad on every tick, and the campaign's
# checkpoint is copied back here and pushed so progress survives too.
#
# Runs in the FOREGROUND: the caller is expected to be a harness-tracked
# background task, and a nohup grandchild gets reaped at tool-call teardown in
# this environment.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
REPO=/home/user/lammps
D=$REPO/.github/dev-docs/kokkos-sync-sweep
cd $REPO || exit 2

mkdir -p $SP/inj
# Refresh the harness from the repository, but never while a campaign is using
# it: replacing campaign.sh under a running bash rereads the file mid-script.
if ! pgrep -f "bash $SP/campaign.sh" >/dev/null; then
  for f in campaign.sh auditbase.sh diagbase.sh runcase.sh runcase_bin.sh \
           inject.py thermo.py report.py cases.txt cases_core.txt \
           sites_reached_filemajor.txt; do
    cmp -s $D/$f $SP/$f || cp $D/$f $SP/$f
  done
  chmod +x $SP/*.sh $SP/*.py 2>/dev/null
fi

# Seed the checkpoint from the repository when the scratchpad copy is behind:
# after a rollback it is, and re-testing sites that already have a verdict is
# hours of work for nothing.
for f in campaign.state campaign.results; do
  if [ -f $D/$f ] && [ "$(sort -u $D/$f 2>/dev/null | wc -l)" -gt "$(sort -u $SP/$f 2>/dev/null | wc -l)" ]; then
    cp $D/$f $SP/$f; echo "restored $f from the repository"
  fi
done

if grep -q "CAMPAIGN COMPLETE" $SP/campaign.log 2>/dev/null; then echo "COMPLETE"; exit 0; fi

# Any leftover injection has to be out of the tree before anything is compiled,
# or the fault is baked into the very binary the diagnosis trusts.
if ! pgrep -f "bash $SP/campaign.sh" >/dev/null; then
  for f in $(grep -l "INJECTED-BUG" src/KOKKOS/*.cpp src/KOKKOS/*.h 2>/dev/null); do
    git checkout -- "$f"
  done
fi

# The poison build runs beside the campaign rather than ahead of it: it takes
# about two hours here, and a site diagnosed before it lands is left
# unclassified by campaign.sh instead of being written down as a blind spot.
if [ ! -x $SP/build-poison/lmp ] && ! pgrep -f "cmake --build $SP/build-poison" >/dev/null; then
  ( flock -n 7 || exit 0
    cmake --build $SP/build-poison -j 2 >> $SP/build-poison.log 2>&1 ) 7>$SP/poison.lock &
  echo "poison build started in the background"
fi

# Clean-run audit baselines, one per screen case.  The audit speaks on correct
# runs too, so a finding only counts when the clean run does not already have it.
if ! grep -q AUDITBASE-DONE $SP/auditbase.log 2>/dev/null; then
  if [ -z "$(git status --porcelain src/KOKKOS)" ]; then
    echo "generating the audit baselines"
    bash $SP/auditbase.sh > $SP/auditbase.log 2>&1
  fi
fi

if pgrep -f "bash $SP/campaign.sh" >/dev/null; then
  echo "alive: $(sort -u $SP/campaign.state | wc -l) verdicts"; exit 0
fi

[ -z "$(git status --porcelain src/KOKKOS)" ] || { echo "HELD: tree dirty beyond injections"; exit 1; }
rm -f $SP/campaign.lock
echo "relaunching at $(sort -u $SP/campaign.state | wc -l) verdicts"
exec bash $SP/campaign.sh $SP/sites_reached_filemajor.txt 6 >> $SP/campaign.log 2>&1
