#!/bin/bash
# Foreground hold.  Nothing is launched or restarted here: this only blocks, so
# the session stays active and the container is not reclaimed while the sweep
# runs.  The sweep itself is started with the Bash tool's run_in_background,
# which is the only launch the harness keeps alive -- a nohup/setsid from inside
# a foreground call is reaped when that call returns.
#
# read -t on a FIFO with no writer blocks for the full timeout and returns
# non-zero: a real wait, no CPU.  The FIFO is opened read-write so open() does
# not block waiting for a writer.  Do NOT read from stdin instead -- it is
# closed, read returns instantly, and the loop spins at 100% CPU, stealing
# cores from the work being waited on.
SP=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad
S=$SP/sync/orchestrate.status
F=$SP/sync/wd.fifo; [ -p $F ] || { rm -f $F; mkfifo $F; }
exec 8<>$F

# the sweeps run the examples in place and a few rewrite files that are checked
# into the repo; those are run byproducts, so drop them rather than let them
# pile up as a dirty tree
( cd /home/user/lammps && git checkout -- examples/ 2>/dev/null )

prev=$(wc -l < $S 2>/dev/null || echo 0)
end=$((SECONDS + ${1:-560}))
while [ $SECONDS -lt $end ]; do
  [ -f $SP/sync/ALL-WORK-DONE ] && { echo "ALL WORK DONE"; break; }
  now=$(wc -l < $S 2>/dev/null || echo 0)
  [ "$now" -gt "$prev" ] && { tail -n +$((prev+1)) $S; break; }
  read -t 30 -u 8 x
done
[ -f $SP/sync/ALL-WORK-DONE ] || tail -1 $S
