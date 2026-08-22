#!/bin/bash
# RUNCASE_TIMEOUT bounds one run.  Screening passes a short one: a clean case
# takes about twenty seconds here, an injected fault that is going to crash
# takes a few minutes at worst, and anything past that is a hang that would
# otherwise cost the full quarter hour before the pool could move on.  Diagnosis
# leaves it at the default, where the poison and watch passes are legitimately
# slow.
: "${RUNCASE_TIMEOUT:=900}"
# runcase.sh <dir> <input> <np> <outfile> [pk args]
# Multi-rank stderr goes through per-rank files and is concatenated afterwards:
# four ranks writing one stream interleave mid-line, and the detector reports
# then parse as garbage.  This is the same fix the ASan logs already needed.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
cd /home/user/lammps/examples/$1 || exit 99
PK="$5"
if [ "$3" -gt 1 ]; then
  D=$(mktemp -d /tmp/rcout.XXXXXX)
  timeout $RUNCASE_TIMEOUT mpirun --allow-run-as-root --oversubscribe -np $3 --output-filename $D \
    timeout $RUNCASE_TIMEOUT $SP/build-rigdbg/lmp -in $2 -log none -screen "$4" -k on -sf kk ${PK:+-pk kokkos $PK} >"$4.err" 2>&1 </dev/null
  rc=$?
  cat $D/1/rank.*/stderr >> "$4.err" 2>/dev/null
  cat $D/1/rank.*/stdout >> "$4.err" 2>/dev/null
  rm -rf $D
  exit $rc
else
  timeout $RUNCASE_TIMEOUT $SP/build-rigdbg/lmp -in $2 -log none -screen "$4" -k on -sf kk ${PK:+-pk kokkos $PK} >"$4.err" 2>&1 </dev/null
fi
