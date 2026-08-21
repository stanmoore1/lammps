#!/bin/bash
# runcase.sh <dir> <input> <np> <outfile> [pk args]
# Multi-rank stderr goes through per-rank files and is concatenated afterwards:
# four ranks writing one stream interleave mid-line, and the detector reports
# then parse as garbage.  This is the same fix the ASan logs already needed.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
cd /home/user/lammps/examples/$1 || exit 99
PK="$5"
if [ "$3" -gt 1 ]; then
  D=$(mktemp -d /tmp/rcout.XXXXXX)
  timeout 900 mpirun --allow-run-as-root --oversubscribe -np $3 --output-filename $D \
    timeout 900 $SP/build-rigdbg/lmp -in $2 -log none -screen "$4" -k on -sf kk ${PK:+-pk kokkos $PK} >"$4.err" 2>&1 </dev/null
  rc=$?
  cat $D/1/rank.*/stderr >> "$4.err" 2>/dev/null
  cat $D/1/rank.*/stdout >> "$4.err" 2>/dev/null
  rm -rf $D
  exit $rc
else
  timeout 900 $SP/build-rigdbg/lmp -in $2 -log none -screen "$4" -k on -sf kk ${PK:+-pk kokkos $PK} >"$4.err" 2>&1 </dev/null
fi
