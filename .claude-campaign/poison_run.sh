#!/bin/bash
# poison_run.sh <dir> <input> <np> <outfile> [pk args]
# Run one case under the poison build in survey mode: every stale access is
# logged and the run keeps going, so one pass collects the whole list instead
# of stopping at the first.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
L=$SP/build-poison/lmp
cd /home/user/lammps/examples/$1 || exit 99
PK="$5"
export LMP_KOKKOS_POISON=1
export ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:print_stacktrace=1:log_path=stderr
if [ "$3" -gt 1 ]; then
  mpirun --allow-run-as-root --oversubscribe -np $3 $L -in $2 -log none -screen "$4" -k on -sf kk ${PK:+-pk kokkos $PK} >"$4.err" 2>&1 </dev/null
else
  $L -in $2 -log none -screen "$4" -k on -sf kk ${PK:+-pk kokkos $PK} >"$4.err" 2>&1 </dev/null
fi
