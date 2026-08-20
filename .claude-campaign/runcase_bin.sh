#!/bin/bash
# runcase_bin.sh <binary> <dir> <input> <np> <outfile> [pk args]
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
L=$1; shift
cd /home/user/lammps/examples/$1 || exit 99
PK="$5"
if [ "$3" -gt 1 ]; then
  mpirun --allow-run-as-root --oversubscribe -np $3 $L -in $2 -log none -screen "$4" -k on -sf kk ${PK:+-pk kokkos $PK} >"$4.err" 2>&1 </dev/null
else
  $L -in $2 -log none -screen "$4" -k on -sf kk ${PK:+-pk kokkos $PK} >"$4.err" 2>&1 </dev/null
fi
