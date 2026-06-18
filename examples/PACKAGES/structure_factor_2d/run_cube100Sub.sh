#!/bin/bash
# Cubic N=100 CPP at 0.9*Tc = 0.980 (SUBCRITICAL), field ladder dUmax = 1,2,3.
# 500k equil + 10M production, 4 ranks. Tags cube100Sub{1,2,3}.
set -e
cd "$(dirname "$0")"
LMP=/home/user/lammps/build/lmp
for dU in 1.0 2.0 3.0; do
  tag="cube100Sub$(printf '%.0f' "$dU")"
  echo "=== launching $tag (dUmax=$dU, T=0.980) ==="
  mpirun --allow-run-as-root -np 4 "$LMP" -in in.cpp_cubic100 \
    -var tag "$tag" -var dUmax "$dU" -var T 0.980 \
    -var Nequil 500000 -var Nprod 10000000 -log "log.$tag"
done
echo "ALL CUBE100Sub RUNS DONE"
