#!/bin/bash
# Cubic N=100 CPP at Tc=1.089, WEAK-field ladder: halve the field below dUmax=2 to
# map where the optimal IK/H mixing alpha drifts. 500k equil + 10M production, 4 ranks.
# Tags encode dUmax: cube100Tc1 (1.0), cube100Tc05 (0.5), cube100Tc025 (0.25).
set -e
cd "$(dirname "$0")"
LMP=/home/user/lammps/build/lmp
declare -A TAGS=( [1.0]=cube100Tc1 [0.5]=cube100Tc05 [0.25]=cube100Tc025 )
for dU in 1.0 0.5 0.25; do
  tag=${TAGS[$dU]}
  echo "=== launching $tag (dUmax=$dU, T=1.089) ==="
  mpirun --allow-run-as-root -np 4 "$LMP" -in in.cpp_cubic100 \
    -var tag "$tag" -var dUmax "$dU" -var T 1.089 \
    -var Nequil 500000 -var Nprod 10000000 -log "log.$tag"
done
echo "ALL CUBE100Tc WEAK RUNS DONE"
