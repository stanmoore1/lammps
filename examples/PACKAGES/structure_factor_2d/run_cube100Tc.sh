#!/bin/bash
# Cubic N=100 CPP runs AT the LJTS critical temperature Tc=1.089 (PeTS), rho_avg~rho_c.
# Field ladder dUmax = 2,3,4; 500k equil + 10M production each; 4 MPI ranks.
set -e
cd "$(dirname "$0")"
LMP=/home/user/lammps/build/lmp
for dU in 2.0 3.0 4.0; do
  tag="cube100Tc$(printf '%.0f' "$dU")"
  echo "=== launching $tag (dUmax=$dU, T=1.089) ==="
  mpirun --allow-run-as-root -np 4 "$LMP" -in in.cpp_cubic100 \
    -var tag "$tag" -var dUmax "$dU" -var T 1.089 \
    -var Nequil 500000 -var Nprod 10000000 -log "log.$tag"
done
echo "ALL CUBE100Tc RUNS DONE"
