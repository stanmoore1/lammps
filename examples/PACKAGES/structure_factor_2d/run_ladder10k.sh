#!/bin/bash
# CPP LJTS 10k-atom CUBIC-box field ladder for the DCF/structure-factor study:
# T = 0.9 Tc (0.980), field ladder dUmax = 0.2 / 0.4 / 0.8.
# Each run: MPI equilibration (1e5 steps) -> restart -> MPI production (1e7 steps)
# with the bin-resolved structure factor + density + H/IK contours + gamma.
# Pure MPI, 2x2x1 lateral decomposition (balanced for the z-pinned slab).
#
# Usage: bash run_ladder10k.sh
#        DUS="0.4" bash run_ladder10k.sh        (single field)
set -u
LMP=/home/user/lammps/build/lmp
NEQ=${NEQ:-100000}
NPR=${NPR:-10000000}
RANKS=${RANKS:-4}
T=${T:-0.980}
export OMP_NUM_THREADS=1
MPI="mpirun --allow-run-as-root -np $RANKS"
read -ra DUS <<< "${DUS:-0.2 0.4 0.8}"
SEED=2000
manifest="ladder10k_T${T}.csv"
echo "# dumax,rho_avg,density_file,sf_file" > "$manifest"

for DU in "${DUS[@]}"; do
  SEED=$((SEED+1))
  TAG="X10k_T${T}_d${DU}"
  echo "=== $(date +%H:%M:%S)  $TAG  equil ($NEQ steps, $RANKS ranks) ==="
  $MPI $LMP -in in.cpp_ljts_equil10k \
       -var T "$T" -var dUmax "$DU" -var seed $SEED -var tag "$TAG" \
       -var Nequil $NEQ > "${TAG}_equil.log" 2>&1
  echo "=== $(date +%H:%M:%S)  $TAG  prod ($NPR steps, $RANKS ranks) ==="
  $MPI $LMP -in in.cpp_ljts_dcf10k \
       -var T "$T" -var dUmax "$DU" -var tag "$TAG" \
       -var Nprod $NPR > "${TAG}_prod.log" 2>&1
  echo "${DU},0.31,${TAG}_dens.out,${TAG}_sf.out" >> "$manifest"
  rm -f "${TAG}.equil.restart"
  echo "=== $(date +%H:%M:%S)  $TAG  DONE ==="
done
echo "wrote $manifest"
echo "ALL 10k LADDER RUNS DONE $(date +%H:%M:%S)"
