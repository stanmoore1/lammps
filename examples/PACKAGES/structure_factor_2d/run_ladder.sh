#!/bin/bash
# CPP LJTS field-ladder: 3 temperatures x 3 field strengths = 9 runs.
# Each run: INTEL-package equilibration (double precision) -> restart -> OPENMP
# production (double precision; the Harasima contour needs per-atom virials that the
# INTEL package cannot provide).  Writes a per-temperature manifest for kernel_fit.py.
#
# Usage: bash run_ladder.sh   (run from examples/PACKAGES/structure_factor_2d/)
set -u
LMP=/home/user/lammps/build/lmp
NEQ=${NEQ:-1000000}
NPR=${NPR:-1000000}
NTH=${NTH:-4}
RHO=0.31
export OMP_NUM_THREADS=$NTH

# temperatures: 0.9 Tc, Tc, 1.1 Tc  (PeTS Tc=1.089)
TEMPS=(0.980 1.089 1.198)
DUS=(0.2 0.4 0.8)
SEED=1000

for T in "${TEMPS[@]}"; do
  manifest="ladder_T${T}.csv"
  echo "# dumax,rho_avg,density_file" > "$manifest"
  for DU in "${DUS[@]}"; do
    SEED=$((SEED+1))
    TAG="T${T}_d${DU}"
    echo "=== $(date +%H:%M:%S)  RUN $TAG  (equil INTEL) ==="
    $LMP -sf intel -pk intel 0 omp $NTH mode double -in in.cpp_ljts_equil \
         -var T "$T" -var dUmax "$DU" -var rho $RHO -var seed $SEED -var tag "$TAG" \
         -var Nequil $NEQ > "${TAG}_equil.log" 2>&1
    echo "=== $(date +%H:%M:%S)  RUN $TAG  (prod OPENMP) ==="
    $LMP -sf omp -pk omp $NTH -in in.cpp_ljts_prod \
         -var T "$T" -var dUmax "$DU" -var tag "$TAG" \
         -var Nprod $NPR > "${TAG}_prod.log" 2>&1
    echo "${DU},${RHO},${TAG}_dens.out" >> "$manifest"
    rm -f "${TAG}.equil.restart"
  done
  echo "wrote $manifest"
done
echo "ALL LADDER RUNS DONE $(date +%H:%M:%S)"
