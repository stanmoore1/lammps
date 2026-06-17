#!/bin/bash
# CPP LJTS field-ladder: 3 temperatures x 3 field strengths = 9 runs.
# Each run: MPI equilibration (double precision) -> restart -> MPI production
# (double precision; the Harasima contour needs the per-atom virials that the plain
# lj/cut style provides).  Writes a per-temperature manifest for kernel_fit.py.
#
# Runs as pure MPI: each job uses $RANKS ranks (default 4), one job at a time.
#
# Usage: bash run_ladder.sh            (run from examples/PACKAGES/structure_factor_2d/)
#        TEMPS="0.980" bash run_ladder.sh     (subset of temperatures, e.g. lowest only)
#        RANKS=4 NEQ=1000000 NPR=1000000 bash run_ladder.sh
set -u
LMP=/home/user/lammps/build/lmp
NEQ=${NEQ:-1000000}
NPR=${NPR:-1000000}
RANKS=${RANKS:-4}
RHO=0.31
export OMP_NUM_THREADS=1
MPI="mpirun --allow-run-as-root -np $RANKS"

# temperatures: 0.9 Tc, Tc, 1.1 Tc  (PeTS Tc=1.089).  Override via TEMPS env var.
read -ra TEMPS <<< "${TEMPS:-0.980 1.089 1.198}"
# field strengths dUmax; override via DUS env var (e.g. stronger fields near Tc).
read -ra DUS <<< "${DUS:-0.2 0.4 0.8}"
SEED=1000

for T in "${TEMPS[@]}"; do
  manifest="ladder_T${T}.csv"
  echo "# dumax,rho_avg,density_file" > "$manifest"
  for DU in "${DUS[@]}"; do
    SEED=$((SEED+1))
    TAG="T${T}_d${DU}"
    echo "=== $(date +%H:%M:%S)  RUN $TAG  (equil, $RANKS ranks) ==="
    $MPI $LMP -in in.cpp_ljts_equil \
         -var T "$T" -var dUmax "$DU" -var rho $RHO -var seed $SEED -var tag "$TAG" \
         -var Nequil $NEQ > "${TAG}_equil.log" 2>&1
    echo "=== $(date +%H:%M:%S)  RUN $TAG  (prod, $RANKS ranks) ==="
    $MPI $LMP -in in.cpp_ljts_prod \
         -var T "$T" -var dUmax "$DU" -var tag "$TAG" \
         -var Nprod $NPR > "${TAG}_prod.log" 2>&1
    echo "${DU},${RHO},${TAG}_dens.out" >> "$manifest"
    rm -f "${TAG}.equil.restart"
  done
  echo "wrote $manifest"
done
echo "ALL LADDER RUNS DONE $(date +%H:%M:%S)"
