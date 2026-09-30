#!/bin/bash
# Make a benchmark case: an oxDNA system tiled n x n x n, one oxDNA-style `input` that runs
# unchanged with standalone oxDNA (CUDA) and both Kokkos benches, and the same system as a
# LAMMPS data file for the LAMMPS KOKKOS reference.
#   ./make_case.sh <model 2|3> <base N8|N64|N512> <n> <outdir> [steps]
# Sizes: N8 = 128 nt, N64 = 1024 nt, N512 = 8192 nt; n = 2 gives 8x that, n = 4 64x.
set -e
H=$(cd "$(dirname "$0")" && pwd); B=$(cd $H/.. && pwd); R=$(cd $B/.. && pwd)
m=$1; base=$2; n=$3; out=$4; steps=${5:-10000}
src=$B/oxdna_kokkos/tests/$base
mkdir -p $out; out=$(cd $out && pwd)
python3 $H/tools/tile.py $src/topology_$base.top $src/init_conf_$base.dat $n $out/top.top $out/conf.dat
python3 $H/tools/ox2lmp.py $out/top.top $out/conf.dat $out/data.lmp > /dev/null
if [ $m = 3 ]; then
  inter="interaction_type = DNA3
use_average_seq = 0
seq_dep_file = $B/oxdna_kokkos/params/oxDNA3_sequence_dependent_parameters.txt"
  cp $R/potentials/oxdna3_lj.cgdna $out/
else
  inter="interaction_type = DNA2"
fi
cat > $out/input <<EOT
# $base x ${n}^3, oxDNA$m, NVE. Same file for standalone oxDNA, bench/oxdna_kokkos and
# bench/oxdna_kokkos_lammps (the benches ignore the backend keys).
backend = CUDA
backend_precision = mixed
CUDA_list = verlet
sim_type = MD
$inter
T = 0.1
salt_concentration = 0.5
dt = 0.003
steps = $steps
thermostat = no
verlet_skin = 0.5
refresh_vel = 1
restart_step_counter = 1
seed = 12345
topology = top.top
conf_file = conf.dat
trajectory_file = /dev/null
lastconf_file = /dev/null
energy_file = energy.dat
print_energy_every = ${steps}
print_conf_interval = 100000000
time_scale = linear
EOT
# LAMMPS-faithful bench in its most faithful mode (LAMMPS framework on)
{ cat $out/input; echo "lammps_overhead = 1"; } > $out/input_lammps_mode
cp $H/lammps/in.perf$m $out/in.lammps
echo "case $out: $(head -1 $out/top.top | awk '{print $1}') nucleotides, model oxDNA$m, $steps steps"
