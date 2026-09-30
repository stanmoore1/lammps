#!/bin/bash
# Regenerate the MD statistics of results_md_statistics.txt (about 15 min on 4 cores).
#   OXDNA=/path/to/oxDNA LMP=/path/to/lmp ./reproduce_md.sh [workdir]
# The lmpV columns need a LAMMPS with the verification switches (see README); with a stock
# LAMMPS they simply repeat lmpT (oxDNA3) / lmp (oxDNA2).
set -e
H=$(cd "$(dirname "$0")" && pwd)
W=${1:-$PWD/md_work}
export SEQ=${SEQ:-$H/../oxdna_kokkos/params/oxDNA3_sequence_dependent_parameters.txt}
: "${OXDNA:?}" "${LMP:?}"
mkdir -p $W
for s in duplex12 nick16 blunt16 bulge12 ss20; do python3 $H/tools/build.py $s $W/$s/start; done
# the bulge starts with overlapping excluded volumes: relax it with MC first
cd $W/bulge12/start
cat > in_relax <<EOT
backend = CPU
sim_type = MC
ensemble = NVT
interaction_type = DNA2
use_average_seq = 1
T = 0.1
salt_concentration = 0.5
steps = 20000
delta_translation = 0.02
delta_rotation = 0.04
verlet_skin = 0.3
topology = top.top
conf_file = conf.dat
trajectory_file = /dev/null
lastconf_file = conf_md.dat
energy_file = en_relax.dat
print_energy_every = 2000
print_conf_interval = 100000
seed = 7
restart_step_counter = 1
time_scale = linear
EOT
"$OXDNA" in_relax > log_relax 2>&1
for s in duplex12 nick16 blunt16 ss20; do cp $W/$s/start/conf.dat $W/$s/start/conf_md.dat; done
# 1e6 steps of Brownian MD per structure and model (200 frames), then every frame through both codes
jobs="duplex12:2:0.1 duplex12:3:0.1 duplex12:2:0.12 duplex12:3:0.12 nick16:2:0.1 nick16:3:0.1 blunt16:2:0.1 blunt16:3:0.1 bulge12:2:0.1 bulge12:3:0.1 ss20:2:0.1 ss20:3:0.1"
for j in $jobs; do
  IFS=: read s m t <<< "$j"; tag=m${m}_T$t; cd $W/$s
  python3 $H/tools/mk_oxdna_input.py $m md start/top.top start/conf_md.dat $tag 1000000 $t 1${m}7 $SEQ > in_md_$tag
  "$OXDNA" in_md_$tag > log_md_$tag 2>&1
  python3 $H/tools/eval_traj.py $m start/top.top traj_$tag.dat $W/$s/eval_$tag 1 4
done
cd $W && python3 $H/tools/analyze.py */eval_*/summary.txt
