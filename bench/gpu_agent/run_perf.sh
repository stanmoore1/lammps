#!/bin/bash
# Run one case (from make_case.sh) with every code whose binary is set, and print timesteps/s.
#   OXDNA=<oxDNA>  OXKK=<bench/oxdna_kokkos binary>  OXKKL=<bench/oxdna_kokkos_lammps binary>
#   LMP=<LAMMPS binary>  ./run_perf.sh <casedir> [steps]
# Unset variables are skipped. LAMMPS runs with ${LMP_ARGS:-"-k on g 1 -sf kk -pk kokkos neigh half newton on comm device"}.
# Logs: <casedir>/log_<code>.txt. The step-0 potential energy per nucleotide is printed as a
# sanity check (all codes start from the same configuration; oxDNA3 LAMMPS differs from the
# others by ~1e-4 because of the physics differences in ../oxdna_lammps_vs_standalone).
set -e
c=$(cd "$1" && pwd); steps=${2:-}
cd $c
[ -n "$steps" ] && sed -i "s/^steps = .*/steps = $steps/; s/^print_energy_every = .*/print_energy_every = $steps/" input input_lammps_mode
steps=$(awk -F= '/^steps/{gsub(/ /,"",$2); print $2}' input)
N=$(head -1 top.top | awk '{print $1}')
LMP_ARGS=${LMP_ARGS:-"-k on g 1 -sf kk -pk kokkos neigh half newton on comm device"}
res=()
run() { local tag=$1; shift; echo "== $tag: $*" >&2; "$@" > log_$tag.txt 2>&1 || { echo "   FAILED (see $c/log_$tag.txt)" >&2; return 1; }; }
if [ -n "$OXDNA" ]; then
  run oxdna "$OXDNA" input && {
    ms=$(grep -o 'per step: [0-9.e+-]* ms' log_oxdna.txt | awk '{print $3}')
    res+=("oxDNA (standalone CUDA)|$(awk -v m=$ms 'BEGIN{printf "%.1f", 1000/m}')|$(head -1 energy.dat 2>/dev/null | awk '{print $2}')"); cp energy.dat energy_oxdna.dat 2>/dev/null || true; }
fi
if [ -n "$OXKK" ]; then
  run oxkk "$OXKK" input && res+=("bench/oxdna_kokkos|$(grep -o '[0-9.]* timesteps/s' log_oxkk.txt | awk '{print $1}')|$(grep -E '^ +[0-9]+ ' log_oxkk.txt | head -1 | awk '{print $3}')")
fi
if [ -n "$OXKKL" ]; then
  run oxkkl_lean "$OXKKL" input && res+=("bench/oxdna_kokkos_lammps (lean)|$(grep -o '[0-9.]* timesteps/s' log_oxkkl_lean.txt | awk '{print $1}')|$(grep -E '^ +[0-9]+ ' log_oxkkl_lean.txt | head -1 | awk '{print $3}')")
  run oxkkl "$OXKKL" input_lammps_mode && res+=("bench/oxdna_kokkos_lammps (lammps_overhead = 1)|$(grep -o '[0-9.]* timesteps/s' log_oxkkl.txt | awk '{print $1}')|$(grep -E '^ +[0-9]+ ' log_oxkkl.txt | head -1 | awk '{print $3}')")
fi
if [ -n "$LMP" ]; then
  run lammps "$LMP" $LMP_ARGS -in in.lammps -var data data.lmp -var steps $steps -log none && res+=("LAMMPS KOKKOS|$(grep -o '[0-9.]* timesteps/s' log_lammps.txt | awk '{print $1}')|$(grep -A1 '^ *Step' log_lammps.txt | tail -1 | awk '{print $2}')")
fi
echo
echo "case $(basename $c): $N nucleotides, $steps steps"
printf '%-48s %14s %14s\n' code timesteps/s "U/nt (step 0)"
for r in "${res[@]}"; do IFS='|' read a b u <<< "$r"; printf '%-48s %14s %14s\n' "$a" "$b" "$u"; done
