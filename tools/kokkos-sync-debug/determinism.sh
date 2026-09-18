#!/bin/bash
# Run each candidate twice with the SAME binary and compare.
#
# Comparison goes through compare-thermo.py, which drops the wall-clock columns.
# Comparing whole thermo rows instead reports every input whose thermo_style
# names CPU or S/CPU as nondeterministic, and three findings in FINDINGS.md were
# exactly that and nothing else.
#
# The divergence stage compares regular-memory against split-memory output, but
# that only means anything for an input that reproduces itself.  Several do not:
# fix balance redistributes on measured time, and several examples seed from
# something that varies.  For those, "the two memory models disagree" is not a
# finding at all.  Run the control first and let it decide which candidates are
# worth reading.
HERE=$(cd "$(dirname "$0")" && pwd)
SP=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad
L=/home/user/lammps
. /home/user/env.sh 2>/dev/null
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false LAMMPS_POTENTIALS=$L/potentials
KK="-k on -sf kk -pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on"
VAR="-var nsteps 200 -var nequil 100 -var nprod 100 -var maxiter 200 -var maxeval 200"
out=$SP/sync/determinism.txt; : > $out
# fd 3, not stdin: mpirun reads stdin and would swallow the rest of the list --
# two inputs got processed out of 42 before this was caught.
while read -r f <&3; do
  d=$(dirname "$f"); i=$(basename "$f")
  cd "$L/$d" 2>/dev/null || continue
  printf 'timer timeout 0:01:30 every 50\ninclude %s\n' "$i" > detcheck.in
  for run in 1 2; do
    nice -n 18 timeout 200 mpirun --host localhost:4 -np 4 $L/build-plain/lmp \
      -in detcheck.in -cite none -log none $KK $VAR 2>/dev/null < /dev/null \
      > /tmp/det.$run
  done
  rm -f detcheck.in
  verdict=$(python3 "$HERE/compare-thermo.py" /tmp/det.1 /tmp/det.2)
  case $verdict in
    NO-OUTPUT) echo "NO-THERMO       $f" >> $out ;;
    SAME)      echo "DETERMINISTIC   $f" >> $out ;;
    *)         echo "NONDETERMINISTIC $f" >> $out ;;
  esac
done 3< "$1"
echo "DONE" >> $out
