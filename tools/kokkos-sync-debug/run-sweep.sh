#!/bin/bash
# One detector sweep over a list of example inputs.
#   $1 binary  $2 tag  $3 input list  $4 timeout (s)  $5 ranks
# Detector selection (LMP_KOKKOS_*) and KKARGS come from the caller.
#
# Only the long examples are shortened, and only through -var: an index
# variable set on the command line binds nothing in a script that does not
# declare it, so the ~40 inputs parameterized as slow get a shorter run and
# every other example keeps its own length.  200 steps still reneighbors many
# times, so a bug that needs an exchange or a rebuild is still reachable.
# The timeout only bounds a hang; anything hitting it is recorded as rc=124 so
# the coverage gap is visible rather than silent.
BIN=$1; TAG=$2; LIST=$3; TMO=${4:-900}; NP=${5:-4}
SP=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad
OUT=$SP/sync/$TAG
mkdir -p $OUT
[ -f $OUT/index.txt ] || : > $OUT/index.txt
. /home/user/env.sh
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
export LAMMPS_POTENTIALS=/home/user/lammps/potentials
WRAP=lmpsweep.in
RUNBUDGET=${RUNBUDGET:-0:02:00}
VAR="-var nsteps 200 -var nequil 100 -var nprod 100 -var maxiter 200 -var maxeval 200"
export LD_LIBRARY_PATH="$(dirname $BIN)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# resume: skip whatever this sweep already recorded
cut -f2 $OUT/index.txt | sort -u > $OUT/.done
if [ -s $OUT/.done ]; then grep -vxF -f $OUT/.done "$LIST" > $OUT/.todo; else cp "$LIST" $OUT/.todo; fi
while read -r f <&3; do
  d=$(dirname "$f"); b=$(basename "$f")
  name=$(echo "$f" | sed 's|examples/||; s|/|_|g')
  rm -rf $OUT/a.$name; mkdir -p $OUT/a.$name
  # Bound the run loops with LAMMPS's own timer, not with a kill.  The
  # detectors report at the end of a run, so a SIGKILL at the hard timeout
  # loses the input's report entirely -- which is how four HEAT examples
  # produced nothing at all.  "timer timeout" stops the loop cleanly and
  # LAMMPS still shuts down and prints.  The hard timeout stays as a backstop
  # for a setup that never reaches a run loop.
  ( cd "/home/user/lammps/$d" && \
    printf 'timer timeout %s every 50\ninclude %s\n' "$RUNBUDGET" "$b" > $WRAP && \
    ASAN_OPTIONS="detect_leaks=0:halt_on_error=0:log_path=$OUT/a.$name/a" \
    timeout $TMO mpirun --host localhost:$NP -np $NP \
      $BIN -in $WRAP -cite none -log none $KKARGS $VAR \
      > $OUT/$name.out 2>&1 < /dev/null; rm -f $WRAP )
  rc=$?
  cat $OUT/a.$name/a.* > $OUT/$name.asan 2>/dev/null
  rm -rf $OUT/a.$name
  [ -s $OUT/$name.asan ] || rm -f $OUT/$name.asan
  grep -a '^\[stale\]\|^\[watch\]\|undeclared changes' $OUT/$name.out > $OUT/$name.rep 2>/dev/null
  [ -s $OUT/$name.rep ] || rm -f $OUT/$name.rep
  gzip -f $OUT/$name.out
  echo -e "$rc\t$f" >> $OUT/index.txt
done 3< $OUT/.todo
touch $OUT/DONE
