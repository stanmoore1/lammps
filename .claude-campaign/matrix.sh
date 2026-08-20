#!/bin/bash
# Run one input under every combination of the KOKKOS comm and sort settings and
# report the largest relative deviation in the thermo output from the default.
# These are performance settings, so they must not change the physics.  Sorting
# and a different ghost ordering do change the order of the summations, so a
# deviation around 1e-16..1e-12 is roundoff; anything larger is a real bug.
#   matrix.sh <lmp> <dir> <input> [nprocs]
LMP=$1; DIR=$2; IN=$3; NP=${4:-1}
S=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
cd "$DIR" || exit 1
run() { # $1 = outfile, $2 = pk options
  if [ "$NP" -gt 1 ]; then
    mpirun --allow-run-as-root --oversubscribe -np "$NP" "$LMP" -in "$IN" -log none -screen "$1" -k on -sf kk ${2:+-pk kokkos $2} >/dev/null 2>&1
  else
    "$LMP" -in "$IN" -log none -screen "$1" -k on -sf kk ${2:+-pk kokkos $2} >/dev/null 2>&1
  fi
}
run /tmp/mx.ref "" || { echo "    default: EXIT $?"; exit 1; }
worst=0
for c in no host device; do
  for s in host device; do
    run /tmp/mx.cur "comm $c sort $s"
    rc=$?
    if [ $rc -ne 0 ]; then echo "    comm=$c,sort=$s: EXIT $rc"; continue; fi
    d=$(python3 $S/cmp.py /tmp/mx.ref /tmp/mx.cur 2>/dev/null)
    case "$d" in SHAPE*) echo "    comm=$c,sort=$s: $d";; *)
      big=$(python3 -c "print(1 if float('$d')>1e-10 else 0)" 2>/dev/null)
      [ "$big" = "1" ] && echo "    comm=$c,sort=$s: max rel dev $d   <-- beyond roundoff"
      worst=$(python3 -c "print(max(float('$worst'),float('$d')))" 2>/dev/null);;
    esac
  done
done
echo "    worst deviation across all combinations: $worst"
