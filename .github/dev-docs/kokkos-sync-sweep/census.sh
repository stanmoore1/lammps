#!/bin/bash
# One clean run per screen case with the copy census on.  The union over cases
# is what a site has to be absent from to be provably inert.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
: > $SP/census.raw
while read d i np pk; do
  [ "$pk" = "-" ] && pk="" || pk=$(echo "$pk" | tr ':' ' ')
  tag="$i.$np${pk:+.dev}"
  LMP_KOKKOS_COPYSTATS=1 bash $SP/runcase.sh $d $i $np /tmp/census.$tag "$pk"
  n=$(grep -c '^\[copies\] ' /tmp/census.$tag.err 2>/dev/null)
  grep '^\[copies\] ' /tmp/census.$tag.err >> $SP/census.raw 2>/dev/null
  echo "  $tag arrays=$n"
done < $SP/cases_core.txt
echo CENSUS-DONE
