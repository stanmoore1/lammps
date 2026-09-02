#!/bin/bash
# Clean-run audit baselines, one per case.  The audit speaks on correct runs
# too (nvt/kk and shake/kk both read arrays they do not declare), so a finding
# only counts when it is not already in the clean run's list.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
cd /home/user/lammps
while read d i np pk; do
  [ "$pk" = "-" ] && pk="" || pk=$(echo "$pk" | tr ':' ' ')
  tag="$i.$np${pk:+.dev}"
  LMP_KOKKOS_AUDIT=1 bash $SP/runcase.sh $d $i $np $SP/inj/base.$tag.a "$pk"
  echo "  $tag audit=$(grep -c ' step(s)$' $SP/inj/base.$tag.a 2>/dev/null)"
done < $SP/cases_core.txt
echo AUDITBASE-DONE
