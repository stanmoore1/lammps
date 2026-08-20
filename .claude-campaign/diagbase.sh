#!/bin/bash
# Regenerate the stale-mode baselines with exactly the env the campaign's
# diagnosis uses, so its label diff compares like with like.
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
cd /home/user/lammps
while read d i np pk; do
  [ "$pk" = "-" ] && pk="" || pk=$(echo "$pk" | tr ':' ' ')
  tag="$i.$np${pk:+.dev}"
  bash $SP/runcase.sh $d $i $np $SP/inj/base.$tag "$pk"
  LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1 \
    LMP_KOKKOS_WATCH_SKIP=comm:k_count,comm:k_buf_send \
    bash $SP/runcase.sh $d $i $np $SP/inj/base.$tag.w "$pk"
  echo "  $tag labels=$(sed -n 's/^\[stale\] \+\([A-Za-z_/][A-Za-z_0-9/]*:[A-Za-z_0-9]*\).*/\1/p' $SP/inj/base.$tag.w.err | sort -u | wc -l)"
done < $SP/cases.txt
echo DIAGBASE-DONE
