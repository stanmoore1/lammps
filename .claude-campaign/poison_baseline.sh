#!/bin/bash
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
cd /home/user/lammps
(cd $SP/build-poison && cmake --build . -j 4 >$SP/pbase.log 2>&1) || { echo "BUILD FAILED"; exit 1; }
while read d i np pk; do
  [ "$pk" = "-" ] && pk="" || pk=$(echo "$pk" | tr ':' ' ')
  tag="$i.$np${pk:+.dev}"
  timeout 3600 bash $SP/poison_run.sh $d $i $np $SP/pbase.$tag "$pk"
  n=$(grep -c 'ERROR: AddressSanitizer' $SP/pbase.$tag.err)
  echo "$tag exit=$? traps=$n"
done < $SP/cases_core.txt
echo POISON-BASELINE-DONE
