#!/bin/bash
# pinject.sh <file> <line> <dir> <input> <np> [pk]
SP=/tmp/claude-0/-home-user-lammps/7fd89ee0-7b78-5513-a9a1-2f910de0c3b9/scratchpad
cd /home/user/lammps
if [ -n "$(git status --porcelain src/KOKKOS cmake doc)" ]; then
  echo "REFUSING: uncommitted changes under src/KOKKOS -- commit them first"; exit 2
fi
python3 $SP/inject.py $1 $2 || exit 1
(cd $SP/build-poison && cmake --build . -j 4 >/dev/null 2>&1) || { python3 $SP/inject.py --restore $1; echo BUILD-FAIL; exit 1; }
rm -rf /tmp/asanlog.inj; mkdir -p /tmp/asanlog.inj
cd examples/$3
pk=$(echo "$6" | tr ':' ' ')
LMP_KOKKOS_POISON=1 ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:log_path=/tmp/asanlog.inj/a \
  timeout 1800 $( [ $5 -gt 1 ] && echo mpirun --allow-run-as-root --oversubscribe -np $5 ) \
  $SP/build-poison/lmp -in $4 -log none -screen /tmp/pinj.scr -k on -sf kk ${pk:+-pk kokkos $pk} </dev/null >/dev/null 2>&1
echo "run=$?"
echo "traps=$(cat /tmp/asanlog.inj/a.* 2>/dev/null | grep -c 'ERROR: AddressSanitizer')"
cat /tmp/asanlog.inj/a.* 2>/dev/null | grep -m1 "SUMMARY: AddressSanitizer" | cut -c1-130
python3 $SP/inject.py --restore $1
