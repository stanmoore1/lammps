#!/bin/bash
S=${LMP_TEST_SCRATCH:-/tmp/claude-0/-home-user-lammps/b7c14453-c726-5fc1-a3f6-52a175964ec4/scratchpad}
K12='^(AtomStylesKokkos|DumpAtomKokkos|DumpCustomKokkos|DumpLocalKokkos|DumpGridKokkos|FixSpringChunkKokkos|ThermoCommandsKokkos|SetPropertyKokkos|PairLDDKokkos|ComputeGlobalKokkos|ComputeChunkKokkos|PhononCommandsKokkos)$'
P12='^(AtomStyles|DumpAtom|DumpCustom|DumpLocal|DumpGrid|FixSpringChunk|ThermoCommands|SetProperty|PairLDD|ComputeGlobal|ComputeChunk|PhononCommands)$'
echo "PIPE2 START $(date -u +%H:%M:%S)"

$S/runchunks.sh /home/user/lammps/build-mixed mixed
echo "STAGE mixed-ci-chunks COMPLETE fails=$(cat $S/results/mixed_*.done 2>/dev/null | grep -o 'fails=[0-9]*' | cut -d= -f2 | paste -sd+ | bc)"

if [ ! -f $S/st-single-build.txt ]; then
  cd /home/user/lammps && cmake --build build-single -j 4 > $S/build-single.log 2>&1
  echo "SINGLE_BUILD=$?" > $S/st-single-build.txt
fi
echo "STAGE single-build COMPLETE $(cat $S/st-single-build.txt)"

if [ -x /home/user/lammps/build-single/lmp ]; then
  cd /home/user/lammps/build-single
  [ -f $S/st-single-k12.txt ] || { ctest --output-on-failure -j 4 -R "$K12" > $S/p-single-k12.log 2>&1; echo "SINGLE_K12=$?" > $S/st-single-k12.txt; }
  echo "STAGE single-k12 COMPLETE $(cat $S/st-single-k12.txt)"
  [ -f $S/st-single-p12.txt ] || { ctest --output-on-failure -j 4 -R "$P12" > $S/p-single-p12.log 2>&1; echo "SINGLE_P12=$?" > $S/st-single-p12.txt; }
  echo "STAGE single-p12 COMPLETE $(cat $S/st-single-p12.txt)"
  S_BUILD=/home/user/lammps/build-single python3 $S/mkchunks.py /home/user/lammps/build-single single > /dev/null 2>&1
  $S/runchunks.sh /home/user/lammps/build-single single
  echo "STAGE single-ci-chunks COMPLETE"
fi

cd /home/user/lammps/build-mixed
[ -f $S/st-mc-k12.txt ] || { ctest -T memcheck --output-on-failure -j 2 -R "$K12" > $S/mc-k12.log 2>&1; echo "MC_K12=$?" > $S/st-mc-k12.txt; }
echo "STAGE memcheck-k12 COMPLETE $(cat $S/st-mc-k12.txt)"
[ -f $S/st-mc-p12.txt ] || { ctest -T memcheck --output-on-failure -j 2 -R "$P12" > $S/mc-p12.log 2>&1; echo "MC_P12=$?" > $S/st-mc-p12.txt; }
echo "STAGE memcheck-p12 COMPLETE $(cat $S/st-mc-p12.txt)"
echo "PIPE2 ALL_DONE $(date -u +%H:%M:%S)"
