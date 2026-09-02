#!/bin/bash
# For each fix: put the pre-fix file back, rebuild lmp, run the input under the
# detectors, restore the file.  Reports land in $OUT/<name>.err
S=/tmp/claude-0/-home-user-lammps/90034feb-5380-5f92-8980-0aec3d6106a8/scratchpad
W=/home/user/lammps-sync; B=$W/build-sync; OUT=$S/ex_reverted; mkdir -p $OUT
export LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1 OMP_NUM_THREADS=1
PKD="-pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on"
PKH="-pk kokkos neigh half newton off comm device sort device atom/map device gpu/aware on"
one() { name=$1; file=$2; dir=$3; inp=$4; pk=$5
  cd $W && git show origin/kk_bugfixes:src/KOKKOS/$file > src/KOKKOS/$file
  ninja -C $B -j3 lmp > $OUT/build_$name.log 2>&1 || { echo "$name BUILD FAILED"; git checkout -q src/KOKKOS/$file; return; }
  (cd $dir && timeout 900 $B/lmp -k on t 1 -sf kk $pk -in $inp -log none -screen $OUT/$name.out 2> $OUT/$name.err)
  echo "$name rc=$? $(grep -c '^\[stale\]\|^\[watch\]' $OUT/$name.err) reports; $(grep -c 'Total wall time' $OUT/$name.out) finished"
  cd $W && git checkout -q src/KOKKOS/$file
}
one tempsphere compute_temp_sphere_kokkos.cpp $S in.tempsphere "$PKD"
one quickmin min_quickmin_kokkos.cpp $S in.quickmin "$PKD"
one peptide fix_shake_kokkos.cpp $W/examples/peptide in.peptide "$PKH"
one tatb compute_reaxff_atom_kokkos.cpp $W/examples/reaxff in.reaxff.tatb "-pk kokkos neigh half newton on comm device sort device atom/map device gpu/aware on"
ninja -C $B -j3 lmp > $OUT/build_restore.log 2>&1; echo "restored rc=$?"
