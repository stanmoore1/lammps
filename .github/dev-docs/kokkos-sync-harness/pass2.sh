#!/bin/bash
S=/tmp/claude-0/-home-user-lammps/90034feb-5380-5f92-8980-0aec3d6106a8/scratchpad
W=/home/user/lammps-sync; B=$W/build-sync; OUT=$S/ex_pass2; mkdir -p $OUT
export LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1 OMP_NUM_THREADS=1
PKD="-pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on"
PKR="-pk kokkos neigh half newton on comm device sort device atom/map device gpu/aware on"
run() { name=$1; dir=$2; inp=$3; pk=$4; (cd $dir && timeout 900 $B/lmp -k on t 1 -sf kk $pk -in $inp -log none -screen $OUT/$name.out 2> $OUT/$name.err); echo "$name rc=$? $(grep -c '^\[stale\]\|^\[watch\]' $OUT/$name.err) reports; $(grep -c 'Total wall time' $OUT/$name.out) finished"; }
# fixed baselines
run quickmin_fixed $S in.quickmin "$PKD"
run tatb_fixed $W/examples/reaxff in.reaxff.tatb "$PKR"
# quickmin reverted, with a thermo style that does not sync the masses itself
cd $W && git show origin/kk_bugfixes:src/KOKKOS/min_quickmin_kokkos.cpp > src/KOKKOS/min_quickmin_kokkos.cpp
ninja -C $B -j3 lmp > $OUT/build_qm.log 2>&1 && run quickmin_reverted $S in.quickmin "$PKD"
cd $W && git checkout -q src/KOKKOS/min_quickmin_kokkos.cpp && ninja -C $B -j3 lmp > $OUT/build_restore.log 2>&1; echo "restore rc=$?"
# test binaries, then the unit-test sweep
ninja -C $B -j3 test_fix_timestep test_pair_style test_bond_style test_angle_style test_dihedral_style test_improper_style test_kspace_styles > $S/build_synct.log 2>&1; echo "tests build rc=$?"
$S/syncrun.sh $B $S/ut_fixed > $S/syncrun_fixed.log 2>&1; echo "sweep done"
