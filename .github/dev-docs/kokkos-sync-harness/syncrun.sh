#!/bin/bash
# Run the force-style / fix-timestep unit tests and a set of example inputs
# under the KOKKOS split-memory debug build and collect the detector reports.
# usage: syncrun.sh <build-dir> <out-dir>
B=$1; OUT=$2; mkdir -p $OUT
export LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1
export OMP_NUM_THREADS=1
SRC=/home/user/lammps-sync
run_tests() {   # binary  yaml-glob
  bin=$1; shift
  for y in "$@"; do
    n=$(basename $y .yaml)
    timeout 600 $B/$bin $y --gtest_filter='*kokkos_serial*' > $OUT/t_$n.out 2> $OUT/t_$n.err
    echo "$n rc=$? $(grep -c '^\[stale\]\|^\[watch\]' $OUT/t_$n.err) reports $(grep -c 'FAILED' $OUT/t_$n.out) failed"
  done
}
T=$SRC/unittest/force-styles/tests
run_tests test_fix_timestep $T/fix-timestep-*.yaml
run_tests test_pair_style $T/mol-pair-*.yaml $T/atomic-pair-*.yaml $T/manybody-pair-*.yaml
run_tests test_bond_style $T/bond-*.yaml
run_tests test_angle_style $T/angle-*.yaml
run_tests test_dihedral_style $T/dihedral-*.yaml
run_tests test_improper_style $T/improper-*.yaml
run_tests test_kspace_styles $T/kspace-*.yaml
