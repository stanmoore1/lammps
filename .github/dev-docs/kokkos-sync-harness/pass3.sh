#!/bin/bash
S=/tmp/claude-0/-home-user-lammps/90034feb-5380-5f92-8980-0aec3d6106a8/scratchpad
W=/home/user/lammps-sync; B=$W/build-sync; OUT=$S/ex_pass3; mkdir -p $OUT
export LMP_KOKKOS_AUDIT=1 OMP_NUM_THREADS=1
PKR="-pk kokkos neigh half newton on comm device sort device atom/map device gpu/aware on"
run() { name=$1; (cd $W/examples/reaxff && timeout 900 $B/lmp -k on t 1 -sf kk $PKR -in in.reaxff.tatb -log none -screen $OUT/$name.out 2> $OUT/$name.err); echo "$name rc=$? $(grep -c '^\[audit\]' $OUT/$name.err) audit lines; $(grep -c 'Total wall time' $OUT/$name.out) finished"; }
run tatb_audit_fixed
cd $W && git show origin/kk_bugfixes:src/KOKKOS/compute_reaxff_atom_kokkos.cpp > src/KOKKOS/compute_reaxff_atom_kokkos.cpp
ninja -C $B -j2 lmp > $OUT/build_rev.log 2>&1 && run tatb_audit_reverted
cd $W && git checkout -q src/KOKKOS/compute_reaxff_atom_kokkos.cpp && ninja -C $B -j2 lmp > $OUT/build_restore.log 2>&1; echo "restore rc=$?"
