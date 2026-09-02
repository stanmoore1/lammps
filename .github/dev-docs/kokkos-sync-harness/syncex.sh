#!/bin/bash
# example inputs with the GPU package settings under the split-memory build
B=$1; OUT=$2; mkdir -p $OUT
export LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1 OMP_NUM_THREADS=1
S=/tmp/claude-0/-home-user-lammps/90034feb-5380-5f92-8980-0aec3d6106a8/scratchpad
PK="-pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on"
ex() { name=$1; dir=$2; inp=$3; shift 3
  (cd $dir && timeout 900 $B/lmp -k on t 1 -sf kk $PK -in $inp -log none -screen $OUT/e_$name.out "$@" 2> $OUT/e_$name.err)
  echo "$name rc=$? $(grep -c '^\[stale\]\|^\[watch\]' $OUT/e_$name.err) reports; $(grep -c 'Total wall time' $OUT/e_$name.out) finished"
}
ex tempsphere $S in.tempsphere
ex quickmin $S in.quickmin
ex respa $S in.respa
ex peptide /home/user/lammps-sync/examples/peptide in.peptide
ex cmap /home/user/lammps-sync/examples/cmap in.cmap_short
ex tatb /home/user/lammps-sync/examples/reaxff in.reaxff.tatb
ex rigid /home/user/lammps-sync/examples/rigid in.rigid.small
ex min /home/user/lammps-sync/examples/min in.min
