#!/bin/bash
# Every remaining stage, in order, idempotent.  Each stage writes a marker when
# it finishes, so a container restart resumes instead of repeating.  The
# watchdog respawns this script if it dies.
#
# The input list was refreshed to the current tree (869 inputs): upstream moved
# examples/{gjf,relres,tracker} under examples/PACKAGES/ and renamed the
# electrode/dielectric inputs, and added fenix, qmmm-xtb and frenkel.  31 old
# paths went away, 55 new ones appeared, so the double-precision sweeps that
# already finished at 845 get a catch-up pass over just the new inputs.
SP=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad
L=/home/user/lammps
R=$SP/sync/run-sweep.sh
ALL=$SP/sync/all-inputs.txt
M=$SP/sync/markers; mkdir -p $M
echo $$ > $SP/sync/pipeline.pid
KK="-k on -sf kk -pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on"
say() { echo "$(date -u +%m-%d\ %H:%M) $*" >> $SP/sync/orchestrate.status; }

build() {  # $1 = build dir -- ninja no-ops when the binary is already current
  say "building $1"
  . /home/user/env.sh 2>/dev/null
  nice -n 12 ninja -C $L/$1 -j3 lmp > $SP/$1.build.log 2>&1
  rc=$?; say "built $1 rc=$rc"
  # a sweep against a stale or missing binary measures nothing: stop instead
  if [ $rc -ne 0 ]; then
    say "STOPPING: $1 failed to build, refusing to sweep with a stale binary"
    exit 1
  fi
  return 0
}

sweep() {  # $1 tag  $2 binary  $3 label ; detector env from caller
  [ -f $M/done.$1 ] && return 0
  say "START $3"
  $R $L/$2/lmp $1 $ALL 360 4
  say "DONE $3"
  touch $M/done.$1
}

# 0. catch up the finished double-precision sweeps on the 55 new inputs
build build-sync
LMP_KOKKOS_AUDIT=1 KKARGS="$KK" \
  sweep audit845 build-sync "audit catch-up (new inputs)"
LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1 KKARGS="$KK" \
  sweep stale845 build-sync "stale watch catch-up (new inputs)"

# 1. mixed precision: refresh the sync binary, then poison and stale
build build-sync-mixed

build build-poison-mixed
LMP_KOKKOS_POISON=1 KKARGS="$KK" sweep mixed-poison845 build-poison-mixed "mixed poison (869)"

LMP_KOKKOS_WATCH= LMP_KOKKOS_STALE= LMP_KOKKOS_STALE_STRICT=1 KKARGS="$KK" \
  sweep mixed-stale845 build-sync-mixed "mixed stale watch (869)"

LMP_KOKKOS_AUDIT=1 KKARGS="$KK" \
  sweep mixed-audit845 build-sync-mixed "mixed audit catch-up (new inputs)"

# 2. plain AddressSanitizer, kokkos double and plain cpu
build build-asan-kk
KKARGS="$KK" sweep asan-kk845 build-asan-kk "asan kokkos double (869)"

build build-asan-cpu
KKARGS="" sweep asan-cpu845 build-asan-cpu "asan plain cpu (869)"

# 3. double-precision poison catch-up.  build-poison was deleted to free disk,
#    so it is reconfigured here rather than held open through the sweeps above.
if [ ! -f $M/done.poison845 ]; then
  if [ ! -x $L/build-poison/lmp ]; then
    say "configuring build-poison"
    . /home/user/env.sh 2>/dev/null
    cmake -S $L/cmake -B $L/build-poison -G Ninja \
      -C $L/cmake/presets/gcc.cmake -C $L/cmake/presets/most.cmake \
      -D PKG_KOKKOS=on -D Kokkos_ENABLE_SERIAL=on -D FFT_KOKKOS=KISS \
      -D KOKKOS_DEBUG_SYNC=on -D KOKKOS_DEBUG_SYNC_ASAN=on \
      -D BUILD_MPI=on -D DOWNLOAD_POTENTIALS=off \
      -D CMAKE_BUILD_TYPE=RelWithDebInfo \
      -D CMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g1" \
      -D CMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
      > $SP/build-poison.cfg.log 2>&1 || say "build-poison configure FAILED"
  fi
  build build-poison
  LMP_KOKKOS_POISON=1 KKARGS="$KK" sweep poison845 build-poison "poison catch-up (new inputs)"
fi

# 3b. re-run, with the poison build rebuilt against the current source, just the
#     inputs that reported under ASan during the mixed poison sweep.  The sweep
#     binary predates the fixes those reports produced, so this is what says
#     whether each one is actually gone.
if [ ! -f $M/done.reverify ]; then
  build build-poison-mixed
  say "START poison re-verify (fixed inputs)"
  LMP_KOKKOS_POISON=1 KKARGS="$KK" \
    $R $L/build-poison-mixed/lmp reverify $SP/sync/reverify.txt 360 4
  say "DONE poison re-verify: $(ls $SP/sync/reverify/*.asan 2>/dev/null | wc -l) still reporting"
  touch $M/done.reverify
fi

# 4. split memory vs regular memory, same source: a value diff catches what the
#    detectors cannot see, because it looks at the consequence not the accessor
build build-plain
KKARGS="$KK" sweep plain845 build-plain "divergence: regular memory (869)"
KKARGS="$KK" sweep split845 build-sync  "divergence: split memory (869)"
if [ ! -f $M/done.divergence ]; then
  say "START divergence diff"
  $SP/sync/diff-divergence.sh > $SP/sync/divergence-report.txt 2>&1
  say "DONE divergence diff: $(grep -c '^DIVERGES' $SP/sync/divergence-report.txt) inputs differ"
  touch $M/done.divergence
fi

say "ALL WORK DONE"
touch $SP/sync/ALL-WORK-DONE
