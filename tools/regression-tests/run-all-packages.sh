#!/bin/bash
# Build LAMMPS with every package that can be built on this host, then compare
# the KOKKOS Serial backend against the plain CPU styles over the whole examples
# tree.
#
# This is a stripped down relative of the build.sh script in the lammps-analyze
# repository, which drives the nightly runs behind
# https://lammps.github.io/lammps-test-results/.  Dropped from it: the bubblewrap
# sandbox, the rsync publishing, Coverity, the coverage build, and the clang
# static analysis.  Kept: the regression driver and the cost based sharding.
#
# Two things make this different from the nightly runs, and both are the point
# of the exercise:
#
#   1. It builds all_on.cmake rather than most.cmake, so the examples of the
#      packages the nightly build leaves out are exercised at all.
#   2. The CPU pass runs with --gen-ref in a copy of the examples tree that has
#      had the bundled reference logs removed, so the KOKKOS pass compares
#      against logs produced by this build on this machine.  The bundled logs
#      come from other machines and compilers, and the floating point
#      differences against them are large enough to hide the ones this test is
#      looking for.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
LAMMPS_DIR=$PWD
WORK=${WORK:-${LAMMPS_DIR}/regression-work}
WORKREL=$(basename "${WORK}")     # run_tests.py resolves its output paths against $PWD
EXAMPLES_REF=${LAMMPS_DIR}/examples-ref
PYTHON=${PYTHON:-python3}
NPROCS=${NPROCS:-4}          # MPI ranks per test
NWORKERS=${NWORKERS:-1}      # tests running at the same time
RUNNER=${LAMMPS_DIR}/tools/regression-tests/run_tests.py

mkdir -p "${WORK}"

# Open MPI refuses to run as root and will not oversubscribe by default; both
# are set here rather than in the committed configuration files so that the
# configurations stay usable on a normal account.
export OMPI_ALLOW_RUN_AS_ROOT=${OMPI_ALLOW_RUN_AS_ROOT:-1}
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=${OMPI_ALLOW_RUN_AS_ROOT_CONFIRM:-1}
export OMPI_MCA_rmaps_base_oversubscribe=${OMPI_MCA_rmaps_base_oversubscribe:-1}
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export LAMMPS_POTENTIALS=${LAMMPS_DIR}/potentials

# run one full set of regression tests
#   $1 LAMMPS binary, $2 output prefix, $3 config file, $4 examples tree,
#   $5 extra run_tests.py arguments
run_regression_tests() {
    local lmpbin=$1 prefix=$2 config=$3 tree=$4 extra=${5:-}
    # Resume where a previous attempt stopped.  A full sweep takes hours, and in
    # a container that can be reclaimed mid-run the progress file is the only
    # thing that makes a restart cheap: run_tests.py skips every input already
    # recorded in it.  Set RESUME=0 to start over.
    if [ "${RESUME:-1}" = "1" ] && [ -s "${WORK}/${prefix}-progress.yaml" ]; then
        extra="${extra} --resume"
        echo "resuming ${prefix}: $(grep -c ': {' "${WORK}/${prefix}-progress.yaml") inputs already recorded"
    fi
    ${PYTHON} "${RUNNER}" \
        --lmp-bin="${lmpbin}" \
        --config-file="${LAMMPS_DIR}/tools/regression-tests/${config}" \
        --examples-top-level="${tree}" \
        --num-workers="${NWORKERS}" \
        --output-file="${WORKREL}/${prefix}.xml" \
        --progress-file="${WORKREL}/${prefix}-progress.yaml" \
        --failure-file="${WORKREL}/${prefix}-failure.yaml" \
        --log-file="${WORKREL}/${prefix}-run.log" \
        ${extra} 2>&1 | tee "${WORK}/${prefix}.out"
    # a timed out test can leave its MPI ranks behind, and they would compete
    # with every test that follows
    pkill -e lmp || true
}

case "${1:-all}" in
  build-cpu|all)
    cmake -S cmake -B build-regression -G Ninja \
        -C cmake/presets/gcc.cmake -C cmake/presets/all_on.cmake \
        -C cmake/presets/download.cmake \
        -D PKG_KOKKOS=off -D PKG_GPU=on -D GPU_API=opencl \
        -D BUILD_MPI=on -D BUILD_OMP=on -D BUILD_SHARED_LIBS=on -D BUILD_TOOLS=off \
        -D FFT=FFTW3 -D WITH_JPEG=on -D WITH_PNG=on -D DOWNLOAD_POTENTIALS=on \
        -D MLIAP_ENABLE_ACE=on -D MLIAP_ENABLE_PYTHON=on \
        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache -D CMAKE_C_COMPILER_LAUNCHER=ccache \
        -D CMAKE_CXX_STANDARD=20 -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold \
        -D CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=mold \
        ${CMAKE_EXTRA_ARGS:-}
    cmake --build build-regression -j "${NPROCS}"
    ;;&
  build-kokkos|all)
    # The same package set plus KOKKOS on the Serial backend.  The compiler flags
    # of the two builds must match exactly, for two reasons: a difference in -std
    # or -O misses on every ccache entry and doubles the build, and comparing a
    # binary built at -O3 against one built at -O2 puts a second variable into a
    # comparison whose whole purpose is to isolate one.  So C++20 (which KOKKOS
    # requires) and Release are used for both.  Debug information is deliberately
    # not added here for the same reason; rebuild an individual case with -g when
    # a crash actually needs a backtrace.
    cmake -S cmake -B build-kokkos -G Ninja \
        -C cmake/presets/gcc.cmake -C cmake/presets/all_on.cmake \
        -C cmake/presets/download.cmake -C cmake/presets/kokkos-serial.cmake \
        -D PKG_KOKKOS=on -D FFT_KOKKOS=KISS -D KOKKOS_PREC=double \
        -D PKG_GPU=on -D GPU_API=opencl \
        -D BUILD_MPI=on -D BUILD_OMP=on -D BUILD_SHARED_LIBS=on -D BUILD_TOOLS=off \
        -D FFT=FFTW3 -D WITH_JPEG=on -D WITH_PNG=on -D DOWNLOAD_POTENTIALS=on \
        -D MLIAP_ENABLE_ACE=on -D MLIAP_ENABLE_PYTHON=on \
        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache -D CMAKE_C_COMPILER_LAUNCHER=ccache \
        -D CMAKE_CXX_STANDARD=20 -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold \
        -D CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=mold \
        ${CMAKE_EXTRA_ARGS:-}
    cmake --build build-kokkos -j "${NPROCS}"
    ;;&
  cpu|all)
    # A copy of the examples tree without the bundled reference logs, so that
    # --gen-ref leaves exactly one reference log per input and find_reference_logs()
    # in run_tests.py cannot pick up a log from another machine by mistake.
    #
    # The copy has to sit next to examples/ rather than inside a subdirectory:
    # 154 files in the tree are relative symlinks that reach outside it, such as
    # examples/eim/ffield.eim -> ../../potentials/ffield.eim.  Those only resolve
    # when the copy is at the same depth, and a dangling potential file turns into
    # a test failure that has nothing to do with the code under test.
    if [ "${RESUME:-1}" = "1" ] && [ -d "${EXAMPLES_REF}" ]; then
        echo "keeping the existing ${EXAMPLES_REF} for a resumed run"
    else
        rm -rf "${EXAMPLES_REF}"
        cp -a examples "${EXAMPLES_REF}"
        find "${EXAMPLES_REF}" -name 'log.*' -delete
    fi
    dangling=$(find "${EXAMPLES_REF}" -type l ! -exec test -e {} \; -print | wc -l)
    [ "${dangling}" -eq 0 ] || { echo "${dangling} dangling symlinks in ${EXAMPLES_REF}"; exit 1; }
    run_regression_tests build-regression/lmp cpu config_mpi4_t120.yaml \
        examples-ref "--gen-ref"
    ;;&
  kokkos|all)
    run_regression_tests build-kokkos/lmp kokkos config_kokkos_serial_mpi4.yaml \
        examples-ref
    ;;&
  classify|all)
    ${PYTHON} "${LAMMPS_DIR}/tools/regression-tests/classify_kokkos.py" \
        --cpu "${WORK}/cpu-progress.yaml" \
        --kokkos "${WORK}/kokkos-progress.yaml" \
        --examples "${EXAMPLES_REF}" \
        --output "${WORK}/kokkos-classification.md"
    ;;
esac
