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
    ${PYTHON} "${RUNNER}" \
        --lmp-bin="${lmpbin}" \
        --config-file="${LAMMPS_DIR}/tools/regression-tests/${config}" \
        --examples-top-level="${tree}" \
        --num-workers="${NWORKERS}" \
        --output-file="${WORK}/${prefix}.xml" \
        --progress-file="${WORK}/${prefix}-progress.yaml" \
        --failure-file="${WORK}/${prefix}-failure.yaml" \
        --log-file="${WORK}/${prefix}-run.log" \
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
    # the same package set plus KOKKOS on the Serial backend.  C++20 is forced
    # for both builds (KOKKOS requires it) so that the two share ccache entries
    # instead of missing on every translation unit over a different -std flag.
    # RelWithDebInfo rather than Release: the crashes this run is looking for
    # are triaged from backtraces.
    cmake -S cmake -B build-kokkos -G Ninja \
        -C cmake/presets/gcc.cmake -C cmake/presets/all_on.cmake \
        -C cmake/presets/download.cmake -C cmake/presets/kokkos-serial.cmake \
        -D PKG_KOKKOS=on -D FFT_KOKKOS=KISS -D KOKKOS_PREC=double \
        -D PKG_GPU=on -D GPU_API=opencl \
        -D BUILD_MPI=on -D BUILD_OMP=on -D BUILD_SHARED_LIBS=on -D BUILD_TOOLS=off \
        -D FFT=FFTW3 -D WITH_JPEG=on -D WITH_PNG=on -D DOWNLOAD_POTENTIALS=on \
        -D MLIAP_ENABLE_ACE=on -D MLIAP_ENABLE_PYTHON=on \
        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache -D CMAKE_C_COMPILER_LAUNCHER=ccache \
        -D CMAKE_CXX_STANDARD=20 -D CMAKE_BUILD_TYPE=RelWithDebInfo \
        -D CMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold \
        -D CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=mold \
        ${CMAKE_EXTRA_ARGS:-}
    cmake --build build-kokkos -j "${NPROCS}"
    ;;&
  cpu|all)
    # a copy of the examples tree without the bundled reference logs, so that
    # --gen-ref leaves exactly one reference log per input and find_reference_logs()
    # in run_tests.py cannot pick up a log from another machine by mistake
    rm -rf "${WORK}/examples-ref"
    cp -a examples "${WORK}/examples-ref"
    find "${WORK}/examples-ref" -name 'log.*' -delete
    run_regression_tests build-regression/lmp cpu config_mpi4_t120.yaml \
        "${WORK}/examples-ref" "--gen-ref"
    ;;&
  kokkos|all)
    run_regression_tests build-kokkos/lmp kokkos config_kokkos_serial_mpi4.yaml \
        "${WORK}/examples-ref"
    ;;&
  classify|all)
    ${PYTHON} "${LAMMPS_DIR}/tools/regression-tests/classify_kokkos.py" \
        --cpu "${WORK}/cpu-progress.yaml" \
        --kokkos "${WORK}/kokkos-progress.yaml" \
        --examples "${WORK}/examples-ref" \
        --output "${WORK}/kokkos-classification.md"
    ;;
esac
