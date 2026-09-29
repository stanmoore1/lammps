cmake_minimum_required(VERSION 3.20)
project(oxdna_kokkos CXX)

# -----------------------------------------------------------------------
# Kokkos: prefer an installed version; fall back to bundled LAMMPS source
# -----------------------------------------------------------------------
find_package(Kokkos QUIET)
if(NOT Kokkos_FOUND)
    # Default to the Kokkos bundled with this LAMMPS tree (bench/oxdna_kokkos -> lib/kokkos)
    set(KOKKOS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/kokkos"
        CACHE PATH "Path to Kokkos source tree")
    if(NOT EXISTS "${KOKKOS_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "Kokkos not found. Install Kokkos or set KOKKOS_SOURCE_DIR "
            "to a Kokkos source tree.")
    endif()
    message(STATUS "Using bundled Kokkos at ${KOKKOS_SOURCE_DIR}")
    add_subdirectory(${KOKKOS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/kokkos)
endif()

# -----------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------
add_executable(oxdna_kokkos
    src/main.cpp
)

target_compile_features(oxdna_kokkos PRIVATE cxx_std_20)
target_link_libraries(oxdna_kokkos PRIVATE Kokkos::kokkos)
target_include_directories(oxdna_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Default oxDNA3 sequence-dependence file (used when the input file has no
# `seq_dep_file` key): the copy of upstream oxDNA's
# oxDNA3_sequence_dependent_parameters.txt shipped in params/.
set(OXDNA_DEFAULT_SEQ_DEP_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/params/oxDNA3_sequence_dependent_parameters.txt"
    CACHE FILEPATH "Default oxDNA3 sequence-dependent parameter file")
target_compile_definitions(oxdna_kokkos PRIVATE
    OXDNA_DEFAULT_SEQ_DEP_FILE="${OXDNA_DEFAULT_SEQ_DEP_FILE}")

# -----------------------------------------------------------------------
# Precision (default: double) -- mirrors the oxDNA CUDA backends:
#   default                     backend_precision = double (CUDA_DOUBLE=ON)
#   -DOXDNA_SINGLE_PRECISION=ON backend_precision = float
#   -DOXDNA_MIXED_PRECISION=ON  backend_precision = mixed (CUDAMixedBackend):
#                               float force kernels, double integrator
# -----------------------------------------------------------------------
option(OXDNA_SINGLE_PRECISION "Use single precision (float), oxDNA backend_precision = float" OFF)
option(OXDNA_MIXED_PRECISION  "Use mixed precision, oxDNA backend_precision = mixed" OFF)
if(OXDNA_SINGLE_PRECISION AND OXDNA_MIXED_PRECISION)
    message(FATAL_ERROR "OXDNA_SINGLE_PRECISION and OXDNA_MIXED_PRECISION are mutually exclusive")
endif()
set(OXDNA_PRECISION_DEFS "")
if(OXDNA_SINGLE_PRECISION)
    list(APPEND OXDNA_PRECISION_DEFS OXDNA_SINGLE_PRECISION)
    message(STATUS "oxdna_kokkos: single precision (backend_precision = float)")
elseif(OXDNA_MIXED_PRECISION)
    list(APPEND OXDNA_PRECISION_DEFS OXDNA_MIXED_PRECISION)
    message(STATUS "oxdna_kokkos: mixed precision (backend_precision = mixed)")
else()
    message(STATUS "oxdna_kokkos: double precision (backend_precision = double, default)")
endif()

# threads_per_block of every kernel (upstream input key, default 2 * warpSize).
# 0 = backend default (64 on CUDA, 128 on HIP).
set(OXDNA_THREADS_PER_BLOCK 0 CACHE STRING "Threads per block of every kernel (0 = 2 * warp size)")
if(NOT OXDNA_THREADS_PER_BLOCK EQUAL 0)
    list(APPEND OXDNA_PRECISION_DEFS OXDNA_THREADS_PER_BLOCK=${OXDNA_THREADS_PER_BLOCK})
endif()
target_compile_definitions(oxdna_kokkos PRIVATE ${OXDNA_PRECISION_DEFS})

# Fast math, as upstream oxDNA compiles every non-debug build: host code with
# -ffast-math (CMakeLists.txt: ADD_DEFINITIONS(-ffast-math)) and CUDA device
# code with -use_fast_math (src/CMakeLists.txt: CUDA_COMMON_FLAGS), i.e.
# flush-to-zero, approximate float division/sqrt and fast float intrinsics.
option(OXDNA_FAST_MATH "Compile with fast math like upstream oxDNA (-ffast-math, CUDA -use_fast_math)" ON)
set(OXDNA_FAST_MATH_FLAGS "")
if(OXDNA_FAST_MATH AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND OXDNA_FAST_MATH_FLAGS -ffast-math)
    if(Kokkos_ENABLE_CUDA)
        list(APPEND OXDNA_FAST_MATH_FLAGS -use_fast_math)
    endif()
    message(STATUS "oxdna_kokkos: fast math (${OXDNA_FAST_MATH_FLAGS})")
endif()
target_compile_options(oxdna_kokkos PRIVATE ${OXDNA_FAST_MATH_FLAGS})
target_link_options(oxdna_kokkos PRIVATE ${OXDNA_FAST_MATH_FLAGS})

# -----------------------------------------------------------------------
# Build type
# -----------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# -----------------------------------------------------------------------
# Usage notes
# -----------------------------------------------------------------------
message(STATUS "
  Build with CUDA backend:
    cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \\
          -DCMAKE_CXX_COMPILER=nvcc_wrapper ..

  Build with OpenMP backend:
    cmake -DKokkos_ENABLE_OPENMP=ON ..

  Build with Serial backend (debug):
    cmake -DKokkos_ENABLE_SERIAL=ON ..
")

# -----------------------------------------------------------------------
# Optional finite-difference force/torque validation tool
#   cmake -DOXDNA_BUILD_TESTS=ON ..   (then run ./fd_test from the source dir)
# -----------------------------------------------------------------------
option(OXDNA_BUILD_TESTS "Build the finite-difference validation tool" OFF)
if(OXDNA_BUILD_TESTS)
    add_executable(fd_test tests/fd_test.cpp)
    target_compile_features(fd_test PRIVATE cxx_std_20)
    target_link_libraries(fd_test PRIVATE Kokkos::kokkos)
    target_include_directories(fd_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(fd_test PRIVATE
        OXDNA_DEFAULT_SEQ_DEP_FILE="${OXDNA_DEFAULT_SEQ_DEP_FILE}")
    target_compile_definitions(fd_test PRIVATE ${OXDNA_PRECISION_DEFS})
    target_compile_options(fd_test PRIVATE ${OXDNA_FAST_MATH_FLAGS})
    target_link_options(fd_test PRIVATE ${OXDNA_FAST_MATH_FLAGS})
endif()

if(OXDNA_BUILD_TESTS)
    add_executable(xcheck tests/xcheck.cpp)
    target_compile_features(xcheck PRIVATE cxx_std_20)
    target_link_libraries(xcheck PRIVATE Kokkos::kokkos)
    target_include_directories(xcheck PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(xcheck PRIVATE
        OXDNA_DEFAULT_SEQ_DEP_FILE="${OXDNA_DEFAULT_SEQ_DEP_FILE}")
    target_compile_definitions(xcheck PRIVATE ${OXDNA_PRECISION_DEFS})
    target_compile_options(xcheck PRIVATE ${OXDNA_FAST_MATH_FLAGS})
    target_link_options(xcheck PRIVATE ${OXDNA_FAST_MATH_FLAGS})
endif()
