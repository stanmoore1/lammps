cmake_minimum_required(VERSION 3.20)
project(oxdna_kokkos_lammps CXX)

# -----------------------------------------------------------------------
# Kokkos: prefer an installed version; fall back to bundled LAMMPS source
# -----------------------------------------------------------------------
find_package(Kokkos QUIET)
if(NOT Kokkos_FOUND)
    # Default to the Kokkos bundled with this LAMMPS tree (bench/oxdna_kokkos_lammps -> lib/kokkos)
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
add_executable(oxdna_kokkos_lammps
    src/main.cpp
)

target_compile_features(oxdna_kokkos_lammps PRIVATE cxx_std_20)
target_link_libraries(oxdna_kokkos_lammps PRIVATE Kokkos::kokkos)
target_include_directories(oxdna_kokkos_lammps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# -----------------------------------------------------------------------
# Precision (default: double)
# -----------------------------------------------------------------------
option(OXDNA_SINGLE_PRECISION "Use single precision (float)" OFF)
if(OXDNA_SINGLE_PRECISION)
    target_compile_definitions(oxdna_kokkos_lammps PRIVATE OXDNA_SINGLE_PRECISION)
    message(STATUS "oxdna_kokkos_lammps: single precision enabled")
else()
    message(STATUS "oxdna_kokkos_lammps: double precision (default)")
endif()

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
    if(OXDNA_SINGLE_PRECISION)
        target_compile_definitions(fd_test PRIVATE OXDNA_SINGLE_PRECISION)
    endif()
endif()

if(OXDNA_BUILD_TESTS)
    add_executable(xcheck tests/xcheck.cpp)
    target_compile_features(xcheck PRIVATE cxx_std_20)
    target_link_libraries(xcheck PRIVATE Kokkos::kokkos)
    target_include_directories(xcheck PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()
