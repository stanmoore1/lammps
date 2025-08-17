/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Common utilities for mixed precision unit tests (PR 4608)
------------------------------------------------------------------------- */

#ifndef TEST_MIXED_PRECISION_UTILS_H
#define TEST_MIXED_PRECISION_UTILS_H

#include "gtest/gtest.h"
#include "lammps.h"
#include "kokkos_type.h"
#include <cmath>
#include <type_traits>
#include <string>

namespace LAMMPS_NS {
namespace TestUtils {

// Precision mode enumeration
enum PrecisionMode {
    DOUBLE_DOUBLE,
    SINGLE_DOUBLE,
    SINGLE_SINGLE
};

// Get current precision mode at compile time
inline PrecisionMode getCurrentPrecisionMode() {
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    return SINGLE_SINGLE;
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    return SINGLE_DOUBLE;
#else
    return DOUBLE_DOUBLE;
#endif
}

// Get precision mode as string
inline std::string getPrecisionModeString() {
    switch (getCurrentPrecisionMode()) {
        case SINGLE_SINGLE: return "single_single";
        case SINGLE_DOUBLE: return "single_double";
        case DOUBLE_DOUBLE: return "double_double";
        default: return "unknown";
    }
}

// Tolerance values for different precision modes
inline double getRelativeTolerance() {
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    return 1e-5;  // Single precision
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    return 1e-6;  // Mixed precision (slightly better than single)
#else
    return 1e-12; // Double precision
#endif
}

inline double getAbsoluteTolerance() {
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    return 1e-6;  // Single precision
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    return 1e-7;  // Mixed precision
#else
    return 1e-14; // Double precision
#endif
}

// Helper to check if two floating point values are approximately equal
// considering the current precision mode
template<typename T>
inline bool approxEqual(T a, T b) {
    const T rel_tol = static_cast<T>(getRelativeTolerance());
    const T abs_tol = static_cast<T>(getAbsoluteTolerance());
    
    // Handle special cases
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) || std::isinf(b)) {
        return a == b;
    }
    
    // Check absolute difference for small numbers
    T diff = std::abs(a - b);
    if (diff < abs_tol) return true;
    
    // Check relative difference for larger numbers
    T max_val = std::max(std::abs(a), std::abs(b));
    return diff < rel_tol * max_val;
}

// Macro for testing with all precision modes
// This would need to be run three times with different compile flags
#define PRECISION_TEST(test_name) \
    TEST(MixedPrecision_##test_name, PrecisionMode_##PRECISION_SUFFIX)

// Get appropriate suffix for current precision
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    #define PRECISION_SUFFIX SingleSingle
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    #define PRECISION_SUFFIX SingleDouble
#else
    #define PRECISION_SUFFIX DoubleDouble
#endif

// Helper to verify type sizes
template<typename T>
inline void verifyTypeSize(const std::string& type_name) {
    std::cout << "Type " << type_name << " has size " << sizeof(T) 
              << " bytes in " << getPrecisionModeString() << " mode\n";
}

// Helper to create precision-aware LAMMPS instance
inline LAMMPS* createLAMMPSInstance(bool use_kokkos = true, 
                                    bool use_gpu = false,
                                    int nthreads = 1) {
    std::vector<std::string> args = {"test", "-log", "none", "-echo", "none", "-nocite"};
    
    if (use_kokkos) {
        args.push_back("-kokkos");
        args.push_back("on");
        
        if (use_gpu) {
            args.push_back("g");
            args.push_back("1");
        } else {
            args.push_back("d");
            args.push_back("1");
        }
        
        args.push_back("t");
        args.push_back(std::to_string(nthreads));
    }
    
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    
    return new LAMMPS(argv.size(), argv.data(), MPI_COMM_WORLD);
}

// Helper to check if running on GPU
inline bool isGPUEnabled() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET)
    return true;
#else
    return false;
#endif
}

// Helper to get device type string
inline std::string getDeviceTypeString() {
#ifdef KOKKOS_ENABLE_CUDA
    return "CUDA";
#elif defined(KOKKOS_ENABLE_HIP)
    return "HIP";
#elif defined(KOKKOS_ENABLE_SYCL)
    return "SYCL";
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
    return "OpenMPTarget";
#elif defined(KOKKOS_ENABLE_OPENMP)
    return "OpenMP";
#elif defined(KOKKOS_ENABLE_THREADS)
    return "Threads";
#else
    return "Serial";
#endif
}

// Macro to skip tests on certain device types
#define SKIP_IF_NO_GPU() \
    if (!isGPUEnabled()) { \
        GTEST_SKIP() << "Test requires GPU support"; \
    }

#define SKIP_IF_SINGLE_PRECISION() \
    if (getCurrentPrecisionMode() == SINGLE_SINGLE) { \
        GTEST_SKIP() << "Test not applicable for single precision"; \
    }

// Helper class for precision-aware test fixtures
class MixedPrecisionTestFixture : public ::testing::Test {
protected:
    LAMMPS* lmp;
    
    void SetUp() override {
        lmp = createLAMMPSInstance();
        
        // Print precision info once
        static bool printed = false;
        if (!printed) {
            std::cout << "\n=== Mixed Precision Test Configuration ===\n";
            std::cout << "Precision Mode: " << getPrecisionModeString() << "\n";
            std::cout << "Device Type: " << getDeviceTypeString() << "\n";
            std::cout << "KK_FLOAT size: " << sizeof(KK_FLOAT) << " bytes\n";
            std::cout << "KK_SUM_FLOAT size: " << sizeof(KK_SUM_FLOAT) << " bytes\n";
            std::cout << "Relative Tolerance: " << getRelativeTolerance() << "\n";
            std::cout << "Absolute Tolerance: " << getAbsoluteTolerance() << "\n";
            std::cout << "==========================================\n\n";
            printed = true;
        }
    }
    
    void TearDown() override {
        delete lmp;
    }
};

// Helper to compare arrays with precision-aware tolerance
template<typename T>
inline void compareArrays(const T* arr1, const T* arr2, int size, 
                          const std::string& desc = "arrays") {
    for (int i = 0; i < size; i++) {
        if (!approxEqual(arr1[i], arr2[i])) {
            ADD_FAILURE() << "Mismatch in " << desc << " at index " << i 
                         << ": " << arr1[i] << " != " << arr2[i]
                         << " (tolerance: " << getRelativeTolerance() << ")";
        }
    }
}

// Helper to verify numerical stability
template<typename T>
inline bool checkNumericalStability(T value) {
    if (std::isnan(value)) {
        ADD_FAILURE() << "NaN detected in computation";
        return false;
    }
    if (std::isinf(value)) {
        ADD_FAILURE() << "Infinity detected in computation";
        return false;
    }
    return true;
}

// Macro for precision-dependent expectations
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    #define EXPECT_PRECISION_EQ(a, b) EXPECT_FLOAT_EQ(a, b)
    #define EXPECT_PRECISION_NEAR(a, b, tol) EXPECT_NEAR(a, b, tol)
#else
    #define EXPECT_PRECISION_EQ(a, b) EXPECT_DOUBLE_EQ(a, b)
    #define EXPECT_PRECISION_NEAR(a, b, tol) EXPECT_NEAR(a, b, tol)
#endif

} // namespace TestUtils
} // namespace LAMMPS_NS

#endif // TEST_MIXED_PRECISION_UTILS_H
