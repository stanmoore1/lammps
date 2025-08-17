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
   Unit tests for PR 4608: KOKKOS Mixed Precision Support
   Testing core type system and precision infrastructure
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "lammps.h"
#include "kokkos.h"
#include "kokkos_type.h"
#include "kokkos_base.h"
#include "atom_kokkos.h"
#include "memory_kokkos.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <limits>

using namespace LAMMPS_NS;

class MixedPrecisionTypesTest : public ::testing::Test {
protected:
    LAMMPS *lmp;
    
    void SetUp() override {
        const char *args[] = {"test", "-log", "none", "-echo", "none", "-nocite", "-kokkos", "on", "d", "1", "t", "1"};
        char **argv = const_cast<char**>(args);
        int argc = sizeof(args) / sizeof(char*);
        
        lmp = new LAMMPS(argc, argv, MPI_COMM_WORLD);
    }
    
    void TearDown() override {
        delete lmp;
    }
};

// Test 1: Verify KK_FLOAT type definition based on precision mode
TEST_F(MixedPrecisionTypesTest, KKFloatTypeDefinition) {
    // Test that KK_FLOAT is defined correctly based on compilation flags
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    EXPECT_EQ(sizeof(KK_FLOAT), sizeof(float));
    EXPECT_TRUE((std::is_same<KK_FLOAT, float>::value));
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    EXPECT_EQ(sizeof(KK_FLOAT), sizeof(float));
    EXPECT_TRUE((std::is_same<KK_FLOAT, float>::value));
#else // LMP_KOKKOS_DOUBLE_DOUBLE (default)
    EXPECT_EQ(sizeof(KK_FLOAT), sizeof(double));
    EXPECT_TRUE((std::is_same<KK_FLOAT, double>::value));
#endif
}

// Test 2: Verify KK_SUM_FLOAT type definition
TEST_F(MixedPrecisionTypesTest, KKSumFloatTypeDefinition) {
    // KK_SUM_FLOAT should be used for accumulation
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    EXPECT_EQ(sizeof(KK_SUM_FLOAT), sizeof(float));
    EXPECT_TRUE((std::is_same<KK_SUM_FLOAT, float>::value));
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    // Mixed precision: accumulate in double
    EXPECT_EQ(sizeof(KK_SUM_FLOAT), sizeof(double));
    EXPECT_TRUE((std::is_same<KK_SUM_FLOAT, double>::value));
#else // LMP_KOKKOS_DOUBLE_DOUBLE
    EXPECT_EQ(sizeof(KK_SUM_FLOAT), sizeof(double));
    EXPECT_TRUE((std::is_same<KK_SUM_FLOAT, double>::value));
#endif
}

// Test 3: TransformView basic functionality
TEST_F(MixedPrecisionTypesTest, TransformViewBasic) {
    using KKDeviceType = typename KKDevice<LMPDeviceType>::value;
    
    // Test TransformView for 1D arrays
    {
        TransformView<KK_FLOAT*, double*, Kokkos::LayoutRight, KKDeviceType> k_data;
        k_data = TransformView<KK_FLOAT*, double*, Kokkos::LayoutRight, KKDeviceType>("test_1d", 100);
        
        EXPECT_EQ(k_data.h_view.extent(0), 100);
        EXPECT_EQ(k_data.d_view.extent(0), 100);
        
        // Host view should always be double*
        EXPECT_TRUE((std::is_same<decltype(k_data.h_view)::value_type, double>::value));
        
        // Device view should be KK_FLOAT*
        EXPECT_TRUE((std::is_same<decltype(k_data.d_view)::value_type, KK_FLOAT>::value));
    }
    
    // Test TransformView for 2D arrays
    {
        TransformView<KK_FLOAT*[3], double*[3], LMPDeviceLayout, KKDeviceType> k_data;
        k_data = TransformView<KK_FLOAT*[3], double*[3], LMPDeviceLayout, KKDeviceType>("test_2d", 50);
        
        EXPECT_EQ(k_data.h_view.extent(0), 50);
        EXPECT_EQ(k_data.h_view.extent(1), 3);
        EXPECT_EQ(k_data.d_view.extent(0), 50);
        EXPECT_EQ(k_data.d_view.extent(1), 3);
    }
}

// Test 4: TransformView sync operations
TEST_F(MixedPrecisionTypesTest, TransformViewSync) {
    using KKDeviceType = typename KKDevice<LMPDeviceType>::value;
    
    TransformView<KK_FLOAT*, double*, Kokkos::LayoutRight, KKDeviceType> k_data;
    k_data = TransformView<KK_FLOAT*, double*, Kokkos::LayoutRight, KKDeviceType>("sync_test", 10);
    
    // Initialize on host
    for (int i = 0; i < 10; i++) {
        k_data.h_view(i) = static_cast<double>(i * 1.5);
    }
    
    // Sync to device
    k_data.modify_host();
    k_data.sync_device();
    
    // Verify sync worked (copy back to check)
    auto h_mirror = Kokkos::create_mirror_view(k_data.d_view);
    Kokkos::deep_copy(h_mirror, k_data.d_view);
    
    for (int i = 0; i < 10; i++) {
        EXPECT_FLOAT_EQ(h_mirror(i), static_cast<KK_FLOAT>(i * 1.5));
    }
}

// Test 5: ArrayTypes precision-aware typedefs
TEST_F(MixedPrecisionTypesTest, ArrayTypesPrecision) {
    using AT = ArrayTypes<LMPDeviceType>;
    
    // Test 1D float types
    {
        typename AT::t_kkfloat_1d d_array("test_1d", 100);
        EXPECT_EQ(d_array.extent(0), 100);
        EXPECT_TRUE((std::is_same<typename AT::t_kkfloat_1d::value_type, KK_FLOAT>::value));
    }
    
    // Test 2D float types  
    {
        typename AT::t_kkfloat_2d d_array("test_2d", 50, 3);
        EXPECT_EQ(d_array.extent(0), 50);
        EXPECT_EQ(d_array.extent(1), 3);
        EXPECT_TRUE((std::is_same<typename AT::t_kkfloat_2d::value_type, KK_FLOAT>::value));
    }
    
    // Test sum types for accumulation
    {
        typename AT::t_kksum_1d d_sum("test_sum", 100);
        EXPECT_TRUE((std::is_same<typename AT::t_kksum_1d::value_type, KK_SUM_FLOAT>::value));
    }
}

// Test 6: Layout selection based on configuration
TEST_F(MixedPrecisionTypesTest, LayoutSelection) {
#ifdef LMP_KOKKOS_LAYOUT_DEFAULT
    // When using default layout, should use LayoutLeft on GPU
    #ifdef KOKKOS_ENABLE_CUDA
        EXPECT_TRUE((std::is_same<LMPDeviceLayout, Kokkos::LayoutLeft>::value));
    #elif defined(KOKKOS_ENABLE_HIP)
        EXPECT_TRUE((std::is_same<LMPDeviceLayout, Kokkos::LayoutLeft>::value));
    #else
        // CPU always uses LayoutRight
        EXPECT_TRUE((std::is_same<LMPDeviceLayout, Kokkos::LayoutRight>::value));
    #endif
#else // LMP_KOKKOS_LAYOUT_LEGACY (default)
    // Legacy layout uses LayoutRight
    EXPECT_TRUE((std::is_same<LMPDeviceLayout, Kokkos::LayoutRight>::value));
#endif
}

// Test 7: Precision conversion accuracy
TEST_F(MixedPrecisionTypesTest, PrecisionConversion) {
    // Test conversion between double and KK_FLOAT
    double d_val = 1.234567890123456;
    KK_FLOAT kk_val = static_cast<KK_FLOAT>(d_val);
    
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    // Single precision - expect loss of precision
    EXPECT_FLOAT_EQ(kk_val, static_cast<float>(d_val));
    EXPECT_NE(kk_val, d_val); // Should not be exactly equal due to precision loss
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    // Mixed precision - KK_FLOAT is float
    EXPECT_FLOAT_EQ(kk_val, static_cast<float>(d_val));
#else
    // Double precision - no loss
    EXPECT_DOUBLE_EQ(kk_val, d_val);
#endif
}

/* FIXME
// Test 8: Atomic operations with precision types
TEST_F(MixedPrecisionTypesTest, AtomicOperations) {
    using KKDeviceType = typename KKDevice<LMPDeviceType>::value;
    
    // Test atomic add with KK_SUM_FLOAT
    Kokkos::View<KK_SUM_FLOAT*, KKDeviceType, Kokkos::MemoryTraits<Kokkos::Atomic>> d_sum("atomic_sum", 1);
    
    // Initialize to zero
    Kokkos::deep_copy(d_sum, 0.0);
    
    // Perform atomic additions
    Kokkos::parallel_for(100, KOKKOS_LAMBDA(const int i) {
        Kokkos::atomic_add(&d_sum(0), static_cast<KK_SUM_FLOAT>(1.0));
    });
    
    // Check result
    auto h_sum = Kokkos::create_mirror_view(d_sum);
    Kokkos::deep_copy(h_sum, d_sum);
    
    EXPECT_FLOAT_EQ(h_sum(0), 100.0);
}
*/

// Test 9: DAT namespace types
TEST_F(MixedPrecisionTypesTest, DATNamespaceTypes) {
    // Test dual views with precision
    {
        DAT::tdual_kkfloat_1d k_data("dat_test", 50);
        EXPECT_EQ(k_data.h_view.extent(0), 50);
        EXPECT_EQ(k_data.d_view.extent(0), 50);
    }
    
    // Test transform views
    {
        DAT::ttransform_kkfloat_1d k_transform("transform_test", 30);
        EXPECT_EQ(k_transform.h_view.extent(0), 30);
        EXPECT_TRUE((std::is_same<decltype(k_transform.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(k_transform.d_view)::value_type, KK_FLOAT>::value));
    }
}

// Test 10: Memory allocation with MemoryKokkos
TEST_F(MixedPrecisionTypesTest, MemoryKokkosAllocation) {
    auto memoryKK = static_cast<MemoryKokkos*>(lmp->memory);
    
    // Test create_kokkos for 1D array
    {
        DAT::ttransform_kkfloat_1d k_array;
        double* h_ptr = nullptr;
        memoryKK->create_kokkos(k_array, h_ptr, 100, "test_array");
        
        EXPECT_NE(h_ptr, nullptr);
        EXPECT_EQ(k_array.h_view.extent(0), 100);
        EXPECT_EQ(k_array.d_view.extent(0), 100);
        
        // Clean up
        memoryKK->destroy_kokkos(k_array, h_ptr);
    }
    
    // Test create_kokkos for 2D array
    {
        DAT::ttransform_kkfloat_2d k_array;
        double** h_ptr = nullptr;
        memoryKK->create_kokkos(k_array, h_ptr, 50, 3, "test_2d");
        
        EXPECT_NE(h_ptr, nullptr);
        EXPECT_EQ(k_array.h_view.extent(0), 50);
        EXPECT_EQ(k_array.h_view.extent(1), 3);
        
        // Clean up
        memoryKK->destroy_kokkos(k_array, h_ptr);
    }
}

// Test 11: Precision limits and special values
TEST_F(MixedPrecisionTypesTest, PrecisionLimits) {
    // Test numeric limits
    KK_FLOAT max_val = std::numeric_limits<KK_FLOAT>::max();
    KK_FLOAT min_val = std::numeric_limits<KK_FLOAT>::min();
    KK_FLOAT epsilon = std::numeric_limits<KK_FLOAT>::epsilon();
    
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    EXPECT_EQ(max_val, std::numeric_limits<float>::max());
    EXPECT_EQ(min_val, std::numeric_limits<float>::min());
    EXPECT_EQ(epsilon, std::numeric_limits<float>::epsilon());
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    EXPECT_EQ(max_val, std::numeric_limits<float>::max());
    EXPECT_EQ(min_val, std::numeric_limits<float>::min());
    EXPECT_EQ(epsilon, std::numeric_limits<float>::epsilon());
#else
    EXPECT_EQ(max_val, std::numeric_limits<double>::max());
    EXPECT_EQ(min_val, std::numeric_limits<double>::min());
    EXPECT_EQ(epsilon, std::numeric_limits<double>::epsilon());
#endif
    
    // Test special values
    KK_FLOAT inf = std::numeric_limits<KK_FLOAT>::infinity();
    KK_FLOAT nan = std::numeric_limits<KK_FLOAT>::quiet_NaN();
    
    EXPECT_TRUE(std::isinf(inf));
    EXPECT_TRUE(std::isnan(nan));
}

// Test 12: View layout with precision types
TEST_F(MixedPrecisionTypesTest, ViewLayoutPrecision) {
    using AT = ArrayTypes<LMPDeviceType>;
    
    // Test LayoutRight views
    {
        typename AT::t_kkfloat_1d_3_lr d_view("layout_right", 100);
        EXPECT_EQ(d_view.extent(0), 100);
        EXPECT_EQ(d_view.extent(1), 3);
        EXPECT_TRUE((std::is_same<typename decltype(d_view)::array_layout, Kokkos::LayoutRight>::value));
    }
    
    // Test device-dependent layout
    {
        typename AT::t_kkfloat_1d_3 d_view("device_layout", 100);
        EXPECT_EQ(d_view.extent(0), 100);
        EXPECT_EQ(d_view.extent(1), 3);
        // Layout depends on LMPDeviceLayout which varies by configuration
    }
}

// Test 13: Pinned memory support
TEST_F(MixedPrecisionTypesTest, PinnedMemorySupport) {
    using KKDeviceType = typename KKDevice<LMPDeviceType>::value;
    
    // Test if pinned memory is properly configured when using CUDA/HIP
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    DAT::ttransform_kkfloat_1d k_pinned("pinned_test", 100);
    
    // With GPUs, HostKK should use pinned memory
    auto host_space = typename decltype(k_pinned.h_view)::memory_space();
    #ifdef KOKKOS_ENABLE_CUDA
        EXPECT_TRUE((std::is_same<decltype(host_space), Kokkos::CudaHostPinnedSpace>::value));
    #elif defined(KOKKOS_ENABLE_HIP)
        EXPECT_TRUE((std::is_same<decltype(host_space), Kokkos::HIPHostPinnedSpace>::value));
    #endif
#endif
}

/* FIXME
// Test 14: Dual view modification flags
TEST_F(MixedPrecisionTypesTest, DualViewModificationFlags) {
    DAT::tdual_kkfloat_1d k_data("mod_test", 10);
    
    // Initially, neither side should be modified
    EXPECT_EQ(k_data.modified_flags(0), 0);  // Host
    EXPECT_EQ(k_data.modified_flags(1), 0);  // Device
    
    // Modify host
    k_data.modify_host();
    EXPECT_GT(k_data.modified_flags(0), 0);
    EXPECT_EQ(k_data.modified_flags(1), 0);
    
    // Sync to device
    k_data.sync_device();
    EXPECT_EQ(k_data.modified_flags(0), 0);
    EXPECT_EQ(k_data.modified_flags(1), 0);
    
    // Modify device
    k_data.modify_device();
    EXPECT_EQ(k_data.modified_flags(0), 0);
    EXPECT_GT(k_data.modified_flags(1), 0);
}
*/


// Test 15: Precision-aware reductions
TEST_F(MixedPrecisionTypesTest, PrecisionReductions) {
    using KKDeviceType = typename KKDevice<LMPDeviceType>::value;
    
    const int N = 1000;
    Kokkos::View<KK_FLOAT*, KKDeviceType> d_data("reduction_data", N);
    
    // Initialize data
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
        d_data(i) = static_cast<KK_FLOAT>(i * 0.001);
    });
    
    // Perform reduction with KK_SUM_FLOAT accumulator
    KK_SUM_FLOAT sum = 0.0;
    Kokkos::parallel_reduce(N, KOKKOS_LAMBDA(const int i, KK_SUM_FLOAT& lsum) {
        lsum += d_data(i);
    }, sum);
    
    // Expected sum: 0 + 0.001 + 0.002 + ... + 0.999 = 0.001 * (0 + 1 + ... + 999) = 0.001 * 499500 = 499.5
    const KK_SUM_FLOAT expected = static_cast<KK_SUM_FLOAT>(499.5);
    
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    // Single precision - allow for more error
    EXPECT_NEAR(sum, expected, 0.01);
#else
    // Double or mixed precision - should be very accurate
    EXPECT_NEAR(sum, expected, 1e-10);
#endif
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    Kokkos::finalize();
    MPI_Finalize();
    
    return result;
}
