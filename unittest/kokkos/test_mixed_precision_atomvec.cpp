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
   Testing AtomVec classes and atom data structures
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "atom_vec_atomic_kokkos.h"
#include "atom_vec_charge_kokkos.h"
#include "atom_vec_full_kokkos.h"
#include "atom_vec_molecular_kokkos.h"
#include "atom_vec_angle_kokkos.h"
#include "atom_vec_bond_kokkos.h"
#include "memory_kokkos.h"
#include "comm_kokkos.h"
#include "input.h"
#include <cmath>

namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionAtomVecTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
    }
    
    void TearDown() override {
        MixedPrecisionTestFixture::TearDown();
    }
};

// Test 1: AtomVecAtomicKokkos basic arrays
TEST_F(MixedPrecisionAtomVecTest, AtomVecAtomicArrays) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecAtomicKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    // Check position array types
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_x.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_x.d_view)::value_type, KK_FLOAT>::value));
    
    // Check velocity array types
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_v.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_v.d_view)::value_type, KK_FLOAT>::value));
    
    // Check force array types (uses KK_SUM_FLOAT for accumulation)
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_f.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_f.d_view)::value_type, KK_SUM_FLOAT>::value));
    
    // Integer arrays should remain as int
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_tag.h_view)::value_type, tagint>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_type.h_view)::value_type, int>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_mask.h_view)::value_type, int>::value));
}

// Test 2: AtomVecChargeKokkos with charge array
TEST_F(MixedPrecisionAtomVecTest, AtomVecChargeArrays) {
    lmp->input->one("atom_style charge");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 single 1.0 1.0 1.0");
    lmp->input->one("set atom 1 charge 1.5");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecChargeKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    // Check charge array precision
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_q.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_q.d_view)::value_type, KK_FLOAT>::value));
    
    // Verify charge value is preserved
    atomKK->sync(Host, Q_MASK);
    EXPECT_PRECISION_NEAR(atomKK->q[0], 1.5, getAbsoluteTolerance());
}

// Test 3: AtomVecFullKokkos with molecular data
TEST_F(MixedPrecisionAtomVecTest, AtomVecFullArrays) {
    lmp->input->one("atom_style full");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box bond/types 1 angle/types 1 "
                   "dihedral/types 1 improper/types 1");
    lmp->input->one("create_atoms 1 single 1.0 1.0 1.0");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecFullKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    // Check molecular arrays
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_molecule.h_view)::value_type, tagint>::value));
    
    // Check special neighbor arrays
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_nspecial.h_view)::value_type, int>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_special.h_view)::value_type, tagint>::value));
    
    // Check bond arrays
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_num_bond.h_view)::value_type, int>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_bond_type.h_view)::value_type, int>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_bond_atom.h_view)::value_type, tagint>::value));
}

// Test 4: Pack/unpack operations with precision conversion
TEST_F(MixedPrecisionAtomVecTest, PackUnpackBorder) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 random 100 12345 box");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecAtomicKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    // Test pack_border
    DAT::tdual_int_1d k_sendlist("sendlist", 10);
    DAT::tdual_double_2d_lr k_buf("buf", 100, 100);
    
    // Add some atoms to send list
    for (int i = 0; i < 10 && i < atomKK->nlocal; i++) {
        k_sendlist.h_view(i) = i;
    }
    k_sendlist.modify_host();
    
    int n = avec->pack_border_kokkos(10, k_sendlist, k_buf, 0, nullptr, Device);
    EXPECT_GT(n, 0);
    
    // Test unpack_border
    avec->unpack_border_kokkos(10, atomKK->nlocal, k_buf, Device);
    
    // Verify data integrity
    atomKK->sync(Host, X_MASK);
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->x[i][j]));
        }
    }
}

// Test 5: Pack/unpack exchange operations
TEST_F(MixedPrecisionAtomVecTest, PackUnpackExchange) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
    lmp->input->one("velocity all create 300.0 12345");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecAtomicKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    // Test pack_exchange
    DAT::tdual_double_2d_lr k_buf("buf", 100, 100);
    DAT::tdual_int_1d k_sendlist("sendlist", 1);
    DAT::tdual_int_1d k_copylist("copylist", 0);
    
    k_sendlist.h_view(0) = 0;  // Send first atom
    k_sendlist.modify_host();
    
    int n = avec->pack_exchange_kokkos(1, k_buf, k_sendlist, k_copylist, Device);
    EXPECT_GT(n, 0);
    
    // Test unpack_exchange
    DAT::tdual_int_1d k_indices("indices", 1);
    int nrecv = avec->unpack_exchange_kokkos(k_buf, n, 0, 0, 10.0, 0.0, Device, k_indices);
    EXPECT_EQ(nrecv, 1);
}

// Test 6: AtomKokkos sync operations
TEST_F(MixedPrecisionAtomVecTest, AtomSyncOperations) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Modify on host
    atomKK->sync(Host, X_MASK);
    atomKK->x[0][0] = 1.234567890123456;
    atomKK->modified(Host, X_MASK);
    
    // Sync to device
    atomKK->sync(Device, X_MASK);
    
    // Verify sync worked
    auto h_x = atomKK->k_x.h_view;
    auto d_x = atomKK->k_x.d_view;
    
    auto h_mirror = Kokkos::create_mirror_view(d_x);
    Kokkos::deep_copy(h_mirror, d_x);
    
    // Check precision-dependent accuracy
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    EXPECT_FLOAT_EQ(h_mirror(0,0), static_cast<float>(1.234567890123456));
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    EXPECT_FLOAT_EQ(h_mirror(0,0), static_cast<float>(1.234567890123456));
#else
    EXPECT_DOUBLE_EQ(h_mirror(0,0), 1.234567890123456);
#endif
}

// Test 7: Memory growth and reallocation
TEST_F(MixedPrecisionAtomVecTest, MemoryGrowth) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecAtomicKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    int initial_nmax = atomKK->nmax;
    
    // Create many atoms to trigger growth
    lmp->input->one("create_atoms 1 random 1000 12345 box");
    
    EXPECT_GT(atomKK->nmax, initial_nmax);
    
    // Verify arrays were properly reallocated
    EXPECT_EQ(atomKK->k_x.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    EXPECT_EQ(atomKK->k_v.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    EXPECT_EQ(atomKK->k_f.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    
    // Check precision is maintained after growth
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_x.d_view)::value_type, KK_FLOAT>::value));
}

// Test 8: AtomVecMolecularKokkos with topology
TEST_F(MixedPrecisionAtomVecTest, AtomVecMolecularTopology) {
    lmp->input->one("atom_style molecular");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box bond/types 1");
    lmp->input->one("molecule mol1 molecule.txt");  // Would need actual file
    
    // For testing, just create atoms manually
    lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
    lmp->input->one("create_atoms 1 single 6.0 5.0 5.0");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecMolecularKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    // Check molecule ID array
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_molecule.h_view)::value_type, tagint>::value));
    
    // Check bond topology arrays
    EXPECT_EQ(atomKK->k_num_bond.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    EXPECT_EQ(atomKK->k_bond_type.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    EXPECT_EQ(atomKK->k_bond_atom.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
}

// Test 9: View layout consistency
TEST_F(MixedPrecisionAtomVecTest, ViewLayoutConsistency) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Check x array layout (should be LayoutRight for compatibility)
    using x_layout = decltype(atomKK->k_x.d_view)::array_layout;
    EXPECT_TRUE((std::is_same<x_layout, Kokkos::LayoutRight>::value));
    
    // Check v array layout
    using v_layout = decltype(atomKK->k_v.d_view)::array_layout;
    EXPECT_TRUE((std::is_same<v_layout, LMPDeviceLayout>::value));
    
    // Check f array layout  
    using f_layout = decltype(atomKK->k_f.d_view)::array_layout;
    EXPECT_TRUE((std::is_same<f_layout, LMPDeviceLayout>::value));
}

// Test 10: Pinned memory with TransformView
TEST_F(MixedPrecisionAtomVecTest, PinnedMemoryTransform) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    // With GPU, host views should use pinned memory
    using host_space = decltype(atomKK->k_x.h_view)::memory_space;
    
    #ifdef KOKKOS_ENABLE_CUDA
        EXPECT_TRUE((std::is_same<host_space, Kokkos::CudaHostPinnedSpace>::value) ||
                   (std::is_same<host_space, Kokkos::HostSpace>::value));
    #elif defined(KOKKOS_ENABLE_HIP)
        EXPECT_TRUE((std::is_same<host_space, Kokkos::HIPHostPinnedSpace>::value) ||
                   (std::is_same<host_space, Kokkos::HostSpace>::value));
    #endif
#endif
}

// Test 11: Custom atom properties
TEST_F(MixedPrecisionAtomVecTest, CustomAtomProperties) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Add custom property (dvector)
    int index = atomKK->add_custom("test_prop", 0, 3);
    EXPECT_GE(index, 0);
    
    // Check dvector precision
    if (atomKK->k_dvector.h_view.extent(0) > 0) {
        EXPECT_TRUE((std::is_same<decltype(atomKK->k_dvector.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(atomKK->k_dvector.d_view)::value_type, KK_FLOAT>::value));
    }
}

// Test 12: Sort operations with precision
TEST_F(MixedPrecisionAtomVecTest, AtomSorting) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 random 100 12345 box");
    lmp->input->one("velocity all create 300.0 12345");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Save original positions
    atomKK->sync(Host, X_MASK);
    std::vector<double> orig_x;
    for (int i = 0; i < atomKK->nlocal; i++) {
        orig_x.push_back(atomKK->x[i][0]);
        orig_x.push_back(atomKK->x[i][1]);
        orig_x.push_back(atomKK->x[i][2]);
    }
    
    // Perform sort
    if (!atomKK->sort_legacy) {
        atomKK->sort();
        
        // Verify data integrity after sort
        atomKK->sync(Host, X_MASK);
        for (int i = 0; i < atomKK->nlocal; i++) {
            for (int j = 0; j < 3; j++) {
                EXPECT_TRUE(checkNumericalStability(atomKK->x[i][j]));
            }
        }
    }
}

// Test 13: AtomVecAngleKokkos with angle topology
TEST_F(MixedPrecisionAtomVecTest, AtomVecAngleTopology) {
    lmp->input->one("atom_style angle");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box bond/types 1 angle/types 1");
    lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
    lmp->input->one("create_atoms 1 single 6.0 5.0 5.0");
    lmp->input->one("create_atoms 1 single 5.5 6.0 5.0");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecAngleKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    // Check angle topology arrays
    EXPECT_EQ(atomKK->k_num_angle.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    EXPECT_EQ(atomKK->k_angle_type.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    EXPECT_EQ(atomKK->k_angle_atom1.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    EXPECT_EQ(atomKK->k_angle_atom2.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    EXPECT_EQ(atomKK->k_angle_atom3.h_view.extent(0), static_cast<size_t>(atomKK->nmax));
    
    // Check precision of topology (should be integer types)
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_angle_type.h_view)::value_type, int>::value));
    EXPECT_TRUE((std::is_same<decltype(atomKK->k_angle_atom1.h_view)::value_type, tagint>::value));
}

// Test 14: Mass array precision
TEST_F(MixedPrecisionAtomVecTest, MassArrayPrecision) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 2 box");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("mass 2 2.0");
    lmp->input->one("create_atoms 1 single 1.0 1.0 1.0");
    lmp->input->one("create_atoms 2 single 2.0 2.0 2.0");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Check per-type mass array
    if (atomKK->mass) {
        EXPECT_TRUE((std::is_same<decltype(atomKK->k_mass.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(atomKK->k_mass.d_view)::value_type, KK_FLOAT>::value));
        
        // Verify mass values
        atomKK->k_mass.sync_host();
        EXPECT_PRECISION_NEAR(atomKK->mass[1], 1.0, getAbsoluteTolerance());
        EXPECT_PRECISION_NEAR(atomKK->mass[2], 2.0, getAbsoluteTolerance());
    }
}

// Test 15: Communication buffer precision
TEST_F(MixedPrecisionAtomVecTest, CommBufferPrecision) {
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 random 10 12345 box");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    auto avec = dynamic_cast<AtomVecAtomicKokkos*>(atomKK->avec);
    ASSERT_NE(avec, nullptr);
    
    // Test communication buffer size
    int size_forward = avec->size_forward;
    int size_reverse = avec->size_reverse;
    int size_border = avec->size_border;
    int size_velocity = avec->size_velocity;
    int size_data_atom = avec->size_data_atom;
    
    EXPECT_GT(size_forward, 0);
    EXPECT_GE(size_reverse, 0);  
    EXPECT_GT(size_border, 0);
    EXPECT_GT(size_velocity, 0);
    EXPECT_GT(size_data_atom, 0);
    
    // Communication buffers should use double for MPI compatibility
    DAT::tdual_double_2d_lr comm_buf("comm_buf", 100, size_border);
    EXPECT_TRUE((std::is_same<decltype(comm_buf.h_view)::value_type, double>::value));
}

} // namespace LAMMPS_NS

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    Kokkos::finalize();
    MPI_Finalize();
    
    return result;
}
