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
   Testing Neighbor List and Domain with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "neighbor_kokkos.h"
#include "npair_kokkos.h"
// nstencil_kokkos.h doesn't exist
#include "domain_kokkos.h"
#include "neigh_list_kokkos.h"
#include "force.h"
#include "modify.h"
#include "input.h"
#include "region.h"
#include "comm_kokkos.h"
#include <cmath>
#include <vector>

namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionNeighborTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
        
        // Create a system for neighbor list testing
        lmp->input->one("units lj");
        lmp->input->one("atom_style atomic");
        lmp->input->one("boundary p p p");
        lmp->input->one("lattice fcc 0.8442");
        lmp->input->one("region box block 0 10 0 10 0 10");
        lmp->input->one("create_box 1 box");
        lmp->input->one("create_atoms 1 box");
        lmp->input->one("mass 1 1.0");
        lmp->input->one("pair_style lj/cut/kk 2.5");
        lmp->input->one("pair_coeff 1 1 1.0 1.0");
        lmp->input->one("neighbor 0.3 bin");
        lmp->input->one("neigh_modify delay 0 every 1");
    }
};

// Test 1: NeighborKokkos basic arrays and precision
TEST_F(MixedPrecisionNeighborTest, NeighborKokkosArrays) {
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    ASSERT_NE(neighborKK, nullptr);
    
    // Trigger neighbor list build
    lmp->input->one("run 0");
    
    // Check cutneighsq array precision (should use KK_FLOAT)
    EXPECT_TRUE((std::is_same<decltype(neighborKK->k_cutneighsq.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(neighborKK->k_cutneighsq.d_view)::value_type, KK_FLOAT>::value));
    
    // Check bboxlo/bboxhi precision
    EXPECT_TRUE((std::is_same<decltype(neighborKK->k_bboxlo.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(neighborKK->k_bboxlo.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(neighborKK->k_bboxhi.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(neighborKK->k_bboxhi.d_view)::value_type, KK_FLOAT>::value));
    
    // Verify values are reasonable
    for (int i = 0; i < 3; i++) {
        EXPECT_TRUE(checkNumericalStability(neighborKK->bboxlo[i]));
        EXPECT_TRUE(checkNumericalStability(neighborKK->bboxhi[i]));
        EXPECT_GT(neighborKK->bboxhi[i], neighborKK->bboxlo[i]);
    }
}

// Test 2: NeighborKokkos distance calculations with precision
TEST_F(MixedPrecisionNeighborTest, NeighborDistanceCalculations) {
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    ASSERT_NE(neighborKK, nullptr);
    
    // Set specific cutoff
    double cutoff = 2.5;
    neighborKK->cutneighmax = cutoff;
    neighborKK->cutneighsq[0][0] = cutoff * cutoff;
    
    // Build neighbor list
    lmp->input->one("run 0");
    
    // Check distance check implementation
    // This would involve testing the distance calculations use KK_FLOAT
    double cutsq = neighborKK->cutneighsq[0][0];
    EXPECT_PRECISION_NEAR(cutsq, cutoff * cutoff, getAbsoluteTolerance());
    
    // Verify skin distance
    EXPECT_GT(neighborKK->skin, 0.0);
    EXPECT_TRUE(checkNumericalStability(neighborKK->skin));
}

// Test 3: NPairKokkos neighbor list building
TEST_F(MixedPrecisionNeighborTest, NPairKokkosBuild) {
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    ASSERT_NE(neighborKK, nullptr);
    
    // Build neighbor lists
    lmp->input->one("run 0");
    
    // Access neighbor list
    ASSERT_GT(neighborKK->nlist, 0);
    auto list = neighborKK->lists[0];
    ASSERT_NE(list, nullptr);
    
    auto listKK = dynamic_cast<NeighListKokkos<LMPDeviceType>*>(list);
    ASSERT_NE(listKK, nullptr);
    
    // Check that neighbor list arrays use correct types
    EXPECT_GT(list->inum, 0);
    EXPECT_GT(list->gnum, 0);
    
    // ilist should be integer
    EXPECT_TRUE((std::is_same<decltype(listKK->k_ilist.h_view)::value_type, int>::value));
    
    // numneigh should be integer
    EXPECT_TRUE((std::is_same<decltype(listKK->k_numneigh.h_view)::value_type, int>::value));
    
    // neighbors should be integer
    EXPECT_TRUE((std::is_same<decltype(listKK->k_neighbors.h_view)::value_type, int>::value));
}

// Test 4: NPairKokkos with ghost atoms
TEST_F(MixedPrecisionNeighborTest, NPairKokkosGhosts) {
    // Create a smaller system to ensure ghosts
    lmp->input->one("clear");
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary p p p");
    lmp->input->one("lattice fcc 0.8442");
    lmp->input->one("region box block 0 3 0 3 0 3");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    lmp->input->one("neighbor 0.3 bin");
    
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    lmp->input->one("run 0");
    
    // Check ghost atom handling
    int nlocal = atomKK->nlocal;
    int nghost = atomKK->nghost;
    
    EXPECT_GT(nlocal, 0);
    EXPECT_GE(nghost, 0);  // May or may not have ghosts depending on decomposition
    
    // Verify neighbor list includes appropriate atoms
    auto list = neighborKK->lists[0];
    EXPECT_LE(list->inum, nlocal);
    EXPECT_EQ(list->gnum, nlocal + nghost);
}

// Test 5: NStencilKokkos stencil creation
TEST_F(MixedPrecisionNeighborTest, NStencilKokkos) {
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    ASSERT_NE(neighborKK, nullptr);
    
    // Trigger stencil creation
    lmp->input->one("run 0");
    
    // Access stencil through neighbor list
    ASSERT_GT(neighborKK->nstencil, 0);
    auto ns = neighborKK->nstencil_list[0];
    ASSERT_NE(ns, nullptr);
    
    // Check stencil properties
    EXPECT_GT(ns->nstencil, 0);  // Should have stencil points
    
    // Stencil arrays should use integer types
    if (ns->stencil) {
        for (int i = 0; i < ns->nstencil; i++) {
            // Stencil values should be reasonable bin offsets
            EXPECT_GE(ns->stencil[i], -27);  // 3x3x3 cube max
            EXPECT_LE(ns->stencil[i], 27);
        }
    }
}

// Test 6: DomainKokkos precision in coordinate transformations
TEST_F(MixedPrecisionNeighborTest, DomainKokkosPrecision) {
    auto domainKK = dynamic_cast<DomainKokkos*>(lmp->domain);
    ASSERT_NE(domainKK, nullptr);
    
    // Check box dimension arrays
    EXPECT_TRUE((std::is_same<decltype(domainKK->k_boxlo.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(domainKK->k_boxlo.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(domainKK->k_boxhi.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(domainKK->k_boxhi.d_view)::value_type, KK_FLOAT>::value));
    
    // Check periodicity arrays
    EXPECT_TRUE((std::is_same<decltype(domainKK->k_prd.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(domainKK->k_prd.d_view)::value_type, KK_FLOAT>::value));
    
    // Verify box dimensions
    for (int i = 0; i < 3; i++) {
        EXPECT_LT(domainKK->boxlo[i], domainKK->boxhi[i]);
        EXPECT_GT(domainKK->prd[i], 0.0);
        EXPECT_TRUE(checkNumericalStability(domainKK->boxlo[i]));
        EXPECT_TRUE(checkNumericalStability(domainKK->boxhi[i]));
        EXPECT_TRUE(checkNumericalStability(domainKK->prd[i]));
    }
}

// Test 7: DomainKokkos periodic boundary conditions
TEST_F(MixedPrecisionNeighborTest, DomainPBC) {
    auto domainKK = dynamic_cast<DomainKokkos*>(lmp->domain);
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Test PBC image flags
    lmp->input->one("run 0");
    
    // Check image array
    atomKK->sync(Host, IMAGE_MASK);
    for (int i = 0; i < atomKK->nlocal; i++) {
        imageint image = atomKK->image[i];
        // Image flags should be reasonable
        int xbox = (image & IMGMASK) - IMGMAX;
        int ybox = ((image >> IMGBITS) & IMGMASK) - IMGMAX;
        int zbox = ((image >> IMG2BITS) & IMGMASK) - IMGMAX;
        
        EXPECT_GE(xbox, -10);
        EXPECT_LE(xbox, 10);
        EXPECT_GE(ybox, -10);
        EXPECT_LE(ybox, 10);
        EXPECT_GE(zbox, -10);
        EXPECT_LE(zbox, 10);
    }
}

// Test 8: Neighbor list rebuild triggering
TEST_F(MixedPrecisionNeighborTest, NeighborListRebuild) {
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Initial build
    lmp->input->one("run 0");
    int initial_nbuilds = neighborKK->ncalls;
    
    // Move atoms slightly
    atomKK->sync(Host, X_MASK);
    for (int i = 0; i < atomKK->nlocal; i++) {
        atomKK->x[i][0] += 0.01;
    }
    atomKK->modified(Host, X_MASK);
    
    // Run again
    lmp->input->one("run 1");
    
    // Check if rebuild logic works correctly
    int final_nbuilds = neighborKK->ncalls;
    EXPECT_GE(final_nbuilds, initial_nbuilds);
    
    // Move atoms significantly to force rebuild
    atomKK->sync(Host, X_MASK);
    for (int i = 0; i < atomKK->nlocal; i++) {
        atomKK->x[i][0] += 0.5;  // Move beyond skin distance
    }
    atomKK->modified(Host, X_MASK);
    
    lmp->input->one("run 1");
    EXPECT_GT(neighborKK->ncalls, final_nbuilds);
}

// Test 9: Binning precision
TEST_F(MixedPrecisionNeighborTest, BinningPrecision) {
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    
    lmp->input->one("run 0");
    
    // Check bin arrays
    EXPECT_GT(neighborKK->mbins, 0);
    EXPECT_GT(neighborKK->nbinx, 0);
    EXPECT_GT(neighborKK->nbiny, 0);
    EXPECT_GT(neighborKK->nbinz, 0);
    
    // Bin sizes should use KK_FLOAT precision
    EXPECT_GT(neighborKK->binsizex, 0.0);
    EXPECT_GT(neighborKK->binsizey, 0.0);
    EXPECT_GT(neighborKK->binsizez, 0.0);
    
    EXPECT_TRUE(checkNumericalStability(neighborKK->binsizex));
    EXPECT_TRUE(checkNumericalStability(neighborKK->binsizey));
    EXPECT_TRUE(checkNumericalStability(neighborKK->binsizez));
    
    // Inverse bin sizes
    EXPECT_TRUE(checkNumericalStability(neighborKK->bininvx));
    EXPECT_TRUE(checkNumericalStability(neighborKK->bininvy));
    EXPECT_TRUE(checkNumericalStability(neighborKK->bininvz));
}

// Test 10: Multi-neighbor list handling
TEST_F(MixedPrecisionNeighborTest, MultipleNeighborLists) {
    // Add a second pair style to create multiple neighbor lists
    lmp->input->one("pair_style hybrid lj/cut/kk 2.5 morse/kk 3.0");
    lmp->input->one("pair_coeff 1 1 lj/cut/kk 1.0 1.0");
    lmp->input->one("pair_coeff 1 1 morse/kk 5.0 1.5 1.0");
    
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    
    lmp->input->one("run 0");
    
    // Should have multiple neighbor lists
    EXPECT_GT(neighborKK->nlist, 1);
    
    // Check each list
    for (int i = 0; i < neighborKK->nlist; i++) {
        auto list = neighborKK->lists[i];
        ASSERT_NE(list, nullptr);
        
        auto listKK = dynamic_cast<NeighListKokkos<LMPDeviceType>*>(list);
        if (listKK) {
            EXPECT_GT(list->inum, 0);
            EXPECT_GT(list->gnum, 0);
        }
    }
}

// Test 11: Neighbor list with exclusions
TEST_F(MixedPrecisionNeighborTest, NeighborExclusions) {
    // Create molecular system with exclusions
    lmp->input->one("clear");
    lmp->input->one("units real");
    lmp->input->one("atom_style full");
    lmp->input->one("boundary p p p");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box bond/types 1");
    
    // Create a few bonded atoms
    lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
    lmp->input->one("create_atoms 1 single 6.0 5.0 5.0");
    lmp->input->one("create_atoms 1 single 7.0 5.0 5.0");
    
    lmp->input->one("mass 1 1.0");
    lmp->input->one("pair_style lj/cut/kk 10.0");
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    lmp->input->one("bond_style harmonic/kk");
    lmp->input->one("bond_coeff 1 100.0 1.0");
    
    lmp->input->one("neighbor 2.0 bin");
    lmp->input->one("neigh_modify exclude bond");
    
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    
    lmp->input->one("run 0");
    
    // Verify exclusions are handled
    EXPECT_GT(neighborKK->nexclude, 0);
}

// Test 12: Domain decomposition with neighbor lists
TEST_F(MixedPrecisionNeighborTest, DomainDecomposition) {
    auto domainKK = dynamic_cast<DomainKokkos*>(lmp->domain);
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    lmp->input->one("run 0");
    
    // Check subbox dimensions
    for (int i = 0; i < 3; i++) {
        EXPECT_LE(domainKK->sublo[i], domainKK->subhi[i]);
        EXPECT_TRUE(checkNumericalStability(domainKK->sublo[i]));
        EXPECT_TRUE(checkNumericalStability(domainKK->subhi[i]));
    }
    
    // Verify all local atoms are within subdomain
    atomKK->sync(Host, X_MASK);
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            // Allow for numerical tolerance
            EXPECT_GE(atomKK->x[i][j], domainKK->sublo[j] - 1e-6);
            EXPECT_LE(atomKK->x[i][j], domainKK->subhi[j] + 1e-6);
        }
    }
}

// Test 13: Triclinic box handling
TEST_F(MixedPrecisionNeighborTest, TriclinicBox) {
    // Test triclinic (non-orthogonal) box
    lmp->input->one("clear");
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary p p p");
    
    // Create triclinic box
    lmp->input->one("lattice fcc 0.8442");
    lmp->input->one("region box prism 0 10 0 10 0 10 1.0 0.0 0.0");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    
    auto domainKK = dynamic_cast<DomainKokkos*>(lmp->domain);
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    
    lmp->input->one("run 0");
    
    // Check triclinic flag
    EXPECT_EQ(domainKK->triclinic, 1);
    
    // Check tilt factors
    EXPECT_TRUE(checkNumericalStability(domainKK->xy));
    EXPECT_TRUE(checkNumericalStability(domainKK->xz));
    EXPECT_TRUE(checkNumericalStability(domainKK->yz));
    
    // Neighbor lists should still work correctly
    ASSERT_GT(neighborKK->nlist, 0);
    auto list = neighborKK->lists[0];
    EXPECT_GT(list->inum, 0);
}

// Test 14: Neighbor list size and reallocation
TEST_F(MixedPrecisionNeighborTest, NeighborListReallocation) {
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    
    // Build initial list
    lmp->input->one("run 0");
    
    auto list = neighborKK->lists[0];
    auto listKK = dynamic_cast<NeighListKokkos<LMPDeviceType>*>(list);
    ASSERT_NE(listKK, nullptr);
    
    int initial_maxneigh = listKK->maxneighs;
    
    // Create a denser system to trigger reallocation
    lmp->input->one("clear");
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary p p p");
    lmp->input->one("lattice fcc 1.2");  // Higher density
    lmp->input->one("region box block 0 5 0 5 0 5");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("pair_style lj/cut/kk 3.0");  // Larger cutoff
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    lmp->input->one("neighbor 0.5 bin");
    
    neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    
    lmp->input->one("run 0");
    
    // Check if reallocation occurred properly
    list = neighborKK->lists[0];
    listKK = dynamic_cast<NeighListKokkos<LMPDeviceType>*>(list);
    
    EXPECT_GT(listKK->maxneighs, 0);
    
    // Verify all neighbor counts are valid
    listKK->k_numneigh.sync_host();
    for (int i = 0; i < list->inum; i++) {
        int n = listKK->numneigh[i];
        EXPECT_GE(n, 0);
        EXPECT_LE(n, listKK->maxneighs);
    }
}

// Test 15: Extreme cutoff values
TEST_F(MixedPrecisionNeighborTest, ExtremeCutoffs) {
    auto neighborKK = dynamic_cast<NeighborKokkos*>(lmp->neighbor);
    
    // Test very small cutoff
    lmp->input->one("pair_style lj/cut/kk 0.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    lmp->input->one("neighbor 0.1 bin");
    lmp->input->one("run 0");
    
    // Should handle small cutoffs without issues
    EXPECT_GT(neighborKK->cutneighmax, 0.0);
    EXPECT_TRUE(checkNumericalStability(neighborKK->cutneighmax));
    
    // Test large cutoff
    lmp->input->one("pair_style lj/cut/kk 5.0");
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    lmp->input->one("neighbor 0.5 bin");
    lmp->input->one("run 0");
    
    // Should handle large cutoffs
    EXPECT_GT(neighborKK->cutneighmax, 4.0);
    EXPECT_TRUE(checkNumericalStability(neighborKK->cutneighmax));
    
    // Verify neighbor lists are built correctly
    auto list = neighborKK->lists[0];
    EXPECT_GT(list->inum, 0);
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
