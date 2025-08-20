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
   Testing compute styles with mixed precision - simplified version
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "compute_temp_kokkos.h"
#include "compute_coord_atom_kokkos.h"
#include "compute.h"
#include "force.h"
#include "neighbor.h"
#include "modify.h"
#include "input.h"
#include <cmath>
#include <vector>

namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionComputesTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
        
        // Create a simple system for testing computes
        lmp->input->one("units lj");
        lmp->input->one("atom_style atomic");
        lmp->input->one("boundary p p p");
        lmp->input->one("lattice fcc 0.8442");
        lmp->input->one("region box block 0 4 0 4 0 4");
        lmp->input->one("create_box 1 box");
        lmp->input->one("create_atoms 1 box");
        lmp->input->one("mass 1 1.0");
        lmp->input->one("pair_style lj/cut/kk 2.5");
        lmp->input->one("pair_coeff 1 1 1.0 1.0");
        lmp->input->one("velocity all create 1.0 12345");
    }
};

// Test 1: ComputeTempKokkos precision types
TEST_F(MixedPrecisionComputesTest, ComputeTempTypes) {
    lmp->input->one("compute mytemp all temp/kk");
    
    int icompute = lmp->modify->find_compute("mytemp");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputeTempKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // Run compute to allocate arrays
    lmp->input->one("run 0");
    
    // Temperature should be calculated using KK_FLOAT internally
    double temp = compute->scalar;
    EXPECT_GT(temp, 0.0);
    EXPECT_TRUE(checkNumericalStability(temp));
    
    // Check that DOF is properly computed
    EXPECT_GT(compute->dof, 0.0);
}

// Test 2: ComputeTempKokkos velocity calculations
TEST_F(MixedPrecisionComputesTest, ComputeTempVelocity) {
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("run 0");
    
    int icompute = lmp->modify->find_compute("mytemp");
    auto compute = dynamic_cast<ComputeTempKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // Temperature should match theoretical value for given velocity
    double temp = compute->scalar;
    EXPECT_PRECISION_NEAR(temp, 1.0, getRelativeTolerance() * 10);
    
    // Check vector values (KE components) if available
    if (compute->vector) {
        for (int i = 0; i < 6; i++) {
            EXPECT_TRUE(checkNumericalStability(compute->vector[i]));
        }
    }
}

// Test 3: ComputeCoordAtomKokkos coordination number
TEST_F(MixedPrecisionComputesTest, ComputeCoordAtom) {
    lmp->input->one("compute mycoord all coord/atom/kk cutoff 1.5");
    lmp->input->one("run 0");
    
    int icompute = lmp->modify->find_compute("mycoord");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputeCoordAtomKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // Check coordination numbers
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    if (compute->vector_atom) {
        double avg_coord = 0.0;
        int count = 0;
        
        for (int i = 0; i < atomKK->nlocal; i++) {
            EXPECT_TRUE(checkNumericalStability(compute->vector_atom[i]));
            EXPECT_GE(compute->vector_atom[i], 0.0);
            
            avg_coord += compute->vector_atom[i];
            count++;
        }
        
        if (count > 0) {
            avg_coord /= count;
            // Average should be reasonable for FCC
            EXPECT_GT(avg_coord, 8.0);  // Lower bound
            EXPECT_LT(avg_coord, 13.0); // Upper bound
        }
    }
}

// Test 4: Regular compute with Kokkos atoms
TEST_F(MixedPrecisionComputesTest, RegularComputeWithKokkos) {
    // Test that regular computes work with Kokkos atom data
    lmp->input->one("compute mype all pe");
    lmp->input->one("compute myke all ke");
    lmp->input->one("run 0");
    
    int ipe = lmp->modify->find_compute("mype");
    int ike = lmp->modify->find_compute("myke");
    
    ASSERT_GE(ipe, 0);
    ASSERT_GE(ike, 0);
    
    double pe = lmp->modify->compute[ipe]->scalar;
    double ke = lmp->modify->compute[ike]->scalar;
    
    EXPECT_LT(pe, 0.0);  // LJ should be negative
    EXPECT_GT(ke, 0.0);  // KE should be positive
    
    EXPECT_TRUE(checkNumericalStability(pe));
    EXPECT_TRUE(checkNumericalStability(ke));
}

// Test 5: Multiple computes together
TEST_F(MixedPrecisionComputesTest, MultipleComputes) {
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("compute mycoord all coord/atom/kk cutoff 1.5");
    lmp->input->one("compute mype all pe");
    
    lmp->input->one("run 1");
    
    // Check all computes work together
    int itemp = lmp->modify->find_compute("mytemp");
    int icoord = lmp->modify->find_compute("mycoord");
    int ipe = lmp->modify->find_compute("mype");
    
    ASSERT_GE(itemp, 0);
    ASSERT_GE(icoord, 0);
    ASSERT_GE(ipe, 0);
    
    double temp = lmp->modify->compute[itemp]->scalar;
    double pe = lmp->modify->compute[ipe]->scalar;
    
    EXPECT_TRUE(checkNumericalStability(temp));
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 6: Compute with fix modification
TEST_F(MixedPrecisionComputesTest, ComputeWithFixModification) {
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("fix 1 all nve");
    lmp->input->one("fix_modify 1 temp mytemp");
    
    lmp->input->one("run 10");
    
    int icompute = lmp->modify->find_compute("mytemp");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputeTempKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    double temp = compute->scalar;
    EXPECT_GT(temp, 0.0);
    EXPECT_TRUE(checkNumericalStability(temp));
}

// Test 7: Statistical precision over time
TEST_F(MixedPrecisionComputesTest, StatisticalPrecision) {
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("fix 1 all nvt temp 1.0 1.0 0.1");
    lmp->input->one("fix_modify 1 temp mytemp");
    
    std::vector<double> temps;
    
    // Collect temperature samples
    for (int step = 0; step < 50; step++) {
        lmp->input->one("run 10");
        
        int icompute = lmp->modify->find_compute("mytemp");
        double temp = lmp->modify->compute[icompute]->scalar;
        temps.push_back(temp);
    }
    
    // Calculate mean
    double mean = 0.0;
    for (double t : temps) {
        mean += t;
    }
    mean /= temps.size();
    
    // Mean should be close to target
    EXPECT_PRECISION_NEAR(mean, 1.0, 0.2);
    
    // Check stability
    for (double t : temps) {
        EXPECT_TRUE(checkNumericalStability(t));
    }
}

// Test 8: Extreme value handling
TEST_F(MixedPrecisionComputesTest, ExtremeValueHandling) {
    // Very high velocities
    lmp->input->one("velocity all create 1000.0 12345");
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("run 0");
    
    int itemp = lmp->modify->find_compute("mytemp");
    double temp = lmp->modify->compute[itemp]->scalar;
    
    EXPECT_TRUE(checkNumericalStability(temp));
    EXPECT_GT(temp, 100.0);
    EXPECT_LT(temp, 1e10);
    
    // Very low velocities
    lmp->input->one("velocity all create 0.001 12345");
    lmp->input->one("run 0");
    
    temp = lmp->modify->compute[itemp]->scalar;
    
    EXPECT_TRUE(checkNumericalStability(temp));
    EXPECT_GT(temp, 0.0);
    EXPECT_LT(temp, 0.01);
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
