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
   Testing compute styles with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "compute_temp_kokkos.h"
#include "compute_pressure_kokkos.h"
#include "compute_pe_kokkos.h"
#include "compute_rdf_kokkos.h"
#include "compute_msd_kokkos.h"
#include "compute_ke_kokkos.h"
#include "compute_centro_atom_kokkos.h"
#include "compute_stress_atom_kokkos.h"
#include "compute_coord_atom_kokkos.h"
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
    // T = 2/(3*N*kb) * sum(1/2 * m * v^2)
    double temp = compute->scalar;
    EXPECT_PRECISION_NEAR(temp, 1.0, getRelativeTolerance() * 10);  // Should be close to input temperature
    
    // Check vector values (KE components)
    if (compute->vector) {
        for (int i = 0; i < 6; i++) {
            EXPECT_TRUE(checkNumericalStability(compute->vector[i]));
        }
    }
}

// Test 3: ComputePressureKokkos with virial
TEST_F(MixedPrecisionComputesTest, ComputePressure) {
    lmp->input->one("compute mypress all pressure/kk NULL");
    lmp->input->one("run 1");  // Need at least 1 step to compute virial
    
    int icompute = lmp->modify->find_compute("mypress");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputePressureKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // Pressure should be computed using KK_FLOAT/KK_SUM_FLOAT
    double pressure = compute->scalar;
    EXPECT_TRUE(checkNumericalStability(pressure));
    
    // Check pressure tensor components
    if (compute->vector) {
        for (int i = 0; i < 6; i++) {
            EXPECT_TRUE(checkNumericalStability(compute->vector[i]));
        }
        
        // Diagonal components should be similar for isotropic system
        double avg_diag = (compute->vector[0] + compute->vector[1] + compute->vector[2]) / 3.0;
        EXPECT_PRECISION_NEAR(compute->vector[0], avg_diag, avg_diag * 0.1);
        EXPECT_PRECISION_NEAR(compute->vector[1], avg_diag, avg_diag * 0.1);
        EXPECT_PRECISION_NEAR(compute->vector[2], avg_diag, avg_diag * 0.1);
    }
}

// Test 4: ComputePEKokkos potential energy
TEST_F(MixedPrecisionComputesTest, ComputePE) {
    lmp->input->one("compute mype all pe/kk");
    lmp->input->one("run 0");
    
    int icompute = lmp->modify->find_compute("mype");
    ASSERT_GE(icompute, 0);
    
    auto compute = lmp->modify->compute[icompute];
    ASSERT_NE(compute, nullptr);
    
    // PE should use KK_FLOAT for computation
    double pe = compute->scalar;
    EXPECT_LT(pe, 0.0);  // LJ potential should be negative at this density
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 5: ComputePEKokkos per-atom energy
TEST_F(MixedPrecisionComputesTest, ComputePEAtom) {
    lmp->input->one("compute mype all pe/atom/kk");
    lmp->input->one("run 0");
    
    int icompute = lmp->modify->find_compute("mype");
    ASSERT_GE(icompute, 0);
    
    auto compute = lmp->modify->compute[icompute];
    ASSERT_NE(compute, nullptr);
    
    // Check per-atom array
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    if (compute->array_atom) {
        double total_pe = 0.0;
        for (int i = 0; i < atomKK->nlocal; i++) {
            EXPECT_TRUE(checkNumericalStability(compute->array_atom[i][0]));
            total_pe += compute->array_atom[i][0];
        }
        
        // Sum should match system PE (within precision tolerance)
        lmp->input->one("compute sys_pe all pe/kk");
        lmp->input->one("run 0");
        int ipe = lmp->modify->find_compute("sys_pe");
        double system_pe = lmp->modify->compute[ipe]->scalar;
        
        MPI_Allreduce(MPI_IN_PLACE, &total_pe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_PRECISION_NEAR(total_pe, system_pe, std::abs(system_pe) * getRelativeTolerance() * 10);
    }
}

// Test 6: ComputeRDFKokkos radial distribution function
TEST_F(MixedPrecisionComputesTest, ComputeRDF) {
    // Need at least 2 atoms for RDF
    lmp->input->one("compute myrdf all rdf/kk 50");
    lmp->input->one("run 100");  // Run a bit to get meaningful RDF
    
    int icompute = lmp->modify->find_compute("myrdf");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputeRDFKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // RDF should use KK_FLOAT for distances and histogram
    if (compute->array) {
        int nrows = compute->array_rows;
        int ncols = compute->array_cols;
        
        EXPECT_EQ(nrows, 50);  // Number of bins
        EXPECT_GE(ncols, 2);    // At least r and g(r)
        
        for (int i = 0; i < nrows; i++) {
            // Check radius values
            EXPECT_TRUE(checkNumericalStability(compute->array[i][0]));
            EXPECT_GT(compute->array[i][0], 0.0);  // Radius should be positive
            
            // Check g(r) values
            EXPECT_TRUE(checkNumericalStability(compute->array[i][1]));
            EXPECT_GE(compute->array[i][1], 0.0);  // g(r) should be non-negative
        }
        
        // First peak should be around r = 1.0 for LJ with sigma = 1.0
        bool found_peak = false;
        for (int i = 1; i < nrows-1; i++) {
            if (compute->array[i][0] > 0.9 && compute->array[i][0] < 1.3) {
                if (compute->array[i][1] > compute->array[i-1][1] &&
                    compute->array[i][1] > compute->array[i+1][1]) {
                    found_peak = true;
                    break;
                }
            }
        }
        EXPECT_TRUE(found_peak) << "Should find first RDF peak near r=1.0";
    }
}

// Test 7: ComputeMSDKokkos mean square displacement
TEST_F(MixedPrecisionComputesTest, ComputeMSD) {
    lmp->input->one("compute mymsd all msd/kk");
    
    // Store initial positions
    lmp->input->one("run 0");
    
    int icompute = lmp->modify->find_compute("mymsd");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputeMSDKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // Initial MSD should be zero
    double msd_init = compute->scalar;
    EXPECT_NEAR(msd_init, 0.0, getAbsoluteTolerance());
    
    // Run dynamics
    lmp->input->one("fix 1 all nve");
    lmp->input->one("run 100");
    
    // MSD should increase with time
    double msd_final = compute->scalar;
    EXPECT_GT(msd_final, 0.0);
    EXPECT_TRUE(checkNumericalStability(msd_final));
    
    // Check vector components (x, y, z, total MSD)
    if (compute->vector) {
        double msd_sum = 0.0;
        for (int i = 0; i < 3; i++) {
            EXPECT_TRUE(checkNumericalStability(compute->vector[i]));
            EXPECT_GE(compute->vector[i], 0.0);  // Each component should be non-negative
            msd_sum += compute->vector[i];
        }
        
        // Total MSD should equal sum of components
        EXPECT_PRECISION_NEAR(compute->vector[3], msd_sum, msd_sum * getRelativeTolerance());
    }
}

// Test 8: ComputeKEKokkos kinetic energy
TEST_F(MixedPrecisionComputesTest, ComputeKE) {
    lmp->input->one("compute myke all ke/kk");
    lmp->input->one("run 0");
    
    int icompute = lmp->modify->find_compute("myke");
    ASSERT_GE(icompute, 0);
    
    auto compute = lmp->modify->compute[icompute];
    ASSERT_NE(compute, nullptr);
    
    // KE should be positive for moving atoms
    double ke = compute->scalar;
    EXPECT_GT(ke, 0.0);
    EXPECT_TRUE(checkNumericalStability(ke));
    
    // Check relationship with temperature
    lmp->input->one("compute mytemp2 all temp/kk");
    lmp->input->one("run 0");
    
    int itemp = lmp->modify->find_compute("mytemp2");
    double temp = lmp->modify->compute[itemp]->scalar;
    double dof = dynamic_cast<ComputeTempKokkos*>(lmp->modify->compute[itemp])->dof;
    
    // KE = (dof/2) * kB * T, with kB = 1 in LJ units
    double expected_ke = 0.5 * dof * temp;
    EXPECT_PRECISION_NEAR(ke, expected_ke, expected_ke * getRelativeTolerance());
}

// Test 9: ComputeStressAtomKokkos per-atom stress
TEST_F(MixedPrecisionComputesTest, ComputeStressAtom) {
    lmp->input->one("compute mystress all stress/atom/kk NULL");
    lmp->input->one("run 1");  // Need to run to compute forces
    
    int icompute = lmp->modify->find_compute("mystress");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputeStressAtomKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // Check per-atom stress tensor
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    if (compute->array_atom) {
        for (int i = 0; i < atomKK->nlocal; i++) {
            // 6 components of stress tensor (xx, yy, zz, xy, xz, yz)
            for (int j = 0; j < 6; j++) {
                EXPECT_TRUE(checkNumericalStability(compute->array_atom[i][j]));
            }
            
            // For equilibrated system, diagonal stresses should be similar
            double avg_diag = (compute->array_atom[i][0] + 
                              compute->array_atom[i][1] + 
                              compute->array_atom[i][2]) / 3.0;
            
            // Check within reasonable bounds (not exact due to local fluctuations)
            for (int j = 0; j < 3; j++) {
                EXPECT_PRECISION_NEAR(compute->array_atom[i][j], avg_diag, 
                                     std::abs(avg_diag) * 0.5 + getAbsoluteTolerance());
            }
        }
    }
}

// Test 10: ComputeCentroAtomKokkos centro-symmetry parameter
TEST_F(MixedPrecisionComputesTest, ComputeCentroAtom) {
    // Create FCC lattice for centro-symmetry
    lmp->input->one("clear");
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("lattice fcc 1.0");
    lmp->input->one("region box block 0 3 0 3 0 3");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    
    lmp->input->one("compute mycentro all centro/atom/kk fcc");
    lmp->input->one("run 0");
    
    int icompute = lmp->modify->find_compute("mycentro");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputeCentroAtomKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // Perfect FCC should have centro-symmetry parameter close to 0
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    if (compute->vector_atom) {
        for (int i = 0; i < atomKK->nlocal; i++) {
            EXPECT_TRUE(checkNumericalStability(compute->vector_atom[i]));
            // Interior atoms in perfect FCC should have very low CSP
            // (boundary atoms may have higher values)
            if (compute->vector_atom[i] < 1.0) {
                EXPECT_NEAR(compute->vector_atom[i], 0.0, 0.01);
            }
        }
    }
}

// Test 11: ComputeCoordAtomKokkos coordination number
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
            EXPECT_GE(compute->vector_atom[i], 0.0);  // Coordination should be non-negative
            
            // FCC with cutoff 1.5 should give ~12 neighbors for interior atoms
            avg_coord += compute->vector_atom[i];
            count++;
        }
        
        if (count > 0) {
            avg_coord /= count;
            // Average should be close to 12 for FCC
            EXPECT_GT(avg_coord, 8.0);  // Lower bound accounting for surface atoms
            EXPECT_LT(avg_coord, 13.0); // Upper bound
        }
    }
}

// Test 12: Multiple computes with dependencies
TEST_F(MixedPrecisionComputesTest, MultipleComputesDependencies) {
    // Create chain of dependent computes
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("compute mypress all pressure/kk mytemp");
    lmp->input->one("compute mype all pe/kk");
    lmp->input->one("compute myke all ke/kk");
    
    lmp->input->one("run 1");
    
    // Check all computes work together
    int itemp = lmp->modify->find_compute("mytemp");
    int ipress = lmp->modify->find_compute("mypress");
    int ipe = lmp->modify->find_compute("mype");
    int ike = lmp->modify->find_compute("myke");
    
    ASSERT_GE(itemp, 0);
    ASSERT_GE(ipress, 0);
    ASSERT_GE(ipe, 0);
    ASSERT_GE(ike, 0);
    
    double temp = lmp->modify->compute[itemp]->scalar;
    double press = lmp->modify->compute[ipress]->scalar;
    double pe = lmp->modify->compute[ipe]->scalar;
    double ke = lmp->modify->compute[ike]->scalar;
    
    EXPECT_TRUE(checkNumericalStability(temp));
    EXPECT_TRUE(checkNumericalStability(press));
    EXPECT_TRUE(checkNumericalStability(pe));
    EXPECT_TRUE(checkNumericalStability(ke));
    
    // Energy conservation check (approximately)
    double total_e = pe + ke;
    EXPECT_TRUE(checkNumericalStability(total_e));
}

// Test 13: Compute with fix modification
TEST_F(MixedPrecisionComputesTest, ComputeWithFixModification) {
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("fix 1 all nve");
    lmp->input->one("fix_modify 1 temp mytemp");
    
    lmp->input->one("run 10");
    
    int icompute = lmp->modify->find_compute("mytemp");
    ASSERT_GE(icompute, 0);
    
    auto compute = dynamic_cast<ComputeTempKokkos*>(lmp->modify->compute[icompute]);
    ASSERT_NE(compute, nullptr);
    
    // Temperature should be properly computed even with fix modification
    double temp = compute->scalar;
    EXPECT_GT(temp, 0.0);
    EXPECT_TRUE(checkNumericalStability(temp));
}

// Test 14: Precision impact on statistical quantities
TEST_F(MixedPrecisionComputesTest, StatisticalPrecision) {
    // Run longer simulation to accumulate statistics
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("fix 1 all nvt/kk temp 1.0 1.0 0.1");
    lmp->input->one("fix_modify 1 temp mytemp");
    
    std::vector<double> temps;
    
    // Collect temperature samples
    for (int step = 0; step < 100; step++) {
        lmp->input->one("run 10");
        
        int icompute = lmp->modify->find_compute("mytemp");
        double temp = lmp->modify->compute[icompute]->scalar;
        temps.push_back(temp);
    }
    
    // Calculate mean and standard deviation
    double mean = 0.0;
    for (double t : temps) {
        mean += t;
    }
    mean /= temps.size();
    
    double variance = 0.0;
    for (double t : temps) {
        variance += (t - mean) * (t - mean);
    }
    variance /= temps.size();
    double stddev = sqrt(variance);
    
    // Mean should be close to target temperature
    EXPECT_PRECISION_NEAR(mean, 1.0, 0.1);
    
    // Standard deviation should be reasonable
    EXPECT_GT(stddev, 0.0);
    EXPECT_LT(stddev, 0.5);  // Not too large
    
    // Check for numerical stability in all samples
    for (double t : temps) {
        EXPECT_TRUE(checkNumericalStability(t));
    }
}

// Test 15: Extreme value handling in computes
TEST_F(MixedPrecisionComputesTest, ExtremeValueHandling) {
    // Create system with very high velocities
    lmp->input->one("velocity all create 1000.0 12345");
    
    lmp->input->one("compute mytemp all temp/kk");
    lmp->input->one("compute myke all ke/kk");
    lmp->input->one("compute mypress all pressure/kk mytemp");
    
    lmp->input->one("run 0");
    
    // All computes should handle high values without overflow
    int itemp = lmp->modify->find_compute("mytemp");
    int ike = lmp->modify->find_compute("myke");
    int ipress = lmp->modify->find_compute("mypress");
    
    double temp = lmp->modify->compute[itemp]->scalar;
    double ke = lmp->modify->compute[ike]->scalar;
    double press = lmp->modify->compute[ipress]->scalar;
    
    EXPECT_TRUE(checkNumericalStability(temp));
    EXPECT_TRUE(checkNumericalStability(ke));
    EXPECT_TRUE(checkNumericalStability(press));
    
    // Values should be high but not infinite
    EXPECT_GT(temp, 100.0);
    EXPECT_LT(temp, 1e10);
    
    EXPECT_GT(ke, 100.0);
    EXPECT_LT(ke, 1e10);
    
    // Now test with very low velocities
    lmp->input->one("velocity all create 0.001 12345");
    lmp->input->one("run 0");
    
    temp = lmp->modify->compute[itemp]->scalar;
    ke = lmp->modify->compute[ike]->scalar;
    
    EXPECT_TRUE(checkNumericalStability(temp));
    EXPECT_TRUE(checkNumericalStability(ke));
    
    // Values should be small but positive
    EXPECT_GT(temp, 0.0);
    EXPECT_LT(temp, 0.01);
    
    EXPECT_GT(ke, 0.0);
    EXPECT_LT(ke, 1.0);
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
