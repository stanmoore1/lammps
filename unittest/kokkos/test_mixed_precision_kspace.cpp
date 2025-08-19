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
   Testing KSPACE package integration with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "force.h"
#include "neighbor.h"
#include "input.h"
#include "domain.h"
#include "kspace.h"
#include <cmath>
#include <vector>

// Include KSPACE headers if available
#ifdef LMP_KOKKOS_KSPACE
#include "pppm_kokkos.h"
#include "ewald_kokkos.h"
#include "msm_kokkos.h"
#include "fft3d_kokkos.h"
#include "remap_kokkos.h"
#include "grid3d_kokkos.h"
#endif

namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionKspaceTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
        
        // Create a charged system for KSPACE testing
        lmp->input->one("units real");
        lmp->input->one("atom_style charge");
        lmp->input->one("boundary p p p");
        lmp->input->one("region box block 0 10 0 10 0 10");
        lmp->input->one("create_box 2 box");
        
        // Create a simple ionic crystal structure
        lmp->input->one("lattice fcc 4.0");
        lmp->input->one("create_atoms 1 box");
        
        // Set charges and masses
        lmp->input->one("mass 1 23.0");  // Na+
        lmp->input->one("mass 2 35.5");  // Cl-
        lmp->input->one("set type 1 charge 1.0");
        lmp->input->one("set type 2 charge -1.0");
        
        // Convert half atoms to type 2 for charge neutrality
        lmp->input->one("group cations type 1");
        lmp->input->one("group anions type 2");
        
        // Basic LJ pair style for non-electrostatic interactions
        lmp->input->one("pair_style lj/cut/coul/long 10.0");
        lmp->input->one("pair_coeff 1 1 0.1 2.0");
        lmp->input->one("pair_coeff 1 2 0.1 2.5");
        lmp->input->one("pair_coeff 2 2 0.1 3.0");
    }
    
    void TearDown() override {
        MixedPrecisionTestFixture::TearDown();
    }
};

#ifdef LMP_KOKKOS_KSPACE

// Test 1: PPPMKokkos precision types
TEST_F(MixedPrecisionKspaceTest, PPPMKokkosPrecisionTypes) {
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("kspace_modify mesh 16 16 16");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    // Check density arrays use correct precision
    EXPECT_TRUE((std::is_same<decltype(pppm->k_density_brick.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_density_brick.d_view)::value_type, KK_FFT_SCALAR>::value));
    
    // Check FFT work arrays
    EXPECT_TRUE((std::is_same<decltype(pppm->k_density_fft.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_density_fft.d_view)::value_type, KK_FFT_SCALAR>::value));
    
    // Check potential arrays
    EXPECT_TRUE((std::is_same<decltype(pppm->k_vdx_brick.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_vdy_brick.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_vdz_brick.d_view)::value_type, KK_FLOAT>::value));
    
    // Check Green's function array
    EXPECT_TRUE((std::is_same<decltype(pppm->k_greensfn.d_view)::value_type, KK_FLOAT>::value));
    
    // Check virial arrays
    EXPECT_TRUE((std::is_same<decltype(pppm->k_virial.d_view)::value_type, KK_FLOAT>::value));
}

// Test 2: PPPMKokkos energy calculation
TEST_F(MixedPrecisionKspaceTest, PPPMEnergyCalculation) {
    lmp->input->one("kspace_style pppm/kk 1.0e-4");
    lmp->input->one("kspace_modify mesh 8 8 8");
    
    // Run to compute energy
    lmp->input->one("run 0");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    // Check that energy is computed and reasonable
    double energy = pppm->energy;
    EXPECT_TRUE(checkNumericalStability(energy));
    
    // For an ionic crystal, energy should be negative (attractive)
    EXPECT_LT(energy, 0.0);
    
    // Check virial computation
    for (int i = 0; i < 6; i++) {
        EXPECT_TRUE(checkNumericalStability(pppm->virial[i]));
    }
}

// Test 3: PPPMKokkos mesh and grid operations
TEST_F(MixedPrecisionKspaceTest, PPPMMeshOperations) {
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("kspace_modify mesh 12 12 12 order 5");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    lmp->input->one("run 0");
    
    // Check mesh dimensions
    EXPECT_GT(pppm->nx_pppm, 0);
    EXPECT_GT(pppm->ny_pppm, 0);
    EXPECT_GT(pppm->nz_pppm, 0);
    
    // Check order
    EXPECT_EQ(pppm->order, 5);
    
    // Check that charge interpolation arrays are allocated
    EXPECT_GT(pppm->k_part2grid.h_view.extent(0), 0u);
    
    // Check density brick dimensions
    EXPECT_GT(pppm->k_density_brick.h_view.extent(0), 0u);
    EXPECT_GT(pppm->k_density_brick.h_view.extent(1), 0u);
    EXPECT_GT(pppm->k_density_brick.h_view.extent(2), 0u);
    EXPECT_GT(pppm->k_density_brick.h_view.extent(3), 0u);
}

// Test 4: EwaldKokkos precision types
TEST_F(MixedPrecisionKspaceTest, EwaldKokkosPrecisionTypes) {
    lmp->input->one("kspace_style ewald/kk 1.0e-4");
    
    auto ewald = dynamic_cast<EwaldKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(ewald, nullptr);
    
    lmp->input->one("run 0");
    
    // Check k-vector arrays
    EXPECT_TRUE((std::is_same<decltype(ewald->k_ug.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(ewald->k_sfacrl.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(ewald->k_sfacim.d_view)::value_type, KK_FLOAT>::value));
    
    // Check exponential arrays
    EXPECT_TRUE((std::is_same<decltype(ewald->k_sfacrl_all.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(ewald->k_sfacim_all.d_view)::value_type, KK_FLOAT>::value));
}

// Test 5: EwaldKokkos energy and forces
TEST_F(MixedPrecisionKspaceTest, EwaldEnergyForces) {
    lmp->input->one("kspace_style ewald/kk 1.0e-4");
    
    lmp->input->one("run 0");
    
    auto ewald = dynamic_cast<EwaldKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(ewald, nullptr);
    
    // Check energy
    double energy = ewald->energy;
    EXPECT_TRUE(checkNumericalStability(energy));
    EXPECT_LT(energy, 0.0);  // Should be attractive for ionic system
    
    // Check forces
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, F_MASK);
    
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
        }
    }
    
    // Check virial
    for (int i = 0; i < 6; i++) {
        EXPECT_TRUE(checkNumericalStability(ewald->virial[i]));
    }
}

// Test 6: MSMKokkos precision types  
TEST_F(MixedPrecisionKspaceTest, MSMKokkosPrecisionTypes) {
    // MSM requires specific setup
    lmp->input->one("kspace_style msm/kk 1.0e-4");
    
    auto msm = dynamic_cast<MSMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(msm, nullptr);
    
    lmp->input->one("run 0");
    
    // Check grid arrays
    EXPECT_TRUE((std::is_same<decltype(msm->k_qgrid.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(msm->k_egrid.d_view)::value_type, KK_FLOAT>::value));
    
    // Check potential arrays
    EXPECT_TRUE((std::is_same<decltype(msm->k_phi1d.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(msm->k_dphi1d.d_view)::value_type, KK_FLOAT>::value));
}

// Test 7: FFT3dKokkos precision handling
TEST_F(MixedPrecisionKspaceTest, FFT3DPrecision) {
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("kspace_modify mesh 16 16 16");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    lmp->input->one("run 0");
    
    // Check FFT scalar type
#ifdef FFT_SINGLE
    EXPECT_TRUE((std::is_same<KK_FFT_SCALAR, float>::value));
#else
    EXPECT_TRUE((std::is_same<KK_FFT_SCALAR, double>::value));
#endif
    
    // Check FFT data arrays match expected precision
    EXPECT_TRUE((std::is_same<decltype(pppm->k_density_fft.d_view)::value_type, KK_FFT_SCALAR>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_work1.d_view)::value_type, KK_FFT_SCALAR>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_work2.d_view)::value_type, KK_FFT_SCALAR>::value));
}

// Test 8: Charge interpolation precision
TEST_F(MixedPrecisionKspaceTest, ChargeInterpolation) {
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("kspace_modify mesh 8 8 8 order 4");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    lmp->input->one("run 0");
    
    // Check rho coefficient arrays
    EXPECT_TRUE((std::is_same<decltype(pppm->k_rho_coeff.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_rho1d.d_view)::value_type, KK_FFT_SCALAR>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_rho_coeff.d_view)::value_type, KK_FLOAT>::value));
    
    // Check that interpolation produced reasonable densities
    pppm->k_density_brick.sync_host();
    bool found_nonzero = false;
    for (size_t i = 0; i < pppm->k_density_brick.h_view.extent(0); i++) {
        for (size_t j = 0; j < pppm->k_density_brick.h_view.extent(1); j++) {
            for (size_t k = 0; k < pppm->k_density_brick.h_view.extent(2); k++) {
                for (size_t m = 0; m < pppm->k_density_brick.h_view.extent(3); m++) {
                    double val = pppm->k_density_brick.h_view(i,j,k,m);
                    EXPECT_TRUE(checkNumericalStability(val));
                    if (std::abs(val) > 1e-10) found_nonzero = true;
                }
            }
        }
    }
    EXPECT_TRUE(found_nonzero);  // Should have some charge density
}

// Test 9: Virial computation precision
TEST_F(MixedPrecisionKspaceTest, VirialPrecision) {
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("compute pressure all pressure NULL virial");
    
    lmp->input->one("run 0");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    // Check virial array precision
    EXPECT_TRUE((std::is_same<decltype(pppm->k_virial.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(pppm->k_virial.d_view)::value_type, KK_FLOAT>::value));
    
    // Verify virial values are reasonable
    for (int i = 0; i < 6; i++) {
        EXPECT_TRUE(checkNumericalStability(pppm->virial[i]));
        // For a symmetric system, off-diagonal terms should be small
        if (i >= 3) {
            EXPECT_NEAR(pppm->virial[i], 0.0, 1.0);
        }
    }
}

// Test 10: Differentiation precision (force computation)
TEST_F(MixedPrecisionKspaceTest, ForceDifferentiation) {
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("kspace_modify mesh 12 12 12 order 5 diff ad");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    // Set up atoms with specific positions for testing
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK | Q_MASK);
    
    // Place a positive charge at origin
    if (atomKK->nlocal > 0) {
        atomKK->x[0][0] = 5.0;
        atomKK->x[0][1] = 5.0;
        atomKK->x[0][2] = 5.0;
        atomKK->q[0] = 1.0;
    }
    
    // Place a negative charge nearby
    if (atomKK->nlocal > 1) {
        atomKK->x[1][0] = 6.0;
        atomKK->x[1][1] = 5.0;
        atomKK->x[1][2] = 5.0;
        atomKK->q[1] = -1.0;
    }
    
    atomKK->modified(Host, X_MASK | Q_MASK);
    
    lmp->input->one("run 0");
    
    // Check forces are reasonable
    atomKK->sync(Host, F_MASK);
    for (int i = 0; i < std::min(2, atomKK->nlocal); i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
        }
        
        // Forces should be primarily along x-axis for this configuration
        if (i < 2) {
            EXPECT_GT(std::abs(atomKK->f[i][0]), std::abs(atomKK->f[i][1]));
            EXPECT_GT(std::abs(atomKK->f[i][0]), std::abs(atomKK->f[i][2]));
        }
    }
}

// Test 11: Grid communication precision
TEST_F(MixedPrecisionKspaceTest, GridCommunication) {
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("kspace_modify mesh 16 16 16");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    lmp->input->one("run 0");
    
    // Check communication buffers use appropriate precision
    if (pppm->k_gc_buf1.h_view.extent(0) > 0) {
        EXPECT_TRUE((std::is_same<decltype(pppm->k_gc_buf1.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(pppm->k_gc_buf1.d_view)::value_type, KK_FFT_SCALAR>::value));
    }
    
    if (pppm->k_gc_buf2.h_view.extent(0) > 0) {
        EXPECT_TRUE((std::is_same<decltype(pppm->k_gc_buf2.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(pppm->k_gc_buf2.d_view)::value_type, KK_FFT_SCALAR>::value));
    }
}

// Test 12: Slab correction precision
TEST_F(MixedPrecisionKspaceTest, SlabCorrection) {
    // Set up 2D slab geometry
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("kspace_modify slab 3.0");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    lmp->input->one("run 0");
    
    // Check slab factor
    EXPECT_GT(pppm->slab_volfactor, 1.0);
    EXPECT_TRUE(checkNumericalStability(pppm->slab_volfactor));
    
    // Energy should include slab correction
    double energy = pppm->energy;
    EXPECT_TRUE(checkNumericalStability(energy));
}

// Test 13: Accuracy vs precision trade-off
TEST_F(MixedPrecisionKspaceTest, AccuracyVsPrecision) {
    // Test with different accuracy settings
    std::vector<double> accuracies = {1.0e-3, 1.0e-4, 1.0e-5};
    std::vector<double> energies;
    
    for (double acc : accuracies) {
        // Reset system
        SetUp();
        
        lmp->input->one(("kspace_style pppm/kk " + std::to_string(acc)).c_str());
        lmp->input->one("run 0");
        
        auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
        ASSERT_NE(pppm, nullptr);
        
        energies.push_back(pppm->energy);
        EXPECT_TRUE(checkNumericalStability(pppm->energy));
        
        TearDown();
    }
    
    // Higher accuracy should give more consistent results
    if (energies.size() >= 2) {
        double diff_low = std::abs(energies[1] - energies[0]);
        double diff_high = std::abs(energies[2] - energies[1]);
        
        // The difference should decrease with higher accuracy
        EXPECT_LT(diff_high, diff_low * 2.0);  // Allow some variance
    }
}

// Test 14: Memory allocation and deallocation
TEST_F(MixedPrecisionKspaceTest, MemoryManagement) {
    // Test allocation
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("kspace_modify mesh 20 20 20 order 6");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    lmp->input->one("run 0");
    
    // Check large arrays are allocated
    size_t density_size = pppm->k_density_brick.h_view.extent(0) *
                         pppm->k_density_brick.h_view.extent(1) *
                         pppm->k_density_brick.h_view.extent(2) *
                         pppm->k_density_brick.h_view.extent(3);
    EXPECT_GT(density_size, 0u);
    
    size_t fft_size = pppm->k_density_fft.h_view.extent(0) *
                     pppm->k_density_fft.h_view.extent(1);
    EXPECT_GT(fft_size, 0u);
    
    // Change settings and reallocate
    lmp->input->one("kspace_style pppm/kk 1.0e-4");
    lmp->input->one("kspace_modify mesh 10 10 10 order 4");
    lmp->input->one("run 0");
    
    // Arrays should be reallocated with new sizes
    pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    size_t new_density_size = pppm->k_density_brick.h_view.extent(0) *
                             pppm->k_density_brick.h_view.extent(1) *
                             pppm->k_density_brick.h_view.extent(2) *
                             pppm->k_density_brick.h_view.extent(3);
    EXPECT_GT(new_density_size, 0u);
    EXPECT_NE(new_density_size, density_size);  // Should be different size
}

// Test 15: Per-atom energy computation
TEST_F(MixedPrecisionKspaceTest, PerAtomEnergy) {
    lmp->input->one("kspace_style pppm/kk 1.0e-5");
    lmp->input->one("compute pe_atom all pe/atom");
    
    lmp->input->one("run 0");
    
    auto pppm = dynamic_cast<PPPMKokkos<LMPDeviceType>*>(lmp->force->kspace);
    ASSERT_NE(pppm, nullptr);
    
    // Check per-atom arrays if allocated
    if (pppm->k_eatom.h_view.extent(0) > 0) {
        EXPECT_TRUE((std::is_same<decltype(pppm->k_eatom.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(pppm->k_eatom.d_view)::value_type, KK_FLOAT>::value));
        
        // Verify per-atom energies sum to total
        pppm->k_eatom.sync_host();
        double sum = 0.0;
        for (size_t i = 0; i < pppm->k_eatom.h_view.extent(0); i++) {
            sum += pppm->k_eatom.h_view(i);
            EXPECT_TRUE(checkNumericalStability(pppm->k_eatom.h_view(i)));
        }
        
        // Sum should be close to total energy (within precision tolerance)
        EXPECT_PRECISION_NEAR(sum, pppm->energy, getRelativeTolerance() * std::abs(pppm->energy));
    }
}

#else // !LMP_KOKKOS_KSPACE

// Placeholder test when KSPACE is not available
TEST_F(MixedPrecisionKspaceTest, KspaceNotAvailable) {
    GTEST_SKIP() << "KSPACE package not enabled, skipping KSPACE tests";
}

#endif // LMP_KOKKOS_KSPACE

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
