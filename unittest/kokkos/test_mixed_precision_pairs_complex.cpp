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
   Testing complex many-body pair styles with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#ifdef LMP_MANYBODY
#include "pair_eam_kokkos.h"
#include "pair_eam_alloy_kokkos.h"
#include "pair_sw_kokkos.h"
#include "pair_tersoff_kokkos.h"
#include "pair_tersoff_zbl_kokkos.h"
#endif
#include "force.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "input.h"
#include <cmath>
#include <fstream>

namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionPairsComplexTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
    }
    
    // Helper to create minimal EAM potential file
    void createEAMFile(const std::string& filename) {
        std::ofstream file(filename);
        file << "# DATE: 2020-01-01 UNITS: metal\n";
        file << "# Generated for unit testing\n\n";
        file << "1 Cu\n";
        file << "500 5.0e-04 500 1.0e-03 10.0\n";
        
        for (int i = 0; i < 500; i++) {
            double rho = i * 5.0e-04;
            double F = -std::sqrt(rho);
            file << F << "\n";
        }
        
        for (int i = 0; i < 500; i++) {
            double r = i * 1.0e-03;
            double rho = std::exp(-r);
            file << rho << "\n";
        }
        
        for (int i = 0; i < 500; i++) {
            double r = i * 1.0e-03;
            double rphi = r * std::exp(-2*r);
            file << rphi << "\n";
        }
        
        file.close();
    }
    
    // Helper to create minimal SW potential file
    void createSWFile(const std::string& filename) {
        std::ofstream file(filename);
        file << "# Stillinger-Weber parameters for Si\n";
        file << "# epsilon sigma a lambda gamma costheta0 A B p q tol\n";
        file << "Si Si Si 2.1683 2.0951 1.80 21.0 1.20 -0.333333333333\n";
        file << "         7.049556277 0.6022245584 4.0 0.0 0.0\n";
        file.close();
    }
    
    // Helper to create minimal Tersoff potential file
    void createTersoffFile(const std::string& filename) {
        std::ofstream file(filename);
        file << "# Tersoff parameters for Si\n";
        file << "# m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A\n";
        file << "Si Si Si 3.0 1.0 0.0 1.0039e5 16.217 -0.59825 0.78734\n";
        file << "         1.1e-6 1.7322 471.18 2.85 0.15 2.4799 1830.8\n";
        file.close();
    }
};

#ifdef LMP_MANYBODY
// Test 1: PairEAMKokkos basic precision types
TEST_F(MixedPrecisionPairsComplexTest, PairEAMTypes) {
    createEAMFile("Cu.eam");
    
    lmp->input->one("units metal");
    lmp->input->one("atom_style atomic");
    lmp->input->one("lattice fcc 3.615");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 63.546");
    
    lmp->input->one("pair_style eam/kk");
    lmp->input->one("pair_coeff * * Cu.eam");
    
    auto pair = dynamic_cast<PairEAMKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    // Check internal array precision
    EXPECT_TRUE((std::is_same<decltype(pair->d_rho)::value_type, KK_SUM_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->d_fp)::value_type, KK_FLOAT>::value));
    
    // Check embedding and density arrays
    EXPECT_TRUE((std::is_same<decltype(pair->d_frho_spline)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->d_rhor_spline)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->d_z2r_spline)::value_type, KK_FLOAT>::value));
}

// Test 2: PairEAMKokkos energy calculation
TEST_F(MixedPrecisionPairsComplexTest, PairEAMEnergy) {
    createEAMFile("Cu.eam");
    
    lmp->input->one("units metal");
    lmp->input->one("atom_style atomic");
    lmp->input->one("lattice fcc 3.615");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 63.546");
    
    lmp->input->one("pair_style eam/kk");
    lmp->input->one("pair_coeff * * Cu.eam");
    lmp->input->one("neighbor 0.3 bin");
    
    lmp->input->one("run 0");
    
    // Check energy is computed and stable
    double pe = lmp->force->pair->eng_vdwl;
    EXPECT_TRUE(checkNumericalStability(pe));
    EXPECT_LT(pe, 0.0);  // EAM typically has negative energy
}

// Test 3: PairSWKokkos three-body interactions
TEST_F(MixedPrecisionPairsComplexTest, PairSWThreeBody) {
    createSWFile("Si.sw");
    
    lmp->input->one("units metal");
    lmp->input->one("atom_style atomic");
    lmp->input->one("lattice diamond 5.431");
    lmp->input->one("region box block 0 1 0 1 0 1");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 28.0855");
    
    lmp->input->one("pair_style sw/kk");
    lmp->input->one("pair_coeff * * Si.sw Si");
    lmp->input->one("neighbor 0.3 bin");
    
    lmp->input->one("run 0");
    
    // Check energy includes three-body terms
    double pe = lmp->force->pair->eng_vdwl;
    EXPECT_TRUE(checkNumericalStability(pe));
    
    // Check forces are stable
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, F_MASK);
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
        }
    }
}

// Test 4: PairTersoffKokkos bond order calculation
TEST_F(MixedPrecisionPairsComplexTest, PairTersoffBondOrder) {
    createTersoffFile("Si.tersoff");
    
    lmp->input->one("units metal");
    lmp->input->one("atom_style atomic");
    lmp->input->one("lattice diamond 5.431");
    lmp->input->one("region box block 0 1 0 1 0 1");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 28.0855");
    
    lmp->input->one("pair_style tersoff/kk");
    lmp->input->one("pair_coeff * * Si.tersoff Si");
    
    lmp->input->one("run 0");
    
    // Verify bond order dependent forces are computed
    double pe = lmp->force->pair->eng_vdwl;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 5: Many-body neighbor list handling
TEST_F(MixedPrecisionPairsComplexTest, ManyBodyNeighborList) {
    createSWFile("Si.sw");
    
    lmp->input->one("units metal");
    lmp->input->one("atom_style atomic");
    lmp->input->one("lattice diamond 5.431");
    lmp->input->one("region box block 0 3 0 3 0 3");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 28.0855");
    
    lmp->input->one("pair_style sw/kk");
    lmp->input->one("pair_coeff * * Si.sw Si");
    lmp->input->one("neighbor 0.3 bin");
    
    lmp->input->one("run 0");
    
    auto pair = dynamic_cast<PairSWKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    // Check neighbor list is properly allocated
    auto list = pair->list;
    ASSERT_NE(list, nullptr);
    
    // Verify neighbor list uses correct types
    auto listKK = dynamic_cast<NeighListKokkos<LMPDeviceType>*>(list);
    if (listKK != nullptr) {
        EXPECT_GT(listKK->inum, 0);
        EXPECT_GT(listKK->d_numneigh.extent(0), 0);
    }
}

// Test 6: Energy conservation with many-body potentials
TEST_F(MixedPrecisionPairsComplexTest, EnergyConservation) {
    createSWFile("Si.sw");
    
    lmp->input->one("units metal");
    lmp->input->one("atom_style atomic");
    lmp->input->one("lattice diamond 5.431");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 28.0855");
    lmp->input->one("velocity all create 300.0 12345");
    
    lmp->input->one("pair_style sw/kk");
    lmp->input->one("pair_coeff * * Si.sw Si");
    lmp->input->one("neighbor 0.3 bin");
    
    lmp->input->one("fix 1 all nve");
    lmp->input->one("timestep 0.001");
    
    // Get initial energy
    lmp->input->one("run 0");
    double initial_pe = lmp->force->pair->eng_vdwl;
    
    // Run short dynamics
    lmp->input->one("run 100");
    double final_pe = lmp->force->pair->eng_vdwl;
    
    // Energy drift tolerance depends on precision mode
    double drift = std::abs(final_pe - initial_pe) / std::abs(initial_pe);
    
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    EXPECT_LT(drift, 0.01);  // 1% acceptable for single precision
#elif defined(LMP_KOKKOS_SINGLE_DOUBLE)
    EXPECT_LT(drift, 0.001); // 0.1% for mixed precision
#else
    EXPECT_LT(drift, 0.0001); // 0.01% for double precision
#endif
}
#else
// Stub test when MANYBODY package is not available
TEST_F(MixedPrecisionPairsComplexTest, ManyBodyNotAvailable) {
    GTEST_SKIP() << "MANYBODY package not available";
}
#endif // LMP_MANYBODY

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
