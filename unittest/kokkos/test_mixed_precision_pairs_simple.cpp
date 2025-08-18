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
   Testing simple pair styles with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "pair.h"     // for Pair and its members (e.g., eng_vdwl)
#include "compute.h"  // for Compute and its members (e.g., scalar)
#include "pair_lj_cut_kokkos.h"
#include "pair_lj_cut_coul_cut_kokkos.h"
#include "pair_morse_kokkos.h"
#include "pair_buck_kokkos.h"
#include "pair_yukawa_kokkos.h"
#include "force.h"
#include "neighbor.h"
#include "input.h"
#include <cmath>


namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionPairsSimpleTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
        
        // Create a simple LJ system
        lmp->input->one("units lj");
        lmp->input->one("atom_style atomic");
        lmp->input->one("lattice fcc 0.8442");
        lmp->input->one("region box block 0 2 0 2 0 2");
        lmp->input->one("create_box 1 box");
        lmp->input->one("create_atoms 1 box");
        lmp->input->one("mass 1 1.0");
        lmp->input->one("neighbor 0.3 bin");
        lmp->input->one("neigh_modify delay 0 every 1 check yes");
    }
};

// Test 1: PairLJCutKokkos precision types
TEST_F(MixedPrecisionPairsSimpleTest, PairLJCutTypes) {
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    auto pair = dynamic_cast<PairLJCutKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    // Check coefficient arrays use correct precision (dual views)
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).lj1), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).lj2), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).lj3), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).lj4), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(pair->k_cutsq.d_view)::value_type, KK_FLOAT>::value));
    
    // Check per-atom arrays
    EXPECT_TRUE((std::is_same<typename decltype(pair->k_eatom.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<typename decltype(pair->k_eatom.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(pair->k_vatom.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<typename decltype(pair->k_vatom.d_view)::value_type, KK_FLOAT>::value));
}

// Test 2: PairLJCutKokkos energy calculation
TEST_F(MixedPrecisionPairsSimpleTest, PairLJCutEnergy) {
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    lmp->input->one("run 0");
    
    // Check energy is computed and reasonable
    double pe = lmp->force->pair->eng_vdwl;
    EXPECT_TRUE(checkNumericalStability(pe));
    EXPECT_LT(pe, 0.0);  // LJ should give negative energy for this configuration
    
    // Check forces
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, F_MASK);
    
    double total_force = 0.0;
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
            total_force += atomKK->f[i][j];
        }
    }
    
    // Total force should be near zero (balanced system)
    EXPECT_NEAR(total_force, 0.0, getAbsoluteTolerance() * 100);
}

// Test 3: PairLJCutCoulCutKokkos with Coulomb
TEST_F(MixedPrecisionPairsSimpleTest, PairLJCutCoulCut) {
    // Need charged atoms for Coulomb
    lmp->input->one("clear");
    lmp->input->one("units real");
    lmp->input->one("atom_style charge");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 2 box");
    lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
    lmp->input->one("create_atoms 2 single 7.0 5.0 5.0");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("mass 2 1.0");
    lmp->input->one("set atom 1 charge 1.0");
    lmp->input->one("set atom 2 charge -1.0");
    
    lmp->input->one("pair_style lj/cut/coul/cut/kk 10.0");
    lmp->input->one("pair_coeff 1 1 0.1 3.0");
    lmp->input->one("pair_coeff 2 2 0.1 3.0");
    lmp->input->one("pair_coeff 1 2 0.1 3.0");
    
    auto pair = dynamic_cast<PairLJCutCoulCutKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    lmp->input->one("run 0");
    
    // Check both VDW and Coulomb energies
    double eng_vdwl = lmp->force->pair->eng_vdwl;
    double eng_coul = lmp->force->pair->eng_coul;
    
    EXPECT_TRUE(checkNumericalStability(eng_vdwl));
    EXPECT_TRUE(checkNumericalStability(eng_coul));
    EXPECT_LT(eng_coul, 0.0);  // Opposite charges should attract
}

// Test 4: PairMorseKokkos exponential potential
TEST_F(MixedPrecisionPairsSimpleTest, PairMorse) {
    lmp->input->one("pair_style morse/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 2.0 1.2 2.5");  // D0 alpha r0 cutoff
    
    auto pair = dynamic_cast<PairMorseKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    // Check Morse-specific arrays (dual views)
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).d0), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).alpha), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).r0), KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    // Morse potential should be stable
    double pe = lmp->force->pair->eng_vdwl;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 5: PairBuckKokkos Buckingham potential
TEST_F(MixedPrecisionPairsSimpleTest, PairBuck) {
    lmp->input->one("pair_style buck/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1000.0 0.36 32.0 2.5");  // A rho C cutoff
    
    auto pair = dynamic_cast<PairBuckKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    // Check Buck-specific arrays (dual views)
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).cutsq), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).a), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).c), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).rhoinv), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).buck1), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).buck2), KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).offset), KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    double pe = lmp->force->pair->eng_vdwl;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 6: PairYukawaKokkos screened potential
TEST_F(MixedPrecisionPairsSimpleTest, PairYukawa) {
    lmp->input->one("pair_style yukawa/kk 2.0 2.5");  // kappa cutoff
    lmp->input->one("pair_coeff 1 1 1.0 2.5");  // A cutoff
    
    auto pair = dynamic_cast<PairYukawaKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    // Check Yukawa-specific arrays (dual views)
    EXPECT_TRUE((std::is_same<decltype(pair->k_params.d_view(0,0).a), KK_FLOAT>::value));
    // Note: kappa is a protected member in base class, cannot directly access
    // EXPECT_EQ(pair->kappa, 2.0);  // kappa is stored as a scalar
    
    lmp->input->one("run 0");
    
    double pe = lmp->force->pair->eng_vdwl;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 7: Neighbor list precision
TEST_F(MixedPrecisionPairsSimpleTest, NeighborListPrecision) {
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    lmp->input->one("run 0");
    
    auto pair = dynamic_cast<PairLJCutKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    // Check neighbor list types
    if (pair->list) {
        // Neighbor list exists and is populated
        EXPECT_GT(pair->list->inum, 0);
        EXPECT_NE(pair->list->ilist, nullptr);
        EXPECT_NE(pair->list->numneigh, nullptr);
        
        // The Kokkos neighbor list has its own structure
        // but we can't directly access k_ilist etc. from base NeighList
    }
}

// Test 8: Force summation precision
TEST_F(MixedPrecisionPairsSimpleTest, ForceSummation) {
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    // Create unbalanced forces
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK);
    
    // Move one atom slightly to create forces
    if (atomKK->nlocal > 0) {
        atomKK->x[0][0] += 0.1;
        atomKK->modified(Host, X_MASK);
    }
    
    lmp->input->one("run 0");
    
    // Forces should sum to near zero (Newton's third law)
    atomKK->sync(Host, F_MASK);
    double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;
    
    for (int i = 0; i < atomKK->nlocal; i++) {
        fx_sum += atomKK->f[i][0];
        fy_sum += atomKK->f[i][1];
        fz_sum += atomKK->f[i][2];
        
        // Check individual forces are reasonable
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
            EXPECT_LT(std::abs(atomKK->f[i][j]), 1000.0);
        }
    }
    
    // Total momentum should be conserved (forces sum to zero)
    // Allow larger tolerance for single precision
    double tol = getCurrentPrecisionMode() == SINGLE_SINGLE ? 1e-3 : 1e-6;
    EXPECT_NEAR(fx_sum, 0.0, tol);
    EXPECT_NEAR(fy_sum, 0.0, tol);
    EXPECT_NEAR(fz_sum, 0.0, tol);
}

// Test 9: Cutoff distance precision
TEST_F(MixedPrecisionPairsSimpleTest, CutoffPrecision) {
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    auto pair = dynamic_cast<PairLJCutKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    // Check cutoff squared values
    pair->k_cutsq.sync_host();
    double cutsq_host = pair->k_cutsq.h_view(1, 1);
    double cutsq_expected = 2.5 * 2.5;
    
    EXPECT_PRECISION_NEAR(cutsq_host, cutsq_expected, 
                         getRelativeTolerance() * cutsq_expected);
}

// Test 10: Virial/pressure calculation
TEST_F(MixedPrecisionPairsSimpleTest, VirialCalculation) {
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    lmp->input->one("compute p all pressure NULL pair");
    
    lmp->input->one("run 0");
    
    // Check virial components
    for (int i = 0; i < 6; i++) {
        EXPECT_TRUE(checkNumericalStability(lmp->force->pair->virial[i]));
    }
    
    // For a cubic system, diagonal components should be similar
    double avg_diag = (lmp->force->pair->virial[0] + 
                       lmp->force->pair->virial[1] + 
                       lmp->force->pair->virial[2]) / 3.0;
    
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(lmp->force->pair->virial[i], avg_diag, 
                   std::abs(avg_diag) * 0.1);  // Within 10%
    }
}

// Test 11: Mixing rules precision
TEST_F(MixedPrecisionPairsSimpleTest, MixingRules) {
    // Create system with 2 atom types
    lmp->input->one("clear");
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 2 0 2 0 2");
    lmp->input->one("create_box 2 box");
    lmp->input->one("create_atoms 1 single 1.0 1.0 1.0");
    lmp->input->one("create_atoms 2 single 1.5 1.0 1.0");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("mass 2 2.0");
    
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    lmp->input->one("pair_coeff 2 2 2.0 1.2");
    lmp->input->one("pair_modify mix arithmetic");
    
    auto pair = dynamic_cast<PairLJCutKokkos<LMPDeviceType>*>(lmp->force->pair);
    ASSERT_NE(pair, nullptr);
    
    lmp->input->one("run 0");
    
    // Check mixed parameters (1-2 interaction)
    pair->k_params.sync_host();
    
    // Arithmetic mixing: eps_12 = (eps_11 + eps_22)/2 = (1.0 + 2.0)/2 = 1.5
    // sigma_12 = (sigma_11 + sigma_22)/2 = (1.0 + 1.2)/2 = 1.1
    // lj1 = 48*eps*sigma^12, lj2 = 24*eps*sigma^6
    
    double eps_mixed = 1.5;
    double sigma_mixed = 1.1;
    double expected_lj1 = 48.0 * eps_mixed * pow(sigma_mixed, 12);
    double expected_lj2 = 24.0 * eps_mixed * pow(sigma_mixed, 6);
    
    // Get actual mixed values
    auto h_lj1 = pair->k_params.h_view(1,2).lj1;
    auto h_lj2 = pair->k_params.h_view(1,2).lj2;
    
    // Mixed parameters might have larger error in single precision
    double tol = getCurrentPrecisionMode() == SINGLE_SINGLE ? 0.01 : 0.0001;
    EXPECT_NEAR(h_lj1, expected_lj1, expected_lj1 * tol);
    EXPECT_NEAR(h_lj2, expected_lj2, expected_lj2 * tol);
}

// Test 12: Short-range interactions
TEST_F(MixedPrecisionPairsSimpleTest, ShortRangeInteractions) {
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    // Create two atoms very close together
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, ALL_MASK);
    
    // Place two atoms at r = 0.5 sigma (very repulsive)
    atomKK->x[0][0] = 0.0;
    atomKK->x[0][1] = 0.0;
    atomKK->x[0][2] = 0.0;
    
    if (atomKK->nlocal > 1) {
        atomKK->x[1][0] = 0.5;
        atomKK->x[1][1] = 0.0;
        atomKK->x[1][2] = 0.0;
    }
    atomKK->modified(Host, X_MASK);
    
    lmp->input->one("run 0");
    
    // Energy should be large and positive (repulsive)
    double pe = lmp->force->pair->eng_vdwl;
    EXPECT_GT(pe, 100.0);  // Very repulsive
    EXPECT_TRUE(checkNumericalStability(pe));
    
    // Forces should be large but finite
    atomKK->sync(Host, F_MASK);
    if (atomKK->nlocal > 1) {
        double f_mag = sqrt(atomKK->f[0][0]*atomKK->f[0][0] + 
                           atomKK->f[0][1]*atomKK->f[0][1] + 
                           atomKK->f[0][2]*atomKK->f[0][2]);
        EXPECT_GT(f_mag, 100.0);  // Large force
        EXPECT_LT(f_mag, 1e10);    // But finite
        EXPECT_TRUE(checkNumericalStability(f_mag));
    }
}

// Test 13: Long-range tail corrections
TEST_F(MixedPrecisionPairsSimpleTest, TailCorrections) {
    lmp->input->one("pair_style lj/cut/kk 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0");
    lmp->input->one("pair_modify tail yes");
    
    lmp->input->one("run 0");
    
    // Check tail correction energy
    double etail = lmp->force->pair->etail;
    double ptail = lmp->force->pair->ptail;
    
    EXPECT_TRUE(checkNumericalStability(etail));
    EXPECT_TRUE(checkNumericalStability(ptail));
    
    // Tail corrections should be negative for attractive potential
    EXPECT_LT(etail, 0.0);
    EXPECT_NE(ptail, 0.0);
}

// Test 14: Pair style switching
TEST_F(MixedPrecisionPairsSimpleTest, PairStyleSwitching) {
    std::vector<std::pair<std::string, std::string>> styles = {
        {"lj/cut/kk 2.5", "pair_coeff 1 1 1.0 1.0"},
        {"morse/kk 2.5", "pair_coeff 1 1 1.0 2.0 1.2"},
        {"buck/kk 2.5", "pair_coeff 1 1 1000.0 0.36 32.0"},
        {"yukawa/kk 2.0 2.5", "pair_coeff 1 1 1.0"}
    };
    
    for (const auto& [style, coeff] : styles) {
        lmp->input->one(("pair_style " + style).c_str());
        lmp->input->one(coeff.c_str());
        lmp->input->one("run 0");
        
        // Verify computation completes without errors
        EXPECT_TRUE(checkNumericalStability(lmp->force->pair->eng_vdwl))
            << "Failed for style: " << style;
        
        // Check forces are reasonable
        auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
        atomKK->sync(Host, F_MASK);
        for (int i = 0; i < atomKK->nlocal && i < 10; i++) {
            for (int j = 0; j < 3; j++) {
                EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]))
                    << "Force unstable for style: " << style;
            }
        }
    }
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
