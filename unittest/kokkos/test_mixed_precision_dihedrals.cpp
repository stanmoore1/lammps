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
   Testing dihedral and improper styles with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "dihedral_charmm_kokkos.h"
#include "dihedral_class2_kokkos.h"
#include "dihedral_harmonic_kokkos.h"
#include "dihedral_opls_kokkos.h"
#include "improper_class2_kokkos.h"
#include "improper_harmonic_kokkos.h"
#include "force.h"
#include "neighbor.h"
#include "input.h"
#include <cmath>

namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionDihedralsTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
        
        // Create a 4-atom system for dihedral testing
        lmp->input->one("units real");
        lmp->input->one("atom_style full");
        lmp->input->one("boundary p p p");
        lmp->input->one("region box block 0 10 0 10 0 10");
        lmp->input->one("create_box 1 box bond/types 1 angle/types 1 "
                       "dihedral/types 1 improper/types 1");
        
        // Create 4 atoms in a chain for dihedral
        lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
        lmp->input->one("create_atoms 1 single 6.0 5.0 5.0");
        lmp->input->one("create_atoms 1 single 6.5 6.0 5.0");
        lmp->input->one("create_atoms 1 single 7.0 6.5 6.0");
        
        lmp->input->one("mass 1 12.0");
        lmp->input->one("pair_style zero 10.0");
        lmp->input->one("pair_coeff * *");
    }
};

// Test 1: DihedralHarmonicKokkos precision types
TEST_F(MixedPrecisionDihedralsTest, DihedralHarmonicTypes) {
    lmp->input->one("dihedral_style harmonic/kk");
    lmp->input->one("dihedral_coeff 1 50.0 1 180");  // k d n (d=1, n=180)
    
    auto dihedral = dynamic_cast<DihedralHarmonicKokkos<LMPDeviceType>*>(lmp->force->dihedral);
    ASSERT_NE(dihedral, nullptr);
    
    // Check coefficient arrays use correct precision (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_multiplicity.d_view)::value_type, int>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_sign.d_view)::value_type, int>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_cos_shift.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_sin_shift.d_view)::value_type, KK_FLOAT>::value));
    
    // Check per-atom arrays
    EXPECT_TRUE((std::is_same<decltype(dihedral->k_eatom.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(dihedral->k_eatom.d_view)::value_type, KK_FLOAT>::value));
}

// Test 2: DihedralHarmonicKokkos energy calculation
TEST_F(MixedPrecisionDihedralsTest, DihedralHarmonicEnergy) {
    lmp->input->one("dihedral_style harmonic/kk");
    lmp->input->one("dihedral_coeff 1 25.0 -1 2");  // k=25, d=-1, n=2
    
    lmp->input->one("run 0");
    
    // Check energy is computed
    double pe = lmp->force->dihedral->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
    EXPECT_GE(pe, 0.0);  // Energy should be non-negative for harmonic
}

// Test 3: DihedralOPLSKokkos with Fourier series
TEST_F(MixedPrecisionDihedralsTest, DihedralOPLS) {
    lmp->input->one("dihedral_style opls/kk");
    lmp->input->one("dihedral_coeff 1 1.0 2.0 3.0 4.0");  // K1 K2 K3 K4
    
    auto dihedral = dynamic_cast<DihedralOPLSKokkos<LMPDeviceType>*>(lmp->force->dihedral);
    ASSERT_NE(dihedral, nullptr);
    
    // Check OPLS coefficient arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k1.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k2.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k3.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k4.d_view)::value_type, KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    // Verify energy computation
    double pe = lmp->force->dihedral->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 4: DihedralCharmmKokkos with weight factor
TEST_F(MixedPrecisionDihedralsTest, DihedralCharmm) {
    lmp->input->one("dihedral_style charmm/kk");
    lmp->input->one("dihedral_coeff 1 10.0 1 180 0.5");  // k n d weight
    
    auto dihedral = dynamic_cast<DihedralCharmmKokkos<LMPDeviceType>*>(lmp->force->dihedral);
    ASSERT_NE(dihedral, nullptr);
    
    // Check Charmm-specific arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_weight.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_cos_shift.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_sin_shift.d_view)::value_type, KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    double pe = lmp->force->dihedral->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 5: DihedralClass2Kokkos with multiple terms
TEST_F(MixedPrecisionDihedralsTest, DihedralClass2) {
    lmp->input->one("dihedral_style class2/kk");
    // Basic coefficients: K1 phi1 K2 phi2 K3 phi3
    lmp->input->one("dihedral_coeff 1 10.0 0 20.0 180 30.0 0");
    // mbt term
    lmp->input->one("dihedral_coeff 1 mbt 5.0 6.0 7.0 1.5");
    // ebt term  
    lmp->input->one("dihedral_coeff 1 ebt 2.0 3.0 4.0 5.0 6.0 7.0 1.5 1.5");
    // at term
    lmp->input->one("dihedral_coeff 1 at 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 120.0 120.0");
    // aat term
    lmp->input->one("dihedral_coeff 1 aat 1.0 120.0 120.0");
    // bb13 term
    lmp->input->one("dihedral_coeff 1 bb13 2.0 1.5 1.5");
    
    auto dihedral = dynamic_cast<DihedralClass2Kokkos<LMPDeviceType>*>(lmp->force->dihedral);
    ASSERT_NE(dihedral, nullptr);
    
    // Check all Class2 coefficient arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k1.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k2.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_k3.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_phi1.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_phi2.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_phi3.d_view)::value_type, KK_FLOAT>::value));
    
    // Check cross-term arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_mbt_f1.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_ebt_f1_1.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_at_f1_1.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_aat_k.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(dihedral->k_bb13t_k.d_view)::value_type, KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    double pe = lmp->force->dihedral->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 6: ImproperHarmonicKokkos precision
TEST_F(MixedPrecisionDihedralsTest, ImproperHarmonic) {
    lmp->input->one("improper_style harmonic/kk");
    lmp->input->one("improper_coeff 1 50.0 180.0");  // k chi0
    
    auto improper = dynamic_cast<ImproperHarmonicKokkos<LMPDeviceType>*>(lmp->force->improper);
    ASSERT_NE(improper, nullptr);
    
    // Check improper coefficient arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_k.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_chi.d_view)::value_type, KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    double pe = lmp->force->improper->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 7: ImproperClass2Kokkos with angle-angle term
TEST_F(MixedPrecisionDihedralsTest, ImproperClass2) {
    lmp->input->one("improper_style class2/kk");
    lmp->input->one("improper_coeff 1 10.0 0.0");  // k chi0
    lmp->input->one("improper_coeff 1 aa 5.0 6.0 7.0 120.0 120.0 120.0");
    
    auto improper = dynamic_cast<ImproperClass2Kokkos<LMPDeviceType>*>(lmp->force->improper);
    ASSERT_NE(improper, nullptr);
    
    // Check Class2 improper arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_k0.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_chi0.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_aa_k1.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_aa_k2.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_aa_k3.d_view)::value_type, KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    double pe = lmp->force->improper->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
}

/*
// Test 8: ImproperHybridKokkos precision
TEST_F(MixedPrecisionDihedralsTest, ImproperCVFF) {
    lmp->input->one("improper_style cvff/kk");
    lmp->input->one("improper_coeff 1 10.0 -1 2");  // k d n
    
    auto improper = dynamic_cast<ImproperCVFFKokkos<LMPDeviceType>*>(lmp->force->improper);
    ASSERT_NE(improper, nullptr);
    
    // Check CVFF improper arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_k.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_sign.d_view)::value_type, int>::value));
    EXPECT_TRUE((std::is_same<typename decltype(improper->k_multiplicity.d_view)::value_type, int>::value));
    
    lmp->input->one("run 0");
    
    double pe = lmp->force->improper->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
}
*/

// Test 9: Dihedral angle calculation precision
TEST_F(MixedPrecisionDihedralsTest, DihedralAngleCalculation) {
    lmp->input->one("dihedral_style harmonic/kk");
    lmp->input->one("dihedral_coeff 1 50.0 1 180");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Set up atoms in a specific dihedral angle
    atomKK->sync(Host, X_MASK);
    if (atomKK->nlocal >= 4) {
        // Create a 90-degree dihedral
        atomKK->x[0][0] = 0.0; atomKK->x[0][1] = 0.0; atomKK->x[0][2] = 0.0;
        atomKK->x[1][0] = 1.0; atomKK->x[1][1] = 0.0; atomKK->x[1][2] = 0.0;
        atomKK->x[2][0] = 1.0; atomKK->x[2][1] = 1.0; atomKK->x[2][2] = 0.0;
        atomKK->x[3][0] = 1.0; atomKK->x[3][1] = 1.0; atomKK->x[3][2] = 1.0;
        atomKK->modified(Host, X_MASK);
    }
    
    lmp->input->one("run 0");
    
    // Check forces are reasonable
    atomKK->sync(Host, F_MASK);
    for (int i = 0; i < atomKK->nlocal && i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
            EXPECT_LT(std::abs(atomKK->f[i][j]), 1e6);
        }
    }
}

// Test 10: Extreme dihedral angles (0 and 180 degrees)
TEST_F(MixedPrecisionDihedralsTest, ExtremeDihedralAngles) {
    lmp->input->one("dihedral_style harmonic/kk");
    lmp->input->one("dihedral_coeff 1 50.0 1 90");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Test planar configuration (dihedral = 0 or 180)
    atomKK->sync(Host, X_MASK);
    if (atomKK->nlocal >= 4) {
        // All atoms in a plane (dihedral = 180)
        atomKK->x[0][0] = 0.0; atomKK->x[0][1] = 0.0; atomKK->x[0][2] = 0.0;
        atomKK->x[1][0] = 1.0; atomKK->x[1][1] = 0.0; atomKK->x[1][2] = 0.0;
        atomKK->x[2][0] = 2.0; atomKK->x[2][1] = 0.0; atomKK->x[2][2] = 0.0;
        atomKK->x[3][0] = 3.0; atomKK->x[3][1] = 0.0; atomKK->x[3][2] = 0.0;
        atomKK->modified(Host, X_MASK);
    }
    
    lmp->input->one("run 0");
    
    // Should handle planar configurations without numerical issues
    double pe = lmp->force->dihedral->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
    
    atomKK->sync(Host, F_MASK);
    for (int i = 0; i < atomKK->nlocal && i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
        }
    }
}

// Test 11: Per-atom energy/virial for dihedrals
TEST_F(MixedPrecisionDihedralsTest, DihedralPerAtomEnergyVirial) {
    lmp->input->one("dihedral_style harmonic/kk");
    lmp->input->one("dihedral_coeff 1 50.0 1 180");
    lmp->input->one("compute pe_atom all pe/atom dihedral");
    lmp->input->one("compute stress_atom all stress/atom NULL virial");
    
    lmp->input->one("run 0");
    
    auto dihedral = dynamic_cast<DihedralHarmonicKokkos<LMPDeviceType>*>(lmp->force->dihedral);
    ASSERT_NE(dihedral, nullptr);
    
    // Check per-atom arrays precision
    if (dihedral->k_eatom.h_view.extent(0) > 0) {
        EXPECT_TRUE((std::is_same<decltype(dihedral->k_eatom.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(dihedral->k_eatom.d_view)::value_type, KK_FLOAT>::value));
    }
    
    if (dihedral->k_vatom.h_view.extent(0) > 0) {
        EXPECT_TRUE((std::is_same<decltype(dihedral->k_vatom.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(dihedral->k_vatom.d_view)::value_type, KK_FLOAT>::value));
    }
}

// Test 12: Dihedral list layout
TEST_F(MixedPrecisionDihedralsTest, DihedralListLayout) {
    lmp->input->one("dihedral_style harmonic/kk");
    lmp->input->one("dihedral_coeff 1 50.0 1 180");
    
    auto dihedral = dynamic_cast<DihedralHarmonicKokkos<LMPDeviceType>*>(lmp->force->dihedral);
    ASSERT_NE(dihedral, nullptr);
    
    // Check dihedrallist layout
    using dihedrallist_type = decltype(dihedral->dihedrallist);
    
    // Should use LayoutRight for dihedral lists
    EXPECT_TRUE((std::is_same<typename dihedrallist_type::array_layout, 
                              Kokkos::LayoutRight>::value));
}

// Test 13: Improper out-of-plane calculations
TEST_F(MixedPrecisionDihedralsTest, ImproperOutOfPlane) {
    lmp->input->one("improper_style harmonic/kk");
    lmp->input->one("improper_coeff 1 100.0 0.0");  // Strong out-of-plane force
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Set up atoms with one out of plane
    atomKK->sync(Host, X_MASK);
    if (atomKK->nlocal >= 4) {
        // Three atoms in xy-plane, fourth out of plane
        atomKK->x[0][0] = 0.0; atomKK->x[0][1] = 0.0; atomKK->x[0][2] = 0.0;
        atomKK->x[1][0] = 1.0; atomKK->x[1][1] = 0.0; atomKK->x[1][2] = 0.0;
        atomKK->x[2][0] = 0.0; atomKK->x[2][1] = 1.0; atomKK->x[2][2] = 0.0;
        atomKK->x[3][0] = 0.5; atomKK->x[3][1] = 0.5; atomKK->x[3][2] = 0.5; // Out of plane
        atomKK->modified(Host, X_MASK);
    }
    
    lmp->input->one("run 0");
    
    // Should generate restoring force
    double pe = lmp->force->improper->energy;
    EXPECT_GT(pe, 0.0);  // Non-zero energy for out-of-plane
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 14: Style switching for dihedrals and impropers
TEST_F(MixedPrecisionDihedralsTest, StyleSwitching) {
    // Test dihedral styles
    std::vector<std::pair<std::string, std::string>> dihedral_styles = {
        {"harmonic/kk", "dihedral_coeff 1 50.0 1 180"},
        {"opls/kk", "dihedral_coeff 1 1.0 2.0 3.0 4.0"},
        {"charmm/kk", "dihedral_coeff 1 10.0 1 180 0.5"}
    };
    
    for (const auto& [style, coeff] : dihedral_styles) {
        lmp->input->one(("dihedral_style " + style).c_str());
        lmp->input->one(coeff.c_str());
        lmp->input->one("run 0");
        
        EXPECT_TRUE(checkNumericalStability(lmp->force->dihedral->energy))
            << "Failed for dihedral style: " << style;
    }
    
    // Test improper styles
    std::vector<std::pair<std::string, std::string>> improper_styles = {
        {"harmonic/kk", "improper_coeff 1 50.0 180.0"},
        {"cvff/kk", "improper_coeff 1 10.0 -1 2"}
    };
    
    for (const auto& [style, coeff] : improper_styles) {
        lmp->input->one(("improper_style " + style).c_str());
        lmp->input->one(coeff.c_str());
        lmp->input->one("run 0");
        
        EXPECT_TRUE(checkNumericalStability(lmp->force->improper->energy))
            << "Failed for improper style: " << style;
    }
}

// Test 15: Force summation with dihedrals
TEST_F(MixedPrecisionDihedralsTest, DihedralForceSummation) {
    lmp->input->one("dihedral_style harmonic/kk");
    lmp->input->one("dihedral_coeff 1 100.0 1 90");  // Strong dihedral
    
    lmp->input->one("run 0");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, F_MASK);
    
    // Total force should sum to zero (Newton's third law)
    double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;
    for (int i = 0; i < atomKK->nlocal; i++) {
        fx_sum += atomKK->f[i][0];
        fy_sum += atomKK->f[i][1];
        fz_sum += atomKK->f[i][2];
    }
    
    // Tolerance depends on precision
    double tol = getCurrentPrecisionMode() == SINGLE_SINGLE ? 1e-3 : 1e-6;
    EXPECT_NEAR(fx_sum, 0.0, tol);
    EXPECT_NEAR(fy_sum, 0.0, tol);
    EXPECT_NEAR(fz_sum, 0.0, tol);
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
