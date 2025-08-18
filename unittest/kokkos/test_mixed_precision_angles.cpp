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
   Testing angle styles with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "angle_harmonic_kokkos.h"
#include "angle_charmm_kokkos.h"
#include "angle_cosine_kokkos.h"
#include "angle_class2_kokkos.h"
#include "force.h"
#include "neighbor.h"
#include "input.h"
#include <cmath>

using namespace LAMMPS_NS;
using namespace TestUtils;

class MixedPrecisionAnglesTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
        
        // Create a simple 3-atom water-like system for angle testing
        lmp->input->one("units real");
        lmp->input->one("atom_style angle");
        lmp->input->one("boundary p p p");
        lmp->input->one("region box block 0 10 0 10 0 10");
        lmp->input->one("create_box 1 box angle/types 1 bond/types 1");
        
        // Create 3 atoms in angle configuration
        lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
        lmp->input->one("create_atoms 1 single 5.8 5.0 5.0");
        lmp->input->one("create_atoms 1 single 5.4 5.7 5.0");
        
        lmp->input->one("mass 1 1.0");
        lmp->input->one("pair_style zero 10.0");
        lmp->input->one("pair_coeff * *");
    }
};

// Test 1: AngleHarmonicKokkos precision types
TEST_F(MixedPrecisionAnglesTest, AngleHarmonicTypes) {
    lmp->input->one("angle_style harmonic/kk");
    lmp->input->one("angle_coeff 1 50.0 109.47");
    
    auto angle = dynamic_cast<AngleHarmonicKokkos<LMPDeviceType>*>(lmp->force->angle);
    ASSERT_NE(angle, nullptr);
    
    // Check that internal arrays use correct precision types (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_k.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_theta0.d_view)::value_type, KK_FLOAT>::value));
    
    // Check dual view types
    EXPECT_TRUE((std::is_same<decltype(angle->k_eatom.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<decltype(angle->k_eatom.d_view)::value_type, KK_FLOAT>::value));
}

// Test 2: AngleHarmonicKokkos computation accuracy
TEST_F(MixedPrecisionAnglesTest, AngleHarmonicComputation) {
    lmp->input->one("angle_style harmonic/kk");
    lmp->input->one("angle_coeff 1 50.0 120.0");  // 120 degree angle
    
    // Manually set up an angle
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, ALL_MASK);
    
    // Set positions to form a 120-degree angle
    double* x = atomKK->x[0];
    x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;  // Atom 1 at origin
    x = atomKK->x[1];
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;  // Atom 2 on x-axis
    x = atomKK->x[2];
    x[0] = 0.5; x[1] = 0.866025; x[2] = 0.0;  // Atom 3 at 120 degrees
    
    atomKK->modified(Host, X_MASK);
    
    // Compute angle energy
    lmp->input->one("run 0");
    
    // Check energy calculation (should be near zero for equilibrium angle)
    double pe = lmp->force->angle->energy;
    EXPECT_PRECISION_NEAR(pe, 0.0, getAbsoluteTolerance() * 100);
}

// Test 3: AngleCharmmKokkos with Urey-Bradley term
TEST_F(MixedPrecisionAnglesTest, AngleCharmmUB) {
    lmp->input->one("angle_style charmm/kk");
    lmp->input->one("angle_coeff 1 50.0 109.47 30.0 2.0");  // With UB term
    
    auto angle = dynamic_cast<AngleCharmmKokkos<LMPDeviceType>*>(lmp->force->angle);
    ASSERT_NE(angle, nullptr);
    
    // Check UB-specific arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_k_ub.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_r_ub.d_view)::value_type, KK_FLOAT>::value));
    
    // Run computation
    lmp->input->one("run 0");
    
    // Verify no NaN/Inf in forces
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, F_MASK);
    
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
        }
    }
}

// Test 4: AngleCosineKokkos precision
TEST_F(MixedPrecisionAnglesTest, AngleCosine) {
    lmp->input->one("angle_style cosine/kk");
    lmp->input->one("angle_coeff 1 50.0");
    
    auto angle = dynamic_cast<AngleCosineKokkos<LMPDeviceType>*>(lmp->force->angle);
    ASSERT_NE(angle, nullptr);
    
    // Check k array precision (dual view)
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_k.d_view)::value_type, KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    // Check energy is computed correctly
    double pe = lmp->force->angle->energy;
    EXPECT_GE(pe, 0.0);  // Energy should be non-negative
}

// Test 5: AngleClass2Kokkos with cross terms
TEST_F(MixedPrecisionAnglesTest, AngleClass2CrossTerms) {
    lmp->input->one("angle_style class2/kk");
    // angle_coeff with multiple terms: theta0 k2 k3 k4
    lmp->input->one("angle_coeff 1 109.47 50.0 -10.0 2.0");
    // bb term: M k r1 r2
    lmp->input->one("angle_coeff 1 bb 20.0 1.0 1.0");
    // ba term: N k1 k2 r1 r2  
    lmp->input->one("angle_coeff 1 ba 15.0 15.0 1.0 1.0");
    
    auto angle = dynamic_cast<AngleClass2Kokkos<LMPDeviceType>*>(lmp->force->angle);
    ASSERT_NE(angle, nullptr);
    
    // Check all coefficient arrays (dual views)
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_k2.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_k3.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_k4.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_bb_k.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(angle->k_ba_k1.d_view)::value_type, KK_FLOAT>::value));
    
    lmp->input->one("run 0");
    
    // Verify energy contributions
    double pe = lmp->force->angle->energy;
    EXPECT_TRUE(checkNumericalStability(pe));
}

// Test 6: Force array precision and accumulation
TEST_F(MixedPrecisionAnglesTest, ForceAccumulation) {
    lmp->input->one("angle_style harmonic/kk");
    lmp->input->one("angle_coeff 1 100.0 90.0");  // Strong angle at 90 degrees
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Set up right angle
    atomKK->sync(Host, X_MASK);
    atomKK->x[0][0] = 0.0; atomKK->x[0][1] = 0.0; atomKK->x[0][2] = 0.0;
    atomKK->x[1][0] = 1.0; atomKK->x[1][1] = 0.0; atomKK->x[1][2] = 0.0;
    atomKK->x[2][0] = 0.0; atomKK->x[2][1] = 1.0; atomKK->x[2][2] = 0.0;
    atomKK->modified(Host, X_MASK);
    
    lmp->input->one("run 0");
    
    // Check force accumulation type
    atomKK->sync(Host, F_MASK);
    
    // Forces should use KK_SUM_FLOAT for accumulation internally
    // Verify forces are reasonable
    for (int i = 0; i < atomKK->nlocal; i++) {
        double fmag = sqrt(atomKK->f[i][0]*atomKK->f[i][0] + 
                          atomKK->f[i][1]*atomKK->f[i][1] + 
                          atomKK->f[i][2]*atomKK->f[i][2]);
        EXPECT_LT(fmag, 1e6);  // Forces shouldn't be enormous
    }
}

// Test 7: View layouts for angle lists
TEST_F(MixedPrecisionAnglesTest, AngleListLayouts) {
    lmp->input->one("angle_style harmonic/kk");
    lmp->input->one("angle_coeff 1 50.0 109.47");
    
    auto angle = dynamic_cast<AngleHarmonicKokkos<LMPDeviceType>*>(lmp->force->angle);
    ASSERT_NE(angle, nullptr);
    
    // Check anglelist layout
    using anglelist_type = decltype(angle->anglelist);
    
    // Should use LayoutRight for angle lists
    EXPECT_TRUE((std::is_same<typename anglelist_type::array_layout, 
                              Kokkos::LayoutRight>::value));
}

// Test 8: Energy/virial accumulation precision
TEST_F(MixedPrecisionAnglesTest, EnergyVirialAccumulation) {
    lmp->input->one("angle_style harmonic/kk");
    lmp->input->one("angle_coeff 1 75.0 100.0");
    lmp->input->one("compute pe all pe");
    lmp->input->one("compute pressure all pressure NULL virial");
    
    // Set up non-equilibrium angle to generate forces
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK);
    atomKK->x[0][0] = 0.0; atomKK->x[0][1] = 0.0; atomKK->x[0][2] = 0.0;
    atomKK->x[1][0] = 1.0; atomKK->x[1][1] = 0.0; atomKK->x[1][2] = 0.0;
    atomKK->x[2][0] = 0.8; atomKK->x[2][1] = 0.6; atomKK->x[2][2] = 0.0;
    atomKK->modified(Host, X_MASK);
    
    lmp->input->one("run 0");
    
    // Check that energy is accumulated correctly
    double pe = lmp->force->angle->energy;
    EXPECT_GT(pe, 0.0);  // Should have positive energy for non-equilibrium
    EXPECT_TRUE(checkNumericalStability(pe));
    
    // Check virial computation
    for (int i = 0; i < 6; i++) {
        EXPECT_TRUE(checkNumericalStability(lmp->force->angle->virial[i]));
    }
}

// Test 9: Precision impact on angle gradients
TEST_F(MixedPrecisionAnglesTest, AngleGradientPrecision) {
    lmp->input->one("angle_style harmonic/kk");
    lmp->input->one("angle_coeff 1 50.0 120.0");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Test gradient at different angles
    std::vector<double> test_angles = {60.0, 90.0, 120.0, 150.0, 180.0};
    
    for (double angle_deg : test_angles) {
        double angle_rad = angle_deg * M_PI / 180.0;
        
        // Set up atoms to form specified angle
        atomKK->sync(Host, X_MASK);
        atomKK->x[0][0] = 0.0; atomKK->x[0][1] = 0.0; atomKK->x[0][2] = 0.0;
        atomKK->x[1][0] = 1.0; atomKK->x[1][1] = 0.0; atomKK->x[1][2] = 0.0;
        atomKK->x[2][0] = cos(angle_rad); 
        atomKK->x[2][1] = sin(angle_rad); 
        atomKK->x[2][2] = 0.0;
        atomKK->modified(Host, X_MASK);
        
        lmp->input->one("run 0 pre yes post no");
        
        // Check force computation stability
        atomKK->sync(Host, F_MASK);
        for (int i = 0; i < atomKK->nlocal; i++) {
            for (int j = 0; j < 3; j++) {
                EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]))
                    << "Force unstable at angle " << angle_deg << " degrees";
            }
        }
    }
}

// Test 10: Mixed precision with per-atom energy/stress
TEST_F(MixedPrecisionAnglesTest, PerAtomEnergyStress) {
    lmp->input->one("angle_style harmonic/kk");
    lmp->input->one("angle_coeff 1 50.0 109.47");
    lmp->input->one("compute pe_atom all pe/atom angle");
    lmp->input->one("compute stress_atom all stress/atom NULL virial");
    
    lmp->input->one("run 0");
    
    auto angle = dynamic_cast<AngleHarmonicKokkos<LMPDeviceType>*>(lmp->force->angle);
    ASSERT_NE(angle, nullptr);
    
    // Check per-atom arrays use correct precision
    if (angle->k_eatom.h_view.extent(0) > 0) {
        EXPECT_TRUE((std::is_same<decltype(angle->k_eatom.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(angle->k_eatom.d_view)::value_type, KK_FLOAT>::value));
    }
    
    if (angle->k_vatom.h_view.extent(0) > 0) {
        EXPECT_TRUE((std::is_same<decltype(angle->k_vatom.h_view)::value_type, double>::value));
        EXPECT_TRUE((std::is_same<decltype(angle->k_vatom.d_view)::value_type, KK_FLOAT>::value));
    }
}

// Test 11: Angle style switching and precision consistency
TEST_F(MixedPrecisionAnglesTest, AngleStyleSwitching) {
    // Test switching between angle styles maintains precision
    std::vector<std::string> styles = {
        "harmonic/kk",
        "cosine/kk",
        "charmm/kk"
    };
    
    for (const auto& style : styles) {
        lmp->input->one(("angle_style " + style).c_str());
        
        if (style == "charmm/kk") {
            lmp->input->one("angle_coeff 1 50.0 109.47 30.0 2.0");
        } else if (style == "cosine/kk") {
            lmp->input->one("angle_coeff 1 50.0");
        } else {
            lmp->input->one("angle_coeff 1 50.0 109.47");
        }
        
        lmp->input->one("run 0");
        
        // Verify computation completes without errors
        EXPECT_TRUE(checkNumericalStability(lmp->force->angle->energy))
            << "Failed for style: " << style;
    }
}

// Test 12: Extreme angle values (near 0 and 180 degrees)
TEST_F(MixedPrecisionAnglesTest, ExtremeAngles) {
    lmp->input->one("angle_style harmonic/kk");
    lmp->input->one("angle_coeff 1 50.0 120.0");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Test near-linear configuration (almost 180 degrees)
    atomKK->sync(Host, X_MASK);
    atomKK->x[0][0] = 0.0; atomKK->x[0][1] = 0.0; atomKK->x[0][2] = 0.0;
    atomKK->x[1][0] = 1.0; atomKK->x[1][1] = 0.0; atomKK->x[1][2] = 0.0;
    atomKK->x[2][0] = 1.999; atomKK->x[2][1] = 0.001; atomKK->x[2][2] = 0.0;
    atomKK->modified(Host, X_MASK);
    
    lmp->input->one("run 0");
    
    // Should handle near-linear angles without numerical issues
    atomKK->sync(Host, F_MASK);
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
            EXPECT_FALSE(std::isinf(atomKK->f[i][j]));
        }
    }
    
    // Test near-zero angle (almost 0 degrees)
    atomKK->sync(Host, X_MASK);
    atomKK->x[0][0] = 0.0; atomKK->x[0][1] = 0.0; atomKK->x[0][2] = 0.0;
    atomKK->x[1][0] = 1.0; atomKK->x[1][1] = 0.0; atomKK->x[1][2] = 0.0;
    atomKK->x[2][0] = -0.999; atomKK->x[2][1] = 0.001; atomKK->x[2][2] = 0.0;
    atomKK->modified(Host, X_MASK);
    
    lmp->input->one("run 0 pre yes post no");
    
    atomKK->sync(Host, F_MASK);
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
        }
    }
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
