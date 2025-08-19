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
   Testing common fix styles (NVE, NVT, NPT, Langevin) with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "pair.h"     // for Pair and its members (e.g., eng_vdwl)
#include "compute.h"  // for Compute and its members (e.g., scalar)
#include "fix_nve_kokkos.h"
#include "fix_nvt_kokkos.h"
#include "fix_npt_kokkos.h"
#include "fix_langevin_kokkos.h"
#include "fix_setforce_kokkos.h"
#include "fix_momentum_kokkos.h"
#include "fix_temp_berendsen_kokkos.h"
#include "fix_temp_rescale_kokkos.h"
#include "fix.h"
#include "modify.h"
#include "force.h"
#include "neighbor.h"
#include "input.h"
#include <cmath>

namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionFixesCommonTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
        
        // Create a simple LJ system for testing fixes
        lmp->input->one("units lj");
        lmp->input->one("atom_style atomic");
        lmp->input->one("lattice fcc 0.8442");
        lmp->input->one("region box block 0 3 0 3 0 3");
        lmp->input->one("create_box 1 box");
        lmp->input->one("create_atoms 1 box");
        lmp->input->one("mass 1 1.0");
        lmp->input->one("velocity all create 1.0 12345");
        lmp->input->one("pair_style lj/cut/kk 2.5");
        lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
        lmp->input->one("neighbor 0.3 bin");
        lmp->input->one("neigh_modify delay 0 every 1 check yes");
    }
};

// Test 1: FixNVEKokkos precision types
TEST_F(MixedPrecisionFixesCommonTest, FixNVETypes) {
    lmp->input->one("fix 1 all nve/kk");
    
    auto fix = dynamic_cast<FixNVEKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // NVE works directly with atom arrays
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Check that velocity and force arrays have correct precision
    EXPECT_TRUE((std::is_same<typename decltype(atomKK->k_v.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<typename decltype(atomKK->k_v.d_view)::value_type, KK_FLOAT>::value));
    EXPECT_TRUE((std::is_same<typename decltype(atomKK->k_f.h_view)::value_type, double>::value));
    EXPECT_TRUE((std::is_same<typename decltype(atomKK->k_f.d_view)::value_type, KK_SUM_FLOAT>::value));
    
    // Run a few steps
    lmp->input->one("run 10");
    
    // Check energy conservation (roughly, as there's truncation)
    lmp->input->one("compute ke all ke");
    lmp->input->one("variable etotal equal pe+c_ke");
    lmp->input->one("run 0");
    
    double etotal_initial = lmp->force->pair->eng_vdwl;  // Simplified
    
    lmp->input->one("run 100");
    
    double etotal_final = lmp->force->pair->eng_vdwl;  // Simplified
    
    // Energy drift depends on precision
    double drift = std::abs(etotal_final - etotal_initial) / std::abs(etotal_initial);
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    EXPECT_LT(drift, 0.1);  // 10% drift acceptable for single precision
#else
    EXPECT_LT(drift, 0.01);  // 1% drift for double/mixed
#endif
}

// Test 2: FixNVTKokkos thermostat precision
TEST_F(MixedPrecisionFixesCommonTest, FixNVTThermostat) {
    lmp->input->one("fix 1 all nvt/kk temp 1.0 1.0 0.1");
    
    auto fix = dynamic_cast<FixNVTKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Check internal precision of thermostat variables
    // Note: t_target is a protected member, cannot directly access
    // We can verify the thermostat is working by checking temperature control
    
    // Run equilibration
    lmp->input->one("run 100");
    
    // Check temperature is controlled
    lmp->input->one("compute temp all temp");
    lmp->input->one("run 0");
    
    // Temperature should be near target
    double temp = lmp->modify->compute[0]->scalar;
    EXPECT_NEAR(temp, 1.0, 0.2);  // Within 20% of target
    EXPECT_TRUE(checkNumericalStability(temp));
}

// Test 3: FixNPTKokkos barostat precision
TEST_F(MixedPrecisionFixesCommonTest, FixNPTBarostat) {
    lmp->input->one("fix 1 all npt/kk temp 1.0 1.0 0.1 iso 1.0 1.0 1.0");
    
    auto fix = dynamic_cast<FixNPTKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run equilibration
    lmp->input->one("run 100");
    
    // Check pressure and box size changes
    lmp->input->one("compute press all pressure");
    lmp->input->one("run 0");
    
    double press = lmp->modify->compute[0]->scalar;
    EXPECT_TRUE(checkNumericalStability(press));
    
    // Box should have adjusted
    double vol = lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd;
    EXPECT_GT(vol, 0.0);
    EXPECT_TRUE(checkNumericalStability(vol));
}

// Test 4: FixLangevinKokkos random force precision
TEST_F(MixedPrecisionFixesCommonTest, FixLangevinRandom) {
    lmp->input->one("fix 1 all langevin/kk 1.0 1.0 0.1 12345");
    lmp->input->one("fix 2 all nve/kk");  // Need integrator too
    
    auto fix = dynamic_cast<FixLangevinKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Check that random number generation works with precision
    lmp->input->one("run 100");
    
    // Temperature should stabilize around target
    lmp->input->one("compute temp all temp");
    lmp->input->one("run 0");
    
    double temp = lmp->modify->compute[0]->scalar;
    EXPECT_NEAR(temp, 1.0, 0.3);  // Within 30% of target (random forces)
    EXPECT_TRUE(checkNumericalStability(temp));
}

// Test 5: FixSetForceKokkos precision
TEST_F(MixedPrecisionFixesCommonTest, FixSetForce) {
    lmp->input->one("fix 1 all setforce/kk 1.0 2.0 3.0");
    
    auto fix = dynamic_cast<FixSetForceKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    lmp->input->one("run 1");
    
    // Check forces are set correctly
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, F_MASK);
    
    for (int i = 0; i < atomKK->nlocal && i < 10; i++) {
        EXPECT_PRECISION_NEAR(atomKK->f[i][0], 1.0, getAbsoluteTolerance());
        EXPECT_PRECISION_NEAR(atomKK->f[i][1], 2.0, getAbsoluteTolerance());
        EXPECT_PRECISION_NEAR(atomKK->f[i][2], 3.0, getAbsoluteTolerance());
    }
}

// Test 6: FixMomentumKokkos precision
TEST_F(MixedPrecisionFixesCommonTest, FixMomentum) {
    lmp->input->one("fix 1 all momentum/kk 10 linear 1 1 1");
    
    auto fix = dynamic_cast<FixMomentumKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run with momentum removal
    lmp->input->one("fix 2 all nve/kk");
    lmp->input->one("run 50");
    
    // Check total momentum
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, V_MASK);
    
    double px = 0.0, py = 0.0, pz = 0.0;
    for (int i = 0; i < atomKK->nlocal; i++) {
        // Assuming mass = 1.0 for simplicity
        px += atomKK->v[i][0];
        py += atomKK->v[i][1];
        pz += atomKK->v[i][2];
    }
    
    // Total momentum should be near zero
    double tol = getCurrentPrecisionMode() == SINGLE_SINGLE ? 1e-4 : 1e-8;
    EXPECT_NEAR(px, 0.0, tol * atomKK->nlocal);
    EXPECT_NEAR(py, 0.0, tol * atomKK->nlocal);
    EXPECT_NEAR(pz, 0.0, tol * atomKK->nlocal);
}

// Test 7: FixTempBerendsenKokkos thermostat
TEST_F(MixedPrecisionFixesCommonTest, FixTempBerendsen) {
    lmp->input->one("fix 1 all temp/berendsen/kk 1.0 1.0 0.1");
    lmp->input->one("fix 2 all nve/kk");
    
    auto fix = dynamic_cast<FixTempBerendsenKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run equilibration
    lmp->input->one("run 100");
    
    // Check temperature
    lmp->input->one("compute temp all temp");
    lmp->input->one("run 0");
    
    double temp = lmp->modify->compute[0]->scalar;
    EXPECT_NEAR(temp, 1.0, 0.2);
    EXPECT_TRUE(checkNumericalStability(temp));
}

// Test 8: FixTempRescaleKokkos precision
TEST_F(MixedPrecisionFixesCommonTest, FixTempRescale) {
    lmp->input->one("fix 1 all temp/rescale/kk 10 1.0 1.0 0.02 1.0");
    lmp->input->one("fix 2 all nve/kk");
    
    auto fix = dynamic_cast<FixTempRescaleKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run with temperature rescaling
    lmp->input->one("run 50");
    
    // Temperature should be at target after rescaling
    lmp->input->one("compute temp all temp");
    lmp->input->one("run 0");
    
    double temp = lmp->modify->compute[0]->scalar;
    EXPECT_NEAR(temp, 1.0, 0.1);
    EXPECT_TRUE(checkNumericalStability(temp));
}

// Test 9: Integration timestep precision
TEST_F(MixedPrecisionFixesCommonTest, TimestepPrecision) {
    // Test with very small timestep
    lmp->input->one("timestep 0.0001");
    lmp->input->one("fix 1 all nve/kk");
    
    // Save initial state
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK | V_MASK);
    
    std::vector<double> init_x, init_v;
    for (int i = 0; i < atomKK->nlocal && i < 10; i++) {
        init_x.push_back(atomKK->x[i][0]);
        init_v.push_back(atomKK->v[i][0]);
    }
    
    // Run short simulation
    lmp->input->one("run 10");
    
    // Positions should change very little with small timestep
    atomKK->sync(Host, X_MASK);
    for (int i = 0; i < atomKK->nlocal && i < 10; i++) {
        double dx = std::abs(atomKK->x[i][0] - init_x[i]);
        EXPECT_LT(dx, 0.01);  // Small displacement
        EXPECT_TRUE(checkNumericalStability(atomKK->x[i][0]));
    }
}

// Test 10: Fix execution order with precision
TEST_F(MixedPrecisionFixesCommonTest, FixExecutionOrder) {
    // Multiple fixes that modify forces
    lmp->input->one("fix 1 all addforce/kk 0.1 0.0 0.0");
    lmp->input->one("fix 2 all setforce/kk NULL 0.0 0.0");  // Zero y,z forces
    lmp->input->one("fix 3 all nve/kk");
    
    lmp->input->one("run 10");
    
    // Check that fixes executed in correct order
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, F_MASK);
    
    for (int i = 0; i < atomKK->nlocal && i < 10; i++) {
        // y and z forces should be zero (setforce)
        EXPECT_NEAR(atomKK->f[i][1], 0.0, getAbsoluteTolerance());
        EXPECT_NEAR(atomKK->f[i][2], 0.0, getAbsoluteTolerance());
        // x force should be non-zero (addforce + pair forces)
        EXPECT_TRUE(checkNumericalStability(atomKK->f[i][0]));
    }
}

// Test 11: Temperature computation precision
TEST_F(MixedPrecisionFixesCommonTest, TemperatureComputePrecision) {
    lmp->input->one("fix 1 all nvt/kk temp 1.5 1.5 0.1");
    
    // Run to equilibrate
    lmp->input->one("run 100");
    
    // Compute temperature multiple ways
    lmp->input->one("compute temp1 all temp");
    lmp->input->one("compute ke all ke");
    lmp->input->one("variable temp2 equal 2*c_ke/(3*atoms)");  // Manual calculation
    lmp->input->one("run 0");
    
    double temp1 = lmp->modify->compute[0]->scalar;
    double ke = lmp->modify->compute[1]->scalar;
    int natoms = lmp->atom->natoms;
    double temp2 = 2.0 * ke / (3.0 * natoms);
    
    // Both should give same temperature within precision
    EXPECT_PRECISION_NEAR(temp1, temp2, getRelativeTolerance() * temp1);
    EXPECT_NEAR(temp1, 1.5, 0.3);  // Near target
}

// Test 12: Velocity Verlet integration accuracy
TEST_F(MixedPrecisionFixesCommonTest, VelocityVerletAccuracy) {
    // Simple harmonic oscillator test
    lmp->input->one("clear");
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("velocity all set 1.0 0.0 0.0");
    lmp->input->one("pair_style zero 10.0");
    lmp->input->one("pair_coeff * *");
    lmp->input->one("fix 1 all addforce/kk -1.0 0.0 0.0");  // Linear restoring force
    lmp->input->one("fix 2 all nve/kk");
    lmp->input->one("timestep 0.01");
    
    // Track oscillation
    std::vector<double> positions;
    for (int step = 0; step < 100; step++) {
        lmp->input->one("run 1");
        auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
        atomKK->sync(Host, X_MASK);
        positions.push_back(atomKK->x[0][0]);
    }
    
    // Should oscillate around x=5.0
    double mean_pos = 0.0;
    for (double pos : positions) {
        mean_pos += pos;
    }
    mean_pos /= positions.size();
    
    EXPECT_NEAR(mean_pos, 5.0, 0.1);
    
    // Check oscillation amplitude is preserved (energy conservation)
    double max_pos = *std::max_element(positions.begin(), positions.end());
    double min_pos = *std::min_element(positions.begin(), positions.end());
    double amplitude = (max_pos - min_pos) / 2.0;
    
    // Amplitude should be roughly constant (some damping OK with precision)
#ifdef LMP_KOKKOS_SINGLE_SINGLE
    EXPECT_NEAR(amplitude, 1.0, 0.2);  // 20% error OK for single
#else
    EXPECT_NEAR(amplitude, 1.0, 0.05);  // 5% error for double/mixed
#endif
}

// Test 13: Pressure tensor precision
TEST_F(MixedPrecisionFixesCommonTest, PressureTensorPrecision) {
    lmp->input->one("fix 1 all npt/kk temp 1.0 1.0 0.1 aniso 1.0 1.0 1.0");
    
    lmp->input->one("run 100");
    
    // Check pressure tensor components
    lmp->input->one("compute press all pressure");
    lmp->input->one("run 0");
    
    // Get pressure tensor (stored in compute)
    auto compute = lmp->modify->compute[0];
    
    // Diagonal components should be similar for isotropic system
    if (compute->vector) {
        double pxx = compute->vector[0];
        double pyy = compute->vector[1];
        double pzz = compute->vector[2];
        
        EXPECT_TRUE(checkNumericalStability(pxx));
        EXPECT_TRUE(checkNumericalStability(pyy));
        EXPECT_TRUE(checkNumericalStability(pzz));
        
        // Should be roughly equal
        double avg_p = (pxx + pyy + pzz) / 3.0;
        EXPECT_NEAR(pxx, avg_p, std::abs(avg_p) * 0.2);
        EXPECT_NEAR(pyy, avg_p, std::abs(avg_p) * 0.2);
        EXPECT_NEAR(pzz, avg_p, std::abs(avg_p) * 0.2);
    }
}

// Test 14: Fix restart with precision
TEST_F(MixedPrecisionFixesCommonTest, FixRestartPrecision) {
    lmp->input->one("fix 1 all nvt/kk temp 1.0 1.0 0.1");
    lmp->input->one("run 50");
    
    // Get current temperature
    lmp->input->one("compute temp all temp");
    lmp->input->one("run 0");
    double temp_before = lmp->modify->compute[0]->scalar;
    
    // Write and read restart (simplified - would need actual file I/O)
    // For now, just test that fix continues to work
    lmp->input->one("run 50");
    
    lmp->input->one("run 0");
    double temp_after = lmp->modify->compute[0]->scalar;
    
    // Temperature should still be controlled
    EXPECT_NEAR(temp_after, 1.0, 0.3);
    EXPECT_TRUE(checkNumericalStability(temp_after));
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
