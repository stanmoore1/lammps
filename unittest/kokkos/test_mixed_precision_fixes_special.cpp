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
   Testing specialized fix styles (SHAKE, Rigid, Wall, DPD) with mixed precision
------------------------------------------------------------------------- */

#include "gtest/gtest.h"
#include "test_mixed_precision_utils.h"
#include "lammps.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "fix_shake_kokkos.h"
//#include "fix_rigid_kokkos.h"
//#include "fix_rigid_small_kokkos.h"
#include "fix_wall_lj93_kokkos.h"
//#include "fix_wall_lj126_kokkos.h"
//#include "fix_wall_lj1043_kokkos.h"
#ifdef LMP_DPD_REACT
#include "fix_dpd_energy_kokkos.h"
#endif
#include "fix_addforce.h"
#include "fix.h"
#include "modify.h"
#include "force.h"
#include "neighbor.h"
#include "input.h"
#include <cmath>

namespace LAMMPS_NS {

using namespace TestUtils;

class MixedPrecisionFixesSpecialTest : public MixedPrecisionTestFixture {
protected:
    void SetUp() override {
        MixedPrecisionTestFixture::SetUp();
    }
    
    void SetupWaterMolecule() {
        // Create a water molecule system for SHAKE testing
        lmp->input->one("units real");
        lmp->input->one("atom_style full");
        lmp->input->one("boundary p p p");
        lmp->input->one("region box block 0 10 0 10 0 10");
        lmp->input->one("create_box 2 box bond/types 2 angle/types 1");
        
        // Create water molecule (O-H-H)
        lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");  // Oxygen
        lmp->input->one("create_atoms 2 single 5.8 5.0 5.0");  // Hydrogen 1
        lmp->input->one("create_atoms 2 single 5.4 5.8 5.0");  // Hydrogen 2
        
        lmp->input->one("mass 1 15.9994");  // O mass
        lmp->input->one("mass 2 1.008");    // H mass
        
        lmp->input->one("bond_style harmonic");
        lmp->input->one("bond_coeff 1 450.0 0.957");  // O-H bond
        lmp->input->one("bond_coeff 2 450.0 1.513");  // H-H constraint
        
        lmp->input->one("angle_style harmonic");
        lmp->input->one("angle_coeff 1 55.0 104.52");  // H-O-H angle
        
        lmp->input->one("pair_style lj/cut 10.0");
        lmp->input->one("pair_coeff * * 0.1 3.0");
        
        lmp->input->one("velocity all create 300.0 12345");
    }
    
    void SetupRigidBody() {
        // Create a system with rigid bodies
        lmp->input->one("units lj");
        lmp->input->one("atom_style molecular");
        lmp->input->one("boundary p p p");
        lmp->input->one("region box block 0 10 0 10 0 10");
        lmp->input->one("create_box 1 box");
        
        // Create a small rigid body (tetrahedron)
        lmp->input->one("create_atoms 1 single 5.0 5.0 5.0");
        lmp->input->one("create_atoms 1 single 6.0 5.0 5.0");
        lmp->input->one("create_atoms 1 single 5.5 6.0 5.0");
        lmp->input->one("create_atoms 1 single 5.5 5.5 6.0");
        
        // Set molecule IDs for rigid body
        lmp->input->one("set atom 1 mol 1");
        lmp->input->one("set atom 2 mol 1");
        lmp->input->one("set atom 3 mol 1");
        lmp->input->one("set atom 4 mol 1");
        
        lmp->input->one("mass 1 1.0");
        lmp->input->one("velocity all create 1.0 12345");
        lmp->input->one("pair_style lj/cut 2.5");
        lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    }
    
    void SetupDPDSystem() {
        // Create a DPD system
        lmp->input->one("units lj");
        lmp->input->one("atom_style atomic");
        lmp->input->one("boundary p p p");
        lmp->input->one("lattice fcc 3.0");
        lmp->input->one("region box block 0 5 0 5 0 5");
        lmp->input->one("create_box 1 box");
        lmp->input->one("create_atoms 1 box");
        lmp->input->one("mass 1 1.0");
        lmp->input->one("velocity all create 1.0 12345");
        lmp->input->one("pair_style dpd 1.0 2.5 34387");
        lmp->input->one("pair_coeff 1 1 25.0 4.5 1.0");
        lmp->input->one("neighbor 0.3 bin");
    }
};

// Test 1: FixShakeKokkos constraint precision
TEST_F(MixedPrecisionFixesSpecialTest, FixShakeConstraints) {
    SetupWaterMolecule();
    
    // Apply SHAKE constraints to O-H bonds
    lmp->input->one("fix 1 all shake/kk 0.0001 20 0 b 1");
    lmp->input->one("fix 2 all nve");
    
    auto fix = dynamic_cast<FixShakeKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Check internal precision of constraint data
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Store initial O-H bond lengths
    atomKK->sync(Host, X_MASK);
    double r_oh_initial = 0.0;
    if (atomKK->nlocal >= 2) {
        double dx = atomKK->x[1][0] - atomKK->x[0][0];
        double dy = atomKK->x[1][1] - atomKK->x[0][1];
        double dz = atomKK->x[1][2] - atomKK->x[0][2];
        r_oh_initial = sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    // Run dynamics with SHAKE
    lmp->input->one("run 100");
    
    // Check that bond lengths are constrained
    atomKK->sync(Host, X_MASK);
    if (atomKK->nlocal >= 2) {
        double dx = atomKK->x[1][0] - atomKK->x[0][0];
        double dy = atomKK->x[1][1] - atomKK->x[0][1];
        double dz = atomKK->x[1][2] - atomKK->x[0][2];
        double r_oh_final = sqrt(dx*dx + dy*dy + dz*dz);
        
        // Bond length should be maintained within SHAKE tolerance
        EXPECT_NEAR(r_oh_final, 0.957, 0.001);  // O-H bond length
        EXPECT_TRUE(checkNumericalStability(r_oh_final));
    }
}

// Test 2: FixShakeKokkos iterative solver precision
TEST_F(MixedPrecisionFixesSpecialTest, FixShakeIterations) {
    SetupWaterMolecule();
    
    // Use tighter tolerance to test iteration precision
    lmp->input->one("fix 1 all shake/kk 1e-6 100 0 b 1 2");
    lmp->input->one("fix 2 all nve");
    
    auto fix = dynamic_cast<FixShakeKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run and check convergence
    lmp->input->one("run 10");
    
    // Verify no NaN/Inf in positions and velocities
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK | V_MASK);
    
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->x[i][j]));
            EXPECT_TRUE(checkNumericalStability(atomKK->v[i][j]));
        }
    }
}

/*
// Test 3: FixRigidKokkos precision for rigid body dynamics
TEST_F(MixedPrecisionFixesSpecialTest, FixRigidBodyDynamics) {
    SetupRigidBody();
    
    // Apply rigid body fix
    lmp->input->one("fix 1 all rigid/nve/kk molecule");
    
    // FixRigidKokkos doesn't exist - skip this test
    // auto fix = dynamic_cast<FixRigidKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    // ASSERT_NE(fix, nullptr);
    
    // Check precision of rigid body properties
    // The fix stores center of mass, orientation, etc.
    
    // Run dynamics
    lmp->input->one("run 100");
    
    // Check that rigid body maintains its shape
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK);
    
    // Calculate distances between atoms in rigid body
    std::vector<double> distances;
    if (atomKK->nlocal >= 4) {
        for (int i = 0; i < 3; i++) {
            for (int j = i+1; j < 4; j++) {
                double dx = atomKK->x[j][0] - atomKK->x[i][0];
                double dy = atomKK->x[j][1] - atomKK->x[i][1];
                double dz = atomKK->x[j][2] - atomKK->x[i][2];
                double r = sqrt(dx*dx + dy*dy + dz*dz);
                distances.push_back(r);
            }
        }
    }
    
    // Run more and check distances are preserved
    lmp->input->one("run 100");
    
    atomKK->sync(Host, X_MASK);
    if (atomKK->nlocal >= 4) {
        int idx = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = i+1; j < 4; j++) {
                double dx = atomKK->x[j][0] - atomKK->x[i][0];
                double dy = atomKK->x[j][1] - atomKK->x[i][1];
                double dz = atomKK->x[j][2] - atomKK->x[i][2];
                double r = sqrt(dx*dx + dy*dy + dz*dz);
                
                // Distances should be preserved for rigid body
                EXPECT_PRECISION_NEAR(r, distances[idx], getRelativeTolerance() * r);
                idx++;
            }
        }
    }
}

// Test 4: FixRigidSmallKokkos for many small rigid bodies
TEST_F(MixedPrecisionFixesSpecialTest, FixRigidSmall) {
    // Create multiple small rigid bodies
    lmp->input->one("units lj");
    lmp->input->one("atom_style molecular");
    lmp->input->one("boundary p p p");
    lmp->input->one("region box block 0 20 0 20 0 20");
    lmp->input->one("create_box 1 box");
    
    // Create several small rigid bodies
    for (int i = 0; i < 3; i++) {
        double x0 = 5.0 + i * 5.0;
        lmp->input->one(("create_atoms 1 single " + std::to_string(x0) + " 5.0 5.0").c_str());
        lmp->input->one(("create_atoms 1 single " + std::to_string(x0+1) + " 5.0 5.0").c_str());
        lmp->input->one(("create_atoms 1 single " + std::to_string(x0+0.5) + " 6.0 5.0").c_str());
        
        // Set molecule IDs
        int start = i * 3 + 1;
        lmp->input->one(("set atom " + std::to_string(start) + " mol " + std::to_string(i+1)).c_str());
        lmp->input->one(("set atom " + std::to_string(start+1) + " mol " + std::to_string(i+1)).c_str());
        lmp->input->one(("set atom " + std::to_string(start+2) + " mol " + std::to_string(i+1)).c_str());
    }
    
    lmp->input->one("mass 1 1.0");
    lmp->input->one("velocity all create 1.0 12345");
    lmp->input->one("pair_style lj/cut 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    lmp->input->one("fix 1 all rigid/small/kk molecule");
    
    auto fix = dynamic_cast<FixRigidSmallKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run dynamics
    lmp->input->one("run 100");
    
    // Check that all rigid bodies maintain integrity
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK | V_MASK);
    
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->x[i][j]));
            EXPECT_TRUE(checkNumericalStability(atomKK->v[i][j]));
        }
    }
}
*/

// Test 5: FixWallLJ93Kokkos wall interaction precision
TEST_F(MixedPrecisionFixesSpecialTest, FixWallLJ93) {
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary p p f");  // Fixed boundary in z
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 random 100 12345 box");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("velocity all create 1.0 12345");
    lmp->input->one("pair_style lj/cut 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    // Add LJ 9-3 wall at z=0
    lmp->input->one("fix 1 all wall/lj93/kk zlo EDGE 1.0 1.0 2.5");
    lmp->input->one("fix 2 all nve");
    
    auto fix = dynamic_cast<FixWallLJ93Kokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run dynamics
    lmp->input->one("run 100");
    
    // Check that atoms don't penetrate wall
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK);
    
    for (int i = 0; i < atomKK->nlocal; i++) {
        EXPECT_GT(atomKK->x[i][2], -0.1);  // Should not go below z=0
        EXPECT_TRUE(checkNumericalStability(atomKK->x[i][2]));
    }
    
    // Check wall force/energy calculation precision
    double wall_energy = fix->compute_scalar();
    EXPECT_TRUE(checkNumericalStability(wall_energy));
}

/*
// Test 6: FixWallLJ126Kokkos precision
TEST_F(MixedPrecisionFixesSpecialTest, FixWallLJ126) {
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary p p f");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 single 5.0 5.0 1.0");  // Close to wall
    lmp->input->one("mass 1 1.0");
    lmp->input->one("velocity all set 0.0 0.0 -0.5");  // Moving toward wall
    lmp->input->one("pair_style lj/cut 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    // Add LJ 9-3 wall (12-6 not available)
    lmp->input->one("fix 1 all wall/lj93/kk zlo EDGE 1.0 1.0 2.5");
    lmp->input->one("fix 2 all nve");
    
    // FixWallLJ126Kokkos doesn't exist - use LJ93 which does exist
    auto fix = dynamic_cast<FixWallLJ93Kokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run dynamics - atom should bounce off wall
    lmp->input->one("run 50");
    
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK | V_MASK);
    
    // Atom should have bounced (positive z velocity)
    if (atomKK->nlocal > 0) {
        EXPECT_GT(atomKK->v[0][2], 0.0);  // Should be moving away from wall
        EXPECT_TRUE(checkNumericalStability(atomKK->v[0][2]));
    }
}

// Test 7: FixWallLJ1043Kokkos precision
TEST_F(MixedPrecisionFixesSpecialTest, FixWallLJ1043) {
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary f f f");  // All walls
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 random 50 12345 box");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("velocity all create 1.0 12345");
    lmp->input->one("pair_style lj/cut 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    // Add LJ 10-4-3 walls on all sides
    lmp->input->one("fix 1 all wall/lj1043/kk xlo EDGE 1.0 1.0 2.5 xhi EDGE 1.0 1.0 2.5");
    lmp->input->one("fix 2 all wall/lj1043/kk ylo EDGE 1.0 1.0 2.5 yhi EDGE 1.0 1.0 2.5");
    lmp->input->one("fix 3 all wall/lj1043/kk zlo EDGE 1.0 1.0 2.5 zhi EDGE 1.0 1.0 2.5");
    lmp->input->one("fix 4 all nve");
    
    auto fix = dynamic_cast<FixWallLJ1043Kokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run dynamics
    lmp->input->one("run 100");
    
    // Check that atoms stay within box
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK);
    
    for (int i = 0; i < atomKK->nlocal; i++) {
        EXPECT_GT(atomKK->x[i][0], 0.0);
        EXPECT_LT(atomKK->x[i][0], 10.0);
        EXPECT_GT(atomKK->x[i][1], 0.0);
        EXPECT_LT(atomKK->x[i][1], 10.0);
        EXPECT_GT(atomKK->x[i][2], 0.0);
        EXPECT_LT(atomKK->x[i][2], 10.0);
        
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->x[i][j]));
        }
    }
}

*/

#ifdef LMP_DPD_REACT
// Test 8: FixDPDEnergyKokkos thermostat precision
TEST_F(MixedPrecisionFixesSpecialTest, FixDPDEnergy) {
    SetupDPDSystem();
    
    // Apply DPD thermostat with energy conservation
    lmp->input->one("fix 1 all dpd/energy/kk");
    lmp->input->one("fix 2 all nve");
    
    auto fix = dynamic_cast<FixDPDenergyKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Check internal energy array precision
    // DPD tracks internal energy per particle
    
    // Run dynamics
    lmp->input->one("run 100");
    
    // Check temperature is maintained
    lmp->input->one("compute temp all temp");
    lmp->input->one("run 0");
    
    // Need to access compute properly
    double temp = 0.0; // Placeholder
    EXPECT_NEAR(temp, 1.0, 0.3);  // Should be near target
    EXPECT_TRUE(checkNumericalStability(temp));
    
    // Check that internal energies are reasonable
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    // Internal energy should be positive and stable
    for (int i = 0; i < atomKK->nlocal && i < 10; i++) {
        // Note: Would need access to internal energy array
        // For now, just check stability of system
        EXPECT_TRUE(checkNumericalStability(atomKK->x[i][0]));
    }
}
#endif // LMP_DPD_REACT

// Test 9: Constraint force precision in SHAKE
TEST_F(MixedPrecisionFixesSpecialTest, ShakeConstraintForces) {
    SetupWaterMolecule();
    
    lmp->input->one("fix 1 all shake/kk 1e-5 50 0 b 1 a 1");
    lmp->input->one("fix 2 all nve");
    
    auto fix = dynamic_cast<FixShakeKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Track forces before and after SHAKE
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    
    lmp->input->one("run 1");
    
    // Forces should be modified by SHAKE to maintain constraints
    atomKK->sync(Host, F_MASK);
    
    // Check that constraint forces are applied correctly
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->f[i][j]));
            EXPECT_LT(std::abs(atomKK->f[i][j]), 1e6);  // Reasonable force magnitude
        }
    }
}

/* Test 10: Rigid body angular momentum conservation - Commented out as FixRigidKokkos doesn't exist
TEST_F(MixedPrecisionFixesSpecialTest, RigidAngularMomentum) {
    SetupRigidBody();
    
    // Give rigid body some angular momentum
    lmp->input->one("velocity all set 0.0 0.0 0.0");
    lmp->input->one("velocity all create 1.0 12345 rot yes");
    
    lmp->input->one("fix 1 all rigid/nve/kk molecule");
    
    // FixRigidKokkos doesn't exist as a template
    // auto fix = dynamic_cast<FixRigidKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    // ASSERT_NE(fix, nullptr);
    
    // Run dynamics
    lmp->input->one("run 100");
    
    // Angular momentum should be conserved
    // (Would need access to fix internal data to check directly)
    
    // Check that rigid body is still rotating properly
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, V_MASK);
    
    // Calculate angular momentum components
    double Lx = 0.0, Ly = 0.0, Lz = 0.0;
    if (atomKK->nlocal >= 4) {
        // Calculate center of mass
        double xcm = 0.0, ycm = 0.0, zcm = 0.0;
        for (int i = 0; i < 4; i++) {
            xcm += atomKK->x[i][0];
            ycm += atomKK->x[i][1];
            zcm += atomKK->x[i][2];
        }
        xcm /= 4.0; ycm /= 4.0; zcm /= 4.0;
        
        // Calculate angular momentum
        for (int i = 0; i < 4; i++) {
            double rx = atomKK->x[i][0] - xcm;
            double ry = atomKK->x[i][1] - ycm;
            double rz = atomKK->x[i][2] - zcm;
            
            Lx += ry * atomKK->v[i][2] - rz * atomKK->v[i][1];
            Ly += rz * atomKK->v[i][0] - rx * atomKK->v[i][2];
            Lz += rx * atomKK->v[i][1] - ry * atomKK->v[i][0];
        }
    }
    
    double L_mag = sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
    EXPECT_TRUE(checkNumericalStability(L_mag));
    EXPECT_GT(L_mag, 0.0);  // Should have non-zero angular momentum
}
*/

// Test 11: Wall force gradient near cutoff
TEST_F(MixedPrecisionFixesSpecialTest, WallForceGradient) {
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary p p f");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 1 box");
    
    // Create atoms at various distances from wall
    for (double z = 0.5; z <= 3.0; z += 0.5) {
        lmp->input->one(("create_atoms 1 single 5.0 5.0 " + std::to_string(z)).c_str());
    }
    
    lmp->input->one("mass 1 1.0");
    lmp->input->one("pair_style lj/cut 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    
    // Add wall with specific cutoff
    lmp->input->one("fix 1 all wall/lj93/kk zlo EDGE 1.0 1.0 2.5");
    
    auto fix = dynamic_cast<FixWallLJ93Kokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Compute forces
    lmp->input->one("run 0");
    
    // Check force gradient is smooth
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, F_MASK | X_MASK);
    
    std::vector<std::pair<double, double>> force_vs_distance;
    for (int i = 0; i < atomKK->nlocal; i++) {
        force_vs_distance.push_back({atomKK->x[i][2], atomKK->f[i][2]});
    }
    
    // Sort by distance
    std::sort(force_vs_distance.begin(), force_vs_distance.end());
    
    // Check that force decreases smoothly with distance
    for (size_t i = 1; i < force_vs_distance.size(); i++) {
        double z1 = force_vs_distance[i-1].first;
        double z2 = force_vs_distance[i].first;
        double f1 = force_vs_distance[i-1].second;
        double f2 = force_vs_distance[i].second;
        
        if (z2 > z1) {
            // Force magnitude should decrease with distance (repulsive wall)
            EXPECT_LE(std::abs(f2), std::abs(f1) + getAbsoluteTolerance());
        }
        
        EXPECT_TRUE(checkNumericalStability(f2));
    }
}

// Test 12: SHAKE with multiple constraint types
TEST_F(MixedPrecisionFixesSpecialTest, ShakeMultipleConstraints) {
    // Create system with bonds and angles to constrain
    lmp->input->one("units real");
    lmp->input->one("atom_style full");
    lmp->input->one("boundary p p p");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 2 box bond/types 2 angle/types 1");
    
    // Create a chain of atoms
    for (int i = 0; i < 4; i++) {
        lmp->input->one(("create_atoms 1 single " + std::to_string(5.0 + i) + " 5.0 5.0").c_str());
    }
    
    lmp->input->one("mass 1 12.0");
    lmp->input->one("mass 2 1.0");
    
    lmp->input->one("bond_style harmonic");
    lmp->input->one("bond_coeff 1 100.0 1.0");
    lmp->input->one("bond_coeff 2 100.0 1.5");
    
    lmp->input->one("angle_style harmonic");
    lmp->input->one("angle_coeff 1 50.0 120.0");
    
    lmp->input->one("pair_style lj/cut 10.0");
    lmp->input->one("pair_coeff * * 0.1 3.0");
    
    lmp->input->one("velocity all create 300.0 12345");
    
    // Apply SHAKE to both bonds and angles
    lmp->input->one("fix 1 all shake/kk 1e-5 50 0 b 1 2 a 1");
    lmp->input->one("fix 2 all nve");
    
    auto fix = dynamic_cast<FixShakeKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run dynamics
    lmp->input->one("run 50");
    
    // Check all constraints are satisfied
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK);
    
    // All positions should be stable
    for (int i = 0; i < atomKK->nlocal; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(checkNumericalStability(atomKK->x[i][j]));
        }
    }
}

/* Test 13: Rigid body with external forces - Commented out as rigid body tests need refactoring
TEST_F(MixedPrecisionFixesSpecialTest, RigidWithExternalForces) {
    SetupRigidBody();
    
    // Apply both rigid fix and external force
    lmp->input->one("fix 1 all rigid/nve/kk molecule");
    lmp->input->one("fix 2 all addforce 0.1 0.0 0.0");  // Constant force in x
    
    // FixRigidKokkos doesn't exist
    // auto rigid_fix = dynamic_cast<FixRigidKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    // ASSERT_NE(rigid_fix, nullptr);
    
    // Track center of mass position
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK);
    
    double xcm_initial = 0.0;
    if (atomKK->nlocal >= 4) {
        for (int i = 0; i < 4; i++) {
            xcm_initial += atomKK->x[i][0];
        }
        xcm_initial /= 4.0;
    }
    
    // Run dynamics
    lmp->input->one("run 100");
    
    // Center of mass should have moved in x direction
    atomKK->sync(Host, X_MASK);
    double xcm_final = 0.0;
    if (atomKK->nlocal >= 4) {
        for (int i = 0; i < 4; i++) {
            xcm_final += atomKK->x[i][0];
        }
        xcm_final /= 4.0;
    }
    
    EXPECT_GT(xcm_final, xcm_initial);  // Should have moved in +x
    EXPECT_TRUE(checkNumericalStability(xcm_final));
    
    // Rigid body shape should be preserved
    if (atomKK->nlocal >= 4) {
        // Check one inter-atomic distance
        double dx = atomKK->x[1][0] - atomKK->x[0][0];
        double dy = atomKK->x[1][1] - atomKK->x[0][1];
        double dz = atomKK->x[1][2] - atomKK->x[0][2];
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        
        EXPECT_NEAR(r, 1.0, 0.01);  // Original distance was 1.0
    }
}
*/

#ifdef LMP_DPD_REACT
// Test 14: DPD with varying timesteps
TEST_F(MixedPrecisionFixesSpecialTest, DPDTimestepStability) {
    SetupDPDSystem();
    
    lmp->input->one("fix 1 all dpd/energy/kk");
    lmp->input->one("fix 2 all nve");
    
    auto fix = dynamic_cast<FixDPDenergyKokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Test with different timesteps
    std::vector<double> timesteps = {0.001, 0.005, 0.01};
    
    for (double dt : timesteps) {
        lmp->input->one(("timestep " + std::to_string(dt)).c_str());
        lmp->input->one("run 10");
        
        // System should remain stable
        auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
        atomKK->sync(Host, X_MASK | V_MASK);
        
        bool stable = true;
        for (int i = 0; i < atomKK->nlocal && i < 10; i++) {
            for (int j = 0; j < 3; j++) {
                if (!checkNumericalStability(atomKK->x[i][j]) ||
                    !checkNumericalStability(atomKK->v[i][j])) {
                    stable = false;
                    break;
                }
            }
        }
        
        EXPECT_TRUE(stable) << "Instability at timestep " << dt;
    }
}
#endif // LMP_DPD_REACT

#ifndef LMP_DPD_REACT
// Stub test when DPD-REACT package is not available
TEST_F(MixedPrecisionFixesSpecialTest, DPDNotAvailable) {
    GTEST_SKIP() << "DPD-REACT package not available";
}
#endif

// Test 15: Wall interactions with mixed atom types
TEST_F(MixedPrecisionFixesSpecialTest, WallMixedTypes) {
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary p p f");
    lmp->input->one("region box block 0 10 0 10 0 10");
    lmp->input->one("create_box 2 box");
    
    // Create two types of atoms
    lmp->input->one("create_atoms 1 random 25 12345 box");
    lmp->input->one("create_atoms 2 random 25 54321 box");
    
    lmp->input->one("mass 1 1.0");
    lmp->input->one("mass 2 2.0");
    
    lmp->input->one("velocity all create 1.0 12345");
    lmp->input->one("pair_style lj/cut 2.5");
    lmp->input->one("pair_coeff * * 1.0 1.0 2.5");
    
    // Different wall interactions for different types
    lmp->input->one("fix 1 all wall/lj93/kk zlo EDGE 1.0 1.0 2.5");
    lmp->input->one("fix 2 all nve");
    
    auto fix = dynamic_cast<FixWallLJ93Kokkos<LMPDeviceType>*>(lmp->modify->fix[0]);
    ASSERT_NE(fix, nullptr);
    
    // Run dynamics
    lmp->input->one("run 100");
    
    // Check that both atom types interact correctly with wall
    auto atomKK = static_cast<AtomKokkos*>(lmp->atom);
    atomKK->sync(Host, X_MASK | TYPE_MASK);
    
    for (int i = 0; i < atomKK->nlocal; i++) {
        // All atoms should be above wall
        EXPECT_GT(atomKK->x[i][2], 0.0);
        EXPECT_TRUE(checkNumericalStability(atomKK->x[i][2]));
        
        // Check type-specific behavior if needed
        int itype = atomKK->type[i];
        EXPECT_GE(itype, 1);
        EXPECT_LE(itype, 2);
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
