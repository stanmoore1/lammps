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

// Regression tests for style settings surviving binary restart files.
// Each test writes a restart, reads it back, writes a second restart,
// and requires the two files to be byte-identical: any style setting
// that is parsed in settings() or coeff() but not stored and restored
// by the restart code shows up as a difference in the second file.

#include "lammps.h"

#include "atom.h"
#include "fix.h"
#include "info.h"
#include "library.h"
#include "modify.h"
#include "utils.h"

#include "../testing/core.h"
#include "../testing/utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

using namespace LAMMPS_NS;

bool verbose = false;

class RestartSettingsTest : public LAMMPSTest {
protected:
    void SetUp() override
    {
        testbinary = "RestartSettingsTest";
        LAMMPSTest::SetUp();
    }

    static std::vector<char> read_binary(const std::string &filename)
    {
        std::ifstream in(filename, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    // the IDs of internal fixes created by pair styles embed a per-process
    // instance counter that does not reset on the clear command; mask the
    // counter digits so they do not fail the round-trip comparison

    static void mask_instance_ids(std::vector<char> &buf)
    {
        const std::string tag("NEIGH_HISTORY_HH");
        auto it = buf.begin();
        while ((it = std::search(it, buf.end(), tag.begin(), tag.end())) != buf.end()) {
            it += tag.size();
            while ((it != buf.end()) && isdigit(*it)) *it++ = '#';
        }
    }

    // write -> read -> write round trip; both files must be identical

    void roundtrip_check()
    {
        BEGIN_HIDE_OUTPUT();
        command("run 0 post no");
        command("write_restart roundtrip_a.restart");
        command("clear");
        command("read_restart roundtrip_a.restart");
        command("run 0 post no");
        command("write_restart roundtrip_b.restart");
        END_HIDE_OUTPUT();

        auto a = read_binary("roundtrip_a.restart");
        auto b = read_binary("roundtrip_b.restart");
        mask_instance_ids(a);
        mask_instance_ids(b);
        ASSERT_GT(a.size(), 0);
        EXPECT_EQ(a, b) << "restart file changed in write/read/write round trip: "
                           "some style setting is not preserved";

        delete_file("roundtrip_a.restart");
        delete_file("roundtrip_b.restart");
    }
};

TEST_F(RestartSettingsTest, granular_limit_damping)
{
    if (!Info::has_package("GRANULAR")) GTEST_SKIP();
    // fix neigh/history/kk requires "newton off" for the exchange communication
    if (lmp->suffix_enable) GTEST_SKIP() << "granular pair styles need newton off with KOKKOS";

    BEGIN_HIDE_OUTPUT();
    command("units si");
    command("atom_style sphere");
    command("boundary f f f");
    command("region box block 0 1 0 1 0 1 units box");
    command("create_box 1 box");
    command("create_atoms 1 single 0.5 0.5 0.5 units box");
    command("create_atoms 1 single 0.5 0.5 0.595 units box");
    command("set group all diameter 0.1 density 2500");
    command("pair_style gran/hooke/history 200000 NULL 50 NULL 0.5 1 limit_damping");
    command("pair_coeff * *");
    command("comm_modify vel yes");
    END_HIDE_OUTPUT();
    roundtrip_check();

    // behavioral check that the flag itself survived: for an unloading
    // contact with dominant damping the limited normal force is clamped
    // to zero, while a lost limit_damping flag gives an attractive force

    BEGIN_HIDE_OUTPUT();
    command("velocity all set 0.0 0.0 0.0");
    command("group upper id 2");
    command("velocity upper set 0.0 0.0 1000.0");
    command("run 0 post no");
    END_HIDE_OUTPUT();
    for (int i = 0; i < lmp->atom->nlocal; i++)
        if (lmp->atom->tag[i] == 2) EXPECT_GE(lmp->atom->f[i][2], 0.0);
}

TEST_F(RestartSettingsTest, dsmc_settings)
{
    if (!Info::has_package("MC")) GTEST_SKIP();

    BEGIN_HIDE_OUTPUT();
    command("units si");
    command("boundary f f f");
    command("region box block 0 1e-5 0 1e-5 0 1e-5 units box");
    command("create_box 1 box");
    command("create_atoms 1 single 5e-6 5e-6 5e-6 units box");
    command("mass 1 6.63e-26");
    // all six settings must survive the restart:
    // max_cell_size seed weighting T_ref recompute_stride nsamples
    command("pair_style dsmc 5e-6 12345 10 300.0 10 100");
    command("pair_coeff * * 4.0e-10");
    command("timestep 1e-8");
    END_HIDE_OUTPUT();
    roundtrip_check();
}

TEST_F(RestartSettingsTest, eff_cut_settings)
{
    if (!Info::has_package("EFF")) GTEST_SKIP();

    // the KOKKOS package requires a Kokkos-enabled atom style, and there is
    // no accelerated version of atom style electron
    if (lmp->kokkos && !info->has_style("atom", "electron/kk"))
        GTEST_SKIP() << "atom style electron has no KOKKOS version";

    BEGIN_HIDE_OUTPUT();
    command("units electron");
    command("atom_style electron");
    command("boundary f f f");
    command("region box block -10 10 -10 10 -10 10 units box");
    command("create_box 2 box");
    // one nucleus (spin 0, charge 1) and one electron (spin 1) via data-free setup
    command("create_atoms 1 single 0.0 0.0 0.0 units box");
    command("create_atoms 2 single 0.5 0.0 0.0 units box");
    command("mass * 1.0");
    command("set type 1 charge 1.0");
    command("set type 2 charge -1.0");
    command("set type 2 spin/electron 1");
    command("set type 2 radius/electron 1.0");
    command("pair_style eff/cut 8.0 limit/eradius pressure/evirials");
    command("pair_coeff * *");
    END_HIDE_OUTPUT();
    roundtrip_check();
}

// the global state of a fix must be restored when the restart file was written
// with an accelerated variant of the fix style and read with the plain name, and
// the other way around.  An explicit accelerated style name (e.g. "nvt/kk" with
// suffixes off, or "nvt/omp") is stored in the restart file as is.

TEST_F(RestartSettingsTest, fix_state_accelerator_variant)
{
    // an explicit "nvt/kk" is only stored as is with suffixes off, which the test
    // switches temporarily when running with "-sf kk"

    std::string accel;
    if (lmp->kokkos)
        accel = "nvt/kk";
    else if (info->has_style("fix", "nvt/omp"))
        accel = "nvt/omp";
    else
        GTEST_SKIP() << "no accelerated variant of fix nvt available";

    // suffixes are only toggled when a suffix was given on the command line
    const bool has_suffix = (lmp->suffix != nullptr);

    auto setup = [&]() {
        command("units lj");
        command("atom_style atomic");
        command("lattice fcc 0.8442");
        command("region box block 0 3 0 3 0 3");
        command("create_box 1 box");
        command("create_atoms 1 box");
        command("mass 1 1.0");
        command("velocity all create 3.0 87287 loop geom");
        command("pair_style lj/cut 2.5");
        command("pair_coeff * * 1.0 1.0");
    };
    auto restored_scalar = [&](const std::string &style, bool suffix_on) {
        BEGIN_HIDE_OUTPUT();
        // the run style is created by "clear", so suffixes must be on before it
        if (has_suffix) command("suffix on");
        command("clear");
        command("read_restart accel_variant.restart");
        command("pair_style lj/cut 2.5");
        command("pair_coeff * * 1.0 1.0");
        if (has_suffix && !suffix_on) command("suffix off");
        END_HIDE_OUTPUT();
        auto out = CAPTURE_OUTPUT([&]() {
            command("fix 1 all " + style + " temp 1.0 1.0 0.5");
        });
        EXPECT_THAT(out, testing::HasSubstr("Resetting global fix info from restart file"));
        BEGIN_HIDE_OUTPUT();
        command("run 0 post no");
        END_HIDE_OUTPUT();
        return lmp->modify->get_fix_by_id("1")->compute_scalar();
    };

    // written with the explicit accelerated name, read with the plain name

    BEGIN_HIDE_OUTPUT();
    setup();
    if (has_suffix) command("suffix off");
    command("fix 1 all " + accel + " temp 1.0 1.0 0.5");
    command("run 20 post no");
    command("write_restart accel_variant.restart");
    END_HIDE_OUTPUT();
    double ref = lmp->modify->get_fix_by_id("1")->compute_scalar();
    ASSERT_NE(ref, 0.0);
    EXPECT_DOUBLE_EQ(restored_scalar("nvt", true), ref);

    // written with the plain name, read with the explicit accelerated name

    BEGIN_HIDE_OUTPUT();
    if (has_suffix) command("suffix on");
    command("clear");
    setup();
    command("fix 1 all nvt temp 1.0 1.0 0.5");
    command("run 20 post no");
    command("write_restart accel_variant.restart");
    END_HIDE_OUTPUT();
    ref = lmp->modify->get_fix_by_id("1")->compute_scalar();
    ASSERT_NE(ref, 0.0);
    EXPECT_DOUBLE_EQ(restored_scalar(accel, false), ref);

    if (has_suffix) {
        BEGIN_HIDE_OUTPUT();
        command("suffix on");
        END_HIDE_OUTPUT();
    }
    delete_file("accel_variant.restart");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    // handle arguments passed via environment variable
    if (const char *var = getenv("TEST_ARGS")) {
        std::vector<std::string> env = LAMMPS_NS::utils::split_words(var);
        for (auto arg : env) {
            if (arg == "-v") {
                verbose = true;
            }
        }
    }
    if ((argc > 1) && (strcmp(argv[1], "-v") == 0)) verbose = true;

    int rv = RUN_ALL_TESTS();

    // finalize the KOKKOS package explicitly: otherwise Kokkos is torn down by
    // static destructors at program exit, leading to segfaults in some cases
    // same workaround as the force-style and FFT3d test drivers

    lammps_kokkos_finalize();

    MPI_Finalize();
    return rv;
}
