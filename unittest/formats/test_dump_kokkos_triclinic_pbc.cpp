/* ----------------------------------------------------------------------
   Regression test for GitHub issues #4923 and #4940:
   "dump_modify pbc yes" hands plain host copies of x, v and image to
   Domain::pbc() and, for a triclinic box, to Domain::x2lamda() and
   Domain::lamda2x().  The KOKKOS versions of those three functions work on
   the Kokkos views instead, so they remapped the real atom data and left the
   copies that are written to the dump file untouched.

   The test runs the same input twice in the same process, once without and
   once with the KOKKOS package, and compares the two dump files.  The atoms
   have to have moved out of the box before the dump for the difference to
   show up, so a short run precedes it.
------------------------------------------------------------------------- */

#include "../testing/core.h"
#include "../testing/utils.h"
#include "info.h"
#include "utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <mpi.h>

#include <string>
#include <vector>

using LAMMPS_NS::Info;
using LAMMPS_NS::LAMMPS;
using LAMMPS_NS::utils::split_words;

namespace {

// one line of the dump file: atom id and its coordinates

struct DumpAtom {
    int id;
    double x, y, z;
};

// read the atom block of a dump file, sorted by atom id

std::vector<DumpAtom> read_dump(const std::string &path)
{
    std::vector<DumpAtom> atoms;
    auto lines = read_lines(path);
    std::size_t i = 0;
    for (; i < lines.size(); ++i)
        if (lines[i].rfind("ITEM: ATOMS", 0) == 0) break;
    for (++i; i < lines.size(); ++i) {
        auto words = split_words(lines[i]);
        if (words.size() < 4) continue;
        atoms.push_back({std::stoi(words[0]), std::stod(words[1]), std::stod(words[2]),
                         std::stod(words[3])});
    }
    std::sort(atoms.begin(), atoms.end(),
              [](const DumpAtom &a, const DumpAtom &b) { return a.id < b.id; });
    return atoms;
}

// run the same triclinic input with or without the KOKKOS package and return
// the coordinates that "dump_modify pbc yes" wrote

std::vector<DumpAtom> run_input(bool with_kokkos, const std::string &dumpfile)
{
    LAMMPS::argv args = {"LAMMPS_test", "-log", "none", "-echo", "none", "-screen", "none"};
    if (with_kokkos) {
        for (const auto &arg : {"-k", "on", "t", "2", "-sf", "kk"})
            args.emplace_back(arg);
    }

    delete_file(dumpfile);

    ::testing::internal::CaptureStdout();
    auto *lmp = new LAMMPS(args, MPI_COMM_WORLD);
    lmp->input->one("units lj");
    lmp->input->one("atom_style atomic");
    lmp->input->one("boundary p p p");
    lmp->input->one("lattice fcc 0.8442");
    lmp->input->one("region box prism 0 4 0 4 0 4 1.2 0.8 0.5");
    lmp->input->one("create_box 1 box");
    lmp->input->one("create_atoms 1 box");
    lmp->input->one("mass 1 1.0");
    lmp->input->one("pair_style lj/cut 2.5");
    lmp->input->one("pair_coeff 1 1 1.0 1.0 2.5");
    lmp->input->one("neighbor 0.3 bin");
    lmp->input->one("velocity all create 2.0 12345");
    lmp->input->one("fix 1 all nve");

    // let the atoms drift out of the box, then dump with pbc yes

    lmp->input->one("run 20 post no");
    lmp->input->one(fmt::format("dump d all custom 1 {} id x y z", dumpfile));
    lmp->input->one("dump_modify d pbc yes");
    lmp->input->one("run 0 post no");
    lmp->input->one("undump d");
    delete lmp;
    (void) ::testing::internal::GetCapturedStdout();

    return read_dump(dumpfile);
}

}    // namespace

TEST(DumpKokkosTriclinicPbc, pbc_yes_matches_the_plain_styles)
{
    if (!Info::has_package("KOKKOS")) GTEST_SKIP() << "KOKKOS package not available";
    if (Info::has_accelerator_feature("KOKKOS", "api", "cuda") ||
        Info::has_accelerator_feature("KOKKOS", "api", "hip") ||
        Info::has_accelerator_feature("KOKKOS", "api", "sycl"))
        GTEST_SKIP() << "KOKKOS GPU build needs a GPU, use a CPU-only preset";

    auto plain  = run_input(false, "dump_tric_pbc_plain.melt");
    auto kokkos = run_input(true, "dump_tric_pbc_kokkos.melt");

    ASSERT_FALSE(plain.empty());
    ASSERT_EQ(plain.size(), kokkos.size());

    const double tol = 1.0e-10;
    for (std::size_t i = 0; i < plain.size(); ++i) {
        ASSERT_EQ(plain[i].id, kokkos[i].id);
        EXPECT_NEAR(plain[i].x, kokkos[i].x, tol) << "atom " << plain[i].id;
        EXPECT_NEAR(plain[i].y, kokkos[i].y, tol) << "atom " << plain[i].id;
        EXPECT_NEAR(plain[i].z, kokkos[i].z, tol) << "atom " << plain[i].id;
    }

    delete_file("dump_tric_pbc_plain.melt");
    delete_file("dump_tric_pbc_kokkos.melt");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleMock(&argc, argv);
    const int rv = RUN_ALL_TESTS();
    MPI_Finalize();
    return rv;
}
