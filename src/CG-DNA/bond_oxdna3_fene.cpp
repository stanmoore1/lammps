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
   Contributing author: Oliver Henrich (University of Strathclyde, Glasgow)
------------------------------------------------------------------------- */

#include "bond_oxdna3_fene.h"
#include "constants_oxdna.h"
#include "nucleotide_oxdna.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "potential_file_reader.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   set coeffs - introduces new function to handle KOKKOS compatibility.
   Vanilla oxdna3 "coeff" literally just calls this "coeff_oxdna3_common"
   function. The structure here avoids messy inheritance issues in KOKKOS
   by not calling "BondOxdna3FENE::coeff" directly. We can also avoid
   code duplication of coeff within KOKKOS using this approach.

   "coeff_oxdna3_common" is static and takes a pointer to the base class
   BondOxdnaFene, which means it can be called from both the vanilla and
   KOKKOS versions.
   Can't use "coeff" directly since it is non-static - calling it would
   require an instance of the BondOxdna3Fene class, which is fine for vanilla
   but not for KOKKOS as we don't want KOKKOS to be a child class of BondOxdna3Fene.
------------------------------------------------------------------------- */
void BondOxdna3Fene::coeff_oxdna3_common(BondOxdnaFene *bond, int narg, char **arg)
{
  if (narg != 2) bond->error->all(FLERR, "Incorrect args for bond coefficients in oxdna3/fene, use potential file" + utils::errorurl(21));
  if (!bond->allocated) bond->allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, bond->atom->nbondtypes, ilo, ihi, bond->error);

  int n = bond->atom->ntypes;
  if (n > 4) bond->error->all(FLERR, "bond oxdna3/fene does not support more than 4 atom types for A, C, G and T");

  for (int i = 0; i <= n; i++) {
    for (int j = 0; j <= n; j++) {
      for (int k = 0; k <= n; k++) {
        for (int l = 0; l <= n; l++) {
          bond->Delta[ilo][i][j][k][l] = 0.0;
          bond->r0[ilo][i][j][k][l] = 0.0;
        }
      }
    }
  }

  if (bond->comm->me == 0) { // read values from potential file
    PotentialFileReader reader(bond->lmp, arg[1], "oxdna3 potential", " (fene)");
    reader.set_bufsize(65336);
    char * line;
    std::string iloc, potential_name;

    while ((line = reader.next_line())) {
      try {
        ValueTokenizer values(line);
        iloc = values.next_string();
        potential_name = values.next_string();
        if (iloc == arg[0] && potential_name == "fene") {
          bond->k[ilo] = values.next_double();
          for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
              for (int k = 1; k <= n; k++) {
                for (int l = 1; l <= n; l++) {
                bond->Delta[ilo][i][j][k][l] = values.next_double();
                bond->Delta[ilo][i][j][k][0] += bond->Delta[ilo][i][j][k][l];
                bond->Delta[ilo][0][j][k][l] += bond->Delta[ilo][i][j][k][l];
                bond->Delta[ilo][0][j][k][0] += bond->Delta[ilo][i][j][k][l];
                }
              }
            }
          }
          for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
              for (int k = 1; k <= n; k++) {
                for (int l = 1; l <= n; l++) {
                  bond->r0[ilo][i][j][k][l] = values.next_double();
                  bond->r0[ilo][i][j][k][0] += bond->r0[ilo][i][j][k][l];
                  bond->r0[ilo][0][j][k][l] += bond->r0[ilo][i][j][k][l];
                  bond->r0[ilo][0][j][k][0] += bond->r0[ilo][i][j][k][l];
                }
              }
            }
          }
          break;
        } else continue;
      } catch (std::exception &e) {
        bond->error->one(FLERR, "Problem parsing oxdna3 potential file: {}", e.what());
      }
    }
    if ((iloc != arg[0]) || (potential_name != "fene"))
      bond->error->one(FLERR, "No corresponding fene potential found in file {} for bond type {}",
                 arg[1], arg[0]);

    // calculate sequence-averaged parameters for terminal base step j-k
    for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= n; j++) {
        for (int k = 1; k <= n; k++) {
          bond->Delta[ilo][i][j][k][0] /= n;
          bond->r0[ilo][i][j][k][0] /= n;
        }
      }
    }
    for (int j = 1; j <= n; j++) {
      for (int k = 1; k <= n; k++) {
        for (int l = 1; l <= n; l++) {
          bond->Delta[ilo][0][j][k][l] /= n;
          bond->r0[ilo][0][j][k][l] /= n;
        }
      }
    }
    for (int j = 1; j <= n; j++) {
      for (int k = 1; k <= n; k++) {
        bond->Delta[ilo][0][j][k][0] /= pow(n,2);
        bond->r0[ilo][0][j][k][0] /= pow(n,2);
      }
    }

  }

  // communicate parameters for bond type ilo
  MPI_Bcast(&bond->k[ilo], 1, MPI_DOUBLE, 0, bond->world);
  MPI_Bcast(&bond->Delta[ilo][0][0][0][0], 625, MPI_DOUBLE, 0, bond->world);
  MPI_Bcast(&bond->r0[ilo][0][0][0][0], 625, MPI_DOUBLE, 0, bond->world);

  // set parameters for all other bond types
  int count = 0;
  for (int ib = ilo; ib <= ihi; ib++) {
    bond->k[ib] = bond->k[ilo];
    for (int i = 0; i <= n; i++) {
      for (int j = 0; j <= n; j++) {
        for (int k = 0; k <= n; k++) {
          for (int l = 0; l <= n; l++) {
            bond->Delta[ib][i][j][k][l] = bond->Delta[ilo][i][j][k][l];
            bond->r0[ib][i][j][k][l] = bond->r0[ilo][i][j][k][l];
          }
        }
      }
    }
    bond->setflag[ib] = 1;
    count++;
  }

  if (count == 0) bond->error->all(FLERR, "Incorrect args for bond coefficients in oxdna3/fene" + utils::errorurl(21));
}

void BondOxdna3Fene::coeff(int narg, char **arg) { coeff_oxdna3_common(this, narg, arg); }
