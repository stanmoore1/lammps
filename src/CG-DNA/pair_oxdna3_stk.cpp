// clang-format off
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

#include "pair_oxdna3_stk.h"

#include "atom.h"
#include "comm.h"
#include "constants_oxdna.h"
#include "error.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "mf_oxdna.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "potential_file_reader.h"

#include <cmath>
#include <cstring>
#include <cassert>

using namespace LAMMPS_NS;
using namespace MFOxdna;

/* ---------------------------------------------------------------------- */

PairOxdna3Stk::PairOxdna3Stk(LAMMPS *lmp) : PairOxdnaStk(lmp)
{
  // sequence-specific stacking strength
  // A:0 C:1 G:2 T:3, 3'- [i][j] -5'

  eta_st[0][0] = 1.0227941724360874;
  eta_st[1][0] = 1.0842007228784680;
  eta_st[2][0] = 0.9108059216458568;
  eta_st[3][0] = 1.0280543774734380;

  eta_st[0][1] = 0.9606488536265920;
  eta_st[1][1] = 0.9104663871513519;
  eta_st[2][1] = 0.9242711984623542;
  eta_st[3][1] = 0.9255424203161666;

  eta_st[0][2] = 1.0937706497793123;
  eta_st[1][2] = 1.0588039315598938;
  eta_st[2][2] = 0.9919972586730905;
  eta_st[3][2] = 1.1103321115392364;

  eta_st[0][3] = 0.9225701575742983;
  eta_st[1][3] = 1.0040233335569662;
  eta_st[2][3] = 0.9403493412105425;
  eta_st[3][3] = 0.9109921004070870;

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairOxdna3Stk::coeff(int narg, char **arg)
{
  int count;

  if (narg != 4) error->all(FLERR,"Incorrect args for pair coefficients in oxdna3/stk, use potential file" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi,nlo,nhi,imod4,jmod4,kmod4;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  assert((ilo == jlo) & (ihi == jhi));
  nlo = ilo;
  nhi = ihi;

  // stacking interaction
  count = 0;

  double T, epsilon_st_one, xi_st_one, kappa_st_one, a_st_one, b_st_lo_one, b_st_hi_one;
  double cut_st_0_one, cut_st_c_one, cut_st_lo_one, cut_st_hi_one;
  double cut_st_lc_one, cut_st_hc_one, tmp, shift_st_one;

  double a_st4_one, theta_st4_0_one, dtheta_st4_ast_one;
  double b_st4_one, dtheta_st4_c_one;

  double a_st5_one, theta_st5_0_one, dtheta_st5_ast_one;
  double b_st5_one, dtheta_st5_c_one;

  double a_st6_one, theta_st6_0_one, dtheta_st6_ast_one;
  double b_st6_one, dtheta_st6_c_one;

  double a_st1_one, cosphi_st1_ast_one, b_st1_one, cosphi_st1_c_one;
  double a_st2_one, cosphi_st2_ast_one, b_st2_one, cosphi_st2_c_one;

  seqdepflag = 1; // default sequence-dependent stacking strength in oxDNA3

  T = utils::numeric(FLERR,arg[2],false,lmp);

  cut_st_0[0][0][0][0] = 0.0;
  cut_st_c[0][0][0][0] = 0.0;
  cut_st_lo[0][0][0][0] = 0.0;
  cut_st_hi[0][0][0][0] = 0.0;
  a_st4[0][0][0][0] = 0.0;
  dtheta_st4_ast[0][0][0][0] = 0.0; 

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, arg[3], "oxdna3 potential", " (stk)");
    reader.set_bufsize(65336);
    char * line;
    std::string iloc, jloc, potential_name;

    while ((line = reader.next_line())) {
      try {
        ValueTokenizer values(line);
        iloc = values.next_string();
        jloc = values.next_string();
        potential_name = values.next_string();
        if (iloc == arg[0] && jloc == arg[1] && potential_name == "stk") {

          xi_st_one = values.next_double(); 
          kappa_st_one = values.next_double();
          a_st_one = values.next_double();

          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  cut_st_0[i][j][k][l] = values.next_double();
                  cut_st_0[0][0][0][0] += cut_st_0[i][j][k][l];
                }
              }
            }
          }
          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  cut_st_c[i][j][k][l] = values.next_double();
                  cut_st_c[0][0][0][0] += cut_st_c[i][j][k][l];
                }
              }
            }
          }
          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  cut_st_lo[i][j][k][l] = values.next_double();
                  cut_st_lo[0][0][0][0] += cut_st_lo[i][j][k][l];
                }
              }
            }
          }
          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  cut_st_hi[i][j][k][l] = values.next_double();
                  cut_st_hi[0][0][0][0] += cut_st_hi[i][j][k][l];
                }
              }
            }
          }
          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  a_st4[i][j][k][l] = values.next_double();
                  a_st4[0][0][0][0] += a_st4[i][j][k][l]; 
                }
              }
            }
          }

          theta_st4_0_one = values.next_double();

          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  dtheta_st4_ast[i][j][k][l] = values.next_double();
                  dtheta_st4_ast[0][0][0][0] += dtheta_st4_ast[i][j][k][l];
                }
              }
            }
          }

          a_st5_one = values.next_double();
          theta_st5_0_one = values.next_double();
          dtheta_st5_ast_one = values.next_double();
          a_st6_one = values.next_double();
          theta_st6_0_one = values.next_double();
          dtheta_st6_ast_one = values.next_double();
          a_st1_one = values.next_double();
          cosphi_st1_ast_one = values.next_double();
          a_st2_one = values.next_double();
          cosphi_st2_ast_one = values.next_double();

          break;
        } else continue;
      } catch (std::exception &e) {
        error->one(FLERR, "Problem parsing oxDNA3 potential file: {}", e.what());
      }
    }
    if ((iloc != arg[0]) || (jloc != arg[1]) || (potential_name != "stk"))
      error->one(FLERR, "No corresponding stk potential found in file {} for pair type {} {}",
                 arg[4], arg[0], arg[1]);

    epsilon_st_one = stacking_strength(xi_st_one, kappa_st_one, T);

    // calculate sequence-averaged parameters 
    cut_st_0[0][0][0][0] /= pow(nhi,4);
    cut_st_c[0][0][0][0] /= pow(nhi,4);
    cut_st_lo[0][0][0][0] /= pow(nhi,4);
    cut_st_hi[0][0][0][0] /= pow(nhi,4);
    a_st4[0][0][0][0] /= pow(nhi,4); 
    dtheta_st4_ast[0][0][0][0] /= pow(nhi,4);

    // assign sequence-averaged parameters to terminal bases j
    for (int j = 1; j <= nhi; j++) {
      for (int k = 1; k <= nhi; k++) {
        for (int l = 0; l <= nhi; l++) {
          cut_st_0[0][j][k][l] = cut_st_0[0][0][0][0];
          cut_st_c[0][j][k][l] = cut_st_c[0][0][0][0];
          cut_st_lo[0][j][k][l] = cut_st_lo[0][0][0][0];
          cut_st_hi[0][j][k][l] = cut_st_hi[0][0][0][0];
          a_st4[0][j][k][l] = a_st4[0][0][0][0]; 
          dtheta_st4_ast[0][j][k][l] = dtheta_st4_ast[0][0][0][0];
        }
      }
    }
    // assign sequence-averaged parameters to terminal bases k
    for (int i = 0; i <= nhi; i++) {
      for (int j = 1; j <= nhi; j++) {
        for (int k = 1; k <= nhi; k++) {
          cut_st_0[i][j][k][0] = cut_st_0[0][0][0][0];
          cut_st_c[i][j][k][0] = cut_st_c[0][0][0][0];
          cut_st_lo[i][j][k][0] = cut_st_lo[0][0][0][0];
          cut_st_hi[i][j][k][0] = cut_st_hi[0][0][0][0];
          a_st4[i][j][k][0] = a_st4[0][0][0][0]; 
          dtheta_st4_ast[i][j][k][0] = dtheta_st4_ast[0][0][0][0];
        }
      }
    }

  }

  MPI_Bcast(&epsilon_st_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&a_st_one, 1, MPI_DOUBLE, 0, world);

  MPI_Bcast(&cut_st_0[0][0][0][0], 625, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cut_st_c[0][0][0][0], 625, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cut_st_lo[0][0][0][0], 625, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cut_st_hi[0][0][0][0], 625, MPI_DOUBLE, 0, world);
  MPI_Bcast(&a_st4[0][0][0][0], 625, MPI_DOUBLE, 0, world);

  MPI_Bcast(&theta_st4_0_one, 1, MPI_DOUBLE, 0, world);

  MPI_Bcast(&dtheta_st4_ast[0][0][0][0], 625, MPI_DOUBLE, 0, world);

  MPI_Bcast(&a_st5_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&theta_st5_0_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&dtheta_st5_ast_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&a_st6_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&theta_st6_0_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&dtheta_st6_ast_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&a_st1_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cosphi_st1_ast_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&a_st2_one, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cosphi_st2_ast_one, 1, MPI_DOUBLE, 0, world);

  // smoothing - determined through continuity and differentiability

  // smoothing strength identical for all pairs ij, hence use average tetramer value below
  b_st_lo_one = 2*a_st_one*exp(-a_st_one*(cut_st_lo[0][0][0][0]-cut_st_0[0][0][0][0]))*
      2*a_st_one*exp(-a_st_one*(cut_st_lo[0][0][0][0]-cut_st_0[0][0][0][0]))*
      (1-exp(-a_st_one*(cut_st_lo[0][0][0][0]-cut_st_0[0][0][0][0])))*
      (1-exp(-a_st_one*(cut_st_lo[0][0][0][0]-cut_st_0[0][0][0][0])))/
      (4*((1-exp(-a_st_one*(cut_st_lo[0][0][0][0] -cut_st_0[0][0][0][0])))*
      (1-exp(-a_st_one*(cut_st_lo[0][0][0][0]-cut_st_0[0][0][0][0])))-
      (1-exp(-a_st_one*(cut_st_c[0][0][0][0] -cut_st_0[0][0][0][0])))*
      (1-exp(-a_st_one*(cut_st_c[0][0][0][0]-cut_st_0[0][0][0][0])))));

  // smoothing strength identical for all pairs ij, hence use average tetramer value below
  b_st_hi_one = 2*a_st_one*exp(-a_st_one*(cut_st_hi[0][0][0][0]-cut_st_0[0][0][0][0]))*
      2*a_st_one*exp(-a_st_one*(cut_st_hi[0][0][0][0]-cut_st_0[0][0][0][0]))*
      (1-exp(-a_st_one*(cut_st_hi[0][0][0][0]-cut_st_0[0][0][0][0])))*
      (1-exp(-a_st_one*(cut_st_hi[0][0][0][0]-cut_st_0[0][0][0][0])))/
      (4*((1-exp(-a_st_one*(cut_st_hi[0][0][0][0] -cut_st_0[0][0][0][0])))*
      (1-exp(-a_st_one*(cut_st_hi[0][0][0][0]-cut_st_0[0][0][0][0])))-
      (1-exp(-a_st_one*(cut_st_c[0][0][0][0] -cut_st_0[0][0][0][0])))*
      (1-exp(-a_st_one*(cut_st_c[0][0][0][0]-cut_st_0[0][0][0][0])))));

  b_st5_one = a_st5_one*a_st5_one*dtheta_st5_ast_one*dtheta_st5_ast_one/
      (1-a_st5_one*dtheta_st5_ast_one*dtheta_st5_ast_one);
  dtheta_st5_c_one = 1/(a_st5_one*dtheta_st5_ast_one);

  b_st6_one = a_st6_one*a_st6_one*dtheta_st6_ast_one*dtheta_st6_ast_one/
      (1-a_st6_one*dtheta_st6_ast_one*dtheta_st6_ast_one);
  dtheta_st6_c_one = 1/(a_st6_one*dtheta_st6_ast_one);

  b_st1_one = a_st1_one*a_st1_one*cosphi_st1_ast_one*cosphi_st1_ast_one/
      (1-a_st1_one*cosphi_st1_ast_one*cosphi_st1_ast_one);
  cosphi_st1_c_one = 1/(a_st1_one*cosphi_st1_ast_one);

  b_st2_one = a_st2_one*a_st2_one*cosphi_st2_ast_one*cosphi_st2_ast_one/
      (1-a_st2_one*cosphi_st2_ast_one*cosphi_st2_ast_one);
  cosphi_st2_c_one = 1/(a_st2_one*cosphi_st2_ast_one);


  // parameters, uniform or depending on base step
  for (int i = nlo; i <= nhi; i++) {
    imod4 = i%4;
    if (imod4 == 0) imod4 = 4;

    for (int j = nlo; j <= nhi; j++) {
      jmod4 = j%4;
      if (jmod4 == 0) jmod4 = 4;

      epsilon_st[i][j] = epsilon_st_one;
      if (seqdepflag) epsilon_st[i][j] *= eta_st[imod4-1][jmod4-1];

      a_st[i][j] = a_st_one;
      b_st_lo[i][j] = b_st_lo_one;
      b_st_hi[i][j] = b_st_hi_one;

      theta_st4_0[i][j] = theta_st4_0_one;

      a_st5[i][j] = a_st5_one;
      theta_st5_0[i][j] = theta_st5_0_one;
      dtheta_st5_ast[i][j] = dtheta_st5_ast_one;
      b_st5[i][j] = b_st5_one;
      dtheta_st5_c[i][j] = dtheta_st5_c_one;

      a_st6[i][j] = a_st6_one;
      theta_st6_0[i][j] = theta_st6_0_one;
      dtheta_st6_ast[i][j] = dtheta_st6_ast_one;
      b_st6[i][j] = b_st6_one;
      dtheta_st6_c[i][j] = dtheta_st6_c_one;

      a_st1[i][j] = a_st1_one;
      cosphi_st1_ast[i][j] = cosphi_st1_ast_one;
      b_st1[i][j] = b_st1_one;
      cosphi_st1_c[i][j] = cosphi_st1_c_one;

      a_st2[i][j] = a_st2_one;
      cosphi_st2_ast[i][j] = cosphi_st2_ast_one;
      b_st2[i][j] = b_st2_one;
      cosphi_st2_c[i][j] = cosphi_st2_c_one;

    }
  }

  // parameters depending on tetramer
  for (int i = 0; i <= nhi; i++) {
    for (int j = nlo; j <= nhi; j++) {
      jmod4 = j%4;
      if (jmod4 == 0) jmod4 = 4;

      for (int k = nlo; k <= nhi; k++) {
        kmod4 = k%4;
        if (kmod4 == 0) kmod4 = 4;

        for (int l = 0; l <= nhi; l++) {

          cut_st_lc[i][j][k][l] = cut_st_lo[i][j][k][l] 
                - a_st_one*exp(-a_st_one*(cut_st_lo[i][j][k][l]-cut_st_0[i][j][k][l]))*
                (1-exp(-a_st_one*(cut_st_lo[i][j][k][l]-cut_st_0[i][j][k][l])))/b_st_lo_one;

          cut_st_hc[i][j][k][l] = cut_st_hi[i][j][k][l] 
                - a_st_one*exp(-a_st_one*(cut_st_hi[i][j][k][l]-cut_st_0[i][j][k][l]))*
                (1-exp(-a_st_one*(cut_st_hi[i][j][k][l]-cut_st_0[i][j][k][l])))/b_st_hi_one;

          cutsq_st_hc[i][j][k][l] = cut_st_hc[i][j][k][l]*cut_st_hc[i][j][k][l];

          tmp = 1 - exp(-(cut_st_c[i][j][k][l]-cut_st_0[i][j][k][l]) * a_st_one);
          shift_st[i][j][k][l] = epsilon_st_one * tmp * tmp;
          if (seqdepflag) {
            shift_st[i][j][k][l] *= eta_st[jmod4-1][kmod4-1];
          }

          b_st4[i][j][k][l] = a_st4[i][j][k][l]*a_st4[i][j][k][l]*dtheta_st4_ast[i][j][k][l]*
                dtheta_st4_ast[i][j][k][l]/(1-a_st4[i][j][k][l]*dtheta_st4_ast[i][j][k][l]*dtheta_st4_ast[i][j][k][l]);
          dtheta_st4_c[i][j][k][l] = 1/(a_st4[i][j][k][l]*dtheta_st4_ast[i][j][k][l]);

        }
      }
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients in oxdna3/stk" + utils::errorurl(21));

}
