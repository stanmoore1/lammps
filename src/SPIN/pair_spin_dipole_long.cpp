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

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)
                         Stan Moore (SNL)

   Please cite the related publication:
   Tranchida, J., Plimpton, S. J., Thibaudeau, P., & Thompson, A. P. (2018).
   Massively parallel symplectic algorithm for coupled magnetic spin dynamics
   and molecular dynamics. Journal of Computational Physics.
------------------------------------------------------------------------- */

#include "pair_spin_dipole_long.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "info.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace EwaldConst;

/* ---------------------------------------------------------------------- */

PairSpinDipoleLong::PairSpinDipoleLong(LAMMPS *lmp) : PairSpinDipoleCut(lmp)
{
  ewaldflag = pppmflag = spinflag = 1;
}

/* ---------------------------------------------------------------------- */

void PairSpinDipoleLong::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rinv,r2inv,rsq,local_cut2,evdwl,ecoul;
  double grij,expm2,pre1,pre2,pre3,b1,b2,b3;
  double xi[3],del[3],spi[4],spj[4],fi[3],fmi[3];

  ev_init(eflag,vflag);
  evdwl = ecoul = 0.0;

  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double **x = atom->x;
  double **f = atom->f;
  double **fm = atom->fm;
  double **sp = atom->sp;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // checking size of emag

  if (nlocal_max < nlocal) {
    nlocal_max = nlocal;
    memory->grow(emag,nlocal_max,"pair/spin:emag");
  }

  pre1 = 2.0 * g_ewald / MY_PIS;
  pre2 = 4.0 * pow(g_ewald,3.0) / MY_PIS;
  pre3 = 8.0 * pow(g_ewald,5.0) / MY_PIS;

  // loop over atoms and their neighbors

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xi[0] = x[i][0];
    xi[1] = x[i][1];
    xi[2] = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    spi[0] = sp[i][0];
    spi[1] = sp[i][1];
    spi[2] = sp[i][2];
    spi[3] = sp[i][3];
    emag[i] = 0.0;
    itype = type[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];
      spj[3] = sp[j][3];

      evdwl = 0.0;
      fi[0] = fi[1] = fi[2] = 0.0;
      fmi[0] = fmi[1] = fmi[2] = 0.0;

      del[0] = xi[0] - x[j][0];
      del[1] = xi[1] - x[j][1];
      del[2] = xi[2] - x[j][2];
      rsq = del[0]*del[0] + del[1]*del[1] + del[2]*del[2];

      local_cut2 = cut_spin_long[itype][jtype]*cut_spin_long[itype][jtype];

      // compute Ewald-corrected dipolar interaction

      if (rsq < local_cut2) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

        double r = sqrt(rsq);
        grij = g_ewald * r;
        expm2 = exp(-grij*grij);

        double t = 1.0 / (1.0 + EWALD_P*grij);
        double erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;

        b1 = (erfc*rinv + pre1*expm2) * r2inv;
        b2 = (3.0*b1 + pre2*expm2) * r2inv;
        b3 = (5.0*b2 + pre3*expm2) * r2inv;

        compute_dipolar_long(i,j,del,fmi,spi,spj,b1,b2);

        if (lattice_flag)
          compute_dipolar_mech_long(i,j,del,fi,spi,spj,b2,b3);

        if (eflag) {
          evdwl = -(spi[0]*fmi[0] + spi[1]*fmi[1] + spi[2]*fmi[2]);
          evdwl *= 0.5*hbar;
          emag[i] += evdwl;
        } else evdwl = 0.0;

        f[i][0] += fi[0];
        f[i][1] += fi[1];
        f[i][2] += fi[2];
        if (newton_pair || j < nlocal) {
          f[j][0] -= fi[0];
          f[j][1] -= fi[1];
          f[j][2] -= fi[2];
        }
        fm[i][0] += fmi[0];
        fm[i][1] += fmi[1];
        fm[i][2] += fmi[2];

        double rij[3];
        rij[0] = x[j][0] - xi[0];
        rij[1] = x[j][1] - xi[1];
        rij[2] = x[j][2] - xi[2];
        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
            evdwl,ecoul,fi[0],fi[1],fi[2],rij[0],rij[1],rij[2]);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   update the pair interaction fmi acting on the spin ii
------------------------------------------------------------------------- */

void PairSpinDipoleLong::compute_single_pair(int ii, double fmi[3])
{
  int j,jnum,itype,jtype,ntypes;
  int *jlist,*numneigh,**firstneigh;
  double rsq,rinv,r2inv,local_cut2;
  double grij,expm2,pre1,pre2,b1,b2;
  double xi[3],del[3],spi[4],spj[4];

  int k,locflag;
  int *type = atom->type;
  double **x = atom->x;
  double **sp = atom->sp;

  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // check if interaction applies to type of ii

  itype = type[ii];
  ntypes = atom->ntypes;
  locflag = 0;
  k = 1;
  while (k <= ntypes) {
    if (k <= itype) {
      if (setflag[k][itype] == 1) {
        locflag = 1;
        break;
      }
      k++;
    } else if (k > itype) {
      if (setflag[itype][k] == 1) {
        locflag = 1;
        break;
      }
      k++;
    } else error->all(FLERR,"Wrong type number");
  }

  if (locflag == 1) {

    xi[0] = x[ii][0];
    xi[1] = x[ii][1];
    xi[2] = x[ii][2];
    spi[0] = sp[ii][0];
    spi[1] = sp[ii][1];
    spi[2] = sp[ii][2];
    spi[3] = sp[ii][3];
    jlist = firstneigh[ii];
    jnum = numneigh[ii];

    pre1 = 2.0 * g_ewald / MY_PIS;
    pre2 = 4.0 * pow(g_ewald,3.0) / MY_PIS;

    for (int jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];
      spj[3] = sp[j][3];

      del[0] = xi[0] - x[j][0];
      del[1] = xi[1] - x[j][1];
      del[2] = xi[2] - x[j][2];
      rsq = del[0]*del[0] + del[1]*del[1] + del[2]*del[2];

      local_cut2 = cut_spin_long[itype][jtype]*cut_spin_long[itype][jtype];

      if (rsq < local_cut2) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

        double r = sqrt(rsq);
        grij = g_ewald * r;
        expm2 = exp(-grij*grij);

        double t = 1.0 / (1.0 + EWALD_P*grij);
        double erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;

        b1 = (erfc*rinv + pre1*expm2) * r2inv;
        b2 = (3.0*b1 + pre2*expm2) * r2inv;

        compute_dipolar_long(ii,j,del,fmi,spi,spj,b1,b2);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute Ewald-corrected spin precession field at i due to spin j
   del = xi - xj, b1 = erfc-corrected 1/r^3, b2 = erfc-corrected 3/r^5
------------------------------------------------------------------------- */

void PairSpinDipoleLong::compute_dipolar_long(int /* i */, int /* j */,
    double del[3], double fmi[3], double spi[4], double spj[4],
    double b1, double b2)
{
  double sjdotr,gigjpre;

  sjdotr = spj[0]*del[0] + spj[1]*del[1] + spj[2]*del[2];
  gigjpre = mub2mu0hbinv * spi[3] * spj[3];

  fmi[0] += gigjpre * (b2*sjdotr*del[0] - b1*spj[0]);
  fmi[1] += gigjpre * (b2*sjdotr*del[1] - b1*spj[1]);
  fmi[2] += gigjpre * (b2*sjdotr*del[2] - b1*spj[2]);
}

/* ----------------------------------------------------------------------
   compute Ewald-corrected mechanical force on atom i due to spin-spin
   interaction with atom j.
   del = xi - xj, b2 = erfc-corrected 3/r^5, b3 = erfc-corrected 15/r^7
------------------------------------------------------------------------- */

void PairSpinDipoleLong::compute_dipolar_mech_long(int /* i */, int /* j */,
    double del[3], double fi[3], double spi[4], double spj[4],
    double b2, double b3)
{
  double sisj,sidotr,sjdotr,gigjpre;

  sidotr = spi[0]*del[0] + spi[1]*del[1] + spi[2]*del[2];
  sjdotr = spj[0]*del[0] + spj[1]*del[1] + spj[2]*del[2];
  sisj   = spi[0]*spj[0] + spi[1]*spj[1] + spi[2]*spj[2];

  gigjpre = 0.5 * mub2mu0 * spi[3] * spj[3];

  fi[0] += gigjpre * (del[0]*(sisj*b2 - sidotr*sjdotr*b3)
                      + b2*(sjdotr*spi[0] + sidotr*spj[0]));
  fi[1] += gigjpre * (del[1]*(sisj*b2 - sidotr*sjdotr*b3)
                      + b2*(sjdotr*spi[1] + sidotr*spj[1]));
  fi[2] += gigjpre * (del[2]*(sisj*b2 - sidotr*sjdotr*b3)
                      + b2*(sjdotr*spi[2] + sidotr*spj[2]));
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSpinDipoleLong::init_style()
{
  if (!atom->sp_flag)
    error->all(FLERR,"Pair spin/dipole/long requires atom/spin style");

  // ensure use of KSpace long-range solver, set g_ewald

  if (force->kspace == nullptr)
    error->all(FLERR,"Pair style spin/dipole/long requires a KSpace style");

  g_ewald = force->kspace->g_ewald;

  // call base class init_style for the rest (newton pair, full neigh list, ...)

  PairSpin::init_style();
}

/* ----------------------------------------------------------------------
   extract long-range parameters for use by kspace
------------------------------------------------------------------------- */

void *PairSpinDipoleLong::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut") == 0) {
    dim = 0;
    return (void *) &cut_spin_long_global;
  } else if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cut_spin_long_global;
  } else if (strcmp(str,"ewald_order") == 0) {
    ewald_order = 0;
    ewald_order |= 1<<1;
    ewald_order |= 1<<3;
    dim = 0;
    return (void *) &ewald_order;
  } else if (strcmp(str,"ewald_mix") == 0) {
    dim = 0;
    return (void *) &mix_flag;
  }
  return nullptr;
}
