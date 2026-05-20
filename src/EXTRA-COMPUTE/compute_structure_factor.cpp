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
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "compute_structure_factor.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "math_const.h"
#include "memory.h"
#include "pair.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr double SMALL = 0.00001;

/* ---------------------------------------------------------------------- */

ComputeStructureFactor::ComputeStructureFactor(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg),
  kxvecs(nullptr), kyvecs(nullptr), kzvecs(nullptr),
  sfacrl(nullptr), sfacim(nullptr), sfacrl_all(nullptr), sfacim_all(nullptr),
  cs(nullptr), sn(nullptr)
{
  kmax_created = 0;

  kmax = 0;
  kxvecs = kyvecs = kzvecs = nullptr;
  sfacrl = sfacim = sfacrl_all = sfacim_all = nullptr;

  nmax = 0;
  cs = sn = nullptr;

  kcount = 0;

  array = nullptr;
  array_flag = 1;
  extarray = 1;

  kxmax = 10;
  kymax = 10;
  kzmax = 0;

  kunique = 0;
  ksq2unique = nullptr;

  norms = nullptr;
  weights = nullptr;

  setup();

  size_array_cols = 3;
  size_array_rows = kunique;

  memory->create(array,size_array_rows,size_array_cols,"structure_factor:array");
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

ComputeStructureFactor::~ComputeStructureFactor()
{
  deallocate();
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);
  memory->destroy(array);
  delete [] norms;
  delete [] weights;
  delete [] ksq2unique;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor::init()
{
  if (comm->me == 0) utils::logmesg(lmp,"StructureFactor initialization ...\n");

  // error check

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use StructureFactor with 2d simulation");

  if (domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use non-periodic boundaries with StructureFactor");

  triclinic = domain->triclinic;

  // setup StructureFactor coefficients so can print stats

  setup();

  // stats

  if (comm->me == 0) {
    std::string mesg = fmt::format("  KSpace vectors: actual max1d max3d unique = {} {} {} {}\n",
                        kcount,kmax,kmax3d,kunique);
    mesg += fmt::format("                  kxmax kymax kzmax  = {} {} {}\n",
                        kxmax,kymax,kzmax);
    utils::logmesg(lmp,mesg);
  }
}

/* ----------------------------------------------------------------------
   adjust StructureFactor coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void ComputeStructureFactor::setup()
{
  // volume-dependent factors

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  volume = xprd * yprd * zprd;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd;

  int kmax_old = kmax;

  // determine kmax

  kmax = MAX(kxmax,kymax);
  kmax = MAX(kmax,kzmax);
  kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

  double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
  double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
  double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
  gsqmx = MAX(gsqxmx,gsqymx);
  gsqmx = MAX(gsqmx,gsqzmx);

  gsqmx = unitk[0]*unitk[0]*17; ////

  gsqmx *= 1.00001;

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
    deallocate();
    allocate();

    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    delete [] weights;
    nmax = atom->nmax;
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
    weights = new double[nmax];
    kmax_created = kmax;
  }

  // pre-compute StructureFactor coefficients

  if (domain->triclinic == 0)
    coeffs();
  else
    coeffs_triclinic();

  int kall = 3*kmax*kmax;
  int* ksq_all = new int[kall];

  delete [] norms;
  norms = new int[kall];

  for (int k = 0; k < kall; k++) {
    ksq_all[k] = 0.0;
    norms[k] = 0.0;
  }

  kunique = 0;
  for (int k = 0; k < kcount; k++) {
    int l = kxvecs[k];
    int m = kyvecs[k];
    int n = kzvecs[k];
    int sqk_int = l*l + m*m + n*n;
    if (ksq_all[sqk_int] == 0) kunique++;
    ksq_all[sqk_int] = 1;
    norms[sqk_int]++;
  }

  delete [] ksq2unique;
  ksq2unique = new int[kall];

  int n = 0;
  for (int k = 0; k < kall; k++) {
    if (ksq_all[k] > 0) {
      ksq2unique[k] = n;
      n++;
    }
  }

  delete [] ksq_all;
}

/* ----------------------------------------------------------------------
   compute the structure factor
------------------------------------------------------------------------- */

void ComputeStructureFactor::compute_array()
{
  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    delete [] weights;
    nmax = atom->nmax;
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
    weights = new double[nmax];
    kmax_created = kmax;
  }

  // partial structure factors on each processor
  // total structure factor by summing over procs

  if (triclinic == 0)
    eik_dot_r();
  else
    eik_dot_r_triclinic();

  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  for (int k = 0; k < kunique; k++)
    array[k][1] = 0.0;

  for (int k = 0; k < kcount; k++) {
    int l = kxvecs[k];
    int m = kyvecs[k];
    int n = kzvecs[k];
    int sqk_int = l*l + m*m + n*n;
    double sqk = (double) sqk_int;
    double q = unitk[0]*sqrt(sqk); ////

    //printf("%g %i %i %i %g\n",q,l,m,n,(sfacrl_all[k]*sfacrl_all[k] +
    //                       sfacim_all[k]*sfacim_all[k])/atom->natoms);
    int kunq = ksq2unique[sqk_int];
    //printf("3D %g: %i %i %g\n",q,norms[sqk_int],(int)group->count(igroup),(sfacrl_all[k]*sfacrl_all[k] +
    //                       sfacim_all[k]*sfacim_all[k])/norms[sqk_int]/group->count(igroup));
    array[kunq][0] = q;
    array[kunq][1] += (sfacrl_all[k]*sfacrl_all[k] +
                               sfacim_all[k]*sfacim_all[k])/norms[sqk_int]/group->count(igroup);
    array[kunq][2] = norms[sqk_int];
  }
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor::eik_dot_r()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm;

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    weights[i] = 1.0;

  n = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 2; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      for (i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          cs[0][ic][i] = 1.0;
          sn[0][ic][i] = 0.0;
          cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
          sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
          cs[-1][ic][i] = cs[1][ic][i];
          sn[-1][ic][i] = -sn[1][ic][i];
          cstr1 += weights[i]*cs[1][ic][i];
          sstr1 += weights[i]*sn[1][ic][i];
        }
      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 2; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        for (i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
            cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
              sn[m-1][ic][i]*sn[1][ic][i];
            sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
              cs[m-1][ic][i]*sn[1][ic][i];
            cs[-m][ic][i] = cs[m][ic][i];
            sn[-m][ic][i] = -sn[m][ic][i];
            cstr1 += weights[i]*cs[m][ic][i];
            sstr1 += weights[i]*sn[m][ic][i];
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
            cstr1 += weights[i]*(cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
            sstr1 += weights[i]*(sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
            cstr2 += weights[i]*(cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
            sstr2 += weights[i]*(sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
            cstr1 += weights[i]*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
            sstr1 += weights[i]*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
            cstr2 += weights[i]*(cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
            sstr2 += weights[i]*(sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
            cstr1 += weights[i]*(cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
            sstr1 += weights[i]*(sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
            cstr2 += weights[i]*(cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
            sstr2 += weights[i]*(sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          for (i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
              clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
              slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
              cstr1 += weights[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr1 += weights[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
              slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
              cstr2 += weights[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr2 += weights[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
              slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
              cstr3 += weights[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr3 += weights[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
              slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
              cstr4 += weights[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr4 += weights[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }
          }
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor::eik_dot_r_triclinic()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1;
  double sqk,clpm,slpm;

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double unitk_lamda[3];

  double max_kvecs[3];
  max_kvecs[0] = kxmax;
  max_kvecs[1] = kymax;
  max_kvecs[2] = kzmax;

  for (int i = 0; i < nmax; i++)
    weights[i] = 1.0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 0.0;
    unitk_lamda[ic] = 2.0*MY_PI;
    x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[ic]*unitk_lamda[ic];
    if (sqk <= gsqmx) {
      for (i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          cs[0][ic][i] = 1.0;
          sn[0][ic][i] = 0.0;
          cs[1][ic][i] = cos(unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
          sn[1][ic][i] = sin(unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
          cs[-1][ic][i] = cs[1][ic][i];
          sn[-1][ic][i] = -sn[1][ic][i];
        }
      }
    }
  }

  for (ic = 0; ic < 3; ic++) {
    for (m = 2; m <= max_kvecs[ic]; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 0.0;
      unitk_lamda[2] = 0.0;
      unitk_lamda[ic] = 2.0*MY_PI*m;
      x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[ic]*unitk_lamda[ic];
      for (i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
          sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];
        }
      }
    }
  }

  for (n = 0; n < kcount; n++) {
    k = kxvecs[n];
    l = kyvecs[n];
    m = kzvecs[n];
    cstr1 = 0.0;
    sstr1 = 0.0;
    for (i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
        slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
        cstr1 += weights[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
        sstr1 += weights[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
      }
    }
    sfacrl[n] = cstr1;
    sfacim[n] = sstr1;
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each structure factor K-vector
------------------------------------------------------------------------- */

void ComputeStructureFactor::coeffs()
{
  int k,l,m;
  double sqk;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0]) * (m*unitk[0]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      kcount++;
    }
    sqk = (m*unitk[1]) * (m*unitk[1]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      kcount++;
    }
    /*sqk = (m*unitk[2]) * (m*unitk[2]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      kcount++;
    }*/
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kzvecs[kcount] = 0;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kzvecs[kcount] = 0;
        kcount++;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        kcount++;

        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = -m;
        kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = m;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = -m;
        kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) +
          (unitk[2]*m) * (unitk[2]*m);
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = -m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = -m;
          kcount++;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each structure factor K-vector for a triclinic
   system
------------------------------------------------------------------------- */

void ComputeStructureFactor::coeffs_triclinic()
{
  int k,l,m;
  double sqk;

  double unitk_lamda[3];

  kcount = 0;

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = -kymax; l <= kymax; l++) {
      for (m = -kzmax; m <= kzmax; m++) {
        unitk_lamda[0] = 2.0*MY_PI*k;
        unitk_lamda[1] = 2.0*MY_PI*l;
        unitk_lamda[2] = 2.0*MY_PI*m;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
        sqk = unitk_lamda[0]*unitk_lamda[0] + unitk_lamda[1]*unitk_lamda[1] +
          unitk_lamda[2]*unitk_lamda[2];
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          kcount++;
        }
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = -kzmax; m <= kzmax; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 2.0*MY_PI*l;
      unitk_lamda[2] = 2.0*MY_PI*m;
      x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[1]*unitk_lamda[1] + unitk_lamda[2]*unitk_lamda[2];
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        kcount++;
      }
    }
  }

  // (0,0,m)

  for (m = 1; m <= kmax; m++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 2.0*MY_PI*m;
    x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[2]*unitk_lamda[2];
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      kcount++;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-arrays
------------------------------------------------------------------------- */

void ComputeStructureFactor::allocate()
{
  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];

  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];
  sfacrl_all = new double[kmax3d];
  sfacim_all = new double[kmax3d];
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-arrays
------------------------------------------------------------------------- */

void ComputeStructureFactor::deallocate()
{
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;

  delete [] sfacrl;
  delete [] sfacim;
  delete [] sfacrl_all;
  delete [] sfacim_all;
}

/* ----------------------------------------------------------------------
   convert box coords vector to transposed triclinic lamda (0-1) coords
   vector, lamda = [(H^-1)^T] * v, does not preserve vector magnitude
   v and lamda can point to same 3-vector
------------------------------------------------------------------------- */

void ComputeStructureFactor::x2lamdaT(double *v, double *lamda)
{
  double *h_inv = domain->h_inv;
  double lamda_tmp[3];

  lamda_tmp[0] = h_inv[0]*v[0];
  lamda_tmp[1] = h_inv[5]*v[0] + h_inv[1]*v[1];
  lamda_tmp[2] = h_inv[4]*v[0] + h_inv[3]*v[1] + h_inv[2]*v[2];

  lamda[0] = lamda_tmp[0];
  lamda[1] = lamda_tmp[1];
  lamda[2] = lamda_tmp[2];
}

/* ----------------------------------------------------------------------
   convert lamda (0-1) coords vector to transposed box coords vector
   lamda = (H^T) * v, does not preserve vector magnitude
   v and lamda can point to same 3-vector
------------------------------------------------------------------------- */

void ComputeStructureFactor::lamda2xT(double *lamda, double *v)
{
  double h[6];
  h[0] = domain->h[0];
  h[1] = domain->h[1];
  h[2] = domain->h[2];
  h[3] = fabs(domain->h[3]);
  h[4] = fabs(domain->h[4]);
  h[5] = fabs(domain->h[5]);
  double v_tmp[3];

  v_tmp[0] = h[0]*lamda[0];
  v_tmp[1] = h[5]*lamda[0] + h[1]*lamda[1];
  v_tmp[2] = h[4]*lamda[0] + h[3]*lamda[1] + h[2]*lamda[2];

  v[0] = v_tmp[0];
  v[1] = v_tmp[1];
  v[2] = v_tmp[2];
}

/* ----------------------------------------------------------------------
   convert triclinic lamda (0-1) coords vector to box coords vector
   v = H * lamda, does not preserve vector magnitude
   lamda and v can point to same 3-vector
------------------------------------------------------------------------- */

void ComputeStructureFactor::lamda2xvector(double *lamda, double *v)
{
  double *h = domain->h;

  v[0] = h[0]*lamda[0] + h[5]*lamda[1] + h[4]*lamda[2];
  v[1] = h[1]*lamda[1] + h[3]*lamda[2];
  v[2] = h[2]*lamda[2];
}
