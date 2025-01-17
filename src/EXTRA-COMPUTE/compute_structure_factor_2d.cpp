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
#include "math_const.h"
#include "memory.h"
#include "pair.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr double SMALL = 0.00001;

/* ---------------------------------------------------------------------- */

ComputeStructureFactor::ComputeStructureFactor(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
  kmax_created = 0;

  kxvecs = kyvecs = nullptr;
  sfacrl = sfacim = sfacrl_all = sfacim_all = nullptr;

  nmax = 0;
  cs = sn = nullptr;

  kcount = 0;

  nbins = 1; //////
  kmax = 10; //////

  kmax2d = 2*kmax*kmax + 2*kmax;
  int ksqmax = kmax*kmax;

  vector = nullptr;
  vector_flag = 1;
  extvector = 1;

  setup();

  size_vector = ksqmax;

  bins = nullptr;
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

ComputeStructureFactor::~ComputeStructureFactor()
{
  deallocate();
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);
  if (bins) delete [] bins;
  bins = nullptr;
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

  if (domain->triclinic)
    error->all(FLERR,"Cannot (yet) use StructureFactor with triclinic box");

  // setup StructureFactor coefficients so can print stats

  setup();


  // stats

  if (comm->me == 0) {
    std::string mesg = fmt::format("  KSpace vectors: actual max1d max2d = {} {} {}\n",
                        kcount,kmax,kmax2d);
    mesg += fmt::format("                  kxmax kymax  = {} {}\n",
                        kxmax,kymax);
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

  kxmax = kmax;
  kymax = kmax;

  //int kmax_old = kmax;

  double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
  double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
  gsqmx = MAX(gsqxmx,gsqymx);

  gsqmx *= 1.00001;

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  //if (kmax > kmax_old) {
    deallocate();
    allocate();

    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    if (bins) delete [] bins;
    nmax = atom->nmax;
    memory->create3d_offset(cs,-kmax,kmax,2,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,2,nmax,"ewald:sn");
    bins = new int[nmax];
    kmax_created = kmax;
  //}

  // pre-compute StructureFactor coefficients

  coeffs();
}

/* ----------------------------------------------------------------------
   compute the structure factor
------------------------------------------------------------------------- */

void ComputeStructureFactor::compute_vector()
{
  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    if (bins) delete [] bins;
    nmax = atom->nmax;
    memory->create3d_offset(cs,-kmax,kmax,2,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,2,nmax,"ewald:sn");
    bins = new int[nmax];
    kmax_created = kmax;
  }

  atom2bin1d();

  // partial structure factors on each processor
  // total structure factor by summing over procs

  for (int k = 0; k < kcount; k++) {
    for (int ibin = 0; ibin < nbins; ibin++) {
      sfacrl[k][ibin] = 0.0;
      sfacim[k][ibin] = 0.0;
    }
  }

  eik_dot_r();

  MPI_Allreduce(&sfacrl[0][0],&sfacrl_all[0][0],kmax2d*nbins,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&sfacim[0][0],&sfacim_all[0][0],kmax2d*nbins,MPI_DOUBLE,MPI_SUM,world);

  for (int k = 0; k < kcount; k++) {
    int l = kxvecs[k];
    int m = kyvecs[k];
    int sqk_int = l*l + m*m;
    for (int ibin = 0; ibin < nbins; ibin++) {
      for (int jbin = 0; jbin < nbins; jbin++) {
        //printf("%i %i %i: %g\n",sqk_int,ibin,jbin,sfacrl_all[k][ibin]*sfacrl_all[k][jbin] +
        //                       sfacim_all[k][ibin]*sfacim_all[k][jbin]/norms[sqk_int]);
        vector[sqk_int*nbins*nbins + ibin*nbins + jbin] = sfacrl_all[k][ibin]*sfacrl_all[k][jbin] +
                               sfacim_all[k][ibin]*sfacim_all[k][jbin]/norms[sqk_int];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor::eik_dot_r()
{
  int i,k,l,m,n,ic;
  double sqk;

  double **x = atom->x;
  int nlocal = atom->nlocal;

  n = 0;

  // (k,0), (0,l)

  for (ic = 0; ic < 2; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      for (i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
        sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];

        int ibin = bins[i];
        sfacrl[n][ibin] += cs[1][ic][i];
        sfacim[n][ibin] += sn[1][ic][i];
      }
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 2; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        for (i = 0; i < nlocal; i++) {
          cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
          sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];

          int ibin = bins[i];
          sfacrl[n][ibin] += cs[m][ic][i];
          sfacim[n][ibin] += sn[m][ic][i];
        }
        n++;
      }
    }
  }

  // 1 = (k,l), 2 = (k,-l)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        for (i = 0; i < nlocal; i++) {
          int ibin = bins[i];
          sfacrl[n][ibin] += cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i];
          sfacim[n][ibin] += sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i];
          sfacrl[n+1][ibin] += cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i];
          sfacim[n+1][ibin] += sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i];
        }
        n+=2;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each StructureFactor K-vector
------------------------------------------------------------------------- */

void ComputeStructureFactor::coeffs()
{
  int k,l,m;
  double sqk;

  kcount = 0;

  // (k,0), (0,l)

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0]) * (m*unitk[0]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kcount++;
    }
    sqk = (m*unitk[1]) * (m*unitk[1]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kcount++;
    }
  }

  // 1 = (k,l), 2 = (k,-l)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kcount++;
      }
    }
  }
  //printf("HERE %i %i\n",kcount,kmax2d);

  for (int k = 0; k < kmax*kmax; k++)
    norms[k] = 0;

  for (int k = 0; k < kcount; k++) {
    int m = kxvecs[k];
    int l = kyvecs[k];
    int sqk_int = m*m + l*l;
    norms[sqk_int]++;
  }

  for (int k = 0; k < kmax*kmax; k++)
    printf("%i %i\n",k,norms[k]);
}

/* ----------------------------------------------------------------------
   assign each atom to a 1d spatial bin (layer)
------------------------------------------------------------------------- */

void ComputeStructureFactor::atom2bin1d()
{
  int i, ibin;
  double *boxlo, *boxhi, *prd;
  double xremap;

  double **x = atom->x;
  int nlocal = atom->nlocal;

  boxlo = domain->boxlo;
  boxhi = domain->boxhi;
  prd = domain->prd;

  double delta = domain->zprd/nbins;
  double invdelta = 1.0/delta;
  double offset = 0.0; 

  // remap each atom's relevant coord back into box via PBC if necessary

  for (i = 0; i < nlocal; i++) {
    xremap = x[i][2];
    if (xremap < boxlo[2]) xremap += prd[2];
    if (xremap >= boxhi[2]) xremap -= prd[2];

    ibin = static_cast<int>((xremap - offset) * invdelta);
    if (xremap < offset) ibin--;

    ibin = MAX(ibin, 0);
    ibin = MIN(ibin, nbins-1);

    bins[i] = ibin + 1;
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void ComputeStructureFactor::allocate()
{
  kxvecs = new int[kmax2d];
  kyvecs = new int[kmax2d];

  memory->create(sfacrl,kmax2d,nbins,"structure_factor:sfacrl");
  memory->create(sfacim,kmax2d,nbins,"structure_factor:sfacim");

  memory->create(sfacrl_all,kmax2d,nbins,"structure_factor:sfacrl_all");
  memory->create(sfacim_all,kmax2d,nbins,"structure_factor:sfacim_all");

  memory->create(vector,kmax2d*nbins*nbins,"structure_factor:vector");
  memory->create(norms,kmax*kmax,"structure_factor:norms");
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void ComputeStructureFactor::deallocate()
{
  delete [] kxvecs;
  delete [] kyvecs;

  memory->destroy(sfacrl);
  memory->destroy(sfacim);

  memory->destroy(sfacrl_all);
  memory->destroy(sfacim_all);

  memory->destroy(vector);
  memory->destroy(norms);
}
