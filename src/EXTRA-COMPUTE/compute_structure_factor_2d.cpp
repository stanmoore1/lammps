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

#include "compute_structure_factor_2d.h"

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

ComputeStructureFactor2D::ComputeStructureFactor2D(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg),
  kxvecs(nullptr), kyvecs(nullptr),
  sfacrl(nullptr), sfacim(nullptr), sfacrl_all(nullptr), sfacim_all(nullptr),
  cs(nullptr), sn(nullptr)
{
  kmax_created = 0;

  kmax = 0;
  kxvecs = kyvecs = nullptr;
  sfacrl = sfacim = sfacrl_all = sfacim_all = nullptr;

  nmax = 0;
  cs = sn = nullptr;

  kcount = 0;

  array = nullptr;
  array_flag = 1;
  extarray = 1;

  // parse arguments: kmax nbins

  if (narg != 5) error->all(FLERR,"Illegal compute structure/factor/2d command");

  kxmax = kymax = utils::inumeric(FLERR,arg[3],false,lmp);
  nbins = utils::inumeric(FLERR,arg[4],false,lmp);

  if (kxmax <= 0) error->all(FLERR,"Compute structure/factor/2d kmax must be positive");
  if (nbins <= 0) error->all(FLERR,"Compute structure/factor/2d nbins must be positive");

  kunique = 0;
  ksq2unique = nullptr;

  norms = nullptr;
  weights = nullptr;
  bins = nullptr;
  counts = new int[nbins];
  counts_all = new int[nbins];

  setup();

  size_array_cols = 5;
  size_array_rows = (kunique+1)*nbins*nbins;

  memory->create(array,size_array_rows,size_array_cols,"structure_factor_2d:array");
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

ComputeStructureFactor2D::~ComputeStructureFactor2D()
{
  deallocate();
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);
  memory->destroy(array);
  delete [] norms;
  delete [] weights;
  delete [] ksq2unique;
  delete [] bins;
  delete [] counts;
  delete [] counts_all;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2D::init()
{
  if (comm->me == 0) utils::logmesg(lmp,"StructureFactor2D initialization ...\n");

  // error check

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use StructureFactor2D with 2d simulation");

  if (domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use non-periodic boundaries with StructureFactor2D");

  if (domain->triclinic)
    error->all(FLERR,"Cannot (yet) use StructureFactor2D with triclinic box");

  // this compute assumes a square cross-section (Lx == Ly) so that the in-plane
  // reciprocal-lattice spacing is isotropic and |k| = unitk[0]*sqrt(l*l+m*m)

  if (fabs(domain->xprd - domain->yprd) > SMALL*domain->xprd)
    error->all(FLERR,"Compute structure/factor/2d requires a square box in x and y (Lx == Ly)");

  // setup StructureFactor coefficients so can print stats

  setup();

  // stats

  if (comm->me == 0) {
    std::string mesg = fmt::format("  KSpace vectors: actual max1d max2d unique = {} {} {} {}\n",
                        kcount,kmax,kmax2d,kunique);
    mesg += fmt::format("                  kxmax kymax  = {} {}\n",
                        kxmax,kymax);
    utils::logmesg(lmp,mesg);
  }
}

/* ----------------------------------------------------------------------
   adjust StructureFactor coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void ComputeStructureFactor2D::setup()
{
  // volume-dependent factors

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  volume = xprd * yprd * zprd;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;

  int kmax_old = kmax;

  // determine kmax

  kmax = MAX(kxmax,kymax);
  kmax2d = 6*kmax*kmax + 3*kmax;

  // circular cutoff in reciprocal space with radius kmax
  // (square box guarantees unitk[0] == unitk[1])

  double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
  double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
  gsqmx = MAX(gsqxmx,gsqymx);

  gsqmx *= 1.00001;

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
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
  }

  // pre-compute StructureFactor coefficients

  coeffs();

  int kall = 2*kmax*kmax;
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
    int sqk_int = l*l + m*m;
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
   compute the bin-resolved (planar) structure factor

   the global array has size_array_cols = 5 columns:
     [0] q     = |k_parallel| = unitk[0]*sqrt(l*l+m*m)  (requires Lx == Ly)
     [1] ibin  = z-bin index i
     [2] jbin  = z-bin index j
     [3] S_ij  = <rho_hat_i(k) * conj(rho_hat_j(k))>, direction-averaged over
                 in-plane k-vectors of equal |k|.  On the q == 0 rows this column
                 instead holds the disconnected product N_i*N_j/A and is unused
                 by the OZ inversion.
     [4] density = per-bin number density rho_i, stored only on the q == 0 rows.

   the value emitted here is instantaneous and unnormalized (no division by area
   or time); time-averaging is done with fix ave/time and all normalization for
   the Ornstein-Zernike inversion is done in post-processing.
------------------------------------------------------------------------- */

void ComputeStructureFactor2D::compute_array()
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

  for (int ibin = 0; ibin < nbins; ibin++)
    counts[ibin] = 0;

  atom2bin1d();

  MPI_Allreduce(counts,counts_all,nbins,MPI_INT,MPI_SUM,world);

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  volume = xprd * yprd * zprd;
  double invarea = 1.0/(xprd * yprd);
  double volbin = volume/nbins;
  double volbininv = 1.0/(volbin);
  double volbin2inv = 1.0/(volbin*volbin);

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

  for (int k = 0; k < size_array_rows; k++)
    array[k][3] = 0.0;

  // q = 0

  for (int ibin = 0; ibin < nbins; ibin++) {
    for (int jbin = 0; jbin < nbins; jbin++) {
      double q = 0;
      int kunq = 0;
      int index = kunq*nbins*nbins + ibin*nbins + jbin;
      array[index][0] = q;
      array[index][1] = ibin;
      array[index][2] = jbin;
      array[index][3] = counts_all[ibin]*counts_all[jbin]*invarea;
      array[index][4] = counts_all[ibin]*volbininv;
      //printf("%i %i %i %i\n",ibin,jbin,counts_all[ibin],counts_all[jbin]);
    }
  }

  // q > 0

  for (int k = 0; k < kcount; k++) {
    for (int ibin = 0; ibin < nbins; ibin++) {
      for (int jbin = 0; jbin < nbins; jbin++) {
        int l = kxvecs[k];
        int m = kyvecs[k];
        int sqk_int = l*l + m*m;
        double sqk = (double) sqk_int;
        double q = unitk[0]*sqrt(sqk); ////
        int kunq = ksq2unique[sqk_int]+1;
        int index = kunq*nbins*nbins + ibin*nbins + jbin;
        //printf("2D %g: %i %g %g\n",q,norms[sqk_int],sqrt(counts_all[ibin])*sqrt(counts_all[jbin]),(sfacrl_all[ibin][k]*sfacrl_all[jbin][k] +
        //                       sfacim_all[ibin][k]*sfacim_all[jbin][k])/norms[sqk_int];
        array[index][0] = q;
        array[index][1] = ibin;
        array[index][2] = jbin;
        // sfacrl_all is indexed [kvector][bin]; accumulate the cross-bin product
        array[index][3] += (sfacrl_all[k][ibin]*sfacrl_all[k][jbin] +
                                   sfacim_all[k][ibin]*sfacim_all[k][jbin])/norms[sqk_int];
        array[index][4] = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2D::eik_dot_r()
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
      n++;
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
   pre-compute coefficients for each structure factor K-vector
------------------------------------------------------------------------- */

void ComputeStructureFactor2D::coeffs()
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
}

/* ----------------------------------------------------------------------
   assign each atom to a 1d spatial bin (layer)
------------------------------------------------------------------------- */

void ComputeStructureFactor2D::atom2bin1d()
{
  int i, ibin;
  double *boxlo, *boxhi, *prd;
  double zremap;

  double **x = atom->x;
  int nlocal = atom->nlocal;

  boxlo = domain->boxlo;
  boxhi = domain->boxhi;
  prd = domain->prd;

  double delta = domain->zprd/nbins;
  double invdelta = 1.0/delta;

  // remap each atom's relevant coord back into box via PBC if necessary

  for (i = 0; i < nlocal; i++) {
    //xremap = x[i][0];
    //yremap = x[i][1];
    zremap = x[i][2];

    //if (xremap < boxlo[0]) xremap += prd[0];
    //if (xremap >= boxhi[0]) xremap -= prd[0];

    //if (yremap < boxlo[1]) yremap += prd[1];
    //if (yremap >= boxhi[1]) yremap -= prd[1];

    if (zremap < boxlo[2]) zremap += prd[2];
    if (zremap >= boxhi[2]) zremap -= prd[2];

    ibin = static_cast<int>((zremap - boxlo[2]) * invdelta);
    ibin = MAX(ibin, 0);
    ibin = MIN(ibin, nbins-1);

    bins[i] = ibin;
    counts[ibin]++;
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void ComputeStructureFactor2D::allocate()
{
  kxvecs = new int[kmax2d];
  kyvecs = new int[kmax2d];

  memory->create(sfacrl,kmax2d,nbins,"structure_factor_2d:sfacrl");
  memory->create(sfacim,kmax2d,nbins,"structure_factor_2d:sfacim");

  memory->create(sfacrl_all,kmax2d,nbins,"structure_factor_2d:sfacrl_all");
  memory->create(sfacim_all,kmax2d,nbins,"structure_factor_2d:sfacim_all");
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void ComputeStructureFactor2D::deallocate()
{
  delete [] kxvecs;
  delete [] kyvecs;

  memory->destroy(sfacrl);
  memory->destroy(sfacim);

  memory->destroy(sfacrl_all);
  memory->destroy(sfacim_all);
}
