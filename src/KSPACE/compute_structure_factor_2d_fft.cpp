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

   FFT version of compute structure/factor/2d.  The planar structure factor is a
   stack of independent 2D (in-plane) transforms, one per z-bin (z stays a real-
   space layer).  Atoms are spread onto a full periodic xy mesh per bin with a
   Kaiser-Bessel (NUFFT) kernel; the z-bins are slab-decomposed across ranks
   (reduce-scatter of the spread density), each rank forward-transforms its own
   bins, and the per-bin rho(k) are all-gathered to form the cross-bin S_ij(k).
   The KB window is removed analytically.  Same output as structure/factor/2d.
------------------------------------------------------------------------- */

#include "compute_structure_factor_2d_fft.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "math_const.h"
#include "memory.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr int OFFSET = 16384;
static constexpr double SMALL = 0.00001;

/* ---------------------------------------------------------------------- */

ComputeStructureFactor2DFFT::ComputeStructureFactor2DFFT(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  kxvecs(nullptr), kyvecs(nullptr), norms(nullptr), ksq2unique(nullptr),
  recvcount(nullptr), displs(nullptr), meshloc(nullptr), meshown(nullptr),
  work(nullptr), rho1d(nullptr), fft2d(nullptr),
  rhohat_re(nullptr), rhohat_im(nullptr), rhohat_re_all(nullptr), rhohat_im_all(nullptr),
  part2grid(nullptr), binofatom(nullptr), counts(nullptr), counts_all(nullptr)
{
  me = comm->me;
  nprocs = comm->nprocs;

  if (narg < 5 || narg > 6) error->all(FLERR,"Illegal compute structure/factor/2d/fft command");
  kmax = utils::inumeric(FLERR,arg[3],false,lmp);
  nbins = utils::inumeric(FLERR,arg[4],false,lmp);
  order = 7;
  oversample = 2.0;
  if (narg == 6) order = utils::inumeric(FLERR,arg[5],false,lmp);

  if (kmax <= 0) error->all(FLERR,"compute structure/factor/2d/fft kmax must be positive");
  if (nbins <= 0) error->all(FLERR,"compute structure/factor/2d/fft nbins must be positive");
  if (order < 3 || order % 2 == 0)
    error->all(FLERR,"compute structure/factor/2d/fft order must be odd and >= 3");

  array_flag = 1;
  extarray = 1;
  size_array_cols = 5;
  nmax = 0;
  mesh_allocated = 0;

  kxmax = kymax = kmax;
  ksqmax = kmax*kmax;
  kmax2d = 6*kmax*kmax + 3*kmax;
  allocate_kvecs();

  counts = new int[nbins];
  counts_all = new int[nbins];

  setup();

  size_array_rows = (kunique+1)*nbins*nbins;
  memory->create(array,size_array_rows,size_array_cols,"structure/factor/2d/fft:array");
}

/* ---------------------------------------------------------------------- */

ComputeStructureFactor2DFFT::~ComputeStructureFactor2DFFT()
{
  deallocate_kvecs();
  deallocate_mesh();
  memory->destroy(array);
  delete [] counts;
  delete [] counts_all;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::init()
{
  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use compute structure/factor/2d/fft with 2d simulation");
  if (domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use non-periodic boundaries with compute structure/factor/2d/fft");
  if (domain->triclinic)
    error->all(FLERR,"Cannot use compute structure/factor/2d/fft with triclinic box");
  if (fabs(domain->xprd-domain->yprd) > SMALL*domain->xprd)
    error->all(FLERR,"compute structure/factor/2d/fft requires a square box in x and y");

  setup();

  deallocate_mesh();
  set_grid();

  // slab-decompose the z-bins across ranks (contiguous blocks)
  binlo = (int) ((bigint) me * nbins / nprocs);
  binhi = (int) ((bigint) (me+1) * nbins / nprocs) - 1;
  nbins_local = binhi - binlo + 1;
  if (nbins_local < 0) nbins_local = 0;

  allocate_mesh();

  if (me == 0)
    utils::logmesg(lmp,"StructureFactor2DFFT: KB order {}, xy FFT {}x{}, {} bins, "
                   "beta {:.4}, k-shells {}\n",order,nx_sf,ny_sf,nbins,kb_beta,kunique);
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::setup()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  volume = xprd*yprd*zprd;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;

  gsqmx = unitk[0]*unitk[0]*ksqmax*1.00001;

  coeffs();

  int kall = 2*kmax*kmax + 1;
  int *ksq_all = new int[kall];
  delete [] norms;
  norms = new int[kall];
  for (int k = 0; k < kall; k++) { ksq_all[k] = 0; norms[k] = 0; }

  kunique = 0;
  for (int k = 0; k < kcount; k++) {
    int sqk_int = kxvecs[k]*kxvecs[k] + kyvecs[k]*kyvecs[k];
    if (ksq_all[sqk_int] == 0) kunique++;
    ksq_all[sqk_int] = 1;
    norms[sqk_int]++;
  }

  delete [] ksq2unique;
  ksq2unique = new int[kall];
  int n = 0;
  for (int k = 0; k < kall; k++)
    if (ksq_all[k] > 0) ksq2unique[k] = n++;

  delete [] ksq_all;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::coeffs()
{
  int k,l,m;
  double sqk;
  kcount = 0;

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0])*(m*unitk[0]);
    if (sqk <= gsqmx) { kxvecs[kcount]=m; kyvecs[kcount]=0; kcount++; }
    sqk = (m*unitk[1])*(m*unitk[1]);
    if (sqk <= gsqmx) { kxvecs[kcount]=0; kyvecs[kcount]=m; kcount++; }
  }

  for (k = 1; k <= kxmax; k++)
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k)*(unitk[0]*k) + (unitk[1]*l)*(unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount]=k; kyvecs[kcount]=l;  kcount++;
        kxvecs[kcount]=k; kyvecs[kcount]=-l; kcount++;
      }
    }
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::set_grid()
{
  int nmin = (int) ceil(2.0*oversample*kmax);
  int nn = nmin;
  while (!factorable(nn)) nn++;
  nx_sf = ny_sf = nn;

  double sigma = (0.5*nx_sf)/kmax;
  double arg = (order*order/(sigma*sigma))*(sigma-0.5)*(sigma-0.5) - 0.8;
  if (arg < 0.0) arg = 0.0;
  kb_beta = MY_PI*sqrt(arg);
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::allocate_mesh()
{
  shift = OFFSET + 0.5;
  shiftone = 0.0;
  nlower = -(order-1)/2;
  nupper = order/2;
  delxinv = nx_sf/domain->xprd;
  delyinv = ny_sf/domain->yprd;

  int nxy = nx_sf*ny_sf;
  memory->create(meshloc,(bigint)nbins*nxy,"structure/factor/2d/fft:meshloc");
  memory->create(meshown,(bigint)MAX(nbins_local,1)*nxy,"structure/factor/2d/fft:meshown");
  memory->create(work,2*nxy,"structure/factor/2d/fft:work");
  memory->create2d_offset(rho1d,2,-order/2,order/2,"structure/factor/2d/fft:rho1d");

  // reduce-scatter / allgather bookkeeping over the bin decomposition
  recvcount = new int[nprocs];
  displs = new int[nprocs];
  for (int r = 0; r < nprocs; r++) {
    int lo = (int) ((bigint) r * nbins / nprocs);
    int hi = (int) ((bigint) (r+1) * nbins / nprocs) - 1;
    recvcount[r] = (hi-lo+1)*nxy;
  }

  rhohat_re = new double[MAX(nbins_local,1)*kcount];
  rhohat_im = new double[MAX(nbins_local,1)*kcount];
  rhohat_re_all = new double[(bigint)nbins*kcount];
  rhohat_im_all = new double[(bigint)nbins*kcount];

  int tmp;
  fft2d = new FFT3d(lmp,MPI_COMM_SELF,nx_sf,ny_sf,1,
                    0,nx_sf-1,0,ny_sf-1,0,0,
                    0,nx_sf-1,0,ny_sf-1,0,0,
                    0,0,&tmp,0,0);
  mesh_allocated = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::deallocate_mesh()
{
  if (!mesh_allocated) return;
  delete fft2d; fft2d = nullptr;
  memory->destroy(meshloc);
  memory->destroy(meshown);
  memory->destroy(work);
  memory->destroy2d_offset(rho1d,-order/2);
  delete [] recvcount; recvcount = nullptr;
  delete [] displs; displs = nullptr;
  delete [] rhohat_re; rhohat_re = nullptr;
  delete [] rhohat_im; rhohat_im = nullptr;
  delete [] rhohat_re_all; rhohat_re_all = nullptr;
  delete [] rhohat_im_all; rhohat_im_all = nullptr;
  mesh_allocated = 0;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::allocate_kvecs()
{
  kxvecs = new int[kmax2d];
  kyvecs = new int[kmax2d];
}

void ComputeStructureFactor2DFFT::deallocate_kvecs()
{
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] norms;
  delete [] ksq2unique;
}

/* ----------------------------------------------------------------------
   compute the bin-resolved planar structure factor via per-bin 2D FFTs
------------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::compute_array()
{
  if (atom->nmax > nmax) {
    memory->destroy(part2grid);
    delete [] binofatom;
    nmax = atom->nmax;
    memory->create(part2grid,nmax,2,"structure/factor/2d/fft:part2grid");
    binofatom = new int[nmax];
  }

  boxlo = domain->boxlo;
  int nxy = nx_sf*ny_sf;

  atom2bin();
  particle_map();
  make_rho();

  // distribute the per-bin spread density: each rank receives the global sum
  // for the bins it owns

  MPI_Reduce_scatter(meshloc,meshown,recvcount,MPI_FFT_SCALAR,MPI_SUM,world);

  // forward 2D FFT each owned bin, deconvolve the KB window, store rho(k)

  for (int b = 0; b < nbins_local; b++) {
    FFT_SCALAR *m = &meshown[(bigint)b*nxy];
    for (int i = 0; i < nxy; i++) { work[2*i] = m[i]; work[2*i+1] = (FFT_SCALAR) 0.0; }
    fft2d->compute(work,work,FFT3d::FORWARD);
    for (int k = 0; k < kcount; k++) {
      int l = kxvecs[k];
      int mm = kyvecs[k];
      int ix = (l % nx_sf + nx_sf) % nx_sf;
      int iy = (mm % ny_sf + ny_sf) % ny_sf;
      int idx = iy*nx_sf + ix;
      double w = kb_window(l,nx_sf) * kb_window(mm,ny_sf);
      rhohat_re[b*kcount+k] = work[2*idx]/w;
      rhohat_im[b*kcount+k] = work[2*idx+1]/w;
    }
  }

  // gather all bins' rho(k) to every rank for the cross-bin products

  int *rc = new int[nprocs];
  int *dp = new int[nprocs];
  for (int r = 0; r < nprocs; r++) rc[r] = (recvcount[r]/nxy)*kcount;
  dp[0] = 0;
  for (int r = 1; r < nprocs; r++) dp[r] = dp[r-1] + rc[r-1];
  MPI_Allgatherv(rhohat_re,nbins_local*kcount,MPI_DOUBLE,rhohat_re_all,rc,dp,MPI_DOUBLE,world);
  MPI_Allgatherv(rhohat_im,nbins_local*kcount,MPI_DOUBLE,rhohat_im_all,rc,dp,MPI_DOUBLE,world);
  delete [] rc;
  delete [] dp;

  MPI_Allreduce(counts,counts_all,nbins,MPI_INT,MPI_SUM,world);

  // assemble the output array (matches compute structure/factor/2d)

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double invarea = 1.0/(xprd*yprd);
  double volbininv = nbins/volume;

  for (int k = 0; k < size_array_rows; k++) array[k][3] = 0.0;

  for (int ibin = 0; ibin < nbins; ibin++)
    for (int jbin = 0; jbin < nbins; jbin++) {
      int index = ibin*nbins + jbin;
      array[index][0] = 0.0;
      array[index][1] = ibin;
      array[index][2] = jbin;
      array[index][3] = (double)counts_all[ibin]*counts_all[jbin]*invarea;
      array[index][4] = counts_all[ibin]*volbininv;
    }

  for (int k = 0; k < kcount; k++) {
    int l = kxvecs[k];
    int mm = kyvecs[k];
    int sqk_int = l*l + mm*mm;
    double q = unitk[0]*sqrt((double) sqk_int);
    int kunq = ksq2unique[sqk_int] + 1;
    double inorm = 1.0/norms[sqk_int];
    for (int ibin = 0; ibin < nbins; ibin++) {
      double rei = rhohat_re_all[ibin*kcount+k];
      double imi = rhohat_im_all[ibin*kcount+k];
      for (int jbin = 0; jbin < nbins; jbin++) {
        double rej = rhohat_re_all[jbin*kcount+k];
        double imj = rhohat_im_all[jbin*kcount+k];
        int index = kunq*nbins*nbins + ibin*nbins + jbin;
        array[index][0] = q;
        array[index][1] = ibin;
        array[index][2] = jbin;
        array[index][3] += (rei*rej + imi*imj)*inorm;
        array[index][4] = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::atom2bin()
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *bxlo = domain->boxlo;
  double *bxhi = domain->boxhi;
  double *prd = domain->prd;
  double invdelta = nbins/domain->zprd;

  for (int i = 0; i < nbins; i++) counts[i] = 0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) { binofatom[i] = -1; continue; }
    double zremap = x[i][2];
    if (zremap < bxlo[2]) zremap += prd[2];
    if (zremap >= bxhi[2]) zremap -= prd[2];
    int ibin = static_cast<int>((zremap - bxlo[2])*invdelta);
    ibin = MAX(ibin,0);
    ibin = MIN(ibin,nbins-1);
    binofatom[i] = ibin;
    counts[ibin]++;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::particle_map()
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    part2grid[i][0] = static_cast<int>((x[i][0]-boxlo[0])*delxinv+shift) - OFFSET;
    part2grid[i][1] = static_cast<int>((x[i][1]-boxlo[1])*delyinv+shift) - OFFSET;
  }
}

/* ----------------------------------------------------------------------
   spread atoms (unit weight) onto the full periodic xy mesh of their z-bin
------------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::make_rho()
{
  int nxy = nx_sf*ny_sf;
  memset(meshloc,0,(bigint)nbins*nxy*sizeof(FFT_SCALAR));

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int ibin = binofatom[i];
    int nx = part2grid[i][0];
    int ny = part2grid[i][1];
    FFT_SCALAR dx = nx+shiftone - (x[i][0]-boxlo[0])*delxinv;
    FFT_SCALAR dy = ny+shiftone - (x[i][1]-boxlo[1])*delyinv;
    compute_kb1d(dx,dy);

    FFT_SCALAR *mesh = &meshloc[(bigint)ibin*nxy];
    for (int mm = nlower; mm <= nupper; mm++) {
      int my = ((mm+ny) % ny_sf + ny_sf) % ny_sf;
      FFT_SCALAR y0 = rho1d[1][mm];
      for (int ll = nlower; ll <= nupper; ll++) {
        int mx = ((ll+nx) % nx_sf + nx_sf) % nx_sf;
        mesh[my*nx_sf + mx] += y0*rho1d[0][ll];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactor2DFFT::compute_kb1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy)
{
  double invhalf = 2.0/order;
  for (int k = nlower; k <= nupper; k++) {
    double ax = (k + dx)*invhalf; double sx = 1.0 - ax*ax; if (sx < 0.0) sx = 0.0;
    double ay = (k + dy)*invhalf; double sy = 1.0 - ay*ay; if (sy < 0.0) sy = 0.0;
    rho1d[0][k] = (FFT_SCALAR) bessel_i0(kb_beta*sqrt(sx));
    rho1d[1][k] = (FFT_SCALAR) bessel_i0(kb_beta*sqrt(sy));
  }
}

/* ---------------------------------------------------------------------- */

double ComputeStructureFactor2DFFT::kb_window(int m, int N)
{
  double arg = MY_PI*order*((double) m)/N;
  double t = kb_beta*kb_beta - arg*arg;
  if (t > 0.0) { double s = sqrt(t); return order*sinh(s)/s; }
  else if (t < 0.0) { double s = sqrt(-t); return order*sin(s)/s; }
  return (double) order;
}

/* ---------------------------------------------------------------------- */

int ComputeStructureFactor2DFFT::factorable(int n)
{
  while (n > 1) {
    if (n % 2 == 0) n /= 2;
    else if (n % 3 == 0) n /= 3;
    else if (n % 5 == 0) n /= 5;
    else return 0;
  }
  return 1;
}

/* ---------------------------------------------------------------------- */

double ComputeStructureFactor2DFFT::bessel_i0(double x)
{
  double ax = fabs(x);
  if (ax < 3.75) {
    double t = x/3.75; t *= t;
    return 1.0 + t*(3.5156229 + t*(3.0899424 + t*(1.2067492 +
           t*(0.2659732 + t*(0.0360768 + t*0.0045813)))));
  }
  double t = 3.75/ax;
  return (exp(ax)/sqrt(ax))*(0.39894228 + t*(0.01328592 + t*(0.00225319 +
         t*(-0.00157565 + t*(0.00916281 + t*(-0.02057706 + t*(0.02635537 +
         t*(-0.01647633 + t*0.00392377))))))));
}
