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

   FFT structure factor.  Same output as compute structure/factor (the direct
   O(N*K) Ewald-style DFT), but rho(k) is obtained by spreading the atoms onto
   a distributed FFT mesh (modeled on PPPM) and forward-transforming, an
   O(N + M log M) cost.  The spreading kernel is a Kaiser-Bessel (NUFFT) window
   and is removed analytically by dividing rho_mesh(k) by its exact Fourier
   transform.
------------------------------------------------------------------------- */

#include "compute_structure_factor_fft.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "grid3d.h"
#include "group.h"
#include "math_const.h"
#include "memory.h"
#include "neighbor.h"
#include "remap_wrap.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr int OFFSET = 16384;
static constexpr double SMALL = 0.00001;

/* ---------------------------------------------------------------------- */

ComputeStructureFactorFFT::ComputeStructureFactorFFT(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  kxvecs(nullptr), kyvecs(nullptr), kzvecs(nullptr), norms(nullptr), ksq2unique(nullptr),
  density_brick(nullptr), density_fft(nullptr), work1(nullptr), rho1d(nullptr),
  part2grid(nullptr), gc_buf1(nullptr), gc_buf2(nullptr),
  fft1(nullptr), remap(nullptr), gc(nullptr)
{
  me = comm->me;
  nprocs = comm->nprocs;

  // optional args: ksqmax (radial |k|^2 cutoff in integer units), order
  // defaults reproduce the shells of compute structure/factor (ksqmax = 17)

  ksqmax = 17;
  order = 7;
  oversample = 2.0;
  if (narg >= 4) ksqmax = utils::inumeric(FLERR,arg[3],false,lmp);
  if (narg >= 5) order = utils::inumeric(FLERR,arg[4],false,lmp);
  if (narg > 5) error->all(FLERR,"Illegal compute structure/factor/fft command");
  if (ksqmax <= 0) error->all(FLERR,"compute structure/factor/fft ksqmax must be positive");
  if (order < 3 || order % 2 == 0)
    error->all(FLERR,"compute structure/factor/fft order must be odd and >= 3");

  array_flag = 1;
  extarray = 1;
  size_array_cols = 3;

  nmax = 0;

  kmax = (int) ceil(sqrt((double) ksqmax));
  kxmax = kymax = kzmax = kmax;
  kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;
  allocate_kvecs();

  setup();

  size_array_rows = kunique;
  memory->create(array,size_array_rows,size_array_cols,"structure/factor/fft:array");

  mesh_allocated = 0;
}

/* ---------------------------------------------------------------------- */

ComputeStructureFactorFFT::~ComputeStructureFactorFFT()
{
  deallocate_kvecs();
  deallocate_mesh();
  memory->destroy(array);
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactorFFT::init()
{
  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use compute structure/factor/fft with 2d simulation");
  if (domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use non-periodic boundaries with compute structure/factor/fft");
  if (domain->triclinic)
    error->all(FLERR,"Cannot use compute structure/factor/fft with triclinic box");

  // the integer-|k|^2 shell binning (q = unitk*sqrt(l^2+m^2+n^2)) assumes a cube

  if (fabs(domain->xprd-domain->yprd) > SMALL*domain->xprd ||
      fabs(domain->yprd-domain->zprd) > SMALL*domain->yprd)
    error->all(FLERR,"compute structure/factor/fft requires a cubic box");

  setup();

  // build/refresh the distributed FFT mesh

  deallocate_mesh();
  set_grid();
  set_grid_local();
  allocate_mesh();

  if (me == 0)
    utils::logmesg(lmp,"StructureFactorFFT: KB order {}, FFT grid {}x{}x{}, "
                   "beta {:.4}, k-shells {}\n",order,nx_sf,ny_sf,nz_sf,kb_beta,kunique);
}

/* ----------------------------------------------------------------------
   k-vector enumeration + volume factors (mirrors ComputeStructureFactor)
------------------------------------------------------------------------- */

void ComputeStructureFactorFFT::setup()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  volume = xprd*yprd*zprd;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd;

  gsqmx = unitk[0]*unitk[0]*ksqmax*1.00001;

  coeffs();

  // map each |k|^2 shell (integer l^2+m^2+n^2) to a unique row, count its members

  int kall = 3*kmax*kmax + 1;
  int *ksq_all = new int[kall];
  delete [] norms;
  norms = new int[kall];
  for (int k = 0; k < kall; k++) { ksq_all[k] = 0; norms[k] = 0; }

  kunique = 0;
  for (int k = 0; k < kcount; k++) {
    int sqk_int = kxvecs[k]*kxvecs[k] + kyvecs[k]*kyvecs[k] + kzvecs[k]*kzvecs[k];
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

/* ----------------------------------------------------------------------
   enumerate k-vectors with l^2+m^2+n^2 <= ksqmax (same set as the DFT compute)
------------------------------------------------------------------------- */

void ComputeStructureFactorFFT::coeffs()
{
  int k,l,m;
  double sqk;

  kcount = 0;

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0]) * (m*unitk[0]);
    if (sqk <= gsqmx) { kxvecs[kcount]=m; kyvecs[kcount]=0; kzvecs[kcount]=0; kcount++; }
    sqk = (m*unitk[1]) * (m*unitk[1]);
    if (sqk <= gsqmx) { kxvecs[kcount]=0; kyvecs[kcount]=m; kzvecs[kcount]=0; kcount++; }
    sqk = (m*unitk[2]) * (m*unitk[2]);
    if (sqk <= gsqmx) { kxvecs[kcount]=0; kyvecs[kcount]=0; kzvecs[kcount]=m; kcount++; }
  }

  for (k = 1; k <= kxmax; k++)
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k)*(unitk[0]*k) + (unitk[1]*l)*(unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount]=k; kyvecs[kcount]=l;  kzvecs[kcount]=0; kcount++;
        kxvecs[kcount]=k; kyvecs[kcount]=-l; kzvecs[kcount]=0; kcount++;
      }
    }

  for (l = 1; l <= kymax; l++)
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[1]*l)*(unitk[1]*l) + (unitk[2]*m)*(unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount]=0; kyvecs[kcount]=l; kzvecs[kcount]=m;  kcount++;
        kxvecs[kcount]=0; kyvecs[kcount]=l; kzvecs[kcount]=-m; kcount++;
      }
    }

  for (k = 1; k <= kxmax; k++)
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[0]*k)*(unitk[0]*k) + (unitk[2]*m)*(unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount]=k; kyvecs[kcount]=0; kzvecs[kcount]=m;  kcount++;
        kxvecs[kcount]=k; kyvecs[kcount]=0; kzvecs[kcount]=-m; kcount++;
      }
    }

  for (k = 1; k <= kxmax; k++)
    for (l = 1; l <= kymax; l++)
      for (m = 1; m <= kzmax; m++) {
        sqk = (unitk[0]*k)*(unitk[0]*k) + (unitk[1]*l)*(unitk[1]*l) +
              (unitk[2]*m)*(unitk[2]*m);
        if (sqk <= gsqmx) {
          kxvecs[kcount]=k; kyvecs[kcount]=l;  kzvecs[kcount]=m;  kcount++;
          kxvecs[kcount]=k; kyvecs[kcount]=-l; kzvecs[kcount]=m;  kcount++;
          kxvecs[kcount]=k; kyvecs[kcount]=l;  kzvecs[kcount]=-m; kcount++;
          kxvecs[kcount]=k; kyvecs[kcount]=-l; kzvecs[kcount]=-m; kcount++;
        }
      }
}

/* ----------------------------------------------------------------------
   choose the global FFT mesh (auto from kmax) and the KB shape parameter
------------------------------------------------------------------------- */

void ComputeStructureFactorFFT::set_grid()
{
  // mesh must resolve modes up to kmax with oversampling factor 'oversample';
  // Nyquist = N/2 >= oversample*kmax

  int nmin = (int) ceil(2.0*oversample*kmax);
  int n = nmin;
  while (!factorable(n)) n++;
  nx_sf = ny_sf = nz_sf = n;

  // Beatty (2005) optimal KB shape parameter for width = order, oversample sigma

  double sigma = (0.5*nx_sf)/kmax;
  double arg = (order*order/(sigma*sigma))*(sigma-0.5)*(sigma-0.5) - 0.8;
  if (arg < 0.0) arg = 0.0;
  kb_beta = MY_PI*sqrt(arg);
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactorFFT::set_grid_local()
{
  shift = OFFSET + 0.5;     // order is odd
  shiftone = 0.0;
  nlower = -(order-1)/2;
  nupper = order/2;

  delxinv = nx_sf/domain->xprd;
  delyinv = ny_sf/domain->yprd;
  delzinv = nz_sf/domain->zprd;

  // x-pencil FFT decomposition (full x, blocks of y,z), as in PPPM

  int npey_fft = 1, npez_fft = nprocs;
  procs2grid2d(nprocs,ny_sf,nz_sf,npey_fft,npez_fft);

  int me_y = me % npey_fft;
  int me_z = me / npey_fft;

  nxlo_fft = 0;
  nxhi_fft = nx_sf - 1;
  nylo_fft = me_y*ny_sf/npey_fft;
  nyhi_fft = (me_y+1)*ny_sf/npey_fft - 1;
  nzlo_fft = me_z*nz_sf/npez_fft;
  nzhi_fft = (me_z+1)*nz_sf/npez_fft - 1;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactorFFT::allocate_mesh()
{
  gc = new Grid3d(lmp,world,nx_sf,ny_sf,nz_sf);
  gc->set_distance(0.5*neighbor->skin);
  gc->set_stencil_atom(-nlower,nupper);
  gc->set_shift_atom(0.5,0.5);          // odd order

  gc->setup_grid(nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                 nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out);
  gc->setup_comm(ngc_buf1,ngc_buf2);

  npergrid = 1;
  memory->create(gc_buf1,npergrid*ngc_buf1,"structure/factor/fft:gc_buf1");
  memory->create(gc_buf2,npergrid*ngc_buf2,"structure/factor/fft:gc_buf2");

  ngrid = (nxhi_out-nxlo_out+1)*(nyhi_out-nylo_out+1)*(nzhi_out-nzlo_out+1);
  nfft_brick = (nxhi_in-nxlo_in+1)*(nyhi_in-nylo_in+1)*(nzhi_in-nzlo_in+1);
  nfft = (nxhi_fft-nxlo_fft+1)*(nyhi_fft-nylo_fft+1)*(nzhi_fft-nzlo_fft+1);
  nfft_both = MAX(nfft,nfft_brick);

  memory->create3d_offset(density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"structure/factor/fft:density_brick");
  memory->create(density_fft,nfft_both,"structure/factor/fft:density_fft");
  memory->create(work1,2*nfft_both,"structure/factor/fft:work1");
  memory->create2d_offset(rho1d,3,-order/2,order/2,"structure/factor/fft:rho1d");

  int tmp;
  fft1 = new FFT3d(lmp,world,nx_sf,ny_sf,nz_sf,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   0,0,&tmp,0,0);
  remap = new Remap(lmp,world,
                    nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                    nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                    1,0,0,FFT_PRECISION,0,0);

  mesh_allocated = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactorFFT::deallocate_mesh()
{
  if (!mesh_allocated) return;
  delete gc; gc = nullptr;
  delete fft1; fft1 = nullptr;
  delete remap; remap = nullptr;
  memory->destroy(gc_buf1);
  memory->destroy(gc_buf2);
  memory->destroy3d_offset(density_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy(density_fft);
  memory->destroy(work1);
  memory->destroy2d_offset(rho1d,-order/2);
  mesh_allocated = 0;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactorFFT::allocate_kvecs()
{
  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];
}

void ComputeStructureFactorFFT::deallocate_kvecs()
{
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;
  delete [] norms;
  delete [] ksq2unique;
}

/* ----------------------------------------------------------------------
   compute the structure factor via the FFT mesh
------------------------------------------------------------------------- */

void ComputeStructureFactorFFT::compute_array()
{
  if (atom->nmax > nmax) {
    memory->destroy(part2grid);
    nmax = atom->nmax;
    memory->create(part2grid,nmax,3,"structure/factor/fft:part2grid");
  }

  boxlo = domain->boxlo;

  particle_map();
  make_rho();

  gc->reverse_comm(Grid3d::COMPUTE,this,REVERSE_RHO,1,sizeof(FFT_SCALAR),
                   gc_buf1,gc_buf2,MPI_FFT_SCALAR);

  brick2fft();

  // pack the (real) FFT-decomposition density into the complex work array,
  // then forward FFT -> rho_mesh(k) in work1 (x-pencil layout)

  int np = 0;
  for (int i = 0; i < nfft; i++) {
    work1[np++] = density_fft[i];
    work1[np++] = (FFT_SCALAR) 0.0;
  }

  fft1->compute(work1,work1,FFT3d::FORWARD);

  double natoms = group->count(igroup);
  if (natoms == 0.0) natoms = 1.0;

  // accumulate |rho(k)|^2/N over the k-vectors of each |k|-shell that this proc
  // owns in the FFT decomposition, deconvolving the KB window analytically

  double *sloc = new double[kunique];
  double *sall = new double[kunique];
  for (int u = 0; u < kunique; u++) sloc[u] = 0.0;

  int nyfft = nyhi_fft - nylo_fft + 1;
  int nxfft = nxhi_fft - nxlo_fft + 1;

  for (int k = 0; k < kcount; k++) {
    int l = kxvecs[k];
    int m = kyvecs[k];
    int nn = kzvecs[k];

    int iy = (m % ny_sf + ny_sf) % ny_sf;
    int iz = (nn % nz_sf + nz_sf) % nz_sf;
    if (iy < nylo_fft || iy > nyhi_fft || iz < nzlo_fft || iz > nzhi_fft) continue;
    int ix = (l % nx_sf + nx_sf) % nx_sf;

    int idx = ((iz-nzlo_fft)*nyfft + (iy-nylo_fft))*nxfft + (ix-nxlo_fft);
    double re = work1[2*idx];
    double im = work1[2*idx+1];

    double w = kb_window(l,nx_sf) * kb_window(m,ny_sf) * kb_window(nn,nz_sf);
    re /= w; im /= w;

    int sqk_int = l*l + m*m + nn*nn;
    sloc[ksq2unique[sqk_int]] += (re*re + im*im)/natoms;
  }

  MPI_Allreduce(sloc,sall,kunique,MPI_DOUBLE,MPI_SUM,world);

  for (int k = 0; k < kcount; k++) {
    int sqk_int = kxvecs[k]*kxvecs[k] + kyvecs[k]*kyvecs[k] + kzvecs[k]*kzvecs[k];
    int kunq = ksq2unique[sqk_int];
    array[kunq][0] = unitk[0]*sqrt((double) sqk_int);
    array[kunq][1] = sall[kunq]/norms[sqk_int];
    array[kunq][2] = norms[sqk_int];
  }

  delete [] sloc;
  delete [] sall;
}

/* ----------------------------------------------------------------------
   map atoms to the lower-left grid point of their stencil
------------------------------------------------------------------------- */

void ComputeStructureFactorFFT::particle_map()
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int flag = 0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) { part2grid[i][0] = part2grid[i][1] = part2grid[i][2] = OFFSET; continue; }
    int nx = static_cast<int>((x[i][0]-boxlo[0])*delxinv+shift) - OFFSET;
    int ny = static_cast<int>((x[i][1]-boxlo[1])*delyinv+shift) - OFFSET;
    int nz = static_cast<int>((x[i][2]-boxlo[2])*delzinv+shift) - OFFSET;
    part2grid[i][0] = nx;
    part2grid[i][1] = ny;
    part2grid[i][2] = nz;
    if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
        ny+nlower < nylo_out || ny+nupper > nyhi_out ||
        nz+nlower < nzlo_out || nz+nupper > nzhi_out) flag = 1;
  }
  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) error->all(FLERR,"Out of range atoms - cannot compute structure/factor/fft");
}

/* ----------------------------------------------------------------------
   spread atoms (unit weight) onto the grid with the KB kernel
------------------------------------------------------------------------- */

void ComputeStructureFactorFFT::make_rho()
{
  memset(&(density_brick[nzlo_out][nylo_out][nxlo_out]),0,ngrid*sizeof(FFT_SCALAR));

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int nx = part2grid[i][0];
    int ny = part2grid[i][1];
    int nz = part2grid[i][2];
    FFT_SCALAR dx = nx+shiftone - (x[i][0]-boxlo[0])*delxinv;
    FFT_SCALAR dy = ny+shiftone - (x[i][1]-boxlo[1])*delyinv;
    FFT_SCALAR dz = nz+shiftone - (x[i][2]-boxlo[2])*delzinv;

    compute_kb1d(dx,dy,dz);

    for (int n = nlower; n <= nupper; n++) {
      int mz = n+nz;
      FFT_SCALAR y0 = rho1d[2][n];
      for (int mm = nlower; mm <= nupper; mm++) {
        int my = mm+ny;
        FFT_SCALAR x0 = y0*rho1d[1][mm];
        for (int ll = nlower; ll <= nupper; ll++) {
          int mx = ll+nx;
          density_brick[mz][my][mx] += x0*rho1d[0][ll];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Kaiser-Bessel stencil weights:  w(d) = I0(beta*sqrt(1-(2d/W)^2)), W=order
   d = signed distance (grid units) of stencil point k from the particle
------------------------------------------------------------------------- */

void ComputeStructureFactorFFT::compute_kb1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy,
                                             const FFT_SCALAR &dz)
{
  double invhalf = 2.0/order;
  for (int k = nlower; k <= nupper; k++) {
    double ax = (k + dx)*invhalf;  double sx = 1.0 - ax*ax; if (sx < 0.0) sx = 0.0;
    double ay = (k + dy)*invhalf;  double sy = 1.0 - ay*ay; if (sy < 0.0) sy = 0.0;
    double az = (k + dz)*invhalf;  double sz = 1.0 - az*az; if (sz < 0.0) sz = 0.0;
    rho1d[0][k] = (FFT_SCALAR) bessel_i0(kb_beta*sqrt(sx));
    rho1d[1][k] = (FFT_SCALAR) bessel_i0(kb_beta*sqrt(sy));
    rho1d[2][k] = (FFT_SCALAR) bessel_i0(kb_beta*sqrt(sz));
  }
}

/* ----------------------------------------------------------------------
   analytic Fourier transform of the KB kernel at integer mode m of an N-grid
   psihat(omega) = W * sinh(s)/s,  s = sqrt(beta^2-(omega*W/2)^2), omega=2*pi*m/N
   (sin branch when the argument is imaginary)
------------------------------------------------------------------------- */

double ComputeStructureFactorFFT::kb_window(int m, int N)
{
  double arg = MY_PI*order*((double) m)/N;     // = omega*W/2
  double t = kb_beta*kb_beta - arg*arg;
  if (t > 0.0) {
    double s = sqrt(t);
    return order*sinh(s)/s;
  } else if (t < 0.0) {
    double s = sqrt(-t);
    return order*sin(s)/s;
  }
  return (double) order;
}

/* ---------------------------------------------------------------------- */

void ComputeStructureFactorFFT::brick2fft()
{
  int n = 0;
  for (int iz = nzlo_in; iz <= nzhi_in; iz++)
    for (int iy = nylo_in; iy <= nyhi_in; iy++)
      for (int ix = nxlo_in; ix <= nxhi_in; ix++)
        density_fft[n++] = density_brick[iz][iy][ix];
  remap->perform(density_fft,density_fft,work1);
}

/* ----------------------------------------------------------------------
   Grid3d ghost-cell callbacks (sum ghost density into owned cells)
------------------------------------------------------------------------- */

void ComputeStructureFactorFFT::pack_reverse_grid(int /*flag*/, void *vbuf, int nlist, int *list)
{
  auto *buf = (FFT_SCALAR *) vbuf;
  FFT_SCALAR *src = &density_brick[nzlo_out][nylo_out][nxlo_out];
  for (int i = 0; i < nlist; i++) buf[i] = src[list[i]];
}

void ComputeStructureFactorFFT::unpack_reverse_grid(int /*flag*/, void *vbuf, int nlist, int *list)
{
  auto *buf = (FFT_SCALAR *) vbuf;
  FFT_SCALAR *dest = &density_brick[nzlo_out][nylo_out][nxlo_out];
  for (int i = 0; i < nlist; i++) dest[list[i]] += buf[i];
}

/* ---------------------------------------------------------------------- */

int ComputeStructureFactorFFT::factorable(int n)
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

void ComputeStructureFactorFFT::procs2grid2d(int nprocs_in, int nx, int ny, int &px, int &py)
{
  int bestsurf = 2*(nx+ny);
  int bestboxx = 0, bestboxy = 0;
  int ipx = 1;
  while (ipx <= nprocs_in) {
    if (nprocs_in % ipx == 0) {
      int ipy = nprocs_in/ipx;
      int boxx = nx/ipx; if (nx % ipx) boxx++;
      int boxy = ny/ipy; if (ny % ipy) boxy++;
      int surf = boxx + boxy;
      if ((surf < bestsurf) || ((surf == bestsurf) && (boxx*boxy > bestboxx*bestboxy))) {
        bestsurf = surf; bestboxx = boxx; bestboxy = boxy; px = ipx; py = ipy;
      }
    }
    ipx++;
  }
}

/* ----------------------------------------------------------------------
   modified Bessel function I0(x), Abramowitz & Stegun 9.8.1 / 9.8.2
------------------------------------------------------------------------- */

double ComputeStructureFactorFFT::bessel_i0(double x)
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
