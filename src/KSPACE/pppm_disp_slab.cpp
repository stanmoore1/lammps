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
   Contributing authors: Stan Moore (SNL), Dean Wheeler (BYU)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Mesh-accelerated damped slab-based dispersion Ewald (pppm/disp/slab).
   The dispersion-weighted density (geometric mixing) varies only in z, so the
   smooth reciprocal part is a 1-D convolution in z: spread the B-weighted
   density onto a z grid, FFT in z, apply the damped influence function,
   inverse-FFT the z-force field, and interpolate.  The reciprocal energy/force
   reproduce the exact ewald/disp/slab (damped) result as the grid is refined;
   the real-space slab correction corr() and the H/IK pressure profiles are the
   identical (verified) formulas shared with ewald/disp/slab.  Damped (SSB) only.

   References: S. Moore, dissertation (BYU); this paper.
------------------------------------------------------------------------- */

#include "pppm_disp_slab.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "pair.h"
#include "utils.h"

#include <cctype>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr double EULER = 0.57721566490153286061;
static constexpr int OFFSET = 16384;
static constexpr int MAXORDER = 8;

/* ---------------------------------------------------------------------- */

PPPMDispSlab::PPPMDispSlab(LAMMPS *lmp) :
    KSpace(lmp), pt_profile(nullptr), pn_profile(nullptr), B(nullptr), dens(nullptr), fre(nullptr),
    fim(nullptr), Gk(nullptr), fz_grid(nullptr), ugrid(nullptr), rho_coeff(nullptr), peatom(nullptr)
{
  dispersionflag = 1;
  contour_flag = 0;
  profile_flag = 0;
  npro = 0;
  dim = 2;
  lat1 = 0;
  lat2 = 1;
  nz = 0;
  order = 6;
  corr_mode = 0;
  bin_dz_user = 0.0;
  bin_nbins = 0;
  g_ewald_set = 0.0;
  order_allocated = 0;
  nmax = 0;
  accuracy_relative = 0.0;
}

/* ---------------------------------------------------------------------- */

PPPMDispSlab::~PPPMDispSlab()
{
  if (copymode) return;
  delete[] B;
  memory->destroy(dens);
  memory->destroy(fre);
  memory->destroy(fim);
  memory->destroy(Gk);
  memory->destroy(fz_grid);
  memory->destroy(ugrid);
  memory->destroy(peatom);
  memory->destroy(pt_profile);
  memory->destroy(pn_profile);
  if (rho_coeff) memory->destroy(rho_coeff);
}

/* ---------------------------------------------------------------------- */

void PPPMDispSlab::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal kspace_style {} command", force->kspace_style);
  accuracy_relative = fabs(utils::numeric(FLERR, arg[0], false, lmp));
  if (accuracy_relative > 1.0)
    error->all(FLERR, "Invalid relative accuracy {:g} for kspace_style {}", accuracy_relative,
               force->kspace_style);
}

/* ----------------------------------------------------------------------
   per-style kspace_modify keywords (the base KSpace parser also handles the
   dispersion keywords mesh/disp, order/disp, and gewald/disp for this style):
     corr raw|bin [dz]    -- real-space correction: pairwise or z-binned
     contour h|ik         -- pressure-profile contour
     pressure/profile <n> -- enable P_T(z)/P_N(z) on an n-point z grid
------------------------------------------------------------------------- */

int PPPMDispSlab::modify_param(int narg, char **arg)
{
  // mesh/disp, order/disp, gewald/disp are consumed by the base KSpace parser
  // (they set nz_pppm_6/gridflag_6, order_6, g_ewald_6/gewaldflag_6).
  if (strcmp(arg[0], "corr") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify corr", error);
    if (strcmp(arg[1], "raw") == 0) {
      corr_mode = 0;
    } else if (strcmp(arg[1], "bin") == 0) {
      corr_mode = 1;
      if (narg >= 3 && arg[2][0] != '\0' && (isdigit(arg[2][0]) || arg[2][0] == '.')) {
        bin_dz_user = utils::numeric(FLERR, arg[2], false, lmp);
        if (bin_dz_user <= 0.0) error->all(FLERR, "kspace_modify corr bin <dz> must be > 0");
        return 3;
      }
    } else {
      error->all(FLERR, "kspace_modify corr must be raw or bin");
    }
    return 2;
  }
  if (strcmp(arg[0], "contour") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify contour", error);
    if (strcmp(arg[1], "h") == 0)
      contour_flag = 0;
    else if (strcmp(arg[1], "ik") == 0)
      contour_flag = 1;
    else
      error->all(FLERR, "kspace_modify contour must be h or ik");
    return 2;
  }
  if (strcmp(arg[0], "pressure/profile") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify pressure/profile", error);
    npro = utils::inumeric(FLERR, arg[1], false, lmp);
    profile_flag = (npro > 0);
    return 2;
  }
  if (strcmp(arg[0], "dim") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify dim", error);
    if (strcmp(arg[1], "x") == 0) dim = 0;
    else if (strcmp(arg[1], "y") == 0) dim = 1;
    else if (strcmp(arg[1], "z") == 0) dim = 2;
    else error->all(FLERR, "kspace_modify dim must be x, y, or z");
    lat1 = (dim + 1) % 3;
    lat2 = (dim + 2) % 3;
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void PPPMDispSlab::init()
{
  if (comm->me == 0) utils::logmesg(lmp, "PPPM slab-based dispersion Ewald (pppm/disp/slab) ...\n");

  triclinic_check();
  if (domain->dimension == 2) error->all(FLERR, "Cannot use pppm/disp/slab with 2d simulation");
  if (domain->triclinic) error->all(FLERR, "Cannot use pppm/disp/slab with triclinic box");
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR, "pppm/disp/slab requires periodic boundaries in all dimensions");
  if (slabflag)
    error->all(FLERR, "Cannot use slab correction (kspace_modify slab) with pppm/disp/slab");
  if (force->pair == nullptr)
    error->all(FLERR, "KSpace style pppm/disp/slab requires a pair style");

  // LJ cutoff from the pair style

  int itmp;
  double *p = (double *) force->pair->extract("cut_lj", itmp);
  if (p == nullptr) p = (double *) force->pair->extract("cut_LJ", itmp);
  if (p == nullptr)
    error->all(FLERR, "Pair style is incompatible with kspace_style pppm/disp/slab");
  cutoff = *p;
  rc2 = cutoff * cutoff;

  // accuracy in force units

  two_charge();
  if (accuracy_absolute >= 0.0)
    accuracy = accuracy_absolute;
  else
    accuracy = accuracy_relative * two_charge_force;

  // per-type dispersion amplitude B = 2 sqrt(eps) sigma^3 (geometric mixing).
  // kspace->init() runs before pair->init(), so lj4 may not be populated yet;
  // epsilon/sigma (set by pair_coeff) give the identical value B = sqrt(lj4).

  int n = atom->ntypes, dim;
  auto **eps = (double **) force->pair->extract("epsilon", dim);
  auto **sig = (double **) force->pair->extract("sigma", dim);
  if (eps == nullptr || sig == nullptr)
    error->all(FLERR, "Pair style does not provide epsilon/sigma for pppm/disp/slab");
  delete[] B;
  B = new double[n + 1];
  B[0] = 0.0;
  for (int t = 1; t <= n; t++) B[t] = 2.0 * sqrt(eps[t][t]) * sig[t][t] * sig[t][t] * sig[t][t];

  // stencil order from kspace_modify order/disp (base member; default 5)
  order = order_6;
  if (order < 2 || order > MAXORDER)
    error->all(FLERR, "pppm/disp/slab order/disp must be between 2 and {}", MAXORDER);

  estimate_params();    // sets g_ewald and the z grid size nz

  setup();

  if (comm->me == 0) {
    utils::logmesg(lmp, "  damped, z grid = {}, stencil order = {}, g_ewald = {:.6g}\n", nz, order,
                   g_ewald);
    utils::logmesg(lmp, "  estimated absolute RMS force accuracy = {:.6g}\n",
                   estimated_force_accuracy);
    utils::logmesg(lmp, "  estimated relative force accuracy = {:.6g}\n",
                   estimated_force_accuracy / two_charge_force);
  }
}

/* ----------------------------------------------------------------------
   Hockney-Eastwood optimal-influence-function residual (ik differentiation),
   specialized to the 1-D z grid: the only aliases are in z, k_a=(m+a*nz)*unitk.
   Per Brillouin-zone mode m (k=m*unitk):
     Q(m) = sum_a D(k_a)^2 k_a^2  -  [sum_a u2_a D(k_a) (k k_a)]^2 / [k^2 (sum_a u2_a)^2]
   with the de-convolved potential coefficient D(k)=GU(k)=coef*Bk(k) (the same
   damped influence as the energy) and u2_a = W(k_a)^2 = sinc(pi(m+a nz)/nz)^(2 order).
   The RMS per-atom force error is then sqrt(sum_m Q) * b2 / sqrt(N), b2=sum B_i^2
   (random-phase model; validated to ~1.4x against measured forces).
------------------------------------------------------------------------- */

double PPPMDispSlab::compute_qopt(int ngrid, int ord)
{
  const double unitk = 2.0 * MY_PI / zprd;
  const double sqpi = sqrt(MY_PI);
  const double coefD = -2.0 * MY_PI * sqpi / (24.0 * volume);    // GU = coefD*Bk
  const int nb = 3;                                              // alias range
  double Q = 0.0;

  for (int m = 1; m <= ngrid / 2; m++) {
    double k = m * unitk;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int al = -nb; al <= nb; al++) {
      int meff = m + al * ngrid;
      double ka = meff * unitk, ak = fabs(ka);
      double D = 0.0;
      if (ak > 0.0) {
        double b = ak / (2.0 * g_ewald), b2 = b * b, b3 = b2 * b;
        double Bk = ak * ak * ak * (sqpi * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));
        D = coefD * Bk;
      }
      double a = MY_PI * meff / ngrid;
      double w = (fabs(a) < 1.0e-12) ? 1.0 : sin(a) / a;
      double u2 = pow(w, 2 * ord);
      s1 += D * D * ka * ka;
      s2 += u2 * D * (k * ka);
      s3 += u2;
    }
    double Qm = s1 - s2 * s2 / (k * k * s3 * s3);
    if (Qm > 0.0) Q += 2.0 * Qm;    // +/- m
  }
  return Q;
}

/* ----------------------------------------------------------------------
   choose g_ewald and the z grid size nz from the target force accuracy.

   g_ewald sets the smooth/sharp split width (cost tradeoff: larger g => sharper
   real-space corr but wider reciprocal spectrum needing a finer grid).  Use the
   damped-tail heuristic (g*rcut)^2 = -2 ln(accuracy_rel), honoring a user
   kspace_modify gewald/disp override.

   nz is the accuracy control: the mesh RMS force error
     df = sqrt(qopt(nz)) * b2 / sqrt(N)   (1-D Hockney-Eastwood, b2 = sum B_i^2)
   captures the aliasing/interpolation AND truncation error of the grid.  Pick
   the smallest power-of-two nz with df < accuracy.
------------------------------------------------------------------------- */

void PPPMDispSlab::estimate_params()
{
  set_grid_params();

  double acc = accuracy / two_charge_force;
  if (acc <= 0.0 || acc >= 1.0) acc = 1.0e-4;
  if (gewaldflag_6)
    g_ewald = g_ewald_6;    // kspace_modify gewald/disp
  else if (gewaldflag)
    ;    // kspace_modify gewald (g_ewald already set by the base parser)
  else
    g_ewald = sqrt(-2.0 * log(acc)) / cutoff;
  g_ewald_set = g_ewald;

  // dispersion sum b2 = sum_i B_i^2 (full system)

  int *type = atom->type;
  int nlocal = atom->nlocal;
  double b2_local = 0.0;
  for (int i = 0; i < nlocal; i++) b2_local += B[type[i]] * B[type[i]];
  double b2;
  MPI_Allreduce(&b2_local, &b2, 1, MPI_DOUBLE, MPI_SUM, world);
  double natoms = (double) atom->natoms;
  if (natoms < 1.0) natoms = 1.0;
  const double pref = b2 / sqrt(natoms);

  if (gridflag_6 && nz_pppm_6 > 0) {
    nz = 1;
    while (nz < nz_pppm_6) nz <<= 1;    // round up to a power of two for the FFT
  } else {
    int ngrid = 16;
    while (ngrid < 65536) {
      double df = sqrt(compute_qopt(ngrid, order)) * pref;
      if (df < accuracy) break;
      ngrid <<= 1;
    }
    nz = ngrid;
  }
  if (nz < 8) nz = 8;

  estimated_force_accuracy = sqrt(compute_qopt(nz, order)) * pref;
}

/* ----------------------------------------------------------------------
   geometry, grid spacing, and B-spline stencil parameters
------------------------------------------------------------------------- */

void PPPMDispSlab::set_grid_params()
{
  lat1 = (dim + 1) % 3;
  lat2 = (dim + 2) % 3;
  zprd = domain->prd[dim];
  area = domain->prd[lat1] * domain->prd[lat2];
  volume = area * zprd;
  zlo = domain->boxlo[dim];

  // grid-assignment shift and stencil bounds (LAMMPS PPPM convention)
  if (order % 2)
    shiftone = 0.0;
  else
    shiftone = 0.5;
  nlower = -(order - 1) / 2;
  nupper = order / 2;
}

/* ---------------------------------------------------------------------- */

void PPPMDispSlab::setup()
{
  set_grid_params();
  delzinv = nz / zprd;

  memory->destroy(dens);
  memory->destroy(fre);
  memory->destroy(fim);
  memory->destroy(Gk);
  memory->destroy(fz_grid);
  memory->destroy(ugrid);
  memory->create(dens, nz, "pppm/disp/slab:dens");
  memory->create(fre, nz, "pppm/disp/slab:fre");
  memory->create(fim, nz, "pppm/disp/slab:fim");
  memory->create(Gk, nz, "pppm/disp/slab:Gk");
  memory->create(fz_grid, nz, "pppm/disp/slab:fz_grid");
  memory->create(ugrid, nz, "pppm/disp/slab:ugrid");

  if (rho_coeff == nullptr || order != order_allocated) {
    if (rho_coeff) memory->destroy(rho_coeff);
    memory->create(rho_coeff, order, order, "pppm/disp/slab:rho_coeff");
    order_allocated = order;
  }
  compute_rho_coeff();

  influence_function();

  // size the corr bin grid to the requested force accuracy (auto, unless the
  // user fixed the width with kspace_modify corr bin <dz>)
  if (corr_mode == 1 && bin_dz_user <= 0.0) calibrate_bin();
}

/* ----------------------------------------------------------------------
   de-convolved damped influence function on the z grid modes.

   The PPPM mesh energy E = sum_{m=0}^{nz-1} Gk[m] |rho_hat_m|^2 (full FFT
   spectrum) must equal the exact ewald/disp/slab energy
   E = GU[0]|S_0|^2 + sum_{k>=1} GU[k]|S_k|^2.  Matching the spectra term by
   term (each +/- mode appears once in the FFT sum) gives the physical
   per-mode coefficient W_E(k) = GU[0] for m=0 and GU[|m|]/2 for m != 0.
   De-convolving the order-p assignment (transfer W(k)=sinc(pi m/nz)^order):
     Gk[m] = W_E(k_m) / W(k_m)^2 .
   GU is the damped ewald/disp/slab coefficient coef*Bk (coef=-2pi^1.5/24V).
------------------------------------------------------------------------- */

void PPPMDispSlab::influence_function()
{
  const double sqpi = sqrt(MY_PI);
  const double coef = -2.0 * MY_PI * sqpi / (24.0 * volume);
  // m=0 (homogeneous tail) term: W_E(0) = GU[0] = -pi^1.5 g^3 / (6 V)
  Gk[0] = -MY_PI * sqpi * g_ewald * g_ewald * g_ewald / (6.0 * volume);
  for (int m = 1; m < nz; m++) {
    int mm = (m <= nz / 2) ? m : m - nz;    // signed mode index
    double k = mm * 2.0 * MY_PI / zprd;
    double ak = fabs(k);
    double b = ak / (2.0 * g_ewald), b2 = b * b, b3 = b2 * b;
    double Bk = ak * ak * ak * (sqpi * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));
    double WE = 0.5 * coef * Bk;    // GU[|mm|]/2  (full-spectrum per-mode coeff)
    double s = sin(MY_PI * mm / nz) / (MY_PI * mm / nz);
    double w2 = pow(s, 2 * order);
    Gk[m] = WE / w2;
  }
}

/* ----------------------------------------------------------------------
   B-spline assignment polynomial coefficients (LAMMPS PPPM convention).
   rho_coeff[l][s] gives the order-l term for stencil point s = 0..order-1
   (grid offset nlower+s); weight at fractional offset dz = sum_l rho_coeff[l][s] dz^l.
------------------------------------------------------------------------- */

void PPPMDispSlab::compute_rho_coeff()
{
  int j, k, l, m;
  double s;
  const int o = order;

  // a[l][k+o], k in [-o, o]
  auto *adata = new double[o * (2 * o + 1)];
  auto A = [&](int ll, int kk) -> double & {
    return adata[ll * (2 * o + 1) + (kk + o)];
  };
  for (k = -o; k <= o; k++)
    for (l = 0; l < o; l++) A(l, k) = 0.0;

  A(0, 0) = 1.0;
  for (j = 1; j < o; j++) {
    for (k = -j; k <= j; k += 2) {
      s = 0.0;
      for (l = 0; l < j; l++) {
        A(l + 1, k) = (A(l, k + 1) - A(l, k - 1)) / (l + 1);
        s += pow(0.5, (double) l + 1) * (A(l, k - 1) + pow(-1.0, (double) l) * A(l, k + 1)) /
            (l + 1);
      }
      A(0, k) = s;
    }
  }

  m = 0;    // maps to stencil point s = 0 .. order-1 (grid offsets nlower..nupper)
  for (k = -(o - 1); k < o; k += 2) {
    for (l = 0; l < o; l++) rho_coeff[l][m] = A(l, k);
    m++;
  }

  delete[] adata;
}

/* ----------------------------------------------------------------------
   assignment weights w[0..order-1] at fractional offset dz (Horner in dz)
------------------------------------------------------------------------- */

void PPPMDispSlab::compute_rho1d(double dz, double *w)
{
  for (int s = 0; s < order; s++) {
    double r = 0.0;
    for (int l = order - 1; l >= 0; l--) r = rho_coeff[l][s] + r * dz;
    w[s] = r;
  }
}

/* ----------------------------------------------------------------------
   d/d(dz) of the assignment weights: dw[s] = sum_{l>=1} l*rho_coeff[l][s] dz^(l-1)
   (used for the energy-conserving B-spline corr force)
------------------------------------------------------------------------- */

void PPPMDispSlab::compute_drho1d(double dz, double *dw)
{
  for (int s = 0; s < order; s++) {
    double r = 0.0;
    for (int l = order - 1; l >= 1; l--) r = l * rho_coeff[l][s] + r * dz;
    dw[s] = r;
  }
}

/* ----------------------------------------------------------------------
   spread the B-weighted density onto the global z grid (Allreduce)
------------------------------------------------------------------------- */

void PPPMDispSlab::make_rho()
{
  for (int g = 0; g < nz; g++) dens[g] = 0.0;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double w[MAXORDER];

  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zlo) * delzinv;
    int g0 = (int) (u + (order % 2 ? OFFSET + 0.5 : OFFSET)) - OFFSET;    // nearest grid pt
    double dz = g0 + shiftone - u;
    compute_rho1d(dz, w);
    const double bi = B[type[i]];
    for (int s = 0; s < order; s++) {
      int g = g0 + nlower + s;
      g = ((g % nz) + nz) % nz;
      dens[g] += bi * w[s];
    }
  }

  double *tmp;
  memory->create(tmp, nz, "pppm/disp/slab:tmp");
  MPI_Allreduce(dens, tmp, nz, MPI_DOUBLE, MPI_SUM, world);
  for (int g = 0; g < nz; g++) dens[g] = tmp[g];
  memory->destroy(tmp);
}

/* ----------------------------------------------------------------------
   radix-2 in-place FFT (n a power of two); sign=-1 forward, +1 inverse,
   both unnormalized (X_m = sum_g x_g e^{-2pi i m g/n} for sign=-1)
------------------------------------------------------------------------- */

void PPPMDispSlab::fft1d(double *re, double *im, int n, int sign)
{
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      double tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      double ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = sign * 2.0 * MY_PI / len;
    double wr = cos(ang), wi = sin(ang);
    for (int i = 0; i < n; i += len) {
      double cr = 1.0, ci = 0.0;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k, bb = i + k + len / 2;
        double tr = re[bb] * cr - im[bb] * ci;
        double ti = re[bb] * ci + im[bb] * cr;
        re[bb] = re[a] - tr;
        im[bb] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        double ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   FFT the density, accumulate the reciprocal energy and tangential virial,
   build the z-force field, and (if per-atom requested) the potential field.

   Energy:  E = sum_m Gk[m] |rho_hat_m|^2   (rho_hat = unnormalized FFT of dens).
   Force field:  f_grid = IFFT[ -i k_m * 2 Gk[m] * rho_hat_m ]  (= exact z-gradient
   of the mesh energy, hence energy-conserving, matching GF=2k*GU in ewald).
   Potential field (per-atom):  u_grid = IFFT[ 2 Gk[m] * rho_hat_m ].
------------------------------------------------------------------------- */

void PPPMDispSlab::poisson()
{
  for (int g = 0; g < nz; g++) {
    fre[g] = dens[g];
    fim[g] = 0.0;
  }
  fft1d(fre, fim, nz, -1);    // rho_hat in (fre,fim)

  // reciprocal energy (full system value) + tangential virial (xx=yy, GT=GU)

  double e = 0.0;
  for (int m = 0; m < nz; m++) e += Gk[m] * (fre[m] * fre[m] + fim[m] * fim[m]);
  e_recip_mesh = e;
  if (eflag_global) energy += e;
  if (vflag_global) {
    virial[lat1] += e;
    virial[lat2] += e;
  }

  // per-atom potential field u_grid = IFFT[2 Gk rho_hat]

  if (evflag_atom) {
    double *ur, *ui;
    memory->create(ur, nz, "pppm/disp/slab:ur");
    memory->create(ui, nz, "pppm/disp/slab:ui");
    for (int m = 0; m < nz; m++) {
      double g2 = 2.0 * Gk[m];
      ur[m] = g2 * fre[m];
      ui[m] = g2 * fim[m];
    }
    fft1d(ur, ui, nz, +1);
    for (int g = 0; g < nz; g++) ugrid[g] = ur[g];
    memory->destroy(ur);
    memory->destroy(ui);
  }

  // z-force field: F_hat_m = -i k_m * 2 Gk[m] * rho_hat_m

  for (int m = 0; m < nz; m++) {
    int mm = (m <= nz / 2) ? m : m - nz;
    double k = mm * 2.0 * MY_PI / zprd;
    double g2k = 2.0 * Gk[m] * k;
    double a = fre[m], bb = fim[m];
    fre[m] = g2k * bb;    // Re(-i k 2Gk (a+ib)) = 2Gk k b
    fim[m] = -g2k * a;    // Im = -2Gk k a
  }
  fft1d(fre, fim, nz, +1);
  for (int g = 0; g < nz; g++) fz_grid[g] = fre[g];
}

/* ----------------------------------------------------------------------
   interpolate the z-force field (and per-atom energy/virial) to atoms
------------------------------------------------------------------------- */

void PPPMDispSlab::fieldforce()
{
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double w[MAXORDER];

  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zlo) * delzinv;
    int g0 = (int) (u + (order % 2 ? OFFSET + 0.5 : OFFSET)) - OFFSET;
    double dz = g0 + shiftone - u;
    compute_rho1d(dz, w);
    const double bi = B[type[i]];

    double fz = 0.0, uu = 0.0;
    for (int s = 0; s < order; s++) {
      int g = g0 + nlower + s;
      g = ((g % nz) + nz) % nz;
      fz += w[s] * fz_grid[g];
      if (evflag_atom) uu += w[s] * ugrid[g];
    }
    f[i][dim] += bi * fz;

    if (evflag_atom) {
      double pe = 0.5 * bi * uu;    // per-atom reciprocal energy
      peatom[i] += pe;
      if (vflag_atom) {
        vatom[i][lat1] += pe;    // tangential (GT=GU)
        vatom[i][lat2] += pe;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PPPMDispSlab::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // grow per-atom energy buffer if needed

  if (atom->nmax > nmax) {
    memory->destroy(peatom);
    nmax = atom->nmax;
    memory->create(peatom, nmax, "pppm/disp/slab:peatom");
  }
  if (evflag_atom)
    for (int i = 0; i < atom->nlocal; i++) peatom[i] = 0.0;

  make_rho();
  poisson();
  fieldforce();

  // damped real-space slab correction (adds to energy, corr_energy, tangential
  // virial, per-atom energy buffer; zz set from the trace below)

  corr_energy = 0.0;
  corr();

  // normal (zz) virial from the exact 1/r^6 virial trace: sum r.f = 6 U, so
  // virial_zz = 6*E_kspace - virial_xx - virial_yy (total pressure, and per-atom
  // the IK-contour local normal pressure).

  if (vflag_global) virial[dim] = 6.0 * (e_recip_mesh + corr_energy) - virial[lat1] - virial[lat2];
  if (vflag_atom)
    for (int i = 0; i < atom->nlocal; i++)
      vatom[i][dim] = 6.0 * peatom[i] - vatom[i][lat1] - vatom[i][lat2];

  if (eflag_atom)
    for (int i = 0; i < atom->nlocal; i++) eatom[i] += peatom[i];

  if (profile_flag) compute_pressure_profile();
}

/* ----------------------------------------------------------------------
   --- shared with ewald/disp/slab (identical, verified formulas) ---
------------------------------------------------------------------------- */

void PPPMDispSlab::corr()
{
  if (corr_mode == 1)
    corr_bin();
  else
    corr_raw();
}

/* ----------------------------------------------------------------------
   damped slab-correction kernels at squared z-separation x2 = (z_i-z_j)^2
------------------------------------------------------------------------- */

void PPPMDispSlab::corr_kernels(double x2, double &w2, double &f2, double &pt2)
{
  const double g2 = g_ewald * g_ewald;
  const double g4 = g2 * g2, g6 = g4 * g2, g8 = g4 * g4, g10 = g8 * g2, g12 = g10 * g2;
  const double rc4 = rc2 * rc2, rc6 = rc4 * rc2;
  const double area = domain->prd[(dim + 1) % 3] * domain->prd[(dim + 2) % 3];

  if (x2 < 1.0e-3) {
    const double x4 = x2 * x2, x6 = x4 * x2;
    w2 = 0.5 * MY_PI *
        (0.5 * g4 - x2 * g6 / 3.0 + x4 * g8 / 8.0 - x6 * g10 / 30.0 +
         exp(-rc2 * g2) * (1.0 / rc4 + g2 / rc2) - 1.0 / rc4) /
        area;
    f2 = 2.0 * MY_PI * (g6 / 6.0 - x2 * g8 / 8.0 + x4 * g10 / 20.0 - x6 * g12 / 72.0) / area;
    pt2 = 0.5 * MY_PI *
        (0.5 * g4 - x2 * g6 / 3.0 + x4 * g8 / 8.0 - x6 * g10 / 30.0 +
         exp(-rc2 * g2) *
             (3.0 / rc4 - 2.0 * x2 / rc6 + g4 + 3.0 * g2 / rc2 - x2 * g4 / rc2 - x2 * g2 / rc4) -
         3.0 / rc4 + 2.0 * x2 / rc6) /
        area;
  } else {
    const double x4 = x2 * x2, x6 = x4 * x2;
    w2 = 0.5 * MY_PI *
        (1.0 / x4 - exp(-x2 * g2) * (1.0 / x4 + g2 / x2) + exp(-rc2 * g2) * (1.0 / rc4 + g2 / rc2) -
         1.0 / rc4) /
        area;
    f2 = 2.0 * MY_PI * (1.0 / x6 - exp(-x2 * g2) * (1.0 / x6 + g2 / x4 + 0.5 * g4 / x2)) / area;
    pt2 = 0.5 * MY_PI *
        (1.0 / x4 - exp(-x2 * g2) * (1.0 / x4 + g2 / x2) +
         exp(-rc2 * g2) *
             (3.0 / rc4 - 2.0 * x2 / rc6 + g4 + 3.0 * g2 / rc2 - x2 * g4 / rc2 - x2 * g2 / rc4) -
         3.0 / rc4 + 2.0 * x2 / rc6) /
        area;
  }
}

/* ----------------------------------------------------------------------
   exact pairwise slab correction (global z-gather; see ewald/disp/slab)
------------------------------------------------------------------------- */

void PPPMDispSlab::corr_raw()
{
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  int nprocs = comm->nprocs;
  int *recvcounts = new int[nprocs];
  int *displs = new int[nprocs];
  MPI_Allgather(&nlocal, 1, MPI_INT, recvcounts, 1, MPI_INT, world);
  int natoms_all = 0;
  for (int p = 0; p < nprocs; p++) {
    displs[p] = natoms_all;
    natoms_all += recvcounts[p];
  }
  int myoff = displs[comm->me];

  auto *zloc = new double[nlocal > 0 ? nlocal : 1];
  auto *bloc = new double[nlocal > 0 ? nlocal : 1];
  for (int i = 0; i < nlocal; i++) {
    zloc[i] = x[i][dim];
    bloc[i] = B[type[i]];
  }
  auto *zall = new double[natoms_all > 0 ? natoms_all : 1];
  auto *ball = new double[natoms_all > 0 ? natoms_all : 1];
  MPI_Allgatherv(zloc, nlocal, MPI_DOUBLE, zall, recvcounts, displs, MPI_DOUBLE, world);
  MPI_Allgatherv(bloc, nlocal, MPI_DOUBLE, ball, recvcounts, displs, MPI_DOUBLE, world);

  double w0, f0, pt0;
  corr_kernels(0.0, w0, f0, pt0);
  const double w2_self = 0.5 * w0, pt2_self = 0.5 * pt0;

  double e_local = 0.0;
  double v_local[2] = {0.0, 0.0};

  double bsqsum_local = 0.0;
  for (int i = 0; i < nlocal; i++) bsqsum_local += B[type[i]] * B[type[i]];
  e_local += bsqsum_local * w2_self;
  v_local[0] += bsqsum_local * pt2_self;
  v_local[1] += bsqsum_local * pt2_self;
  if (evflag_atom)
    for (int i = 0; i < nlocal; i++) peatom[i] += B[type[i]] * B[type[i]] * w2_self;
  if (vflag_atom)
    for (int i = 0; i < nlocal; i++) {
      vatom[i][lat1] += B[type[i]] * B[type[i]] * pt2_self;
      vatom[i][lat2] += B[type[i]] * B[type[i]] * pt2_self;
    }

  for (int i = 0; i < nlocal; i++) {
    const double zi = x[i][dim];
    const double bi = B[type[i]];
    const int iglob = myoff + i;
    double fz_i = 0.0;

    for (int jg = 0; jg < natoms_all; jg++) {
      if (jg == iglob) continue;
      double delz = zi - zall[jg];
      delz -= zprd * trunc(2.0 * delz / zprd);
      double x2 = delz * delz;
      if (x2 >= rc2) continue;

      double w2, f2, pt2;
      corr_kernels(x2, w2, f2, pt2);
      const double bij = bi * ball[jg];

      e_local += 0.5 * bij * w2;
      fz_i += delz * bij * f2;
      v_local[0] += 0.5 * bij * pt2;
      v_local[1] += 0.5 * bij * pt2;

      if (evflag_atom) peatom[i] += 0.5 * bij * w2;
      if (vflag_atom) {
        vatom[i][lat1] += 0.5 * bij * pt2;
        vatom[i][lat2] += 0.5 * bij * pt2;
      }
    }

    f[i][dim] += fz_i;
  }

  double e_all;
  MPI_Allreduce(&e_local, &e_all, 1, MPI_DOUBLE, MPI_SUM, world);
  corr_energy = e_all;
  if (eflag_global) energy += e_all;
  if (vflag_global) {
    double v_all[2];
    MPI_Allreduce(v_local, v_all, 2, MPI_DOUBLE, MPI_SUM, world);
    virial[lat1] += v_all[0];
    virial[lat2] += v_all[1];
  }

  delete[] recvcounts;
  delete[] displs;
  delete[] zloc;
  delete[] bloc;
  delete[] zall;
  delete[] ball;
}

/* ----------------------------------------------------------------------
   z-binned slab correction (1D particle-mesh CIC; see ewald/disp/slab)
------------------------------------------------------------------------- */

void PPPMDispSlab::corr_bin()
{
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  int nbins;
  if (bin_dz_user > 0.0)
    nbins = (int) (zprd / bin_dz_user + 0.5);
  else if (bin_nbins > 0)
    nbins = bin_nbins;    // tied to the requested accuracy (calibrate_bin)
  else
    nbins = (int) (zprd / MIN(0.025 / g_ewald, 0.025 * cutoff) + 0.5);
  if (nbins < 4) nbins = 4;
  const double dz = zprd / nbins;
  int nwin = (int) (cutoff / dz) + 1;
  if (nwin > nbins / 2) nwin = nbins / 2;

  auto *bdens = new double[nbins];
  auto *phiW = new double[nbins];
  auto *phiPT = new double[nbins];
  for (int b = 0; b < nbins; b++) bdens[b] = 0.0;

  // order-p B-spline assignment of the B-weighted density onto the corr grid
  // (reuses the reciprocal's rho_coeff; CIC is just the order=2 special case).
  const double delzc = 1.0 / dz;    // = nbins/zprd
  const double shift = (order % 2) ? OFFSET + 0.5 : OFFSET;
  auto *ag0 = new int[nlocal > 0 ? nlocal : 1];
  auto *adz = new double[nlocal > 0 ? nlocal : 1];
  double w[MAXORDER];
  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zlo) * delzc;
    int g0 = (int) (u + shift) - OFFSET;
    double dzf = g0 + shiftone - u;
    compute_rho1d(dzf, w);
    const double bi = B[type[i]];
    for (int s = 0; s < order; s++) {
      int g = g0 + nlower + s;
      g = ((g % nbins) + nbins) % nbins;
      bdens[g] += bi * w[s];
    }
    ag0[i] = g0;
    adz[i] = dzf;
  }

  auto *dens_all = new double[nbins];
  MPI_Allreduce(bdens, dens_all, nbins, MPI_DOUBLE, MPI_SUM, world);

  auto *Kw = new double[nwin + 1];
  auto *Kpt = new double[nwin + 1];
  for (int d = 0; d <= nwin; d++) {
    double xi = d * dz;
    double x2 = xi * xi;
    double w2, f2, pt2;
    if (x2 >= rc2) {
      Kw[d] = Kpt[d] = 0.0;
    } else {
      corr_kernels(x2, w2, f2, pt2);
      Kw[d] = w2;
      Kpt[d] = pt2;
    }
  }

  for (int b = 0; b < nbins; b++) {
    double sw = Kw[0] * dens_all[b];
    double spt = Kpt[0] * dens_all[b];
    for (int d = 1; d <= nwin; d++) {
      int bp = b + d;
      if (bp >= nbins) bp -= nbins;
      int bm = b - d;
      if (bm < 0) bm += nbins;
      double s = dens_all[bp] + dens_all[bm];
      sw += Kw[d] * s;
      spt += Kpt[d] * s;
    }
    phiW[b] = sw;
    phiPT[b] = spt;
  }

  double e = 0.0;
  for (int b = 0; b < nbins; b++) e += dens_all[b] * phiW[b];
  corr_energy = 0.5 * e;
  if (eflag_global) energy += corr_energy;
  if (vflag_global) {
    double vpt = 0.0;
    for (int b = 0; b < nbins; b++) vpt += dens_all[b] * phiPT[b];
    virial[lat1] += 0.5 * vpt;
    virial[lat2] += 0.5 * vpt;
  }

  // forces (exact z-gradient of the binned energy via the B-spline derivative)
  // and per-atom energy/virial (interpolated with the same assignment weights)
  double dw[MAXORDER];
  for (int i = 0; i < nlocal; i++) {
    int g0 = ag0[i];
    double dzf = adz[i];
    compute_rho1d(dzf, w);
    compute_drho1d(dzf, dw);
    const double bi = B[type[i]];
    double fz = 0.0, pe = 0.0, pt = 0.0;
    for (int s = 0; s < order; s++) {
      int g = g0 + nlower + s;
      g = ((g % nbins) + nbins) % nbins;
      fz += dw[s] * phiW[g];
      if (evflag_atom) pe += w[s] * phiW[g];
      if (vflag_atom) pt += w[s] * phiPT[g];
    }
    f[i][dim] += bi * delzc * fz;
    if (evflag_atom) peatom[i] += 0.5 * bi * pe;
    if (vflag_atom) {
      vatom[i][lat1] += 0.5 * bi * pt;
      vatom[i][lat2] += 0.5 * bi * pt;
    }
  }

  delete[] bdens;
  delete[] dens_all;
  delete[] phiW;
  delete[] phiPT;
  delete[] Kw;
  delete[] Kpt;
  delete[] ag0;
  delete[] adz;
}

/* ----------------------------------------------------------------------
   lean force-only z-binned corr at a given bin count, written to fzloc[nlocal].
   Same B-spline assignment + convolution as corr_bin() but no energy/virial/
   per-atom bookkeeping -- used by calibrate_bin() to size the grid.
------------------------------------------------------------------------- */

void PPPMDispSlab::corr_bin_force(int nbins, double *fzloc)
{
  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  const double dz = zprd / nbins;
  int nwin = (int) (cutoff / dz) + 1;
  if (nwin > nbins / 2) nwin = nbins / 2;

  auto *bdens = new double[nbins];
  for (int b = 0; b < nbins; b++) bdens[b] = 0.0;
  const double delzc = 1.0 / dz;
  const double shift = (order % 2) ? OFFSET + 0.5 : OFFSET;
  auto *ag0 = new int[nlocal > 0 ? nlocal : 1];
  auto *adz = new double[nlocal > 0 ? nlocal : 1];
  double w[MAXORDER];
  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zlo) * delzc;
    int g0 = (int) (u + shift) - OFFSET;
    double dzf = g0 + shiftone - u;
    compute_rho1d(dzf, w);
    const double bi = B[type[i]];
    for (int s = 0; s < order; s++) {
      int g = ((g0 + nlower + s) % nbins + nbins) % nbins;
      bdens[g] += bi * w[s];
    }
    ag0[i] = g0;
    adz[i] = dzf;
  }
  auto *dens_all = new double[nbins];
  MPI_Allreduce(bdens, dens_all, nbins, MPI_DOUBLE, MPI_SUM, world);

  auto *Kw = new double[nwin + 1];
  for (int d = 0; d <= nwin; d++) {
    double x2 = (d * dz) * (d * dz);
    double w2, f2, pt2;
    if (x2 >= rc2)
      Kw[d] = 0.0;
    else {
      corr_kernels(x2, w2, f2, pt2);
      Kw[d] = w2;
    }
  }
  auto *phiW = new double[nbins];
  for (int b = 0; b < nbins; b++) {
    double sw = Kw[0] * dens_all[b];
    for (int d = 1; d <= nwin; d++) {
      int bp = b + d;
      if (bp >= nbins) bp -= nbins;
      int bm = b - d;
      if (bm < 0) bm += nbins;
      sw += Kw[d] * (dens_all[bp] + dens_all[bm]);
    }
    phiW[b] = sw;
  }

  double dw[MAXORDER];
  for (int i = 0; i < nlocal; i++) {
    compute_drho1d(adz[i], dw);
    int g0 = ag0[i];
    double fz = 0.0;
    for (int s = 0; s < order; s++) {
      int g = ((g0 + nlower + s) % nbins + nbins) % nbins;
      fz += dw[s] * phiW[g];
    }
    fzloc[i] = B[type[i]] * delzc * fz;
  }

  delete[] bdens;
  delete[] dens_all;
  delete[] Kw;
  delete[] phiW;
  delete[] ag0;
  delete[] adz;
}

/* ----------------------------------------------------------------------
   lean exact-pairwise corr z-force (global z-gather), written to fzloc[nlocal].
   Calibration reference: corr_bin() should reproduce this to target accuracy.
------------------------------------------------------------------------- */

void PPPMDispSlab::corr_raw_force(double *fzloc)
{
  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  int nprocs = comm->nprocs;
  int *recvcounts = new int[nprocs];
  int *displs = new int[nprocs];
  MPI_Allgather(&nlocal, 1, MPI_INT, recvcounts, 1, MPI_INT, world);
  int natoms_all = 0;
  for (int p = 0; p < nprocs; p++) {
    displs[p] = natoms_all;
    natoms_all += recvcounts[p];
  }
  int myoff = displs[comm->me];

  auto *zloc = new double[nlocal > 0 ? nlocal : 1];
  auto *bloc = new double[nlocal > 0 ? nlocal : 1];
  for (int i = 0; i < nlocal; i++) {
    zloc[i] = x[i][dim];
    bloc[i] = B[type[i]];
  }
  auto *zall = new double[natoms_all > 0 ? natoms_all : 1];
  auto *ball = new double[natoms_all > 0 ? natoms_all : 1];
  MPI_Allgatherv(zloc, nlocal, MPI_DOUBLE, zall, recvcounts, displs, MPI_DOUBLE, world);
  MPI_Allgatherv(bloc, nlocal, MPI_DOUBLE, ball, recvcounts, displs, MPI_DOUBLE, world);

  for (int i = 0; i < nlocal; i++) {
    const double zi = x[i][dim];
    const double bi = B[type[i]];
    const int iglob = myoff + i;
    double fz = 0.0;
    for (int jg = 0; jg < natoms_all; jg++) {
      if (jg == iglob) continue;
      double delz = zi - zall[jg];
      delz -= zprd * trunc(2.0 * delz / zprd);
      double x2 = delz * delz;
      if (x2 >= rc2) continue;
      double w2, f2, pt2;
      corr_kernels(x2, w2, f2, pt2);
      fz += delz * bi * ball[jg] * f2;
    }
    fzloc[i] = fz;
  }

  delete[] recvcounts;
  delete[] displs;
  delete[] zloc;
  delete[] bloc;
  delete[] zall;
  delete[] ball;
}

/* ----------------------------------------------------------------------
   choose the corr bin count so the binned corr force matches the EXACT
   pairwise corr force to the target RMS force accuracy.  The exact force is
   computed once at setup; the bin count is doubled until the binned force
   agrees with it.  NOTE: binning a sharp cutoff interaction converges only as
   ~sqrt(dz) at fine grids (random sub-grid aliasing at the near field / rcut
   boundary, where the force kernel f2 does not vanish), so it reaches raw as
   dz->0 but slowly -- very tight accuracy needs an impractically fine grid and
   corr raw (exact per-pair) is then preferable; this is flagged with a warning.
------------------------------------------------------------------------- */

void PPPMDispSlab::calibrate_bin()
{
  int nlocal = atom->nlocal;
  double natoms = (double) atom->natoms;
  if (natoms < 1.0) natoms = 1.0;
  auto *fref = new double[nlocal > 0 ? nlocal : 1];
  auto *fb = new double[nlocal > 0 ? nlocal : 1];

  corr_raw_force(fref);    // exact target (once)

  const int nb_cap = (int) (zprd / (0.2 * cutoff / 600.0) + 0.5);    // ~ zprd/0.001 cap
  int nb = (int) (zprd / 0.1 + 0.5);                                 // start near dz = 0.1 sigma
  if (nb < 8) nb = 8;
  int chosen = nb;
  double err = 0.0;
  for (int it = 0; it < 20; it++) {
    corr_bin_force(nb, fb);
    double s = 0.0;
    for (int i = 0; i < nlocal; i++) {
      double d = fb[i] - fref[i];
      s += d * d;
    }
    double sall;
    MPI_Allreduce(&s, &sall, 1, MPI_DOUBLE, MPI_SUM, world);
    err = sqrt(sall / natoms);    // RMS(binned - exact) corr force
    chosen = nb;
    if (err < accuracy || nb >= nb_cap) break;
    nb *= 2;
  }
  bin_nbins = chosen;
  if (comm->me == 0) {
    utils::logmesg(lmp, "  corr bin: {} bins (dz = {:.4g}), force error {:.3g} vs target {:.3g}\n",
                   bin_nbins, zprd / bin_nbins, err, accuracy);
    if (err > accuracy)
      error->warning(FLERR,
                     "pppm/disp/slab corr bin did not reach the target force accuracy {:.3g} "
                     "(reached {:.3g}); use kspace_modify corr raw for tighter accuracy",
                     accuracy, err);
  }

  delete[] fref;
  delete[] fb;
}

/* ----------------------------------------------------------------------
   IK pressure building blocks Phi(h), Psi(h)  (see ewald/disp/slab)
------------------------------------------------------------------------- */

double PPPMDispSlab::ik_phi(double h)
{
  if (fabs(h) < 1.0e-300) return 0.0;
  const double ah = fabs(h);
  double A[8], Bc[8];
  sici_chain(ah * cutoff, A, Bc);
  const double sii5 = A[5] / 4.0 - A[1] / (4.0 * pow(ah * cutoff, 4));
  const double phi = MY_PI / 576.0 - sii5 + A[7] - Bc[6];
  const double ah4 = ah * ah * ah * ah;
  return (h >= 0.0 ? 1.0 : -1.0) * ah4 * phi;
}

/* ---------------------------------------------------------------------- */

double PPPMDispSlab::ik_psi(double h)
{
  if (fabs(h) < 1.0e-300) return 0.0;
  const double ah = fabs(h);
  double A[8], Bc[8];
  sici_chain(ah * cutoff, A, Bc);
  const double psi = MY_PI / 288.0 - A[7] + Bc[6];
  const double ah4 = ah * ah * ah * ah;
  return (h >= 0.0 ? 1.0 : -1.0) * ah4 * psi;
}

/* ----------------------------------------------------------------------
   long-range pressure profiles P_T(z), P_N(z) (Harasima or Irving-Kirkwood).
   Computes the number-density Fourier coefficients S_n directly from atoms,
   then assembles the profile with the same coefficients as ewald/disp/slab.
------------------------------------------------------------------------- */

void PPPMDispSlab::compute_pressure_profile()
{
  const double unitk = 2.0 * MY_PI / zprd;
  const double rc3 = cutoff * cutoff * cutoff;
  const int K = nz / 2 - 1;    // highest resolved mode
  if (npro < 1 || K < 1) return;

  memory->destroy(pt_profile);
  memory->destroy(pn_profile);
  memory->create(pt_profile, npro, "pppm/disp/slab:pt_profile");
  memory->create(pn_profile, npro, "pppm/disp/slab:pn_profile");

  // structure factors sfac[n] = sum_j B_j exp(i n unitk z_j), n=0..K
  auto *srl = new double[K + 1];
  auto *sim = new double[K + 1];
  auto *srl_all = new double[K + 1];
  auto *sim_all = new double[K + 1];
  for (int n = 0; n <= K; n++) srl[n] = sim[n] = 0.0;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    const double bi = B[type[i]];
    double c1 = cos(unitk * x[i][dim]), s1 = sin(unitk * x[i][dim]);
    double cn = 1.0, sn = 0.0;    // cos/sin(n*unitk*z), recurrence
    srl[0] += bi;
    for (int n = 1; n <= K; n++) {
      double cnn = cn * c1 - sn * s1;
      double snn = sn * c1 + cn * s1;
      cn = cnn;
      sn = snn;
      srl[n] += bi * cn;
      sim[n] += bi * sn;
    }
  }
  MPI_Allreduce(srl, srl_all, K + 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(sim, sim_all, K + 1, MPI_DOUBLE, MPI_SUM, world);

  // number-density coefficients S_n = (1/V)(sfacrl - i sfacim)
  auto *Sre = new double[K + 1];
  auto *Sim = new double[K + 1];
  for (int n = 0; n <= K; n++) {
    Sre[n] = srl_all[n] / volume;
    Sim[n] = -sim_all[n] / volume;
  }
  const double c0 = -4.0 * MY_PI * Sre[0] / (3.0 * rc3);

  if (contour_flag == 0) {

    // Harasima: P(z) = rho_B(z) [ c0 + sum_n Re(C_n S_n e^{i h_n z}) ]
    auto *bdens = new double[npro];
    for (int g = 0; g < npro; g++) bdens[g] = 0.0;
    for (int i = 0; i < nlocal; i++) {
      double u = (x[i][dim] - zlo) / zprd * npro;
      u -= npro * floor(u / npro);
      int g = (int) u;
      if (g >= npro) g -= npro;
      bdens[g] += B[type[i]];
    }
    auto *bdens_all = new double[npro];
    MPI_Allreduce(bdens, bdens_all, npro, MPI_DOUBLE, MPI_SUM, world);
    const double dz = zprd / npro;
    for (int g = 0; g < npro; g++) {
      double z = zlo + (g + 0.5) * dz;
      double gt = c0, gn = c0;
      for (int n = 1; n <= K; n++) {
        double hn = n * unitk, xc = hn * cutoff;
        double AA[8], BB[8];
        sici_chain(xc, AA, BB);
        double Tn = -24.0 * MY_PI * hn * hn * hn * (MY_PI / 288.0 - AA[7] + BB[6]);
        double Nn =
            -24.0 * MY_PI * hn * hn * hn * (MY_PI / 72.0 - AA[5] + 2.0 * AA[7] - 2.0 * BB[6]);
        double cz = cos(hn * z), sz = sin(hn * z);
        gt += Tn * (Sre[n] * cz - Sim[n] * sz);
        gn += Nn * (Sre[n] * cz - Sim[n] * sz);
      }
      double rhoz = bdens_all[g] / (area * dz);
      pt_profile[g] = rhoz * gt;
      pn_profile[g] = rhoz * gn;
    }
    delete[] bdens;
    delete[] bdens_all;

  } else {

    // Irving-Kirkwood: total-mode amplitudes A^T_p, A^N_p (p=n+m), then grid sum
    int P = 2 * K;
    auto *ATre = new double[P + 1];
    auto *ATim = new double[P + 1];
    auto *ANre = new double[P + 1];
    auto *ANim = new double[P + 1];
    for (int p = 0; p <= P; p++) ATre[p] = ATim[p] = ANre[p] = ANim[p] = 0.0;
    auto Sn = [&](int n, double &re, double &im) {
      int an = n < 0 ? -n : n;
      re = Sre[an];
      im = (n < 0) ? -Sim[an] : Sim[an];
    };
    for (int n = -K; n <= K; n++) {
      double hn = n * unitk;
      for (int m = -K; m <= K; m++) {
        int p = n + m;
        if (p < 0) continue;
        double hm = m * unitk, H = hn + hm;
        double snr, sni, smr, smi;
        Sn(n, snr, sni);
        Sn(m, smr, smi);
        double sre = snr * smr - sni * smi, simv = snr * smi + sni * smr;
        double CT, CN;
        if (n == 0 && m == 0) {
          CT = CN = -4.0 * MY_PI / (3.0 * rc3);
        } else if (fabs(H) < 1.0e-300) {
          double ah = fabs(hn), xc = ah * cutoff, AA[8], BB[8];
          sici_chain(xc, AA, BB);
          CT = -12.0 * MY_PI * ah * ah * ah * (MY_PI / 288.0 - AA[7] + BB[6]);
          CN = -24.0 * MY_PI * ah * ah * ah * (MY_PI / 72.0 - AA[5] + 2.0 * AA[7] - 2.0 * BB[6]) /
              2.0;
        } else {
          CT = -6.0 * MY_PI / H * (ik_phi(hm) + ik_phi(hn));
          CN = -12.0 * MY_PI / H * (ik_psi(hm) + ik_psi(hn));
        }
        ATre[p] += CT * sre;
        ATim[p] += CT * simv;
        ANre[p] += CN * sre;
        ANim[p] += CN * simv;
      }
    }
    const double dz = zprd / npro;
    for (int g = 0; g < npro; g++) {
      double z = zlo + (g + 0.5) * dz;
      double pt = ATre[0], pn = ANre[0];
      for (int p = 1; p <= P; p++) {
        double cz = cos(p * unitk * z), sz = sin(p * unitk * z);
        pt += 2.0 * (ATre[p] * cz - ATim[p] * sz);
        pn += 2.0 * (ANre[p] * cz - ANim[p] * sz);
      }
      pt_profile[g] = pt;
      pn_profile[g] = pn;
    }
    delete[] ATre;
    delete[] ATim;
    delete[] ANre;
    delete[] ANim;
  }

  delete[] srl;
  delete[] sim;
  delete[] srl_all;
  delete[] sim_all;
  delete[] Sre;
  delete[] Sim;
}

/* ----------------------------------------------------------------------
   standard sine/cosine integrals (series x<=2, Lentz CF x>2); see ewald/disp/slab
------------------------------------------------------------------------- */

void PPPMDispSlab::cisi(double x, double &si, double &ci)
{
  if (x <= 2.0) {
    double term = x, s = x;
    for (int k = 1; k < 60; k++) {
      term *= -x * x / ((2.0 * k) * (2.0 * k + 1.0));
      double add = term / (2.0 * k + 1.0);
      s += add;
      if (fabs(add) < 1.0e-18 * fabs(s)) break;
    }
    si = s;
    double cterm = 1.0, cin = 0.0;
    for (int k = 1; k < 60; k++) {
      cterm *= -x * x / ((2.0 * k - 1.0) * (2.0 * k));
      double add = -cterm / (2.0 * k);
      cin += add;
      if (fabs(add) < 1.0e-18 * (fabs(cin) + 1.0e-300)) break;
    }
    ci = EULER + log(x) - cin;
  } else {
    const double tiny = 1.0e-300;
    double br = 1.0, bi = x;
    double cr = 1.0e308, cii = 0.0;
    double den = br * br + bi * bi;
    double dr = br / den, di = -bi / den;
    double hr = dr, hi = di;
    for (int i = 1; i < 400; i++) {
      double a = -(double) i * i;
      br += 2.0;
      double tr = a * dr + br, ti = a * di + bi;
      den = tr * tr + ti * ti;
      if (den < tiny) den = tiny;
      dr = tr / den;
      di = -ti / den;
      double cden = cr * cr + cii * cii;
      if (cden < tiny) cden = tiny;
      cr = br + a * cr / cden;
      cii = bi - a * cii / cden;
      double delr = cr * dr - cii * di;
      double deli = cr * di + cii * dr;
      double nhr = hr * delr - hi * deli;
      double nhi = hr * deli + hi * delr;
      hr = nhr;
      hi = nhi;
      if (fabs(delr - 1.0) + fabs(deli) < 1.0e-17) break;
    }
    double cx = cos(x), sx = sin(x);
    double fr = hr * cx + hi * sx;
    double fi = -hr * sx + hi * cx;
    ci = -fr;
    si = MY_PI / 2.0 + fi;
  }
}

/* ----------------------------------------------------------------------
   generalized integrals A_m=Si_m, B_m=Ci_m via recurrence; fills [1..7]
------------------------------------------------------------------------- */

void PPPMDispSlab::sici_chain(double x, double *Aarr, double *Barr)
{
  double si, ci;
  cisi(x, si, ci);
  Aarr[1] = si;
  Barr[1] = ci - EULER;
  double sx = sin(x), cx = cos(x);
  for (int m = 2; m <= 7; m++) {
    double xm = pow(x, 1 - m);
    Aarr[m] = -sx * xm / (m - 1) + Barr[m - 1] / (m - 1);
    Barr[m] = -cx * xm / (m - 1) - Aarr[m - 1] / (m - 1);
  }
}

/* ---------------------------------------------------------------------- */

double PPPMDispSlab::memory_usage()
{
  double bytes = 6.0 * nz * sizeof(double);
  bytes += (double) nmax * sizeof(double);
  bytes += (double) order * order * sizeof(double);
  if (profile_flag) bytes += 2.0 * (double) npro * sizeof(double);
  return bytes;
}
