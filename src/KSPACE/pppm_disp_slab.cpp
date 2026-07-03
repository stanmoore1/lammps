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
   Mesh-accelerated smooth-damped slab-based dispersion Ewald (pppm/disp/slab).
   The dispersion-weighted density (geometric mixing) varies only in the chosen
   inhomogeneous dimension, so the smooth reciprocal part is a 1-D convolution:
   spread the B-weighted density onto a 1-D grid, FFT, apply the damped influence
   function, inverse-FFT the z-force field, and interpolate.  The reciprocal
   energy/force reproduce the exact ewald/disp/slab result as the grid is
   refined.

   Matched to a lj/cut/dispswitch pair that fades the 1/r^6 dispersion out
   smoothly over [rcut, rcut+Delta].  The real-space slab correction is a
   z-convolution of the density, diagonal in the grid's Fourier basis, so it is
   folded directly into the influence function (influence_function): one spread +
   FFT + combined kernel + interpolation does the reciprocal sum AND the
   correction (energy, ik force, full pressure tensor, per-atom) in one pass with
   no separate real-space correction step.

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

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr int OFFSET = 16384;
static constexpr int MAXORDER = 8;
static constexpr double EULER = 0.57721566490153286061;

/* ---------------------------------------------------------------------- */

PPPMDispSlab::PPPMDispSlab(LAMMPS *lmp) :
    KSpace(lmp), B(nullptr), dens(nullptr), fre(nullptr), fim(nullptr), Gk(nullptr), GTk(nullptr),
    GNk(nullptr), fz_grid(nullptr), ugrid(nullptr), uTgrid(nullptr), uNgrid(nullptr),
    rho_coeff(nullptr), peatom(nullptr)
{
  dispersionflag = 1;
  dim = 2;
  lat1 = 0;
  lat2 = 1;
  nz = 0;
  order = 6;
  sw_width = 0.0;
  cWgrid = nullptr;
  cWraw = nullptr;
  ncgrid = 0;
  cwdz = 0.0;
  Araw_tab = Braw_tab = nullptr;
  nkap = 0;
  kap_dk = 0.0;
  kap_max = 0.0;
  g_ewald_set = 0.0;
  order_allocated = 0;
  nmax = 0;
  accuracy_relative = 0.0;
  prof_kmax_cached = 0;
  prof_kmax_nz = 0;
  prof_kmax_zprd = 0.0;
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
  memory->destroy(GTk);
  memory->destroy(GNk);
  memory->destroy(fz_grid);
  memory->destroy(ugrid);
  memory->destroy(uTgrid);
  memory->destroy(uNgrid);
  delete[] cWgrid;
  delete[] cWraw;
  delete[] Araw_tab;
  delete[] Braw_tab;
  memory->destroy(peatom);
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
     dim x|y|z    -- select the inhomogeneous direction (default z)
------------------------------------------------------------------------- */

int PPPMDispSlab::modify_param(int narg, char **arg)
{
  // mesh/disp, order/disp, gewald/disp are consumed by the base KSpace parser
  // (they set nz_pppm_6/gridflag_6, order_6, g_ewald_6/gewaldflag_6).
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

  // the matched lj/cut/dispswitch pair fades the 1/r^6 dispersion out over
  // [rcut, rcut+Delta]; the smooth switched corr is merged into the influence
  // function (no real-space corr step; see influence_function).

  int itmp2;
  double *p_dz = (double *) force->pair->extract("disp_switch_width", itmp2);
  if (p_dz == nullptr || *p_dz <= 0.0)
    error->all(FLERR,
               "kspace_style pppm/disp/slab requires the matched lj/cut/dispswitch pair style "
               "to switch off the dispersion smoothly at the cutoff; use "
               "pair_style lj/cut/dispswitch <rcut> <Delta>");
  sw_width = *p_dz;

  // accuracy in force units

  two_charge();
  if (accuracy_absolute >= 0.0)
    accuracy = accuracy_absolute;
  else
    accuracy = accuracy_relative * two_charge_force;

  // per-type dispersion amplitude B = 2 sqrt(eps) sigma^3 (geometric mixing).
  // kspace->init() runs before pair->init(), so lj4 may not be populated yet;
  // epsilon/sigma (set by pair_coeff) give the identical value B = sqrt(lj4).

  int n = atom->ntypes, edim;
  auto **eps = (double **) force->pair->extract("epsilon", edim);
  auto **sig = (double **) force->pair->extract("sigma", edim);
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
    utils::logmesg(lmp,
                   "  smooth-damped, z grid = {}, stencil order = {}, g_ewald = {:.6g}, switch "
                   "width Delta = {:.6g}\n",
                   nz, order, g_ewald, sw_width);
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
   the smallest power-of-two nz with df < accuracy.  The grid must also resolve
   the merged corr kernel (its two features: the Gaussian-screened peak of width
   ~1/g_ewald, and the (1-S) switch shell of width ~Delta -- whichever is sharper).
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
    const int ngrid_max = 65536;
    int ngrid = 16;
    while (ngrid < ngrid_max) {
      double df = sqrt(compute_qopt(ngrid, order)) * pref;
      if (df < accuracy) break;
      ngrid <<= 1;
    }
    nz = ngrid;
    // the same grid must resolve the merged corr kernel, which has TWO features --
    // the Gaussian-screened peak (width ~1/g_ewald) and, when Delta < 1/g_ewald, the
    // sharper (1-S) switch shell (width ~Delta).  The grid must resolve the SHARPER
    // of the two.  Measured vs the exact ewald/disp/slab corr raw (bench slab, order
    // 5, acc 1e-5): dz ~ 0.35/g reaches ~2e-7, and for Delta < 1/g, dz ~ 0.39*Delta
    // reaches ~1e-5; use 0.35*min(1/g, Delta) with the same ~2x margin as the
    // Gaussian floor.  Delta >= 1/g reduces to the original 1/g-only floor.
    const double s = pow(acc / 1.0e-5, 0.25);
    double feat = MIN(1.0 / g_ewald, sw_width);    // sharper corr-kernel feature
    double dz_target = 0.35 * s * feat;
    dz_target = MAX(dz_target, 0.02);              // sanity floor (avoid runaway nz)
    int nzc = 1;
    while (nzc < (int) (zprd / dz_target)) nzc <<= 1;
    if (nz < nzc) nz = nzc;
  }
  if (nz < 8) nz = 8;

  // reported RMS force error = the Gaussian grid aliasing (qopt) PLUS the merged
  // corr force content the grid cannot resolve (modes beyond nz/2).  The corr
  // Fourier tail decays only ~k^-5, so it dominates; qopt alone (Gaussian only)
  // is far too optimistic.  Same random-phase corr-tail term as ewald/disp/slab.
  build_corr_kernels();
  const double uk = 2.0 * MY_PI / zprd;
  const double invLz = 1.0 / zprd;
  double ctk = 0.0;
  for (int k = nz / 2 + 1; k <= 8 * nz; k++) {
    double w2t, kw2p;
    corr_tilde(k * uk, w2t, kw2p);
    const double cf = 2.0 * (k * uk) * w2t * invLz;
    ctk += cf * cf;
  }
  estimated_force_accuracy = pref * sqrt(compute_qopt(nz, order) + 0.5 * ctk);
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
  memory->destroy(GTk);
  memory->destroy(GNk);
  memory->destroy(fz_grid);
  memory->destroy(ugrid);
  memory->destroy(uTgrid);
  memory->destroy(uNgrid);
  memory->create(dens, nz, "pppm/disp/slab:dens");
  memory->create(fre, nz, "pppm/disp/slab:fre");
  memory->create(fim, nz, "pppm/disp/slab:fim");
  memory->create(Gk, nz, "pppm/disp/slab:Gk");
  memory->create(GTk, nz, "pppm/disp/slab:GTk");
  memory->create(GNk, nz, "pppm/disp/slab:GNk");
  memory->create(fz_grid, nz, "pppm/disp/slab:fz_grid");
  memory->create(ugrid, nz, "pppm/disp/slab:ugrid");
  memory->create(uTgrid, nz, "pppm/disp/slab:uTgrid");
  memory->create(uNgrid, nz, "pppm/disp/slab:uNgrid");

  if (rho_coeff == nullptr || order != order_allocated) {
    if (rho_coeff) memory->destroy(rho_coeff);
    memory->create(rho_coeff, order, order, "pppm/disp/slab:rho_coeff");
    order_allocated = order;
  }
  compute_rho_coeff();

  build_corr_kernels();
  influence_function();
}

/* ----------------------------------------------------------------------
   de-convolved smooth-damped influence function on the z grid modes.

   The PPPM mesh energy E = sum_{m=0}^{nz-1} Gk[m] |rho_hat_m|^2 (full FFT
   spectrum) must equal the exact ewald/disp/slab energy
   E = GU[0]|S_0|^2 + sum_{k>=1} GU[k]|S_k|^2.  Matching the spectra term by term
   (each +/- mode appears once in the FFT sum) gives the physical per-mode
   coefficient W_E(k) = GU[0] for m=0 and GU[|m|]/2 for m != 0.  De-convolving the
   order-p assignment (transfer W(k)=sinc(pi m/nz)^order): Gk[m] = W_E(k_m)/W(k_m)^2.

   Merged smooth switched corr: the binned corr convolution is diagonal in the
   grid's Fourier basis (E_corr = sum_k [0.5 W~2(k)/Lz]|S_k|^2 with W~2 the 1-D
   transform of the smooth corr kernel w2(|dz|)), so it folds into the influence
   function.  Virial: the corr tangential coefficient equals its energy coefficient
   (pt2 = w2, boundary term ~ acc^2), so GTk = Gk; the normal is the exact strain
   derivative GN = GU + h dGU/dh for the reciprocal part plus CN = 0.5(W~2 + k
   dW~2/dk)/Lz for the corr.
------------------------------------------------------------------------- */

void PPPMDispSlab::influence_function()
{
  const double sqpi = sqrt(MY_PI);
  const double coef = -2.0 * MY_PI * sqpi / (24.0 * volume);
  // merged corr coefficient = (2*pi/volume) times the interpolated box-independent
  // Fourier transform A(k) (and A - k B for the normal); no per-step quadrature.
  const double pre2 = 2.0 * MY_PI / volume;

  // m=0 (homogeneous tail) term: W_E(0) = GU[0] = -pi^1.5 g^3 / (6 V), plus corr
  Gk[0] = -MY_PI * sqpi * g_ewald * g_ewald * g_ewald / (6.0 * volume);
  double A0, B0;
  ft_interp(0.0, A0, B0);
  const double ce0 = 0.5 * pre2 * A0;
  GNk[0] = Gk[0] + ce0;    // reciprocal GN(k=0) = GU(0)
  Gk[0] += ce0;
  GTk[0] = Gk[0];

  for (int m = 1; m < nz; m++) {
    int mm = (m <= nz / 2) ? m : m - nz;    // signed mode index
    double k = mm * 2.0 * MY_PI / zprd;
    double ak = fabs(k);
    double b = ak / (2.0 * g_ewald), b2 = b * b, b3 = b2 * b;
    double Bk = ak * ak * ak * (sqpi * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));
    double WE = 0.5 * coef * Bk;    // GU[|mm|]/2  (full-spectrum per-mode coeff)
    double s = sin(MY_PI * mm / nz) / (MY_PI * mm / nz);
    double w2 = pow(s, 2 * order);

    double A, Bv;
    ft_interp(ak, A, Bv);
    const double CE = 0.5 * pre2 * A;
    const double CN = 0.5 * pre2 * (A - ak * Bv);
    const double WN = 0.5 * coef * (4.0 * Bk - 1.5 * ak * ak * ak * exp(-b2) / b3);
    Gk[m] = (WE + CE) / w2;
    GTk[m] = Gk[m];
    GNk[m] = (WN + CN) / w2;
  }
}

/* ----------------------------------------------------------------------
   smooth (Gaussian-screened) 1/r^6 = u(r) - u_short(r); Taylor near r=0.
------------------------------------------------------------------------- */

double PPPMDispSlab::u_smooth(double r)
{
  const double g2 = g_ewald * g_ewald;
  const double r2 = r * r;
  const double a2 = g2 * r2;
  if (a2 < 0.1) {
    const double g6 = g2 * g2 * g2, g8 = g6 * g2, g10 = g8 * g2, g12 = g10 * g2;
    return g6 / 6.0 - g8 * r2 / 8.0 + g10 * r2 * r2 / 20.0 - g12 * r2 * r2 * r2 / 72.0;
  }
  const double r6 = r2 * r2 * r2;
  return (1.0 - (1.0 + a2 + 0.5 * a2 * a2) * exp(-a2)) / r6;
}

/* ----------------------------------------------------------------------
   tabulate the smooth switched corr energy kernel w2(|dz|) over [0, rcut+Delta]:
   w2 = (2 pi/area) int_{|dz|}^{b} r corr_e(r) dr,
   corr_e(r) = u_smooth(r) - [r>rcut] S(r)/r^6 (vanishes smoothly at b).
   Same math as ewald/disp/slab build_corr_kernels.
------------------------------------------------------------------------- */

void PPPMDispSlab::build_corr_kernels()
{
  const double a = cutoff, b = cutoff + sw_width;
  ncgrid = 2048;
  cwdz = b / ncgrid;

  // BOX-INDEPENDENT kernel integral, precomputed once (NPT hot loop -> just rescale)
  if (cWraw == nullptr) {
    cWraw = new double[ncgrid + 1];
    for (int g = 0; g <= ncgrid; g++) {
      const double adz = g * cwdz;
      const int n = 400;
      const double hr = (b - adz) / n;
      double IE = 0.0;
      if (hr > 0.0) {
        for (int i = 0; i <= n; i++) {
          const double r = adz + i * hr;
          const double rr = (r > 1.0e-300) ? r : 1.0e-300;
          double ce = u_smooth(rr);
          if (rr > a) {
            const double r6 = rr * rr * rr * rr * rr * rr;
            ce -= switch_S((rr - a) / sw_width) / r6;
          }
          const double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
          IE += w * rr * ce;
        }
        IE *= hr / 3.0;
      }
      cWraw[g] = IE;
    }
  }

  const double pre = 2.0 * MY_PI / area;
  delete[] cWgrid;
  cWgrid = new double[ncgrid + 1];
  for (int g = 0; g <= ncgrid; g++) cWgrid[g] = pre * cWraw[g];

  // ensure the FT tables cover the grid modes k = (nz/2)*(2*pi/zprd)
  build_corr_ft_tables((nz / 2) * (2.0 * MY_PI / zprd));
}

/* ----------------------------------------------------------------------
   1-D Fourier transforms of the tabulated corr kernel (Simpson over the table):
     w2t  = W~2(k)        = 2 int_0^b w2(z) cos(kz) dz
     kw2p = k dW~2(k)/dk  = -2 k int_0^b z w2(z) sin(kz) dz
   Exact reference (used to build the interpolation tables).
------------------------------------------------------------------------- */

void PPPMDispSlab::corr_tilde(double k, double &w2t, double &kw2p)
{
  double sc = 0.0, ss = 0.0;
  for (int g = 0; g <= ncgrid; g++) {
    const double z = g * cwdz;
    const double w = (g == 0 || g == ncgrid) ? 1.0 : (g % 2 ? 4.0 : 2.0);
    sc += w * cWgrid[g] * cos(k * z);
    ss += w * z * cWgrid[g] * sin(k * z);
  }
  w2t = 2.0 * sc * cwdz / 3.0;
  kw2p = -2.0 * k * ss * cwdz / 3.0;
}

/* ----------------------------------------------------------------------
   (re)build the box-independent Fourier-transform tables of cWraw on a uniform
   wavenumber grid (grow-only); see ewald/disp/slab for the derivation.
     A(kap) = 2 int cWraw cos(kap z) dz,  B(kap) = 2 int z cWraw sin(kap z) dz
   so W~2(k) = (2*pi/area) A(k) and k dW~2/dk = -(2*pi/area) k B(k).
------------------------------------------------------------------------- */

void PPPMDispSlab::build_corr_ft_tables(double kap_need)
{
  const double target = 1.5 * MAX(kap_need, 1.0e-6);
  if (Araw_tab && target <= kap_max) return;

  kap_dk = (2.0 * MY_PI / (cutoff + sw_width)) / 100.0;    // ~100 points per oscillation
  nkap = (int) (target / kap_dk) + 4;
  kap_max = nkap * kap_dk;
  delete[] Araw_tab;
  delete[] Braw_tab;
  Araw_tab = new double[nkap + 1];
  Braw_tab = new double[nkap + 1];
  const double c = 2.0 * cwdz / 3.0;
  for (int j = 0; j <= nkap; j++) {
    const double kap = j * kap_dk;
    double sc = 0.0, ss = 0.0;
    for (int g = 0; g <= ncgrid; g++) {
      const double z = g * cwdz;
      const double w = (g == 0 || g == ncgrid) ? 1.0 : (g % 2 ? 4.0 : 2.0);
      sc += w * cWraw[g] * cos(kap * z);
      ss += w * z * cWraw[g] * sin(kap * z);
    }
    Araw_tab[j] = c * sc;
    Braw_tab[j] = c * ss;
  }
}

/* ----------------------------------------------------------------------
   4-point (cubic) Lagrange interpolation of the FT tables at wavenumber kap.
------------------------------------------------------------------------- */

void PPPMDispSlab::ft_interp(double kap, double &A, double &B)
{
  double x = kap / kap_dk;
  int j = (int) x - 1;
  if (j < 0) j = 0;
  if (j > nkap - 3) j = nkap - 3;
  const double t = x - j;
  const double L0 = -(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0;
  const double L1 = t * (t - 2.0) * (t - 3.0) / 2.0;
  const double L2 = -t * (t - 1.0) * (t - 3.0) / 2.0;
  const double L3 = t * (t - 1.0) * (t - 2.0) / 6.0;
  A = Araw_tab[j] * L0 + Araw_tab[j + 1] * L1 + Araw_tab[j + 2] * L2 + Araw_tab[j + 3] * L3;
  B = Braw_tab[j] * L0 + Braw_tab[j + 1] * L1 + Braw_tab[j + 2] * L2 + Braw_tab[j + 3] * L3;
}

/* ----------------------------------------------------------------------
   C3 septic smoothstep S(t) and its derivative (the matched lj/cut/dispswitch
   pair fades the 1/r^6 dispersion out by (1-S) over [rcut, rcut+Delta]).
------------------------------------------------------------------------- */

double PPPMDispSlab::switch_S(double t)
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t;
  const double t3 = t2 * t, t4 = t3 * t;
  return t4 * (35.0 - 84.0 * t + 70.0 * t2 - 20.0 * t3);
}

/* ---------------------------------------------------------------------- */

double PPPMDispSlab::switch_dS(double t)
{
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double tu = t * (1.0 - t);
  return 140.0 * tu * tu * tu;    // 140 (t(1-t))^3
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
   FFT the density, accumulate the reciprocal energy and virial, build the
   z-force field, and (if per-atom requested) the potential/virial fields.

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

  // reciprocal energy (full system value) + virial: explicit tangential (GTk) and
  // normal (GNk) kernels (the homogeneity trace does not hold for the merged corr)

  double e = 0.0;
  for (int m = 0; m < nz; m++) e += Gk[m] * (fre[m] * fre[m] + fim[m] * fim[m]);
  if (eflag_global) energy += e;
  if (vflag_global) {
    double vt = 0.0, vn = 0.0;
    for (int m = 0; m < nz; m++) {
      double uk = fre[m] * fre[m] + fim[m] * fim[m];
      vt += GTk[m] * uk;
      vn += GNk[m] * uk;
    }
    virial[lat1] += vt;
    virial[lat2] += vt;
    virial[dim] += vn;
  }

  // per-atom potential/virial fields (u = IFFT[2 Gk rho_hat], uT, uN)

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
    for (int m = 0; m < nz; m++) {
      double gt2 = 2.0 * GTk[m];
      ur[m] = gt2 * fre[m];
      ui[m] = gt2 * fim[m];
    }
    fft1d(ur, ui, nz, +1);
    for (int g = 0; g < nz; g++) uTgrid[g] = ur[g];
    for (int m = 0; m < nz; m++) {
      double gn2 = 2.0 * GNk[m];
      ur[m] = gn2 * fre[m];
      ui[m] = gn2 * fim[m];
    }
    fft1d(ur, ui, nz, +1);
    for (int g = 0; g < nz; g++) uNgrid[g] = ur[g];
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

    double fz = 0.0, uu = 0.0, uT = 0.0, uN = 0.0;
    for (int s = 0; s < order; s++) {
      int g = g0 + nlower + s;
      g = ((g % nz) + nz) % nz;
      fz += w[s] * fz_grid[g];
      if (evflag_atom) {
        uu += w[s] * ugrid[g];
        uT += w[s] * uTgrid[g];
        uN += w[s] * uNgrid[g];
      }
    }
    f[i][dim] += bi * fz;

    if (evflag_atom) {
      double pe = 0.5 * bi * uu;    // per-atom reciprocal energy
      peatom[i] += pe;
      if (vflag_atom) {
        vatom[i][lat1] += 0.5 * bi * uT;
        vatom[i][lat2] += 0.5 * bi * uT;
        vatom[i][dim] += 0.5 * bi * uN;
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
  // the smooth switched corr is merged into the influence function -- energy, ik
  // force, full pressure tensor and per-atom terms are all handled in
  // poisson()/fieldforce(), with no real-space correction step.

  if (eflag_atom)
    for (int i = 0; i < atom->nlocal; i++) eatom[i] += peatom[i];
}


/* ----------------------------------------------------------------------
   energy shell integral t5 = int_rcut^{rcut+Delta} S(r) r^-5 sin(h r) dr
   (10-point Gauss-Legendre per panel, panel count scaled to the oscillation
   count so the result is accurate ~1e-13 for all h).
------------------------------------------------------------------------- */

double PPPMDispSlab::switch_trans5(double h)
{
  static const double gx[10] = {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
                                -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
                                0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
                                0.9739065285171717};
  static const double gw[10] = {0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                                0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
                                0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
                                0.0666713443086881};
  const double a = cutoff, dz = sw_width;
  int np = (int) (8.0 * h * dz / (2.0 * MY_PI)) + 1;
  np = MAX(8, np);
  np = MIN(np, 20000);
  const double hp = dz / np;    // panel width
  double s5 = 0.0;
  for (int p = 0; p < np; p++) {
    const double c0 = a + (p + 0.5) * hp;    // panel center
    for (int g = 0; g < 10; g++) {
      const double r = c0 + 0.5 * hp * gx[g];
      const double S = switch_S((r - a) / dz);
      const double r2 = r * r, r4 = r2 * r2;
      s5 += gw[g] * S * sin(h * r) / (r4 * r);    // r^-5 sin
    }
  }
  return 0.5 * hp * s5;
}

/* ----------------------------------------------------------------------
   shell virial integrals over [rcut, rcut+Delta] with the smooth force
   phi'(r) = (S u)' = -S'/r^6 + 6 S/r^7:
     sGT = int phi'(r) A_T dr, sGN = int phi'(r) A_N dr.
   GT = GT_tail - (pi/V) sGT, GN = GN_tail - (2 pi/V) sGN.
------------------------------------------------------------------------- */

void PPPMDispSlab::switch_shell_virial(double h, double &sGT, double &sGN)
{
  static const double gx[10] = {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
                                -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
                                0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
                                0.9739065285171717};
  static const double gw[10] = {0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                                0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
                                0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
                                0.0666713443086881};
  const double a = cutoff, dz = sw_width;
  int np = (int) (8.0 * h * dz / (2.0 * MY_PI)) + 1;
  np = MAX(8, np);
  np = MIN(np, 20000);
  const double hp = dz / np;
  const double h2 = h * h, h3 = h2 * h;
  double accT = 0.0, accN = 0.0;
  for (int p = 0; p < np; p++) {
    const double c0 = a + (p + 0.5) * hp;
    for (int g = 0; g < 10; g++) {
      const double r = c0 + 0.5 * hp * gx[g];
      const double t = (r - a) / dz;
      const double S = switch_S(t);
      const double Sp = switch_dS(t) / dz;    // S'(r)
      const double r2 = r * r, r6 = r2 * r2 * r2, r7 = r6 * r;
      const double phip = -Sp / r6 + 6.0 * S / r7;    // (S u)' = S'u + S u'
      const double sr = sin(h * r), cr = cos(h * r);
      const double AT = -4.0 * r * cr / h2 + 4.0 * sr / h3;
      const double AN = 2.0 * r2 * sr / h + 4.0 * r * cr / h2 - 4.0 * sr / h3;
      accT += gw[g] * phip * AT;
      accN += gw[g] * phip * AN;
    }
  }
  sGT = 0.5 * hp * accT;
  sGN = 0.5 * hp * accN;
}

/* ----------------------------------------------------------------------
   compact-switch energy coefficient GU at mesh mode k (k = m*unitk): non-damped
   tail at rcut+Delta plus the numerically integrated shell transition.
------------------------------------------------------------------------- */

double PPPMDispSlab::gu_switch(int k)
{
  const double kcell = k * (2.0 * MY_PI / zprd);
  const double kcell3 = kcell * kcell * kcell;
  const double c = cutoff + sw_width;
  double C[8], D[8];
  sici_compl_chain(kcell * c, C, D);
  const double t5 = switch_trans5(kcell);
  return (-4.0 * MY_PI * kcell3 / volume) * C[5] - (4.0 * MY_PI / volume) * t5 / kcell;
}

/* ----------------------------------------------------------------------
   compact-switch k=0 energy coefficient.
------------------------------------------------------------------------- */

double PPPMDispSlab::gu0_switch()
{
  const double a = cutoff, b = cutoff + sw_width, dz = sw_width;
  const int n = 256;
  const double dr = (b - a) / n;
  double s = 0.0;
  for (int i = 0; i <= n; i++) {
    const double r = a + i * dr;
    const double S = switch_S((r - a) / dz);
    const double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    s += w * S / (r * r * r * r);
  }
  const double trans = dr / 3.0 * s;
  const double tail = 1.0 / (3.0 * b * b * b);
  return -(2.0 * MY_PI / volume) * (trans + tail);
}

/* ----------------------------------------------------------------------
   complementary chain C[m]=A[m](inf)-A[m], D[m]=B[m](inf)-B[m] (cancellation
   free high-k tail coefficients); see ewald/disp/slab.
------------------------------------------------------------------------- */

void PPPMDispSlab::sici_compl_chain(double x, double *Carr, double *Darr)
{
  double si, ci;
  cisi(x, si, ci);
  Carr[1] = MY_PI / 2.0 - si;    // A[1](inf) - A[1] = pi/2 - Si(x)
  Darr[1] = -ci;                 // B[1](inf) - B[1] = -Ci(x)
  const double sx = sin(x), cx = cos(x);
  for (int m = 2; m <= 7; m++) {
    const double xm = pow(x, 1 - m);
    Carr[m] = (Darr[m - 1] + sx * xm) / (m - 1);
    Darr[m] = (cx * xm - Carr[m - 1]) / (m - 1);
  }
}

/* ----------------------------------------------------------------------
   IK pressure building blocks Phi(h), Psi(h)  (see ewald/disp/slab)
------------------------------------------------------------------------- */

double PPPMDispSlab::ik_phi(double h)
{
  if (fabs(h) < 1.0e-300) return 0.0;
  const double ah = fabs(h);
  // compact switch: anchor the tail at the OUTER cutoff rcut+Delta and add the
  // switch-shell integral (ported from pppm/disp/planar; sharp as Delta->0)
  const double c = cutoff + sw_width;
  double A[8], Bc[8];
  sici_chain(ah * c, A, Bc);
  const double sii5 = A[5] / 4.0 - A[1] / (4.0 * pow(ah * c, 4));
  double phi = MY_PI / 576.0 - sii5 + A[7] - Bc[6];
  phi += prof_shell(PROF_PHI, ah);
  const double ah4 = ah * ah * ah * ah;
  return (h >= 0.0 ? 1.0 : -1.0) * ah4 * phi;
}

/* ---------------------------------------------------------------------- */

double PPPMDispSlab::ik_psi(double h)
{
  if (fabs(h) < 1.0e-300) return 0.0;
  const double ah = fabs(h);
  const double c = cutoff + sw_width;
  double A[8], Bc[8];
  sici_chain(ah * c, A, Bc);
  double psi = MY_PI / 288.0 - A[7] + Bc[6];
  psi += prof_shell(PROF_T, ah);
  const double ah4 = ah * ah * ah * ah;
  return (h >= 0.0 ? 1.0 : -1.0) * ah4 * psi;
}

/* ----------------------------------------------------------------------
   potential-form integrand g(r) of a profile coefficient (x = h r); ported
   from pppm/disp/planar (see ewald/disp/slab for the derivation notes).
------------------------------------------------------------------------- */

double PPPMDispSlab::prof_integrand(int which, double r, double h)
{
  const double x = h * r;
  const double sx = sin(x), cx = cos(x);
  const double h2 = h * h, h4 = h2 * h2, h5 = h4 * h, h6 = h4 * h2;
  const double r2 = r * r, r4 = r2 * r2, r5 = r4 * r, r6 = r4 * r2, r7 = r6 * r;
  if (which == PROF_T) {
    return sx / (h6 * r7) - cx / (h5 * r6);
  } else if (which == PROF_N) {
    return sx / (h4 * r5) - 2.0 * sx / (h6 * r7) + 2.0 * cx / (h5 * r6);
  } else {    // PROF_PHI
    double si, ci;
    cisi(x, si, ci);
    return si / (h4 * r5) - sx / (h6 * r7) + cx / (h5 * r6);
  }
}

/* ----------------------------------------------------------------------
   compact-switch shell correction of a profile coefficient:
   int_rcut^{rcut+Delta} W(r) g(r) dr, W = S - S' r/6 (see ewald/disp/slab).
------------------------------------------------------------------------- */

double PPPMDispSlab::prof_shell(int which, double h)
{
  static const double gx[10] = {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
                                -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
                                0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
                                0.9739065285171717};
  static const double gw[10] = {0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                                0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
                                0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
                                0.0666713443086881};
  const double a = cutoff, dzs = sw_width;
  int np = (int) (8.0 * h * dzs / (2.0 * MY_PI)) + 1;
  np = MAX(8, np);
  np = MIN(np, 20000);
  const double hp = dzs / np;
  double acc = 0.0;
  for (int p = 0; p < np; p++) {
    const double c0 = a + (p + 0.5) * hp;
    for (int g = 0; g < 10; g++) {
      const double r = c0 + 0.5 * hp * gx[g];
      const double t = (r - a) / dzs;
      const double W = switch_S(t) - (switch_dS(t) / dzs) * r / 6.0;
      acc += gw[g] * W * prof_integrand(which, r, h);
    }
  }
  return 0.5 * hp * acc;
}

/* ----------------------------------------------------------------------
   shell-correction virial per profile bin (CSB), dispatched on corr_mode so the
   contour profile uses the IDENTICAL corr_csb correction as the box average
   (raw = exact per-atom shell virial spread IK along each bond; bin = density
   convolution).  Ported from pppm/disp/planar (geometric mixing).
------------------------------------------------------------------------- */

void PPPMDispSlab::shell_profile_virial(int nbins, double /*lo*/, double /*dz*/,
                                        double * /*dens_all*/, double *shellT, double *shellN)
{
  // No shell subtraction for the merged-damped variant.  Unlike the compact-switch
  // method (where the pair evaluated the FULL shell and corr_csb removed the plane
  // mean field), here the pair fades the dispersion by (1-S) and the kspace
  // coefficients GT[k]/GN[k] already carry the full plane mean field of S*u
  // (Gaussian split + merged corr), so the reciprocal double sum needs no
  // additional shell correction -- the switched pair supplies the laterally
  // resolved (1-S) shell separately in the stress/cartesian pair term.
  for (int g = 0; g < nbins; g++) shellT[g] = shellN[g] = 0.0;
}

/* ----------------------------------------------------------------------
   force-accuracy mode cutoff K_prof for the profile: the FFT grid over-resolves
   the physical mode content, so truncate the O(K^2) assembly at the force-
   converged kmax (same random-phase model + kbig tail scan as the estimators).
   Ported from pppm/disp/planar.
------------------------------------------------------------------------- */

int PPPMDispSlab::profile_kmax()
{
  const int Kgrid = nz / 2 - 1;
  if (Kgrid <= 8) return MAX(1, Kgrid);

  if (prof_kmax_cached > 0 && prof_kmax_nz == nz && fabs(prof_kmax_zprd - zprd) < 1.0e-12 * zprd)
    return prof_kmax_cached;

  int *type = atom->type;
  int nlocal = atom->nlocal;
  double b2_local = 0.0;
  for (int i = 0; i < nlocal; i++) b2_local += B[type[i]] * B[type[i]];
  double b2;
  MPI_Allreduce(&b2_local, &b2, 1, MPI_DOUBLE, MPI_SUM, world);
  double natoms = (double) atom->natoms;
  if (natoms < 1.0) natoms = 1.0;

  const double prefac = 0.5 * b2 * b2 / natoms;
  const double bias = 1.6;
  const double target = accuracy * accuracy / (bias * bias);
  const double uk = 2.0 * MY_PI / zprd;
  const int kbig = 8192;
  auto *gf2 = new double[kbig + 1];
  const double tail_floor = 1.0e-3 * accuracy * accuracy / (16.0 * prefac);
  int kstop = kbig;
  for (int k = 1; k <= kbig; k++) {
    double g = 2.0 * (k * uk) * gu_switch(k);
    gf2[k] = g * g;
    if (k > 16 && gf2[k] * k / 9.0 < tail_floor && gf2[k] < gf2[k - 1]) {
      kstop = k;
      break;
    }
  }
  for (int k = kstop + 1; k <= kbig; k++) gf2[k] = 0.0;
  double tail = 0.0;
  int chosen = kbig;
  for (int kmx = kbig - 1; kmx >= 4; kmx--) {
    tail += gf2[kmx + 1];
    if (prefac * tail >= target) {
      chosen = kmx + 1;
      break;
    }
    chosen = kmx;
  }
  delete[] gf2;

  const int kmax_phys = MAX(8, MIN(chosen, kbig));
  prof_kmax_cached = MIN(kmax_phys, Kgrid);
  prof_kmax_nz = nz;
  prof_kmax_zprd = zprd;
  return prof_kmax_cached;
}

/* ----------------------------------------------------------------------
   raw per-mode tangential/normal box-pressure coefficients GT[k], GN[k],
   k=0..K (the compact-switch coefficients ewald/disp/slab computes in coeffs();
   NOT the de-convolved mesh GTk/GNk).  Ported from pppm/disp/planar.
------------------------------------------------------------------------- */

void PPPMDispSlab::profile_GTGN_raw(int K, double *GTr, double *GNr)
{
  const double unitk = 2.0 * MY_PI / zprd;
  const double c = cutoff + sw_width;
  const double a = cutoff, dzs = sw_width;
  const int ni = 2000;
  const double dr = dzs / ni;
  double iJ = 0.0, iT = 0.0;
  for (int i = 0; i <= ni; i++) {
    const double r = a + i * dr;
    const double t = (r - a) / dzs;
    const double S = switch_S(t);
    const double Sp = switch_dS(t) / dzs;
    const double r3 = r * r * r, r4 = r3 * r;
    const double w = (i == 0 || i == ni) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    iJ += w * Sp / r3;
    iT += w * S / r4;
  }
  const double Jint = dr / 3.0 * iJ;
  const double trans = dr / 3.0 * iT;
  GTr[0] = GNr[0] = -(2.0 * MY_PI / (3.0 * volume)) * (-Jint + 6.0 * trans + 2.0 / (c * c * c));
  for (int k = 1; k <= K; k++) {
    const double kcell = k * unitk;
    const double kcell3 = kcell * kcell * kcell;
    double C[8], D[8];
    sici_compl_chain(kcell * c, C, D);
    const double GTtail = (-24.0 * MY_PI * kcell3 / volume) * (C[7] - D[6]);
    const double GNtail = (-24.0 * MY_PI * kcell3 / volume) * (C[5] - 2.0 * C[7] + 2.0 * D[6]);
    double sGT, sGN;
    switch_shell_virial(kcell, sGT, sGN);
    GTr[k] = GTtail - (MY_PI / volume) * sGT;
    GNr[k] = GNtail - (2.0 * MY_PI / volume) * sGN;
  }
}

/* ----------------------------------------------------------------------
   Irving-Kirkwood profile assembly (see ewald/disp/slab pressure_profile_long
   for the derivation); ported from pppm/disp/planar.
------------------------------------------------------------------------- */

void PPPMDispSlab::profile_assemble(int K, int nbins, double lo, double width, const double *Sre,
                                    const double *Sim, const double *GTr, const double *GNr,
                                    const double *shellT, const double *shellN, double *pN,
                                    double *pT)
{
  const double unitk = 2.0 * MY_PI / zprd;
  const double inv_adz = 1.0 / (area * width);
  const int P = 2 * K;
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
  auto *phiP = new double[K + 1];
  auto *psiP = new double[K + 1];
  for (int k = 0; k <= K; k++) {
    phiP[k] = ik_phi(k * unitk);
    psiP[k] = ik_psi(k * unitk);
  }
  auto PHI = [&](int n) { return n < 0 ? -phiP[-n] : phiP[n]; };
  auto PSI = [&](int n) { return n < 0 ? -psiP[-n] : psiP[n]; };
  for (int n = -K; n <= K; n++) {
    double hn = n * unitk;
    for (int m = -K; m <= K; m++) {
      int p = n + m;
      if (p < 0) continue;    // Hermitian symmetry; keep p>=0
      double hm = m * unitk, H = hn + hm;
      double snr, sni, smr, smi;
      Sn(n, snr, sni);
      Sn(m, smr, smi);
      double sre = snr * smr - sni * smi, simv = snr * smi + sni * smr;
      double CT, CN;
      if (n == 0 && m == 0) {
        CT = CN = volume * GTr[0];
      } else if (fabs(H) < 1.0e-300) {
        int kk = (n < 0) ? -n : n;
        CT = 0.5 * volume * GTr[kk];
        CN = 0.5 * volume * GNr[kk];
      } else {
        CT = -6.0 * MY_PI / H * (PHI(m) + PHI(n));
        CN = -12.0 * MY_PI / H * (PSI(m) + PSI(n));
      }
      ATre[p] += CT * sre;
      ATim[p] += CT * simv;
      ANre[p] += CN * sre;
      ANim[p] += CN * simv;
    }
  }
  for (int g = 0; g < nbins; g++) {
    double z = lo + (g + 0.5) * width;
    double pt = ATre[0], pn = ANre[0];
    for (int p = 1; p <= P; p++) {
      double cz = cos(p * unitk * z), sz = sin(p * unitk * z);
      pt += 2.0 * (ATre[p] * cz - ATim[p] * sz);
      pn += 2.0 * (ANre[p] * cz - ANim[p] * sz);
    }
    pT[g] = pt - shellT[g] * inv_adz;
    pN[g] = pn - shellN[g] * inv_adz;
  }
  delete[] ATre;
  delete[] ATim;
  delete[] ANre;
  delete[] ANim;
  delete[] phiP;
  delete[] psiP;
}

/* ----------------------------------------------------------------------
   long-range Irving-Kirkwood pressure profiles on the caller's z grid.  The
   merged-damped kspace represents the identical switched tail S(r)*u(r) as the
   compact-switch method (the pair fades the dispersion by (1-S)), so the same
   S*u pressure building blocks (ik_phi/ik_psi + switch shell) apply.
------------------------------------------------------------------------- */

int PPPMDispSlab::pressure_profile_long(int dir, int nbins, double lo, double width,
                                        double *pN, double *pT)
{
  if (dir != dim)
    error->all(FLERR,
               "compute stress/cartesian binning direction must match the inhomogeneous axis "
               "of pppm/disp/slab");

  const double unitk = 2.0 * MY_PI / zprd;
  const int K = profile_kmax();

  if (nbins <= 2 * K)
    error->all(FLERR,
               "compute stress/cartesian with pppm/disp/slab kspace: {} bins along the "
               "inhomogeneous axis is too coarse; need > {} (= 2*K_prof, the force-accuracy "
               "mode cutoff) to resolve the Irving-Kirkwood reciprocal modes without aliasing "
               "(use a finer bin width, looser accuracy, or wider switch)",
               nbins, 2 * K);

  auto *GTr = new double[K + 1];
  auto *GNr = new double[K + 1];
  profile_GTGN_raw(K, GTr, GNr);

  // EXACT structure factors sfac[n] = sum_j B_j exp(i n unitk z_j), n=0..K
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

  auto *Sre = new double[K + 1];
  auto *Sim = new double[K + 1];
  for (int n = 0; n <= K; n++) {
    Sre[n] = srl_all[n] / volume;
    Sim[n] = -sim_all[n] / volume;
  }

  // shell virial subtraction is zero for the merged-damped variant (the kspace
  // GT/GN already carry the full plane mean field of S*u); keep the (zeroed)
  // arrays so profile_assemble's signature is unchanged.
  auto *shellT = new double[nbins];
  auto *shellN = new double[nbins];
  shell_profile_virial(nbins, lo, width, nullptr, shellT, shellN);

  profile_assemble(K, nbins, lo, width, Sre, Sim, GTr, GNr, shellT, shellN, pN, pT);

  delete[] GTr;
  delete[] GNr;
  delete[] srl;
  delete[] sim;
  delete[] srl_all;
  delete[] sim_all;
  delete[] Sre;
  delete[] Sim;
  delete[] shellT;
  delete[] shellN;
  return 1;
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
  double bytes = 10.0 * nz * sizeof(double);    // dens,fre,fim,Gk,GTk,GNk,fz_grid,ugrid,uTgrid,uNgrid
  bytes += (double) nmax * sizeof(double);
  bytes += (double) order * order * sizeof(double);
  bytes += (double) (ncgrid + 1) * sizeof(double);    // corr energy kernel
  return bytes;
}
