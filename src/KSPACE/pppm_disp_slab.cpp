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
  ncgrid = 0;
  cwdz = 0.0;
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
  memory->destroy(GTk);
  memory->destroy(GNk);
  memory->destroy(fz_grid);
  memory->destroy(ugrid);
  memory->destroy(uTgrid);
  memory->destroy(uNgrid);
  delete[] cWgrid;
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

  // m=0 (homogeneous tail) term: W_E(0) = GU[0] = -pi^1.5 g^3 / (6 V), plus corr
  Gk[0] = -MY_PI * sqpi * g_ewald * g_ewald * g_ewald / (6.0 * volume);
  double w2t0, kw2p0;
  corr_tilde(0.0, w2t0, kw2p0);
  const double ce0 = 0.5 * w2t0 / zprd;
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

    double w2t, kw2p;
    corr_tilde(ak, w2t, kw2p);
    const double CE = 0.5 * w2t / zprd;
    const double CN = 0.5 * (w2t + kw2p) / zprd;
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
  const double pre = 2.0 * MY_PI / area;
  ncgrid = 2048;
  cwdz = b / ncgrid;
  delete[] cWgrid;
  cWgrid = new double[ncgrid + 1];
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
    cWgrid[g] = pre * IE;
  }
}

/* ----------------------------------------------------------------------
   1-D Fourier transforms of the tabulated corr kernel (Simpson over the table):
     w2t  = W~2(k)        = 2 int_0^b w2(z) cos(kz) dz
     kw2p = k dW~2(k)/dk  = -2 k int_0^b z w2(z) sin(kz) dz
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

/* ---------------------------------------------------------------------- */

double PPPMDispSlab::memory_usage()
{
  double bytes = 10.0 * nz * sizeof(double);    // dens,fre,fim,Gk,GTk,GNk,fz_grid,ugrid,uTgrid,uNgrid
  bytes += (double) nmax * sizeof(double);
  bytes += (double) order * order * sizeof(double);
  bytes += (double) (ncgrid + 1) * sizeof(double);    // corr energy kernel
  return bytes;
}
