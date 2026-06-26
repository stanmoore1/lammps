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
   Mesh-accelerated planar dispersion Ewald (pppm/disp/planar), the FFT version
   of ewald/disp/planar.  The dispersion-weighted density (geometric mixing)
   varies only in z, so the smooth reciprocal part of the C3-switched 1/r^6 is a
   1-D convolution in z: spread the B-weighted density onto a z grid, FFT in z,
   apply the de-convolved compact-switch influence function, inverse-FFT the
   z-force field, and interpolate.  The reciprocal energy/force reproduce the
   exact ewald/disp/planar result as the grid is refined; the shell correction
   corr_shell() and the H/IK pressure profiles use the same formulas as
   ewald/disp/planar.

   References: S. Moore, dissertation (BYU); this paper.
------------------------------------------------------------------------- */

#include "pppm_disp_planar.h"

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

PPPMDispPlanar::PPPMDispPlanar(LAMMPS *lmp) :
    KSpace(lmp), pt_profile(nullptr), pn_profile(nullptr), B(nullptr), dens(nullptr), fre(nullptr),
    fim(nullptr), Gk(nullptr), GTk(nullptr), GNk(nullptr), fz_grid(nullptr), ugrid(nullptr),
    uTgrid(nullptr), uNgrid(nullptr), wEgrid(nullptr), wFgrid(nullptr), wTgrid(nullptr),
    wNgrid(nullptr), rho_coeff(nullptr), peatom(nullptr)
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
  sw_width = 0.0;
  mix_flag = 0;
  nchan = 1;
  mix_disp_user = -1;
  corr_mode = 0;
  bin_dz_user = 0.0;
  order_allocated = 0;
  nmax = 0;
  accuracy_relative = 0.0;
}

/* ---------------------------------------------------------------------- */

PPPMDispPlanar::~PPPMDispPlanar()
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
  delete[] wEgrid;
  delete[] wFgrid;
  delete[] wTgrid;
  delete[] wNgrid;
  memory->destroy(peatom);
  memory->destroy(pt_profile);
  memory->destroy(pn_profile);
  if (rho_coeff) memory->destroy(rho_coeff);
}

/* ---------------------------------------------------------------------- */

void PPPMDispPlanar::settings(int narg, char **arg)
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

int PPPMDispPlanar::modify_param(int narg, char **arg)
{
  // mesh/disp, order/disp are consumed by the base KSpace parser
  // (they set nz_pppm_6/gridflag_6, order_6).
  if (strcmp(arg[0], "mix/disp") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify mix/disp", error);
    if (strcmp(arg[1], "geom") == 0)
      mix_disp_user = 0;    // force geometric mixing
    else if (strcmp(arg[1], "arith") == 0)
      mix_disp_user = 1;    // force arithmetic / Lorentz-Berthelot mixing
    else if (strcmp(arg[1], "pair") == 0 || strcmp(arg[1], "none") == 0)
      mix_disp_user = -1;    // follow the pair style's mixing rule
    else
      error->all(FLERR, "kspace_modify mix/disp must be geom, arith, pair, or none");
    return 2;
  }
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

void PPPMDispPlanar::init()
{
  if (comm->me == 0) utils::logmesg(lmp, "PPPM slab-based dispersion Ewald (pppm/disp/planar) ...\n");

  triclinic_check();
  if (domain->dimension == 2) error->all(FLERR, "Cannot use pppm/disp/planar with 2d simulation");
  if (domain->triclinic) error->all(FLERR, "Cannot use pppm/disp/planar with triclinic box");
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR, "pppm/disp/planar requires periodic boundaries in all dimensions");
  if (slabflag)
    error->all(FLERR, "Cannot use slab correction (kspace_modify slab) with pppm/disp/planar");
  if (force->pair == nullptr)
    error->all(FLERR, "KSpace style pppm/disp/planar requires a pair style");

  // LJ cutoff from the pair style

  int itmp;
  double *p = (double *) force->pair->extract("cut_lj", itmp);
  if (p == nullptr) p = (double *) force->pair->extract("cut_LJ", itmp);
  if (p == nullptr)
    error->all(FLERR, "Pair style is incompatible with kspace_style pppm/disp/planar");
  cutoff = *p;
  rc2 = cutoff * cutoff;

  // the matched pair style supplies the switch width Delta; "cut_lj" above is the
  // inner rcut.  The pair evaluates the full dispersion u over the shell
  // [rcut, rcut+Delta] (exact 3-D) and corr_shell() removes the reciprocal sum's
  // plane mean-field S*u there, eliminating the lateral-correlation residual in
  // energy and pressure.

  {
    int itmp2;
    double *p_dz = (double *) force->pair->extract("disp_switch_width", itmp2);
    if (p_dz == nullptr)
      error->all(FLERR,
                 "kspace_style pppm/disp/planar requires a pair style that provides the dispersion "
                 "switch width (use pair_style lj/cut/dispplanar)");
    sw_width = *p_dz;
    if (sw_width <= 0.0) error->all(FLERR, "pppm/disp/planar switch width must be > 0");
  }

  // accuracy in force units

  two_charge();
  if (accuracy_absolute >= 0.0)
    accuracy = accuracy_absolute;
  else
    accuracy = accuracy_relative * two_charge_force;

  // dispersion mixing rule (nchan) and the per-type B amplitude array.
  // kspace->init() runs before pair->init(), so lj4 may not be populated yet;
  // init_coeffs() builds B from epsilon/sigma (already set by pair_coeff).

  init_coeffs();

  // stencil order from kspace_modify order/disp (base member; default 5)
  order = order_6;
  if (order < 2 || order > MAXORDER)
    error->all(FLERR, "pppm/disp/planar order/disp must be between 2 and {}", MAXORDER);

  estimate_params();    // choose the z grid size nz

  setup();

  if (comm->me == 0) {
    utils::logmesg(lmp,
                   "  planar dispersion PPPM, z grid = {}, stencil order = {}, switch width "
                   "Delta = {:.6g}\n",
                   nz, order, sw_width);
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

double PPPMDispPlanar::compute_qopt(int ngrid, int ord)
{
  const double unitk = 2.0 * MY_PI / zprd;
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
        // compact switch: the de-convolved potential coefficient is the CSB GU
        // at the aliased mode (no g_ewald; the switch sets the spectrum).  gu_switch
        // is roundoff-limited at high modes -- its analytic ~k^-5 tail is formed
        // by a k^3 cancellation, so below ~(4pi/V) k^3 * 1e-13 the value is noise.
        // The true contribution there is negligible, so zero it; this keeps qopt
        // monotonically decreasing in the grid (otherwise the noise floor, summed as
        // D^2 k^2, makes qopt spuriously blow up at fine grids / tight targets).
        D = gu_switch(meff < 0 ? -meff : meff);
        if (fabs(D) < (4.0 * MY_PI / volume) * ak * ak * ak * 1.0e-13) D = 0.0;
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
   choose the z grid size nz from the target force accuracy.

   nz is the accuracy control: the mesh RMS force error
     df = sqrt(qopt(nz)) * b2 / sqrt(N)   (1-D Hockney-Eastwood, b2 = sum B_i^2)
   captures the aliasing/interpolation AND truncation error of the grid.  Pick
   the smallest power-of-two nz with df < accuracy.
------------------------------------------------------------------------- */

void PPPMDispPlanar::estimate_params()
{
  set_grid_params();

  // the compact switch has no g_ewald: the switch width Delta, fixed by the pair,
  // sets the reciprocal spectrum.

  // dispersion sum b2 = sum_i B_i^2 (full system).  B_t = 2 sqrt(eps_tt) sigma_tt^3
  // = sqrt(C6_tt) is the per-type self amplitude, independent of the mixing rule, so
  // the grid estimate is the same for the geometric and arithmetic B layouts.

  int *type = atom->type;
  int nlocal = atom->nlocal;
  int ntypes = atom->ntypes;
  int etmp;
  auto **eps = (double **) force->pair->extract("epsilon", etmp);
  auto **sig = (double **) force->pair->extract("sigma", etmp);
  auto *Bt = new double[ntypes + 1];
  for (int t = 1; t <= ntypes; t++)
    Bt[t] = (eps && sig) ? 2.0 * sqrt(eps[t][t]) * sig[t][t] * sig[t][t] * sig[t][t]
                         : (nchan == 1 ? B[t] : B[7 * t]);
  double b2_local = 0.0;
  for (int i = 0; i < nlocal; i++) b2_local += Bt[type[i]] * Bt[type[i]];
  delete[] Bt;
  double b2;
  MPI_Allreduce(&b2_local, &b2, 1, MPI_DOUBLE, MPI_SUM, world);
  double natoms = (double) atom->natoms;
  if (natoms < 1.0) natoms = 1.0;
  const double pref = b2 / sqrt(natoms);

  if (gridflag_6 && nz_pppm_6 > 0) {
    nz = 1;
    while (nz < nz_pppm_6) nz <<= 1;    // round up to a power of two for the FFT
  } else {
    // The qopt model (optimal influence function) under-predicts the de-convolved
    // mesh force error by ~1.8x for the compact switch (measured vs the RMS force
    // calculator); fold that into the selection so the chosen grid meets the target.
    // gu_switch noise at fine grids is clamped in compute_qopt, so qopt is monotone
    // and the search terminates without the old pathological run-up; keep a generous
    // ceiling only as a safety net.
    const double bias = 1.8;
    const int ngrid_max = 16384;
    int ngrid = 16;
    while (ngrid < ngrid_max) {
      double df = bias * sqrt(compute_qopt(ngrid, order)) * pref;
      if (df < accuracy) break;
      ngrid <<= 1;
    }
    nz = ngrid;
  }
  if (nz < 8) nz = 8;

  const double bias = 1.8;
  estimated_force_accuracy = bias * sqrt(compute_qopt(nz, order)) * pref;
  if (nz >= 16384 && estimated_force_accuracy > accuracy && comm->me == 0)
    error->warning(FLERR,
                   "pppm/disp/planar: grid capped at nz={}; estimated force "
                   "accuracy {:.3g} exceeds the target {:.3g}",
                   nz, estimated_force_accuracy, accuracy);
}

/* ----------------------------------------------------------------------
   geometry, grid spacing, and B-spline stencil parameters
------------------------------------------------------------------------- */

void PPPMDispPlanar::set_grid_params()
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

/* ----------------------------------------------------------------------
   set the dispersion mixing rule (nchan) and build the per-type B amplitude
   array.  Identical layout/normalization to ewald/disp/planar:
     geometric (mix_flag 0): B[t] = 2 sqrt(eps_t) sigma_t^3, one per type (n+1).
     arithmetic (mix_flag 1): the 7-channel binomial expansion of
       (0.5(sigma_i+sigma_j))^6, B[7*t+j] = sigma_t^j sqrt(eps_t) c[j],
       c={1,sqrt6,sqrt15,sqrt20,sqrt15,sqrt6,1}, so the cross amplitude
       sum_j B[7*i+j] B[7*j_type+(6-j)] reproduces 4 sqrt(eps_i eps_j)
       ((sigma_i+sigma_j)/2)^6.  For a single type this reduces to
       4 eps sigma^6 = B_geom^2, so single-type results are bit-identical.
   The mixing rule follows the pair style (extract "ewald_mix") unless the user
   forced it via kspace_modify mix/disp.
------------------------------------------------------------------------- */

void PPPMDispPlanar::init_coeffs()
{
  int tmp;
  int n = atom->ntypes;

  int *p_mix = (int *) force->pair->extract("ewald_mix", tmp);
  int pair_mix = p_mix ? *p_mix : Pair::GEOMETRIC;
  if (mix_disp_user == 0)
    mix_flag = 0;
  else if (mix_disp_user == 1)
    mix_flag = 1;
  else if (pair_mix == Pair::GEOMETRIC)
    mix_flag = 0;
  else if (pair_mix == Pair::ARITHMETIC)
    mix_flag = 1;
  else
    error->all(FLERR,
               "Unsupported pair mixing rule for kspace_style pppm/disp/planar "
               "(use pair_modify mix geometric|arithmetic, or kspace_modify mix/disp)");
  nchan = mix_flag ? 7 : 1;

  delete[] B;

  if (mix_flag == 0) {    // geometric: single per-type amplitude B[t]=2 sqrt(eps) sigma^3
    auto **eps = (double **) force->pair->extract("epsilon", tmp);
    auto **sig = (double **) force->pair->extract("sigma", tmp);
    if (eps == nullptr || sig == nullptr)
      error->all(FLERR, "Pair style does not provide epsilon/sigma for pppm/disp/planar");
    B = new double[n + 1];
    B[0] = 0.0;
    for (int t = 1; t <= n; t++) B[t] = 2.0 * sqrt(eps[t][t]) * sig[t][t] * sig[t][t] * sig[t][t];
  } else {    // arithmetic (Lorentz-Berthelot): 7-channel binomial expansion
    auto **epsilon = (double **) force->pair->extract("epsilon", tmp);
    auto **sigma = (double **) force->pair->extract("sigma", tmp);
    if (!(epsilon && sigma))
      error->all(FLERR,
                 "Pair style does not provide epsilon/sigma for arithmetic mixing in "
                 "pppm/disp/planar");
    B = new double[7 * n + 7];
    const double c[7] = {1.0, sqrt(6.0), sqrt(15.0), sqrt(20.0), sqrt(15.0), sqrt(6.0), 1.0};
    for (int j = 0; j < 7; ++j) B[j] = 0.0;    // type 0 (unused)
    for (int i = 1; i <= n; ++i) {
      const double eps_i = sqrt(epsilon[i][i]);
      const double sigma_i = sigma[i][i];
      double sigma_p = 1.0;
      for (int j = 0; j < 7; ++j) {
        B[7 * i + j] = sigma_p * eps_i * c[j];
        sigma_p *= sigma_i;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PPPMDispPlanar::setup()
{
  set_grid_params();
  // reset the mixing rule (nchan) in case the pair style changed; sizes the
  // per-channel density / field grids below
  init_coeffs();
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
  // dens and the force/potential fields carry nchan channels (1 geom, 7 arith);
  // the FFT workspace (fre/fim) and the influence functions (Gk/GTk/GNk) are
  // scalar per-mode (mixing-independent).
  memory->create(dens, nz * nchan, "pppm/disp/planar:dens");
  memory->create(fre, nz, "pppm/disp/planar:fre");
  memory->create(fim, nz, "pppm/disp/planar:fim");
  memory->create(Gk, nz, "pppm/disp/planar:Gk");
  memory->create(GTk, nz, "pppm/disp/planar:GTk");
  memory->create(GNk, nz, "pppm/disp/planar:GNk");
  memory->create(fz_grid, nz * nchan, "pppm/disp/planar:fz_grid");
  memory->create(ugrid, nz * nchan, "pppm/disp/planar:ugrid");
  memory->create(uTgrid, nz * nchan, "pppm/disp/planar:uTgrid");
  memory->create(uNgrid, nz * nchan, "pppm/disp/planar:uNgrid");

  if (rho_coeff == nullptr || order != order_allocated) {
    if (rho_coeff) memory->destroy(rho_coeff);
    memory->create(rho_coeff, order, order, "pppm/disp/planar:rho_coeff");
    order_allocated = order;
  }
  compute_rho_coeff();

  influence_function();

  build_shell_vkernels();
}

/* ----------------------------------------------------------------------
   de-convolved compact-switch influence function on the z grid modes.

   The PPPM mesh energy E = sum_{m=0}^{nz-1} Gk[m] |rho_hat_m|^2 (full FFT
   spectrum) must equal the exact ewald/disp/planar energy
   E = GU[0]|S_0|^2 + sum_{k>=1} GU[k]|S_k|^2.  Matching the spectra term by
   term (each +/- mode appears once in the FFT sum) gives the physical
   per-mode coefficient W_E(k) = GU[0] for m=0 and GU[|m|]/2 for m != 0.
   De-convolving the order-p assignment (transfer W(k)=sinc(pi m/nz)^order):
     Gk[m] = W_E(k_m) / W(k_m)^2 .
   GU/GT/GN are the compact-switch coefficients (gu_switch/switch_shell_virial),
   built at k = |mm| * unitk; there is no g_ewald and no real-space correction.
------------------------------------------------------------------------- */

void PPPMDispPlanar::influence_function()
{
  {
    const double c = cutoff + sw_width;
    Gk[0] = gu0_switch();
    // k=0 (uniform) virial of S*u; GT[0]=GN[0] (includes the S'u switch term)
    {
      const double a = cutoff, dz = sw_width;
      const int n = 2000;
      const double dr = dz / n;
      double iJ = 0.0, iT = 0.0;
      for (int i = 0; i <= n; i++) {
        const double r = a + i * dr;
        const double t = (r - a) / dz;
        const double S = switch_S(t);
        const double Sp = switch_dS(t) / dz;    // S'(r)
        const double r3 = r * r * r, r4 = r3 * r;
        const double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        iJ += w * Sp / r3;
        iT += w * S / r4;
      }
      const double Jint = dr / 3.0 * iJ;
      const double trans = dr / 3.0 * iT;
      GTk[0] = GNk[0] =
          -(2.0 * MY_PI / (3.0 * volume)) * (-Jint + 6.0 * trans + 2.0 / (c * c * c));
    }
    for (int m = 1; m < nz; m++) {
      int mm = (m <= nz / 2) ? m : m - nz;    // signed mode index
      int am = mm < 0 ? -mm : mm;             // |mm|
      double kcell = am * 2.0 * MY_PI / zprd;
      double kcell3 = kcell * kcell * kcell;
      double s = sin(MY_PI * mm / nz) / (MY_PI * mm / nz);
      double w2 = pow(s, 2 * order);
      const double inv = 0.5 / w2;    // 0.5 = GU[|mm|]/2 double-count, /w2 de-conv

      double C[8], D[8];
      sici_compl_chain(kcell * c, C, D);
      const double t5 = switch_trans5(kcell);
      const double GU = (-4.0 * MY_PI * kcell3 / volume) * C[5] -
          (4.0 * MY_PI / volume) * t5 / kcell;
      const double GTtail = (-24.0 * MY_PI * kcell3 / volume) * (C[7] - D[6]);
      const double GNtail =
          (-24.0 * MY_PI * kcell3 / volume) * (C[5] - 2.0 * C[7] + 2.0 * D[6]);
      double sGT, sGN;
      switch_shell_virial(kcell, sGT, sGN);
      const double GT = GTtail - (MY_PI / volume) * sGT;
      const double GN = GNtail - (2.0 * MY_PI / volume) * sGN;

      Gk[m] = GU * inv;
      GTk[m] = GT * inv;
      GNk[m] = GN * inv;
    }
  }
}

/* ----------------------------------------------------------------------
   compact-switch reciprocal-coefficient helpers (verbatim from
   ewald/disp/planar; see that file for the derivations and accuracy notes).
------------------------------------------------------------------------- */

double PPPMDispPlanar::switch_S(double t)
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t, t3 = t2 * t, t4 = t3 * t;    // C3 septic smoothstep
  return t4 * (35.0 - 84.0 * t + 70.0 * t2 - 20.0 * t3);
}

/* ---------------------------------------------------------------------- */

double PPPMDispPlanar::switch_dS(double t)
{
  // dS/dt of the C3 septic smoothstep = 7!/(3!)^2 (t(1-t))^3 = 140 (t(1-t))^3
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double tu = t * (1.0 - t);
  return 140.0 * tu * tu * tu;
}

/* ----------------------------------------------------------------------
   energy shell integral t5 = int_rcut^{rcut+Delta} S(r) r^-5 sin(h r) dr
   (10-point Gauss-Legendre per panel, panel count scaled to the oscillation
   count so the result is accurate ~1e-13 for all h).
------------------------------------------------------------------------- */

double PPPMDispPlanar::switch_trans5(double h)
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

void PPPMDispPlanar::switch_shell_virial(double h, double &sGT, double &sGN)
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

double PPPMDispPlanar::gu_switch(int k)
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

double PPPMDispPlanar::gu0_switch()
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
   free high-k tail coefficients); see ewald/disp/planar.
------------------------------------------------------------------------- */

void PPPMDispPlanar::sici_compl_chain(double x, double *Carr, double *Darr)
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
   B-spline assignment polynomial coefficients (LAMMPS PPPM convention).
   rho_coeff[l][s] gives the order-l term for stencil point s = 0..order-1
   (grid offset nlower+s); weight at fractional offset dz = sum_l rho_coeff[l][s] dz^l.
------------------------------------------------------------------------- */

void PPPMDispPlanar::compute_rho_coeff()
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

void PPPMDispPlanar::compute_rho1d(double dz, double *w)
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

void PPPMDispPlanar::compute_drho1d(double dz, double *dw)
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

void PPPMDispPlanar::make_rho()
{
  for (int g = 0; g < nz * nchan; g++) dens[g] = 0.0;    // channel-major: dens[m*nz+g]

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double w[MAXORDER];

  if (nchan == 1) {    // geometric: single B-weighted density grid

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

  } else {    // arithmetic: spread 7 density channels dens[m*nz+g] (m=0..6)

    for (int i = 0; i < nlocal; i++) {
      double u = (x[i][dim] - zlo) * delzinv;
      int g0 = (int) (u + (order % 2 ? OFFSET + 0.5 : OFFSET)) - OFFSET;
      double dz = g0 + shiftone - u;
      compute_rho1d(dz, w);
      const double *bi = &B[7 * type[i]];    // 7 channel amplitudes of atom i
      for (int s = 0; s < order; s++) {
        int g = g0 + nlower + s;
        g = ((g % nz) + nz) % nz;
        const double ws = w[s];
        for (int m = 0; m < 7; m++) dens[m * nz + g] += bi[m] * ws;
      }
    }
  }

  double *tmp;
  memory->create(tmp, nz * nchan, "pppm/disp/planar:tmp");
  MPI_Allreduce(dens, tmp, nz * nchan, MPI_DOUBLE, MPI_SUM, world);
  for (int g = 0; g < nz * nchan; g++) dens[g] = tmp[g];
  memory->destroy(tmp);
}

/* ----------------------------------------------------------------------
   radix-2 in-place FFT (n a power of two); sign=-1 forward, +1 inverse,
   both unnormalized (X_m = sum_g x_g e^{-2pi i m g/n} for sign=-1)
------------------------------------------------------------------------- */

void PPPMDispPlanar::fft1d(double *re, double *im, int n, int sign)
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

void PPPMDispPlanar::poisson()
{
  if (nchan == 1) {    // geometric: single B-weighted density grid

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
      // compact switch: explicit tangential (GTk) and normal (GNk) kernels (the
      // homogeneity trace relation does not hold for the non-power-law S*u)
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

    // per-atom potential field u_grid = IFFT[2 Gk rho_hat]

    if (evflag_atom) {
      double *ur, *ui;
      memory->create(ur, nz, "pppm/disp/planar:ur");
      memory->create(ui, nz, "pppm/disp/planar:ui");
      for (int m = 0; m < nz; m++) {
        double g2 = 2.0 * Gk[m];
        ur[m] = g2 * fre[m];
        ui[m] = g2 * fim[m];
      }
      fft1d(ur, ui, nz, +1);
      for (int g = 0; g < nz; g++) ugrid[g] = ur[g];
      // compact switch: tangential/normal per-atom virial fields (GTk/GNk kernels)
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
    return;
  }

  // ----- arithmetic (Lorentz-Berthelot): 7 density channels -----
  //
  // FFT each channel's density to rho_hat_m, then form the per-mode channel
  // pairing (the mesh analog of ewald/disp/planar's R_k):
  //   R[mode] = Re(rho0 conj(rho6)) + Re(rho1 conj(rho5)) + Re(rho2 conj(rho4))
  //             + 0.5 |rho3|^2          (folded m<->6-m pairing, each pair once)
  // The energy E = AS_E sum_mode Gk[mode] R[mode] with AS_E = 1/8 (the channels
  // expand (sigma_i+sigma_j)^6 = 16 C6_ij; the m<->6-m folding halves it again).
  // For a single type R/8 == |rho_hat|^2 exactly, so the geometric path is
  // recovered bit-for-bit.  The influence functions Gk/GTk/GNk are unchanged.

  const double as_e = 0.125;    // 1/8  energy / virial normalization

  // store the 7 FFT'd density channels (channel-major rho_hat_m[mode])
  auto *rre = new double[7 * nz];
  auto *rim = new double[7 * nz];
  for (int m = 0; m < 7; m++) {
    for (int g = 0; g < nz; g++) {
      rre[m * nz + g] = dens[m * nz + g];
      rim[m * nz + g] = 0.0;
    }
    fft1d(&rre[m * nz], &rim[m * nz], nz, -1);
  }

  // per-mode folded channel pairing R[mode]
  auto Rmode = [&](int mode) -> double {
    const double *r0 = &rre[0 * nz], *i0 = &rim[0 * nz];
    const double *r1 = &rre[1 * nz], *i1 = &rim[1 * nz];
    const double *r2 = &rre[2 * nz], *i2 = &rim[2 * nz];
    const double *r3 = &rre[3 * nz], *i3 = &rim[3 * nz];
    const double *r4 = &rre[4 * nz], *i4 = &rim[4 * nz];
    const double *r5 = &rre[5 * nz], *i5 = &rim[5 * nz];
    const double *r6 = &rre[6 * nz], *i6 = &rim[6 * nz];
    return (r0[mode] * r6[mode] + i0[mode] * i6[mode]) +
        (r1[mode] * r5[mode] + i1[mode] * i5[mode]) +
        (r2[mode] * r4[mode] + i2[mode] * i4[mode]) +
        0.5 * (r3[mode] * r3[mode] + i3[mode] * i3[mode]);
  };

  double e = 0.0;
  for (int mode = 0; mode < nz; mode++) e += Gk[mode] * Rmode(mode);
  e *= as_e;
  e_recip_mesh = e;
  if (eflag_global) energy += e;
  if (vflag_global) {
    double vt = 0.0, vn = 0.0;
    for (int mode = 0; mode < nz; mode++) {
      double R = Rmode(mode);
      vt += GTk[mode] * R;
      vn += GNk[mode] * R;
    }
    virial[lat1] += as_e * vt;
    virial[lat2] += as_e * vt;
    virial[dim] += as_e * vn;
  }

  // per-atom potential / virial fields, one per channel:
  //   ugrid[m*nz+.] = IFFT[2 Gk rho_hat_m]  (and GTk/GNk variants).
  // fieldforce() pairs atom channel (6-m) with field channel m and applies the
  // per-atom 0.25*as_e normalization (see fieldforce()).

  if (evflag_atom) {
    double *ur, *ui;
    memory->create(ur, nz, "pppm/disp/planar:ur");
    memory->create(ui, nz, "pppm/disp/planar:ui");
    for (int m = 0; m < 7; m++) {
      for (int mode = 0; mode < nz; mode++) {
        double g2 = 2.0 * Gk[mode];
        ur[mode] = g2 * rre[m * nz + mode];
        ui[mode] = g2 * rim[m * nz + mode];
      }
      fft1d(ur, ui, nz, +1);
      for (int g = 0; g < nz; g++) ugrid[m * nz + g] = ur[g];
      for (int mode = 0; mode < nz; mode++) {
        double gt2 = 2.0 * GTk[mode];
        ur[mode] = gt2 * rre[m * nz + mode];
        ui[mode] = gt2 * rim[m * nz + mode];
      }
      fft1d(ur, ui, nz, +1);
      for (int g = 0; g < nz; g++) uTgrid[m * nz + g] = ur[g];
      for (int mode = 0; mode < nz; mode++) {
        double gn2 = 2.0 * GNk[mode];
        ur[mode] = gn2 * rre[m * nz + mode];
        ui[mode] = gn2 * rim[m * nz + mode];
      }
      fft1d(ur, ui, nz, +1);
      for (int g = 0; g < nz; g++) uNgrid[m * nz + g] = ur[g];
    }
    memory->destroy(ur);
    memory->destroy(ui);
  }

  // per-channel z-force field: Ffield_m = IFFT[-i k 2 Gk rho_hat_m].
  // fieldforce() applies f = AS_F sum_m bi[6-m] Ffield_m(z_i), AS_F = 1/16.

  double *fr, *fi;
  memory->create(fr, nz, "pppm/disp/planar:fr");
  memory->create(fi, nz, "pppm/disp/planar:fi");
  for (int m = 0; m < 7; m++) {
    for (int mode = 0; mode < nz; mode++) {
      int mm = (mode <= nz / 2) ? mode : mode - nz;
      double k = mm * 2.0 * MY_PI / zprd;
      double g2k = 2.0 * Gk[mode] * k;
      double a = rre[m * nz + mode], bb = rim[m * nz + mode];
      fr[mode] = g2k * bb;     // Re(-i k 2Gk (a+ib)) = 2Gk k b
      fi[mode] = -g2k * a;     // Im = -2Gk k a
    }
    fft1d(fr, fi, nz, +1);
    for (int g = 0; g < nz; g++) fz_grid[m * nz + g] = fr[g];
  }
  memory->destroy(fr);
  memory->destroy(fi);

  delete[] rre;
  delete[] rim;
}

/* ----------------------------------------------------------------------
   interpolate the z-force field (and per-atom energy/virial) to atoms
------------------------------------------------------------------------- */

void PPPMDispPlanar::fieldforce()
{
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double w[MAXORDER];

  if (nchan == 1) {    // geometric: single B-weighted field

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
          // explicit tangential (GTk) and normal (GNk) per-atom virial fields
          vatom[i][lat1] += 0.5 * bi * uT;
          vatom[i][lat2] += 0.5 * bi * uT;
          vatom[i][dim] += 0.5 * bi * uN;
        }
      }
    }
    return;
  }

  // ----- arithmetic: pair atom channel (6-m) with field channel m -----
  //
  // z-force:  f = AS_F sum_m bi[6-m] Ffield_m(z_i),  AS_F = 1/16 = AS_E/2 (the
  //   force differentiates both indices of the bilinear, restoring the factor 2).
  //   For a single type sum_m bi[6-m] Ffield_m == 64 * (geometric field)/B, so
  //   AS_F=1/16 reproduces the geometric B-weighted force exactly.
  // per-atom energy/virial use the ordered channel sum with the field built from
  //   2 Gk rho_hat (same FFTs as the force); the per-atom normalization is then
  //   0.25*AS_E (the 2 Gk field factor vs ewald's 0.5*AS_E on the raw products).

  const double as_f = 1.0 / 16.0;          // z-force normalization
  const double as_pe = 0.25 * 0.125;       // 0.25*AS_E per-atom (= 1/32)

  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zlo) * delzinv;
    int g0 = (int) (u + (order % 2 ? OFFSET + 0.5 : OFFSET)) - OFFSET;
    double dz = g0 + shiftone - u;
    compute_rho1d(dz, w);
    const double *bi = &B[7 * type[i]];

    double fz = 0.0, uu = 0.0, uT = 0.0, uN = 0.0;
    for (int s = 0; s < order; s++) {
      int g = g0 + nlower + s;
      g = ((g % nz) + nz) % nz;
      const double ws = w[s];
      for (int m = 0; m < 7; m++) {
        const double a = bi[6 - m];    // atom channel (6-m) pairs with field channel m
        fz += a * ws * fz_grid[m * nz + g];
        if (evflag_atom) {
          uu += a * ws * ugrid[m * nz + g];
          uT += a * ws * uTgrid[m * nz + g];
          uN += a * ws * uNgrid[m * nz + g];
        }
      }
    }
    f[i][dim] += as_f * fz;

    if (evflag_atom) {
      peatom[i] += as_pe * uu;
      if (vflag_atom) {
        vatom[i][lat1] += as_pe * uT;
        vatom[i][lat2] += as_pe * uT;
        vatom[i][dim] += as_pe * uN;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PPPMDispPlanar::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // grow per-atom energy buffer if needed

  if (atom->nmax > nmax) {
    memory->destroy(peatom);
    nmax = atom->nmax;
    memory->create(peatom, nmax, "pppm/disp/planar:peatom");
  }
  if (evflag_atom)
    for (int i = 0; i < atom->nlocal; i++) peatom[i] = 0.0;

  make_rho();
  poisson();
  fieldforce();

  // compact-switch shell correction: subtract the reciprocal sum's plane mean-field
  // S*u over the shell [rcut, rcut+Delta] (energy, z-force, virial) so the pair's
  // exact 3-D shell remains.  It must run every step (the z-force is removed
  // unconditionally, else it is double counted).  The normal (zz) virial is the
  // explicit GNk kernel accumulated in poisson()/fieldforce() (the switch is
  // non-homogeneous, so the trace identity 6U = sum r.f does not apply).

  corr_energy = 0.0;
  corr_shell();

  if (eflag_atom)
    for (int i = 0; i < atom->nlocal; i++) eatom[i] += peatom[i];

  if (profile_flag) compute_pressure_profile();
}

/* ----------------------------------------------------------------------
   tabulate the plane (mean-field) energy, z-force and virial kernels of the
   long-range part S(r)*u(r), u=-1/r^6, over the shell [rcut, rcut+Delta] as
   functions of |dz|.  These are what the reciprocal sum injects there with a
   laterally-uniform density (each atom seen as a uniform sheet of areal
   B-density 1/area); corr_shell subtracts them so the pair's exact 3-D shell
   interaction replaces the mean field.  Per (B_i B_j), with pre = pi/area (the
   plane integral's 2 pi/area times the Ewald 1/2 folded into GU[k], see below):
     wE(dz) = -pre       int_{rlo}^{b} S(r) r^-5 dr               (energy)
     wF(dz) =  pre       S(|dz|) |dz|^-6   for |dz|>rcut, else 0   (so that the
               plane z-force on i from j is delz * B_i B_j * wF, the z-gradient
               of wE; vanishes for |dz|<rcut where the limit rlo=rcut is fixed)
     wT(dz) = -pre/2     int_{rlo}^{b} (S u)'(r) (r^2 - dz^2) dr   (tangential)
     wN(dz) = -pre dz^2  int_{rlo}^{b} (S u)'(r) dr               (normal)
   with rlo = max(|dz|,rcut), b = rcut+Delta, (S u)'(r) = -S'(r)/r^6 + 6 S(r)/r^7.
   Identical math to ewald/disp/planar (the kernels do not depend on how the
   reciprocal sum is evaluated -- they cancel the same plane mean field).
------------------------------------------------------------------------- */

void PPPMDispPlanar::build_shell_vkernels()
{
  const double a = cutoff, b = cutoff + sw_width;
  const double area = domain->prd[lat1] * domain->prd[lat2];
  // pre = (1/2)(2 pi/area): the plane integral of S*u carries 2 pi/area; the 1/2 is
  // the Ewald factor folded into the reciprocal coefficients (E = sum_k GU[k]|S_k|^2
  // with GU[k]=(1/2) u-tilde(k)).  Verified by <kernel>_z == GU[0]/GT[0] shell parts.
  const double pre = MY_PI / area;
  nwgrid = 1024;
  wdz = b / nwgrid;
  delete[] wEgrid;
  delete[] wFgrid;
  delete[] wTgrid;
  delete[] wNgrid;
  wEgrid = new double[nwgrid + 1];
  wFgrid = new double[nwgrid + 1];
  wTgrid = new double[nwgrid + 1];
  wNgrid = new double[nwgrid + 1];
  for (int g = 0; g <= nwgrid; g++) {
    const double adz = g * wdz;
    const double rlo = MAX(adz, a);
    if (rlo >= b) {
      wEgrid[g] = wFgrid[g] = wTgrid[g] = wNgrid[g] = 0.0;
      continue;
    }
    const int n = 600;
    const double hr = (b - rlo) / n;
    double IE = 0.0, IT = 0.0, IN = 0.0;
    for (int i = 0; i <= n; i++) {
      const double r = rlo + i * hr;
      const double t = (r - a) / sw_width;
      const double S = switch_S(t);
      const double Sp = switch_dS(t) / sw_width;    // S'(r)
      const double r2 = r * r, r4 = r2 * r2, r5 = r4 * r, r6 = r4 * r2, r7 = r6 * r;
      const double Sup = -Sp / r6 + 6.0 * S / r7;    // (S u)'
      const double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
      IE += w * S / r5;
      IT += w * Sup * (r2 - adz * adz);
      IN += w * Sup;
    }
    IE *= hr / 3.0;
    IT *= hr / 3.0;
    IN *= hr / 3.0;
    wEgrid[g] = -pre * IE;
    // z-force kernel: only the moving limit rlo=|dz| (|dz|>rcut) contributes
    const double S_dz = (adz > a) ? switch_S((adz - a) / sw_width) : 0.0;
    const double adz6 = (adz > 0.0) ? adz * adz * adz * adz * adz * adz : 1.0;
    wFgrid[g] = (adz > a) ? pre * S_dz / adz6 : 0.0;
    wTgrid[g] = -0.5 * pre * IT;
    wNgrid[g] = -pre * adz * adz * IN;
  }
}

/* ---------------------------------------------------------------------- */

void PPPMDispPlanar::shell_vkernel(double adz, double &wE, double &wF, double &wT, double &wN)
{
  if (adz >= nwgrid * wdz) {
    wE = wF = wT = wN = 0.0;
    return;
  }
  const double x = adz / wdz;
  int g = (int) x;
  if (g >= nwgrid) g = nwgrid - 1;
  const double f = x - g;
  wE = wEgrid[g] * (1.0 - f) + wEgrid[g + 1] * f;
  wF = wFgrid[g] * (1.0 - f) + wFgrid[g + 1] * f;
  wT = wTgrid[g] * (1.0 - f) + wTgrid[g + 1] * f;
  wN = wNgrid[g] * (1.0 - f) + wNgrid[g + 1] * f;
}

/* ----------------------------------------------------------------------
   compact-switch shell correction dispatcher
------------------------------------------------------------------------- */

void PPPMDispPlanar::corr_shell()
{
  if (corr_mode == 1)
    corr_shell_bin();
  else
    corr_shell_raw();
}

/* ----------------------------------------------------------------------
   exact (global z-gather) subtraction of the plane (mean-field) shell energy,
   z-force and virial.  Every proc gathers the global (z, B) list and each local
   atom sums the plane kernel over all global atoms in its |dz| < rcut+Delta
   window (slab-slab).  Removes what the reciprocal sum put in the shell with a
   laterally-uniform density so the matched pair's exact 3-D shell interaction
   (full u to rcut+Delta) is what remains.  Full ordered double sum incl. self,
   so energy/virial carry no 1/2; the z-force = -d E/d z_i differentiates both
   pair indices and so carries a factor 2.
------------------------------------------------------------------------- */

void PPPMDispPlanar::corr_shell_raw()
{
  const double zprd = domain->prd[dim];
  const double bcut = cutoff + sw_width;
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

  // gather z and the nchan dispersion channels of every atom.  For the arithmetic
  // path each atom carries its 7 binomial channels B[7t+0..6]; the per-pair C6 cross
  // amplitude is then (1/16) sum_m a_i[m] a_j[6-m] (the full ordered binomial sum
  // (sigma_i+sigma_j)^6 = 16 C6_ij), matching the geometric bij = B_i B_j = C6_ij.
  auto *zloc = new double[nlocal > 0 ? nlocal : 1];
  auto *bloc = new double[(nlocal > 0 ? nlocal : 1) * nchan];
  for (int i = 0; i < nlocal; i++) {
    zloc[i] = x[i][dim];
    if (nchan == 1)
      bloc[i] = B[type[i]];
    else
      for (int m = 0; m < 7; m++) bloc[i * 7 + m] = B[7 * type[i] + m];
  }
  auto *zall = new double[natoms_all > 0 ? natoms_all : 1];
  auto *ball = new double[(natoms_all > 0 ? natoms_all : 1) * nchan];
  MPI_Allgatherv(zloc, nlocal, MPI_DOUBLE, zall, recvcounts, displs, MPI_DOUBLE, world);
  int *rc_b = new int[nprocs];
  int *dp_b = new int[nprocs];
  for (int p = 0; p < nprocs; p++) {
    rc_b[p] = recvcounts[p] * nchan;
    dp_b[p] = displs[p] * nchan;
  }
  MPI_Allgatherv(bloc, nlocal * nchan, MPI_DOUBLE, ball, rc_b, dp_b, MPI_DOUBLE, world);
  delete[] rc_b;
  delete[] dp_b;

  const double as_shell = 1.0 / 16.0;    // arithmetic C6 cross normalization

  double e_local = 0.0, vt_local = 0.0, vn_local = 0.0;
  for (int i = 0; i < nlocal; i++) {
    const double zi = x[i][dim];
    const double bi = (nchan == 1) ? B[type[i]] : 0.0;
    const double *ai = (nchan == 1) ? nullptr : &B[7 * type[i]];
    double e_i = 0.0, fz_i = 0.0, vt_i = 0.0, vn_i = 0.0;
    for (int jg = 0; jg < natoms_all; jg++) {
      double delz = zi - zall[jg];
      delz -= zprd * floor(delz / zprd + 0.5);    // nearest image
      const double adz = fabs(delz);
      if (adz >= bcut) continue;
      double wE, wF, wT, wN;
      shell_vkernel(adz, wE, wF, wT, wN);
      double bij;
      if (nchan == 1) {
        bij = bi * ball[jg];
      } else {
        const double *aj = &ball[jg * 7];
        double cross = 0.0;
        for (int m = 0; m < 7; m++) cross += ai[m] * aj[6 - m];
        bij = as_shell * cross;
      }
      e_i += bij * wE;
      fz_i += 2.0 * delz * bij * wF;    // remove the plane z-force (factor 2: see above)
      vt_i += bij * wT;
      vn_i += bij * wN;
    }
    e_local += e_i;
    vt_local += vt_i;
    vn_local += vn_i;
    f[i][dim] += fz_i;    // f -= (plane force); plane force = -2 sum delz bij wF
    if (evflag_atom) peatom[i] -= e_i;    // per-atom energy of i = sum_j bij wE
    if (vflag_atom) {
      vatom[i][lat1] -= vt_i;
      vatom[i][lat2] -= vt_i;
      vatom[i][dim] -= vn_i;
    }
  }

  if (eflag_global || vflag_global) {
    double e_all;
    MPI_Allreduce(&e_local, &e_all, 1, MPI_DOUBLE, MPI_SUM, world);
    corr_energy -= e_all;
    if (eflag_global) energy -= e_all;
  }
  if (vflag_global) {
    double vt_all, vn_all;
    MPI_Allreduce(&vt_local, &vt_all, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&vn_local, &vn_all, 1, MPI_DOUBLE, MPI_SUM, world);
    virial[lat1] -= vt_all;
    virial[lat2] -= vt_all;
    virial[dim] -= vn_all;
  }

  delete[] recvcounts;
  delete[] displs;
  delete[] zloc;
  delete[] bloc;
  delete[] zall;
  delete[] ball;
}

/* ----------------------------------------------------------------------
   z-binned version of the shell correction (1D particle-mesh, CIC).  Bins the
   B-weighted density, convolves with the plane kernels, interpolates back.
   The force is the exact gradient of the binned energy (conserves energy).
------------------------------------------------------------------------- */

void PPPMDispPlanar::corr_shell_bin()
{
  const double zprd = domain->prd[dim];
  const double zloc0 = domain->boxlo[dim];
  const double bcut = cutoff + sw_width;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double dz_target = (bin_dz_user > 0.0) ? bin_dz_user : MIN(0.02 * bcut, 0.5 * sw_width);
  int nbins = (int) (zprd / dz_target + 0.5);
  if (nbins < 4) nbins = 4;
  const double dz = zprd / nbins;
  int nwin = (int) (bcut / dz) + 1;
  if (nwin > nbins / 2) nwin = nbins / 2;

  // nchan density channels (1 geometric, 7 arithmetic), flattened as dens[b*nchan+m].
  // For arithmetic the binned energy mimics (1/16) sum_ij sum_m a_i[m] a_j[6-m] wE,
  // so channel m of the density pairs with the field of channel (6-m).
  const double as_shell = (nchan == 1) ? 1.0 : 1.0 / 16.0;
  auto *dens = new double[nbins * nchan];
  auto *dens_all = new double[nbins * nchan];
  for (int b = 0; b < nbins * nchan; b++) dens[b] = 0.0;

  auto *ab0 = new int[nlocal > 0 ? nlocal : 1];
  auto *afrac = new double[nlocal > 0 ? nlocal : 1];
  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zloc0) / dz;
    u -= nbins * floor(u / nbins);
    int b0 = (int) u;
    if (b0 >= nbins) b0 -= nbins;
    double frac = u - (int) u;
    ab0[i] = b0;
    afrac[i] = frac;
    int b1 = b0 + 1;
    if (b1 >= nbins) b1 -= nbins;
    if (nchan == 1) {
      const double bi = B[type[i]];
      dens[b0] += bi * (1.0 - frac);
      dens[b1] += bi * frac;
    } else {
      const double *bi = &B[7 * type[i]];
      for (int m = 0; m < 7; m++) {
        dens[b0 * 7 + m] += bi[m] * (1.0 - frac);
        dens[b1 * 7 + m] += bi[m] * frac;
      }
    }
  }
  MPI_Allreduce(dens, dens_all, nbins * nchan, MPI_DOUBLE, MPI_SUM, world);

  // energy and virial kernels on the bin offsets (force = gradient of binned energy)
  auto *KE = new double[nwin + 1];
  auto *KT = new double[nwin + 1];
  auto *KN = new double[nwin + 1];
  for (int d = 0; d <= nwin; d++) {
    double wE, wF, wT, wN;
    shell_vkernel(d * dz, wE, wF, wT, wN);
    KE[d] = wE;
    KT[d] = wT;
    KN[d] = wN;
  }

  // per-channel convolved fields phiE_m[b] = sum_d KE[d](dens_m[b+d]+dens_m[b-d])
  auto *phiE = new double[nbins * nchan];
  auto *phiT = new double[nbins * nchan];
  auto *phiN = new double[nbins * nchan];
  for (int m = 0; m < nchan; m++) {
    for (int b = 0; b < nbins; b++) {
      double sE = KE[0] * dens_all[b * nchan + m];
      double sT = KT[0] * dens_all[b * nchan + m];
      double sN = KN[0] * dens_all[b * nchan + m];
      for (int d = 1; d <= nwin; d++) {
        int bp = b + d;
        if (bp >= nbins) bp -= nbins;
        int bm = b - d;
        if (bm < 0) bm += nbins;
        double s = dens_all[bp * nchan + m] + dens_all[bm * nchan + m];
        sE += KE[d] * s;
        sT += KT[d] * s;
        sN += KN[d] * s;
      }
      phiE[b * nchan + m] = sE;
      phiT[b * nchan + m] = sT;
      phiN[b * nchan + m] = sN;
    }
  }

  // global energy = as_shell sum_b sum_m dens_m phiE_{6-m} (channel cross pairing;
  // full ordered convention, no 1/2).  For nchan==1 this is sum_b dens phiE.
  if (eflag_global || vflag_global) {
    double e = 0.0;
    for (int b = 0; b < nbins; b++)
      for (int m = 0; m < nchan; m++)
        e += dens_all[b * nchan + m] * phiE[b * nchan + (nchan - 1 - m)];
    e *= as_shell;
    corr_energy -= e;
    if (eflag_global) energy -= e;
  }
  if (vflag_global) {
    double vt = 0.0, vn = 0.0;
    for (int b = 0; b < nbins; b++)
      for (int m = 0; m < nchan; m++) {
        vt += dens_all[b * nchan + m] * phiT[b * nchan + (nchan - 1 - m)];
        vn += dens_all[b * nchan + m] * phiN[b * nchan + (nchan - 1 - m)];
      }
    vt *= as_shell;
    vn *= as_shell;
    virial[lat1] -= vt;
    virial[lat2] -= vt;
    virial[dim] -= vn;
  }

  // forces (CIC gradient of the binned energy; factor 2 from the ordered double
  // sum) and per-atom energy/virial.  Atom i channel m pairs with field channel 6-m.
  for (int i = 0; i < nlocal; i++) {
    int b0 = ab0[i];
    double frac = afrac[i];
    int b1 = b0 + 1;
    if (b1 >= nbins) b1 -= nbins;
    if (nchan == 1) {
      const double bi = B[type[i]];
      // f += -d/dz_i[-E] = +B_i d/dz_i E,  E = sum dens phiE,  dE/d dens = 2 phiE
      f[i][dim] += bi * 2.0 * (phiE[b1] - phiE[b0]) / dz;
      if (evflag_atom) peatom[i] -= bi * (phiE[b0] * (1.0 - frac) + phiE[b1] * frac);
      if (vflag_atom) {
        const double pT = phiT[b0] * (1.0 - frac) + phiT[b1] * frac;
        const double pN = phiN[b0] * (1.0 - frac) + phiN[b1] * frac;
        vatom[i][lat1] -= bi * pT;
        vatom[i][lat2] -= bi * pT;
        vatom[i][dim] -= bi * pN;
      }
    } else {
      const double *bi = &B[7 * type[i]];
      double fz = 0.0, pe = 0.0, pT = 0.0, pN = 0.0;
      for (int m = 0; m < 7; m++) {
        const int n = 6 - m;    // atom channel m pairs with field channel 6-m
        fz += bi[m] * (phiE[b1 * 7 + n] - phiE[b0 * 7 + n]);
        if (evflag_atom)
          pe += bi[m] * (phiE[b0 * 7 + n] * (1.0 - frac) + phiE[b1 * 7 + n] * frac);
        if (vflag_atom) {
          pT += bi[m] * (phiT[b0 * 7 + n] * (1.0 - frac) + phiT[b1 * 7 + n] * frac);
          pN += bi[m] * (phiN[b0 * 7 + n] * (1.0 - frac) + phiN[b1 * 7 + n] * frac);
        }
      }
      f[i][dim] += as_shell * 2.0 * fz / dz;
      if (evflag_atom) peatom[i] -= as_shell * pe;
      if (vflag_atom) {
        vatom[i][lat1] -= as_shell * pT;
        vatom[i][lat2] -= as_shell * pT;
        vatom[i][dim] -= as_shell * pN;
      }
    }
  }

  delete[] dens;
  delete[] dens_all;
  delete[] KE;
  delete[] KT;
  delete[] KN;
  delete[] phiE;
  delete[] phiT;
  delete[] phiN;
  delete[] ab0;
  delete[] afrac;
}

/* ----------------------------------------------------------------------
   IK pressure building blocks Phi(h), Psi(h)  (see ewald/disp/planar)
------------------------------------------------------------------------- */

double PPPMDispPlanar::ik_phi(double h)
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

double PPPMDispPlanar::ik_psi(double h)
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
   then assembles the profile with the same coefficients as ewald/disp/planar.
------------------------------------------------------------------------- */

void PPPMDispPlanar::compute_pressure_profile()
{
  const double unitk = 2.0 * MY_PI / zprd;
  const double rc3 = cutoff * cutoff * cutoff;
  const int K = nz / 2 - 1;    // highest resolved mode
  if (npro < 1 || K < 1) return;

  memory->destroy(pt_profile);
  memory->destroy(pn_profile);
  memory->create(pt_profile, npro, "pppm/disp/planar:pt_profile");
  memory->create(pn_profile, npro, "pppm/disp/planar:pn_profile");

  // structure factors sfac[n] = sum_j B_j exp(i n unitk z_j), n=0..K.
  // The pressure profile uses a single scalar dispersion weight per atom; under
  // arithmetic mixing (7-channel B) this is approximated by the per-type self
  // amplitude Bt = 2 sqrt(eps_t) sigma_t^3 = sqrt(C6_tt) (same single-channel
  // approximation as ewald/disp/planar's profile; the profile is an optional
  // diagnostic, off by default, and is exact only for geometric mixing).
  int ntypes = atom->ntypes;
  auto *Bt = new double[ntypes + 1];
  if (nchan == 1) {
    for (int t = 0; t <= ntypes; t++) Bt[t] = B[t];
  } else {
    // B[7t+3] = sigma^3 sqrt(eps) sqrt(20)  ->  Bt = 2 sqrt(eps) sigma^3 = 2 B[7t+3]/sqrt(20)
    Bt[0] = 0.0;
    for (int t = 1; t <= ntypes; t++) Bt[t] = 2.0 * B[7 * t + 3] / sqrt(20.0);
  }

  auto *srl = new double[K + 1];
  auto *sim = new double[K + 1];
  auto *srl_all = new double[K + 1];
  auto *sim_all = new double[K + 1];
  for (int n = 0; n <= K; n++) srl[n] = sim[n] = 0.0;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    const double bi = Bt[type[i]];
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
      bdens[g] += Bt[type[i]];
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

  delete[] Bt;
  delete[] srl;
  delete[] sim;
  delete[] srl_all;
  delete[] sim_all;
  delete[] Sre;
  delete[] Sim;
}

/* ----------------------------------------------------------------------
   standard sine/cosine integrals (series x<=2, Lentz CF x>2); see ewald/disp/planar
------------------------------------------------------------------------- */

void PPPMDispPlanar::cisi(double x, double &si, double &ci)
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

void PPPMDispPlanar::sici_chain(double x, double *Aarr, double *Barr)
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

double PPPMDispPlanar::memory_usage()
{
  // fre,fim,Gk,GTk,GNk are scalar per-mode; dens,fz_grid,ugrid,uTgrid,uNgrid are nchan-strided
  double bytes = 5.0 * nz * sizeof(double);          // fre,fim,Gk,GTk,GNk
  bytes += 5.0 * nz * nchan * sizeof(double);        // dens,fz_grid,ugrid,uTgrid,uNgrid
  bytes += (double) nmax * sizeof(double);
  bytes += (double) order * order * sizeof(double);
  if (profile_flag) bytes += 2.0 * (double) npro * sizeof(double);
  return bytes;
}
