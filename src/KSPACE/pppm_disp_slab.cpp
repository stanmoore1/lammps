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
  damp_flag = 0;
  sw_width = 0.0;
  switch_order = 3;
  corr_mode = 0;
  corr_switch = 0;
  prof_kmax_cached = 0;
  prof_kmax_nz = 0;
  prof_kmax_zprd = 0.0;
  cWgrid = nullptr;
  ncgrid = 0;
  cwdz = 0.0;
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
  memory->destroy(GTk);
  memory->destroy(GNk);
  memory->destroy(fz_grid);
  memory->destroy(ugrid);
  memory->destroy(uTgrid);
  memory->destroy(uNgrid);
  delete[] cWgrid;
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
  if (strcmp(arg[0], "damp") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify damp", error);
    // "compact"/"switch" selects the compact-switch (CSB) variant; "yes" keeps the
    // default damped (SSB) mesh.  The non-damped (SB) variant is not meshed here.
    if (strcmp(arg[1], "compact") == 0 || strcmp(arg[1], "switch") == 0)
      damp_flag = 2;
    else if (utils::logical(FLERR, arg[1], false, lmp))
      damp_flag = 0;
    else
      error->all(FLERR, "pppm/disp/slab supports only damp yes (SSB) or damp compact (CSB)");
    return 2;
  }
  if (strcmp(arg[0], "disp/switch/order") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify disp/switch/order", error);
    switch_order = utils::inumeric(FLERR, arg[1], false, lmp);
    if (switch_order < 1 || switch_order == 4 || switch_order == 6 || switch_order > 7)
      error->all(FLERR, "kspace_modify disp/switch/order must be 1, 2, 3, 5, or 7");
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

  // compact-switch (CSB) variant: the matched pair style supplies the switch width
  // Delta; "cut_lj" above is the inner rcut.  The pair evaluates the full dispersion
  // u over the shell [rcut, rcut+Delta] (exact 3-D) and corr_csb() removes the
  // reciprocal sum's plane mean-field S*u there, eliminating the lateral-correlation
  // residual in energy and pressure.

  if (damp_flag == 2) {
    int itmp2;
    double *p_dz = (double *) force->pair->extract("disp_switch_width", itmp2);
    if (p_dz == nullptr)
      error->all(FLERR,
                 "kspace_modify damp compact requires a pair style that provides the dispersion "
                 "switch width (e.g. pair_style lj/cut/dispswitch)");
    sw_width = *p_dz;
    if (sw_width <= 0.0) error->all(FLERR, "pppm/disp/slab compact switch width must be > 0");

    // tell the pair to evaluate the FULL dispersion u over the shell (exact 3-D),
    // not the (1-S)*u complement: corr_csb() removes the plane mean-field S*u there.
    int *p_full = (int *) force->pair->extract("csb_full_shell", itmp2);
    if (p_full) *p_full = 1;
  }

  // damped variant: if the matched pair supplies a dispersion switch width
  // (lj/cut/dispswitch in its default (1-S) mode), the smooth switched corr is
  // merged into the influence function (corr_switch; no real-space corr step).
  corr_switch = 0;
  if (damp_flag != 2) {
    int itmp2;
    double *p_dz = (double *) force->pair->extract("disp_switch_width", itmp2);
    if (p_dz && *p_dz > 0.0) {
      sw_width = *p_dz;
      corr_switch = 1;
      int *p_full = (int *) force->pair->extract("csb_full_shell", itmp2);
      if (p_full) *p_full = 0;
    }
  }

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
    if (damp_flag == 2)
      utils::logmesg(lmp,
                     "  compact-switch, z grid = {}, stencil order = {}, switch width Delta = "
                     "{:.6g}\n",
                     nz, order, sw_width);
    else
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
        if (damp_flag == 2) {
          // compact switch: the de-convolved potential coefficient is the CSB GU
          // at the aliased mode (no g_ewald; the switch sets the spectrum).  gu_switch
          // is roundoff-limited at high modes -- its analytic ~k^-(n+2) tail is formed
          // by a k^3 cancellation, so below ~(4pi/V) k^3 * 1e-13 the value is noise.
          // The true contribution there is negligible, so zero it; this keeps qopt
          // monotonically decreasing in the grid (otherwise the noise floor, summed as
          // D^2 k^2, makes qopt spuriously blow up at fine grids / tight targets).
          D = gu_switch(meff < 0 ? -meff : meff);
          if (fabs(D) < (4.0 * MY_PI / volume) * ak * ak * ak * 1.0e-13) D = 0.0;
        } else {
          double b = ak / (2.0 * g_ewald), b2 = b * b, b3 = b2 * b;
          double Bk = ak * ak * ak * (sqpi * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));
          D = coefD * Bk;
        }
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
  // the compact switch has no g_ewald (the switch width Delta, fixed by the pair,
  // sets the reciprocal spectrum), so skip the splitting-parameter heuristic.
  if (damp_flag != 2) {
    if (gewaldflag_6)
      g_ewald = g_ewald_6;    // kspace_modify gewald/disp
    else if (gewaldflag)
      ;    // kspace_modify gewald (g_ewald already set by the base parser)
    else
      g_ewald = sqrt(-2.0 * log(acc)) / cutoff;
    g_ewald_set = g_ewald;
  }

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
    // The qopt model (optimal influence function) under-predicts the de-convolved
    // mesh force error by ~1.8x for the compact switch (measured vs the RMS force
    // calculator); fold that into the selection so the chosen grid meets the target.
    // gu_switch noise at fine grids is clamped in compute_qopt, so qopt is monotone
    // and the search terminates without the old pathological run-up; keep a generous
    // ceiling only as a safety net.
    const double bias = (damp_flag == 2) ? 1.8 : 1.0;
    const int ngrid_max = (damp_flag == 2) ? 16384 : 65536;
    int ngrid = 16;
    while (ngrid < ngrid_max) {
      double df = bias * sqrt(compute_qopt(ngrid, order)) * pref;
      if (df < accuracy) break;
      ngrid <<= 1;
    }
    nz = ngrid;
    // merged smooth corr (corr_switch): the same grid must also resolve the corr
    // kernel (peak width ~1/g_ewald).  Measured vs the exact ewald/disp/slab corr
    // raw (bench slab, order 5): dz*g = 0.6 -> 3e-5, 0.3 -> 4.8e-6, 0.16 -> 2.4e-7
    // relative force error, i.e. ~(dz*g)^4; invert with a ~2x safety margin.
    if (corr_switch) {
      double dzg = 0.35 * pow(acc / 1.0e-5, 0.25);
      dzg = MAX(0.12, MIN(0.7, dzg));
      int nzc = 1;
      while (nzc < (int) (zprd * g_ewald / dzg)) nzc <<= 1;
      if (nz < nzc) nz = nzc;
    }
  }
  if (nz < 8) nz = 8;

  const double bias = (damp_flag == 2) ? 1.8 : 1.0;
  estimated_force_accuracy = bias * sqrt(compute_qopt(nz, order)) * pref;
  if (damp_flag == 2 && nz >= 16384 && estimated_force_accuracy > accuracy && comm->me == 0)
    error->warning(FLERR,
                   "pppm/disp/slab compact switch: grid capped at nz={}; estimated force "
                   "accuracy {:.3g} exceeds the target {:.3g}",
                   nz, estimated_force_accuracy, accuracy);
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
  if (damp_flag == 2 || corr_switch) {
    memory->create(uTgrid, nz, "pppm/disp/slab:uTgrid");
    memory->create(uNgrid, nz, "pppm/disp/slab:uNgrid");
  }

  if (rho_coeff == nullptr || order != order_allocated) {
    if (rho_coeff) memory->destroy(rho_coeff);
    memory->create(rho_coeff, order, order, "pppm/disp/slab:rho_coeff");
    order_allocated = order;
  }
  compute_rho_coeff();

  if (corr_switch) build_corr_kernels();
  influence_function();

  if (damp_flag == 2) build_shell_vkernels();

  // size the corr bin grid to the requested force accuracy (auto, unless the
  // user fixed the width with kspace_modify corr bin <dz>).  The compact-switch
  // shell correction (corr_csb_bin) sizes its own grid, so skip this calibration.
  if (damp_flag != 2 && !corr_switch && corr_mode == 1 && bin_dz_user <= 0.0) calibrate_bin();
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

  if (damp_flag == 2) {

    // compact switch (CSB): the per-mode reciprocal coefficients are the same
    // GU/GT/GN as ewald/disp/slab (built at k = |mm| * unitk by gu_switch /
    // switch_shell_virial), with the +/- m double-count factor 0.5 for m != 0 and
    // the order-p assignment de-convolution.  No g_ewald, no slab correction.

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
    return;
  }

  const double coef = -2.0 * MY_PI * sqpi / (24.0 * volume);
  // m=0 (homogeneous tail) term: W_E(0) = GU[0] = -pi^1.5 g^3 / (6 V)
  Gk[0] = -MY_PI * sqpi * g_ewald * g_ewald * g_ewald / (6.0 * volume);
  if (corr_switch) {
    // merged smooth switched corr.  The binned corr convolution is diagonal in the
    // grid's Fourier basis: E_corr = sum_k [0.5 W~2(k)/Lz] |S_k|^2 with W~2 the 1-D
    // transform of the smooth corr kernel w2(|dz|) (corr_e vanishes smoothly at
    // rcut+Delta, so W~2 decays fast and the grid resolves it).  Fold it into the
    // influence function: one spread + FFT + combined kernel + interpolation does
    // recip AND corr (energy, ik force, per-atom) with no real-space corr step.
    // Virial: the corr tangential coefficient equals its energy coefficient
    // (pt2 = w2, boundary term ~ acc^2), so GTk = Gk merged; the normal is the
    // exact strain derivative: reciprocal GN = GU + h dGU/dh =
    // 0.5 coef (4 Bk - 1.5 h^3 e^{-b^2}/b^3) per mode, corr
    // CN = 0.5 (W~2 + k dW~2/dk)/Lz (the same identity structure).
    double w2t0, kw2p0;
    corr_tilde(0.0, w2t0, kw2p0);
    const double ce0 = 0.5 * w2t0 / zprd;
    GNk[0] = Gk[0] + ce0;    // reciprocal GN(k=0) = GU(0)
    Gk[0] += ce0;
    GTk[0] = Gk[0];
  }
  for (int m = 1; m < nz; m++) {
    int mm = (m <= nz / 2) ? m : m - nz;    // signed mode index
    double k = mm * 2.0 * MY_PI / zprd;
    double ak = fabs(k);
    double b = ak / (2.0 * g_ewald), b2 = b * b, b3 = b2 * b;
    double Bk = ak * ak * ak * (sqpi * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));
    double WE = 0.5 * coef * Bk;    // GU[|mm|]/2  (full-spectrum per-mode coeff)
    double s = sin(MY_PI * mm / nz) / (MY_PI * mm / nz);
    double w2 = pow(s, 2 * order);
    if (corr_switch) {
      double w2t, kw2p;
      corr_tilde(ak, w2t, kw2p);
      const double CE = 0.5 * w2t / zprd;
      const double CN = 0.5 * (w2t + kw2p) / zprd;
      const double WN = 0.5 * coef * (4.0 * Bk - 1.5 * ak * ak * ak * exp(-b2) / b3);
      Gk[m] = (WE + CE) / w2;
      GTk[m] = Gk[m];
      GNk[m] = (WN + CN) / w2;
    } else {
      Gk[m] = WE / w2;
    }
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
   compact-switch (CSB) reciprocal-coefficient helpers (verbatim from
   ewald/disp/slab; see that file for the derivations and accuracy notes).
------------------------------------------------------------------------- */

double PPPMDispSlab::switch_S(double t)
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t;
  if (switch_order == 1) return t2 * (3.0 - 2.0 * t);                       // cubic, C1
  if (switch_order == 2) return t2 * t * (10.0 - 15.0 * t + 6.0 * t2);      // quintic, C2
  if (switch_order == 5) {
    const double t6 = t2 * t2 * t2;
    return t6 * (462.0 + t * (-1980.0 + t * (3465.0 + t * (-3080.0 + t * (1386.0 - 252.0 * t)))));
  }
  if (switch_order == 7) {
    const double t4 = t2 * t2, t8 = t4 * t4;
    return t8 * (6435.0 + t * (-40040.0 + t * (108108.0 + t * (-163800.0 +
           t * (150150.0 + t * (-83160.0 + t * (25740.0 - 3432.0 * t)))))));
  }
  const double t3 = t2 * t, t4 = t3 * t;    // n=3 septic
  return t4 * (35.0 - 84.0 * t + 70.0 * t2 - 20.0 * t3);
}

/* ---------------------------------------------------------------------- */

double PPPMDispSlab::switch_dS(double t)
{
  // dS/dt of the order-n smoothstep is (2n+1)!/(n!)^2 * (t(1-t))^n
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double tu = t * (1.0 - t);
  if (switch_order == 1) return 6.0 * tu;
  if (switch_order == 2) return 30.0 * tu * tu;
  if (switch_order == 5) {
    const double tu2 = tu * tu;
    return 2772.0 * tu2 * tu2 * tu;
  }
  if (switch_order == 7) {
    const double tu2 = tu * tu, tu3 = tu2 * tu;
    return 51480.0 * tu3 * tu3 * tu;
  }
  return 140.0 * tu * tu * tu;
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
    if (damp_flag == 2 || corr_switch) {
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
    } else {
      virial[lat1] += e;    // tangential (GT = GU)
      virial[lat2] += e;
    }
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
    // compact switch / merged smooth corr: per-atom virial fields (GTk/GNk kernels)
    if (damp_flag == 2 || corr_switch) {
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
    }
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
        if (damp_flag == 2 || corr_switch) {
          uT += w[s] * uTgrid[g];
          uN += w[s] * uNgrid[g];
        }
      }
    }
    f[i][dim] += bi * fz;

    if (evflag_atom) {
      double pe = 0.5 * bi * uu;    // per-atom reciprocal energy
      peatom[i] += pe;
      if (vflag_atom) {
        if (damp_flag == 2 || corr_switch) {
          // explicit tangential (GTk) and normal (GNk) per-atom virial fields
          vatom[i][lat1] += 0.5 * bi * uT;
          vatom[i][lat2] += 0.5 * bi * uT;
          vatom[i][dim] += 0.5 * bi * uN;
        } else {
          vatom[i][lat1] += pe;    // tangential (GT=GU)
          vatom[i][lat2] += pe;
        }
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
  // virial, per-atom energy buffer; zz set from the trace below).  For the compact
  // switch, corr_csb() instead subtracts the reciprocal sum's plane mean-field S*u
  // over the shell (energy, z-force, virial) so the pair's exact 3-D shell remains;
  // it must run every step (the z-force is removed unconditionally).

  corr_energy = 0.0;
  if (damp_flag == 2) corr_csb();
  else if (!corr_switch) corr();
  // corr_switch: the smooth switched corr is merged into the influence function
  // (energy, ik force, virial, per-atom all handled in poisson()/fieldforce())

  // normal (zz) virial.  For the damped (power-law) variant the homogeneity trace
  // gives it cheaply: sum r.f = 6 U => virial_zz = 6*E - virial_xx - virial_yy.
  // The compact switch is non-homogeneous (S varies), so its normal is the
  // explicit GNk kernel accumulated in poisson()/fieldforce() instead.

  if (damp_flag != 2 && !corr_switch) {
    if (vflag_global)
      virial[dim] = 6.0 * (e_recip_mesh + corr_energy) - virial[lat1] - virial[lat2];
    if (vflag_atom)
      for (int i = 0; i < atom->nlocal; i++)
        vatom[i][dim] = 6.0 * peatom[i] - vatom[i][lat1] - vatom[i][lat2];
  }

  if (eflag_atom)
    for (int i = 0; i < atom->nlocal; i++) eatom[i] += peatom[i];

  if (profile_flag) compute_pressure_profile();
}

/* ----------------------------------------------------------------------
   tabulate the plane (mean-field) energy, z-force and virial kernels of the
   long-range part S(r)*u(r), u=-1/r^6, over the shell [rcut, rcut+Delta] as
   functions of |dz|.  These are what the reciprocal sum injects there with a
   laterally-uniform density (each atom seen as a uniform sheet of areal
   B-density 1/area); corr_csb subtracts them so the pair's exact 3-D shell
   interaction replaces the mean field.  Per (B_i B_j), with pre = pi/area (the
   plane integral's 2 pi/area times the Ewald 1/2 folded into GU[k], see below):
     wE(dz) = -pre       int_{rlo}^{b} S(r) r^-5 dr               (energy)
     wF(dz) =  pre       S(|dz|) |dz|^-6   for |dz|>rcut, else 0   (so that the
               plane z-force on i from j is delz * B_i B_j * wF, the z-gradient
               of wE; vanishes for |dz|<rcut where the limit rlo=rcut is fixed)
     wT(dz) = -pre/2     int_{rlo}^{b} (S u)'(r) (r^2 - dz^2) dr   (tangential)
     wN(dz) = -pre dz^2  int_{rlo}^{b} (S u)'(r) dr               (normal)
   with rlo = max(|dz|,rcut), b = rcut+Delta, (S u)'(r) = -S'(r)/r^6 + 6 S(r)/r^7.
   Identical math to ewald/disp/slab (the kernels do not depend on how the
   reciprocal sum is evaluated -- they cancel the same plane mean field).
------------------------------------------------------------------------- */

void PPPMDispSlab::build_shell_vkernels()
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

void PPPMDispSlab::shell_vkernel(double adz, double &wE, double &wF, double &wT, double &wN)
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

void PPPMDispSlab::corr_csb()
{
  if (corr_mode == 1)
    corr_csb_bin();
  else
    corr_csb_raw();
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

void PPPMDispSlab::corr_csb_raw()
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

  double e_local = 0.0, vt_local = 0.0, vn_local = 0.0;
  for (int i = 0; i < nlocal; i++) {
    const double zi = x[i][dim];
    const double bi = B[type[i]];
    double e_i = 0.0, fz_i = 0.0, vt_i = 0.0, vn_i = 0.0;
    for (int jg = 0; jg < natoms_all; jg++) {
      double delz = zi - zall[jg];
      delz -= zprd * floor(delz / zprd + 0.5);    // nearest image
      const double adz = fabs(delz);
      if (adz >= bcut) continue;
      double wE, wF, wT, wN;
      shell_vkernel(adz, wE, wF, wT, wN);
      const double bij = bi * ball[jg];
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

void PPPMDispSlab::corr_csb_bin()
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

  auto *dens = new double[nbins];
  auto *dens_all = new double[nbins];
  for (int b = 0; b < nbins; b++) dens[b] = 0.0;

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
    const double bi = B[type[i]];
    dens[b0] += bi * (1.0 - frac);
    dens[b1] += bi * frac;
  }
  MPI_Allreduce(dens, dens_all, nbins, MPI_DOUBLE, MPI_SUM, world);

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

  auto *phiE = new double[nbins];
  auto *phiT = new double[nbins];
  auto *phiN = new double[nbins];
  for (int b = 0; b < nbins; b++) {
    double sE = KE[0] * dens_all[b];
    double sT = KT[0] * dens_all[b];
    double sN = KN[0] * dens_all[b];
    for (int d = 1; d <= nwin; d++) {
      int bp = b + d;
      if (bp >= nbins) bp -= nbins;
      int bm = b - d;
      if (bm < 0) bm += nbins;
      double s = dens_all[bp] + dens_all[bm];
      sE += KE[d] * s;
      sT += KT[d] * s;
      sN += KN[d] * s;
    }
    phiE[b] = sE;
    phiT[b] = sT;
    phiN[b] = sN;
  }

  if (eflag_global || vflag_global) {
    double e = 0.0;
    for (int b = 0; b < nbins; b++) e += dens_all[b] * phiE[b];
    corr_energy -= e;
    if (eflag_global) energy -= e;
  }
  if (vflag_global) {
    double vt = 0.0, vn = 0.0;
    for (int b = 0; b < nbins; b++) {
      vt += dens_all[b] * phiT[b];
      vn += dens_all[b] * phiN[b];
    }
    virial[lat1] -= vt;
    virial[lat2] -= vt;
    virial[dim] -= vn;
  }

  for (int i = 0; i < nlocal; i++) {
    int b0 = ab0[i];
    double frac = afrac[i];
    int b1 = b0 + 1;
    if (b1 >= nbins) b1 -= nbins;
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
  int *type = atom->type;
  double natoms = (double) atom->natoms;
  if (natoms < 1.0) natoms = 1.0;

  // skip if the dispersion amplitudes B are not populated yet (defensive parity
  // with ewald/disp/slab; pppm fills B in init() before setup(), so this is
  // currently always satisfied, but guard in case the call order changes)
  double bmax = 0.0;
  for (int i = 0; i < nlocal; i++) bmax = MAX(bmax, fabs(B[type[i]]));
  double bmax_all;
  MPI_Allreduce(&bmax, &bmax_all, 1, MPI_DOUBLE, MPI_MAX, world);
  if (bmax_all == 0.0) return;

  auto *fref = new double[nlocal > 0 ? nlocal : 1];
  auto *fb = new double[nlocal > 0 ? nlocal : 1];

  corr_raw_force(fref);    // exact target (once)

  // Refine the bin count by doubling, but stop as soon as refinement stops
  // paying off.  The binned correction has an intrinsic error floor (set by the
  // CIC/B-spline assignment), so a target below that floor can never be met --
  // without the knee test below the loop would run to nb_cap, and since the
  // per-step correction is O(nbins^2) (nwin grows with nbins) that makes the
  // default pathologically slow.  Back off to the coarsest grid that is still
  // within ~30% of the floor.
  const int nb_cap = (int) (zprd / (0.02 * cutoff) + 0.5);    // dz >= 0.02*cutoff
  int nb = (int) (zprd / 0.1 + 0.5);                          // start near dz = 0.1 sigma
  if (nb < 8) nb = 8;
  int chosen = nb;
  double err = 0.0, prev_err = -1.0;
  int prev_nb = nb;
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
    if (err < accuracy) break;                          // target met
    if (prev_err > 0.0 && err > 0.7 * prev_err) {       // diminishing returns: at the floor
      chosen = prev_nb;                                 // keep coarser grid (~same error)
      err = prev_err;
      break;
    }
    if (nb >= nb_cap) break;                            // safety cap
    prev_err = err;
    prev_nb = nb;
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
  // compact switch: anchor the tail at the OUTER cutoff rcut+Delta and add the
  // switch-shell integral (ported from pppm/disp/planar; sharp as Delta->0)
  const double c = (damp_flag == 2) ? cutoff + sw_width : cutoff;
  double A[8], Bc[8];
  sici_chain(ah * c, A, Bc);
  const double sii5 = A[5] / 4.0 - A[1] / (4.0 * pow(ah * c, 4));
  double phi = MY_PI / 576.0 - sii5 + A[7] - Bc[6];
  if (damp_flag == 2) phi += prof_shell(PROF_PHI, ah);
  const double ah4 = ah * ah * ah * ah;
  return (h >= 0.0 ? 1.0 : -1.0) * ah4 * phi;
}

/* ---------------------------------------------------------------------- */

double PPPMDispSlab::ik_psi(double h)
{
  if (fabs(h) < 1.0e-300) return 0.0;
  const double ah = fabs(h);
  const double c = (damp_flag == 2) ? cutoff + sw_width : cutoff;
  double A[8], Bc[8];
  sici_chain(ah * c, A, Bc);
  double psi = MY_PI / 288.0 - A[7] + Bc[6];
  if (damp_flag == 2) psi += prof_shell(PROF_T, ah);
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

void PPPMDispSlab::shell_profile_virial(int nbins, double lo, double dz, double *dens_all,
                                        double *shellT, double *shellN)
{
  const double zpd = domain->prd[dim];
  const double bcut = cutoff + sw_width;
  for (int g = 0; g < nbins; g++) shellT[g] = shellN[g] = 0.0;

  if (corr_mode != 0) {    // BIN: density-density convolution (matches corr_csb_bin)
    for (int g = 0; g < nbins; g++) {
      for (int gp = 0; gp < nbins; gp++) {
        double ddz = (gp - g) * dz;
        ddz -= zpd * floor(ddz / zpd + 0.5);
        double wE, wF, wT, wN;
        shell_vkernel(fabs(ddz), wE, wF, wT, wN);
        if (wT == 0.0 && wN == 0.0) continue;
        const double pT = dens_all[g] * dens_all[gp] * wT;
        const double pN = dens_all[g] * dens_all[gp] * wN;
        const int nspan = (int) (fabs(ddz) / dz + 0.5) + 1;
        const int step = (ddz >= 0.0) ? 1 : -1;
        const double iT = pT / nspan, iN = pN / nspan;
        for (int sp = 0; sp < nspan; sp++) {
          int b = (g + sp * step) % nbins;
          if (b < 0) b += nbins;
          shellT[b] += iT;
          shellN[b] += iN;
        }
      }
    }
    return;
  }

  // RAW: exact per-atom shell virial, spread IK along each bond (matches corr_csb_raw)
  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nprocs = comm->nprocs;
  int *rc = new int[nprocs];
  int *dp = new int[nprocs];
  MPI_Allgather(&nlocal, 1, MPI_INT, rc, 1, MPI_INT, world);
  int natoms_all = 0;
  for (int p = 0; p < nprocs; p++) {
    dp[p] = natoms_all;
    natoms_all += rc[p];
  }
  auto *zloc = new double[nlocal > 0 ? nlocal : 1];
  auto *bloc = new double[nlocal > 0 ? nlocal : 1];
  for (int i = 0; i < nlocal; i++) {
    zloc[i] = x[i][dim];
    bloc[i] = B[type[i]];
  }
  auto *zall = new double[natoms_all > 0 ? natoms_all : 1];
  auto *ball = new double[natoms_all > 0 ? natoms_all : 1];
  MPI_Allgatherv(zloc, nlocal, MPI_DOUBLE, zall, rc, dp, MPI_DOUBLE, world);
  MPI_Allgatherv(bloc, nlocal, MPI_DOUBLE, ball, rc, dp, MPI_DOUBLE, world);

  auto *sTloc = new double[nbins];
  auto *sNloc = new double[nbins];
  for (int g = 0; g < nbins; g++) sTloc[g] = sNloc[g] = 0.0;
  for (int i = 0; i < nlocal; i++) {
    double zi = x[i][dim];
    double bi = B[type[i]];
    double u = (zi - lo) / dz;
    u -= nbins * floor(u / nbins);
    int g = (int) u;
    if (g >= nbins) g -= nbins;
    for (int jg = 0; jg < natoms_all; jg++) {
      double delz = zi - zall[jg];
      delz -= zpd * floor(delz / zpd + 0.5);
      double adz = fabs(delz);
      if (adz >= bcut) continue;
      double wE, wF, wT, wN;
      shell_vkernel(adz, wE, wF, wT, wN);
      const double bij = bi * ball[jg];
      const int nspan = (int) (adz / dz + 0.5) + 1;
      const int step = (delz <= 0.0) ? 1 : -1;
      const double iT = bij * wT / nspan, iN = bij * wN / nspan;
      for (int sp = 0; sp < nspan; sp++) {
        int b = (g + sp * step) % nbins;
        if (b < 0) b += nbins;
        sTloc[b] += iT;
        sNloc[b] += iN;
      }
    }
  }
  MPI_Allreduce(sTloc, shellT, nbins, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(sNloc, shellN, nbins, MPI_DOUBLE, MPI_SUM, world);
  delete[] rc;
  delete[] dp;
  delete[] zloc;
  delete[] bloc;
  delete[] zall;
  delete[] ball;
  delete[] sTloc;
  delete[] sNloc;
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
   long-range Irving-Kirkwood pressure profiles on the caller's z grid (ported
   from pppm/disp/planar; compact-switch variant only for now).
------------------------------------------------------------------------- */

int PPPMDispSlab::pressure_profile_long(int dir, int nbins, double lo, double width,
                                        double *pN, double *pT)
{
  if (damp_flag != 2)
    error->all(FLERR,
               "compute stress/cartesian kspace with pppm/disp/slab currently requires the "
               "compact-switch variant (kspace_modify damp compact); use kspace_modify "
               "pressure/profile for the damped variant");
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

  // bin the B-weighted density (BIN-mode shell convolution source)
  auto *dens_b = new double[nbins];
  for (int g = 0; g < nbins; g++) dens_b[g] = 0.0;
  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - lo) / width;
    u -= nbins * floor(u / nbins);
    int g = (int) u;
    if (g >= nbins) g -= nbins;
    dens_b[g] += B[type[i]];
  }
  auto *dens_all = new double[nbins];
  MPI_Allreduce(dens_b, dens_all, nbins, MPI_DOUBLE, MPI_SUM, world);

  auto *shellT = new double[nbins];
  auto *shellN = new double[nbins];
  shell_profile_virial(nbins, lo, width, dens_all, shellT, shellN);

  profile_assemble(K, nbins, lo, width, Sre, Sim, GTr, GNr, shellT, shellN, pN, pT);

  delete[] GTr;
  delete[] GNr;
  delete[] srl;
  delete[] sim;
  delete[] srl_all;
  delete[] sim_all;
  delete[] Sre;
  delete[] Sim;
  delete[] dens_b;
  delete[] dens_all;
  delete[] shellT;
  delete[] shellN;
  return 1;
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
  double bytes = 8.0 * nz * sizeof(double);    // dens,fre,fim,Gk,GTk,GNk,fz_grid,ugrid
  if (damp_flag == 2) bytes += 2.0 * nz * sizeof(double);    // uTgrid, uNgrid
  bytes += (double) nmax * sizeof(double);
  bytes += (double) order * order * sizeof(double);
  if (profile_flag) bytes += 2.0 * (double) npro * sizeof(double);
  return bytes;
}
