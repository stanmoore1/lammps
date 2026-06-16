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
   Slab-based (SB) Ewald summation for 1/r^6 dispersion interactions in
   systems whose mean density varies in one (z) direction only -- e.g. planar
   liquid-vapor interfaces.  The long-range dispersion energy is evaluated as a
   1-D Fourier sum over z wavevectors h_n = 2*pi*n/Lz of the dispersion-weighted
   structure factor, with analytic x-y tail corrections.

   Two variants (kspace_modify damp no|yes):
     - non-damped (SB):  sharp truncation; reciprocal coefficients use the
       generalized sine/cosine integrals Si_m, Ci_m.
     - damped (SSB):     Gaussian (erfc) smoothing like 3-D dispersion Ewald;
       adds a real-space "slab" correction term (corr()).

   References: S. Moore, dissertation (BYU); this paper.
------------------------------------------------------------------------- */

#include "ewald_disp_slab.h"

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

/* ---------------------------------------------------------------------- */

EwaldDispSlab::EwaldDispSlab(LAMMPS *lmp) :
    KSpace(lmp), GU(nullptr), GF(nullptr), GT(nullptr), ek(nullptr), peatom(nullptr),
    sfacrl(nullptr), sfacim(nullptr), sfacrl_all(nullptr), sfacim_all(nullptr), cs(nullptr),
    sn(nullptr), B(nullptr)
{
  dispersionflag = 1;
  dim = 2;
  lat1 = 0;
  lat2 = 1;
  damp_flag = 0;
  corr_mode = 0;
  bin_dz_user = 0.0;
  sw_width = 0.0;
  contour_flag = 0;
  profile_flag = 0;
  npro = 0;
  pt_profile = pn_profile = nullptr;
  kmax = 0;
  kcount = 0;
  kmax_created = 0;
  kmax_user = 0;
  nmax = 0;
  accuracy_relative = 0.0;
}

/* ---------------------------------------------------------------------- */

EwaldDispSlab::~EwaldDispSlab()
{
  deallocate();
  memory->destroy(ek);
  memory->destroy(peatom);
  memory->destroy(cs);
  memory->destroy(sn);
  delete[] B;
  memory->destroy(pt_profile);
  memory->destroy(pn_profile);
}

/* ---------------------------------------------------------------------- */

void EwaldDispSlab::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal kspace_style {} command", force->kspace_style);
  accuracy_relative = fabs(utils::numeric(FLERR, arg[0], false, lmp));
  if (accuracy_relative > 1.0)
    error->all(FLERR, "Invalid relative accuracy {:g} for kspace_style {}", accuracy_relative,
               force->kspace_style);
}

/* ----------------------------------------------------------------------
   handle the per-style kspace_modify keywords:
     damp yes|no       -- select damped (SSB) vs non-damped (SB)
     kmax <N>          -- override the number of z wavevectors
     corr raw|bin [dz] -- damped correction: exact pairwise, or z-binned (faster)
   returns number of args consumed (0 -> base errors on unknown keyword)
------------------------------------------------------------------------- */

int EwaldDispSlab::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0], "damp") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify damp", error);
    // "compact"/"switch" selects the compact-switch (CSB) variant; otherwise yes/no
    if (strcmp(arg[1], "compact") == 0 || strcmp(arg[1], "switch") == 0)
      damp_flag = 2;
    else
      damp_flag = utils::logical(FLERR, arg[1], false, lmp);
    return 2;
  }
  if (strcmp(arg[0], "kmax") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify kmax", error);
    kmax_user = utils::inumeric(FLERR, arg[1], false, lmp);
    if (kmax_user < 2) error->all(FLERR, "kspace_modify kmax must be >= 2");
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

void EwaldDispSlab::init()
{
  if (comm->me == 0) utils::logmesg(lmp, "Slab-based dispersion Ewald (ewald/disp/slab) ...\n");

  // error checks

  triclinic_check();
  if (domain->dimension == 2) error->all(FLERR, "Cannot use ewald/disp/slab with 2d simulation");
  if (domain->triclinic) error->all(FLERR, "Cannot use ewald/disp/slab with triclinic box");
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR, "ewald/disp/slab requires periodic boundaries in all dimensions");
  if (slabflag)
    error->all(FLERR, "Cannot use slab correction (kspace_modify slab) with ewald/disp/slab");

  // SB Ewald pairs with a plain-cutoff LJ pair style (e.g. lj/cut): the pair
  // computes the full LJ to rcut and this kspace adds the r>rcut tail.  So we do
  // not require pair->dispersionflag; instead we validate via extract() below.

  if (force->pair == nullptr)
    error->all(FLERR, "KSpace style ewald/disp/slab requires a pair style");

  // extract the LJ cutoff and dispersion amplitudes B from the pair style

  int itmp;
  double *p_cutoff = (double *) force->pair->extract("cut_lj", itmp);
  if (p_cutoff == nullptr) p_cutoff = (double *) force->pair->extract("cut_LJ", itmp);
  if (p_cutoff == nullptr)
    error->all(FLERR, "Pair style is incompatible with kspace_style ewald/disp/slab");
  cutoff = *p_cutoff;
  rc2 = cutoff * cutoff;

  // compact-switch (CSB) variant: the matched pair style supplies the switch
  // width Delta (and the (1-S)*u shell complement over [rcut, rcut+Delta]).  The
  // pair's interaction cutoff is rcut+Delta; "cut_lj" above is the inner rcut.

  if (damp_flag == 2) {
    int itmp2;
    double *p_dz = (double *) force->pair->extract("disp_switch_width", itmp2);
    if (p_dz == nullptr)
      error->all(FLERR,
                 "kspace_modify damp compact requires a pair style that provides the dispersion "
                 "switch width (e.g. pair_style lj/cut/dispswitch)");
    sw_width = *p_dz;
    if (sw_width <= 0.0) error->all(FLERR, "ewald/disp/slab compact switch width must be > 0");
  }

  // accuracy in force units

  two_charge();
  if (accuracy_absolute >= 0.0)
    accuracy = accuracy_absolute;
  else
    accuracy = accuracy_relative * two_charge_force;

  // choose the splitting parameter g_ewald (damped) and the number of z
  // wavevectors kmax from the target accuracy (unless the user set them).
  // init_coeffs() first so the dispersion amplitudes B are available.

  init_coeffs();
  estimate_params();

  setup();

  if (comm->me == 0) {
    const char *variant =
        (damp_flag == 2) ? "compact-switch" : (damp_flag == 1) ? "damped" : "non-damped";
    utils::logmesg(lmp, "  {} slab-based dispersion Ewald, {} z wavevectors\n", variant, kmax);
    if (damp_flag == 1) utils::logmesg(lmp, "  g_ewald = {:.6g}\n", g_ewald);
    if (damp_flag == 2) utils::logmesg(lmp, "  switch width Delta = {:.6g}\n", sw_width);
    utils::logmesg(lmp, "  estimated absolute RMS force accuracy = {:.6g}\n",
                   estimated_force_accuracy);
    utils::logmesg(lmp, "  estimated relative force accuracy = {:.6g}\n",
                   estimated_force_accuracy / two_charge_force);
  }
}

/* ----------------------------------------------------------------------
   force coefficient GF for a single z mode k (k>=1); requires volume, unitk,
   cutoff (and g_ewald for the damped variant) to be set
------------------------------------------------------------------------- */

double EwaldDispSlab::gf_of_k(int k)
{
  const double kcell = k * unitk;
  const double kcell3 = kcell * kcell * kcell;
  if (damp_flag == 2) {
    // compact switch: force is the exact z-gradient of the energy term, GF=2k*GU
    return 2.0 * kcell * gu_switch(k);
  } else if (!damp_flag) {
    double A[8], Bc[8];
    sici_chain(kcell * cutoff, A, Bc);
    const double GUk = (-4.0 * MY_PI * kcell3 / volume) * (MY_PI / 48.0 - A[5]);
    return 2.0 * kcell * GUk;
  } else {
    const double b = kcell / (2.0 * g_ewald);
    const double b2 = b * b, b3 = b2 * b;
    const double coef = -2.0 * MY_PI * sqrt(MY_PI) / (24.0 * volume);
    const double Bk = kcell3 * (sqrt(MY_PI) * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));
    return coef * 2.0 * kcell * Bk;
  }
}

/* ----------------------------------------------------------------------
   estimate g_ewald (damped) and kmax from the target force accuracy.

   RMS per-atom force truncation error (random-phase model, validated against
   measured forces):  rms(kmax) = sqrt(0.5 * b2^2 / N * sum_{k>kmax} GF[k]^2),
   with b2 = sum_i B_i^2.  Pick the smallest kmax with rms < accuracy/2 (the
   model slightly under-predicts for correlated/interfacial systems).

   g_ewald is set so the neglected short-range tail (beyond rcut, since the pair
   style is a hard cutoff) is below the target: (g*rcut)^2 = -2*ln(accuracy).
------------------------------------------------------------------------- */

void EwaldDispSlab::estimate_params()
{
  lat1 = (dim + 1) % 3;
  lat2 = (dim + 2) % 3;
  volume = domain->prd[0] * domain->prd[1] * domain->prd[2];
  unitk = 2.0 * MY_PI / domain->prd[dim];

  // g_ewald for the damped variant (Gaussian short-range tail criterion)

  double acc = accuracy / two_charge_force;    // relative target for the log
  if (acc <= 0.0 || acc >= 1.0) acc = 1.0e-4;
  if (damp_flag == 1 && !gewaldflag) g_ewald = sqrt(-2.0 * log(acc)) / cutoff;

  // dispersion sum b2 = sum_i B_i^2 (full system).  NOTE kspace->init() runs
  // before pair->init(), so lj4 (hence B) is not populated yet -- compute the
  // per-type amplitude B_t = 2 sqrt(eps_tt) sigma_tt^3 from epsilon/sigma, which
  // pair_coeff has already set.

  int *type = atom->type;
  int nlocal = atom->nlocal;
  int ntypes = atom->ntypes;
  int dim;
  auto **eps = (double **) force->pair->extract("epsilon", dim);
  auto **sig = (double **) force->pair->extract("sigma", dim);
  auto *Bt = new double[ntypes + 1];
  for (int t = 1; t <= ntypes; t++)
    Bt[t] = (eps && sig) ? 2.0 * sqrt(eps[t][t]) * sig[t][t] * sig[t][t] * sig[t][t] : B[t];
  double b2_local = 0.0;
  for (int i = 0; i < nlocal; i++) b2_local += Bt[type[i]] * Bt[type[i]];
  delete[] Bt;
  double b2;
  MPI_Allreduce(&b2_local, &b2, 1, MPI_DOUBLE, MPI_SUM, world);
  double natoms = (double) atom->natoms;
  if (natoms < 1.0) natoms = 1.0;

  if (kmax_user > 0) {
    kmax = kmax_user;
    return;
  }

  // cumulative tail of GF[k]^2 from a large cap; pick kmax for target accuracy

  const int kbig = 8192;
  const double prefac = 0.5 * b2 * b2 / natoms;
  // require predicted rms < accuracy/4 (safety margin: the random-phase model
  // under-predicts for correlated/interfacial systems by a factor of a few)
  const double target = accuracy * accuracy / 16.0;

  // sum GF^2 from the top down so tail(kmax) = sum_{k>kmax}
  auto *gf2 = new double[kbig + 1];
  if (damp_flag == 2) {
    // the compact-switch GF^2 decays fast; compute upward and stop once it has
    // been negligible for a run of modes (avoids costly high-k quadrature).
    double gf2max = 0.0;
    int nsmall = 0, kstop = kbig;
    for (int k = 1; k <= kbig; k++) {
      double g = gf_of_k(k);
      gf2[k] = g * g;
      gf2max = MAX(gf2max, gf2[k]);
      if (gf2[k] < 1.0e-16 * gf2max) {
        if (++nsmall >= 32) { kstop = k; break; }
      } else
        nsmall = 0;
    }
    for (int k = kstop + 1; k <= kbig; k++) gf2[k] = 0.0;
  } else {
    for (int k = 1; k <= kbig; k++) {
      double g = gf_of_k(k);
      gf2[k] = g * g;
    }
  }
  double tail = 0.0;
  int chosen = kbig;
  for (int kmx = kbig - 1; kmx >= 4; kmx--) {
    tail += gf2[kmx + 1];
    if (prefac * tail >= target) {    // first kmax (scanning down) that fails -> kmax+1 is enough
      chosen = kmx + 1;
      break;
    }
    chosen = kmx;
  }

  kmax = MAX(8, MIN(chosen, kbig));

  // predicted RMS per-atom force error at the chosen kmax
  double tk = 0.0;
  for (int k = kmax + 1; k <= kbig; k++) tk += gf2[k];
  estimated_force_accuracy = sqrt(prefac * tk);
  delete[] gf2;

  // the non-damped reciprocal converges only algebraically (~kmax^-0.7) because
  // of Gibbs ringing; warn if the target is not reachable within the cap
  if (chosen >= kbig && damp_flag == 0 && comm->me == 0)
    error->warning(FLERR,
                   "ewald/disp/slab: target accuracy needs kmax >= {} (capped); the non-damped "
                   "variant converges slowly -- use kspace_modify damp yes (or damp compact) for "
                   "tighter accuracy",
                   kbig);
}

/* ----------------------------------------------------------------------
   adjust coefficients, called initially and whenever the volume changes
------------------------------------------------------------------------- */

void EwaldDispSlab::setup()
{
  volume = domain->prd[0] * domain->prd[1] * domain->prd[2];
  unitk = 2.0 * MY_PI / domain->prd[dim];

  deallocate();
  allocate();

  if (atom->nmax > nmax) {
    memory->destroy(ek);
    memory->destroy(peatom);
    memory->destroy(cs);
    memory->destroy(sn);
    nmax = atom->nmax;
    memory->create(ek, nmax, "ewald/disp/slab:ek");
    memory->create(peatom, nmax, "ewald/disp/slab:peatom");
    memory->create(cs, kmax, nmax, "ewald/disp/slab:cs");
    memory->create(sn, kmax, nmax, "ewald/disp/slab:sn");
    kmax_created = kmax;
  } else if (kmax != kmax_created) {
    memory->destroy(cs);
    memory->destroy(sn);
    memory->create(cs, kmax, nmax, "ewald/disp/slab:cs");
    memory->create(sn, kmax, nmax, "ewald/disp/slab:sn");
    kmax_created = kmax;
  }

  init_coeffs();
  coeffs();
}

/* ----------------------------------------------------------------------
   extract per-type dispersion amplitude B[i] = sqrt(|lj4[i][i]|) = 2*sqrt(eps)*sigma^3
------------------------------------------------------------------------- */

void EwaldDispSlab::init_coeffs()
{
  int tmp;
  int n = atom->ntypes;
  auto **b = (double **) force->pair->extract("B", tmp);
  if (b == nullptr)
    error->all(FLERR, "Pair style does not provide dispersion coefficient B for ewald/disp/slab");
  delete[] B;
  B = new double[n + 1];
  B[0] = 0.0;
  for (int i = 1; i <= n; ++i) B[i] = sqrt(fabs(b[i][i]));
}

/* ----------------------------------------------------------------------
   pre-compute the reciprocal-space coefficients for each z wavevector
------------------------------------------------------------------------- */

void EwaldDispSlab::coeffs()
{
  int k;
  double kcell, kcell3, kcell4, kcutoff;
  double A[8], Bc[8];

  kcount = kmax;

  if (damp_flag == 2) {

    // compact switch (CSB): smoothed truncation over [rcut, rcut+Delta].  The
    // long-range part S(r)*u(r) vanishes inside rcut, so no real-space (slab)
    // correction is needed; the kernel is smooth at rcut, so the reciprocal sum
    // converges fast (no Gibbs ringing).  GU/GT = analytic tail evaluated at
    // rcut+Delta plus a numerically integrated transition over the shell.

    GU[0] = gu0_switch();
    GF[0] = 0.0;
    GT[0] = 2.0 * GU[0];    // tail tangential constant is 2x the energy constant

    const double c = cutoff + sw_width;
    for (k = 1; k < kcount; k++) {
      kcell = k * unitk;
      kcell3 = kcell * kcell * kcell;
      sici_chain(kcell * c, A, Bc);    // tail evaluated at rcut+Delta
      double t5, t7, t6;
      switch_transitions(kcell, t5, t7, t6);

      // energy: GU = [non-damped tail at c] + [-(4pi/V)(1/k) * t5]
      GU[k] = (-4.0 * MY_PI * kcell3 / volume) * (MY_PI / 48.0 - A[5]) -
          (4.0 * MY_PI / volume) * t5 / kcell;
      GF[k] = 2.0 * kcell * GU[k];    // exact z-gradient of the energy term
      // tangential pressure: [non-damped tail at c] + transition (Si7/Ci6 parts)
      GT[k] = (-24.0 * MY_PI * kcell3 / volume) * (MY_PI / 288.0 - A[7] + Bc[6]) -
          (24.0 * MY_PI / volume) * (t7 / kcell3 - t6 / (kcell * kcell));
    }

  } else if (!damp_flag) {

    // non-damped (SB): sharp truncation, Si_m/Ci_m coefficients

    double cutoff3 = cutoff * cutoff * cutoff;
    GU[0] = -2.0 * MY_PI / 3.0 / cutoff3 / volume;
    GF[0] = 0.0;
    GT[0] = -4.0 * MY_PI / 3.0 / cutoff3 / volume;

    for (k = 1; k < kcount; k++) {
      kcell = k * unitk;
      kcell3 = kcell * kcell * kcell;
      kcell4 = kcell * kcell3;
      kcutoff = kcell * cutoff;
      sici_chain(kcutoff, A, Bc);
      const double si5 = A[5], si7 = A[7], ci6 = Bc[6];

      GU[k] = (-4.0 * MY_PI * kcell3 / volume) * (MY_PI / 48.0 - si5);
      // Force coefficient = exact z-gradient of the energy term GU[k]*|S_k|^2, i.e.
      // GF[k] = 2*kcell*GU[k].  This (not the paper's separately-derived F_n with
      // Si_7/Ci_6, which is flagged "fix this") makes the force conserve energy;
      // verified by finite difference.  The damped branch already uses this form.
      GF[k] = 2.0 * kcell * GU[k];
      GT[k] = (-24.0 * MY_PI * kcell3) * (MY_PI / 288.0 - si7 + ci6) / volume;
    }

  } else {

    // damped (SSB): Gaussian smoothing, erfc coefficients

    double g3 = g_ewald * g_ewald * g_ewald;
    GU[0] = -MY_PI * sqrt(MY_PI) * g3 / (6.0 * volume);
    GF[0] = 0.0;
    GT[0] = GU[0];

    for (k = 1; k < kcount; k++) {
      kcell = k * unitk;
      kcell3 = kcell * kcell * kcell;
      double b = kcell / (2.0 * g_ewald);
      double b2 = b * b;
      double b3 = b2 * b;
      double coef = -2.0 * MY_PI * sqrt(MY_PI) / (24.0 * volume);
      double Bk = kcell3 * (sqrt(MY_PI) * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));

      GU[k] = coef * Bk;
      GF[k] = coef * 2.0 * kcell * Bk;
      GT[k] = GU[k];
    }
  }
}

/* ----------------------------------------------------------------------
   compact-switch (CSB) smoothstep S(t), C3 (septic) on t in [0,1]:
   S(0)=0, S(1)=1, S=S'=S''=S'''=0 at both ends.  In r: S=0 for r<=rcut,
   S=1 for r>=rcut+Delta.  The long-range part fed to the reciprocal sum is
   S(r)*u(r); it vanishes inside rcut (so no slab correction) and is smooth at
   rcut (so the z-Fourier coefficients decay fast -- no Gibbs ringing).
------------------------------------------------------------------------- */

double EwaldDispSlab::switch_S(double t)
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t, t3 = t2 * t, t4 = t3 * t;
  return t4 * (35.0 - 84.0 * t + 70.0 * t2 - 20.0 * t3);
}

/* ----------------------------------------------------------------------
   transition integrals over the switch shell [rcut, rcut+Delta]:
     t5 = int S(r) r^-5 sin(h r) dr   (energy)
     t7 = int S(r) r^-7 sin(h r) dr   (tangential, Si7 part)
     t6 = int S(r) r^-6 cos(h r) dr   (tangential, Ci6 part)
   composite Simpson with the panel count scaled to the oscillation count
   (h*Delta) so accuracy is k-independent.  Built once at setup, so cheap.
------------------------------------------------------------------------- */

void EwaldDispSlab::switch_transitions(double h, double &t5, double &t7, double &t6)
{
  const double a = cutoff, b = cutoff + sw_width, dz = sw_width;
  int n = (int) (24.0 * h * dz / (2.0 * MY_PI)) + 1;
  n = MAX(64, n);
  n = MIN(n, 200000);
  if (n % 2) n++;    // even count for Simpson
  const double dr = (b - a) / n;
  double s5 = 0.0, s7 = 0.0, s6 = 0.0;
  for (int i = 0; i <= n; i++) {
    const double r = a + i * dr;
    const double S = switch_S((r - a) / dz);
    const double sr = sin(h * r), cr = cos(h * r);
    const double r2 = r * r, r4 = r2 * r2;
    const double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    s5 += w * S * sr / (r4 * r);          // r^-5 sin
    s7 += w * S * sr / (r4 * r2 * r);     // r^-7 sin
    s6 += w * S * cr / (r4 * r2);         // r^-6 cos
  }
  const double f = dr / 3.0;
  t5 = f * s5;
  t7 = f * s7;
  t6 = f * s6;
}

/* ----------------------------------------------------------------------
   compact-switch energy coefficient GU[k] (k>=1): non-damped tail evaluated at
   rcut+Delta plus the numerically integrated shell transition.
------------------------------------------------------------------------- */

double EwaldDispSlab::gu_switch(int k)
{
  const double kcell = k * unitk;
  const double kcell3 = kcell * kcell * kcell;
  const double c = cutoff + sw_width;
  double A[8], Bc[8];
  sici_chain(kcell * c, A, Bc);
  double t5, t7, t6;
  switch_transitions(kcell, t5, t7, t6);
  return (-4.0 * MY_PI * kcell3 / volume) * (MY_PI / 48.0 - A[5]) -
      (4.0 * MY_PI / volume) * t5 / kcell;
}

/* ----------------------------------------------------------------------
   compact-switch k=0 energy coefficient:
     GU[0] = -(2pi/V) [ int_rcut^{rcut+Delta} S(r) r^-4 dr + 1/(3 (rcut+Delta)^3) ]
   (-> the non-damped -2pi/(3 rcut^3 V) as Delta->0).
------------------------------------------------------------------------- */

double EwaldDispSlab::gu0_switch()
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
   1-D dispersion-weighted structure factors S(h_n) = sum_j B_j exp(-i h_n z_j)
------------------------------------------------------------------------- */

void EwaldDispSlab::eik_dot_r()
{
  int i, k;
  double **x = atom->x;
  int nlocal = atom->nlocal;
  int *type = atom->type;

  memset(sfacrl, 0, kcount * sizeof(double));
  memset(sfacim, 0, kcount * sizeof(double));

  for (i = 0; i < nlocal; i++) {
    const double bi = B[type[i]];

    cs[0][i] = 1.0;
    sn[0][i] = 0.0;
    sfacrl[0] += bi;

    if (kcount > 1) {
      cs[1][i] = cos(unitk * x[i][dim]);
      sn[1][i] = sin(unitk * x[i][dim]);
      sfacrl[1] += bi * cs[1][i];
      sfacim[1] += bi * sn[1][i];
    }

    for (k = 2; k < kcount; k++) {
      cs[k][i] = cs[k - 1][i] * cs[1][i] - sn[k - 1][i] * sn[1][i];
      sn[k][i] = sn[k - 1][i] * cs[1][i] + cs[k - 1][i] * sn[1][i];
      sfacrl[k] += bi * cs[k][i];
      sfacim[k] += bi * sn[k][i];
    }
  }
}

/* ----------------------------------------------------------------------
   compute the slab-based dispersion long-range force, energy, virial
------------------------------------------------------------------------- */

void EwaldDispSlab::compute(int eflag, int vflag)
{
  int i, k;

  ev_init(eflag, vflag);

  // grow per-atom arrays if needed

  if (atom->nmax > nmax) {
    memory->destroy(ek);
    memory->destroy(peatom);
    memory->destroy(cs);
    memory->destroy(sn);
    nmax = atom->nmax;
    memory->create(ek, nmax, "ewald/disp/slab:ek");
    memory->create(peatom, nmax, "ewald/disp/slab:peatom");
    memory->create(cs, kmax, nmax, "ewald/disp/slab:cs");
    memory->create(sn, kmax, nmax, "ewald/disp/slab:sn");
    kmax_created = kmax;
  }

  // partial structure factors per proc, then global total

  eik_dot_r();
  MPI_Allreduce(sfacrl, sfacrl_all, kcount, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(sfacim, sfacim_all, kcount, MPI_DOUBLE, MPI_SUM, world);

  double **f = atom->f;
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double exprl, expim, partial, partial_peratom;

  for (i = 0; i < nlocal; i++) ek[i] = 0.0;
  if (evflag_atom)
    for (i = 0; i < nlocal; i++) peatom[i] = 0.0;

  for (k = 0; k < kcount; k++) {
    const double srl = sfacrl_all[k], sim = sfacim_all[k];
    for (i = 0; i < nlocal; i++) {
      exprl = cs[k][i];
      expim = sn[k][i];
      partial = expim * srl - exprl * sim;
      ek[i] += partial * GF[k];

      if (evflag_atom) {
        partial_peratom = exprl * srl + expim * sim;
        // accumulate per-atom energy in a buffer (needed for the zz virial trace
        // even when only the per-atom virial, not energy, is requested)
        peatom[i] += GU[k] * partial_peratom;
        if (vflag_atom) {
          // tangential from GT; normal (dim) set from the virial trace
          vatom[i][lat1] += GT[k] * partial_peratom;
          vatom[i][lat2] += GT[k] * partial_peratom;
        }
      }
    }
  }

  // reciprocal z-force on each atom (scaled by its own B)

  for (i = 0; i < nlocal; i++) f[i][dim] += B[type[i]] * ek[i];

  // reciprocal energy (full system value, identical on every proc); always
  // evaluated when the virial is needed (the zz trace uses it)

  double e_recip = 0.0;
  if (eflag_global || vflag_global) {
    for (k = 0; k < kcount; k++)
      e_recip += GU[k] * (sfacrl_all[k] * sfacrl_all[k] + sfacim_all[k] * sfacim_all[k]);
  }
  if (eflag_global) energy += e_recip;

  // global tangential virial (xx=yy from GT); zz set from the trace after corr

  if (vflag_global) {
    for (k = 0; k < kcount; k++) {
      double uk = sfacrl_all[k] * sfacrl_all[k] + sfacim_all[k] * sfacim_all[k];
      virial[lat1] += uk * GT[k];
      virial[lat2] += uk * GT[k];
    }
  }

  // scale per-atom energy buffer / virial by each atom's B

  if (evflag_atom)
    for (i = 0; i < nlocal; i++) peatom[i] *= B[type[i]];
  if (vflag_atom)
    for (i = 0; i < nlocal; i++) {
      vatom[i][lat1] *= B[type[i]];
      vatom[i][lat2] *= B[type[i]];
    }

  // damped variant: real-space "slab" correction (adds to energy, corr_energy,
  // tangential virial, and the per-atom energy buffer peatom; the zz virial is
  // set from the trace below)

  corr_energy = 0.0;
  if (damp_flag == 1) corr();

  // normal (zz) virial from the exact 1/r^6 virial trace:  sum r.f = 6 U, so
  // virial_zz = 6*E_kspace - virial_xx - virial_yy.  This is the total pressure
  // (contour-independent) and, per-atom, the IK-contour local normal pressure.

  if (vflag_global) virial[dim] = 6.0 * (e_recip + corr_energy) - virial[lat1] - virial[lat2];
  if (vflag_atom)
    for (i = 0; i < nlocal; i++) vatom[i][dim] = 6.0 * peatom[i] - vatom[i][lat1] - vatom[i][lat2];

  // report per-atom energy (from the buffer) when requested
  if (eflag_atom)
    for (i = 0; i < nlocal; i++) eatom[i] += peatom[i];

  // long-range pressure profiles P_T(z), P_N(z) (Harasima or Irving-Kirkwood)
  if (profile_flag) compute_pressure_profile();
}

/* ----------------------------------------------------------------------
   IK tangential building block Phi(h) = sgn(h)|h|^4 [pi/576 - Sii5 + Si7 - Ci6]
   (the IK normal uses Psi(h) = sgn(h)|h|^4 [pi/288 - Si7 + Ci6]).
------------------------------------------------------------------------- */

double EwaldDispSlab::ik_phi(double h)
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

double EwaldDispSlab::ik_psi(double h)
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
   long-range pressure profiles P_T(z), P_N(z) on an npro-point z grid.
   Harasima (contour_flag=0):  P(z) = rho_B(z) [ c0 + sum_n Re(C_n S_n e^{i h_n z}) ]
       with C_n = T_n (tangential) or N_n (normal), c0 = -16 pi rho_avg/(3 rc^3).
   Irving-Kirkwood (contour_flag=1):  P(z) = sum_{n,m} S_n S_m C_{n,m} e^{i(h_n+h_m)z},
       C^T_{n,m} = -24 pi/(h_n+h_m)[Phi(h_m)+Phi(h_n)],
       C^N_{n,m} = -48 pi/(h_n+h_m)[Psi(h_m)+Psi(h_n)],
       with the n=-m (T_n/2, N_n/2) and (0,0) (-16pi/3rc^3) special cases.
   S_n are the number-density Fourier coefficients = (1/V) sum_j B_j e^{-i h_n z_j}.
------------------------------------------------------------------------- */

void EwaldDispSlab::compute_pressure_profile()
{
  const double zprd = domain->prd[dim], zlo = domain->boxlo[dim];
  const double area = domain->prd[lat1] * domain->prd[lat2];
  const double rc3 = cutoff * cutoff * cutoff;
  const int K = kcount - 1;    // highest mode index

  if (npro < 1) return;
  memory->destroy(pt_profile);
  memory->destroy(pn_profile);
  memory->create(pt_profile, npro, "ewald/disp/slab:pt_profile");
  memory->create(pn_profile, npro, "ewald/disp/slab:pn_profile");

  // number-density Fourier coefficients S_n = (1/V)(sfacrl - i sfacim) for n>=0
  // (S_{-n} = conj(S_n)); store Sre[n], Sim[n] for n=0..K
  auto *Sre = new double[K + 1];
  auto *Sim = new double[K + 1];
  for (int n = 0; n <= K; n++) {
    Sre[n] = sfacrl_all[n] / volume;
    Sim[n] = -sfacim_all[n] / volume;
  }

  const double c0 = -4.0 * MY_PI * Sre[0] / (3.0 * rc3);    // -16 pi rho_avg/(3 rc^3)

  if (contour_flag == 0) {

    // Harasima: needs the B-weighted density profile rho_B(z) (bin B onto the grid)
    auto *dens = new double[npro];
    for (int g = 0; g < npro; g++) dens[g] = 0.0;
    int *type = atom->type;
    double **x = atom->x;
    for (int i = 0; i < atom->nlocal; i++) {
      double u = (x[i][dim] - zlo) / zprd * npro;
      u -= npro * floor(u / npro);
      int g = (int) u;
      if (g >= npro) g -= npro;
      dens[g] += B[type[i]];
    }
    auto *dens_all = new double[npro];
    MPI_Allreduce(dens, dens_all, npro, MPI_DOUBLE, MPI_SUM, world);
    const double dz = zprd / npro;
    for (int g = 0; g < npro; g++) {
      double z = zlo + (g + 0.5) * dz;
      double gt = c0, gn = c0;    // field multiplying rho(z)
      for (int n = 1; n <= K; n++) {
        double hn = n * unitk, x = hn * cutoff;
        double AA[8], BB[8];
        sici_chain(x, AA, BB);
        double Tn = -24.0 * MY_PI * hn * hn * hn * (MY_PI / 288.0 - AA[7] + BB[6]);
        double Nn =
            -24.0 * MY_PI * hn * hn * hn * (MY_PI / 72.0 - AA[5] + 2.0 * AA[7] - 2.0 * BB[6]);
        double cz = cos(hn * z), sz = sin(hn * z);
        // Re(C_n S_n e^{i h z}) = C_n (Sre cz - Sim sz)
        gt += Tn * (Sre[n] * cz - Sim[n] * sz);
        gn += Nn * (Sre[n] * cz - Sim[n] * sz);
      }
      double rhoz = dens_all[g] / (area * dz);    // areal volume density
      pt_profile[g] = rhoz * gt;
      pn_profile[g] = rhoz * gn;
    }
    delete[] dens;
    delete[] dens_all;

  } else {

    // Irving-Kirkwood: total-mode amplitudes A^T_p, A^N_p (p = n+m), then grid sum
    int P = 2 * K;
    auto *ATre = new double[P + 1];
    auto *ATim = new double[P + 1];
    auto *ANre = new double[P + 1];
    auto *ANim = new double[P + 1];
    for (int p = 0; p <= P; p++) ATre[p] = ATim[p] = ANre[p] = ANim[p] = 0.0;
    // precompute Phi, Psi for h = n*unitk, n in [-K,K]; Tn2/Nn2 for n=-m case
    auto Sn = [&](int n, double &re, double &im) {
      int an = n < 0 ? -n : n;
      re = Sre[an];
      im = (n < 0) ? -Sim[an] : Sim[an];
    };
    for (int n = -K; n <= K; n++) {
      double hn = n * unitk;
      for (int m = -K; m <= K; m++) {
        int p = n + m;
        if (p < 0) continue;    // use Hermitian symmetry; keep p>=0
        double hm = m * unitk, H = hn + hm;
        double snr, sni, smr, smi;
        Sn(n, snr, sni);
        Sn(m, smr, smi);
        // S_n * S_m (complex)
        double sre = snr * smr - sni * smi, sim = snr * smi + sni * smr;
        double CT, CN;
        if (n == 0 && m == 0) {
          CT = CN = -4.0 * MY_PI / (3.0 * rc3);
        } else if (fabs(H) < 1.0e-300) {    // n = -m
          double ah = fabs(hn), x = ah * cutoff, AA[8], BB[8];
          sici_chain(x, AA, BB);
          CT = -12.0 * MY_PI * ah * ah * ah * (MY_PI / 288.0 - AA[7] + BB[6]);    // T_n/2
          CN = -24.0 * MY_PI * ah * ah * ah * (MY_PI / 72.0 - AA[5] + 2.0 * AA[7] - 2.0 * BB[6]) /
              2.0;    // N_n/2
        } else {
          CT = -6.0 * MY_PI / H * (ik_phi(hm) + ik_phi(hn));
          CN = -12.0 * MY_PI / H * (ik_psi(hm) + ik_psi(hn));
        }
        ATre[p] += CT * sre;
        ATim[p] += CT * sim;
        ANre[p] += CN * sre;
        ANim[p] += CN * sim;
      }
    }
    const double dz = zprd / npro;
    for (int g = 0; g < npro; g++) {
      double z = zlo + (g + 0.5) * dz;
      double pt = ATre[0], pn = ANre[0];    // p=0 term (real)
      for (int p = 1; p <= P; p++) {
        double cz = cos(p * unitk * z), sz = sin(p * unitk * z);
        pt += 2.0 * (ATre[p] * cz - ATim[p] * sz);    // Hermitian: +c.c.
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
  delete[] Sre;
  delete[] Sim;
}

/* ----------------------------------------------------------------------
   damped slab-correction kernels at squared z-separation x2 = (z_i-z_j)^2.
   w2  = energy kernel; f2 = -2 d(w2)/d(x2) (energy-conserving z-force);
   pt2 = tangential-pressure kernel.  All are per unit area (1/A).
   The kernel depends ONLY on the z-separation (it is the x-y slab integral of
   the smooth/long-range part of the dispersion interaction for r < rcut).
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_kernels(double x2, double &w2, double &f2, double &pt2)
{
  const double g2 = g_ewald * g_ewald;
  const double g4 = g2 * g2, g6 = g4 * g2, g8 = g4 * g4, g10 = g8 * g2, g12 = g10 * g2;
  const double rc4 = rc2 * rc2, rc6 = rc4 * rc2;
  const double area = domain->prd[(dim + 1) % 3] * domain->prd[(dim + 2) % 3];

  if (x2 < 1.0e-3) {    // Taylor branch (avoids 1/x^n cancellation near x=0)
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
   damped slab correction dispatcher: exact pairwise or z-binned
------------------------------------------------------------------------- */

void EwaldDispSlab::corr()
{
  if (corr_mode == 1)
    corr_bin();
  else
    corr_raw();
}

/* ----------------------------------------------------------------------
   exact pairwise slab correction.
   The kernel acts between every pair with |z_i - z_j| < rcut regardless of x-y
   separation (a slab-slab interaction).  A 3-D neighbor list would miss most
   such pairs, so every proc gathers the global (z, B) list and each local atom
   sums over all global atoms in its z-window (full sum, no Newton across procs).
   Force is z-only and applied locally; energy/virial are reduced to full-system
   values (KSpace convention).  Equivalent to 0.5*sum_{i,j} B_i B_j w2(z_i-z_j)
   over all pairs including i=j (the i=j term is the self contribution).
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_raw()
{
  const double zprd = domain->prd[dim];
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // gather global (z, B) lists across all procs

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

  // self term = 0.5 * kernel(0)  (the i=j contribution)

  double w0, f0, pt0;
  corr_kernels(0.0, w0, f0, pt0);
  const double w2_self = 0.5 * w0, pt2_self = 0.5 * pt0;

  double e_local = 0.0;
  double v_local[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  double bsqsum_local = 0.0;
  for (int i = 0; i < nlocal; i++) bsqsum_local += B[type[i]] * B[type[i]];
  e_local += bsqsum_local * w2_self;
  v_local[lat1] += bsqsum_local * pt2_self;
  v_local[lat2] += bsqsum_local * pt2_self;
  if (evflag_atom)
    for (int i = 0; i < nlocal; i++) peatom[i] += B[type[i]] * B[type[i]] * w2_self;
  if (vflag_atom)
    for (int i = 0; i < nlocal; i++) {
      vatom[i][lat1] += B[type[i]] * B[type[i]] * pt2_self;
      vatom[i][lat2] += B[type[i]] * B[type[i]] * pt2_self;
    }

  // pair contributions: local i vs all global j in the z-window (full sum)

  for (int i = 0; i < nlocal; i++) {
    const double zi = x[i][dim];
    const double bi = B[type[i]];
    const int iglob = myoff + i;
    double fz_i = 0.0;

    for (int jg = 0; jg < natoms_all; jg++) {
      if (jg == iglob) continue;
      double delz = zi - zall[jg];
      delz -= zprd * trunc(2.0 * delz / zprd);    // nearest image in z
      double x2 = delz * delz;
      if (x2 >= rc2) continue;

      double w2, f2, pt2;
      corr_kernels(x2, w2, f2, pt2);
      const double bij = bi * ball[jg];

      // each unordered pair is summed from both ends -> 0.5 weight on energy/virial.
      // normal (dim) virial is set from the trace in compute(), not here.
      e_local += 0.5 * bij * w2;
      fz_i += delz * bij * f2;
      v_local[lat1] += 0.5 * bij * pt2;    // tangential lat1
      v_local[lat2] += 0.5 * bij * pt2;    // tangential lat2

      if (evflag_atom) peatom[i] += 0.5 * bij * w2;
      if (vflag_atom) {
        vatom[i][lat1] += 0.5 * bij * pt2;
        vatom[i][lat2] += 0.5 * bij * pt2;
      }
    }

    f[i][dim] += fz_i;
  }

  // corr energy reduced to a full-system value (always, for the virial trace)
  double e_all;
  MPI_Allreduce(&e_local, &e_all, 1, MPI_DOUBLE, MPI_SUM, world);
  corr_energy = e_all;
  if (eflag_global) energy += e_all;
  if (vflag_global) {
    double v_all[6];
    MPI_Allreduce(v_local, v_all, 6, MPI_DOUBLE, MPI_SUM, world);
    virial[lat1] += v_all[lat1];
    virial[lat2] += v_all[lat2];
  }

  delete[] recvcounts;
  delete[] displs;
  delete[] zloc;
  delete[] bloc;
  delete[] zall;
  delete[] ball;
}

/* ----------------------------------------------------------------------
   z-binned slab correction (1D particle-mesh, cloud-in-cell).
   Bins the B-weighted density onto a periodic z-grid, convolves with the
   slab kernels, and interpolates energy/force/virial back to atoms.  Reproduces
   the exact pairwise result as the bin width -> 0, at O(nbins*nwin)+O(N) cost
   instead of O(N*N_slice).  Force is the exact z-gradient of the binned energy
   (CIC), so it conserves energy.
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_bin()
{
  const double zprd = domain->prd[dim];
  const double zlo = domain->boxlo[dim];
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // grid: bin width must resolve the kernel peak (width ~1/g_ewald).  Default
  // dz = 1/(40 g_ewald) (~0.4% error), capped at 0.025*cutoff; user-tunable.

  double dz_target = (bin_dz_user > 0.0) ? bin_dz_user : MIN(0.025 / g_ewald, 0.025 * cutoff);
  int nbins = (int) (zprd / dz_target + 0.5);
  if (nbins < 4) nbins = 4;
  const double dz = zprd / nbins;
  int nwin = (int) (cutoff / dz) + 1;
  if (nwin > nbins / 2) nwin = nbins / 2;    // kernel window cannot exceed half the box

  auto *dens = new double[nbins];
  auto *phiW = new double[nbins];
  auto *phiPT = new double[nbins];
  for (int b = 0; b < nbins; b++) dens[b] = 0.0;

  // CIC-assign B-weighted density; remember each atom's (b0, frac)

  auto *ab0 = new int[nlocal > 0 ? nlocal : 1];
  auto *afrac = new double[nlocal > 0 ? nlocal : 1];
  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zlo) / dz;
    u -= nbins * floor(u / nbins);    // wrap into [0,nbins)
    int b0 = (int) u;
    if (b0 >= nbins) b0 -= nbins;
    double frac = u - (int) u;
    int b1 = b0 + 1;
    if (b1 >= nbins) b1 -= nbins;
    const double bi = B[type[i]];
    dens[b0] += bi * (1.0 - frac);
    dens[b1] += bi * frac;
    ab0[i] = b0;
    afrac[i] = frac;
  }

  // global density (full grid on every proc)

  auto *dens_all = new double[nbins];
  MPI_Allreduce(dens, dens_all, nbins, MPI_DOUBLE, MPI_SUM, world);

  // precompute symmetric kernels vs bin offset.
  // NOTE: the slab kernel is sharply peaked at xi=0 with width ~1/g_ewald, so the
  // bin width must resolve it (dz <~ 1/(10 g_ewald)); the default below does this.

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

  // convolve density with kernels (periodic)

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

  // global energy = 0.5 * sum_b dens_all[b] * phiW[b] (full on every proc); zz
  // virial is handled by the trace in compute(), so only tangential here

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

  // forces (CIC gradient of the binned energy), per-atom energy buffer / virial

  for (int i = 0; i < nlocal; i++) {
    int b0 = ab0[i];
    double frac = afrac[i];
    int b1 = b0 + 1;
    if (b1 >= nbins) b1 -= nbins;
    const double bi = B[type[i]];
    // f = -B_i d/dz_i [0.5 sum dens phi] = -B_i (phiW[b1]-phiW[b0])/dz
    f[i][dim] += -bi * (phiW[b1] - phiW[b0]) / dz;
    if (evflag_atom) peatom[i] += 0.5 * bi * (phiW[b0] * (1.0 - frac) + phiW[b1] * frac);
    if (vflag_atom) {
      double pt = phiPT[b0] * (1.0 - frac) + phiPT[b1] * frac;
      vatom[i][lat1] += 0.5 * bi * pt;
      vatom[i][lat2] += 0.5 * bi * pt;
    }
  }

  delete[] dens;
  delete[] dens_all;
  delete[] phiW;
  delete[] phiPT;
  delete[] Kw;
  delete[] Kpt;
  delete[] ab0;
  delete[] afrac;
}

/* ----------------------------------------------------------------------
   standard sine/cosine integrals Si(x), Ci(x)
     series for x <= 2, Lentz continued fraction (exp-integral) for x > 2
------------------------------------------------------------------------- */

void EwaldDispSlab::cisi(double x, double &si, double &ci)
{
  if (x <= 2.0) {
    double term = x, s = x;    // Si series: sum (-1)^k x^(2k+1)/((2k+1)(2k+1)!)
    for (int k = 1; k < 60; k++) {
      term *= -x * x / ((2.0 * k) * (2.0 * k + 1.0));
      double add = term / (2.0 * k + 1.0);
      s += add;
      if (fabs(add) < 1.0e-18 * fabs(s)) break;
    }
    si = s;
    // Cin series: sum_{k>=1} (-1)^(k+1) x^(2k)/((2k)(2k)!); Ci = gamma + ln x - Cin
    double cterm = 1.0, cin = 0.0;
    for (int k = 1; k < 60; k++) {
      cterm *= -x * x / ((2.0 * k - 1.0) * (2.0 * k));
      double add = -cterm / (2.0 * k);
      cin += add;
      if (fabs(add) < 1.0e-18 * (fabs(cin) + 1.0e-300)) break;
    }
    ci = EULER + log(x) - cin;
  } else {
    // modified Lentz continued fraction for the complex exponential integral
    const double tiny = 1.0e-300;
    double br = 1.0, bi = x;           // b = 1 + i x
    double cr = 1.0e308, cii = 0.0;    // c = big
    // d = 1/b
    double den = br * br + bi * bi;
    double dr = br / den, di = -bi / den;
    double hr = dr, hi = di;
    for (int i = 1; i < 400; i++) {
      double a = -(double) i * i;
      br += 2.0;
      // d = 1/(a*d + b)
      double tr = a * dr + br, ti = a * di + bi;
      den = tr * tr + ti * ti;
      if (den < tiny) den = tiny;
      dr = tr / den;
      di = -ti / den;
      // c = b + a/c
      double cden = cr * cr + cii * cii;
      if (cden < tiny) cden = tiny;
      cr = br + a * cr / cden;
      cii = bi - a * cii / cden;
      // delta = c*d ; h *= delta
      double delr = cr * dr - cii * di;
      double deli = cr * di + cii * dr;
      double nhr = hr * delr - hi * deli;
      double nhi = hr * deli + hi * delr;
      hr = nhr;
      hi = nhi;
      if (fabs(delr - 1.0) + fabs(deli) < 1.0e-17) break;
    }
    // h *= (cos x - i sin x)
    double cx = cos(x), sx = sin(x);
    double fr = hr * cx + hi * sx;
    double fi = -hr * sx + hi * cx;
    ci = -fr;
    si = MY_PI / 2.0 + fi;
  }
}

/* ----------------------------------------------------------------------
   generalized integrals A_m = Si_m, B_m = Ci_m via integration-by-parts
   recurrence anchored at the standard Si/Ci.  Fills Aarr[1..7], Barr[1..7].
     A_m = -sin x x^{1-m}/(m-1) + B_{m-1}/(m-1)
     B_m = -cos x x^{1-m}/(m-1) - A_{m-1}/(m-1)
------------------------------------------------------------------------- */

void EwaldDispSlab::sici_chain(double x, double *Aarr, double *Barr)
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

/* ----------------------------------------------------------------------
   allocate K-vector-dependent arrays
------------------------------------------------------------------------- */

void EwaldDispSlab::allocate()
{
  memory->create(GU, kmax, "ewald/disp/slab:GU");
  memory->create(GF, kmax, "ewald/disp/slab:GF");
  memory->create(GT, kmax, "ewald/disp/slab:GT");
  memory->create(sfacrl, kmax, "ewald/disp/slab:sfacrl");
  memory->create(sfacim, kmax, "ewald/disp/slab:sfacim");
  memory->create(sfacrl_all, kmax, "ewald/disp/slab:sfacrl_all");
  memory->create(sfacim_all, kmax, "ewald/disp/slab:sfacim_all");
}

/* ---------------------------------------------------------------------- */

void EwaldDispSlab::deallocate()
{
  memory->destroy(GU);
  memory->destroy(GF);
  memory->destroy(GT);
  memory->destroy(sfacrl);
  memory->destroy(sfacim);
  memory->destroy(sfacrl_all);
  memory->destroy(sfacim_all);
  GU = GF = GT = nullptr;
  sfacrl = sfacim = sfacrl_all = sfacim_all = nullptr;
}

/* ---------------------------------------------------------------------- */

double EwaldDispSlab::memory_usage()
{
  double bytes = 7.0 * kmax * sizeof(double);
  bytes += (double) nmax * sizeof(double);
  bytes += 2.0 * (double) kmax * nmax * sizeof(double);
  if (profile_flag) bytes += 2.0 * (double) npro * sizeof(double);
  return bytes;
}
