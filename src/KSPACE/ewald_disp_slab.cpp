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
    KSpace(lmp), GU(nullptr), GF(nullptr), GT(nullptr), GN(nullptr), ek(nullptr), peatom(nullptr),
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
  bin_nbins = 0;
  sw_width = 0.0;
  switch_order = 3;
  wEgrid = wFgrid = wTgrid = wNgrid = nullptr;
  nwgrid = 0;
  wdz = 0.0;
  corr_switch = 0;
  cWgrid = nullptr;
  ncgrid = 0;
  cwdz = 0.0;
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
  delete[] wEgrid;
  delete[] wFgrid;
  delete[] wTgrid;
  delete[] wNgrid;
  delete[] cWgrid;
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

    // tell the pair to evaluate the FULL dispersion (repulsion + 1/r^6) over the
    // shell [rcut, rcut+Delta] (exact 3-D): corr_csb() below removes the reciprocal
    // sum's plane mean-field S*u there, so the pair supplies the laterally-
    // correlated shell interaction.  (pppm/disp/slab also has corr_csb and sets
    // this flag the same way.)
    int *p_full = (int *) force->pair->extract("csb_full_shell", itmp2);
    if (p_full) *p_full = 1;
  }

  // damped (SSB) variant: if the matched pair supplies a dispersion switch width
  // (lj/cut/dispswitch in its default (1-S) mode), the 1/r^6 dispersion is faded
  // out over [rcut, rcut+Delta].  The corr then removes u_smooth out to rcut+Delta
  // and adds back the faded S/r^6 shell, so the corr potential -> 0 smoothly at
  // rcut+Delta (no force discontinuity at rcut).  This "smooth corr" is what the
  // high-order binned corr (corr bin) needs.  Exact corr raw works either way.
  corr_switch = 0;
  if (damp_flag == 1) {
    int itmp2;
    double *p_dz = (double *) force->pair->extract("disp_switch_width", itmp2);
    if (p_dz && *p_dz > 0.0) {
      sw_width = *p_dz;
      corr_switch = 1;
      // ensure the pair runs the (1-S) faded-dispersion path, not the CSB full shell
      int *p_full = (int *) force->pair->extract("csb_full_shell", itmp2);
      if (p_full) *p_full = 0;
    } else if (corr_mode == 1) {
      error->all(FLERR,
                 "kspace_modify corr bin (damped ewald/disp/slab) requires the matched "
                 "lj/cut/dispswitch pair style to switch off the dispersion smoothly at the "
                 "cutoff; use pair_style lj/cut/dispswitch <rcut> <Delta>, or kspace_modify corr raw");
    }
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
  int etmp;    // extract() out-param; do NOT shadow the member `dim`
  auto **eps = (double **) force->pair->extract("epsilon", etmp);
  auto **sig = (double **) force->pair->extract("sigma", etmp);
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

  // (a user-set kmax still gets a predicted RMS force error reported below)

  // cumulative tail of GF[k]^2 from a large cap; pick kmax for target accuracy

  const int kbig = 8192;
  const double prefac = 0.5 * b2 * b2 / natoms;
  // The random-phase model rms = sqrt(prefac * sum_{k>kmax} GF[k]^2) under-predicts
  // the true per-atom force error by a roughly constant factor (the cross-term /
  // diagonal-approximation contribution; |S_k|^2 -> b2 in the relevant high-k tail).
  // For the compact switch this factor is ~1.4 for the default C3 septic and the
  // faster-decaying C5/C7; it rises toward ~2x only for the gentle C2 at loose
  // accuracy, whose slow ~k^-4 tail picks a small kmax where the random-phase
  // diagonal approximation is weakest.  Measured against the RMS force calculator
  // over kmax and switch order.  Fold in ~1.6 (the chosen kmax then meets the
  // requested accuracy within ~1.5x across orders, vs the old fixed 8x over-margin
  // that over-resolved by ~5x) and select with no extra margin.  The damped and
  // non-damped variants keep the original conservative 8x margin (not recalibrated).
  const double bias = (damp_flag == 2) ? 1.6 : 1.0;
  const double safety = (damp_flag == 2) ? 1.0 : 8.0;
  const double target = accuracy * accuracy / (bias * bias * safety * safety);

  // sum GF^2 from the top down so tail(kmax) = sum_{k>kmax}
  auto *gf2 = new double[kbig + 1];
  if (damp_flag == 2) {
    // The compact-switch GF decays only ALGEBRAICALLY (~k^-5) past the switch
    // bandwidth, so the tail is not negligible relative to the *peak* -- only
    // once its remaining sum falls below the tightest tail the target accuracy
    // needs.  Compute upward and stop when the estimated remaining tail
    // (Sum_{j>k} gf2 ~ gf2[k]*k/9 for a ~k^-10 spectrum) is a small fraction of
    // that target tail; the genuinely-negligible remainder is zeroed.  (The old
    // "10 orders below peak" cutoff truncated this algebraic tail, making the
    // estimator report zero at high kmax.)  C[m]/D[m] keep the high-k gf2 free
    // of cancellation roundoff so the summed tail is real, not noise.
    const double tail_floor = 1.0e-3 * accuracy * accuracy / (16.0 * prefac);
    int kstop = kbig;
    for (int k = 1; k <= kbig; k++) {
      double g = gf_of_k(k);
      gf2[k] = g * g;
      if (k > 16 && gf2[k] * k / 9.0 < tail_floor && gf2[k] < gf2[k - 1]) {
        kstop = k;
        break;
      }
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

  if (kmax_user > 0) kmax = kmax_user;
  else kmax = MAX(8, MIN(chosen, kbig));

  // predicted RMS per-atom force error at the chosen kmax (bias-corrected)
  double tk = 0.0;
  for (int k = kmax + 1; k <= kbig; k++) tk += gf2[k];
  estimated_force_accuracy = bias * sqrt(prefac * tk);
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
  if (damp_flag == 2) build_shell_vkernels();
  if (corr_switch) build_corr_kernels();

  // size the corr bin count to the target force accuracy (auto, unless the user
  // fixed the bin width); the compact switch uses corr_csb, not corr(), so skip.
  bin_nbins = 0;
  // the smooth switched corr_bin is high-order (cubic+Gauss); its default grid is
  // set from dz_target, not the CIC force-error calibration (which sizes the O(h)
  // CIC bin and would vastly over-provision the cubic grid).
  if (damp_flag == 1 && corr_mode == 1 && bin_dz_user <= 0.0 && !corr_switch) calibrate_bin();
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
    // converges fast (no Gibbs ringing).
    //
    // Energy/force, tangential (GT) and normal (GN) pressure are all the strain
    // derivative of the SAME functional U = sum_k GU[k]|S_k|^2.  Under strain
    // r_ij scales against the fixed lengths rcut,Delta, so the pressure picks up
    // the switch derivative: the relevant force is phi' = (S u)' = S' u + S u'.
    // The S u' piece is the smooth (non-damped) tail at rcut+Delta; the S' u piece
    // is a shell-localized term (the consistent counterpart of the matched pair's
    // -S'u switch-force virial).  GN is computed explicitly (the homogeneity trace
    // relation GN+2GT=6GU holds only for the pure power law, not for S*u).

    const double c = cutoff + sw_width;
    GU[0] = gu0_switch();           // energy
    GF[0] = 0.0;
    // k=0 (uniform) virial of the S*u long-range part: P = -(rho_B^2/6) int r phi'
    // 4 pi r^2 dr with phi=S*u, giving GT[0]=GN[0] = -(2 pi/3V)(-J + 6 trans + 2/c^3),
    // J = int_rcut^c S'(r)/r^3 dr, trans = int_rcut^c S(r)/r^4 dr.  This is the
    // pure-power-law value 2*GU[0] PLUS the (2 pi/3V) J switch-derivative (S'u) term
    // that the old "GT[0]=2*GU[0]" shortcut dropped, leaving a ~1/rcut^3 isotropic
    // pressure offset.  NOTE J does NOT vanish as Delta->0 (S' -> a delta at rcut, so
    // J -> 1/rcut^3); the plane S'u mean field this term represents is removed by
    // corr_csb() and replaced by the matched pair's exact 3-D shell, so the TOTAL
    // pressure -> the sharp non-damped value for any Delta (verified to Delta=0.01).
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
        const double r2 = r * r, r3 = r2 * r, r4 = r2 * r2;
        const double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        iJ += w * Sp / r3;
        iT += w * S / r4;
      }
      const double Jint = dr / 3.0 * iJ;
      const double trans = dr / 3.0 * iT;
      GT[0] = GN[0] = -(2.0 * MY_PI / (3.0 * volume)) * (-Jint + 6.0 * trans + 2.0 / (c * c * c));
    }
    for (k = 1; k < kcount; k++) {
      kcell = k * unitk;
      kcell3 = kcell * kcell * kcell;
      const double t5 = switch_trans5(kcell);

      // energy: smoothed tail at rcut+Delta plus the shell transition.  The tail
      // combinations (pi/48 - A[5]) etc. are taken from the complementary chain
      // C[m]=A[m](inf)-A[m], D[m]=B[m](inf)-B[m] so the small high-k coefficients
      // are computed without the A[m]=A[m](inf)+(tiny) roundoff cancellation.
      double C[8], D[8];
      sici_compl_chain(kcell * c, C, D);    // evaluated at rcut+Delta
      GU[k] = (-4.0 * MY_PI * kcell3 / volume) * C[5] - (4.0 * MY_PI / volume) * t5 / kcell;
      GF[k] = 2.0 * kcell * GU[k];    // exact z-gradient of the energy term

      // pressure: consistent (S u)' mean-field = non-damped tail at rcut+Delta plus
      // the shell virial of phi' = (S u)' = S'u + S u'.  This is the strain
      // derivative of the same S*u functional as the energy/force (no sharp split);
      // its shell-correlation residual is removed by the real-space correction below.
      const double GTtail = (-24.0 * MY_PI * kcell3 / volume) * (C[7] - D[6]);
      const double GNtail =
          (-24.0 * MY_PI * kcell3 / volume) * (C[5] - 2.0 * C[7] + 2.0 * D[6]);
      double sGT, sGN;
      switch_shell_virial(kcell, sGT, sGN);
      GT[k] = GTtail - (MY_PI / volume) * sGT;
      GN[k] = GNtail - (2.0 * MY_PI / volume) * sGN;
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
    GN[0] = GU[0];

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
      // exact per-mode normal (zz) coefficient: under an eps_zz strain, S_k is
      // invariant (h_k z_i fixed), so GN = -dGU/deps = GU + h dGU/dh.  With
      // Bk = h^3 F(b), b = h/(2g): F'(b) = -(3/2) e^{-b^2}/b^4 (the erfc and
      // exponential terms cancel), giving h dBk/dh = 3 Bk - (3/2) h^3 e^{-b^2}/b^3.
      // Used by the smooth switched variant (corr_switch), whose kspace share is
      // not degree -6 homogeneous, so the 6E trace does not apply.
      GN[k] = coef * (4.0 * Bk - 1.5 * kcell3 * exp(-b2) / b3);
    }
  }
}

/* ----------------------------------------------------------------------
   compact-switch (CSB) smoothstep S(t), the order-n "smootherstep" on t in
   [0,1]: S(0)=0, S(1)=1 with the first n derivatives zero at both ends (C^n).
   In r: S=0 for r<=rcut, S=1 for r>=rcut+Delta.  The long-range part fed to the
   reciprocal sum is S(r)*u(r); it vanishes inside rcut (so no slab correction)
   and meets r>=rcut with C^n continuity, so the z-Fourier coefficients decay as
   ~k^-(n+2) (no Gibbs ringing).  n=3 (septic) default; 5 or 7 decay faster but
   the transition is steeper.  Selectable via kspace_modify disp/switch/order.
------------------------------------------------------------------------- */

double EwaldDispSlab::switch_S(double t)
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

/* ----------------------------------------------------------------------
   transition integrals over the switch shell [rcut, rcut+Delta]:
     t5 = int S(r) r^-5 sin(h r) dr   (energy)
     t7 = int S(r) r^-7 sin(h r) dr   (tangential, Si7 part)
     t6 = int S(r) r^-6 cos(h r) dr   (tangential, Ci6 part)
   composite Simpson with the panel count scaled to the oscillation count
   (h*Delta) so accuracy is k-independent.  Built once at setup, so cheap.
------------------------------------------------------------------------- */

double EwaldDispSlab::switch_trans5(double h)
{
  // 10-point Gauss-Legendre per panel; panel count scaled to the oscillation
  // count (h*Delta) so the result is accurate (~1e-13) for all h.  High accuracy
  // is required because GU = [tail at rcut+Delta] + [transition] is a difference
  // of two slowly-decaying (ringing) terms whose cancellation gives the true
  // fast-decaying coefficient.
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

/* ---------------------------------------------------------------------- */

double EwaldDispSlab::switch_dS(double t)
{
  // dS/dt of the order-n smoothstep is (2n+1)!/(n!)^2 * (t(1-t))^n
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double tu = t * (1.0 - t);
  if (switch_order == 1) return 6.0 * tu;            // 6 t(1-t)
  if (switch_order == 2) return 30.0 * tu * tu;      // 30 (t(1-t))^2
  if (switch_order == 5) {
    const double tu2 = tu * tu;
    return 2772.0 * tu2 * tu2 * tu;    // 2772 (t(1-t))^5
  }
  if (switch_order == 7) {
    const double tu2 = tu * tu, tu3 = tu2 * tu;
    return 51480.0 * tu3 * tu3 * tu;    // 51480 (t(1-t))^7
  }
  return 140.0 * tu * tu * tu;    // 140 (t(1-t))^3
}

/* ----------------------------------------------------------------------
   shell virial integrals over [rcut, rcut+Delta]:
     sGT = int phi'(r) A_T(r,h) dr,   sGN = int phi'(r) A_N(r,h) dr,
   with the FULL switched-dispersion force phi'(r) = (S u)'(r) = -S'(r)/r^6 +
   6 S(r)/r^7 -- i.e. the consistent strain derivative of the energy functional
   sum_k GU[k]|S_k|^2 (the S'(r)u "switch-force" term is INCLUDED).  This plane
   mean field over the shell is what corr_csb() then subtracts and replaces with
   the matched pair's exact 3-D shell virial, so the residual is removed by the
   real-space correction, not by dropping the S'u term here.  Angular factors:
     A_T = -4 r cos(hr)/h^2 + 4 sin(hr)/h^3,
     A_N =  2 r^2 sin(hr)/h + 4 r cos(hr)/h^2 - 4 sin(hr)/h^3.
   GT = GT_tail - (pi/V) sGT, GN = GN_tail - (2 pi/V) sGN.
------------------------------------------------------------------------- */

void EwaldDispSlab::switch_shell_virial(double h, double &sGT, double &sGN)
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
   compact-switch energy coefficient GU[k] (k>=1): non-damped tail evaluated at
   rcut+Delta plus the numerically integrated shell transition.
------------------------------------------------------------------------- */

double EwaldDispSlab::gu_switch(int k)
{
  const double kcell = k * unitk;
  const double kcell3 = kcell * kcell * kcell;
  const double c = cutoff + sw_width;
  double C[8], D[8];
  sici_compl_chain(kcell * c, C, D);    // (pi/48 - A[5]) = C[5], cancellation-free
  const double t5 = switch_trans5(kcell);
  return (-4.0 * MY_PI * kcell3 / volume) * C[5] - (4.0 * MY_PI / volume) * t5 / kcell;
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
------------------------------------------------------------------------- */

void EwaldDispSlab::build_shell_vkernels()
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

void EwaldDispSlab::shell_vkernel(double adz, double &wE, double &wF, double &wT, double &wN)
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
   compact-switch shell virial correction dispatcher
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_csb()
{
  if (corr_mode == 1)
    corr_csb_bin();
  else
    corr_csb_raw();
}

/* ----------------------------------------------------------------------
   exact (global z-gather) subtraction of the plane (mean-field) shell energy,
   z-force and virial.  Mirrors corr_raw: every proc gathers the global (z, B)
   list and each local atom sums the plane kernel over all global atoms in its
   |dz| < rcut+Delta window (slab-slab).  Removes what the reciprocal sum put in
   the shell with a laterally-uniform density so the matched pair's exact 3-D
   shell interaction (full u to rcut+Delta) is what remains.  Matches the kspace
   |S_k|^2 convention: full ordered double sum incl. self, so the energy/virial
   carry no 1/2; the z-force = -d E/d z_i differentiates both pair indices and so
   carries a factor 2 (as the reciprocal GF[k]=2k GU[k] force does).
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_csb_raw()
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
   z-binned version of the shell virial correction (1D particle-mesh, CIC).
   Bins the B-weighted density, convolves with the plane kernels, interpolates
   back.  O(nbins*nwin)+O(N) instead of O(N*N_slice).
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_csb_bin()
{
  const double zprd = domain->prd[dim];
  const double zlo = domain->boxlo[dim];
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
    double u = (x[i][dim] - zlo) / dz;
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

  // precompute energy and virial kernels on the bin offsets (the force is the
  // exact gradient of the binned energy, so no separate force kernel is needed)
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

  // global energy = sum_b dens phiE (full ordered convention, no 1/2); subtracted
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

  // forces (CIC gradient of the binned energy; factor 2 from the ordered double
  // sum, matching the reciprocal GF[k]=2k GU[k]) and per-atom energy/virial
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
          // tangential from GT; normal (dim) from GN (compact switch) or the
          // virial trace (other variants, set after corr below)
          vatom[i][lat1] += GT[k] * partial_peratom;
          vatom[i][lat2] += GT[k] * partial_peratom;
          if (damp_flag == 2 || corr_switch) vatom[i][dim] += GN[k] * partial_peratom;
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
      if (damp_flag == 2 || corr_switch) virial[dim] += uk * GN[k];    // explicit normal kernel
    }
  }

  // scale per-atom energy buffer / virial by each atom's B

  if (evflag_atom)
    for (i = 0; i < nlocal; i++) peatom[i] *= B[type[i]];
  if (vflag_atom)
    for (i = 0; i < nlocal; i++) {
      vatom[i][lat1] *= B[type[i]];
      vatom[i][lat2] *= B[type[i]];
      if (damp_flag == 2 || corr_switch) vatom[i][dim] *= B[type[i]];
    }

  // damped variant: real-space "slab" correction (adds to energy, corr_energy,
  // tangential virial, and the per-atom energy buffer peatom; the zz virial is
  // set from the trace below)

  corr_energy = 0.0;
  if (damp_flag == 1) corr();

  // compact-switch (CSB) shell correction.  The reciprocal sum treats the shell
  // [rcut, rcut+Delta] with a laterally-uniform density (plane mean field), which
  // leaves a lateral-correlation residual in energy AND pressure that grows with
  // Delta.  corr_csb() subtracts that plane mean field (energy, z-force, virial)
  // so the matched pair -- which now evaluates the FULL dispersion u to rcut+Delta
  // with exact 3-D correlation -- supplies the shell interaction instead.  Must run
  // every step (the z-force is removed unconditionally, else it is double counted).
  if (damp_flag == 2) corr_csb();

  // normal (zz) virial.  For the sharp/Gaussian variants the dispersion is the
  // exact power law (homogeneous degree -6), so the trace gives it cheaply:
  // sum r.f = 6 U => virial_zz = 6*E_kspace - virial_xx - virial_yy.  The compact
  // switch and the smooth switched damped variant (corr_switch) are
  // non-homogeneous (S varies), so their normal is explicit: the reciprocal GN
  // kernel accumulated above plus (corr_switch) the corr dz^2*f2 term added in
  // corr_raw/corr_bin_smooth.
  if (damp_flag != 2 && !corr_switch) {
    if (vflag_global) virial[dim] = 6.0 * (e_recip + corr_energy) - virial[lat1] - virial[lat2];
    if (vflag_atom)
      for (i = 0; i < nlocal; i++)
        vatom[i][dim] = 6.0 * peatom[i] - vatom[i][lat1] - vatom[i][lat2];
  }

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
  // compact switch: anchor the tail at the OUTER cutoff rcut+Delta and add the
  // switch-shell integral so Phi is consistent with the switched potential S(r)/r^6
  // (ported from ewald/disp/planar; reduces to the sharp form as Delta->0).
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

double EwaldDispSlab::ik_psi(double h)
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
   potential-form integrand g(r) of a profile coefficient: the sharp coefficient
   is int_rcut^inf g(r) dr.  (x = h r; ported from ewald/disp/planar.)
     PROF_T   (Tn & Psi):  sin(hr)/(h^6 r^7) - cos(hr)/(h^5 r^6)
     PROF_N   (Nn):        sin(hr)/(h^4 r^5) - 2 sin(hr)/(h^6 r^7) + 2 cos(hr)/(h^5 r^6)
     PROF_PHI (Phi):       Si(hr)/(h^4 r^5) - sin(hr)/(h^6 r^7) + cos(hr)/(h^5 r^6)
------------------------------------------------------------------------- */

double EwaldDispSlab::prof_integrand(int which, double r, double h)
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
   compact-switch shell correction for a profile coefficient:
     int_rcut^{rcut+Delta} W(r) g(r) dr,  W(r) = S(r) - S'(r) r / 6
   (the force-reweight (S u)'/(6/r^7) that makes the shell term identical to the
   global switch_shell_virial; sharp result recovered as Delta->0).  10-point
   Gauss-Legendre, panel count scaled to the oscillation count h*Delta.
------------------------------------------------------------------------- */

double EwaldDispSlab::prof_shell(int which, double h)
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
  double acc = 0.0;
  for (int p = 0; p < np; p++) {
    const double c0 = a + (p + 0.5) * hp;
    for (int g = 0; g < 10; g++) {
      const double r = c0 + 0.5 * hp * gx[g];
      const double t = (r - a) / dz;
      const double W = switch_S(t) - (switch_dS(t) / dz) * r / 6.0;    // (S u)'/(6/r^7)
      acc += gw[g] * W * prof_integrand(which, r, h);
    }
  }
  return 0.5 * hp * acc;
}

/* ----------------------------------------------------------------------
   shell-correction virial per profile bin (compact switch), dispatched on
   corr_mode so the contour profile uses the IDENTICAL real-space corr_csb
   correction as the box average (raw = exact per-atom shell virial spread
   Irving-Kirkwood along each bond; bin = density-density convolution, also
   IK-spread).  Ported from ewald/disp/planar (geometric mixing).
------------------------------------------------------------------------- */

void EwaldDispSlab::shell_profile_virial(int nbins, double lo, double dz, double *dens_all,
                                         double *shellT, double *shellN)
{
  const double zprd = domain->prd[dim];
  const double bcut = cutoff + sw_width;
  for (int g = 0; g < nbins; g++) shellT[g] = shellN[g] = 0.0;

  if (corr_mode != 0) {    // BIN: density-density convolution (matches corr_csb_bin)
    for (int g = 0; g < nbins; g++) {
      for (int gp = 0; gp < nbins; gp++) {
        double ddz = (gp - g) * dz;
        ddz -= zprd * floor(ddz / zprd + 0.5);
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
      delz -= zprd * floor(delz / zprd + 0.5);
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
   long-range Irving-Kirkwood pressure profiles P_T(z), P_N(z) on the caller's
   nbins z grid (bin centers lo+(g+0.5)*width); the caller allocates pT/pN.
   Ported from ewald/disp/planar; compact-switch (CSB) variant only for now (the
   sharp/damped variants keep the internal kspace_modify pressure/profile path).
   P(z) = sum_{n,m} S_n S_m C_{n,m} e^{i(h_n+h_m)z} - shell(z); p=n+m=0 pinned to
   the global GT/GN; off-diagonal C^T = -6pi/H [Phi(h_m)+Phi(h_n)],
   C^N = -12pi/H [Psi(h_m)+Psi(h_n)].  Requires nbins > 2*kmax (anti-aliasing).
------------------------------------------------------------------------- */

int EwaldDispSlab::pressure_profile_long(int dir, int nbins, double lo, double width,
                                         double *pN, double *pT)
{
  if (damp_flag != 2)
    error->all(FLERR,
               "compute stress/cartesian kspace with ewald/disp/slab currently requires the "
               "compact-switch variant (kspace_modify damp compact); use kspace_modify "
               "pressure/profile for the sharp/damped variants");
  if (dir != dim)
    error->all(FLERR,
               "compute stress/cartesian binning direction must match the inhomogeneous axis "
               "of ewald/disp/slab");

  const double area = domain->prd[lat1] * domain->prd[lat2];
  const int K = kcount - 1;    // highest mode index

  // anti-aliasing: the IK profile sums modes e^{i p unitk z} with |p| up to 2*kmax
  if (nbins <= 2 * K)
    error->all(FLERR,
               "compute stress/cartesian with ewald/disp/slab kspace: {} bins along the "
               "inhomogeneous axis is too coarse; need > {} (= 2*kmax) to resolve the "
               "Irving-Kirkwood reciprocal modes without aliasing (use a finer bin width or "
               "smaller kmax)",
               nbins, 2 * K);

  auto *Sre = new double[K + 1];
  auto *Sim = new double[K + 1];
  for (int n = 0; n <= K; n++) {
    Sre[n] = sfacrl_all[n] / volume;
    Sim[n] = -sfacim_all[n] / volume;
  }

  // bin the B-weighted density (BIN-mode shell convolution source)
  const double dz = width;
  auto *dens = new double[nbins];
  for (int g = 0; g < nbins; g++) dens[g] = 0.0;
  {
    int *type = atom->type;
    double **x = atom->x;
    for (int i = 0; i < atom->nlocal; i++) {
      double u = (x[i][dim] - lo) / width;
      u -= nbins * floor(u / nbins);
      int g = (int) u;
      if (g >= nbins) g -= nbins;
      dens[g] += B[type[i]];
    }
  }
  auto *dens_all = new double[nbins];
  MPI_Allreduce(dens, dens_all, nbins, MPI_DOUBLE, MPI_SUM, world);

  // shell-correction virial per bin: same real-space corr_csb correction as the box
  auto *shellT = new double[nbins];
  auto *shellN = new double[nbins];
  shell_profile_virial(nbins, lo, width, dens_all, shellT, shellN);
  const double inv_adz = 1.0 / (area * dz);

  {
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
    // precompute the per-mode IK shape kernels once (odd in h), collapsing the
    // off-diagonal double sum from O(K^2) transcendental evaluations to O(K)
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
        double sre = snr * smr - sni * smi, sim = snr * smi + sni * smr;
        double CT, CN;
        if (n == 0 && m == 0) {
          CT = CN = volume * GT[0];    // (0,0): pinned to the global GT[0]=GN[0]
        } else if (fabs(H) < 1.0e-300) {    // n = -m diagonal: V*GT[k]/2, V*GN[k]/2
          int kk = (n < 0) ? -n : n;
          CT = 0.5 * volume * GT[kk];
          CN = 0.5 * volume * GN[kk];
        } else {    // off-diagonal: switch-aware Phi/Psi (sets the profile SHAPE)
          CT = -6.0 * MY_PI / H * (PHI(m) + PHI(n));
          CN = -12.0 * MY_PI / H * (PSI(m) + PSI(n));
        }
        ATre[p] += CT * sre;
        ATim[p] += CT * sim;
        ANre[p] += CN * sre;
        ANim[p] += CN * sim;
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
  delete[] dens;
  delete[] dens_all;
  delete[] shellT;
  delete[] shellN;
  delete[] Sre;
  delete[] Sim;
  return 1;
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
   smooth (Gaussian-screened) 1/r^6 = u(r) - u_short(r), the long-range part the
   reciprocal sum represents.  Taylor series near r=0 to avoid 1/r^6 cancellation.
------------------------------------------------------------------------- */

double EwaldDispSlab::u_smooth(double r)
{
  const double g2 = g_ewald * g_ewald;
  const double r2 = r * r;
  const double a2 = g2 * r2;    // (g_ewald * r)^2
  if (a2 < 0.1) {
    const double g6 = g2 * g2 * g2, g8 = g6 * g2, g10 = g8 * g2, g12 = g10 * g2;
    return g6 / 6.0 - g8 * r2 / 8.0 + g10 * r2 * r2 / 20.0 - g12 * r2 * r2 * r2 / 72.0;
  }
  const double r6 = r2 * r2 * r2;
  return (1.0 - (1.0 + a2 + 0.5 * a2 * a2) * exp(-a2)) / r6;
}

/* ----------------------------------------------------------------------
   tabulate the smooth (switched-pair) damped correction kernels over [0, rcut+Delta].
   With the matched lj/cut/dispswitch pair the 1/r^6 dispersion is faded out by (1-S)
   over [rcut, rcut+Delta], so the corr potential
       corr_e(r) = u_smooth(r) - [r>rcut] S(r)/r^6
   vanishes smoothly at rcut+Delta (corr_e(rcut+Delta) = u_short(rcut+Delta) ~ acc^2).
   The z-force kernel f2 = (2 pi/area) corr_e(|dz|) is analytic (see corr_smooth_kernels);
   here we tabulate the energy kernel w2 = (2 pi/area) int_{|dz|}^{b} r corr_e(r) dr by
   quadrature (Simpson).  Matches the sharp corr_kernels conventions (so corr_raw/corr_bin
   use it the same way), but the smooth upper limit removes the rcut force discontinuity.
------------------------------------------------------------------------- */

void EwaldDispSlab::build_corr_kernels()
{
  const double a = cutoff, b = cutoff + sw_width;
  const double area = domain->prd[lat1] * domain->prd[lat2];
  const double pre = 2.0 * MY_PI / area;
  ncgrid = 1024;
  cwdz = b / ncgrid;
  delete[] cWgrid;
  cWgrid = new double[ncgrid + 1];
  for (int g = 0; g <= ncgrid; g++) {
    const double adz = g * cwdz;
    const int n = 800;
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
   evaluate the smooth switched corr kernels: analytic z-force f2, interpolated
   energy w2, tangential pt2.  |dz| measured in z; support is [0, rcut+Delta].
   pt2 = w2 exactly: the tangential strain derivative -(pi/A) int (r^2-dz^2)
   phi'(r) dr integrates by parts to (2pi/A) int r phi dr = w2 because the
   switched corr potential phi(rcut+Delta) ~ acc^2 ~ 0 (for the sharp lj/cut
   kernel the rcut boundary term is what makes corr_kernels' pt2 != w2).
   The normal (zz) virial is per-pair dz^2 * f2, accumulated by the callers.
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_smooth_kernels(double adz, double &w2, double &f2, double &pt2)
{
  const double b = cutoff + sw_width;
  if (adz >= b) {
    w2 = f2 = pt2 = 0.0;
    return;
  }
  const double area = domain->prd[lat1] * domain->prd[lat2];
  const double pre = 2.0 * MY_PI / area;
  const double rr = (adz > 1.0e-300) ? adz : 1.0e-300;
  double ce = u_smooth(rr);
  if (adz > cutoff) {
    const double r6 = rr * rr * rr * rr * rr * rr;
    ce -= switch_S((adz - cutoff) / sw_width) / r6;
  }
  f2 = pre * ce;
  const double xg = adz / cwdz;
  int g = (int) xg;
  if (g >= ncgrid) g = ncgrid - 1;
  const double fr = xg - g;
  w2 = cWgrid[g] * (1.0 - fr) + cWgrid[g + 1] * fr;
  pt2 = w2;
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

  const double cut2 = corr_switch ? (cutoff + sw_width) * (cutoff + sw_width) : rc2;
  double w0, f0, pt0;
  if (corr_switch)
    corr_smooth_kernels(0.0, w0, f0, pt0);
  else
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
      if (x2 >= cut2) continue;

      double w2, f2, pt2;
      if (corr_switch)
        corr_smooth_kernels(sqrt(x2), w2, f2, pt2);
      else
        corr_kernels(x2, w2, f2, pt2);
      const double bij = bi * ball[jg];

      // each unordered pair is summed from both ends -> 0.5 weight on energy/virial.
      // normal (dim) virial: explicit dz^2*f2 (r_z f_z) for the smooth switched
      // kernel; the sharp kernel keeps the trace in compute() instead.
      e_local += 0.5 * bij * w2;
      fz_i += delz * bij * f2;
      v_local[lat1] += 0.5 * bij * pt2;    // tangential lat1
      v_local[lat2] += 0.5 * bij * pt2;    // tangential lat2
      if (corr_switch) v_local[dim] += 0.5 * bij * x2 * f2;

      if (evflag_atom) peatom[i] += 0.5 * bij * w2;
      if (vflag_atom) {
        vatom[i][lat1] += 0.5 * bij * pt2;
        vatom[i][lat2] += 0.5 * bij * pt2;
        if (corr_switch) vatom[i][dim] += 0.5 * bij * x2 * f2;
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
    if (corr_switch) virial[dim] += v_all[dim];
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
  // smooth switched kernel -> high-order (cubic-moment + Gauss) binning
  if (corr_switch) {
    corr_bin_smooth();
    return;
  }

  const double zprd = domain->prd[dim];
  const double zlo = domain->boxlo[dim];
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // grid: bin width must resolve the kernel peak (width ~1/g_ewald).  Default
  // dz = 1/(40 g_ewald) (~0.4% error), capped at 0.025*cutoff; user-tunable.

  const double rwin = corr_switch ? (cutoff + sw_width) : cutoff;
  const double cut2 = corr_switch ? rwin * rwin : rc2;
  double dz_target = (bin_dz_user > 0.0) ? bin_dz_user : MIN(0.025 / g_ewald, 0.025 * cutoff);
  int nbins = (bin_nbins > 0) ? bin_nbins : (int) (zprd / dz_target + 0.5);
  if (nbins < 4) nbins = 4;
  const double dz = zprd / nbins;
  int nwin = (int) (rwin / dz) + 1;
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
    if (x2 >= cut2) {
      Kw[d] = Kpt[d] = 0.0;
    } else if (corr_switch) {
      corr_smooth_kernels(xi, w2, f2, pt2);
      Kw[d] = w2;
      Kpt[d] = pt2;
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
   high-order binned corr for the smooth switched kernel.  Each atom is binned
   into one cell; the cell's B-weighted density is reconstructed as a cubic
   rho(s)=c0+c1 s+c2 s^2+c3 s^3 from its first four moments m_k=sum B (z-zc)^k,
   and the corr energy/force/tangential convolutions are evaluated by 8-point
   Gauss quadrature against the (smooth, analytic) kernel.  The smooth kernel has
   no force discontinuity, so this converges ~O(h^3) (1e-5 at ~600 bins) where the
   CIC corr_bin is O(h).  Force = sum_j B_i B_j (z_i-z_j) f2 is exactly the
   z-gradient of the binned energy (f2 = -d w2/d dz), so energy is conserved.
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_bin_smooth()
{
  const double zprd = domain->prd[dim];
  const double zlo = domain->boxlo[dim];
  const double bcut = cutoff + sw_width;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double dz_target = (bin_dz_user > 0.0) ? bin_dz_user : MIN(0.025 / g_ewald, 0.025 * cutoff);
  int nbins = (bin_nbins > 0) ? bin_nbins : (int) (zprd / dz_target + 0.5);
  if (nbins < 4) nbins = 4;
  const double dz = zprd / nbins;
  const double half = 0.5 * dz;

  // per-cell moments m_k = sum_j B_j (z_j - zc)^k, k=0..3
  const int NM = 4;
  auto *M = new double[NM * nbins];
  for (int t = 0; t < NM * nbins; t++) M[t] = 0.0;
  auto *acell = new int[nlocal > 0 ? nlocal : 1];
  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zlo) / dz;
    u -= nbins * floor(u / nbins);
    int c = (int) u;
    if (c >= nbins) c -= nbins;
    acell[i] = c;
    const double zc = zlo + (c + 0.5) * dz;
    double s = x[i][dim] - zc;
    s -= zprd * floor(s / zprd + 0.5);
    double sp = B[type[i]];
    for (int k = 0; k < NM; k++) {
      M[k * nbins + c] += sp;
      sp *= s;
    }
  }
  auto *Mall = new double[NM * nbins];
  MPI_Allreduce(M, Mall, NM * nbins, MPI_DOUBLE, MPI_SUM, world);

  // cubic moment matrix on [-half,half] is block-diagonal: even (c0,c2)<-(m0,m2),
  // odd (c1,c3)<-(m1,m3).  p_k = int_{-half}^{half} s^k ds.
  const double p0 = dz;
  const double p2 = dz * dz * dz / 12.0;
  const double p4 = dz * dz * dz * dz * dz / 80.0;
  const double p6 = dz * dz * dz * dz * dz * dz * dz / 448.0;
  const double detE = p0 * p4 - p2 * p2;
  const double detO = p2 * p6 - p4 * p4;

  static const double gx[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                               -0.1834346424956498, 0.1834346424956498,  0.5255324099163290,
                               0.7966664774136267,  0.9602898564975363};
  static const double gw[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                               0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                               0.2223810344533745, 0.1012285362903763};

  const int reach = (int) ((bcut + half) / dz) + 1;

  double e_local = 0.0, vt_local = 0.0, vn_local = 0.0;
  for (int i = 0; i < nlocal; i++) {
    const double zi = x[i][dim];
    const double bi = B[type[i]];
    const int ci = acell[i];
    double accF = 0.0, accE = 0.0, accT = 0.0, accN = 0.0;
    for (int off = -reach; off <= reach; off++) {
      int c = (ci + off) % nbins;
      if (c < 0) c += nbins;
      const double zc = zlo + (c + 0.5) * dz;
      double dc = zi - zc;
      dc -= zprd * floor(dc / zprd + 0.5);
      if (fabs(dc) >= bcut + half) continue;
      const double m0 = Mall[0 * nbins + c], m1 = Mall[1 * nbins + c], m2 = Mall[2 * nbins + c],
                   m3 = Mall[3 * nbins + c];
      const double c0 = (p4 * m0 - p2 * m2) / detE;
      const double c2 = (-p2 * m0 + p0 * m2) / detE;
      const double c1 = (p6 * m1 - p4 * m3) / detO;
      const double c3 = (-p4 * m1 + p2 * m3) / detO;
      for (int q = 0; q < 8; q++) {
        const double s = half * gx[q];
        const double W = half * gw[q];
        const double rho = c0 + s * (c1 + s * (c2 + s * c3));
        const double d = dc - s;
        const double ad = fabs(d);
        if (ad >= bcut) continue;
        double w2, f2, pt2;
        corr_smooth_kernels(ad, w2, f2, pt2);
        accF += W * rho * d * f2;    // KFz(d) = d * f2(|d|)
        accE += W * rho * w2;
        accT += W * rho * pt2;
        accN += W * rho * d * d * f2;    // normal (zz): r_z f_z = d^2 f2
      }
    }
    f[i][dim] += bi * accF;
    e_local += 0.5 * bi * accE;
    vt_local += 0.5 * bi * accT;
    vn_local += 0.5 * bi * accN;
    if (evflag_atom) peatom[i] += 0.5 * bi * accE;
    if (vflag_atom) {
      vatom[i][lat1] += 0.5 * bi * accT;
      vatom[i][lat2] += 0.5 * bi * accT;
      vatom[i][dim] += 0.5 * bi * accN;
    }
  }

  double e_all;
  MPI_Allreduce(&e_local, &e_all, 1, MPI_DOUBLE, MPI_SUM, world);
  corr_energy = e_all;
  if (eflag_global) energy += e_all;
  if (vflag_global) {
    double vtn[2] = {vt_local, vn_local}, vtn_all[2];
    MPI_Allreduce(vtn, vtn_all, 2, MPI_DOUBLE, MPI_SUM, world);
    virial[lat1] += vtn_all[0];
    virial[lat2] += vtn_all[0];
    virial[dim] += vtn_all[1];
  }

  delete[] M;
  delete[] Mall;
  delete[] acell;
}

/* ----------------------------------------------------------------------
   lean exact-pairwise corr z-force (global z-gather) -> fzloc[nlocal].
   Calibration reference: corr_bin() should reproduce this to target accuracy.
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_raw_force(double *fzloc)
{
  const double zprd = domain->prd[dim];
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
      if (x2 >= (corr_switch ? (cutoff + sw_width) * (cutoff + sw_width) : rc2)) continue;
      double w2, f2, pt2;
      if (corr_switch)
        corr_smooth_kernels(sqrt(x2), w2, f2, pt2);
      else
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
   binned corr z-force at a given bin count -> fzloc[nlocal] (CIC, force only;
   used by calibrate_bin to size the grid; no energy/virial/state changes).
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_bin_force(int nbins, double *fzloc)
{
  const double zprd = domain->prd[dim];
  const double zlo = domain->boxlo[dim];
  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  if (nbins < 4) nbins = 4;
  const double dz = zprd / nbins;
  const double rwin = corr_switch ? (cutoff + sw_width) : cutoff;
  const double cut2 = corr_switch ? rwin * rwin : rc2;
  int nwin = (int) (rwin / dz) + 1;
  if (nwin > nbins / 2) nwin = nbins / 2;

  auto *dens = new double[nbins];
  auto *phiW = new double[nbins];
  for (int b = 0; b < nbins; b++) dens[b] = 0.0;
  auto *ab0 = new int[nlocal > 0 ? nlocal : 1];
  for (int i = 0; i < nlocal; i++) {
    double u = (x[i][dim] - zlo) / dz;
    u -= nbins * floor(u / nbins);
    int b0 = (int) u;
    if (b0 >= nbins) b0 -= nbins;
    double frac = u - (int) u;
    int b1 = b0 + 1;
    if (b1 >= nbins) b1 -= nbins;
    const double bi = B[type[i]];
    dens[b0] += bi * (1.0 - frac);
    dens[b1] += bi * frac;
    ab0[i] = b0;
  }
  auto *dens_all = new double[nbins];
  MPI_Allreduce(dens, dens_all, nbins, MPI_DOUBLE, MPI_SUM, world);

  auto *Kw = new double[nwin + 1];
  for (int d = 0; d <= nwin; d++) {
    double xi = d * dz, x2 = xi * xi, w2, f2, pt2;
    if (x2 >= cut2)
      Kw[d] = 0.0;
    else if (corr_switch) {
      corr_smooth_kernels(xi, w2, f2, pt2);
      Kw[d] = w2;
    } else {
      corr_kernels(x2, w2, f2, pt2);
      Kw[d] = w2;
    }
  }
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
  for (int i = 0; i < nlocal; i++) {
    int b0 = ab0[i];
    int b1 = b0 + 1;
    if (b1 >= nbins) b1 -= nbins;
    fzloc[i] = -B[type[i]] * (phiW[b1] - phiW[b0]) / dz;
  }

  delete[] dens;
  delete[] phiW;
  delete[] dens_all;
  delete[] Kw;
  delete[] ab0;
}

/* ----------------------------------------------------------------------
   choose the corr bin count so the binned corr force matches the exact pairwise
   corr force to the target RMS force accuracy (mirrors pppm/disp/slab).  Binning
   a sharp-cutoff kernel converges only ~sqrt(dz), so tight force targets cannot
   be met -- back off at the error floor and warn (corr raw is exact).
------------------------------------------------------------------------- */

void EwaldDispSlab::calibrate_bin()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double natoms = (double) atom->natoms;
  if (natoms < 1.0) natoms = 1.0;

  // skip if the dispersion amplitudes B are not populated yet (kspace->init()
  // runs before pair->init(), so the first setup() has B = 0); the production
  // setup() after pair->init() recalibrates with valid B.
  double bmax = 0.0;
  for (int i = 0; i < nlocal; i++) bmax = MAX(bmax, fabs(B[type[i]]));
  double bmax_all;
  MPI_Allreduce(&bmax, &bmax_all, 1, MPI_DOUBLE, MPI_MAX, world);
  if (bmax_all == 0.0) return;

  const double zprd = domain->prd[dim];
  auto *fref = new double[nlocal > 0 ? nlocal : 1];
  auto *fb = new double[nlocal > 0 ? nlocal : 1];

  corr_raw_force(fref);    // exact target (once)

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
    err = sqrt(sall / natoms);
    chosen = nb;
    if (err < accuracy) break;                          // target met
    if (prev_err > 0.0 && err > 0.7 * prev_err) {       // diminishing returns: at the floor
      chosen = prev_nb;
      err = prev_err;
      break;
    }
    if (nb >= nb_cap) break;
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
                     "ewald/disp/slab corr bin did not reach the target force accuracy {:.3g} "
                     "(reached {:.3g}); use kspace_modify corr raw for tighter accuracy",
                     accuracy, err);
  }

  delete[] fref;
  delete[] fb;
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
   complementary chain  C[m] = A[m](inf) - A[m],  D[m] = B[m](inf) - B[m],
   i.e. the *small* tail parts that the reciprocal coefficients actually need:
     (pi/48  - A[5])              = C[5]
     (pi/288 - A[7] + B[6])       = C[7] - D[6]
     (pi/72  - A[5] + 2A[7] -2B[6]) = C[5] - 2C[7] + 2D[6]
   Computing them via this recurrence (seeded by C[1]=pi/2-Si, D[1]=-Ci) avoids
   the catastrophic A[m] = A[m](inf) + (tiny) cancellation, so the high-k
   coefficients (and forces/virial) no longer hit the ~1e-6 roundoff floor.
------------------------------------------------------------------------- */

void EwaldDispSlab::sici_compl_chain(double x, double *Carr, double *Darr)
{
  double si, ci;
  cisi(x, si, ci);
  Carr[1] = MY_PI / 2.0 - si;    // A[1](inf) - A[1] = pi/2 - Si(x)
  Darr[1] = -ci;                 // B[1](inf) - B[1] = -gamma - (Ci - gamma) = -Ci(x)
  const double sx = sin(x), cx = cos(x);
  for (int m = 2; m <= 7; m++) {
    const double xm = pow(x, 1 - m);
    Carr[m] = (Darr[m - 1] + sx * xm) / (m - 1);
    Darr[m] = (cx * xm - Carr[m - 1]) / (m - 1);
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
  memory->create(GN, kmax, "ewald/disp/slab:GN");
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
  memory->destroy(GN);
  memory->destroy(sfacrl);
  memory->destroy(sfacim);
  memory->destroy(sfacrl_all);
  memory->destroy(sfacim_all);
  GU = GF = GT = GN = nullptr;
  sfacrl = sfacim = sfacrl_all = sfacim_all = nullptr;
}

/* ---------------------------------------------------------------------- */

double EwaldDispSlab::memory_usage()
{
  double bytes = 8.0 * kmax * sizeof(double);    // GU,GF,GT,GN,sfacrl/im,sfacrl/im_all
  bytes += (double) nmax * sizeof(double);
  bytes += 2.0 * (double) kmax * nmax * sizeof(double);
  if (damp_flag == 2) bytes += 4.0 * (nwgrid + 1) * sizeof(double);    // shell kernels
  if (profile_flag) bytes += 2.0 * (double) npro * sizeof(double);
  return bytes;
}
