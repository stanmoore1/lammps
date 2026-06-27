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
   Planar Ewald summation for 1/r^6 dispersion interactions in systems whose
   mean density varies in one (z) direction only -- e.g. planar liquid-vapor
   interfaces.  The long-range dispersion energy is evaluated as a 1-D Fourier
   sum over z wavevectors h_n = 2*pi*n/Lz of the dispersion-weighted structure
   factor, with analytic x-y tail corrections.

   The 1/r^6 dispersion is split at the inner cutoff rcut by a C3 (septic)
   smoothstep S(r) over the shell [rcut, rcut+Delta]: S(r)*u(r) is the smooth
   long-range part fed to the reciprocal sum (it vanishes inside rcut, so there
   is no real-space slab correction, and it is C3-continuous at rcut, so the
   z-Fourier coefficients decay as ~k^-5 with no Gibbs ringing).  The matched
   pair style lj/cut/dispplanar evaluates the full LJ to rcut+Delta; a shell
   correction (corr_shell()) subtracts the reciprocal sum's plane mean-field
   S*u over the shell so the pair supplies the exact 3-D shell interaction.

   References: S. Moore, dissertation (BYU); this paper.
------------------------------------------------------------------------- */

#include "ewald_disp_planar.h"

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

EwaldDispPlanar::EwaldDispPlanar(LAMMPS *lmp) :
    KSpace(lmp), GU(nullptr), GF(nullptr), GT(nullptr), GN(nullptr), ek(nullptr), peatom(nullptr),
    sfacrl(nullptr), sfacim(nullptr), sfacrl_all(nullptr), sfacim_all(nullptr), cs(nullptr),
    sn(nullptr), B(nullptr)
{
  dispersionflag = 1;
  dim = 2;
  lat1 = 0;
  lat2 = 1;
  mix_flag = 0;
  nchan = 1;
  corr_mode = 0;
  bin_dz_user = 0.0;
  sw_width = 0.0;
  wEgrid = wFgrid = wTgrid = wNgrid = nullptr;
  nwgrid = 0;
  wdz = 0.0;
  kmax = 0;
  kcount = 0;
  kmax_created = 0;
  kmax_user = 0;
  nmax = 0;
  accuracy_relative = 0.0;
}

/* ---------------------------------------------------------------------- */

EwaldDispPlanar::~EwaldDispPlanar()
{
  deallocate();
  memory->destroy(ek);
  memory->destroy(peatom);
  memory->destroy(cs);
  memory->destroy(sn);
  delete[] B;
  delete[] wEgrid;
  delete[] wFgrid;
  delete[] wTgrid;
  delete[] wNgrid;
}

/* ---------------------------------------------------------------------- */

void EwaldDispPlanar::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal kspace_style {} command", force->kspace_style);
  accuracy_relative = fabs(utils::numeric(FLERR, arg[0], false, lmp));
  if (accuracy_relative > 1.0)
    error->all(FLERR, "Invalid relative accuracy {:g} for kspace_style {}", accuracy_relative,
               force->kspace_style);
}

/* ----------------------------------------------------------------------
   handle the per-style kspace_modify keywords:
     kmax <N>          -- override the number of z wavevectors
     corr raw|bin [dz] -- shell correction: exact pairwise, or z-binned (faster)
     (the local pressure profile is the Irving-Kirkwood contour; the Harasima contour is
      the per-atom virial, available via compute stress/atom + fix ave/chunk)
     dim x|y|z         -- the inhomogeneous direction (default z)
   returns number of args consumed (0 -> base errors on unknown keyword)
------------------------------------------------------------------------- */

int EwaldDispPlanar::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0], "kmax") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify kmax", error);
    kmax_user = utils::inumeric(FLERR, arg[1], false, lmp);
    if (kmax_user < 2) error->all(FLERR, "kspace_modify kmax must be >= 2");
    return 2;
  }
  // mix/disp is parsed by the base KSpace::modify_params (sets mixflag); see init_coeffs.
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

void EwaldDispPlanar::init()
{
  if (comm->me == 0) utils::logmesg(lmp, "Planar dispersion Ewald (ewald/disp/planar) ...\n");

  // error checks

  triclinic_check();
  if (domain->dimension == 2) error->all(FLERR, "Cannot use ewald/disp/planar with 2d simulation");
  if (domain->triclinic) error->all(FLERR, "Cannot use ewald/disp/planar with triclinic box");
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR, "ewald/disp/planar requires periodic boundaries in all dimensions");
  if (slabflag)
    error->all(FLERR, "Cannot use slab correction (kspace_modify slab) with ewald/disp/planar");

  // ewald/disp/planar pairs with the matched lj/cut/dispplanar pair style: the
  // pair computes the full LJ to rcut+Delta and this kspace adds the r>rcut tail
  // of the C3-switched 1/r^6.  Validate the coupling via extract() below.

  if (force->pair == nullptr)
    error->all(FLERR, "KSpace style ewald/disp/planar requires a pair style");

  // extract the LJ cutoff and dispersion amplitudes B from the pair style

  int itmp;
  double *p_cutoff = (double *) force->pair->extract("cut_lj", itmp);
  if (p_cutoff == nullptr) p_cutoff = (double *) force->pair->extract("cut_LJ", itmp);
  if (p_cutoff == nullptr)
    error->all(FLERR, "Pair style is incompatible with kspace_style ewald/disp/planar");
  cutoff = *p_cutoff;
  rc2 = cutoff * cutoff;

  // the matched pair style supplies the switch width Delta and evaluates the full
  // dispersion (repulsion + 1/r^6) over the shell [rcut, rcut+Delta] (exact 3-D);
  // corr_shell() below removes the reciprocal sum's plane mean-field S*u there, so
  // the pair supplies the laterally-correlated shell interaction.  The pair's
  // interaction cutoff is rcut+Delta; "cut_lj" above is the inner rcut.

  {
    int itmp2;
    double *p_dz = (double *) force->pair->extract("disp_switch_width", itmp2);
    if (p_dz == nullptr)
      error->all(FLERR,
                 "kspace_style ewald/disp/planar requires a pair style that provides the dispersion "
                 "switch width (use pair_style lj/cut/dispplanar)");
    sw_width = *p_dz;
    if (sw_width <= 0.0) error->all(FLERR, "ewald/disp/planar switch width must be > 0");
  }

  // accuracy in force units

  two_charge();
  if (accuracy_absolute >= 0.0)
    accuracy = accuracy_absolute;
  else
    accuracy = accuracy_relative * two_charge_force;

  // choose the number of z wavevectors kmax from the target accuracy (unless the
  // user set it).  init_coeffs() first so the dispersion amplitudes B are available.

  init_coeffs();
  estimate_params();

  setup();

  if (comm->me == 0) {
    utils::logmesg(lmp, "  planar dispersion Ewald, {} z wavevectors\n", kmax);
    utils::logmesg(lmp, "  switch width Delta = {:.6g}\n", sw_width);
    utils::logmesg(lmp, "  estimated absolute RMS force accuracy = {:.6g}\n",
                   estimated_force_accuracy);
    utils::logmesg(lmp, "  estimated relative force accuracy = {:.6g}\n",
                   estimated_force_accuracy / two_charge_force);
  }
}

/* ----------------------------------------------------------------------
   force coefficient GF for a single z mode k (k>=1); requires volume, unitk,
   cutoff and the switch width to be set
------------------------------------------------------------------------- */

double EwaldDispPlanar::gf_of_k(int k)
{
  // compact switch: force is the exact z-gradient of the energy term, GF=2k*GU
  const double kcell = k * unitk;
  return 2.0 * kcell * gu_switch(k);
}

/* ----------------------------------------------------------------------
   estimate kmax from the target force accuracy.

   RMS per-atom force truncation error (random-phase model, validated against
   measured forces):  rms(kmax) = sqrt(0.5 * b2^2 / N * sum_{k>kmax} GF[k]^2),
   with b2 = sum_i B_i^2.  Pick the smallest kmax with rms < accuracy (the model
   slightly under-predicts for correlated/interfacial systems, folded into bias).
------------------------------------------------------------------------- */

void EwaldDispPlanar::estimate_params()
{
  lat1 = (dim + 1) % 3;
  lat2 = (dim + 2) % 3;
  volume = domain->prd[0] * domain->prd[1] * domain->prd[2];
  unitk = 2.0 * MY_PI / domain->prd[dim];

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
  // B_t = 2 sqrt(eps_tt) sigma_tt^3 = sqrt(C6_tt); independent of the mixing rule, so
  // this kmax estimate is the same for the geometric and arithmetic B layouts.
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

  // (a user-set kmax still gets a predicted RMS force error reported below)

  // cumulative tail of GF[k]^2 from a large cap; pick kmax for target accuracy

  const int kbig = 8192;
  const double prefac = 0.5 * b2 * b2 / natoms;
  // The random-phase model rms = sqrt(prefac * sum_{k>kmax} GF[k]^2) under-predicts
  // the true per-atom force error by a roughly constant factor (the cross-term /
  // diagonal-approximation contribution; |S_k|^2 -> b2 in the relevant high-k tail).
  // For the C3 septic switch this factor is ~1.4; fold in ~1.6 (the chosen kmax then
  // meets the requested accuracy within ~1.5x) and select with no extra margin.
  const double bias = 1.6;
  const double target = accuracy * accuracy / (bias * bias);

  // sum GF^2 from the top down so tail(kmax) = sum_{k>kmax}.
  // The compact-switch GF decays only ALGEBRAICALLY (~k^-5) past the switch
  // bandwidth, so the tail is not negligible relative to the *peak* -- only
  // once its remaining sum falls below the tightest tail the target accuracy
  // needs.  Compute upward and stop when the estimated remaining tail
  // (Sum_{j>k} gf2 ~ gf2[k]*k/9 for a ~k^-10 spectrum) is a small fraction of
  // that target tail; the genuinely-negligible remainder is zeroed.  C[m]/D[m]
  // keep the high-k gf2 free of cancellation roundoff so the summed tail is real.
  auto *gf2 = new double[kbig + 1];
  {
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
}

/* ----------------------------------------------------------------------
   adjust coefficients, called initially and whenever the volume changes
------------------------------------------------------------------------- */

void EwaldDispPlanar::setup()
{
  volume = domain->prd[0] * domain->prd[1] * domain->prd[2];
  unitk = 2.0 * MY_PI / domain->prd[dim];

  // set the mixing rule / channel count (nchan) before allocating the structure
  // factors, whose size is kmax*nchan
  init_coeffs();

  deallocate();
  allocate();

  if (atom->nmax > nmax) {
    memory->destroy(ek);
    memory->destroy(peatom);
    memory->destroy(cs);
    memory->destroy(sn);
    nmax = atom->nmax;
    memory->create(ek, nmax, "ewald/disp/planar:ek");
    memory->create(peatom, nmax, "ewald/disp/planar:peatom");
    memory->create(cs, kmax, nmax, "ewald/disp/planar:cs");
    memory->create(sn, kmax, nmax, "ewald/disp/planar:sn");
    kmax_created = kmax;
  } else if (kmax != kmax_created) {
    memory->destroy(cs);
    memory->destroy(sn);
    memory->create(cs, kmax, nmax, "ewald/disp/planar:cs");
    memory->create(sn, kmax, nmax, "ewald/disp/planar:sn");
    kmax_created = kmax;
  }

  init_coeffs();
  coeffs();
  build_shell_vkernels();
}

/* ----------------------------------------------------------------------
   extract per-type dispersion amplitude B[i] = sqrt(|lj4[i][i]|) = 2*sqrt(eps)*sigma^3
------------------------------------------------------------------------- */

void EwaldDispPlanar::init_coeffs()
{
  int tmp;
  int n = atom->ntypes;

  // select the dispersion mixing rule for the C6 cross term.  By default follow the
  // pair style's rule (Pair::mix_flag, read via the pair's ewald_mix extract:
  // GEOMETRIC=0 or ARITHMETIC=1).  kspace_modify mix/disp overrides it through the
  // base-class flag KSpace::mixflag (0 = pair/follow, 1 = geom/force geometric,
  // 2 = none).  Only geometric and arithmetic (Lorentz-Berthelot) are supported; the
  // eigenvalue-split "none" rule of pppm/disp does not apply to the planar single-axis
  // 1/r^6 sum.  Request arithmetic mixing with pair_modify mix arithmetic (matching
  // upstream ewald/disp and pppm/disp, which also take it from the pair).

  int *p_mix = (int *) force->pair->extract("ewald_mix", tmp);
  int pair_mix = p_mix ? *p_mix : Pair::GEOMETRIC;
  if (mixflag == 1) {    // kspace_modify mix/disp geom: force geometric
    mix_flag = 0;
  } else if (mixflag == 2) {    // kspace_modify mix/disp none
    error->all(FLERR,
               "kspace_modify mix/disp none is not supported by ewald/disp/planar; use "
               "geometric or arithmetic mixing (pair_modify mix geometric|arithmetic)");
  } else {    // mixflag 0 (default): follow the pair style
    if (pair_mix == Pair::GEOMETRIC)
      mix_flag = 0;
    else if (pair_mix == Pair::ARITHMETIC)
      mix_flag = 1;
    else
      error->all(FLERR,
                 "Unsupported pair mixing rule for kspace_style ewald/disp/planar "
                 "(use pair_modify mix geometric|arithmetic)");
  }
  nchan = mix_flag ? 7 : 1;

  delete[] B;

  if (mix_flag == 0) {    // geometric: single per-type amplitude B[i]=sqrt(|lj4[i][i]|)
    auto **b = (double **) force->pair->extract("B", tmp);
    if (b == nullptr)
      error->all(FLERR,
                 "Pair style does not provide dispersion coefficient B for ewald/disp/planar");
    B = new double[n + 1];
    B[0] = 0.0;
    for (int i = 1; i <= n; ++i) B[i] = sqrt(fabs(b[i][i]));
  } else {    // arithmetic (Lorentz-Berthelot): 7-channel binomial expansion
    auto **epsilon = (double **) force->pair->extract("epsilon", tmp);
    auto **sigma = (double **) force->pair->extract("sigma", tmp);
    if (!(epsilon && sigma))
      error->all(FLERR,
                 "Pair style does not provide epsilon/sigma for arithmetic mixing in "
                 "ewald/disp/planar");
    B = new double[7 * n + 7];

    // the seven per-type coefficients of the binomial expansion of
    // (0.5*(sigma_i+sigma_j))^6 : sqrt(eps_i)*c[j]*sigma_i^j, j = 0..6.  The cross
    // amplitude is C6_ij = 4 sqrt(eps_i eps_j) ((sigma_i+sigma_j)/2)^6 =
    // sum_{j=0}^6 B[7*i+j] * B[7*j_type+(6-j)].  For a single type this reduces to
    // C6_ii = 4 eps_i sigma_i^6 = (2 sqrt(eps_i) sigma_i^3)^2, identical to the
    // geometric path's B[i]^2, so single-type results are bit-identical.

    const double c[7] = {1.0, sqrt(6.0), sqrt(15.0), sqrt(20.0), sqrt(15.0), sqrt(6.0), 1.0};
    for (int j = 0; j < 7; ++j) B[j] = 0.0;    // type 0 (unused) channels
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

/* ----------------------------------------------------------------------
   pre-compute the reciprocal-space coefficients for each z wavevector
------------------------------------------------------------------------- */

void EwaldDispPlanar::coeffs()
{
  int k;
  double kcell, kcell3;

  kcount = kmax;

  {

    // compact switch: smoothed truncation over [rcut, rcut+Delta].  The
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
    // corr_shell() and replaced by the matched pair's exact 3-D shell, so the TOTAL
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
  }
}

/* ----------------------------------------------------------------------
   compact-switch smoothstep S(t), the C3 (septic) "smootherstep" on t in [0,1]:
   S(0)=0, S(1)=1 with the first three derivatives zero at both ends (C3).
   In r: S=0 for r<=rcut, S=1 for r>=rcut+Delta.  The long-range part fed to the
   reciprocal sum is S(r)*u(r); it vanishes inside rcut (so no slab correction)
   and meets r>=rcut with C3 continuity, so the z-Fourier coefficients decay as
   ~k^-5 (no Gibbs ringing).
------------------------------------------------------------------------- */

double EwaldDispPlanar::switch_S(double t)
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t, t3 = t2 * t, t4 = t3 * t;    // C3 septic smoothstep
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

double EwaldDispPlanar::switch_trans5(double h)
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

double EwaldDispPlanar::switch_dS(double t)
{
  // dS/dt of the C3 septic smoothstep = 7!/(3!)^2 (t(1-t))^3 = 140 (t(1-t))^3
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double tu = t * (1.0 - t);
  return 140.0 * tu * tu * tu;
}

/* ----------------------------------------------------------------------
   shell virial integrals over [rcut, rcut+Delta]:
     sGT = int phi'(r) A_T(r,h) dr,   sGN = int phi'(r) A_N(r,h) dr,
   with the FULL switched-dispersion force phi'(r) = (S u)'(r) = -S'(r)/r^6 +
   6 S(r)/r^7 -- i.e. the consistent strain derivative of the energy functional
   sum_k GU[k]|S_k|^2 (the S'(r)u "switch-force" term is INCLUDED).  This plane
   mean field over the shell is what corr_shell() then subtracts and replaces with
   the matched pair's exact 3-D shell virial, so the residual is removed by the
   real-space correction, not by dropping the S'u term here.  Angular factors:
     A_T = -4 r cos(hr)/h^2 + 4 sin(hr)/h^3,
     A_N =  2 r^2 sin(hr)/h + 4 r cos(hr)/h^2 - 4 sin(hr)/h^3.
   GT = GT_tail - (pi/V) sGT, GN = GN_tail - (2 pi/V) sGN.
------------------------------------------------------------------------- */

void EwaldDispPlanar::switch_shell_virial(double h, double &sGT, double &sGN)
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

double EwaldDispPlanar::gu_switch(int k)
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

double EwaldDispPlanar::gu0_switch()
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
------------------------------------------------------------------------- */

void EwaldDispPlanar::build_shell_vkernels()
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

void EwaldDispPlanar::shell_vkernel(double adz, double &wE, double &wF, double &wT, double &wN)
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

void EwaldDispPlanar::corr_shell()
{
  if (corr_mode == 1)
    corr_shell_bin();
  else
    corr_shell_raw();
}

/* ----------------------------------------------------------------------
   exact (global z-gather) subtraction of the plane (mean-field) shell energy,
   z-force and virial.  Every proc gathers the global (z, B)
   list and each local atom sums the plane kernel over all global atoms in its
   |dz| < rcut+Delta window (slab-slab).  Removes what the reciprocal sum put in
   the shell with a laterally-uniform density so the matched pair's exact 3-D
   shell interaction (full u to rcut+Delta) is what remains.  Matches the kspace
   |S_k|^2 convention: full ordered double sum incl. self, so the energy/virial
   carry no 1/2; the z-force = -d E/d z_i differentiates both pair indices and so
   carries a factor 2 (as the reciprocal GF[k]=2k GU[k] force does).
------------------------------------------------------------------------- */

void EwaldDispPlanar::corr_shell_raw()
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
  // amplitude is then sum_m a_i[m] a_j[6-m] / 16 (the full ordered binomial sum
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
  // channel counts/displacements (scaled by nchan) for the B-channel gather
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
   z-binned version of the shell virial correction (1D particle-mesh, CIC).
   Bins the B-weighted density, convolves with the plane kernels, interpolates
   back.  O(nbins*nwin)+O(N) instead of O(N*N_slice).
------------------------------------------------------------------------- */

void EwaldDispPlanar::corr_shell_bin()
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
    double u = (x[i][dim] - zlo) / dz;
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
  // full ordered convention, no 1/2); subtracted.  For nchan==1 this is sum_b dens phiE.
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
  // sum, matching the reciprocal GF[k]=2k GU[k]) and per-atom energy/virial.  Atom i
  // contributes channel m and pairs with the field of channel (6-m).
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
   1-D dispersion-weighted structure factors S(h_n) = sum_j B_j exp(-i h_n z_j)
------------------------------------------------------------------------- */

void EwaldDispPlanar::eik_dot_r()
{
  int i, k;
  double **x = atom->x;
  int nlocal = atom->nlocal;
  int *type = atom->type;

  memset(sfacrl, 0, kcount * nchan * sizeof(double));
  memset(sfacim, 0, kcount * nchan * sizeof(double));

  if (nchan == 1) {    // geometric: single B-weighted structure factor per mode

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

  } else {    // arithmetic: 7 channel structure factors S_m = sum_i B[7*t_i+m] e^{-i h z_i}

    for (i = 0; i < nlocal; i++) {
      const double *bi = &B[7 * type[i]];

      cs[0][i] = 1.0;
      sn[0][i] = 0.0;
      for (int m = 0; m < 7; m++) sfacrl[0 * 7 + m] += bi[m];

      if (kcount > 1) {
        cs[1][i] = cos(unitk * x[i][dim]);
        sn[1][i] = sin(unitk * x[i][dim]);
        for (int m = 0; m < 7; m++) {
          sfacrl[1 * 7 + m] += bi[m] * cs[1][i];
          sfacim[1 * 7 + m] += bi[m] * sn[1][i];
        }
      }

      for (k = 2; k < kcount; k++) {
        cs[k][i] = cs[k - 1][i] * cs[1][i] - sn[k - 1][i] * sn[1][i];
        sn[k][i] = sn[k - 1][i] * cs[1][i] + cs[k - 1][i] * sn[1][i];
        for (int m = 0; m < 7; m++) {
          sfacrl[k * 7 + m] += bi[m] * cs[k][i];
          sfacim[k * 7 + m] += bi[m] * sn[k][i];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute the slab-based dispersion long-range force, energy, virial
------------------------------------------------------------------------- */

void EwaldDispPlanar::compute(int eflag, int vflag)
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
    memory->create(ek, nmax, "ewald/disp/planar:ek");
    memory->create(peatom, nmax, "ewald/disp/planar:peatom");
    memory->create(cs, kmax, nmax, "ewald/disp/planar:cs");
    memory->create(sn, kmax, nmax, "ewald/disp/planar:sn");
    kmax_created = kmax;
  }

  // partial structure factors per proc, then global total

  eik_dot_r();
  MPI_Allreduce(sfacrl, sfacrl_all, kcount * nchan, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(sfacim, sfacim_all, kcount * nchan, MPI_DOUBLE, MPI_SUM, world);

  double **f = atom->f;
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double exprl, expim, partial, partial_peratom;

  for (i = 0; i < nlocal; i++) ek[i] = 0.0;
  if (evflag_atom)
    for (i = 0; i < nlocal; i++) peatom[i] = 0.0;

  double e_recip = 0.0;

  if (nchan == 1) {    // geometric: single B-weighted structure factor per mode

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
            // tangential from GT, normal (dim) from the explicit GN kernel
            vatom[i][lat1] += GT[k] * partial_peratom;
            vatom[i][lat2] += GT[k] * partial_peratom;
            vatom[i][dim] += GN[k] * partial_peratom;
          }
        }
      }
    }

    // reciprocal z-force on each atom (scaled by its own B)

    for (i = 0; i < nlocal; i++) f[i][dim] += B[type[i]] * ek[i];

    // reciprocal energy (full system value, identical on every proc); always
    // evaluated when the virial is needed (the zz trace uses it)

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
        virial[dim] += uk * GN[k];    // explicit normal kernel
      }
    }

    // scale per-atom energy buffer / virial by each atom's B

    if (evflag_atom)
      for (i = 0; i < nlocal; i++) peatom[i] *= B[type[i]];
    if (vflag_atom)
      for (i = 0; i < nlocal; i++) {
        vatom[i][lat1] *= B[type[i]];
        vatom[i][lat2] *= B[type[i]];
        vatom[i][dim] *= B[type[i]];
      }

  } else {    // arithmetic (Lorentz-Berthelot): 7-channel cross pairing

    // For each mode k the seven global structure factors S_0..S_6 (channels) pair
    // as R_k = (sr0 sr6 + si0 si6) + (sr1 sr5 + si1 si5) + (sr2 sr4 + si2 si4)
    //        + 0.5 (sr3^2 + si3^2), reproducing the LB cross amplitude.  Per atom i
    // and mode k the z-force pairs atom channel (6-m) with global channel m; the
    // per-atom energy/virial carry the same channel pairing with a 0.5 (the global
    // value sums to GU[k] R_k).  No 0.5 in the force (it differentiates both indices).
    //
    // The channel amplitudes B[7t+j]=sigma^j sqrt(eps) c[j] expand
    // (sigma_i+sigma_j)^6, so the folded channel pairing R_k reproduces the C6 cross
    // double sum times 8 (= 16 from the (sigma_i+sigma_j)^6 vs C6_ij=.../16 binomial
    // normalization, halved again by the folding of the m<->6-m pairs).  GU/GF/GT/GN
    // carry the geometric (B_i B_j == C6_ij) normalization, so the arithmetic
    // reciprocal channels are scaled by AS = 1/8 to recover the same convention.
    // (For a single type R_k/8 == |S_k(geom)|^2 exactly: LB cross == geometric cross.)

    // Energy/virial use the folded channel pairing R_k (each m<->6-m pair counted
    // once), normalized by AS_E = 1/8.  The z-force is the exact gradient of that
    // energy: differentiating the bilinear R_k w.r.t. z_i restores the dropped factor
    // of 2 (both channels of each pair contain atom i), so the force uses AS_F = 1/16
    // = AS_E/2 with the FULL ordered channel sum bi[6-m]*S_m.  (For a single type
    // this gives exactly B^2 sum_k GF[k](sn S_re - cs S_im), the geometric z-force.)
    const double as_e = 0.125;       // 1/8  energy / virial normalization
    const double as_f = 1.0 / 16.0;  // 1/16 z-force normalization (= as_e/2)

    for (i = 0; i < nlocal; i++) {
      const double *bi = &B[7 * type[i]];
      double fz_i = 0.0, pe_i = 0.0, pT_i = 0.0, pN_i = 0.0;
      for (k = 0; k < kcount; k++) {
        const double ci = cs[k][i], si = sn[k][i];
        const double *sr = &sfacrl_all[k * 7];
        const double *sm = &sfacim_all[k * 7];
        double fsum = 0.0, esum = 0.0;
        for (int m = 0; m < 7; m++) {
          const double a = bi[6 - m];    // atom channel (6-m) pairs with global m
          fsum += a * (si * sr[m] - ci * sm[m]);
          if (evflag_atom) esum += a * (ci * sr[m] + si * sm[m]);
        }
        fz_i += GF[k] * fsum;
        if (evflag_atom) {
          pe_i += GU[k] * 0.5 * esum;
          if (vflag_atom) {
            pT_i += GT[k] * 0.5 * esum;
            pN_i += GN[k] * 0.5 * esum;
          }
        }
      }
      f[i][dim] += as_f * fz_i;
      if (evflag_atom) {
        peatom[i] = as_e * pe_i;
        if (vflag_atom) {
          vatom[i][lat1] += as_e * pT_i;
          vatom[i][lat2] += as_e * pT_i;
          vatom[i][dim] += as_e * pN_i;
        }
      }
    }

    // global reciprocal energy / virial from the per-mode channel pairing R_k
    if (eflag_global || vflag_global) {
      for (k = 0; k < kcount; k++) {
        const double *sr = &sfacrl_all[k * 7];
        const double *sm = &sfacim_all[k * 7];
        const double R = (sr[0] * sr[6] + sm[0] * sm[6]) + (sr[1] * sr[5] + sm[1] * sm[5]) +
            (sr[2] * sr[4] + sm[2] * sm[4]) + 0.5 * (sr[3] * sr[3] + sm[3] * sm[3]);
        e_recip += GU[k] * R;
        if (vflag_global) {
          virial[lat1] += as_e * R * GT[k];
          virial[lat2] += as_e * R * GT[k];
          virial[dim] += as_e * R * GN[k];    // explicit normal kernel
        }
      }
    }
    e_recip *= as_e;
    if (eflag_global) energy += e_recip;
  }

  // compact-switch shell correction.  The reciprocal sum treats the shell
  // [rcut, rcut+Delta] with a laterally-uniform density (plane mean field), which
  // leaves a lateral-correlation residual in energy AND pressure that grows with
  // Delta.  corr_shell() subtracts that plane mean field (energy, z-force, virial)
  // so the matched pair -- which now evaluates the FULL dispersion u to rcut+Delta
  // with exact 3-D correlation -- supplies the shell interaction instead.  Must run
  // every step (the z-force is removed unconditionally, else it is double counted).
  // The normal (zz) virial is the explicit GN kernel accumulated above (the switch
  // is non-homogeneous, so the trace identity 6U = sum r.f does not apply).
  corr_energy = 0.0;
  corr_shell();

  // report per-atom energy (from the buffer) when requested
  if (eflag_atom)
    for (i = 0; i < nlocal; i++) eatom[i] += peatom[i];
}

/* ----------------------------------------------------------------------
   potential-form integrand g(r) of a profile coefficient: the sharp coefficient
   is int_rcut^inf g(r) dr.  (x = h r)
     PROF_T   (combo_GT, Tn & Psi):  sin(hr)/(h^6 r^7) - cos(hr)/(h^5 r^6)
     PROF_N   (combo_GN, Nn):        sin(hr)/(h^4 r^5) - 2 sin(hr)/(h^6 r^7)
                                       + 2 cos(hr)/(h^5 r^6)
     PROF_PHI (combo_phi, Phi):      Si(hr)/(h^4 r^5) - sin(hr)/(h^6 r^7)
                                       + cos(hr)/(h^5 r^6)
------------------------------------------------------------------------- */

double EwaldDispPlanar::prof_integrand(int which, double r, double h)
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
   global switch_shell_virial; W -> S as Delta->0 with S'->delta at rcut so the
   shell shrinks to nothing and the sharp result is recovered).  10-point
   Gauss-Legendre, panel count scaled to the oscillation count h*Delta (~1e-12).
------------------------------------------------------------------------- */

double EwaldDispPlanar::prof_shell(int which, double h)
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
   IK tangential building block Phi(h) = sgn(h)|h|^4 [pi/576 - Sii5 + Si7 - Ci6]
   (the IK normal uses Psi(h) = sgn(h)|h|^4 [pi/288 - Si7 + Ci6]).
   Compact-switch aware: the closed-form tail combo is evaluated at the OUTER
   cutoff rcut+Delta and the switch-shell integral prof_shell(...) is added, so
   Phi/Psi are consistent with the switched potential S(r)/r^6 (sharp as Delta->0).
------------------------------------------------------------------------- */

double EwaldDispPlanar::ik_phi(double h)
{
  if (fabs(h) < 1.0e-300) return 0.0;
  const double ah = fabs(h);
  const double c = cutoff + sw_width;
  double A[8], Bc[8];
  sici_chain(ah * c, A, Bc);    // tail anchored at the outer cutoff
  const double sii5 = A[5] / 4.0 - A[1] / (4.0 * pow(ah * c, 4));
  const double phi = MY_PI / 576.0 - sii5 + A[7] - Bc[6] + prof_shell(PROF_PHI, ah);
  const double ah4 = ah * ah * ah * ah;
  return (h >= 0.0 ? 1.0 : -1.0) * ah4 * phi;
}

/* ---------------------------------------------------------------------- */

double EwaldDispPlanar::ik_psi(double h)
{
  if (fabs(h) < 1.0e-300) return 0.0;
  const double ah = fabs(h);
  const double c = cutoff + sw_width;
  double A[8], Bc[8];
  sici_chain(ah * c, A, Bc);    // tail anchored at the outer cutoff
  const double psi = MY_PI / 288.0 - A[7] + Bc[6] + prof_shell(PROF_T, ah);
  const double ah4 = ah * ah * ah * ah;
  return (h >= 0.0 ? 1.0 : -1.0) * ah4 * psi;
}

/* ----------------------------------------------------------------------
   shell-correction virial per profile bin, dispatched on corr_mode so the contour
   profile uses the IDENTICAL real-space corr_shell correction as the box average:
     corr raw (0): each atom's EXACT shell virial vt_i = sum_j bij wT(|z_i-z_j|),
       vn_i = sum_j bij wN, binned by z_i; sum over bins == vt_all/vn_all exactly, so
       box-avg(profile) == box pressure independent of nbins.
     corr bin (1): density-density convolution (the binned approximation).
   geometric (nchan==1): bij = B_i B_j.  arithmetic (nchan==7): bij = (1/16) sum_m
   a_i[m] a_j[6-m] (the same C6 cross used by corr_shell_raw).
------------------------------------------------------------------------- */

void EwaldDispPlanar::shell_profile_virial(int nbins, double lo, double dz, double *dens_all,
                                           double *shellT, double *shellN)
{
  const double zprd = domain->prd[dim];
  const double bcut = cutoff + sw_width;
  for (int g = 0; g < nbins; g++) shellT[g] = shellN[g] = 0.0;

  if (corr_mode != 0) {    // BIN: density-density convolution (matches corr_shell_bin)
    for (int g = 0; g < nbins; g++) {
      double sT = 0.0, sN = 0.0;
      for (int gp = 0; gp < nbins; gp++) {
        double ddz = (g - gp) * dz;
        ddz -= zprd * floor(ddz / zprd + 0.5);
        double wE, wF, wT, wN;
        shell_vkernel(fabs(ddz), wE, wF, wT, wN);
        sT += dens_all[gp] * wT;
        sN += dens_all[gp] * wN;
      }
      shellT[g] = dens_all[g] * sT;
      shellN[g] = dens_all[g] * sN;
    }
    return;
  }

  // RAW: exact per-atom shell virial binned by z (matches corr_shell_raw, the default).
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
  MPI_Allgatherv(zloc, nlocal, MPI_DOUBLE, zall, rc, dp, MPI_DOUBLE, world);
  int *rcb = new int[nprocs];
  int *dpb = new int[nprocs];
  for (int p = 0; p < nprocs; p++) {
    rcb[p] = rc[p] * nchan;
    dpb[p] = dp[p] * nchan;
  }
  MPI_Allgatherv(bloc, nlocal * nchan, MPI_DOUBLE, ball, rcb, dpb, MPI_DOUBLE, world);

  const double as_shell = 1.0 / 16.0;
  auto *sTloc = new double[nbins];
  auto *sNloc = new double[nbins];
  for (int g = 0; g < nbins; g++) sTloc[g] = sNloc[g] = 0.0;
  for (int i = 0; i < nlocal; i++) {
    double zi = x[i][dim];
    double bi = (nchan == 1) ? B[type[i]] : 0.0;
    const double *ai = (nchan == 1) ? nullptr : &B[7 * type[i]];
    double u = (zi - lo) / dz;
    u -= nbins * floor(u / nbins);
    int g = (int) u;
    if (g >= nbins) g -= nbins;
    double vt = 0.0, vn = 0.0;
    for (int jg = 0; jg < natoms_all; jg++) {
      double delz = zi - zall[jg];
      delz -= zprd * floor(delz / zprd + 0.5);
      double adz = fabs(delz);
      if (adz >= bcut) continue;
      double wE, wF, wT, wN;
      shell_vkernel(adz, wE, wF, wT, wN);
      double bij;
      if (nchan == 1)
        bij = bi * ball[jg];
      else {
        const double *aj = &ball[jg * 7];
        double cross = 0.0;
        for (int m = 0; m < 7; m++) cross += ai[m] * aj[6 - m];
        bij = as_shell * cross;
      }
      vt += bij * wT;
      vn += bij * wN;
    }
    sTloc[g] += vt;
    sNloc[g] += vn;
  }
  MPI_Allreduce(sTloc, shellT, nbins, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(sNloc, shellN, nbins, MPI_DOUBLE, MPI_SUM, world);
  delete[] rc;
  delete[] dp;
  delete[] rcb;
  delete[] dpb;
  delete[] zloc;
  delete[] bloc;
  delete[] zall;
  delete[] ball;
  delete[] sTloc;
  delete[] sNloc;
}

/* ----------------------------------------------------------------------
   long-range Irving-Kirkwood pressure profiles P_T(z), P_N(z) on the caller's nbins z grid
   (bin centers lo+(g+0.5)*width); the caller allocates pT/pN.  Returns 1 on success.
   (The Harasima contour is the per-atom virial, obtained via compute stress/atom +
   fix ave/chunk; only the IK contour -- which cannot be written per-atom -- is here.)
     P(z) = sum_{n,m} S_n S_m C_{n,m} e^{i(h_n+h_m)z} - shell(z),
   p=n+m=0 coefficients pinned to the verified global GT/GN ((0,0)=V*GT[0], n=-m diagonal
   = V*GT[k]/2), off-diagonal C^{T}_{n,m}=-6pi/(h_n+h_m)[Phi(h_m)+Phi(h_n)],
   C^{N}_{n,m}=-12pi/(h_n+h_m)[Psi(h_m)+Psi(h_n)] (these set only the SHAPE).  The shell
   mean field uses the SAME corr_shell correction as the box average (corr_mode).
   S_n = (1/V) sum_j B_j e^{-i h_n z_j}.  Requires nbins > 2*kmax (anti-aliasing).
------------------------------------------------------------------------- */

int EwaldDispPlanar::pressure_profile_long(int dir, int nbins, double lo, double width,
                                           double *pN, double *pT)
{
  if (dir != dim)
    error->all(FLERR,
               "compute stress/cartesian binning direction must match the inhomogeneous axis "
               "(kspace_modify dim) of ewald/disp/planar");

  const double zprd = domain->prd[dim];
  const double area = domain->prd[lat1] * domain->prd[lat2];
  const int K = kcount - 1;    // highest mode index

  // anti-aliasing requirement: the Irving-Kirkwood profile sums reciprocal modes
  // e^{i p unitk z} with |p|=|n+m| up to 2*kmax.  The z grid must resolve them or the
  // high modes alias onto low ones and corrupt both the profile shape and its box-average
  // (which must equal the global pressure).  K = kcount-1 is the highest mode index, so
  // require nbins > 2K.
  if (nbins <= 2 * K)
    error->all(FLERR,
               "compute stress/cartesian with ewald/disp/planar kspace: {} bins along the "
               "inhomogeneous axis is too coarse; need > {} (= 2*kmax) to resolve the "
               "Irving-Kirkwood reciprocal modes without aliasing (use a finer bin width or "
               "smaller kmax)",
               nbins, 2 * K);

  // number-density Fourier coefficients S_n = (1/V)(sfacrl - i sfacim) for n>=0
  // (S_{-n} = conj(S_n)); store Sre[n], Sim[n] for n=0..K
  auto *Sre = new double[K + 1];
  auto *Sim = new double[K + 1];
  for (int n = 0; n <= K; n++) {
    Sre[n] = sfacrl_all[n] / volume;
    Sim[n] = -sfacim_all[n] / volume;
  }

  // bin the B-weighted density rho_B(z) -- the Harasima rho multiplier and the BIN-mode
  // shell convolution source; shared by both contours.
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

  // shell-correction VIRIAL per bin shellT[g]/shellN[g].  The contour profile MUST use
  // the SAME real-space corr_shell correction as the box average so box-avg(profile)
  // == box pressure; dispatch on corr_mode exactly like the global correction (raw =
  // exact per-atom shell virial binned by z, nbins-independent exact; bin = density
  // convolution).  The kspace NET long-range pressure is reciprocal - shell, so subtract
  // shellT[g]/(area*dz) (a pressure) from the reciprocal profile below.
  auto *shellT = new double[nbins];
  auto *shellN = new double[nbins];
  shell_profile_virial(nbins, lo, width, dens_all, shellT, shellN);
  const double inv_adz = 1.0 / (area * dz);

  {

    // Irving-Kirkwood pressure profile.  (The Harasima contour is just the per-atom
    // virial, obtained from compute stress/atom + fix ave/chunk -- not computed here.)
    // P(z) = sum_{n,m} S_n S_m C_{n,m} e^{i(h_n+h_m)z}.  Only the
    // p=n+m=0 terms survive the box-average, so they fix the integral (=> the global
    // pressure / surface tension), while p!=0 (the off-diagonal Phi/Psi kernels) set
    // only the SHAPE of the IK profile.  The box-average-relevant p=0 coefficients are
    // pinned to the verified global GT/GN exactly as the Harasima single-sum: the
    // (0,0) term -> V*GT[0], and each n=-m diagonal -> V*GT[k]/2 (the 1/2 since both
    // +n and -n are summed).  The compact-switch shell mean-field is laterally uniform
    // (contour-independent) so the SAME shellT/shellN field (from corr_mode) is subtracted.
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
        if (p < 0) continue;    // use Hermitian symmetry; keep p>=0
        double hm = m * unitk, H = hn + hm;
        double snr, sni, smr, smi;
        Sn(n, snr, sni);
        Sn(m, smr, smi);
        // S_n * S_m (complex)
        double sre = snr * smr - sni * smi, sim = snr * smi + sni * smr;
        double CT, CN;
        if (n == 0 && m == 0) {
          CT = CN = volume * GT[0];    // (0,0): box-average-pinned to global GT[0]=GN[0]
        } else if (fabs(H) < 1.0e-300) {    // n = -m diagonal: V*GT[k]/2, V*GN[k]/2
          int kk = (n < 0) ? -n : n;
          CT = 0.5 * volume * GT[kk];
          CN = 0.5 * volume * GN[kk];
        } else {    // off-diagonal: switch-aware Phi/Psi (sets the IK profile SHAPE)
          CT = -6.0 * MY_PI / H * (ik_phi(hm) + ik_phi(hn));
          CN = -12.0 * MY_PI / H * (ik_psi(hm) + ik_psi(hn));
        }
        ATre[p] += CT * sre;
        ATim[p] += CT * sim;
        ANre[p] += CN * sre;
        ANim[p] += CN * sim;
      }
    }
    for (int g = 0; g < nbins; g++) {
      double z = lo + (g + 0.5) * width;
      double pt = ATre[0], pn = ANre[0];    // p=0 term (real)
      for (int p = 1; p <= P; p++) {
        double cz = cos(p * unitk * z), sz = sin(p * unitk * z);
        pt += 2.0 * (ATre[p] * cz - ATim[p] * sz);    // Hermitian: +c.c.
        pn += 2.0 * (ANre[p] * cz - ANim[p] * sz);
      }
      // subtract the laterally-uniform shell mean field (same corr correction as box)
      pT[g] = pt - shellT[g] * inv_adz;
      pN[g] = pn - shellN[g] * inv_adz;
    }
    delete[] ATre;
    delete[] ATim;
    delete[] ANre;
    delete[] ANim;
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
   standard sine/cosine integrals Si(x), Ci(x)
     series for x <= 2, Lentz continued fraction (exp-integral) for x > 2
------------------------------------------------------------------------- */

void EwaldDispPlanar::cisi(double x, double &si, double &ci)
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

void EwaldDispPlanar::sici_chain(double x, double *Aarr, double *Barr)
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

void EwaldDispPlanar::sici_compl_chain(double x, double *Carr, double *Darr)
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

void EwaldDispPlanar::allocate()
{
  memory->create(GU, kmax, "ewald/disp/planar:GU");
  memory->create(GF, kmax, "ewald/disp/planar:GF");
  memory->create(GT, kmax, "ewald/disp/planar:GT");
  memory->create(GN, kmax, "ewald/disp/planar:GN");
  // structure factors: nchan channels per mode (1 geometric, 7 arithmetic),
  // flattened as sfacrl[k*nchan+m].  nchan is set by init_coeffs() which runs
  // first (see setup()); allocate 7 channels in the arithmetic case.
  memory->create(sfacrl, kmax * nchan, "ewald/disp/planar:sfacrl");
  memory->create(sfacim, kmax * nchan, "ewald/disp/planar:sfacim");
  memory->create(sfacrl_all, kmax * nchan, "ewald/disp/planar:sfacrl_all");
  memory->create(sfacim_all, kmax * nchan, "ewald/disp/planar:sfacim_all");
}

/* ---------------------------------------------------------------------- */

void EwaldDispPlanar::deallocate()
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

double EwaldDispPlanar::memory_usage()
{
  // GU,GF,GT,GN (kmax each) + sfacrl/im, sfacrl/im_all (kmax*nchan each)
  double bytes = (4.0 * kmax + 4.0 * kmax * nchan) * sizeof(double);
  bytes += (double) nmax * sizeof(double);
  bytes += 2.0 * (double) kmax * nmax * sizeof(double);
  bytes += 4.0 * (nwgrid + 1) * sizeof(double);    // shell kernels
  return bytes;
}
