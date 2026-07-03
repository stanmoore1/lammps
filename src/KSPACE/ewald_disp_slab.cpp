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

   Smooth-damped variant: Gaussian (erfc) smoothing like 3-D dispersion Ewald,
   matched to a lj/cut/dispswitch pair that fades the 1/r^6 dispersion out
   smoothly over [rcut, rcut+Delta].  The real-space slab correction (removing
   u_smooth inside rcut+Delta and adding back the faded S/r^6 shell) is a
   z-convolution of the dispersion-weighted density, diagonal in the reciprocal
   basis, so it folds directly into the per-mode coefficients GU/GF/GT/GN
   (merge_corr_coeffs) -- one reciprocal pass yields energy, force and the full
   pressure tensor with no separate real-space correction step.

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
  mix_flag = 0;
  nchan = 1;
  sw_width = 0.0;
  cWgrid = nullptr;
  cWraw = nullptr;
  ncgrid = 0;
  cwdz = 0.0;
  Araw_tab = Braw_tab = nullptr;
  nkap = 0;
  kap_dk = 0.0;
  kap_max = 0.0;
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
  delete[] cWgrid;
  delete[] cWraw;
  delete[] Araw_tab;
  delete[] Braw_tab;
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
     kmax <N>   -- override the number of z wavevectors
     dim x|y|z  -- select the inhomogeneous direction (default z)
   returns number of args consumed (0 -> base errors on unknown keyword)
------------------------------------------------------------------------- */

int EwaldDispSlab::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0], "kmax") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "kspace_modify kmax", error);
    kmax_user = utils::inumeric(FLERR, arg[1], false, lmp);
    if (kmax_user < 2) error->all(FLERR, "kspace_modify kmax must be >= 2");
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

  // ewald/disp/slab pairs with the matched lj/cut/dispswitch pair style: the pair
  // computes the full LJ to rcut and fades the 1/r^6 dispersion out over
  // [rcut, rcut+Delta]; this kspace adds the smooth r>rcut tail.

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

  // the matched lj/cut/dispswitch pair fades the 1/r^6 dispersion out over
  // [rcut, rcut+Delta], so the corr potential corr_e(r) = u_smooth(r) -
  // [r>rcut] S(r)/r^6 vanishes smoothly at rcut+Delta (no force discontinuity at
  // rcut).  Its Fourier content folds into the reciprocal coefficients
  // (merge_corr_coeffs), so no real-space correction step is needed.

  int itmp2;
  double *p_dz = (double *) force->pair->extract("disp_switch_width", itmp2);
  if (p_dz == nullptr || *p_dz <= 0.0)
    error->all(FLERR,
               "kspace_style ewald/disp/slab requires the matched lj/cut/dispswitch pair style "
               "to switch off the dispersion smoothly at the cutoff; use "
               "pair_style lj/cut/dispswitch <rcut> <Delta>");
  sw_width = *p_dz;

  // accuracy in force units

  two_charge();
  if (accuracy_absolute >= 0.0)
    accuracy = accuracy_absolute;
  else
    accuracy = accuracy_relative * two_charge_force;

  // choose the splitting parameter g_ewald and the number of z wavevectors kmax
  // from the target accuracy (unless the user set them).  init_coeffs() first so
  // the dispersion amplitudes B are available.

  init_coeffs();
  estimate_params();

  setup();

  if (comm->me == 0) {
    utils::logmesg(lmp, "  smooth-damped slab-based dispersion Ewald, {} z wavevectors\n", kmax);
    utils::logmesg(lmp, "  g_ewald = {:.6g}\n", g_ewald);
    utils::logmesg(lmp, "  switch width Delta = {:.6g}\n", sw_width);
    utils::logmesg(lmp, "  estimated absolute RMS force accuracy = {:.6g}\n",
                   estimated_force_accuracy);
    utils::logmesg(lmp, "  estimated relative force accuracy = {:.6g}\n",
                   estimated_force_accuracy / two_charge_force);
  }
}

/* ----------------------------------------------------------------------
   force coefficient GF for a single z mode k (k>=1); requires volume, unitk,
   cutoff and g_ewald to be set
------------------------------------------------------------------------- */

double EwaldDispSlab::gf_of_k(int k)
{
  const double kcell = k * unitk;
  const double kcell3 = kcell * kcell * kcell;
  const double b = kcell / (2.0 * g_ewald);
  const double b2 = b * b, b3 = b2 * b;
  const double coef = -2.0 * MY_PI * sqrt(MY_PI) / (24.0 * volume);
  const double Bk = kcell3 * (sqrt(MY_PI) * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));
  return coef * 2.0 * kcell * Bk;
}

/* ----------------------------------------------------------------------
   estimate g_ewald and kmax from the target force accuracy.

   RMS per-atom force truncation error (random-phase model, validated against
   measured forces):  rms(kmax) = sqrt(0.5 * b2^2 / N * sum_{k>kmax} GF[k]^2),
   with b2 = sum_i B_i^2.  Pick the smallest kmax with rms < accuracy.

   g_ewald is set so the neglected short-range tail (beyond rcut) is below the
   target: (g*rcut)^2 = -2*ln(accuracy).
------------------------------------------------------------------------- */

void EwaldDispSlab::estimate_params()
{
  lat1 = (dim + 1) % 3;
  lat2 = (dim + 2) % 3;
  volume = domain->prd[0] * domain->prd[1] * domain->prd[2];
  unitk = 2.0 * MY_PI / domain->prd[dim];

  // g_ewald (Gaussian short-range tail criterion)

  double acc = accuracy / two_charge_force;    // relative target for the log
  if (acc <= 0.0 || acc >= 1.0) acc = 1.0e-4;
  if (!gewaldflag) g_ewald = sqrt(-2.0 * log(acc)) / cutoff;

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

  // cumulative tail of GF[k]^2 from a large cap; pick kmax for target accuracy.
  // The random-phase model rms = sqrt(prefac * sum_{k>kmax} GF[k]^2) under-predicts
  // the true per-atom force error, so keep a conservative 8x margin.

  const int kbig = 8192;
  const double prefac = 0.5 * b2 * b2 / natoms;
  const double safety = 8.0;
  const double target = accuracy * accuracy / (safety * safety);

  auto *gf2 = new double[kbig + 1];
  for (int k = 1; k <= kbig; k++) {
    double g = gf_of_k(k);
    gf2[k] = g * g;
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

  // The merged smooth corr adds W~2(k)/Lz to each mode; its Fourier content decays
  // only ~k^-5.5 (the switch feature is sharper than the g_ewald Gaussian), so at
  // any usable kmax the corr force tail 2k*W~2(k)/Lz DOMINATES the (super-
  // exponentially small) Gaussian reciprocal tail.  kmax must resolve it: scan the
  // same random-phase tail model and target the corr force tail at ~accuracy.
  // (Measured on the interfacial slab example, err ~ kmax^-5 and the random-phase
  // model tracks the true error to ~1.2x, so targeting accuracy^2 lands the
  // delivered RMS force error at ~accuracy; see estimate_params verification.)

  build_corr_kernels();
  const double invLz = 1.0 / domain->prd[dim];
  const double uk = 2.0 * MY_PI / domain->prd[dim];
  const int ccap = MIN(kbig, 8 * kmax + 256);
  auto *cf2 = new double[ccap + 1];
  for (int k = 1; k <= ccap; k++) {
    double w2t, kw2p;
    corr_tilde(k * uk, w2t, kw2p);
    const double cf = 2.0 * (k * uk) * w2t * invLz;    // corr force per mode
    cf2[k] = cf * cf;
  }
  if (kmax_user == 0) {
    const double ctarget = accuracy * accuracy;    // target the corr force tail at ~accuracy
    double ctail = 0.0;
    int ck = ccap;
    for (int kmx = ccap - 1; kmx >= 4; kmx--) {
      ctail += cf2[kmx + 1];
      if (prefac * ctail >= ctarget) {
        ck = kmx + 1;
        break;
      }
      ck = kmx;
    }
    if (ck > kmax) kmax = MIN(ck, kbig);
  }

  // predicted RMS per-atom force error at the chosen kmax = the Gaussian reciprocal
  // tail PLUS the (dominant) merged corr force tail
  double tk = 0.0;
  for (int k = kmax + 1; k <= kbig; k++) tk += gf2[k];
  double ctk = 0.0;
  for (int k = kmax + 1; k <= ccap; k++) ctk += cf2[k];
  estimated_force_accuracy = sqrt(prefac * (tk + ctk));
  delete[] gf2;
  delete[] cf2;
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

  // fold the smooth corr into GU/GF/GT/GN (no real-space corr step): the corr
  // energy is a z-convolution of the B-density with the tabulated kernel w2(z),
  // diagonal in the reciprocal basis, so it adds to the per-mode coefficients.
  build_corr_kernels();
  merge_corr_coeffs();
}

/* ----------------------------------------------------------------------
   extract per-type dispersion amplitude B[i] = sqrt(|lj4[i][i]|) = 2*sqrt(eps)*sigma^3
------------------------------------------------------------------------- */

void EwaldDispSlab::init_coeffs()
{
  int tmp;
  int n = atom->ntypes;

  // select the C6 cross-term mixing rule.  By default follow the pair style's rule
  // (Pair::mix_flag: GEOMETRIC or ARITHMETIC); kspace_modify mix/disp overrides it via
  // the base flag KSpace::mixflag (0 = follow pair, 1 = force geometric, 2 = none).
  // The eigenvalue-split "none" rule of pppm/disp does not apply to the single-axis
  // 1/r^6 sum, so only geometric and arithmetic (Lorentz-Berthelot) are supported.
  int *p_mix = (int *) force->pair->extract("ewald_mix", tmp);
  const int pair_mix = p_mix ? *p_mix : Pair::GEOMETRIC;
  if (mixflag == 1) {
    mix_flag = 0;    // kspace_modify mix/disp geom
  } else if (mixflag == 2) {
    error->all(FLERR,
               "kspace_modify mix/disp none is not supported by ewald/disp/slab; use "
               "geometric or arithmetic mixing (pair_modify mix geometric|arithmetic)");
  } else {
    if (pair_mix == Pair::GEOMETRIC)
      mix_flag = 0;
    else if (pair_mix == Pair::ARITHMETIC)
      mix_flag = 1;
    else
      error->all(FLERR,
                 "Unsupported pair mixing rule for kspace_style ewald/disp/slab (use "
                 "pair_modify mix geometric|arithmetic)");
  }
  nchan = mix_flag ? 7 : 1;

  delete[] B;

  if (mix_flag == 0) {    // geometric: single per-type amplitude B[i]=sqrt(|lj4[i][i]|)
    auto **b = (double **) force->pair->extract("B", tmp);
    if (b == nullptr)
      error->all(FLERR,
                 "Pair style does not provide dispersion coefficient B for ewald/disp/slab");
    B = new double[n + 1];
    B[0] = 0.0;
    for (int i = 1; i <= n; ++i) B[i] = sqrt(fabs(b[i][i]));
  } else {    // arithmetic (Lorentz-Berthelot): 7-channel binomial expansion
    auto **epsilon = (double **) force->pair->extract("epsilon", tmp);
    auto **sigma = (double **) force->pair->extract("sigma", tmp);
    if (!(epsilon && sigma))
      error->all(FLERR,
                 "Pair style does not provide epsilon/sigma for arithmetic mixing in "
                 "ewald/disp/slab");
    B = new double[7 * n + 7];
    // the seven per-type coefficients of the binomial expansion of
    // (0.5(sigma_i+sigma_j))^6: sqrt(eps_i) c[j] sigma_i^j, j=0..6, with
    // c[j]=sqrt(C(6,j)).  The cross amplitude is
    //   C6_ij = 4 sqrt(eps_i eps_j)((sigma_i+sigma_j)/2)^6
    //         = (1/16) sum_{j=0}^6 B[7i+j] B[7*jtype+(6-j)].
    // For a single type this reduces to (2 sqrt(eps) sigma^3)^2 = geometric B[i]^2,
    // so single-type results are bit-identical to the geometric path.
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

/* ----------------------------------------------------------------------
   pre-compute the reciprocal-space coefficients for each z wavevector

   Gaussian (erfc) smoothing.  Energy GU, z-force GF and tangential pressure GT
   coefficients from the standard dispersion-Ewald erfc form; the normal (zz)
   pressure GN is the exact per-mode strain derivative GN = GU + h dGU/dh, needed
   because the merged smooth corr (added by merge_corr_coeffs) makes the kspace
   share non-homogeneous, so the 6E trace relation does not apply.
------------------------------------------------------------------------- */

void EwaldDispSlab::coeffs()
{
  int k;
  double kcell, kcell3;

  kcount = kmax;

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
    GN[k] = coef * (4.0 * Bk - 1.5 * kcell3 * exp(-b2) / b3);
  }
}

/* ----------------------------------------------------------------------
   C3 septic smoothstep S(t) on t in [0,1]: S(0)=0, S(1)=1 with the first three
   derivatives zero at both ends.  In r: S=0 for r<=rcut, S=1 for r>=rcut+Delta.
   The matched pair fades the 1/r^6 dispersion out by (1-S) over [rcut, rcut+Delta],
   so the corr potential meets rcut+Delta with C3 continuity (its z-Fourier
   coefficients decay as ~k^-5, no Gibbs ringing).
------------------------------------------------------------------------- */

double EwaldDispSlab::switch_S(double t)
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t;
  const double t3 = t2 * t, t4 = t3 * t;
  return t4 * (35.0 - 84.0 * t + 70.0 * t2 - 20.0 * t3);
}

/* ---------------------------------------------------------------------- */

double EwaldDispSlab::switch_dS(double t)
{
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double tu = t * (1.0 - t);
  return 140.0 * tu * tu * tu;    // 140 (t(1-t))^3
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
  } else {    // arithmetic: 7 binomial-channel structure factors per mode
    for (i = 0; i < nlocal; i++) {
      const double *bi = &B[7 * type[i]];

      cs[0][i] = 1.0;
      sn[0][i] = 0.0;
      for (int m = 0; m < 7; m++) sfacrl[m] += bi[m];

      if (kcount > 1) {
        cs[1][i] = cos(unitk * x[i][dim]);
        sn[1][i] = sin(unitk * x[i][dim]);
        for (int m = 0; m < 7; m++) {
          sfacrl[7 + m] += bi[m] * cs[1][i];
          sfacim[7 + m] += bi[m] * sn[1][i];
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
  MPI_Allreduce(sfacrl, sfacrl_all, kcount * nchan, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(sfacim, sfacim_all, kcount * nchan, MPI_DOUBLE, MPI_SUM, world);

  double **f = atom->f;
  int nlocal = atom->nlocal;
  int *type = atom->type;

  for (i = 0; i < nlocal; i++) ek[i] = 0.0;
  if (eflag_atom)
    for (i = 0; i < nlocal; i++) peatom[i] = 0.0;

  if (nchan == 1) {    // ------- geometric mixing -------

    double exprl, expim, partial, partial_peratom;
    for (k = 0; k < kcount; k++) {
      const double srl = sfacrl_all[k], sim = sfacim_all[k];
      for (i = 0; i < nlocal; i++) {
        exprl = cs[k][i];
        expim = sn[k][i];
        partial = expim * srl - exprl * sim;
        ek[i] += partial * GF[k];

        if (evflag_atom) {
          partial_peratom = exprl * srl + expim * sim;
          if (eflag_atom) peatom[i] += GU[k] * partial_peratom;
          if (vflag_atom) {
            vatom[i][lat1] += GT[k] * partial_peratom;
            vatom[i][lat2] += GT[k] * partial_peratom;
            vatom[i][dim] += GN[k] * partial_peratom;
          }
        }
      }
    }

    for (i = 0; i < nlocal; i++) f[i][dim] += B[type[i]] * ek[i];

    if (eflag_global) {
      double e_recip = 0.0;
      for (k = 0; k < kcount; k++)
        e_recip += GU[k] * (sfacrl_all[k] * sfacrl_all[k] + sfacim_all[k] * sfacim_all[k]);
      energy += e_recip;
    }
    if (vflag_global) {
      for (k = 0; k < kcount; k++) {
        double uk = sfacrl_all[k] * sfacrl_all[k] + sfacim_all[k] * sfacim_all[k];
        virial[lat1] += uk * GT[k];
        virial[lat2] += uk * GT[k];
        virial[dim] += uk * GN[k];
      }
    }

    if (eflag_atom)
      for (i = 0; i < nlocal; i++) eatom[i] += B[type[i]] * peatom[i];
    if (vflag_atom)
      for (i = 0; i < nlocal; i++) {
        vatom[i][lat1] *= B[type[i]];
        vatom[i][lat2] *= B[type[i]];
        vatom[i][dim] *= B[type[i]];
      }

  } else {    // ------- arithmetic (Lorentz-Berthelot) mixing -------

    // The C6 cross term folds into the per-mode channel pairing
    //   R_k = sum_{m=0}^6 S^(m) S^(6-m)  (each unordered pair counted twice except
    // the self m=3, hence the 0.5 there), with the energy/virial normalization
    // as_e = 1/8 and the z-force as_f = as_e/2 = 1/16 (the force differentiates both
    // pair indices).  For a single type R_k/8 == |S_k(geom)|^2 exactly.
    const double as_e = 0.125;
    const double as_f = 1.0 / 16.0;

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

    if (eflag_global || vflag_global) {
      double e_recip = 0.0;
      for (k = 0; k < kcount; k++) {
        const double *sr = &sfacrl_all[k * 7];
        const double *sm = &sfacim_all[k * 7];
        const double R = (sr[0] * sr[6] + sm[0] * sm[6]) + (sr[1] * sr[5] + sm[1] * sm[5]) +
            (sr[2] * sr[4] + sm[2] * sm[4]) + 0.5 * (sr[3] * sr[3] + sm[3] * sm[3]);
        e_recip += GU[k] * R;
        if (vflag_global) {
          virial[lat1] += as_e * R * GT[k];
          virial[lat2] += as_e * R * GT[k];
          virial[dim] += as_e * R * GN[k];
        }
      }
      if (eflag_global) energy += as_e * e_recip;
    }

    if (eflag_atom)
      for (i = 0; i < nlocal; i++) eatom[i] += peatom[i];
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
   tabulate the smooth (switched-pair) damped correction energy kernel over
   [0, rcut+Delta].  With the matched lj/cut/dispswitch pair the 1/r^6 dispersion
   is faded out by (1-S) over [rcut, rcut+Delta], so the corr potential
       corr_e(r) = u_smooth(r) - [r>rcut] S(r)/r^6
   vanishes smoothly at rcut+Delta (corr_e(rcut+Delta) = u_short(rcut+Delta) ~ acc^2).
   Here we tabulate the energy kernel w2 = (2 pi/area) int_{|dz|}^{b} r corr_e(r) dr
   by Simpson quadrature; corr_tilde() Fourier-transforms it for merge_corr_coeffs().
------------------------------------------------------------------------- */

void EwaldDispSlab::build_corr_kernels()
{
  const double a = cutoff, b = cutoff + sw_width;
  ncgrid = 1024;
  cwdz = b / ncgrid;

  // BOX-INDEPENDENT kernel integral IE[g] = int_{z_g}^b r*corr_e(r) dr (g_ewald,
  // cutoff, Delta are all fixed after init), precomputed once.  Under NPT only the
  // 2*pi/area prefactor changes, so the per-step build is just a rescale.
  if (cWraw == nullptr) {
    cWraw = new double[ncgrid + 1];
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
      cWraw[g] = IE;
    }
  }

  const double pre = 2.0 * MY_PI / (domain->prd[lat1] * domain->prd[lat2]);
  delete[] cWgrid;
  cWgrid = new double[ncgrid + 1];
  for (int g = 0; g <= ncgrid; g++) cWgrid[g] = pre * cWraw[g];

  // ensure the box-independent Fourier-transform tables cover the current modes
  build_corr_ft_tables((kcount > 0 ? (kcount - 1) : kmax) * unitk);
}

/* ----------------------------------------------------------------------
   1-D Fourier transforms of the tabulated corr kernel (Simpson over the table):
     w2t  = W~2(k)        = 2 int_0^b w2(z) cos(kz) dz
     kw2p = k dW~2(k)/dk  = -2 k int_0^b z w2(z) sin(kz) dz
   Exact reference (used to build the interpolation tables).
------------------------------------------------------------------------- */

void EwaldDispSlab::corr_tilde(double k, double &w2t, double &kw2p)
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
   (re)build the box-independent Fourier-transform tables of the corr kernel on
   a uniform wavenumber grid kap_j = j*kap_dk covering [0, kap_need] with margin.
   Grow-only: rebuilt only when the modes outgrow the current range (an NPT box
   that shrinks).  A(kap)=2 int cWraw cos(kap z) dz, B(kap)=2 int z cWraw sin, so
   W~2(k)=(2*pi/area) A(k) and k dW~2/dk = -(2*pi/area) k B(k).
------------------------------------------------------------------------- */

void EwaldDispSlab::build_corr_ft_tables(double kap_need)
{
  const double target = 1.5 * MAX(kap_need, 1.0e-6);    // 50% headroom for NPT shrink
  if (Araw_tab && target <= kap_max) return;             // current tables suffice

  // resolve the FT oscillation (period 2*pi/b in kap); ~100 points per period
  kap_dk = (2.0 * MY_PI / (cutoff + sw_width)) / 100.0;
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
   The tables are ~100 points per oscillation, so the error is ~1e-9 relative --
   far below the reciprocal accuracy the merge targets.
------------------------------------------------------------------------- */

void EwaldDispSlab::ft_interp(double kap, double &A, double &B)
{
  double x = kap / kap_dk;
  int j = (int) x - 1;    // centered 4-point stencil j..j+3 (t in [1,2] interior)
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
   fold the smooth switched corr into the reciprocal coefficients GU/GF/GT/GN.
   E_corr = sum_n [W~2(k_n)/Lz] |S_n|^2 is the same bilinear form as the reciprocal
   sum, so per mode: GU += CE, GT += CE (corr tangential = corr energy since
   pt2 = w2), GN += CN (the normal strain derivative), and GF is the exact
   z-gradient 2k*GU of the merged energy.  The +/- k double-count gives a 0.5 on
   the k=0 term and 1.0 for k>=1 (each of the FFT's +-k modes carries 0.5, so this
   matches the verified pppm/disp/slab merge exactly).  After this, compute() does
   energy/force/virial + corr in one reciprocal pass -- no O(N^2) raw, no binning.
------------------------------------------------------------------------- */

void EwaldDispSlab::merge_corr_coeffs()
{
  // W~2(k) = (2*pi/area) A(k) and k dW~2/dk = -(2*pi/area) k B(k), so the merge
  // coefficients are (2*pi/volume) times the box-independent A(k), (A - k B)(k).
  // Interpolate A, B from the precomputed tables at the current modes -- no
  // per-step quadrature (NPT-proof).
  const double area = domain->prd[lat1] * domain->prd[lat2];
  const double pre2 = 2.0 * MY_PI / (area * domain->prd[dim]);    // 2*pi/volume
  for (int k = 0; k < kcount; k++) {
    const double kc = k * unitk;
    double A, Bv;
    ft_interp(kc, A, Bv);
    const double f = (k == 0) ? 0.5 : 1.0;
    const double CE = f * pre2 * A;
    const double CN = f * pre2 * (A - kc * Bv);
    GU[k] += CE;
    GT[k] += CE;
    GN[k] += CN;
    GF[k] = 2.0 * kc * GU[k];    // exact z-gradient of the merged energy
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
  // structure factors carry nchan channels per mode (1 geometric, 7 arithmetic)
  memory->create(sfacrl, kmax * nchan, "ewald/disp/slab:sfacrl");
  memory->create(sfacim, kmax * nchan, "ewald/disp/slab:sfacim");
  memory->create(sfacrl_all, kmax * nchan, "ewald/disp/slab:sfacrl_all");
  memory->create(sfacim_all, kmax * nchan, "ewald/disp/slab:sfacim_all");
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

/* ----------------------------------------------------------------------
   Irving-Kirkwood long-range pressure profile (compute stress/cartesian hook).
   The merged-damped reciprocal represents the identical switched tail S(r)*u(r)
   as pppm/disp/slab (the pair fades the dispersion by (1-S)), so the same S*u
   pressure building blocks apply.  Ported from pppm/disp/slab (which was ported
   from pppm/disp/planar); the only difference here is the exact-sum mode cutoff
   K = kcount-1 (no mesh over-resolution to truncate).  Special functions and
   kernels below are self-contained in terms of the S*u potential (cutoff,
   sw_width, B, volume), independent of the reciprocal solve.
------------------------------------------------------------------------- */

void EwaldDispSlab::cisi(double x, double &si, double &ci)
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

void EwaldDispSlab::sici_compl_chain(double x, double *Carr, double *Darr)
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

double EwaldDispSlab::ik_phi(double h)
{
  if (fabs(h) < 1.0e-300) return 0.0;
  const double ah = fabs(h);
  const double c = cutoff + sw_width;
  double A[8], Bc[8];
  sici_chain(ah * c, A, Bc);
  const double sii5 = A[5] / 4.0 - A[1] / (4.0 * pow(ah * c, 4));
  double phi = MY_PI / 576.0 - sii5 + A[7] - Bc[6];
  phi += prof_shell(PROF_PHI, ah);
  const double ah4 = ah * ah * ah * ah;
  return (h >= 0.0 ? 1.0 : -1.0) * ah4 * phi;
}

double EwaldDispSlab::ik_psi(double h)
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

void EwaldDispSlab::shell_profile_virial(int nbins, double /*lo*/, double /*dz*/,
                                         double * /*dens_all*/, double *shellT, double *shellN)
{
  // No shell subtraction for the merged-damped variant: the pair fades the
  // dispersion by (1-S) and the kspace GT[k]/GN[k] already carry the full plane
  // mean field of S*u, so the reciprocal double sum needs no shell correction.
  for (int g = 0; g < nbins; g++) shellT[g] = shellN[g] = 0.0;
}

void EwaldDispSlab::profile_GTGN_raw(int K, double *GTr, double *GNr)
{
  const double zprd = domain->prd[dim];
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

void EwaldDispSlab::profile_assemble(int K, int nbins, double lo, double width, const double *Sre,
                                     const double *Sim, const double *GTr, const double *GNr,
                                     const double *shellT, const double *shellN, double *pN,
                                     double *pT)
{
  const double zprd = domain->prd[dim];
  const double area = domain->prd[lat1] * domain->prd[lat2];
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

int EwaldDispSlab::pressure_profile_long(int dir, int nbins, double lo, double width, double *pN,
                                         double *pT)
{
  if (dir != dim)
    error->all(FLERR,
               "compute stress/cartesian binning direction must match the inhomogeneous axis "
               "of ewald/disp/slab");

  const double unitk = 2.0 * MY_PI / domain->prd[dim];
  const int K = kcount - 1;    // highest resolved mode (force-accuracy sized)

  if (nbins <= 2 * K)
    error->all(FLERR,
               "compute stress/cartesian with ewald/disp/slab kspace: {} bins along the "
               "inhomogeneous axis is too coarse; need > {} (= 2*kmax) to resolve the "
               "Irving-Kirkwood reciprocal modes without aliasing (use a finer bin width, "
               "looser accuracy, or wider switch)",
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

  // no shell subtraction for the merged-damped variant (GT/GN carry the full S*u
  // plane mean field); keep the zeroed arrays so profile_assemble's signature is
  // unchanged.
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

/* ---------------------------------------------------------------------- */

double EwaldDispSlab::memory_usage()
{
  double bytes = 8.0 * kmax * sizeof(double);    // GU,GF,GT,GN,sfacrl/im,sfacrl/im_all
  bytes += 2.0 * (double) nmax * sizeof(double);    // ek, peatom
  bytes += 2.0 * (double) kmax * nmax * sizeof(double);    // cs, sn
  bytes += (double) (ncgrid + 1) * sizeof(double);    // corr energy kernel
  return bytes;
}
