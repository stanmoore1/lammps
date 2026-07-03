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
  sw_width = 0.0;
  cWgrid = nullptr;
  ncgrid = 0;
  cwdz = 0.0;
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
  // SLOWER than the Gaussian reciprocal (the switch feature is sharper than the
  // g_ewald Gaussian), so kmax must also resolve the corr force tail 2k*W~2(k)/Lz.
  // Same random-phase tail criterion, but that model over-predicts the corr force
  // error by ~12-20x (measured err ~ kmax^-5.5); target 4*accuracy so the scan's
  // over-prediction lands the true error at ~acc/3.

  if (kmax_user == 0) {
    build_corr_kernels();
    const double invLz = 1.0 / domain->prd[dim];
    const double uk = 2.0 * MY_PI / domain->prd[dim];
    const int ccap = MIN(kbig, 8 * kmax + 128);
    auto *cf2 = new double[ccap + 1];
    for (int k = 1; k <= ccap; k++) {
      double w2t, kw2p;
      corr_tilde(k * uk, w2t, kw2p);
      const double cf = 2.0 * (k * uk) * w2t * invLz;    // corr force per mode
      cf2[k] = cf * cf;
    }
    const double ctarget = 16.0 * accuracy * accuracy;    // (4*acc)^2, see above
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
    delete[] cf2;
    if (ck > kmax) kmax = MIN(ck, kbig);
  }

  // predicted RMS per-atom force error at the chosen kmax
  double tk = 0.0;
  for (int k = kmax + 1; k <= kbig; k++) tk += gf2[k];
  estimated_force_accuracy = sqrt(prefac * tk);
  delete[] gf2;
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
  if (eflag_atom)
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
        if (eflag_atom) peatom[i] += GU[k] * partial_peratom;
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

  // reciprocal energy (full system value, identical on every proc)

  if (eflag_global) {
    double e_recip = 0.0;
    for (k = 0; k < kcount; k++)
      e_recip += GU[k] * (sfacrl_all[k] * sfacrl_all[k] + sfacim_all[k] * sfacim_all[k]);
    energy += e_recip;
  }

  // global virial: tangential (xx=yy) from GT, normal (zz) from the explicit GN

  if (vflag_global) {
    for (k = 0; k < kcount; k++) {
      double uk = sfacrl_all[k] * sfacrl_all[k] + sfacim_all[k] * sfacim_all[k];
      virial[lat1] += uk * GT[k];
      virial[lat2] += uk * GT[k];
      virial[dim] += uk * GN[k];
    }
  }

  // scale per-atom energy / virial by each atom's B and report

  if (eflag_atom)
    for (i = 0; i < nlocal; i++) eatom[i] += B[type[i]] * peatom[i];
  if (vflag_atom)
    for (i = 0; i < nlocal; i++) {
      vatom[i][lat1] *= B[type[i]];
      vatom[i][lat2] *= B[type[i]];
      vatom[i][dim] *= B[type[i]];
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
   1-D Fourier transforms of the tabulated corr kernel (Simpson over the table):
     w2t  = W~2(k)        = 2 int_0^b w2(z) cos(kz) dz
     kw2p = k dW~2(k)/dk  = -2 k int_0^b z w2(z) sin(kz) dz
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
  const double invLz = 1.0 / domain->prd[dim];
  for (int k = 0; k < kcount; k++) {
    double w2t, kw2p;
    corr_tilde(k * unitk, w2t, kw2p);
    const double f = (k == 0) ? 0.5 : 1.0;
    const double CE = f * w2t * invLz;
    const double CN = f * (w2t + kw2p) * invLz;
    GU[k] += CE;
    GT[k] += CE;
    GN[k] += CN;
    GF[k] = 2.0 * (k * unitk) * GU[k];    // exact z-gradient of the merged energy
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
  bytes += 2.0 * (double) nmax * sizeof(double);    // ek, peatom
  bytes += 2.0 * (double) kmax * nmax * sizeof(double);    // cs, sn
  bytes += (double) (ncgrid + 1) * sizeof(double);    // corr energy kernel
  return bytes;
}
