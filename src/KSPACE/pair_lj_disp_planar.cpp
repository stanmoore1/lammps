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

#include "pair_lj_disp_planar.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "utils.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJDispPlanar::PairLJDispPlanar(LAMMPS *lmp) : PairLJCut(lmp)
{
  sw_width = 0.1;      // default switch width Delta
  inner_rc2 = 0.0;
  sw_order = 3;        // default C3 septic switch
  respa_enable = 0;    // rRESPA inner/middle/outer not supported with the switch
}

/* ----------------------------------------------------------------------
   Generalized C^n smootherstep S_n(t) (degree 2n+1 Hermite interpolant, first n
   derivatives vanishing at t=0,1) and its derivative S_n'(t) = c_n (t(1-t))^n
   with c_n = (2n+1)!/(n!)^2.  n=3 is the septic default; higher n gives faster
   reciprocal-coefficient decay (h^-(n+2)) and a smaller grid at fixed accuracy.
   MUST match the kspace ewald/disp/planar / pppm/disp/planar switch_S/switch_dS.
------------------------------------------------------------------------- */

double PairLJDispPlanar::sw_S(double t) const
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
  if (sw_order == 4)
    return t5 * (126.0 + t * (-420.0 + t * (540.0 + t * (-315.0 + 70.0 * t))));
  if (sw_order == 5) {
    const double t6 = t5 * t;
    return t6 * (462.0 + t * (-1980.0 + t * (3465.0 + t * (-3080.0 + t * (1386.0 - 252.0 * t)))));
  }
  return t4 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)));    // n=3 (default)
}

double PairLJDispPlanar::sw_dS(double t) const
{
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double tu = t * (1.0 - t), tu2 = tu * tu;
  if (sw_order == 4) return 630.0 * tu2 * tu2;             // 630 (t(1-t))^4
  if (sw_order == 5) return 2772.0 * tu2 * tu2 * tu;       // 2772 (t(1-t))^5
  return 140.0 * tu2 * tu;                                 // 140 (t(1-t))^3 (default)
}

/* ----------------------------------------------------------------------
   pair_style lj/disp/planar <rcut> [Delta] [order]
------------------------------------------------------------------------- */

void PairLJDispPlanar::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 3) error->all(FLERR, "Illegal pair_style lj/disp/planar command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (narg >= 2) sw_width = utils::numeric(FLERR, arg[1], false, lmp);    // else default 0.1
  if (sw_width <= 0.0) error->all(FLERR, "pair_style lj/disp/planar switch width must be > 0");
  if (narg == 3) {
    sw_order = utils::inumeric(FLERR, arg[2], false, lmp);
    if (sw_order < 3 || sw_order > 5)
      error->all(FLERR, "pair_style lj/disp/planar switch order must be 3, 4, or 5");
  }
  inv_sw_width = 1.0 / sw_width;    // precomputed for the hot loop

  // reset per-type cutoffs (always global here)
  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global + sw_width;
  }
}

/* ----------------------------------------------------------------------
   inner = rcut; neighbor/interaction cutoff = rcut + Delta; no energy shift
------------------------------------------------------------------------- */

double PairLJDispPlanar::init_one(int i, int j)
{
  PairLJCut::init_one(i, j);    // mix epsilon/sigma -> lj1..lj4
  cut[i][j] = cut[j][i] = cut_global + sw_width;
  offset[i][j] = offset[j][i] = 0.0;    // kspace continues the tail: no shift
  inner_rc2 = cut_global * cut_global;
  inv_sw_width = 1.0 / sw_width;    // precomputed for the hot loop (NPT-safe: Delta fixed)

  // The matched kspace derives the i-j dispersion C6 cross term from the per-type
  // diagonals via its mixing rule (geometric, or arithmetic Lorentz-Berthelot); an
  // explicit non-conforming pair_coeff i j is represented in the pair but NOT in the
  // reciprocal tail.  Warn if they disagree (reference built from the diagonals, so
  // this is independent of the order in which init_one is called).
  if (i != j) {
    const double si6 = pow(sigma[i][i], 6.0), sj6 = pow(sigma[j][j], 6.0);
    double ref;
    if (mix_flag == ARITHMETIC) {
      const double sij = 0.5 * (sigma[i][i] + sigma[j][j]);
      ref = 4.0 * sqrt(epsilon[i][i] * epsilon[j][j]) * pow(sij, 6.0);
    } else {    // geometric (the kspace treats any other rule as geometric)
      ref = sqrt((4.0 * epsilon[i][i] * si6) * (4.0 * epsilon[j][j] * sj6));
    }
    if (fabs(lj4[i][j] - ref) > 1.0e-6 * MAX(fabs(ref), 1.0e-300))
      error->warning(FLERR,
                     "pair lj/disp/planar: the {}-{} dispersion C6 does not match the "
                     "kspace mixing rule; the long-range tail will use the mixed value, "
                     "not the explicit pair_coeff",
                     i, j);
  }
  return cut_global + sw_width;
}

/* ----------------------------------------------------------------------
   the matched kspace style handles the 1/r^6 dispersion beyond the cutoff, so
   the analytic long-range tail correction would double count it
------------------------------------------------------------------------- */

void PairLJDispPlanar::init_style()
{
  if (tail_flag)
    error->all(FLERR,
               "Pair style lj/disp/planar is incompatible with pair_modify tail yes "
               "(the dispersion tail is handled by the matched kspace style)");
  PairLJCut::init_style();
}

/* ----------------------------------------------------------------------
   proc 0 writes/reads to restart file.  The base writes cut_global etc.; the
   switch width Delta must be persisted too, or it resets to 0 on restart (which
   collapses the cutoff to the inner rcut and aborts the matched kspace).
------------------------------------------------------------------------- */

void PairLJDispPlanar::write_restart_settings(FILE *fp)
{
  PairLJCut::write_restart_settings(fp);
  fwrite(&sw_width, sizeof(double), 1, fp);
  fwrite(&sw_order, sizeof(int), 1, fp);
}

void PairLJDispPlanar::read_restart_settings(FILE *fp)
{
  PairLJCut::read_restart_settings(fp);
  if (comm->me == 0) {
    utils::sfread(FLERR, &sw_width, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &sw_order, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&sw_width, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&sw_order, 1, MPI_INT, 0, world);
  inv_sw_width = 1.0 / sw_width;
}

/* ---------------------------------------------------------------------- */

void PairLJDispPlanar::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // hoist the switch constants out of the pair loop
  const double rc_inner = cut_global;    // inner cutoff rcut
  const double invsw = inv_sw_width;     // 1/Delta

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq >= cutsq[itype][jtype]) continue;

      r2inv = 1.0 / rsq;
      r6inv = r2inv * r2inv * r2inv;

      if (rsq < inner_rc2) {

        // full LJ to rcut (inner; the matched kspace continues the dispersion tail).
        // The switch only splits the 1/r^6 dispersion between real and reciprocal
        // space; the 1/r^12 repulsion is short-range and is computed in full here.
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;
        if (eflag) evdwl = factor_lj * r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]);

      } else {

        // shell [rcut, rcut+Delta]: only the attractive 1/r^6 dispersion is split
        // between real and reciprocal space (switched by (1 - S)); the 1/r^12
        // repulsion is short-range and is evaluated in FULL here, exactly as inside
        // rcut, so the potential and force are continuous across rcut (they are
        // truncated only at rcut+Delta, where 1/r^12 is negligible).  The kspace
        // style supplies the plane S*u dispersion tail.  E = lj3 r^-12 - (1-S) lj4
        // r^-6 ; fpair = lj1 r^-14 - lj4[S'(t)/Delta r^-7 + 6 (1-S) r^-8].  Here t is
        // guaranteed in [0,1) (rcut <= r < rcut+Delta), so the switch polynomials are
        // inlined without the end clamps, and all divisions are replaced by the
        // precomputed 1/Delta and rinv = sqrt(r2inv) (no 1/r, no /Delta).
        const double rinv = sqrt(r2inv);
        const double r = rsq * rinv;    // = sqrt(rsq), one mul not a div
        const double t = (r - rc_inner) * invsw;
        const double S = sw_S(t), dS = sw_dS(t);    // C^n smootherstep (order sw_order)
        const double oneMinusS = 1.0 - S;
        const double lj1ij = lj1[itype][jtype], lj3ij = lj3[itype][jtype];
        const double lj4ij = lj4[itype][jtype];
        const double frep = lj1ij * r6inv * r6inv * r2inv;    // full 1/r^12 repulsion
        const double fatt = lj4ij * r6inv * ((dS * invsw) * rinv + 6.0 * oneMinusS * r2inv);
        fpair = factor_lj * (frep - fatt);
        if (eflag) evdwl = factor_lj * (lj3ij * r6inv * r6inv - oneMinusS * lj4ij * r6inv);
      }

      f[i][0] += delx * fpair;
      f[i][1] += dely * fpair;
      f[i][2] += delz * fpair;
      if (newton_pair || j < nlocal) {
        f[j][0] -= delx * fpair;
        f[j][1] -= dely * fpair;
        f[j][2] -= delz * fpair;
      }

      if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

double PairLJDispPlanar::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                                double /*factor_coul*/, double factor_lj, double &fforce)
{
  if (rsq >= cutsq[itype][jtype]) {
    fforce = 0.0;
    return 0.0;
  }
  const double r2inv = 1.0 / rsq;
  const double r6inv = r2inv * r2inv * r2inv;
  double phi, forcelj;
  if (rsq < inner_rc2) {
    // full LJ to rcut (the matched kspace continues the dispersion tail)
    forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
    fforce = factor_lj * forcelj * r2inv;
    phi = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]);
  } else {
    const double rinv = sqrt(r2inv), r = rsq * rinv;
    const double t = (r - cut_global) * inv_sw_width;
    const double S = sw_S(t), dS = sw_dS(t), oneMinusS = 1.0 - S;
    const double lj1ij = lj1[itype][jtype], lj3ij = lj3[itype][jtype];
    const double lj4ij = lj4[itype][jtype];
    // full 1/r^12 repulsion + switched (1-S) 1/r^6 attraction (matches compute())
    fforce = factor_lj *
        (lj1ij * r6inv * r6inv * r2inv -
         lj4ij * r6inv * ((dS * inv_sw_width) * rinv + 6.0 * oneMinusS * r2inv));
    phi = lj3ij * r6inv * r6inv - oneMinusS * lj4ij * r6inv;
  }
  return factor_lj * phi;
}

/* ---------------------------------------------------------------------- */

void *PairLJDispPlanar::extract(const char *str, int &dim)
{
  if (strcmp(str, "disp_switch_width") == 0) {
    dim = 0;
    return (void *) &sw_width;
  }
  if (strcmp(str, "disp_switch_order") == 0) {    // C^n switch order for the matched kspace
    dim = 0;
    return (void *) &sw_order;
  }
  if (strcmp(str, "ewald_mix") == 0) {    // C6 mixing rule for the matched kspace style
    dim = 0;
    return (void *) &mix_flag;
  }
  return PairLJCut::extract(str, dim);    // cut_lj -> rcut (inner), B, epsilon, sigma
}
