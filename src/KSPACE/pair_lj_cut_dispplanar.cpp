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
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "pair_lj_cut_dispplanar.h"

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

PairLJCutDispPlanar::PairLJCutDispPlanar(LAMMPS *lmp) : PairLJCut(lmp)
{
  sw_width = 0.0;
  inner_rc2 = 0.0;
  full_shell = 0;    // default (1-S)*u; the matched kspace sets 1 via extract
  respa_enable = 0;    // rRESPA inner/middle/outer not supported with the switch
}

/* ----------------------------------------------------------------------
   C3 (septic) smoothstep S(t) and its derivative S'(t)=140 t^3 (1-t)^3,
   identical to the planar kspace compact switch.
------------------------------------------------------------------------- */

double PairLJCutDispPlanar::sw_S(double t)
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t, t3 = t2 * t, t4 = t3 * t;
  return t4 * (35.0 - 84.0 * t + 70.0 * t2 - 20.0 * t3);
}

double PairLJCutDispPlanar::sw_dS(double t)
{
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double u = 1.0 - t;
  return 140.0 * t * t * t * u * u * u;
}

/* ----------------------------------------------------------------------
   pair_style lj/cut/dispplanar <rcut> <Delta>
------------------------------------------------------------------------- */

void PairLJCutDispPlanar::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Illegal pair_style lj/cut/dispplanar command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  sw_width = utils::numeric(FLERR, arg[1], false, lmp);
  if (sw_width <= 0.0) error->all(FLERR, "pair_style lj/cut/dispplanar switch width must be > 0");

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

double PairLJCutDispPlanar::init_one(int i, int j)
{
  PairLJCut::init_one(i, j);    // mix epsilon/sigma -> lj1..lj4
  cut[i][j] = cut[j][i] = cut_global + sw_width;
  offset[i][j] = offset[j][i] = 0.0;    // kspace continues the tail: no shift
  inner_rc2 = cut_global * cut_global;
  return cut_global + sw_width;
}

/* ----------------------------------------------------------------------
   the matched kspace style handles the 1/r^6 dispersion beyond the cutoff, so
   the analytic long-range tail correction would double count it
------------------------------------------------------------------------- */

void PairLJCutDispPlanar::init_style()
{
  if (tail_flag)
    error->all(FLERR, "Pair style lj/cut/dispplanar is incompatible with pair_modify tail yes "
                      "(the dispersion tail is handled by the matched kspace style)");
  PairLJCut::init_style();
}

/* ----------------------------------------------------------------------
   proc 0 writes/reads to restart file.  The base writes cut_global etc.; the
   switch width Delta must be persisted too, or it resets to 0 on restart (which
   collapses the cutoff to the inner rcut and aborts the matched kspace).
------------------------------------------------------------------------- */

void PairLJCutDispPlanar::write_restart_settings(FILE *fp)
{
  PairLJCut::write_restart_settings(fp);
  fwrite(&sw_width, sizeof(double), 1, fp);
}

void PairLJCutDispPlanar::read_restart_settings(FILE *fp)
{
  PairLJCut::read_restart_settings(fp);
  if (comm->me == 0) utils::sfread(FLERR, &sw_width, sizeof(double), 1, fp, nullptr, error);
  MPI_Bcast(&sw_width, 1, MPI_DOUBLE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairLJCutDispPlanar::compute(int eflag, int vflag)
{
  // full_shell evaluates full LJ over the whole [0, rcut+Delta] range (the
  // switch lives entirely in the kspace S*u split + shell correction), which is
  // exactly what the optimized base PairLJCut::compute does with cut = rcut+Delta
  // and no energy offset.  Delegate to it so the matched pair is as fast as lj/cut.
  if (full_shell) {
    PairLJCut::compute(eflag, vflag);
    return;
  }

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

        // full LJ to rcut (inner); the matched kspace continues the dispersion
        // tail.  The switch only splits the 1/r^6 dispersion between real and
        // reciprocal space; the 1/r^12 repulsion is short-range and computed in
        // full here.
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;
        if (eflag) evdwl = factor_lj * r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]);

      } else {

        // shell [rcut, rcut+Delta]: attractive dispersion switched by (1 - S).
        // E = -(1-S) lj4 r^-6 ; fpair = -lj4[ S'(t)/Delta r^-7 + 6 (1-S) r^-8 ].
        // The kspace style supplies the plane S*u tail.
        const double r = sqrt(rsq);
        const double t = (r - cut_global) / sw_width;
        const double S = sw_S(t);
        const double dS = sw_dS(t);    // dS/dt
        const double oneMinusS = 1.0 - S;
        const double lj4ij = lj4[itype][jtype];
        const double rinv = 1.0 / r;
        fpair = -factor_lj * lj4ij *
            ((dS / sw_width) * r6inv * rinv + 6.0 * oneMinusS * r6inv * r2inv);
        if (eflag) evdwl = -factor_lj * oneMinusS * lj4ij * r6inv;
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

double PairLJCutDispPlanar::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                                   double /*factor_coul*/, double factor_lj, double &fforce)
{
  if (rsq >= cutsq[itype][jtype]) {
    fforce = 0.0;
    return 0.0;
  }
  const double r2inv = 1.0 / rsq;
  const double r6inv = r2inv * r2inv * r2inv;
  double phi, forcelj;
  if (rsq < inner_rc2 || full_shell) {
    // full LJ (full_shell evaluates the repulsion + full dispersion to rcut+Delta;
    // the kspace plane mean-field S*u is removed by the shell correction)
    forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
    fforce = factor_lj * forcelj * r2inv;
    phi = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]);
  } else {
    const double r = sqrt(rsq), rinv = 1.0 / r;
    const double t = (r - cut_global) / sw_width;
    const double S = sw_S(t), dS = sw_dS(t), oneMinusS = 1.0 - S;
    const double lj4ij = lj4[itype][jtype];
    fforce = -lj4ij * ((dS / sw_width) * r6inv * rinv + 6.0 * oneMinusS * r6inv * r2inv);
    fforce *= factor_lj;
    phi = -oneMinusS * lj4ij * r6inv;
  }
  return factor_lj * phi;
}

/* ----------------------------------------------------------------------
   expose the inner cutoff, dispersion amplitude, and switch parameters so the
   matched planar kspace style is self-contained (no edit to pair_lj_cut needed)
------------------------------------------------------------------------- */

void *PairLJCutDispPlanar::extract(const char *str, int &dim)
{
  if (strcmp(str, "disp_switch_width") == 0) {
    dim = 0;
    return (void *) &sw_width;
  }
  if (strcmp(str, "disp_full_shell") == 0) {
    dim = 0;
    return (void *) &full_shell;
  }
  if (strcmp(str, "cut_lj") == 0) {
    dim = 0;
    return (void *) &cut_global;    // inner cutoff rcut
  }
  if (strcmp(str, "B") == 0) {
    dim = 2;
    return (void *) lj4;    // 4*eps*sigma^6 (dispersion C6); B[i] = sqrt(|lj4[i][i]|)
  }
  return PairLJCut::extract(str, dim);    // epsilon, sigma
}
