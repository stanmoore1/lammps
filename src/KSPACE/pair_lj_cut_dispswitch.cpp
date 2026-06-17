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

#include "pair_lj_cut_dispswitch.h"

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

PairLJCutDispSwitch::PairLJCutDispSwitch(LAMMPS *lmp) : PairLJCut(lmp)
{
  sw_width = 0.0;
  inner_rc2 = 0.0;
  csb_full_shell = 0;    // default (1-S)*u; ewald/disp/slab sets 1 via extract
  respa_enable = 0;    // rRESPA inner/middle/outer not supported with the switch
}

/* ----------------------------------------------------------------------
   C3 (septic) smoothstep S(t) and its derivative S'(t)=140 t^3 (1-t)^3,
   identical to the kspace ewald/disp/slab compact switch.
------------------------------------------------------------------------- */

double PairLJCutDispSwitch::sw_S(double t)
{
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  const double t2 = t * t, t3 = t2 * t, t4 = t3 * t;
  return t4 * (35.0 - 84.0 * t + 70.0 * t2 - 20.0 * t3);
}

double PairLJCutDispSwitch::sw_dS(double t)
{
  if (t <= 0.0 || t >= 1.0) return 0.0;
  const double u = 1.0 - t;
  return 140.0 * t * t * t * u * u * u;
}

/* ----------------------------------------------------------------------
   pair_style lj/cut/dispswitch <rcut> <Delta>
------------------------------------------------------------------------- */

void PairLJCutDispSwitch::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Illegal pair_style lj/cut/dispswitch command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  sw_width = utils::numeric(FLERR, arg[1], false, lmp);
  if (sw_width <= 0.0) error->all(FLERR, "pair_style lj/cut/dispswitch switch width must be > 0");

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

double PairLJCutDispSwitch::init_one(int i, int j)
{
  PairLJCut::init_one(i, j);    // mix epsilon/sigma -> lj1..lj4
  cut[i][j] = cut[j][i] = cut_global + sw_width;
  offset[i][j] = offset[j][i] = 0.0;    // kspace continues the tail: no shift
  inner_rc2 = cut_global * cut_global;
  return cut_global + sw_width;
}

/* ---------------------------------------------------------------------- */

void PairLJCutDispSwitch::compute(int eflag, int vflag)
{
  // csb_full_shell evaluates full LJ over the whole [0, rcut+Delta] range (the
  // switch lives entirely in the kspace S*u split + corr_csb), which is exactly
  // what the optimized base PairLJCut::compute does with cut = rcut+Delta and no
  // energy offset.  Delegate to it so the matched pair is as fast as lj/cut.
  if (csb_full_shell) {
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

      if (rsq < inner_rc2 || csb_full_shell) {

        // full LJ to rcut+Delta (csb_full_shell) or to rcut (inner; kspace continues
        // the dispersion tail).  The switch only splits the 1/r^6 dispersion between
        // real and reciprocal space; the 1/r^12 repulsion is short-range and is
        // computed in full here.  In csb_full_shell mode the reciprocal sum's plane
        // mean-field S*u over the shell is removed by corr_csb(), so the pair supplies
        // the exact 3-D dispersion as well as the repulsion to rcut+Delta.
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;
        if (eflag) evdwl = factor_lj * r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]);

      } else {

        // shell [rcut, rcut+Delta]: attractive dispersion switched by (1 - S).
        // E = -(1-S) lj4 r^-6 ; fpair = -lj4[ S'(t)/Delta r^-7 + 6 (1-S) r^-8 ].
        // The kspace style supplies the plane S*u tail (used by pppm/disp/slab,
        // which has no real-space shell correction).
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

double PairLJCutDispSwitch::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                                   double /*factor_coul*/, double factor_lj, double &fforce)
{
  if (rsq >= cutsq[itype][jtype]) {
    fforce = 0.0;
    return 0.0;
  }
  const double r2inv = 1.0 / rsq;
  const double r6inv = r2inv * r2inv * r2inv;
  double phi, forcelj;
  if (rsq < inner_rc2 || csb_full_shell) {
    // full LJ (csb_full_shell evaluates the repulsion + full dispersion to rcut+Delta;
    // the kspace plane mean-field S*u is removed by corr_csb)
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

/* ---------------------------------------------------------------------- */

void *PairLJCutDispSwitch::extract(const char *str, int &dim)
{
  if (strcmp(str, "disp_switch_width") == 0) {
    dim = 0;
    return (void *) &sw_width;
  }
  if (strcmp(str, "csb_full_shell") == 0) {
    dim = 0;
    return (void *) &csb_full_shell;
  }
  return PairLJCut::extract(str, dim);    // cut_lj -> rcut (inner), B, epsilon, sigma
}
