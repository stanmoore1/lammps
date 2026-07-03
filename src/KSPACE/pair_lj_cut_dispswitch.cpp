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

double PairLJCutDispSwitch::init_one(int i, int j)
{
  PairLJCut::init_one(i, j);    // mix epsilon/sigma -> lj1..lj4
  cut[i][j] = cut[j][i] = cut_global + sw_width;
  offset[i][j] = offset[j][i] = 0.0;    // kspace continues the tail: no shift
  inner_rc2 = cut_global * cut_global;
  inv_sw_width = 1.0 / sw_width;    // precomputed for the hot loop (NPT-safe: Delta fixed)
  return cut_global + sw_width;
}

/* ----------------------------------------------------------------------
   the matched kspace style handles the 1/r^6 dispersion beyond the cutoff, so
   the analytic long-range tail correction would double count it
------------------------------------------------------------------------- */

void PairLJCutDispSwitch::init_style()
{
  if (tail_flag)
    error->all(FLERR, "Pair style lj/cut/dispswitch is incompatible with pair_modify tail yes "
                      "(the dispersion tail is handled by the matched kspace style)");
  PairLJCut::init_style();
}

/* ----------------------------------------------------------------------
   proc 0 writes/reads to restart file.  The base writes cut_global etc.; the
   switch width Delta must be persisted too, or it resets to 0 on restart (which
   collapses the cutoff to the inner rcut and aborts the matched kspace).
------------------------------------------------------------------------- */

void PairLJCutDispSwitch::write_restart_settings(FILE *fp)
{
  PairLJCut::write_restart_settings(fp);
  fwrite(&sw_width, sizeof(double), 1, fp);
}

void PairLJCutDispSwitch::read_restart_settings(FILE *fp)
{
  PairLJCut::read_restart_settings(fp);
  if (comm->me == 0) utils::sfread(FLERR, &sw_width, sizeof(double), 1, fp, nullptr, error);
  MPI_Bcast(&sw_width, 1, MPI_DOUBLE, 0, world);
  inv_sw_width = 1.0 / sw_width;
}

/* ---------------------------------------------------------------------- */

void PairLJCutDispSwitch::compute(int eflag, int vflag)
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
  const double invsw = inv_sw_width;      // 1/Delta

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

        // shell [rcut, rcut+Delta]: attractive dispersion switched by (1 - S).
        // E = -(1-S) lj4 r^-6 ; fpair = -lj4[ S'(t)/Delta r^-7 + 6 (1-S) r^-8 ].
        // The kspace style supplies the plane S*u tail (used by pppm/disp/slab,
        // which has no real-space shell correction).  Here t is guaranteed in
        // [0,1) (rcut <= r < rcut+Delta), so the switch polynomials are inlined
        // without the end clamps, and all divisions are replaced by the
        // precomputed 1/Delta and rinv = sqrt(r2inv) (no 1/r, no /Delta).
        const double rinv = sqrt(r2inv);
        const double r = rsq * rinv;              // = sqrt(rsq), one mul not a div
        const double t = (r - rc_inner) * invsw;
        const double t2 = t * t, t3 = t2 * t, t4 = t3 * t;
        const double S = t4 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)));    // Horner
        const double u = 1.0 - t;
        const double dS = 140.0 * t3 * u * u * u;    // dS/dt
        const double oneMinusS = 1.0 - S;
        const double lj4ij = lj4[itype][jtype];
        fpair = -factor_lj * lj4ij * r6inv *
            ((dS * invsw) * rinv + 6.0 * oneMinusS * r2inv);
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
  if (rsq < inner_rc2) {
    // full LJ to rcut (the matched kspace continues the dispersion tail)
    forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
    fforce = factor_lj * forcelj * r2inv;
    phi = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]);
  } else {
    const double rinv = sqrt(r2inv), r = rsq * rinv;
    const double t = (r - cut_global) * inv_sw_width;
    const double S = sw_S(t), dS = sw_dS(t), oneMinusS = 1.0 - S;
    const double lj4ij = lj4[itype][jtype];
    fforce = -factor_lj * lj4ij * r6inv * ((dS * inv_sw_width) * rinv + 6.0 * oneMinusS * r2inv);
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
  if (strcmp(str, "ewald_mix") == 0) {    // C6 mixing rule for the matched kspace style
    dim = 0;
    return (void *) &mix_flag;
  }
  return PairLJCut::extract(str, dim);    // cut_lj -> rcut (inner), B, epsilon, sigma
}
