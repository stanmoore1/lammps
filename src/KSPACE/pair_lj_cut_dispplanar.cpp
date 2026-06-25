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
#include "utils.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJCutDispPlanar::PairLJCutDispPlanar(LAMMPS *lmp) : PairLJCut(lmp)
{
  sw_width = 0.0;
  inner_cut = 0.0;
  respa_enable = 0;    // pair is a companion to the planar kspace; no rRESPA split
}

/* ----------------------------------------------------------------------
   pair_style lj/cut/dispplanar <rcut> <Delta>
   rcut is the total (outer) cutoff -- the same cutoff used by the other planar
   Ewald sums -- and the C3 switch ramps inward over [rcut-Delta, rcut].  The pair
   evaluates the full LJ to rcut; the matched kspace style places the switch on
   [rcut-Delta, rcut] using the inner cutoff (cut_lj = rcut-Delta) it reads here.
------------------------------------------------------------------------- */

void PairLJCutDispPlanar::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Illegal pair_style lj/cut/dispplanar command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  sw_width = utils::numeric(FLERR, arg[1], false, lmp);
  if (sw_width <= 0.0) error->all(FLERR, "pair_style lj/cut/dispplanar switch width must be > 0");
  if (sw_width >= cut_global)
    error->all(FLERR, "pair_style lj/cut/dispplanar switch width must be < the cutoff");
  inner_cut = cut_global - sw_width;    // where the switch starts (S=0)

  // reset per-type cutoffs (always global here)
  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   neighbor/interaction cutoff = rcut (the total cutoff); the switch ramps over
   the inner shell [rcut-Delta, rcut]; no energy shift
------------------------------------------------------------------------- */

double PairLJCutDispPlanar::init_one(int i, int j)
{
  PairLJCut::init_one(i, j);    // mix epsilon/sigma -> lj1..lj4
  cut[i][j] = cut[j][i] = cut_global;
  offset[i][j] = offset[j][i] = 0.0;    // kspace continues the tail: no shift
  return cut_global;
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
  inner_cut = cut_global - sw_width;
}

/* ----------------------------------------------------------------------
   expose the inner cutoff, dispersion amplitude, and switch parameters so the
   matched planar kspace style is self-contained (no edit to pair_lj_cut needed).
   The force/energy computation (full LJ to rcut+Delta) is inherited from PairLJCut.
------------------------------------------------------------------------- */

void *PairLJCutDispPlanar::extract(const char *str, int &dim)
{
  if (strcmp(str, "disp_switch_width") == 0) {
    dim = 0;
    return (void *) &sw_width;
  }
  if (strcmp(str, "cut_lj") == 0) {
    dim = 0;
    return (void *) &inner_cut;    // inner cutoff rcut-Delta (switch start)
  }
  if (strcmp(str, "B") == 0) {
    dim = 2;
    return (void *) lj4;    // 4*eps*sigma^6 (dispersion C6); B[i] = sqrt(|lj4[i][i]|)
  }
  return PairLJCut::extract(str, dim);    // epsilon, sigma
}
