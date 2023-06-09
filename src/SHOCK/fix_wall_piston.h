/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(wall/piston,FixWallPiston);
// clang-format on
#else

#ifndef FIX_WALL_PISTON_H
#define FIX_WALL_PISTON_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallPiston : public Fix {
 public:
  FixWallPiston(class LAMMPS *, int, char **);
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double compute_scalar() override;
  double compute_vector(int) override;

 private:
  int dim,side;
  double coord,Edeep3,Rdeep,cutoff;
  double offset;
  double wall[4],wall_all[4];
  int wall_flag;
  int nlevels_respa;
  int ifix_mw;
  class Fix *fix_mw; // fix_mw

  int xloflag, xhiflag, yloflag, yhiflag, zloflag, zhiflag;
  int scaleflag, roughflag, rampflag, rampNL1flag, rampNL2flag, rampNL3flag, rampNL4flag,
      rampNL5flag;
  double roughdist, roughoff, x0, y0, z0, vx, vy, vz, maxvx, maxvy, maxvz, paccelx, paccely,
      paccelz, angfreq;
  int tempflag, tseed;
  double t_target, t_period, t_extent;
  class RanMars *randomt;
  double *gfactor1, *gfactor2;
};

}    // namespace LAMMPS_NS

#endif
#endif

