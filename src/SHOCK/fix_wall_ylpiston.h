/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(wall/ylpiston,FixWallYLPiston);

#else

#ifndef FIX_WALL_YLPISTON
#define FIX_WALL_YLPISTON

#include "fix.h"

namespace LAMMPS_NS {

class FixWallYLPiston : public Fix {
 public:
  FixWallYLPiston(class LAMMPS *, int, char **);
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double compute_scalar() override;
  double compute_vector(int) override;

 protected:
  int dim,side;
  double coord,Edeep3,Rdeep,cutoff;
  double offset;
  double wall[4],wall_all[4];
  int wall_flag;
  int nlevels_respa;
  int ifix_mw;

  class Fix *fix_mw; // fix_mw
};

}

#endif
#endif

