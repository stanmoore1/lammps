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

FixStyle(wall/piston,FixWallPiston)

#else

#ifndef FIX_WALL_PISTON_H
#define FIX_WALL_PISTON_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallPiston : public Fix {
 public:
  FixWallPiston(class LAMMPS *, int, char **);
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);

 private:
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

