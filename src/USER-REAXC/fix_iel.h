/* -*- c++ -*- ----------------------------------------------------------
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

FixStyle(iel,FixIEL)

#else

#ifndef LMP_FIX_IEL_H
#define LMP_FIX_IEL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixIEL : public Fix {
 public:
  FixIEL(class LAMMPS *, int, char **);
  virtual ~FixIEL() {}
  int setmask();
  void init();
  void initial_integrate(int);
  void final_integrate();
  void reset_dt();

  void get_names(char *,double *&);
  double kinetic_latent();
  void Berendersen(const double);
  void Langevin(const double);
  double parallel_vector_acc(double *);

 private:
  double dtv,dtf,dth;

  class RanMars *random;
  int seed;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
