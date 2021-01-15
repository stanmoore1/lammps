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

FixStyle(xlmd/reax,FixXLMDReax)

#else

#ifndef LMP_FIX_XLMD_REAX_H
#define LMP_FIX_XLMD_REAX_H

#include "fix_qeq_reax.h"

namespace LAMMPS_NS {

class FixXLMDReax : public FixQEqReax {
 public:
  FixXLMDReax(class LAMMPS *, int, char **);
  virtual ~FixXLMDReax();
  void post_constructor();
  int setmask();
  void init();
  void initial_integrate(int);
  void final_integrate();
  void reset_dt();
  void setup_pre_force(int);
  void pre_force(int);

 private:

  void calculate_XLMD();
  double kinetic_latent();
  void Berendersen(const double);
  void Langevin(const double);

  double dtv,dtf,dth;

  int xlmd_flag; // 0 -> C-XLMD, 1 -> B/C-XLMD, 2 -> S/C-XLMD
  int setup_flag;
  double *qLatent,*pLatent,*fLatent;
  double mLatent, tLatent, tauLatent; // mass, temperature and coupling parameter
  double qConst; // Chemical potential

  class RanMars *random;
  int seed;

  void calculate_Q();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
};

}

#endif
#endif
