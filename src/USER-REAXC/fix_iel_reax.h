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

/* ----------------------------------------------------------------------
   Contributing author: Hasan Metin Aktulga, Purdue University
   (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)

   Please cite the related publication:
   H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
   "Parallel Reactive Molecular Dynamics: Numerical Methods and
   Algorithmic Techniques", Parallel Computing, in press.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(iel/reax,FixIELReax)

#else

#ifndef LMP_FIX_IEL_REAX_H
#define LMP_FIX_IEL_REAX_H

#include "fix_qeq_reax.h"

namespace LAMMPS_NS {

class FixIELReax : public FixQEqReax {
 public:
  FixIELReax(class LAMMPS *, int, char **);
  virtual ~FixIELReax();
  int setmask();
  void init();
  void initial_integrate(int);
  void final_integrate();
  void reset_dt();
  void setup_pre_force(int);
  void pre_force(int);
  void end_of_step();

 private:

  void calculate_XLMD();
  void get_names(char *,double *&);
  double kinetic_latent();
  void Berendersen(const double);
  void Langevin(const double);

  double dtv,dtf,dth;

  class RanMars *random;
  int seed;

  void calculate_Q();
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *); 
};

}

#endif
#endif
