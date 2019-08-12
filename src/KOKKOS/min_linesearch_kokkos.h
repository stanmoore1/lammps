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

#ifndef LMP_MIN_LSRCH_KOKKOS_H
#define LMP_MIN_LSRCH_KOKKOS_H

#include "min_kokkos.h"

namespace LAMMPS_NS {

class MinLineSearch : public Min {
 public:
  MinLineSearch(class LAMMPS *);
  ~MinLineSearch();
  void init();
  void setup_style();
  void reset_vectors();

 protected:
  // vectors needed by linesearch minimizers
  // allocated and stored by fix_minimize
  // x,f are stored by parent or Atom class or Pair class

  double d_x0;                 // coords at start of linesearch
  double d_g;                  // old gradient vector
  double d_h;                  // search direction vector

  typedef int (MinLineSearchKokkos::*FnPtr)(double, double &);
  FnPtr linemin;
  int linemin_quadratic(double, double &);

  double alpha_step(double, int);
  double compute_dir_deriv(double &);
};

}

#endif
