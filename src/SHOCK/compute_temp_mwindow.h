/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   Subroutine made especially for Moving Window technique

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: You Lin (USF)
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(temp/mwindow,ComputeTempMWindow)

#else

#ifndef COMPUTE_TEMP_MWINDOW_H
#define COMPUTE_TEMP_MWINDOW_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeTempMWindow : public Compute {
 public:
  ComputeTempMWindow(class LAMMPS *, int, char **);
  ~ComputeTempMWindow() override;
  void init() override;
  double compute_scalar() override;
  void compute_vector() override;

  void remove_bias(int, double *) override;
  void remove_bias_all() override;
  void restore_bias(int, double *) override;
  void restore_bias_all() override;

 protected:
  int fix_dof;
  double tfactor,masstotal;
  double vbias[3];    // stored velocity bias for one atom

  void dof_compute();

};

}

#endif
#endif
