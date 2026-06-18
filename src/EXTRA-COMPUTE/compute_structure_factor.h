/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(structure/factor,ComputeStructureFactor);
// clang-format on
#else

#ifndef LMP_COMPUTE_STRUCTURE_FACTOR_H
#define LMP_COMPUTE_STRUCTURE_FACTOR_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeStructureFactor : public Compute {
 public:
  ComputeStructureFactor(class LAMMPS *, int, char **);
  ~ComputeStructureFactor() override;
  void init() override;
  void setup() override;
  void compute_array() override;

 protected:
  int kxmax, kymax, kzmax;
  int kcount, kmax, kmax3d, kmax_created;
  double gsqmx, volume;
  int nmax;

  double unitk[3];
  int *kxvecs, *kyvecs, *kzvecs;
  int kxmax_orig, kymax_orig, kzmax_orig;
  double *sfacrl, *sfacim, *sfacrl_all, *sfacim_all;
  double ***cs, ***sn;

  virtual void eik_dot_r();
  virtual void coeffs();
  virtual void allocate();
  virtual void deallocate();

  // triclinic

  int triclinic;
  void eik_dot_r_triclinic();
  void coeffs_triclinic();
  void x2lamdaT(double *, double *);
  void lamda2xT(double *, double *);
  void lamda2xvector(double *, double *);

  int kunique;
  int *norms;
  double *weights;
  int *ksq2unique;
};

}    // namespace LAMMPS_NS

#endif
#endif
