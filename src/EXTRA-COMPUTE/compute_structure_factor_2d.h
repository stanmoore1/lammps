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
ComputeStyle(structure/factor/2d,ComputeStructureFactor2D);
// clang-format on
#else

#ifndef LMP_COMPUTE_STRUCTURE_FACTOR_2D_H
#define LMP_COMPUTE_STRUCTURE_FACTOR_2D_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeStructureFactor2D : public Compute {
 public:
  ComputeStructureFactor2D(class LAMMPS *, int, char **);
  ~ComputeStructureFactor2D() override;
  void init() override;
  void setup() override;
  void compute_array() override;

 protected:
  int kxmax, kymax;
  int kcount, kmax, kmax2d, kmax_created;
  double gsqmx, volume;
  int nmax, nbins;

  double unitk[2];
  int *kxvecs, *kyvecs;
  int kxmax_orig, kymax_orig;
  double **sfacrl, **sfacim, **sfacrl_all, **sfacim_all;
  double ***cs, ***sn;

  virtual void eik_dot_r();
  virtual void coeffs();
  void atom2bin1d();
  virtual void allocate();
  virtual void deallocate();

  int kunique;
  int *norms;
  double *weights;
  int *ksq2unique;
  int *bins;
  int *counts, *counts_all;
};

}    // namespace LAMMPS_NS

#endif
#endif
