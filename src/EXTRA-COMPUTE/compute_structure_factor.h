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
  void compute_vector() override;

 protected:
  int kxmax, kymax, kzmax;
  int kcount, kmax, kmax2d, kmax_created;
  double gsqmx, volume;
  int nmax, nbins;

  double unitk[2];
  int *kxvecs, *kyvecs, *kzvecs;
  int kxmax_orig, kymax_orig, kzmax_orig;
  double **sfacrl, **sfacim, **sfacrl_all, **sfacim_all;
  double ***cs, ***sn;

  virtual void eik_dot_r();
  virtual void coeffs();
  void atom2bin1d();
  virtual void allocate();
  virtual void deallocate();

  int *bins;
};

}    // namespace LAMMPS_NS

#endif
#endif
