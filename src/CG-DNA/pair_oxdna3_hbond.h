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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(oxdna3/hbond,PairOxdna3Hbond);
// clang-format on
#else

#ifndef LMP_PAIR_OXDNA3_HBOND_H
#define LMP_PAIR_OXDNA3_HBOND_H

#include "pair_oxdna_hbond.h"

namespace LAMMPS_NS {

class PairOxdna3Hbond : public PairOxdnaHbond {
 public:
  PairOxdna3Hbond(class LAMMPS *lmp) : PairOxdnaHbond(lmp) {}
  void compute_base_site(int, double *, double *, double *, double *) const override;
  void coeff(int, char **) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
