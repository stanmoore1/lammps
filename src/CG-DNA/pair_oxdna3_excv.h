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
PairStyle(oxdna3/excv,PairOxdna3Excv);
// clang-format on
#else

#ifndef LMP_PAIR_OXDNA3_EXCV_H
#define LMP_PAIR_OXDNA3_EXCV_H

#include "pair_oxdna2_excv.h"

namespace LAMMPS_NS {

class PairOxdna3Excv : public PairOxdna2Excv {
 public:
  PairOxdna3Excv(class LAMMPS *lmp) : PairOxdna2Excv(lmp) {}
  void compute_base_site(int, double *, double *, double *, double *) const override;
  void coeff(int, char **) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
