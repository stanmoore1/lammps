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
PairStyle(spin/dipole/long,PairSpinDipoleLong);
// clang-format on
#else

#ifndef LMP_PAIR_SPIN_DIPOLE_LONG_H
#define LMP_PAIR_SPIN_DIPOLE_LONG_H

#include "pair_spin_dipole_cut.h"

namespace LAMMPS_NS {

class PairSpinDipoleLong : public PairSpinDipoleCut {
 public:
  PairSpinDipoleLong(class LAMMPS *);
  ~PairSpinDipoleLong() override = default;

  void compute(int, int) override;
  void compute_single_pair(int, double *) override;
  void init_style() override;
  void *extract(const char *, int &) override;

  void compute_dipolar_long(int, int, double *, double *, double *, double *,
                            double, double);
  void compute_dipolar_mech_long(int, int, double *, double *, double *, double *,
                                 double, double);
};

}    // namespace LAMMPS_NS

#endif
#endif
