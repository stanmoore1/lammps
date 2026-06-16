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
PairStyle(lj/cut/dispswitch,PairLJCutDispSwitch);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_DISPSWITCH_H
#define LMP_PAIR_LJ_CUT_DISPSWITCH_H

#include "pair_lj_cut.h"

namespace LAMMPS_NS {

// Matched short-range pair style for kspace_style ewald/disp/slab with
// "kspace_modify damp compact".  The full LJ is computed to the inner cutoff
// rcut; the attractive 1/r^6 dispersion is then smoothly switched off over the
// shell [rcut, rcut+Delta] by the factor (1 - S(r)), so that pair + kspace
// (which supplies S(r)*tail) reproduce the full dispersion.  S is the same C3
// septic smoothstep used by the kspace style.

class PairLJCutDispSwitch : public PairLJCut {
 public:
  PairLJCutDispSwitch(class LAMMPS *);
  void compute(int, int) override;
  void settings(int, char **) override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  double sw_width;     // switch width Delta
  double inner_rc2;    // rcut^2 (inner boundary; full LJ for r < rcut)
  static double sw_S(double t);
  static double sw_dS(double t);
};

}    // namespace LAMMPS_NS

#endif
#endif
