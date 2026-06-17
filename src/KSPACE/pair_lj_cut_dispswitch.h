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

// Matched short-range pair style for the slab dispersion solvers with
// "kspace_modify damp compact".  The full LJ is computed to the inner cutoff
// rcut.  Over the shell [rcut, rcut+Delta] the 1/r^6 dispersion is handled in one
// of two ways, selected by the matched kspace via extract("csb_full_shell"):
//   0 (default, e.g. pppm/disp/slab): the attractive dispersion is switched off
//     by (1 - S(r)); the reciprocal sum supplies the plane S(r)*u there.
//   1 (ewald/disp/slab): the FULL dispersion u is evaluated (exact 3-D); the
//     reciprocal sum's plane mean-field S*u over the shell is removed by its
//     corr_csb(), so the pair gives the laterally-correlated shell interaction
//     and the lateral-correlation residual in energy/pressure is eliminated.
// S is the same C3 septic smoothstep used by the kspace style.

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
  // shell [rcut, rcut+Delta] treatment, set via extract("csb_full_shell") by the
  // matched kspace: 0 = (1-S)*u complement (kspace supplies the plane S*u, e.g.
  // pppm/disp/slab); 1 = full u (ewald/disp/slab removes the plane S*u in corr_csb
  // so the pair gives the exact 3-D shell interaction).
  int csb_full_shell;
  static double sw_S(double t);
  static double sw_dS(double t);
};

}    // namespace LAMMPS_NS

#endif
#endif
