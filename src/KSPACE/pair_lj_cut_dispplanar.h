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
PairStyle(lj/cut/dispplanar,PairLJCutDispPlanar);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_DISPPLANAR_H
#define LMP_PAIR_LJ_CUT_DISPPLANAR_H

#include "pair_lj_cut.h"

namespace LAMMPS_NS {

// Matched short-range pair style for the planar dispersion solvers
// ewald/disp/planar and pppm/disp/planar.  The full LJ is computed to the inner
// cutoff rcut.  Over the shell [rcut, rcut+Delta] the 1/r^6 dispersion is split
// between this pair and the reciprocal sum by the C3 septic smoothstep S(r), via
// the flag full_shell set by the matched kspace through extract("disp_full_shell"):
//   1 (the live path -- both ewald/disp/planar and pppm/disp/planar set this):
//     full LJ is evaluated over the whole [0, rcut+Delta] range (repulsion + full
//     dispersion, exact 3-D); the reciprocal sum's plane mean-field S*u over the
//     shell is removed by the kspace shell correction, so the pair supplies the
//     exact laterally-correlated shell interaction.  compute() delegates to PairLJCut.
//   0 (fallback, unused by the shipped kspace styles): over the shell the
//     attractive dispersion is switched off by (1-S(r)) and the reciprocal sum
//     supplies the plane S(r)*u with no real-space shell correction.
// The switch only splits the 1/r^6 dispersion; the 1/r^12 repulsion is short-
// range and is always evaluated in full.  S is the C3 (septic) smoothstep
// S(t) = t^4(35 - 84t + 70t^2 - 20t^3), matching the planar kspace styles.

class PairLJCutDispPlanar : public PairLJCut {
 public:
  PairLJCutDispPlanar(class LAMMPS *);
  void compute(int, int) override;
  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void *extract(const char *, int &) override;
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  double sw_width;     // switch width Delta
  double inner_rc2;    // rcut^2 (inner boundary; full LJ for r < rcut)
  // shell [rcut, rcut+Delta] treatment, set via extract("disp_full_shell") by the
  // matched kspace: 1 = full LJ (the live path; kspace removes the plane S*u in
  // its shell correction), 0 = (1-S)*u complement fallback (unused by the shipped
  // kspace styles).
  int full_shell;
  static double sw_S(double t);
  static double sw_dS(double t);
};

}    // namespace LAMMPS_NS

#endif
#endif
