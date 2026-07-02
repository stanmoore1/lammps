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

// Matched short-range pair style for the slab dispersion solvers.  rcut (the
// pair_style argument) is the TOTAL (outer) interaction cutoff -- the pair never
// interacts beyond it, so this style shares one neighbor cutoff/list with plain
// lj/cut.  The C3 switch S(r) ramps inward over the shell [rcut-Delta, rcut]
// (S=0 at rcut-Delta, S=1 at rcut); the matched kspace reads the INNER boundary
// rcut-Delta via extract("cut_lj") and reconstructs its own outer boundary as
// cutoff+Delta = rcut, so no kspace-side change is needed for this convention
// (mirrors the planar lj/cut/dispplanar geometry).  Full LJ is always evaluated
// for r < rcut-Delta.  Over the shell, the split is controlled by the flag
// csb_full_shell, set by the matched kspace through extract("csb_full_shell"):
//   1 ("kspace_modify damp compact", the compact-switch variant): full LJ
//     (repulsion + full dispersion, exact 3-D) is evaluated over the whole
//     [0, rcut] range; the reciprocal sum's plane mean-field S*u over the shell
//     is removed by the kspace corr_csb(), so the pair supplies the exact
//     laterally-correlated shell interaction.  compute() delegates to PairLJCut.
//   0 ("kspace_modify damp yes" with the smooth switched corr, corr_switch):
//     over the shell the attractive dispersion is switched off by (1-S(r)),
//     smoothly reaching zero exactly at the pair's cutoff rcut; the reciprocal
//     sum supplies the plane S(r)*u there, folded into the influence function
//     with no separate real-space shell correction.
// The switch only splits the 1/r^6 dispersion; the 1/r^12 repulsion is short-
// range and is always evaluated in full.

class PairLJCutDispSwitch : public PairLJCut {
 public:
  PairLJCutDispSwitch(class LAMMPS *);
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
  double inner_cut;    // inner cutoff rcut-Delta (where the switch starts, S=0)
  double inner_rc2;    // inner_cut^2; full LJ for r < inner_cut
  // shell [rcut-Delta, rcut] treatment, set via extract("csb_full_shell") by the
  // matched kspace: 1 = full LJ (compact switch; kspace removes the plane S*u in
  // corr_csb), 0 = (1-S)*u complement (smooth switched corr, folded into kspace).
  int csb_full_shell;
  static double sw_S(double t);
  static double sw_dS(double t);
};

}    // namespace LAMMPS_NS

#endif
#endif
