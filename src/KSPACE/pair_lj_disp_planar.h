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
PairStyle(lj/disp/planar,PairLJDispPlanar);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_DISP_PLANAR_H
#define LMP_PAIR_LJ_DISP_PLANAR_H

#include "pair_lj_cut.h"

namespace LAMMPS_NS {

// Matched short-range pair style for the planar dispersion solvers ewald/disp/planar
// and pppm/disp/planar.  The full LJ is computed to the inner cutoff rcut.  Over the
// shell [rcut, rcut+Delta] the attractive 1/r^6 dispersion is switched off by
// (1-S(r)) with S the C3 septic smoothstep, and the matched reciprocal sum
// supplies the plane S(r)*u tail (folded into its influence-function correction).
// The switch only splits the 1/r^6 dispersion; the 1/r^12 repulsion is short-
// range and is always evaluated in full.  S here matches the kspace smoothstep.

class PairLJDispPlanar : public PairLJCut {
 public:
  PairLJDispPlanar(class LAMMPS *);
  void compute(int, int) override;
  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void *extract(const char *, int &) override;
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  double sw_width;        // switch width Delta
  double inv_sw_width;    // 1/Delta, precomputed for the hot loop
  double inner_rc2;       // rcut^2 (inner boundary; full LJ for r < rcut)
  static double sw_S(double t);
  static double sw_dS(double t);
};

}    // namespace LAMMPS_NS

#endif
#endif
