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
// ewald/disp/planar and pppm/disp/planar.  It is a plain lj/cut evaluated to the
// extended cutoff rcut+Delta (the inner cutoff rcut plus the switch width Delta),
// with no energy offset.  The matched kspace style applies the C3 septic
// smoothstep S(r) over the shell [rcut, rcut+Delta]: its reciprocal sum carries
// S*u and a shell correction subtracts that mean field over the shell, so the
// full 1/r^6 dispersion this pair computes there is replaced by the exact 3-D
// laterally-correlated interaction.  This pair therefore exposes the inner cutoff
// rcut (cut_lj), the switch width Delta (disp_switch_width), and the dispersion
// amplitude B to the kspace via extract(); it must be paired with one of the
// planar kspace styles (it does not apply the switch itself).

class PairLJCutDispPlanar : public PairLJCut {
 public:
  PairLJCutDispPlanar(class LAMMPS *);
  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void *extract(const char *, int &) override;

 protected:
  double sw_width;     // switch width Delta
};

}    // namespace LAMMPS_NS

#endif
#endif
