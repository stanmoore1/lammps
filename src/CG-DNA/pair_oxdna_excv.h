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
PairStyle(oxdna/excv,PairOxdnaExcv);
// clang-format on
#else

#ifndef LMP_PAIR_OXDNA_EXCV_H
#define LMP_PAIR_OXDNA_EXCV_H

#include "pair.h"

namespace LAMMPS_NS {

class PairOxdnaExcv : public Pair {
 public:
  PairOxdnaExcv(class LAMMPS *);
  ~PairOxdnaExcv() override;
  virtual void compute_backbone_site(double *, double *, double *, double *) const;
  template <int N>
  void compute_base_site(double *, double *, double *, double *) const;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_list(int, class NeighList *) override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void *extract(const char *, int &) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 protected:
  // s=sugar-phosphate backbone site, b=base site, st=stacking site

  // excluded volume interaction
  // base step-dependent coefficients
  double **epsilon_ss, **sigma_ss, **cut_ss_ast, **cutsq_ss_ast;
  double **lj1_ss, **lj2_ss, **b_ss, **cut_ss_c, **cutsq_ss_c;
  double **epsilon_sb, **sigma_sb, **cut_sb_ast, **cutsq_sb_ast;
  double **lj1_sb, **lj2_sb, **b_sb, **cut_sb_c, **cutsq_sb_c;
  double **epsilon_bb, **sigma_bb, **cut_bb_ast, **cutsq_bb_ast;
  double **lj1_bb, **lj2_bb, **b_bb, **cut_bb_c, **cutsq_bb_c;
  // tetramer-dependent coefficients
  double ****sigma4_sb, ****cut4_sb_ast, ****cut4sq_sb_ast;
  double ****lj14_sb, ****lj24_sb, ****b4_sb, ****cut4_sb_c, ****cut4sq_sb_c;
  double ****sigma4_bb, ****cut4_bb_ast, ****cut4sq_bb_ast;
  double ****lj14_bb, ****lj24_bb, ****b4_bb, ****cut4_bb_c, ****cut4sq_bb_c;

  double **nx, **ny, **nz;    // per-atom arrays for local unit vectors
  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
