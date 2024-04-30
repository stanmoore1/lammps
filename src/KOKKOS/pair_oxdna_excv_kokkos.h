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
PairStyle(oxdna/excv/kk,PairOxdnaExcvKokkos<LMPDeviceType>);
PairStyle(oxdna/excv/kk/device,PairOxdnaExcvKokkos<LMPDeviceType>);
PairStyle(oxdna/excv/kk/host,PairOxdnaExcvKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_OXDNA_EXCV_KOKKOS_H
#define LMP_PAIR_OXDNA_EXCV_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_oxdna_excv.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairOxdnaExcvKokkos : public PairOxdnaExcv {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairOxdnaExcvKokkos(class LAMMPS *);
  ~PairOxdnaExcvKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style();
  double init_one(int, int) override;

 protected:
  typename AT::t_x_array_randomread x;
  //typename AT::t_x_array c_x;
  typename AT::t_f_array f;
  typename ArrayTypes<DeviceType>::t_f_array torque;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;

  int neighflag;
  int nlocal, eflag, vflag, anum, alist, blist, numneigh, firstneigh;

  // s=sugar-phosphate backbone site, b=base site, st=stacking site
  // excluded volume interaction
  typename AT::tdual_ffloat_2d k_epsilon_ss, k_sigma_ss, k_cut_ss_ast, k_cutsq_ss_ast;
  typename AT::tdual_ffloat_2d k_lj1_ss, k_lj2_ss, k_b_ss, k_cut_ss_c, k_cutsq_ss_c;
  typename AT::tdual_ffloat_2d k_epsilon_sb, k_sigma_sb, k_cut_sb_ast, k_cutsq_sb_ast;
  typename AT::tdual_ffloat_2d k_lj1_sb, k_lj2_sb, k_b_sb, k_cut_sb_c, k_cutsq_sb_c;
  typename AT::tdual_ffloat_2d k_epsilon_bb, k_sigma_bb, k_cut_bb_ast, k_cutsq_bb_ast;
  typename AT::tdual_ffloat_2d k_lj1_bb, k_lj2_bb, k_b_bb, k_cut_bb_c, k_cutsq_bb_c;
  typename AT::t_ffloat_2d d_epsilon_ss, d_sigma_ss, d_cut_ss_ast, d_cutsq_ss_ast;
  typename AT::t_ffloat_2d d_lj1_ss, d_lj2_ss, d_b_ss, d_cut_ss_c, d_cutsq_ss_c;
  typename AT::t_ffloat_2d d_epsilon_sb, d_sigma_sb, d_cut_sb_ast, d_cutsq_sb_ast;
  typename AT::t_ffloat_2d d_lj1_sb, d_lj2_sb, d_b_sb, d_cut_sb_c, d_cutsq_sb_c;
  typename AT::t_ffloat_2d d_epsilon_bb, d_sigma_bb, d_cut_bb_ast, d_cutsq_bb_ast;
  typename AT::t_ffloat_2d d_lj1_bb, d_lj2_bb, d_b_bb, d_cut_bb_c, d_cutsq_bb_c;
  // per-atom arrays for local unit vectors
  typename AT::tdual_ffloat_2d k_nx, k_ny, k_nz;
  typename AT::t_ffloat_2d d_nx, d_ny, d_nz;

  /*typename AT::t_ffloat_2d epsilon_ss, sigma_ss, cut_ss_ast, cutsq_ss_ast;
  typename AT::t_ffloat_2d lj1_ss, lj2_ss, b_ss, cut_ss_c, cutsq_ss_c;
  typename AT::t_ffloat_2d epsilon_sb, sigma_sb, cut_sb_ast, cutsq_sb_ast;
  typename AT::t_ffloat_2d lj1_sb, lj2_sb, b_sb, cut_sb_c, cutsq_sb_c;
  typename AT::t_ffloat_2d epsilon_bb, sigma_bb, cut_bb_ast, cutsq_bb_ast;
  typename AT::t_ffloat_2d lj1_bb, lj2_bb, b_bb, cut_bb_c, cutsq_bb_c;
  typename AT::t_ffloat_2d nx, ny, nz;    // per-atom arrays for local unit vectors*/

  void allocate() override;
  friend struct PairComputeFunctor<PairOxdnaExcvKokkos,FULL,true,0>;
  friend struct PairComputeFunctor<PairOxdnaExcvKokkos,FULL,true,1>;
  friend struct PairComputeFunctor<PairOxdnaExcvKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairOxdnaExcvKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairOxdnaExcvKokkos,FULL,false,0>;
  friend struct PairComputeFunctor<PairOxdnaExcvKokkos,FULL,false,1>;
  friend struct PairComputeFunctor<PairOxdnaExcvKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairOxdnaExcvKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairOxdnaExcvKokkos,FULL,0>(PairOxdnaExcvKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairOxdnaExcvKokkos,FULL,1>(PairOxdnaExcvKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairOxdnaExcvKokkos,HALF>(PairOxdnaExcvKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairOxdnaExcvKokkos,HALFTHREAD>(PairOxdnaExcvKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairOxdnaExcvKokkos>(PairOxdnaExcvKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairOxdnaExcvKokkos>(PairOxdnaExcvKokkos*);
};

}

#endif
#endif

