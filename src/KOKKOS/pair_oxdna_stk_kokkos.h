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
PairStyle(oxdna/stk/kk,PairOxdnaStkKokkos<LMPDeviceType>);
PairStyle(oxdna/stk/kk/device,PairOxdnaStkKokkos<LMPDeviceType>);
PairStyle(oxdna/stk/kk/host,PairOxdnaStkKokkos<LMPHostType>);
PairStyle(oxdna2/stk/kk,PairOxdnaStkKokkos<LMPDeviceType>);
PairStyle(oxdna2/stk/kk/device,PairOxdnaStkKokkos<LMPDeviceType>);
PairStyle(oxdna2/stk/kk/host,PairOxdnaStkKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_OXDNA_STK_KOKKOS_H
#define LMP_PAIR_OXDNA_STK_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "pair_oxdna_stk.h"
#include "neigh_list_kokkos.h"

#include "mf_oxdna_kokkos.h"
#include "atom_vec_ellipsoid_kokkos.h"

namespace LAMMPS_NS {

template<int NEIGHFLAG, int NEWTON_BOND, int EVFLAG>
struct TagPairOxdnaStkCompute{};

template<class DeviceType>
class PairOxdnaStkKokkos : public PairOxdnaStk, public KokkosBase {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairOxdnaStkKokkos(class LAMMPS *);
  ~PairOxdnaStkKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style();
  double init_one(int, int) override;

  template<int NEIGHFLAG, int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaStkCompute<NEIGHFLAG,NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaStkCompute<NEIGHFLAG,NEWTON_BOND,EVFLAG>, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j, const int &nlocal, const int &newton_bond,\
      const F_FLOAT &evdwl, const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,\
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const;

 protected:

  class NeighborKokkos *neighborKK;
  class mfOxdnaKokkos<DeviceType> *mfOxdnaKK;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_f_array torque;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_2d bondlist;
  typename AT::t_tagint_1d tag;
  typename AT::t_tagint_1d id5p;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int neighflag, nbondlist;
  int nlocal, newton_bond, eflag, vflag;

  typename AT::t_neighbors_2d_randomread d_neighbors;
  typename AT::t_int_1d_randomread d_alist;
  typename AT::t_int_1d_randomread d_numneigh;

  // stacking interaction parameters
  typename AT::tdual_ffloat_2d k_epsilon_st, k_a_st, k_cut_st_0, k_cut_st_c;
  typename AT::tdual_ffloat_2d k_cut_st_lo, k_cut_st_hi;
  typename AT::tdual_ffloat_2d k_cut_st_lc, k_cut_st_hc, k_b_st_lo, k_b_st_hi;
  typename AT::tdual_ffloat_2d k_shift_st, k_cutsq_st_hc;
  typename AT::tdual_ffloat_2d k_a_st4, k_theta_st4_0, k_dtheta_st4_ast;
  typename AT::tdual_ffloat_2d k_b_st4, k_dtheta_st4_c;
  typename AT::tdual_ffloat_2d k_a_st5, k_theta_st5_0, k_dtheta_st5_ast;
  typename AT::tdual_ffloat_2d k_b_st5, k_dtheta_st5_c;
  typename AT::tdual_ffloat_2d k_a_st6, k_theta_st6_0, k_dtheta_st6_ast;
  typename AT::tdual_ffloat_2d k_b_st6, k_dtheta_st6_c;
  typename AT::tdual_ffloat_2d k_a_st1, k_cosphi_st1_ast, k_b_st1, k_cosphi_st1_c;
  typename AT::tdual_ffloat_2d k_a_st2, k_cosphi_st2_ast, k_b_st2, k_cosphi_st2_c;
  typename AT::t_ffloat_2d d_epsilon_st, d_a_st, d_cut_st_0, d_cut_st_c;
  typename AT::t_ffloat_2d d_cut_st_lo, d_cut_st_hi;
  typename AT::t_ffloat_2d d_cut_st_lc, d_cut_st_hc, d_b_st_lo, d_b_st_hi;
  typename AT::t_ffloat_2d d_shift_st, d_cutsq_st_hc;
  typename AT::t_ffloat_2d d_a_st4, d_theta_st4_0, d_dtheta_st4_ast;
  typename AT::t_ffloat_2d d_b_st4, d_dtheta_st4_c;
  typename AT::t_ffloat_2d d_a_st5, d_theta_st5_0, d_dtheta_st5_ast;
  typename AT::t_ffloat_2d d_b_st5, d_dtheta_st5_c;
  typename AT::t_ffloat_2d d_a_st6, d_theta_st6_0, d_dtheta_st6_ast;
  typename AT::t_ffloat_2d d_b_st6, d_dtheta_st6_c;
  typename AT::t_ffloat_2d d_a_st1, d_cosphi_st1_ast, d_b_st1, d_cosphi_st1_c;
  typename AT::t_ffloat_2d d_a_st2, d_cosphi_st2_ast, d_b_st2, d_cosphi_st2_c;
  // per-atom arrays for local unit vectors
  DAT::tdual_x_array k_nx_xtrct, k_ny_xtrct, k_nz_xtrct;
  typename AT::t_x_array d_nx_xtrct, d_ny_xtrct, d_nz_xtrct;

  int first;
  typename AT::t_int_1d d_sendlist;
  typename AT::t_xfloat_1d_um v_buf;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, \
  KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, \
  KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<F_FLOAT*[3], typename AT::t_f_array::array_layout> dup_f;
  DupScatterView<F_FLOAT*[3], typename AT::t_f_array::array_layout> dup_torque;
  DupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout> dup_eatom;
  DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> dup_vatom;
  NonDupScatterView<F_FLOAT*[3], typename AT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<F_FLOAT*[3], typename AT::t_f_array::array_layout> ndup_torque;
  NonDupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout> ndup_eatom;
  NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  void allocate() override;
 
  friend void pair_virial_fdotr_compute<PairOxdnaStkKokkos>(PairOxdnaStkKokkos*);


};

}

#endif
#endif

