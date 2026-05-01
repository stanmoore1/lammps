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
PairStyle(oxdna/hbond/kk,PairOxdnaHbondKokkos<LMPDeviceType>);
PairStyle(oxdna/hbond/kk/device,PairOxdnaHbondKokkos<LMPDeviceType>);
PairStyle(oxdna/hbond/kk/host,PairOxdnaHbondKokkos<LMPHostType>);
PairStyle(oxdna2/hbond/kk,PairOxdnaHbondKokkos<LMPDeviceType>);
PairStyle(oxdna2/hbond/kk/device,PairOxdnaHbondKokkos<LMPDeviceType>);
PairStyle(oxdna2/hbond/kk/host,PairOxdnaHbondKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_OXDNA_HBOND_KOKKOS_H
#define LMP_PAIR_OXDNA_HBOND_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "pair_oxdna_hbond.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixOxdnaLRFKokkos;  // forward declaration

template<class DeviceType>
class FixOxdnaNpairKokkos;  // forward declaration

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairOxdnaHbondCompute{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairOxdnaHbondComputeGPUPair{};

template<class DeviceType>
class PairOxdnaHbondKokkos : public PairOxdnaHbond, public KokkosBase {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairOxdnaHbondKokkos(class LAMMPS *);
  ~PairOxdnaHbondKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  // Standard non-GPU Compute Functor(s). 1 with EV_FLOAT, 1 without.

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaHbondCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaHbondCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&) const;

// GPU ComputeGPUPair Functor(s). 1 with EV_FLOAT, 1 without.

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaHbondComputeGPUPair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaHbondComputeGPUPair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const KK_FLOAT &epair, const KK_FLOAT &fx, const KK_FLOAT &fy, const KK_FLOAT &fz, const KK_FLOAT &delx,
                  const KK_FLOAT &dely, const KK_FLOAT &delz) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

 protected:

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkfloat_1d_3 f;
  typename AT::t_kkfloat_1d_3 torque;
  typename AT::t_int_1d_randomread type;

  DAT::ttransform_kkfloat_1d k_eatom;
  DAT::ttransform_kkfloat_1d_6 k_vatom;
  typename AT::t_kkfloat_1d d_eatom;
  typename AT::t_kkfloat_1d_6 d_vatom;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_kkfloat_2d k_cutsq;
  typename AT::t_kkfloat_2d d_cutsq;

  int neighflag;
  int nlocal, eflag, vflag;
  int anum;

  typename AT::t_neighbors_2d_randomread d_neighbors;
  typename AT::t_int_1d_randomread d_alist;
  typename AT::t_int_1d_randomread d_numneigh;
  // GPU-specific: screened neighbor arrays for npair fix
  DAT::tdual_uint64_1d k_pairs_screened;
  typename AT::t_uint64_1d d_pairs_screened;
  int screened_pair_count;

  // hydrogen-bonding interaction parameters
  typename AT::tdual_kkfloat_2d k_epsilon_hb, k_a_hb, k_cut_hb_0, k_cut_hb_c;
  typename AT::tdual_kkfloat_2d k_cut_hb_lo, k_cut_hb_hi;
  typename AT::tdual_kkfloat_2d k_cut_hb_lc, k_cut_hb_hc, k_b_hb_lo, k_b_hb_hi;
  typename AT::tdual_kkfloat_2d k_shift_hb, k_cutsq_hb_hc;
  typename AT::tdual_kkfloat_2d k_a_hb1, k_theta_hb1_0, k_dtheta_hb1_ast;
  typename AT::tdual_kkfloat_2d k_b_hb1, k_dtheta_hb1_c;
  typename AT::tdual_kkfloat_2d k_a_hb2, k_theta_hb2_0, k_dtheta_hb2_ast;
  typename AT::tdual_kkfloat_2d k_b_hb2, k_dtheta_hb2_c;
  typename AT::tdual_kkfloat_2d k_a_hb3, k_theta_hb3_0, k_dtheta_hb3_ast;
  typename AT::tdual_kkfloat_2d k_b_hb3, k_dtheta_hb3_c;
  typename AT::tdual_kkfloat_2d k_a_hb4, k_theta_hb4_0, k_dtheta_hb4_ast;
  typename AT::tdual_kkfloat_2d k_b_hb4, k_dtheta_hb4_c;
  typename AT::tdual_kkfloat_2d k_a_hb7, k_theta_hb7_0, k_dtheta_hb7_ast;
  typename AT::tdual_kkfloat_2d k_b_hb7, k_dtheta_hb7_c;
  typename AT::tdual_kkfloat_2d k_a_hb8, k_theta_hb8_0, k_dtheta_hb8_ast;
  typename AT::tdual_kkfloat_2d k_b_hb8, k_dtheta_hb8_c;
  typename AT::t_kkfloat_2d d_epsilon_hb, d_a_hb, d_cut_hb_0, d_cut_hb_c;
  typename AT::t_kkfloat_2d d_cut_hb_lo, d_cut_hb_hi;
  typename AT::t_kkfloat_2d d_cut_hb_lc, d_cut_hb_hc, d_b_hb_lo, d_b_hb_hi;
  typename AT::t_kkfloat_2d d_shift_hb, d_cutsq_hb_hc;
  typename AT::t_kkfloat_2d d_a_hb1, d_theta_hb1_0, d_dtheta_hb1_ast;
  typename AT::t_kkfloat_2d d_b_hb1, d_dtheta_hb1_c;
  typename AT::t_kkfloat_2d d_a_hb2, d_theta_hb2_0, d_dtheta_hb2_ast;
  typename AT::t_kkfloat_2d d_b_hb2, d_dtheta_hb2_c;
  typename AT::t_kkfloat_2d d_a_hb3, d_theta_hb3_0, d_dtheta_hb3_ast;
  typename AT::t_kkfloat_2d d_b_hb3, d_dtheta_hb3_c;
  typename AT::t_kkfloat_2d d_a_hb4, d_theta_hb4_0, d_dtheta_hb4_ast;
  typename AT::t_kkfloat_2d d_b_hb4, d_dtheta_hb4_c;
  typename AT::t_kkfloat_2d d_a_hb7, d_theta_hb7_0, d_dtheta_hb7_ast;
  typename AT::t_kkfloat_2d d_b_hb7, d_dtheta_hb7_c;
  typename AT::t_kkfloat_2d d_a_hb8, d_theta_hb8_0, d_dtheta_hb8_ast;
  typename AT::t_kkfloat_2d d_b_hb8, d_dtheta_hb8_c;
  // per-atom arrays for local unit vectors
  DAT::tdual_kkfloat_1d_3 k_nx_xtrct, k_ny_xtrct, k_nz_xtrct;
  typename AT::t_kkfloat_1d_3 d_nx_xtrct, d_ny_xtrct, d_nz_xtrct;

  int first;
  typename AT::t_int_1d d_sendlist;
  typename AT::t_double_1d_um v_buf;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, \
  KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, \
  KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<KK_FLOAT*[3], typename AT::t_kkfloat_1d_3::array_layout> dup_f;
  DupScatterView<KK_FLOAT*[3], typename AT::t_kkfloat_1d_3::array_layout> dup_torque;
  DupScatterView<KK_FLOAT*, typename DAT::t_kkfloat_1d::array_layout> dup_eatom;
  DupScatterView<KK_FLOAT*[6], typename DAT::t_kkfloat_1d_6::array_layout> dup_vatom;
  NonDupScatterView<KK_FLOAT*[3], typename AT::t_kkfloat_1d_3::array_layout> ndup_f;
  NonDupScatterView<KK_FLOAT*[3], typename AT::t_kkfloat_1d_3::array_layout> ndup_torque;
  NonDupScatterView<KK_FLOAT*, typename DAT::t_kkfloat_1d::array_layout> ndup_eatom;
  NonDupScatterView<KK_FLOAT*[6], typename DAT::t_kkfloat_1d_6::array_layout> ndup_vatom;

  void allocate() override;
 
  friend void pair_virial_fdotr_compute<PairOxdnaHbondKokkos>(PairOxdnaHbondKokkos*);

  FixOxdnaLRFKokkos<DeviceType> *fix_oxdna_lrfKK;    // ptr to oxdna/lrf/kk fix
  FixOxdnaNpairKokkos<DeviceType> *fix_oxdna_npairKK;    // ptr to oxdna/npair/kk fix
};

}

#endif
#endif

