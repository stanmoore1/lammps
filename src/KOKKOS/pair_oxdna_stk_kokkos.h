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

namespace LAMMPS_NS {

template<class DeviceType>
class FixOxdnaLRFKokkos;  // forward declaration

struct TagPairOxdnaStkPrecomputeBondPrimeNeighs{};

template<int NEWTON_BOND, int EVFLAG>
struct TagPairOxdnaStkCompute{};

template<class DeviceType>
class PairOxdnaStkKokkos : public PairOxdnaStk, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairOxdnaStkKokkos(class LAMMPS *);
  ~PairOxdnaStkKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaStkPrecomputeBondPrimeNeighs, const int&) const;

  template<int NEWTON_BOND, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaStkCompute<NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaStkCompute<NEWTON_BOND,EVFLAG>, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j, const int &nlocal, const int &newton_bond,\
      const KK_FLOAT &evdwl, const KK_FLOAT &fx, const KK_FLOAT &fy, const KK_FLOAT &fz,\
      const KK_FLOAT &delx, const KK_FLOAT &dely, const KK_FLOAT &delz) const;

 protected:

  class NeighborKokkos *neighborKK;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkfloat_1d_3 f;
  typename AT::t_kkfloat_1d_3 torque;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_2d_lr bondlist;
  typename AT::t_tagint_1d tag;
  typename AT::t_tagint_1d id5p;
  typename AT::t_tagint_1d id3p;

  DAT::ttransform_kkfloat_1d k_eatom;
  DAT::ttransform_kkfloat_1d_6 k_vatom;
  typename AT::t_kkfloat_1d d_eatom;
  typename AT::t_kkfloat_1d_6 d_vatom;

  int nbondlist;
  int nlocal, newton_bond, eflag, vflag;
  // bond->prime-neigh table only changes on reneighbor; cache the lastcall it
  // was built for so it is recomputed once per neighbor build, not every step.
  bigint last_precompute_lastcall = -1;

  // stacking interaction parameters
  typename AT::tdual_kkfloat_2d k_epsilon_st, k_a_st;
  typename AT::tdual_kkfloat_4d k_cut_st_0, k_cut_st_c, k_cut_st_lo, k_cut_st_hi;
  typename AT::tdual_kkfloat_4d k_cut_st_lc, k_cut_st_hc;
  typename AT::tdual_kkfloat_2d k_b_st_lo, k_b_st_hi;
  typename AT::tdual_kkfloat_4d k_shift_st, k_cutsq_st_hc;
  typename AT::tdual_kkfloat_4d k_a_st4;
  typename AT::tdual_kkfloat_2d k_theta_st4_0;
  typename AT::tdual_kkfloat_4d k_dtheta_st4_ast;
  typename AT::tdual_kkfloat_4d k_b_st4, k_dtheta_st4_c;
  typename AT::tdual_kkfloat_2d k_a_st5, k_theta_st5_0, k_dtheta_st5_ast;
  typename AT::tdual_kkfloat_2d k_b_st5, k_dtheta_st5_c;
  typename AT::tdual_kkfloat_2d k_a_st6, k_theta_st6_0, k_dtheta_st6_ast;
  typename AT::tdual_kkfloat_2d k_b_st6, k_dtheta_st6_c;
  typename AT::tdual_kkfloat_2d k_a_st1, k_cosphi_st1_ast, k_b_st1, k_cosphi_st1_c;
  typename AT::tdual_kkfloat_2d k_a_st2, k_cosphi_st2_ast, k_b_st2, k_cosphi_st2_c;
  typename AT::t_kkfloat_2d_randomread d_epsilon_st, d_a_st;
  typename AT::t_kkfloat_4d_randomread d_cut_st_0, d_cut_st_c, d_cut_st_lo, d_cut_st_hi;
  typename AT::t_kkfloat_4d_randomread d_cut_st_lc, d_cut_st_hc;
  typename AT::t_kkfloat_2d_randomread d_b_st_lo, d_b_st_hi;
  typename AT::t_kkfloat_4d_randomread d_shift_st, d_cutsq_st_hc;
  typename AT::t_kkfloat_4d_randomread d_a_st4;
  typename AT::t_kkfloat_2d_randomread d_theta_st4_0;
  typename AT::t_kkfloat_4d_randomread d_dtheta_st4_ast;
  typename AT::t_kkfloat_4d_randomread d_b_st4, d_dtheta_st4_c;
  typename AT::t_kkfloat_2d_randomread d_a_st5, d_theta_st5_0, d_dtheta_st5_ast;
  typename AT::t_kkfloat_2d_randomread d_b_st5, d_dtheta_st5_c;
  typename AT::t_kkfloat_2d_randomread d_a_st6, d_theta_st6_0, d_dtheta_st6_ast;
  typename AT::t_kkfloat_2d_randomread d_b_st6, d_dtheta_st6_c;
  typename AT::t_kkfloat_2d_randomread d_a_st1, d_cosphi_st1_ast, d_b_st1, d_cosphi_st1_c;
  typename AT::t_kkfloat_2d_randomread d_a_st2, d_cosphi_st2_ast, d_b_st2, d_cosphi_st2_c;
  // per-atom arrays for local unit vectors
  DAT::tdual_kkfloat_1d_3 k_nx_xtrct, k_ny_xtrct, k_nz_xtrct;
  typename AT::t_kkfloat_1d_3_randomread d_nx_xtrct, d_ny_xtrct, d_nz_xtrct;

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
 
  friend void pair_virial_fdotr_compute<PairOxdnaStkKokkos>(PairOxdnaStkKokkos*);

  FixOxdnaLRFKokkos<DeviceType> *fix_oxdna_lrfKK;    // ptr to oxdna/lrf/kk fix

  // Atom Mapping
  int map_style;
  DAT::tdual_int_1d k_map_array;
  dual_hash_type k_map_hash;
  DAT::tdual_int_1d k_sametag;
  typename AT::t_int_1d d_sametag;
  // Precomputed atom a/b 3'/5' directionality and atom mapping of their 3' and 5' neighbors.
  // 0-3 : atom a, atom b, id3p[a], id5p[b] for each bond.
  // Internal scratch: written by the precompute kernel and read by the compute
  // kernel, both in the same (device) execution space, so a single device-space
  // View suffices -- no DualView/host mirror (cf. bond_prime_neighs in the
  // oxdna_kokkos standalone). A DualView would force a device<->host deep_copy
  // of this int*[4] array on sync, which is both needless and fails on GPU.
  typename AT::t_int_1d_4 d_bond_prime_neighs;
};

}    // namespace LAMMPS_NS

#endif
#endif

