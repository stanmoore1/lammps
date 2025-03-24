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
PairStyle(oxdna2/dh/kk,PairOxdna2DhKokkos<LMPDeviceType>);
PairStyle(oxdna2/dh/kk/device,PairOxdna2DhKokkos<LMPDeviceType>);
PairStyle(oxdna2/dh/kk/host,PairOxdna2DhKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_OXDNA2_DH_KOKKOS_H
#define LMP_PAIR_OXDNA2_DH_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "pair_oxdna2_dh.h"
#include "neigh_list_kokkos.h"

#include "atom_vec_ellipsoid_kokkos.h"

namespace LAMMPS_NS {

template<int OXDNAFLAG, int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairOxdna2DhCompute{};

template<class DeviceType>
class PairOxdna2DhKokkos : public PairOxdna2Dh, public KokkosBase {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairOxdna2DhKokkos(class LAMMPS *);
  ~PairOxdna2DhKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  template<int OXDNAFLAG, int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdna2DhCompute<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int OXDNAFLAG, int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdna2DhCompute<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

 protected:

  int oxdnaflag;
  enum EnabledOXDNAFlag{OXDNA2=1,OXRNA2=2};

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_f_array torque;
  typename AT::t_int_1d_randomread type;

  typename AtomVecEllipsoidKokkosBonusArray<DeviceType>::t_bonus_1d bonus;
  typename AT::t_int_1d_randomread ellipsoid;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;

  int neighflag;
  int nlocal, eflag, vflag;
  int anum;

  typename AT::t_neighbors_2d_randomread d_neighbors;
  typename AT::t_int_1d_randomread d_alist;
  typename AT::t_int_1d_randomread d_numneigh;

  // debye huckel interaction parameters
  typename AT::tdual_ffloat_2d k_qeff_dh_pf, k_kappa_dh;
  typename AT::tdual_ffloat_2d k_b_dh, k_cut_dh_ast, k_cutsq_dh_ast;
  typename AT::tdual_ffloat_2d k_cut_dh_c, k_cutsq_dh_c;
  typename AT::t_ffloat_2d d_qeff_dh_pf, d_kappa_dh;
  typename AT::t_ffloat_2d d_b_dh, d_cut_dh_ast, d_cutsq_dh_ast;
  typename AT::t_ffloat_2d d_cut_dh_c, d_cutsq_dh_c;
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
 
  friend void pair_virial_fdotr_compute<PairOxdna2DhKokkos>(PairOxdna2DhKokkos*);
};

}

#endif
#endif

