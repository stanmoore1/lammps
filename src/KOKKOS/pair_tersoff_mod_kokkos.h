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
PairStyle(tersoff/mod/kk,PairTersoffMODKokkos<LMPDeviceType>);
PairStyle(tersoff/mod/kk/device,PairTersoffMODKokkos<LMPDeviceType>);
PairStyle(tersoff/mod/kk/host,PairTersoffMODKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_TERSOFF_MOD_KOKKOS_H
#define LMP_PAIR_TERSOFF_MOD_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_tersoff_mod.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<int NEIGHFLAG, int EVFLAG>
struct TagPairTersoffMODCompute{};

struct TagPairTersoffMODComputeShortNeigh{};

template<class DeviceType>
class PairTersoffMODKokkos : public PairTersoffMOD {
 public:
  enum {EnabledNeighFlags=HALF|HALFTHREAD};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  // Static blocking size for PairTersoffCompute, EVFLAG == 0
  static constexpr int block_size_compute_tersoff_force = 128;
  // EVFLAG == 1, intentionally different due to how Kokkos implements
  // reductions vs simple parallel_for
  static constexpr int block_size_compute_tersoff_energy = 256;

  PairTersoffMODKokkos(class LAMMPS *);
  ~PairTersoffMODKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;

  // RangePolicy versions
  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODCompute<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODCompute<NEIGHFLAG,EVFLAG>, const int&) const;

  // MDRangePolicy versions
  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODCompute<NEIGHFLAG,EVFLAG>, const int&, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODCompute<NEIGHFLAG,EVFLAG>, const int&, const int&) const;

  // TeamPolicy versions
  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODCompute<NEIGHFLAG,EVFLAG>, const typename Kokkos::TeamPolicy<DeviceType, TagPairTersoffMODCompute<NEIGHFLAG,EVFLAG> >::member_type&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODCompute<NEIGHFLAG,EVFLAG>, const typename Kokkos::TeamPolicy<DeviceType, TagPairTersoffMODCompute<NEIGHFLAG,EVFLAG> >::member_type&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairTersoffMODComputeShortNeigh, const int&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void tersoff_mod_compute(const int&, EV_FLOAT&) const;

  KOKKOS_INLINE_FUNCTION
  double ters_fc_k(const Param& param, const double &r) const;

  KOKKOS_INLINE_FUNCTION
  double ters_dfc(const Param& param, const double &r) const;

  KOKKOS_INLINE_FUNCTION
  double ters_fa_k(const Param& param, const double &r) const;

  KOKKOS_INLINE_FUNCTION
  double ters_dfa(const Param& param, const double &r) const;

  KOKKOS_INLINE_FUNCTION
  double ters_bij_k(const Param& param, const double &bo) const;

  KOKKOS_INLINE_FUNCTION
  double ters_dbij(const Param& param, const double &bo) const;

  KOKKOS_INLINE_FUNCTION
  double bondorder(const Param& param,
              const double &rij, const double &dx1, const double &dy1, const double &dz1,
              const double &rik, const double &dx2, const double &dy2, const double &dz2) const;

  KOKKOS_INLINE_FUNCTION
  double ters_gijk(const Param& param, const double &cos) const;

  KOKKOS_INLINE_FUNCTION
  double ters_dgijk(const Param& param, const double &cos) const;

  KOKKOS_INLINE_FUNCTION
  void ters_dthb(const Param& param, const double &prefactor,
              const double &rij, const double &dx1, const double &dy1, const double &dz1,
              const double &rik, const double &dx2, const double &dy2, const double &dz2,
              double *fi, double *fj, double *fk) const;

  KOKKOS_INLINE_FUNCTION
  void ters_dthbj(const Param& param, const double &prefactor,
              const double &rij, const double &dx1, const double &dy1, const double &dz1,
              const double &rik, const double &dx2, const double &dy2, const double &dz2,
              double *fj, double *fk) const;

  KOKKOS_INLINE_FUNCTION
  void ters_dthbk(const Param& param, const double &prefactor,
              const double &rij, const double &dx1, const double &dy1, const double &dz1,
              const double &rik, const double &dx2, const double &dy2, const double &dz2,
              double *fk) const;

  KOKKOS_INLINE_FUNCTION
  double vec3_dot(const double x[3], const double y[3]) const {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }

  KOKKOS_INLINE_FUNCTION
  void vec3_add(const double x[3], const double y[3], double * const z) const {
    z[0] = x[0]+y[0]; z[1] = x[1]+y[1]; z[2] = x[2]+y[2];
  }

  KOKKOS_INLINE_FUNCTION
  void vec3_scale(const double k, const double x[3], double y[3]) const {
    y[0] = k*x[0]; y[1] = k*x[1]; y[2] = k*x[2];
  }

  KOKKOS_INLINE_FUNCTION
  void vec3_scaleadd(const double k, const double x[3], const double y[3], double * const z) const {
    z[0] = k*x[0]+y[0]; z[1] = k*x[1]+y[1]; z[2] = k*x[2]+y[2];
  }

  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const double &epair, const double &fpair, const double &delx,
                  const double &dely, const double &delz) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void v_tally3(EV_FLOAT &ev, const int &i, const int &j, const int &k,
                double *fj, double *fk, double *drij, double *drik) const;

  KOKKOS_INLINE_FUNCTION
  void v_tally3_atom(EV_FLOAT &ev, const int &i, const int &j, const int &k,
                double *fj, double *fk, double *drji, double *drjk) const;

  void setup_params() override;

 protected:
  typedef Kokkos::DualView<int***,DeviceType> tdual_int_3d;
  typedef typename tdual_int_3d::t_dev_const_randomread t_int_3d_randomread;
  typedef typename tdual_int_3d::t_host t_host_int_3d;

  t_int_3d_randomread d_elem3param;
  typename AT::t_int_1d_randomread d_map;

  typedef Kokkos::DualView<Param*,DeviceType> tdual_param_1d;
  typedef typename tdual_param_1d::t_dev t_param_1d;
  typedef typename tdual_param_1d::t_host t_host_param_1d;

  t_param_1d d_params;

  int inum;
  typename AT::t_f_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;
  typename AT::t_tagint_1d tag;

  DAT::tdual_double_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_double_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int need_dup;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<double*[3], typename DAT::t_f_array::array_layout> dup_f;
  DupScatterView<double*, typename DAT::t_double_1d::array_layout> dup_eatom;
  DupScatterView<double*[6], typename DAT::t_virial_array::array_layout> dup_vatom;

  NonDupScatterView<double*[3], typename DAT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<double*, typename DAT::t_double_1d::array_layout> ndup_eatom;
  NonDupScatterView<double*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  typedef Kokkos::DualView<double**[7],Kokkos::LayoutRight,DeviceType> tdual_double_2d_n7;
  typedef typename tdual_double_2d_n7::t_dev_const_randomread t_double_2d_n7_randomread;
  typedef typename tdual_double_2d_n7::t_host t_host_double_2d_n7;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  Kokkos::View<int**,DeviceType> d_neighbors_short;
  Kokkos::View<int*,DeviceType> d_numneigh_short;

  friend void pair_virial_fdotr_compute<PairTersoffMODKokkos>(PairTersoffMODKokkos*);
};

}

#endif
#endif

