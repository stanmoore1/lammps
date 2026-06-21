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

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(chimesFF/kk,PairCHIMESKokkosDevice<LMPDeviceType>);
PairStyle(chimesFF/kk/device,PairCHIMESKokkosDevice<LMPDeviceType>);
#ifdef LMP_KOKKOS_GPU
PairStyle(chimesFF/kk/host,PairCHIMESKokkosHost<LMPHostType>);
#else
PairStyle(chimesFF/kk/host,PairCHIMESKokkosDevice<LMPHostType>);
#endif
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_CHIMES_KOKKOS_H
#define LMP_PAIR_CHIMES_KOKKOS_H

#include "chimesFF_kokkos.h"
#include "pair_chimes.h"

// Per-backend vector (sub-group / warp) length for the 3B/4B Compute
// TeamPolicies, mirroring the Kokkos SNAP convention (kokkos_type.h:
// SNAP_KOKKOS_*_VECLEN). The host backend always uses 1.
#ifdef LMP_KOKKOS_GPU
  #if defined(KOKKOS_ENABLE_SYCL)
    #define CHIMES_KOKKOS_DEVICE_VECLEN 16
  #else
    #define CHIMES_KOKKOS_DEVICE_VECLEN 32
  #endif
#else
  #define CHIMES_KOKKOS_DEVICE_VECLEN 1
#endif
#define CHIMES_KOKKOS_HOST_VECLEN 1

namespace LAMMPS_NS {

// LaunchBounds occupancy hint (minimum resident blocks per SM) for the 3B/4B
// Compute TeamPolicies; tune per architecture (mirrors SNAP's
// min_blocks_compute_*). Raising these lowers the per-thread register cap to
// improve occupancy, at the risk of spills — validate on the target GPU.
constexpr int chimes_min_blocks_3b = 1;
constexpr int chimes_min_blocks_4b = 1;

template<class DeviceType, int vector_length_>
class PairCHIMESKokkos : public PairCHIMES
{
 public:
  static constexpr int vector_length = vector_length_;

  struct TagPairCHIMESZero{};

  struct TagPairCHIMESComputeNeigh4Body{};

  template<int NEIGHFLAG>
  struct TagPairCHIMESCompute1Body{};

  template<int NEIGHFLAG, int EVFLAG>
  struct TagPairCHIMESCompute2Body{};

  template<int NEIGHFLAG, int EVFLAG>
  struct TagPairCHIMESCompute3Body{};

  template<int NEIGHFLAG, int EVFLAG>
  struct TagPairCHIMESCompute4Body{};

  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;
  typedef typename Kokkos::TeamPolicy<DeviceType>::member_type t_team;

  PairCHIMESKokkos(class LAMMPS *);
  ~PairCHIMESKokkos() override;
  void settings(int narg, char **arg) override;
  void init_style() override;
  void coeff(int narg, char **arg) override;
  void allocate() override;
  void compute(int eflag, int vflag) override;
  void build_mb_neighlists() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairCHIMESZero, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void neigh_2B_item(const int& ii, int& offset, const bool& final) const;

  KOKKOS_INLINE_FUNCTION
  void neigh_3B_item(const int& ii, int& offset, const bool& final) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESComputeNeigh4Body,const int& ii) const;
  //void neigh_4B_item(const int& ii, int& offset, const bool& final) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute1Body<NEIGHFLAG>,const int& ii, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>,const t_team& team) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>,const t_team& team, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>,const t_team& team) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>,const t_team& team, EV_FLOAT&) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT get_dist(int i, int j, KK_FLOAT* dr) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT get_dist(int i, int j) const;

 private:
  int neighflag;
  int inum, maxneigh, chunk_size, chunk_offset, chunksize;
  int host_flag, max_2mers, max_3mers, max_4mers;
  int size_2mers, size_3mers, size_4mers;

  KK_FLOAT maxcut_3b_padded, maxcut_4b_padded;

  int eflag, vflag;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  typedef Kokkos::DualView<KK_FLOAT**, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq, k_scale;
  typedef Kokkos::View<KK_FLOAT**, DeviceType> t_fparams;
  t_fparams d_cutsq, d_scale;

  typename AT::t_int_1d d_chimes_type,d_map;

  typename AT::t_int_1d_2 d_neighborlist_2mers;
  typename AT::t_int_1d_3 d_neighborlist_3mers;
  typename AT::t_int_1d_4 d_neighborlist_4mers;

  typename AT::t_int_scalar d_size_4mers;

  chimesFFKokkos<DeviceType> chimes_calculatorKK; // chimesFF instance

  int need_dup;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout> dup_f;
  DupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout> dup_eatom;
  DupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> dup_vatom;

  NonDupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout> ndup_f;
  NonDupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout> ndup_eatom;
  NonDupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairCHIMESKokkos>(PairCHIMESKokkos*);

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void ev_tally_mb(int ninteractionatoms, int npairs,
                   int atmpairidxlst[6][2],
                   KK_FLOAT, KK_FLOAT[6],
                   EV_FLOAT &ev) const;

};

template <class DeviceType, int vector_length>
struct PairCHIMESComputeNeigh2BodyFunctor  {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef int value_type;
  PairCHIMESKokkos<DeviceType, vector_length> c;
  PairCHIMESComputeNeigh2BodyFunctor(PairCHIMESKokkos<DeviceType, vector_length>* c_ptr):c(*c_ptr) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &offset, const bool &final) const {
    c.neigh_2B_item(ii,offset,final);
  }
};

template <class DeviceType, int vector_length>
struct PairCHIMESComputeNeigh3BodyFunctor  {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef int value_type;
  PairCHIMESKokkos<DeviceType, vector_length> c;
  PairCHIMESComputeNeigh3BodyFunctor(PairCHIMESKokkos<DeviceType, vector_length>* c_ptr):c(*c_ptr) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &offset, const bool &final) const {
    c.neigh_3B_item(ii,offset,final);
  }
};

template <class DeviceType, int vector_length>
struct PairCHIMESComputeNeigh4BodyFunctor  {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef int value_type;
  PairCHIMESKokkos<DeviceType, vector_length> c;
  PairCHIMESComputeNeigh4BodyFunctor(PairCHIMESKokkos<DeviceType, vector_length>* c_ptr):c(*c_ptr) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &offset, const bool &final) const {
    c.neigh_4B_item(ii,offset,final);
  }
};

// Wrapper subclasses fixing the vector_length template parameter per backend,
// so the pair-style factory (which only supplies the DeviceType argument) can
// instantiate the right specialization. Mirrors PairSNAPKokkosDevice/Host.

template <class DeviceType>
class PairCHIMESKokkosDevice : public PairCHIMESKokkos<DeviceType, CHIMES_KOKKOS_DEVICE_VECLEN> {
 private:
  using Base = PairCHIMESKokkos<DeviceType, CHIMES_KOKKOS_DEVICE_VECLEN>;
 public:
  PairCHIMESKokkosDevice(class LAMMPS *lmp) : Base(lmp) {}
  void coeff(int narg, char **arg) override { Base::coeff(narg, arg); }
  void init_style() override { Base::init_style(); }
  void allocate() override { Base::allocate(); }
  void compute(int eflag, int vflag) override { Base::compute(eflag, vflag); }
  void build_mb_neighlists() override { Base::build_mb_neighlists(); }
};

#ifdef LMP_KOKKOS_GPU
template <class DeviceType>
class PairCHIMESKokkosHost : public PairCHIMESKokkos<DeviceType, CHIMES_KOKKOS_HOST_VECLEN> {
 private:
  using Base = PairCHIMESKokkos<DeviceType, CHIMES_KOKKOS_HOST_VECLEN>;
 public:
  PairCHIMESKokkosHost(class LAMMPS *lmp) : Base(lmp) {}
  void coeff(int narg, char **arg) override { Base::coeff(narg, arg); }
  void init_style() override { Base::init_style(); }
  void allocate() override { Base::allocate(); }
  void compute(int eflag, int vflag) override { Base::compute(eflag, vflag); }
  void build_mb_neighlists() override { Base::build_mb_neighlists(); }
};
#endif

}    // namespace LAMMPS_NS

#endif
#endif
