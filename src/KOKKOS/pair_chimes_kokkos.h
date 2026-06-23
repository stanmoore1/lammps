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
PairStyle(chimesFF/kk,PairCHIMESKokkos<LMPDeviceType>);
PairStyle(chimesFF/kk/device,PairCHIMESKokkos<LMPDeviceType>);
PairStyle(chimesFF/kk/host,PairCHIMESKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_CHIMES_KOKKOS_H
#define LMP_PAIR_CHIMES_KOKKOS_H

#include "chimesFF_kokkos.h"
#include "pair_chimes.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairCHIMESKokkos : public PairCHIMES
{
 public:
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

  PairCHIMESKokkos(class LAMMPS *);
  ~PairCHIMESKokkos() override;
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
  void operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT get_dist(int i, int j, KK_FLOAT* dr) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT get_dist(int i, int j) const;

 private:
  int neighflag;
  int inum, maxneigh, chunk_size, chunk_offset;
  int host_flag, max_2mers, max_3mers, max_4mers;
  int size_2mers, size_3mers, size_4mers;

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

  // Cluster (m-mer) index lists stored LayoutLeft so the cluster index is the
  // fastest-varying dimension. Adjacent clusters then sit at adjacent addresses,
  // coalescing the per-cluster index writes during the parallel_scan/atomic list
  // build and the reads in the RangePolicy 2-body compute (where neighboring
  // threads handle neighboring clusters). The 3B/4B team kernels read one
  // cluster's indices as a broadcast, so they are unaffected.
  Kokkos::View<int*[2], Kokkos::LayoutLeft, DeviceType> d_neighborlist_2mers;
  Kokkos::View<int*[3], Kokkos::LayoutLeft, DeviceType> d_neighborlist_3mers;
  Kokkos::View<int*[4], Kokkos::LayoutLeft, DeviceType> d_neighborlist_4mers;

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

template <class DeviceType>
struct PairCHIMESComputeNeigh2BodyFunctor  {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef int value_type;
  PairCHIMESKokkos<DeviceType> c;
  PairCHIMESComputeNeigh2BodyFunctor(PairCHIMESKokkos<DeviceType>* c_ptr):c(*c_ptr) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &offset, const bool &final) const {
    c.neigh_2B_item(ii,offset,final);
  }
};

template <class DeviceType>
struct PairCHIMESComputeNeigh3BodyFunctor  {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef int value_type;
  PairCHIMESKokkos<DeviceType> c;
  PairCHIMESComputeNeigh3BodyFunctor(PairCHIMESKokkos<DeviceType>* c_ptr):c(*c_ptr) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &offset, const bool &final) const {
    c.neigh_3B_item(ii,offset,final);
  }
};

template <class DeviceType>
struct PairCHIMESComputeNeigh4BodyFunctor  {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef int value_type;
  PairCHIMESKokkos<DeviceType> c;
  PairCHIMESComputeNeigh4BodyFunctor(PairCHIMESKokkos<DeviceType>* c_ptr):c(*c_ptr) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &offset, const bool &final) const {
    c.neigh_4B_item(ii,offset,final);
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
