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

#include <vector>

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

  // True when this instantiation executes on the host.  The host has the
  // batched, type-grouped chimesFF evaluators, a cluster build that works in
  // squared distances, and no limit on the polynomial order, so it takes an
  // entirely different code path from the device kernels below.

  static constexpr int host_flag = (ExecutionSpaceFromDevice<DeviceType>::space == LAMMPS_NS::HostKK);

  PairCHIMESKokkos(class LAMMPS *);
  ~PairCHIMESKokkos() override;
  void init_style() override;
  void coeff(int narg, char **arg) override;
  void allocate() override;
  void compute(int eflag, int vflag) override;
  void build_mb_neighlists() override;
  void setup_neighlist_ptrs() override;

  // Host path: the batched chimesFF evaluators, driven over chunks of the
  // type-sorted cluster lists so the work threads.

  void compute_host(int eflag, int vflag);
  void host_setup_chunks();

  template<int NEIGHFLAG>
  void host_launch(EV_FLOAT &ev);

  template<int NEIGHFLAG>
  void host_2body_chunk(const int chunk, EV_FLOAT &ev) const;

  template<int NEIGHFLAG>
  void host_3body_chunk(const int chunk, EV_FLOAT &ev) const;

  template<int NEIGHFLAG>
  void host_4body_chunk(const int chunk, EV_FLOAT &ev) const;

  template<class EAtomAccess, class VAtomAccess>
  void host_tally(int ninteractionatoms, const int *atmlist, double evdwl,
                  const double *stress, EV_FLOAT &ev, const EAtomAccess &a_eatom,
                  const VAtomAccess &a_vatom) const;

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
  int max_2mers, max_3mers, max_4mers;
  int size_2mers, size_3mers, size_4mers;

  // Host path state.  The cluster lists are the base class's type-sorted flat
  // arrays; each work item takes one contiguous chunk of them and runs the
  // batched evaluator over it, so a batch only ever breaks at a chunk edge.

  int host_nchunk_2b, host_nchunk_3b, host_nchunk_4b;

  // Per-chunk scratch, allocated once per run rather than per launch: the
  // batch objects own heap buffers, and the 2-body path stages one partial
  // lane group per chemical-pair key.

  std::vector<chimes3BBatch> host_batch3;
  std::vector<chimes4BBatch> host_batch4;

  std::vector<int> host_b2_cnt;
  std::vector<int> host_b2_i, host_b2_j;
  std::vector<double> host_b2_dist, host_b2_dr;

  // Neighbor list in the form the shared CPU loops expect.  d_neighbors is one
  // padded 2d block, so a row is already contiguous whenever its inner stride
  // is one; when it is not, the rows are compacted into host_neigh_buf.

  std::vector<int *> host_firstneigh;
  std::vector<int> host_neigh_buf;

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

// Host cluster kernels.  These hold a pointer rather than a copy of the pair
// style: Kokkos copies the functor into every launch, and the style owns the
// coefficient tables and the cluster lists, which run to tens of megabytes.
// They are host-only, so the pointer is always dereferenceable.

template <class DeviceType, int NBODY, int NEIGHFLAG>
struct PairCHIMESHostClusterFunctor {
  typedef DeviceType execution_space;
  typedef EV_FLOAT value_type;

  PairCHIMESKokkos<DeviceType> *p;

  PairCHIMESHostClusterFunctor(PairCHIMESKokkos<DeviceType> *p_in) : p(p_in) {}

  void operator()(const int &chunk, EV_FLOAT &ev) const
  {
    if (NBODY == 2)
      p->template host_2body_chunk<NEIGHFLAG>(chunk, ev);
    else if (NBODY == 3)
      p->template host_3body_chunk<NEIGHFLAG>(chunk, ev);
    else
      p->template host_4body_chunk<NEIGHFLAG>(chunk, ev);
  }
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
