#ifdef FIX_CLASS
// clang-format off
FixStyle(cluster_switch/kk,FixClusterSwitchKokkos<LMPDeviceType>);
FixStyle(cluster_switch/kk/device,FixClusterSwitchKokkos<LMPDeviceType>);
FixStyle(cluster_switch/kk/host,FixClusterSwitchKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_CLUSTER_SWITCH_KOKKOS_H
#define LMP_FIX_CLUSTER_SWITCH_KOKKOS_H

#include "fix_cluster_switch.h"
#include "kokkos_type.h"
#include "kokkos_base.h"
#include "rand_pool_wrap_kokkos.h"
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Random.hpp>

namespace LAMMPS_NS {

struct TagFixClusterSwitchInit{};
struct TagFixClusterSwitchCheckCluster1{};
struct TagFixClusterSwitchCheckCluster2{};
struct TagFixClusterSwitchCheckCluster3{};
struct TagFixClusterSwitchCheckCluster4{};
struct TagFixClusterSwitchAttemptSwitch1{};
struct TagFixClusterSwitchAttemptSwitch2{};
struct TagFixClusterSwitchAttemptSwitch3{};
struct TagFixClusterSwitchAttemptSwitch4{};

template<int PBC_FLAG>
struct TagFixClusterSwitchPackForwardComm{};

struct TagFixClusterSwitchUnpackForwardComm{};

template<class DeviceType>
class FixClusterSwitchKokkos : public FixClusterSwitch, public KokkosBase {
 public:

  struct REDUCE_DOUBLE_6 {
    double d0, d1, d2, d3, d4, d5;
    KOKKOS_INLINE_FUNCTION
    REDUCE_DOUBLE_6() {
      d0 = d1 = d2 = d3 = d4 = d5 = 0.0;
    }
    KOKKOS_INLINE_FUNCTION
    REDUCE_DOUBLE_6& operator+=(const REDUCE_DOUBLE_6 &rhs) {
      d0 += rhs.d0;
      d1 += rhs.d1;
      d2 += rhs.d2;
      d3 += rhs.d3;
      d4 += rhs.d4;
      d5 += rhs.d5;
      return *this;
    }
  };

  typedef DeviceType device_type;
  typedef double value_type;
  typedef ArrayTypes<DeviceType> AT;

  FixClusterSwitchKokkos(class LAMMPS *, int, char **);
  ~FixClusterSwitchKokkos() override;
  void allocate(int) override;
  void init() override;
  void pre_exchange() override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_2d, int, DAT::tdual_xfloat_1d&,
                       int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchInit, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchCheckCluster1, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchCheckCluster2, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchCheckCluster3, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchCheckCluster4, const int&, double &) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchAttemptSwitch1, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchAttemptSwitch2, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchAttemptSwitch3, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchAttemptSwitch4, const int&) const;

  template<int PBC_FLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchPackForwardComm<PBC_FLAG>, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchUnpackForwardComm, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void gather_statistics_item(const int&, REDUCE_DOUBLE_6&) const;

 private:
#ifdef LMP_KOKKOS_DEBUG
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#else
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#endif

  KOKKOS_INLINE_FUNCTION
  int confirm_molecule(tagint) const; // checks molID state (returns 1 for ON and 0 for off)

  KOKKOS_INLINE_FUNCTION
  int switch_flag(int, rand_type &rand_gen) const; // uses random to decided if this molID should switch state (returns 1 for YES)

  void attempt_switch(); // where all the MC switching happens
  void check_cluster(); // checks recursively molecules part of central cluster and updates mol_restrict
  void gather_statistics(); // uses newly communicated mol arrays to gather MC statistics

  // hash map (key value) to keep track of mols

  typedef Kokkos::UnorderedMap<int,int> hash_type;
  hash_type d_hash;

  DAT::tdual_int_1d k_atomtypesON; // atom types associated with ON
  DAT::tdual_int_1d k_atomtypesOFF; // atom types associated with OFF
  DAT::tdual_int_3d k_contactMap; // atom types associated with successful intermolecule contact
  typename AT::t_int_1d d_atomtypesON;
  typename AT::t_int_1d d_atomtypesOFF;
  typename AT::t_int_3d d_contactMap;

  DAT::tdual_int_1d k_mol_restrict; // list of molecule ID tags that are open to switching (mol cluster can consider more mols than mol_restrict)
  //mol_restrict should be updated such that mols part of cluster are turned off to switching
  //is equal to 1 when open to switching, otherwise -1
  typename AT::t_int_1d d_mol_restrict;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d d_ilist;
  typename AT::t_int_1d d_numneigh;

  typename AT::t_x_array x;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;
  typename AT::t_tagint_1d molecule;
  
  DAT::tdual_tagint_2d k_mol_atoms; // 2D list of (mol ID), (internal atom type) = atom ID tag
  DAT::tdual_int_1d k_mol_state; //list of molecules current state (0 = off, 1 = on)
  DAT::tdual_int_1d k_mol_accept; //list of switching decisions for each molecule (-1 initial, 0 fail, 1 switch true
  DAT::tdual_int_1d k_mol_cluster; //list of cluster IDs for each molecule

  typename AT::t_tagint_2d d_mol_atoms;
  typename AT::t_int_1d d_mol_state;
  typename AT::t_int_1d d_mol_accept;
  typename AT::t_int_1d d_mol_cluster;
  typename AT::t_tagint_1d d_mol_count;
  typename AT::t_int_1d d_sumState;

  typename AT::t_int_1d d_mol_cluster_local;
  typename AT::t_int_1d d_mol_accept_local;

  typename AT::t_int_scalar d_done;

  int iswap,first,nsend;

  typename AT::t_int_2d d_sendlist;
  typename AT::t_xfloat_1d_um d_buf;

  typename AT::t_int_1d d_exchange_sendlist;
  typename AT::t_int_1d d_copylist;
  typename AT::t_int_1d d_indices;

  X_FLOAT dx,dy,dz;
};

template <class DeviceType>
struct FixClusterSwitchGatherStatisticsFunctor {
  typedef DeviceType device_type;
  typedef typename FixClusterSwitchKokkos<DeviceType>::REDUCE_DOUBLE_6 value_type;
  FixClusterSwitchKokkos<DeviceType> c;
  FixClusterSwitchGatherStatisticsFunctor(FixClusterSwitchKokkos<DeviceType>* c_ptr):c(*c_ptr) {c.set_copymode(1);}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &reduce) const {
    c.gather_statistics_item(i, reduce);
  }
};

}

#endif
#endif
