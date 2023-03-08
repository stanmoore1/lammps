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
#include <Kokkos_UnorderedMap.hpp>

namespace LAMMPS_NS {

struct TagFixClusterSwitchCheckCluster{};

template<class DeviceType>
class FixClusterSwitchKokkos : public FixClusterSwitch {
public:
typedef DeviceType device_type;
typedef EV_FLOAT value_type;
typedef ArrayTypes<DeviceType> AT;

  FixClusterSwitchKokkos(class LAMMPS *, int, char **);
  ~FixClusterSwitchKokkos() override;
  void allocate(int) override;
  void init() override;
  void pre_exchange() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixClusterSwitchCheckCluster, const int&) const;

private:
  KOKKOS_INLINE_FUNCTION
  int confirm_molecule(tagint) const; // checks molID state (returns 1 for ON and 0 for off)

  KOKKOS_INLINE_FUNCTION
  int switch_flag(int) const; // uses random to decided if this molID should switch state (returns 1 for YES)

  void attempt_switch(); // where all the MC switching happens
  void check_cluster(); // checks recursively molecules part of central cluster and updates mol_restrict
  void gather_statistics(); // uses newly communicated mol arrays to gather MC statistics

  // hash map (key value) to keep track of mols

  typedef Kokkos::UnorderedMap<int,int> hash_type;
  hash_type d_hash;

  DAT::tdual_int_1d k_atomtypesON; // atom types associated with ON
  DAT::tdual_int_1d k_atomtypesOFF; // atom types associated with OFF
  DAT::tdual_int_3d k_contactMap; // atom types associated with successful intermolecule contact
  DAT::t_int_1d d_atomtypesON;
  DAT::t_int_1d d_atomtypesOFF;
  DAT::t_int_3d d_contactMap;


  DAT::tdual_int_1d k_mol_restrict; // list of molecule ID tags that are open to switching (mol cluster can consider more mols than mol_restrict)
  //mol_restrict should be updated such that mols part of cluster are turned off to switching
  //is equal to 1 when open to switching, otherwise -1
  DAT::t_int_1d d_mol_restrict;

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

  DAT::t_tagint_2d d_mol_atoms;
  DAT::t_int_1d d_mol_state;
  DAT::t_int_1d d_mol_accept;
  DAT::t_int_1d d_mol_cluster;

  DAT::t_int_1d d_mol_cluster_local;
  DAT::t_int_1d d_mol_accept_local;

  DAT::t_int_scalar d_done;
};

}

#endif
#endif
