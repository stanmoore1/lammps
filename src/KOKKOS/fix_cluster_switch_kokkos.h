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

template<class DeviceType>
class FixClusterSwitchKokkos : public FixClusterSwitch {
public:
typedef DeviceType device_type;
typedef EV_FLOAT value_type;
typedef ArrayTypes<DeviceType> AT;

  FixClusterSwitchKokkos(class LAMMPS *, int, char **);
  ~FixClusterSwitchKokkos() override;
  void allocate() override;
  void init() override;
  void pre_exchange() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;

private:
  int confirm_molecule(tagint); // checks molID state (returns 1 for ON and 0 for off)
  int switch_flag(int); // uses random to decided if this molID should switch state (returns 1 for YES)
  void attempt_switch(); // where all the MC switching happens
  void check_cluster(); // checks recursively molecules part of central cluster and updates mol_restrict
  void gather_statistics(); //uses newly communicated mol arrays to gather MC statistics

  // hash map (key value) to keep track of mols

  typedef Kokkos::UnorderedMap<int,int> hash_type;
  hash_type hash_kk;

  int *atomtypesON; // atom types associated with ON
  int *atomtypesOFF; // atom types associated with OFF
  int ***contactMap; // atom types associated with successful intermolecule contact
  class RanPark *random_equal; //rand num gen (same across procs)
  class RanPark *random_unequal; // rand num gen (diff across procs)


  int *mol_restrict; // list of molecule ID tags that are open to switching (mol cluster can consider more mols than mol_restrict)
  //mol_restrict should be updated such that mols part of cluster are turned off to switching
  //is equal to 1 when open to switching, otherwise -1

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  typename AT::t_x_array x;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;
  typename AT::t_tagint_1d molecule;
  
  tagint **mol_atoms; // 2D list of (mol ID), (internal atom type) = atom ID tag
  int *mol_state; //list of molecules current state (0 = off, 1 = on)
  int *mol_accept; //list of switching decisions for each molecule (-1 initial, 0 fail, 1 switch true
  int *mol_cluster; //list of cluster IDs for each molecule
};

}

#endif
#endif
