/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(shake/kk,FixShakeKokkos<LMPDeviceType>)
FixStyle(shake/kk/device,FixShakeKokkos<LMPDeviceType>)
FixStyle(shake/kk/host,FixShakeKokkos<LMPHostType>)

#else

#ifndef LMP_FIX_SHAKE_KOKKOS_H
#define LMP_FIX_SHAKE_KOKKOS_H

#include "fix_shake.h"

namespace LAMMPS_NS {

class FixShakeKokkos : public FixShake {

 //friend class FixEHEX;

 public:
  FixShakeKokkos(class LAMMPS *, int, char **);
  virtual ~FixShakeKokkos();
  virtual void init();
  void setup(int);
  void pre_neighbor();
  virtual void post_force(int);

  virtual double memory_usage();
  virtual void grow_arrays(int);
  virtual void copy_arrays(int, int, int);
  void set_arrays(int);
  virtual void update_arrays(int, int);
  void set_molecule(int, tagint, int, double *, double *, double *);

  virtual int pack_exchange(int, double *);
  virtual int unpack_exchange(int, double *);
  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);

  virtual void shake_end_of_step(int vflag);
  virtual void correct_coordinates(int vflag);
  virtual void correct_velocities();

  int dof(int);
  virtual void reset_dt();

 protected:

  typename ArrayTypes<DeviceType>::t_x_array x;
  typename ArrayTypes<DeviceType>::t_v_array v;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_float_1d rmass;
  typename ArrayTypes<DeviceType>::t_float_1d mass;
  typename ArrayTypes<DeviceType>::t_int_1d type;

                                         // settings from input command
  typename ArrayTypes<DeviceType>::t_int_1d d_bond_flag; // bond/angle types to constrain
  typename ArrayTypes<DeviceType>::t_int_1d d_angle_flag;
  typename ArrayTypes<DeviceType>::t_float_1d d_bond_distance; // constraint distances
  typename ArrayTypes<DeviceType>::t_float_1d d_angle_distance;

                                         // atom-based arrays
  typename ArrayTypes<DeviceType>::t_int_1d d_shake_list; // 0 if atom not in SHAKE cluster
                                         // 1 = size 3 angle cluster
                                         // 2,3,4 = size of bond-only cluster
  typename ArrayTypes<DeviceType>::t_tagint_2d d_shake_atom; // global IDs of atoms in cluster
                                         // central atom is 1st
                                         // lowest global ID is 1st for size 2
  typename ArrayTypes<DeviceType>::t_int_2d d_shake_type; // bondtype of each bond in cluster
                                         // for angle cluster, 3rd value
                                         //   is angletype
  typename ArrayTypes<DeviceType>::t_x_array d_xshake; // unconstrained atom coords
  typename ArrayTypes<DeviceType>::t_int_1d d_nshake; // count

  typename ArrayTypes<DeviceType>::t_int_1d d_list // list of clusters to SHAKE

  class Molecule **atommols;            // atom style template pointer
  class Molecule **onemols;             // molecule added on-the-fly

  void find_clusters();
  void atom_owners();
  void partner_info(int *, tagint **, int **, int **, int **, int **);
  void nshake_info(int *, tagint **, int **);
  void shake_info(int *, tagint **, int **);

  int masscheck(double);
  void unconstrained_update();
  void unconstrained_update_respa(int);
  void shake(int);
  void shake3(int);
  void shake4(int);
  void shake3angle(int);
  void stats();
  int bondtype_findset(int, tagint, tagint, int);
  int angletype_findset(int, tagint, tagint, int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use fix shake with non-molecular system

Your choice of atom style does not have bonds.

*/
