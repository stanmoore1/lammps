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

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(ellipsoid/kk,AtomVecEllipsoidKokkos);
AtomStyle(ellipsoid/kk/device,AtomVecEllipsoidKokkos);
AtomStyle(ellipsoid/kk/host,AtomVecEllipsoidKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_ELLIPSOID_KOKKOS_H
#define LMP_ATOM_VEC_ELLIPSOID_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_ellipsoid.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

/* ---------------------------------------------------------------------- */
// DualViews for Bonus struct - shape,quat,ilocal.

template <class DeviceType>
struct AtomVecEllipsoidKokkosBonusArray;

template <>
struct AtomVecEllipsoidKokkosBonusArray<LMPDeviceType> {
  typedef Kokkos::
    DualView<AtomVecEllipsoid::Bonus*,
    LMPDeviceType::array_layout,LMPDeviceType> tdual_bonus_1d;
  typedef tdual_bonus_1d::t_dev t_bonus_1d;
};
#ifdef LMP_KOKKOS_GPU
template <>
struct AtomVecEllipsoidKokkosBonusArray<LMPHostType> {
  typedef Kokkos::
    DualView<AtomVecEllipsoid::Bonus*,
    LMPHostType::array_layout,LMPHostType> tdual_bonus_1d;
  typedef tdual_bonus_1d::t_host t_bonus_1d;
};
#endif

typedef AtomVecEllipsoidKokkosBonusArray<LMPDeviceType> DEllipsoidBonusAT;
typedef AtomVecEllipsoidKokkosBonusArray<LMPHostType> HEllipsoidBonusAT;

/* ---------------------------------------------------------------------- */

class AtomVecEllipsoidKokkos : public AtomVecKokkos, public AtomVecEllipsoid {
 public:
  AtomVecEllipsoidKokkos(class LAMMPS *);

  void grow(int) override;
  void grow_pointers() override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;

  int pack_comm_kokkos(const int &n, const DAT::tdual_int_2d &k_sendlist,
                       const int & iswap,
                       const DAT::tdual_xfloat_2d &buf,
                       const int &pbc_flag, const int pbc[]) override;
  void unpack_comm_kokkos(const int &n, const int &nfirst,
                          const DAT::tdual_xfloat_2d &buf) override;
  int pack_comm_vel_kokkos(const int &n, const DAT::tdual_int_2d &k_sendlist,
                           const int & iswap,
                           const DAT::tdual_xfloat_2d &buf,
                           const int &pbc_flag, const int pbc[]) override;
  void unpack_comm_vel_kokkos(const int &n, const int &nfirst,
                              const DAT::tdual_xfloat_2d &buf) override;
  int pack_comm_self(const int &n, const DAT::tdual_int_2d &list,
                     const int & iswap, const int nfirst,
                     const int &pbc_flag, const int pbc[]) override;
  int pack_border_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                         DAT::tdual_xfloat_2d buf,int iswap,
                         int pbc_flag, int *pbc, ExecutionSpace space) override;
  void unpack_border_kokkos(const int &n, const int &nfirst,
                            const DAT::tdual_xfloat_2d &buf,
                            ExecutionSpace space) override;
  int pack_border_vel_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                             DAT::tdual_xfloat_2d buf,int iswap,
                             int pbc_flag, int *pbc, ExecutionSpace space) override;
  void unpack_border_vel_kokkos(const int &n, const int &nfirst,
                                const DAT::tdual_xfloat_2d &buf,
                                ExecutionSpace space) override;
  int pack_exchange_kokkos(const int &nsend,DAT::tdual_xfloat_2d &buf,
                           DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist,
                           ExecutionSpace space) override;
  int unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv,
                             int nlocal, int dim, X_FLOAT lo, X_FLOAT hi,
                             ExecutionSpace space, DAT::tdual_int_1d &k_indices) override;    
  int pack_comm_bonus_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                             DAT::tdual_xfloat_2d buf,int iswap, 
                             ExecutionSpace space) override;
  void unpack_comm_bonus_kokkos(const int &n, const int &nfirst,
                             const DAT::tdual_xfloat_2d &buf,
                             ExecutionSpace space) override;
  int pack_border_bonus_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                             DAT::tdual_xfloat_2d buf,int iswap,
                             ExecutionSpace space) override;
  void unpack_border_bonus_kokkos(const int &n, const int & nfirst,
                               const DAT::tdual_xfloat_2d & buf,
                               ExecutionSpace space) override;
  int pack_exchange_bonus_kokkos(const int &nsend, 
                               DAT::tdual_xfloat_2d &buf,
                               DAT::tdual_int_1d k_sendlist,
                               DAT::tdual_int_1d k_copylist,
                               ExecutionSpace space) override;
  int unpack_exchange_bonus_kokkos(DAT::tdual_xfloat_2d &k_buf, 
                                 int nrecv, int nlocal,
                                 int dim, X_FLOAT lo, X_FLOAT hi,
                                 ExecutionSpace space,
                                 DAT::tdual_int_1d &k_indices) override;

  void sync(ExecutionSpace space, unsigned int mask) override;
  void modified(ExecutionSpace space, unsigned int mask) override;
  void sync_overlapping_device(ExecutionSpace space, unsigned int mask) override;

  // Bonus struct

  void grow_bonus() override; 

  //typedef Kokkos::DualView<Bonus*,LMPDeviceType> tdual_bonus_1d;
  //typedef typename tdual_bonus_1d::t_dev t_bonus_1d;
  //typedef typename tdual_bonus_1d::t_host t_host_bonus_1d;
    
 private:
  //int *ellipsoid; // IS THIS CORRECT?
  double **torque;  
    
  DAT::t_tagint_1d d_tag;
  HAT::t_tagint_1d h_tag;
  DAT::t_imageint_1d d_image;
  HAT::t_imageint_1d h_image;
  DAT::t_int_1d d_type, d_mask;
  HAT::t_int_1d h_type, h_mask;
    
  DAT::t_x_array d_x;
  //HAT::t_x_array h_x; // NOT SURE THESE 3 HATs ARE NEEDED - EXIST (PROTECTED MEMBERS) WITHIN AtomVecKokkos
  DAT::t_v_array d_v;
  //HAT::t_v_array h_v;
  DAT::t_f_array d_f;
  //DAT::t_f_array h_f;
    
  DAT::t_float_1d d_rmass;
  HAT::t_float_1d h_rmass;
  DAT::t_v_array d_angmom;
  HAT::t_v_array h_angmom;
  DAT::t_f_array d_torque;
  HAT::t_f_array h_torque;
  DAT::t_int_1d d_ellipsoid;
  HAT::t_int_1d h_ellipsoid;
    
  DEllipsoidBonusAT::tdual_bonus_1d k_bonus; 
  DEllipsoidBonusAT::t_bonus_1d d_bonus; 
  HEllipsoidBonusAT::t_bonus_1d h_bonus;

};

}    // namespace LAMMPS_NS

#endif
#endif
