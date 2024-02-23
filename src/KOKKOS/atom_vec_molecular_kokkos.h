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
AtomStyle(molecular/kk,AtomVecMolecularKokkos);
AtomStyle(molecular/kk/device,AtomVecMolecularKokkos);
AtomStyle(molecular/kk/host,AtomVecMolecularKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_MOLECULAR_KOKKOS_H
#define LMP_ATOM_VEC_MOLECULAR_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_molecular.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomVecMolecularKokkos : public AtomVecKokkos, public AtomVecMolecular {
 public:
  AtomVecMolecularKokkos(class LAMMPS *);

  void grow(int) override;
  void grow_pointers() override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;

  template<class DeviceType>
  int pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                         DAT::tdual_xfloat_2d buf,
                         int pbc_flag, int *pbc) override;

  template<class DeviceType>
  void unpack_border_kokkos(const int &n, const int &nfirst,
                            const DAT::tdual_xfloat_2d &buf) override;

  template<class DeviceType>
  int pack_exchange_kokkos(const int &nsend,DAT::tdual_xfloat_2d &buf,
                           DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist) override;

  template<class DeviceType>
  int unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv,
                             int nlocal, int dim, X_FLOAT lo, X_FLOAT hi,
                             DAT::tdual_int_1d &k_indices) override;

  void sync(ExecutionSpace space, unsigned int mask) override;
  void modified(ExecutionSpace space, unsigned int mask) override;
  void sync_overlapping_device(ExecutionSpace space, unsigned int mask) override;

 private:
  tagint *molecule;
  tagint **special;
  tagint **bond_atom;
  tagint **angle_atom1,**angle_atom2,**angle_atom3;
  tagint **dihedral_atom1,**dihedral_atom2,**dihedral_atom3,**dihedral_atom4;
  tagint **improper_atom1,**improper_atom2,**improper_atom3,**improper_atom4;

  DAT::t_tagint_1d d_tag;
  DAT::t_imageint_1d d_image;
  DAT::t_int_1d d_type, d_mask;
  DAT::t_x_array d_x;
  DAT::t_v_array d_v;
  DAT::t_f_array d_f;
  DAT::t_tagint_1d d_molecule;
  DAT::t_int_2d d_nspecial;
  DAT::t_tagint_2d d_special;
  DAT::t_int_1d d_num_bond;
  DAT::t_int_2d d_bond_type;
  DAT::t_tagint_2d d_bond_atom;
  DAT::t_int_1d d_num_angle;
  DAT::t_int_2d d_angle_type;
  DAT::t_tagint_2d d_angle_atom1,d_angle_atom2,d_angle_atom3;
  DAT::t_int_1d d_num_dihedral;
  DAT::t_int_2d d_dihedral_type;
  DAT::t_tagint_2d d_dihedral_atom1,d_dihedral_atom2,
    d_dihedral_atom3,d_dihedral_atom4;
  DAT::t_int_1d d_num_improper;
  DAT::t_int_2d d_improper_type;
  DAT::t_tagint_2d d_improper_atom1,d_improper_atom2,
    d_improper_atom3,d_improper_atom4;
};

}    // namespace LAMMPS_NS

#endif
#endif
