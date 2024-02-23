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
AtomStyle(atomic/kk,AtomVecAtomicKokkos);
AtomStyle(atomic/kk/device,AtomVecAtomicKokkos);
AtomStyle(atomic/kk/host,AtomVecAtomicKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_ATOMIC_KOKKOS_H
#define LMP_ATOM_VEC_ATOMIC_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_atomic.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomVecAtomicKokkos : public AtomVecKokkos, public AtomVecAtomic {
 public:
  AtomVecAtomicKokkos(class LAMMPS *);

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
  DAT::t_tagint_1d d_tag;
  DAT::t_imageint_1d d_image;
  DAT::t_int_1d d_type, d_mask;
  DAT::t_x_array d_x;
  DAT::t_v_array d_v;
  DAT::t_f_array d_f;
};

}    // namespace LAMMPS_NS

#endif
#endif
