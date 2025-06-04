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
AtomStyle(oxdna/kk,AtomVecOxdnaKokkos);
AtomStyle(oxdna/kk/device,AtomVecOxdnaKokkos);
AtomStyle(oxdna/kk/host,AtomVecOxdnaKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_OXDNA_KOKKOS_H
#define LMP_ATOM_VEC_OXDNA_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_oxdna.h"

namespace LAMMPS_NS {

class AtomVecOxdnaKokkos : public AtomVecKokkos, public AtomVecOxdna {
 public:
  AtomVecOxdnaKokkos(class LAMMPS *);

  void grow(int) override;
  void grow_pointers() override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;

  // [un]packs not needed for atom_vec_oxdna[_kokkos], however need to be
  // overridden to avoid pure virtual function compile error
  int pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                         DAT::tdual_double_2d_lr buf,
                         int pbc_flag, int *pbc, ExecutionSpace space) 
                         override {return 0;};
  void unpack_border_kokkos(const int &n, const int &nfirst,
                            const DAT::tdual_double_2d_lr &buf,
                            ExecutionSpace space) override {};
  int pack_exchange_kokkos(const int &nsend,DAT::tdual_double_2d_lr &buf,
                           DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist,
                           DAT::tdual_int_1d k_sendlist_bonus,
                           DAT::tdual_int_1d k_copylist_bonus,
                           ExecutionSpace space) 
                           override {return 0;};
  int unpack_exchange_kokkos(DAT::tdual_double_2d_lr &k_buf, int nrecv,
                             int nlocal, int dim, double lo, double hi,
                             ExecutionSpace space,
                             DAT::tdual_int_1d &k_indices) 
                             override {return 0;};

  void sync(ExecutionSpace space, unsigned int mask) override;
  void modified(ExecutionSpace space, unsigned int mask) override;
  void sync_pinned_device(ExecutionSpace space, unsigned int mask) override;

 private:
   DAT::t_tagint_1d d_id3p;
   HAT::t_tagint_1d h_id3p;
   DAT::t_tagint_1d d_id5p;
   HAT::t_tagint_1d h_id5p;

};

}    // namespace LAMMPS_NS

#endif
#endif
