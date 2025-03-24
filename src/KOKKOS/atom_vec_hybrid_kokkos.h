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
AtomStyle(hybrid/kk,AtomVecHybridKokkos);
AtomStyle(hybrid/kk/device,AtomVecHybridKokkos);
AtomStyle(hybrid/kk/host,AtomVecHybridKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_HYBRID_KOKKOS_H
#define LMP_ATOM_VEC_HYBRID_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_hybrid.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomVecHybridKokkos : public AtomVecKokkos, public AtomVecHybrid {
 public:
  AtomVecHybridKokkos(class LAMMPS *);
  void process_args(int, char **) override;

  void grow(int) override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;

  int pack_comm_kokkos(const int &n, const DAT::tdual_int_1d &k_sendlist,
                       const DAT::tdual_xfloat_2d &buf,
                       const int &pbc_flag, const int pbc[]) override;
  void unpack_comm_kokkos(const int &n, const int &nfirst,
                          const DAT::tdual_xfloat_2d &buf) override;
  int pack_comm_self(const int &n, const DAT::tdual_int_1d &list,
                     const int nfirst,
                     const int &pbc_flag, const int pbc[]) override;
  int pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                         DAT::tdual_xfloat_2d buf,
                         int pbc_flag, int *pbc, ExecutionSpace space) override;
  void unpack_border_kokkos(const int &n, const int &nfirst,
                            const DAT::tdual_xfloat_2d &buf,
                            ExecutionSpace space) override;
  int pack_exchange_kokkos(const int &nsend,DAT::tdual_xfloat_2d &buf,
                           DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist,
                           DAT::tdual_int_1d k_sendlist_bonus,
                           DAT::tdual_int_1d k_copylist_bonus,
                           ExecutionSpace space) override;
  int unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv,
                             int nlocal, int dim, X_FLOAT lo, X_FLOAT hi,
                             ExecutionSpace space,
                             DAT::tdual_int_1d &k_indices) override;

  void sync(ExecutionSpace space, unsigned int mask) override;
  void modified(ExecutionSpace space, unsigned int mask) override;
  void sync_overlapping_device(ExecutionSpace space, unsigned int mask) override;

 private:
  AtomVecKokkos **nstyles_cast;
};

} // namespace LAMMPS_NS

#endif
#endif
