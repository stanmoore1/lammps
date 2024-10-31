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

  void sync(ExecutionSpace space, unsigned int mask) override;
  void modified(ExecutionSpace space, unsigned int mask) override;
  void sync_overlapping_device(ExecutionSpace space, unsigned int mask) override;

 protected:


};

}    // namespace LAMMPS_NS

#endif
#endif
