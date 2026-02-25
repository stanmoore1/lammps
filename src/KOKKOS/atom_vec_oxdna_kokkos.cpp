// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
//TODO: Check "BOND_MASK" is the correct MASK for oxdnaKK
#include "atom_vec_oxdna_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "memory_kokkos.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecOxdnaKokkos::AtomVecOxdnaKokkos(LAMMPS *lmp) : AtomVec(lmp),
AtomVecKokkos(lmp), AtomVecOxdna(lmp)
{
}

/* ----------------------------------------------------------------------
   process field strings to initialize data structs for all other methods
------------------------------------------------------------------------- */

void AtomVecOxdnaKokkos::init()
{
  AtomVecOxdna::init();

  set_atom_masks();
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by DELTA
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecOxdnaKokkos::grow(int n)
{
  auto DELTA = LMP_KOKKOS_AV_DELTA;
  int step = MAX(DELTA,nmax*0.01);
  if (n == 0) nmax += step;
  else nmax = n;
  atomKK->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  atomKK->sync(Device,ALL_MASK);
  atomKK->modified(Device,ALL_MASK);

  memoryKK->grow_kokkos(atomKK->k_id3p,atomKK->id3p,nmax,"atom:id3p");
  memoryKK->grow_kokkos(atomKK->k_id5p,atomKK->id5p,nmax,"atom:id5p");

  grow_pointers();
  atomKK->sync(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecOxdnaKokkos::grow_pointers()
{
  id3p = atomKK->id3p;
  d_id3p = atomKK->k_id3p.d_view;
  h_id3p = atomKK->k_id3p.h_view;
  id5p = atomKK->id5p;
  d_id5p = atomKK->k_id5p.d_view;
  h_id5p = atomKK->k_id5p.h_view;
}

/* ----------------------------------------------------------------------
   sort atom arrays on device
------------------------------------------------------------------------- */

void AtomVecOxdnaKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  atomKK->sync(Device, ALL_MASK & ~F_MASK);

  Sorter.sort(LMPDeviceType(), d_id3p);
  Sorter.sort(LMPDeviceType(), d_id5p);

  atomKK->modified(Device, ALL_MASK & ~F_MASK);
}

/* ---------------------------------------------------------------------- */

void AtomVecOxdnaKokkos::sync(ExecutionSpace space, uint64_t mask)
{
  if (space == Device) {
    if (mask & BOND_MASK) atomKK->k_id3p.k_x.sync_device();
    if (mask & BOND_MASK) atomKK->k_id5p.k_x.sync_device();
  } else {
    if (mask & BOND_MASK) atomKK->k_id3p.k_x.sync_host();
    if (mask & BOND_MASK) atomKK->k_id5p.k_x.sync_host();
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecOxdnaKokkos::sync_pinned(ExecutionSpace space, uint64_t mask, int async_flag)
{
  if (space == Device) {
    if ((mask & BOND_MASK) && atomKK->k_id3p.need_sync_device())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_id3p,space,async_flag);
    if ((mask & BOND_MASK) && atomKK->k_id5p.need_sync_device())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_id5p,space,async_flag);
  } else {
    if ((mask & BOND_MASK) && atomKK->k_id3p.need_sync_host())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_id3p,space,async_flag);
    if ((mask & BOND_MASK) && atomKK->k_id5p.need_sync_host())
      perform_pinned_copy<DAT::tdual_tagint_1d>(atomKK->k_id5p,space,async_flag);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecOxdnaKokkos::modified(ExecutionSpace space, uint64_t mask)
{
  if (space == Device) {
    if (mask & BOND_MASK) atomKK->k_id3p.modify_device();
    if (mask & BOND_MASK) atomKK->k_id5p.modify_device();
  } else {
    if (mask & BOND_MASK) atomKK->k_id3p.modify_host();
    if (mask & BOND_MASK) atomKK->k_id5p.modify_host();
  }
}
