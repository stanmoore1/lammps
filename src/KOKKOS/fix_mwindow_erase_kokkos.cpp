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

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "fix_mwindow_erase_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "atom_vec.h"
#include "error.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixMWindowEraseKokkos<DeviceType>::FixMWindowEraseKokkos(
    LAMMPS *lmp, int narg, char **arg)
  : FixMWindowErase(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read  = X_MASK | MASK_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixMWindowEraseKokkos<DeviceType>::~FixMWindowEraseKokkos()
{
  if (copymode) return;
  // parent destructor handles mark/list raw pointers;
  // k_mark dual view cleans itself up
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMWindowEraseKokkos<DeviceType>::pre_exchange()
{
  if (update->ntimestep != next_reneighbor && update->ntimestep != 1) return;

  // grow parent's raw arrays AND the Kokkos dual-view together
  if (atom->nlocal > nmax) {
    memory->sfree(list);
    memory->sfree(mark);
    nmax = atom->nmax;
    list = (int *) memory->smalloc(nmax * sizeof(int), "mwindow/erase:list");
    mark = (int *) memory->smalloc(nmax * sizeof(int), "mwindow/erase:mark");
    k_mark.resize(nmax);
    d_mark = k_mark.template view<DeviceType>();
    h_mark = k_mark.view<LMPHostType>();
  }

  // cache kernel parameters before entering copymode
  d_erase_pos  = mw_erase_position_d;
  d_erase_dim  = mw_erase_dim;
  d_erase_side = mw_erase_side;

  // sync positions and mask to the execution space
  atomKK->sync(execution_space, datamask_read);
  d_x    = atomKK->k_x.view<DeviceType>();
  d_mask = atomKK->k_mask.view<DeviceType>();

  const int nlocal = atom->nlocal;

  // mark atoms on device
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,
    TagFixMWindowEraseMarkAtoms>(0, nlocal), *this);
  copymode = 0;

  // count marked atoms on device
  int nall = 0;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,
    TagFixMWindowEraseCountMarked>(0, nlocal), *this, nall);
  copymode = 0;

  int nwhack = 0;
  MPI_Allreduce(&nall, &nwhack, 1, MPI_INT, MPI_SUM, world);

  // sync mark to host for the serial deletion step
  k_mark.template sync<LMPHostType>();

  // delete marked atoms (must be done on host via avec->copy)
  AtomVec *avec = atom->avec;
  for (int i = nlocal - 1; i >= 0; i--) {
    if (h_mark(i)) {
      avec->copy(atom->nlocal - 1, i, 1);
      atom->nlocal--;
    }
  }

  // reset global natoms; rebuild map if needed
  atom->natoms -= nwhack;
  if (nwhack && atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }

  // mark all atom data as modified on host (avec->copy changed host arrays)
  if (nwhack)
    atomKK->modified(Host, X_MASK | V_MASK | F_MASK | MASK_MASK |
                     TYPE_MASK | RMASS_MASK | IMAGE_MASK);

  ndeleted = nwhack;
  if (update->ntimestep == next_reneighbor)
    next_reneighbor = update->ntimestep + nfreq;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixMWindowEraseKokkos<DeviceType>::operator()(
    TagFixMWindowEraseMarkAtoms, const int &i) const
{
  if (d_mask[i] & groupbit) {
    const double xi = d_x(i, d_erase_dim);
    const double delta = (d_erase_side == -1) ? (xi - d_erase_pos)
                                               : (d_erase_pos - xi);
    d_mark[i] = (delta <= 0.0) ? 1 : 0;
  } else {
    d_mark[i] = 0;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixMWindowEraseKokkos<DeviceType>::operator()(
    TagFixMWindowEraseCountMarked, const int &i, int &count) const
{
  count += d_mark[i];
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixMWindowEraseKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixMWindowEraseKokkos<LMPHostType>;
#endif
}
