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

#include "fix_wall_ylpiston_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixWallYLPistonKokkos<DeviceType>::FixWallYLPistonKokkos(
    LAMMPS *lmp, int narg, char **arg)
  : FixWallYLPiston(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read  = X_MASK | MASK_MASK;
  datamask_modify = F_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixWallYLPistonKokkos<DeviceType>::~FixWallYLPistonKokkos()
{
  if (copymode) return;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixWallYLPistonKokkos<DeviceType>::post_force(int /*vflag*/)
{
  wall[0] = wall[1] = wall[2] = wall[3] = 0.0;
  wall_flag = 0;

  // compute the erasing-plane position (host scalar, from fix_mw)
  d_der    = (ifix_mw < 0) ? 0.0 : fix_mw->compute_scalar();
  d_coord  = coord;
  d_Edeep3 = Edeep3;
  d_cutoff = cutoff;
  d_dim    = dim;
  d_side   = side;
  d_ifix_mw = ifix_mw;

  atomKK->sync(execution_space, datamask_read);
  d_x    = atomKK->k_x.view<DeviceType>();
  d_f    = atomKK->k_f.view<DeviceType>();
  d_mask = atomKK->k_mask.view<DeviceType>();

  const int nlocal = atom->nlocal;

  EWALL wall_kk;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,
    TagFixWallYLPistonForce>(0, nlocal), *this, wall_kk);
  copymode = 0;

  atomKK->modified(execution_space, F_MASK);

  wall[0] = wall_kk.w[0];
  wall[1] = wall_kk.w[1];
  wall[2] = wall_kk.w[2];
  wall[3] = wall_kk.w[3];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixWallYLPistonKokkos<DeviceType>::operator()(
    TagFixWallYLPistonForce, const int &i, EWALL &wall_kk) const
{
  if (!(d_mask[i] & groupbit)) return;

  double delta;
  if (d_side == -1) delta = d_x(i, d_dim) - d_coord;
  else              delta = d_coord - d_x(i, d_dim);

  if (delta <= 0.0 || delta > d_cutoff) return;
  delta = d_cutoff - delta;

  double Etemp = d_Edeep3 * delta;
  if (d_ifix_mw >= 0) {
    if (d_side == -1 && d_x(i, d_dim) < d_der) Etemp = 0.0;
    else if (d_side == 1 && d_x(i, d_dim) > d_der) Etemp = 0.0;
  }

  const double fpiston = Etemp * d_side * 2.0;
  d_f(i, d_dim) -= fpiston;
  wall_kk.w[0]         += Etemp * delta;
  wall_kk.w[d_dim + 1] += fpiston;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixWallYLPistonKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixWallYLPistonKokkos<LMPHostType>;
#endif
}
