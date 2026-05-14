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

#include "compute_temp_mwindow_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "modify.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeTempMWindowKokkos<DeviceType>::ComputeTempMWindowKokkos(LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp, narg, arg)
{
  if (narg != 6) error->all(FLERR, "Illegal compute temp/mwindow command");

  vbias[0] = utils::numeric(FLERR, arg[3], false, lmp);
  vbias[1] = utils::numeric(FLERR, arg[4], false, lmp);
  vbias[2] = utils::numeric(FLERR, arg[5], false, lmp);

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;
  tempflag = 1;
  tempbias = 1;

  vector = new double[6];

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = V_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeTempMWindowKokkos<DeviceType>::~ComputeTempMWindowKokkos()
{
  if (copymode) return;

  delete [] vector;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::init()
{
  fix_dof = 0;
  for (int i = 0; i < modify->nfix; i++)
    fix_dof += modify->fix[i]->dof(igroup);
  dof_compute();
  masstotal = group->mass(igroup);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::dof_compute()
{
  double natoms = group->count(igroup);
  int nper = domain->dimension;
  dof = nper * natoms;
  dof -= extra_dof + fix_dof;
  if (dof > 0) tfactor = force->mvv2e / (dof * force->boltz);
  else tfactor = 0.0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double ComputeTempMWindowKokkos<DeviceType>::compute_scalar()
{
  if (invoked_scalar == update->ntimestep) return scalar;
  invoked_scalar = update->ntimestep;

  atomKK->sync(execution_space, datamask_read);
  if (!atomKK->rmass) atomKK->k_mass.sync<DeviceType>();

  v    = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  if (atomKK->rmass)
    rmass = atomKK->k_rmass.view<DeviceType>();
  else
    mass  = atomKK->k_mass.view<DeviceType>();

  const int nlocal = atom->nlocal;
  if (dynamic) masstotal = group->mass(igroup);

  CTEMP t_kk;
  copymode = 1;
  if (atomKK->rmass)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,
      TagComputeTempMWindowScalar<1>>(0, nlocal), *this, t_kk);
  else
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,
      TagComputeTempMWindowScalar<0>>(0, nlocal), *this, t_kk);
  copymode = 0;

  double t = t_kk.t0;
  MPI_Allreduce(&t, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
  if (dynamic) dof_compute();
  scalar *= tfactor;
  return scalar;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int RMASS>
KOKKOS_INLINE_FUNCTION
void ComputeTempMWindowKokkos<DeviceType>::operator()(
    TagComputeTempMWindowScalar<RMASS>, const int &i, CTEMP &t_kk) const
{
  if (mask[i] & groupbit) {
    const KK_FLOAT vx = v(i,0) - (KK_FLOAT)vbias[0];
    const KK_FLOAT vy = v(i,1) - (KK_FLOAT)vbias[1];
    const KK_FLOAT vz = v(i,2) - (KK_FLOAT)vbias[2];
    const KK_FLOAT m = RMASS ? rmass[i] : mass[type[i]];
    t_kk.t0 += m * (vx*vx + vy*vy + vz*vz);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::compute_vector()
{
  invoked_vector = update->ntimestep;

  atomKK->sync(execution_space, datamask_read);
  if (!atomKK->rmass) atomKK->k_mass.sync<DeviceType>();

  v    = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  if (atomKK->rmass)
    rmass = atomKK->k_rmass.view<DeviceType>();
  else
    mass  = atomKK->k_mass.view<DeviceType>();

  const int nlocal = atom->nlocal;
  if (dynamic) masstotal = group->mass(igroup);

  CTEMP t_kk;
  copymode = 1;
  if (atomKK->rmass)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,
      TagComputeTempMWindowVector<1>>(0, nlocal), *this, t_kk);
  else
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,
      TagComputeTempMWindowVector<0>>(0, nlocal), *this, t_kk);
  copymode = 0;

  double t[6] = {t_kk.t0, t_kk.t1, t_kk.t2, t_kk.t3, t_kk.t4, t_kk.t5};
  MPI_Allreduce(t, vector, 6, MPI_DOUBLE, MPI_SUM, world);
  for (int i = 0; i < 6; i++) vector[i] *= force->mvv2e;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int RMASS>
KOKKOS_INLINE_FUNCTION
void ComputeTempMWindowKokkos<DeviceType>::operator()(
    TagComputeTempMWindowVector<RMASS>, const int &i, CTEMP &t_kk) const
{
  if (mask[i] & groupbit) {
    const KK_FLOAT vx = v(i,0) - (KK_FLOAT)vbias[0];
    const KK_FLOAT vy = v(i,1) - (KK_FLOAT)vbias[1];
    const KK_FLOAT vz = v(i,2) - (KK_FLOAT)vbias[2];
    const KK_FLOAT m = RMASS ? rmass[i] : mass[type[i]];
    t_kk.t0 += m * vx*vx;
    t_kk.t1 += m * vy*vy;
    t_kk.t2 += m * vz*vz;
    t_kk.t3 += m * vx*vy;
    t_kk.t4 += m * vx*vz;
    t_kk.t5 += m * vy*vz;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::remove_bias(int /*i*/, double *vel)
{
  vel[0] -= vbias[0];
  vel[1] -= vbias[1];
  vel[2] -= vbias[2];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::remove_bias_all()
{
  atomKK->sync(execution_space, V_MASK | MASK_MASK);
  v    = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,
    TagComputeTempMWindowRemoveBias>(0, atom->nlocal), *this);
  copymode = 0;

  atomKK->modified(execution_space, V_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void ComputeTempMWindowKokkos<DeviceType>::operator()(
    TagComputeTempMWindowRemoveBias, const int &i) const
{
  if (mask[i] & groupbit) {
    v(i,0) -= (KK_FLOAT)vbias[0];
    v(i,1) -= (KK_FLOAT)vbias[1];
    v(i,2) -= (KK_FLOAT)vbias[2];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::restore_bias(int /*i*/, double *vel)
{
  vel[0] += vbias[0];
  vel[1] += vbias[1];
  vel[2] += vbias[2];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::restore_bias_all()
{
  atomKK->sync(execution_space, V_MASK | MASK_MASK);
  v    = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,
    TagComputeTempMWindowRestoreBias>(0, atom->nlocal), *this);
  copymode = 0;

  atomKK->modified(execution_space, V_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void ComputeTempMWindowKokkos<DeviceType>::operator()(
    TagComputeTempMWindowRestoreBias, const int &i) const
{
  if (mask[i] & groupbit) {
    v(i,0) += (KK_FLOAT)vbias[0];
    v(i,1) += (KK_FLOAT)vbias[1];
    v(i,2) += (KK_FLOAT)vbias[2];
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class ComputeTempMWindowKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeTempMWindowKokkos<LMPHostType>;
#endif
}
