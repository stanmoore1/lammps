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
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group_kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeTempMWindowKokkos<DeviceType>::ComputeTempMWindowKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputeTempMWindow(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  groupKK = (GroupKokkos *) group;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = V_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double ComputeTempMWindowKokkos<DeviceType>::compute_scalar()
{
  atomKK->sync(execution_space,datamask_read);
  atomKK->k_mass.sync<DeviceType>();

  invoked_scalar = update->ntimestep;

  if (dynamic) masstotal = groupKK->mass_kk<DeviceType>(igroup);
  groupKK->vcm_kk<DeviceType>(igroup,masstotal,vbias);

  v = atomKK->k_v.view<DeviceType>();
  if (atomKK->rmass)
    rmass = atomKK->k_rmass.view<DeviceType>();
  else
    mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atom->nlocal;

  double t = 0.0;
  CTEMP t_kk;

  copymode = 1;
  if (atomKK->rmass)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeTempMWindowScalar<1> >(0,nlocal),*this,t_kk);
  else
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeTempMWindowScalar<0> >(0,nlocal),*this,t_kk);
  copymode = 0;

  t = t_kk.t0; // could make this more efficient

  MPI_Allreduce(&t,&scalar,1,MPI_DOUBLE,MPI_SUM,world);
  if (dynamic) dof_compute();
  if (dof < 0.0 && natoms_temp > 0.0)
    error->all(FLERR,"Temperature compute degrees of freedom < 0");
  scalar *= tfactor;

  return scalar;
}

template<class DeviceType>
template<int RMASS>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputeTempMWindowKokkos<DeviceType>::operator()(TagComputeTempMWindowScalar<RMASS>, const int &i, CTEMP& t_kk) const {
  if (RMASS) {
    if (mask[i] & groupbit) {
      const KK_FLOAT vx = v(i,0) - (KK_FLOAT)vbias[0];
      const KK_FLOAT vy = v(i,1) - (KK_FLOAT)vbias[1];
      const KK_FLOAT vz = v(i,2) - (KK_FLOAT)vbias[2];
      t_kk.t0 += (vx*vx + vy*vy + vz*vz) * rmass[i];
    }
  } else {
    if (mask[i] & groupbit) {
      const KK_FLOAT vx = v(i,0) - (KK_FLOAT)vbias[0];
      const KK_FLOAT vy = v(i,1) - (KK_FLOAT)vbias[1];
      const KK_FLOAT vz = v(i,2) - (KK_FLOAT)vbias[2];
      t_kk.t0 += (vx*vx + vy*vy + vz*vz) * mass[type[i]];
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::compute_vector()
{
  atomKK->sync(execution_space,datamask_read);
  atomKK->k_mass.sync<DeviceType>();

  int i;

  invoked_vector = update->ntimestep;

  if (dynamic) masstotal = groupKK->mass_kk<DeviceType>(igroup);
  groupKK->vcm_kk<DeviceType>(igroup,masstotal,vbias);

  v = atomKK->k_v.view<DeviceType>();
  if (atomKK->rmass)
    rmass = atomKK->k_rmass.view<DeviceType>();
  else
    mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atom->nlocal;

  double t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;
  CTEMP t_kk;

  copymode = 1;
  if (atomKK->rmass)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeTempMWindowVector<1> >(0,nlocal),*this,t_kk);
  else
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeTempMWindowVector<0> >(0,nlocal),*this,t_kk);
  copymode = 0;

  t[0] = t_kk.t0;
  t[1] = t_kk.t1;
  t[2] = t_kk.t2;
  t[3] = t_kk.t3;
  t[4] = t_kk.t4;
  t[5] = t_kk.t5;

  MPI_Allreduce(t,vector,6,MPI_DOUBLE,MPI_SUM,world);
  for (i = 0; i < 6; i++) vector[i] *= force->mvv2e;
}

template<class DeviceType>
template<int RMASS>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputeTempMWindowKokkos<DeviceType>::operator()(TagComputeTempMWindowVector<RMASS>, const int &i, CTEMP& t_kk) const {
  if (mask[i] & groupbit) {
    KK_FLOAT massone = 0.0;
    if (RMASS) massone = rmass[i];
    else massone = mass[type[i]];
    const KK_FLOAT vx = v(i,0) - (KK_FLOAT)vbias[0];
    const KK_FLOAT vy = v(i,1) - (KK_FLOAT)vbias[1];
    const KK_FLOAT vz = v(i,2) - (KK_FLOAT)vbias[2];
    t_kk.t0 += massone * vx*vx;
    t_kk.t1 += massone * vy*vy;
    t_kk.t2 += massone * vz*vz;
    t_kk.t3 += massone * vx*vy;
    t_kk.t4 += massone * vx*vz;
    t_kk.t5 += massone * vy*vz;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::remove_bias_all()
{
  remove_bias_all_kk();
  atomKK->sync(Host,V_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::remove_bias_all_kk()
{
  atomKK->sync(execution_space,V_MASK|MASK_MASK);
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atom->nlocal;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeTempMWindowRemoveBias >(0,nlocal), *this);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputeTempMWindowKokkos<DeviceType>::operator()(TagComputeTempMWindowRemoveBias, const int &i) const {
  if (mask[i] & groupbit) {
    v(i,0) -= (KK_FLOAT)vbias[0];
    v(i,1) -= (KK_FLOAT)vbias[1];
    v(i,2) -= (KK_FLOAT)vbias[2];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::restore_bias_all()
{
  restore_bias_all_kk();
  atomKK->sync(Host,V_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempMWindowKokkos<DeviceType>::restore_bias_all_kk()
{
  atomKK->sync(execution_space,V_MASK|MASK_MASK);
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atom->nlocal;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagComputeTempMWindowRestoreBias>(0, atom->nlocal), *this);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputeTempMWindowKokkos<DeviceType>::operator()(TagComputeTempMWindowRestoreBias, const int &i) const {
  if (mask[i] & groupbit) {
    v(i,0) += (KK_FLOAT)vbias[0];
    v(i,1) += (KK_FLOAT)vbias[1];
    v(i,2) += (KK_FLOAT)vbias[2];
  }
}

namespace LAMMPS_NS {
template class ComputeTempMWindowKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeTempMWindowKokkos<LMPHostType>;
#endif
}
