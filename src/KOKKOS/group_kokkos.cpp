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

#include "group_kokkos.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "dump.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "input.h"
#include "math_extra.h"
#include "math_eigen.h"
#include "memory.h"
#include "modify.h"
#include "output.h"
#include "region.h"
#include "tokenizer.h"
#include "variable.h"
#include "exceptions.h"

#include <cmath>
#include <cstring>
#include <map>
#include <utility>

using namespace LAMMPS_NS;

static constexpr int MAX_GROUP = 32;
static constexpr double EPSILON = 1.0e-6;

enum{NONE,TYPE,MOLECULE,ID};
enum{LT,LE,GT,GE,EQ,NEQ,BETWEEN};

#define BIG 1.0e20

/* ----------------------------------------------------------------------
   initialize group memory
------------------------------------------------------------------------- */

GroupKokkos::GroupKokkos(LAMMPS *lmp) : Group(lmp)
{
  atomKK = (AtomKokkos *) atom;
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

GroupKokkos::~GroupKokkos()
{
}

// ----------------------------------------------------------------------
// computations on a group of atoms
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   count atoms in group
------------------------------------------------------------------------- */

template<class DeviceType>
bigint GroupKokkos::count_kokkos(int igroup)
{
  int groupbit = bitmask[igroup];

  auto execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(execution_space,MASS_MASK|RMASS_MASK|MASK_MASK|TYPE_MASK);

  auto mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atom->nlocal;

  int n = 0;
  Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int i, int &one) {
    if (mask[i] & groupbit) n++;
  },one);

  bigint nsingle = n;
  bigint nall;
  MPI_Allreduce(&nsingle,&nall,1,MPI_LMP_BIGINT,MPI_SUM,world);
  return nall;
}

/* ----------------------------------------------------------------------
   compute the total mass of group of atoms
   use either per-type mass or per-atom rmass
------------------------------------------------------------------------- */

template<class DeviceType>
double GroupKokkos::mass_kokkos(int igroup)
{
  const int groupbit = bitmask[igroup];

  auto execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(execution_space,MASS_MASK|RMASS_MASK|MASK_MASK|TYPE_MASK);

  auto mass = atomKK->k_mass.view<DeviceType>();
  auto rmass = atomKK->k_rmass.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  int nlocal = atom->nlocal;

  double one = 0.0;

  if (rmass) {
    Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int i, double &one) {
      if (mask[i] & groupbit) one += rmass[i];
    },one);
  } else {
    Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(int i, double &one) {
      if (mask[i] & groupbit) one += mass[i];
    },one);
  }

  double all;
  MPI_Allreduce(&one,&all,1,MPI_DOUBLE,MPI_SUM,world);
  return all;
}

/* ----------------------------------------------------------------------
   compute the center-of-mass coords of group of atoms
   masstotal = total mass
   return center-of-mass coords in cm[]
   must unwrap atoms to compute center-of-mass correctly
------------------------------------------------------------------------- */

template<class DeviceType>
void GroupKokkos::xcm_kokkos(int igroup, double masstotal, double *cm)
{
  int groupbit = bitmask[igroup];

  auto execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(execution_space,X_MASK|MASK_MASK|TYPE_MASK|IMAGE_MASK|MASS_MASK|RMASS_MASK);

  auto x = atomKK->k_x.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto image = atomKK->k_image.view<DeviceType>();
  auto mass = atomKK->k_mass.view<DeviceType>();
  auto rmass = atomKK->k_rmass.view<DeviceType>();
  int nlocal = atom->nlocal;

  double cmone[3];
  cmone[0] = cmone[1] = cmone[2] = 0.0;

  double massone;
  double unwrap[3];

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        massone = rmass[i];
        domain->unmap(x[i],image[i],unwrap);
        cmone[0] += unwrap[0] * massone;
        cmone[1] += unwrap[1] * massone;
        cmone[2] += unwrap[2] * massone;
      }
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        massone = mass[type[i]];
        domain->unmap(x[i],image[i],unwrap);
        cmone[0] += unwrap[0] * massone;
        cmone[1] += unwrap[1] * massone;
        cmone[2] += unwrap[2] * massone;
      }
  }

  MPI_Allreduce(cmone,cm,3,MPI_DOUBLE,MPI_SUM,world);
  if (masstotal > 0.0) {
    cm[0] /= masstotal;
    cm[1] /= masstotal;
    cm[2] /= masstotal;
  }
}

/* ----------------------------------------------------------------------
   compute the center-of-mass velocity of group of atoms
   masstotal = total mass
   return center-of-mass velocity in cm[]
------------------------------------------------------------------------- */

template<class DeviceType>
void GroupKokkos::vcm_kokkos(int igroup, double masstotal, double *cm)
{
  int groupbit = bitmask[igroup];

  auto execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(execution_space,V_MASK|MASK_MASK|TYPE_MASK|MASS_MASK|RMASS_MASK);

  auto v = atomKK->k_v.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto mass = atomKK->k_mass.view<DeviceType>();
  auto rmass = atomKK->k_rmass.view<DeviceType>();
  int nlocal = atom->nlocal;

  double p[3],massone;
  p[0] = p[1] = p[2] = 0.0;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        massone = rmass[i];
        p[0] += v[i][0]*massone;
        p[1] += v[i][1]*massone;
        p[2] += v[i][2]*massone;
      }
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        massone = mass[type[i]];
        p[0] += v[i][0]*massone;
        p[1] += v[i][1]*massone;
        p[2] += v[i][2]*massone;
      }
  }

  MPI_Allreduce(p,cm,3,MPI_DOUBLE,MPI_SUM,world);
  if (masstotal > 0.0) {
    cm[0] /= masstotal;
    cm[1] /= masstotal;
    cm[2] /= masstotal;
  }
}

/* ----------------------------------------------------------------------
   compute the angular momentum L (lmom) of group
   around center-of-mass cm
   must unwrap atoms to compute L correctly
------------------------------------------------------------------------- */

template<class DeviceType>
void GroupKokkos::angmom_kokkos(int igroup, double *cm, double *lmom)
{
  int groupbit = bitmask[igroup];

  auto execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(execution_space,X_MASK|V_MASK|MASK_MASK|TYPE_MASK|IMAGE_MASK|MASS_MASK|RMASS_MASK);

  auto x = atomKK->k_x.view<DeviceType>();
  auto v = atomKK->k_v.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto image = atomKK->k_image.view<DeviceType>();
  auto mass = atomKK->k_mass.view<DeviceType>();
  auto rmass = atomKK->k_rmass.view<DeviceType>();
  int nlocal = atom->nlocal;

  double dx,dy,dz,massone;
  double unwrap[3];

  double p[3];
  p[0] = p[1] = p[2] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - cm[0];
      dy = unwrap[1] - cm[1];
      dz = unwrap[2] - cm[2];
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      p[0] += massone * (dy*v[i][2] - dz*v[i][1]);
      p[1] += massone * (dz*v[i][0] - dx*v[i][2]);
      p[2] += massone * (dx*v[i][1] - dy*v[i][0]);
    }

  MPI_Allreduce(p,lmom,3,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   compute moment of inertia tensor around center-of-mass cm of group
   must unwrap atoms to compute itensor correctly
------------------------------------------------------------------------- */

template<class DeviceType>
void GroupKokkos::inertia_kokkos(int igroup, double *cm, double itensor[3][3])
{
  int i,j;

  int groupbit = bitmask[igroup];

  auto execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(execution_space,X_MASK|MASK_MASK|TYPE_MASK|IMAGE_MASK|MASS_MASK|RMASS_MASK);

  auto x = atomKK->k_x.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto image = atomKK->k_image.view<DeviceType>();
  auto mass = atomKK->k_mass.view<DeviceType>();
  auto rmass = atomKK->k_rmass.view<DeviceType>();
  int nlocal = atom->nlocal;

  double dx,dy,dz,massone;
  double unwrap[3];

  double ione[3][3];
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      ione[i][j] = 0.0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      domain->unmap(x[i],image[i],unwrap);
      dx = unwrap[0] - cm[0];
      dy = unwrap[1] - cm[1];
      dz = unwrap[2] - cm[2];
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      ione[0][0] += massone * (dy*dy + dz*dz);
      ione[1][1] += massone * (dx*dx + dz*dz);
      ione[2][2] += massone * (dx*dx + dy*dy);
      ione[0][1] -= massone * dx*dy;
      ione[1][2] -= massone * dy*dz;
      ione[0][2] -= massone * dx*dz;
    }
  ione[1][0] = ione[0][1];
  ione[2][1] = ione[1][2];
  ione[2][0] = ione[0][2];

  MPI_Allreduce(&ione[0][0],&itensor[0][0],9,MPI_DOUBLE,MPI_SUM,world);
}

