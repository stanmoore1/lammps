// clang-format off
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

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "fix_recenter_kokkos.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "lattice.h"
#include "modify.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{BOX,LATTICE,FRACTION};

/* ---------------------------------------------------------------------- */

template <class DeviceType>
FixRecenterKokkos<DeviceType>::FixRecenterKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixRecenter(lmp, narg, arg)
{
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixRecenterKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  // target COM
  // bounding box around domain works for both orthogonal and triclinic

  double xtarget,ytarget,ztarget;
  double *bboxlo,*bboxhi;

  if (scaleflag == FRACTION) {
    if (domain->triclinic == 0) {
      bboxlo = domain->boxlo;
      bboxhi = domain->boxhi;
    } else {
      bboxlo = domain->boxlo_bound;
      bboxhi = domain->boxhi_bound;
    }
  }

  if (xinitflag) xtarget = xinit;
  else if (scaleflag == FRACTION)
    xtarget = bboxlo[0] + xcom*(bboxhi[0] - bboxlo[0]);
  else xtarget = xcom;

  if (yinitflag) ytarget = yinit;
  else if (scaleflag == FRACTION)
    ytarget = bboxlo[1] + ycom*(bboxhi[1] - bboxlo[1]);
  else ytarget = ycom;

  if (zinitflag) ztarget = zinit;
  else if (scaleflag == FRACTION)
    ztarget = bboxlo[2] + zcom*(bboxhi[2] - bboxlo[2]);
  else ztarget = zcom;

  // current COM

  double xcm[3];
  if (group->dynamic[igroup])
    masstotal = group->mass(igroup);

  group->xcm(igroup,masstotal,xcm);

  // shift coords by difference between actual COM and requested COM

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  shift[0] = xflag ? (xtarget - xcm[0]) : 0.0;
  shift[1] = yflag ? (ytarget - xcm[1]) : 0.0;
  shift[2] = zflag ? (ztarget - xcm[2]) : 0.0;
  distance = sqrt(shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2]);

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & group2bit) {
      x[i][0] += shift[0];
      x[i][1] += shift[1];
      x[i][2] += shift[2];
    }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixEnforce2DKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixEnforce2DKokkos<LMPHostType>;
#endif
}
