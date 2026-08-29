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

#ifndef LMP_REGION_REMAP_KOKKOS_H
#define LMP_REGION_REMAP_KOKKOS_H

#include "domain.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

/* ----------------------------------------------------------------------
   device-side copy of Domain::remap(double *)

   Region::match() and Region::surface() map a coordinate back into the
   periodic box before testing it, because not every region subclass handles
   a region that reaches past a periodic edge.  The KOKKOS regions run the
   same test inside a kernel, where the Domain instance is not reachable, so
   the few box quantities remap() needs are copied into this struct on the
   host and captured by value with the region object.
------------------------------------------------------------------------- */

struct RegionRemapKokkos {
  int triclinic, xperiodic, yperiodic, zperiodic;
  double lo[3], hi[3], period[3];
  double boxlo[3], h[6], h_inv[6];

  RegionRemapKokkos() : triclinic(0), xperiodic(0), yperiodic(0), zperiodic(0)
  {
    for (int i = 0; i < 3; i++) lo[i] = hi[i] = period[i] = boxlo[i] = 0.0;
    for (int i = 0; i < 6; i++) h[i] = h_inv[i] = 0.0;
  }

  // refresh from the Domain instance; call whenever the box may have changed

  void setup(Domain *domain)
  {
    triclinic = domain->triclinic;
    xperiodic = domain->xperiodic;
    yperiodic = domain->yperiodic;
    zperiodic = domain->zperiodic;
    for (int i = 0; i < 3; i++) {
      if (triclinic) {
        lo[i] = domain->boxlo_lamda[i];
        hi[i] = domain->boxhi_lamda[i];
        period[i] = domain->prd_lamda[i];
      } else {
        lo[i] = domain->boxlo[i];
        hi[i] = domain->boxhi[i];
        period[i] = domain->prd[i];
      }
      boxlo[i] = domain->boxlo[i];
    }
    for (int i = 0; i < 6; i++) {
      h[i] = domain->h[i];
      h_inv[i] = domain->h_inv[i];
    }
  }

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void remap(double &x, double &y, double &z) const
  {
    double coord[3];

    if (triclinic == 0) {
      coord[0] = x; coord[1] = y; coord[2] = z;
    } else {
      const double d0 = x - boxlo[0];
      const double d1 = y - boxlo[1];
      const double d2 = z - boxlo[2];
      coord[0] = h_inv[0]*d0 + h_inv[5]*d1 + h_inv[4]*d2;
      coord[1] = h_inv[1]*d1 + h_inv[3]*d2;
      coord[2] = h_inv[2]*d2;
    }

    if (xperiodic) {
      while (coord[0] < lo[0]) coord[0] += period[0];
      while (coord[0] >= hi[0]) coord[0] -= period[0];
      if (coord[0] < lo[0]) coord[0] = lo[0];
    }

    if (yperiodic) {
      while (coord[1] < lo[1]) coord[1] += period[1];
      while (coord[1] >= hi[1]) coord[1] -= period[1];
      if (coord[1] < lo[1]) coord[1] = lo[1];
    }

    if (zperiodic) {
      while (coord[2] < lo[2]) coord[2] += period[2];
      while (coord[2] >= hi[2]) coord[2] -= period[2];
      if (coord[2] < lo[2]) coord[2] = lo[2];
    }

    if (triclinic == 0) {
      x = coord[0]; y = coord[1]; z = coord[2];
    } else {
      x = h[0]*coord[0] + h[5]*coord[1] + h[4]*coord[2] + boxlo[0];
      y = h[1]*coord[1] + h[3]*coord[2] + boxlo[1];
      z = h[2]*coord[2] + boxlo[2];
    }
  }
};

}    // namespace LAMMPS_NS

#endif
