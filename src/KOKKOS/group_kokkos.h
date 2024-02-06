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

#ifndef LMP_GROUP_KOKKOS_H
#define LMP_GROUP_KOKKOS_H

#include "group.h"

namespace LAMMPS_NS {

class GroupKokkos : public Grouop {
 public:
  GroupKokkos(class LAMMPS *);
  ~GroupKokkos() override;

  bigint count_kokkos(int);              // count atoms in group
  double mass_kokkos(int);               // total mass of atoms in group
  void xcm_kokkos(int, double, double *);    // center-of-mass coords of group
  void vcm_kokkos(int, double, double *);    // center-of-mass velocity of group
  void angmom_kokkos(int, double *, double *);    // angular momentum of group
  void inertia_kokkos(int, double *, double[3][3]);    // inertia tensor

 private:
};

}    // namespace LAMMPS_NS

#endif
