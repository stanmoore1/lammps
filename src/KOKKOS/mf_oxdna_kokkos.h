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

#ifndef MF_OXDNA_KOKKOS_H
#define MF_OXDNA_KOKKOS_H

#include "kokkos_base.h"

namespace LAMMPS_NS {

template<class DeviceType>
class mfOxdnaKokkos : public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  mfOxdnaKokkos(class LAMMPS *);
  ~mfOxdnaKokkos();

  //class mfOxdnaKokkos *&mfOxdnaKK;

  KOKKOS_INLINE_FUNCTION
  F_FLOAT oxDNA_F1_KK(F_FLOAT, F_FLOAT, F_FLOAT, F_FLOAT, F_FLOAT, F_FLOAT, F_FLOAT, F_FLOAT, F_FLOAT, F_FLOAT,
                 F_FLOAT);

};

}    // namespace LAMMPS_NS

#endif