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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(oxdna3/hbond/kk,PairOxdna3HbondKokkos<LMPDeviceType>);
PairStyle(oxdna3/hbond/kk/device,PairOxdna3HbondKokkos<LMPDeviceType>);
PairStyle(oxdna3/hbond/kk/host,PairOxdna3HbondKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_OXDNA3_HBOND_KOKKOS_H
#define LMP_PAIR_OXDNA3_HBOND_KOKKOS_H

#include "pair_oxdna_hbond_kokkos.h"
#include "pair_oxdna3_hbond.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairOxdna3HbondKokkos :
    public PairOxdnaHbondKokkosT<DeviceType, PairOxdna3Hbond, PairOxdnaHbondModel::OXDNA3> {
 public:
  PairOxdna3HbondKokkos(class LAMMPS *lmp) :
      PairOxdnaHbondKokkosT<DeviceType, PairOxdna3Hbond, PairOxdnaHbondModel::OXDNA3>(lmp) {}
};

}    // namespace LAMMPS_NS

#endif
#endif
