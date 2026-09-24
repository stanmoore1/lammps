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
PairStyle(oxrna2/hbond/kk,PairOxrna2HbondKokkos<LMPDeviceType>);
PairStyle(oxrna2/hbond/kk/device,PairOxrna2HbondKokkos<LMPDeviceType>);
PairStyle(oxrna2/hbond/kk/host,PairOxrna2HbondKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_OXRNA2_HBOND_KOKKOS_H
#define LMP_PAIR_OXRNA2_HBOND_KOKKOS_H

#include "pair_oxdna_hbond_kokkos.h"
#include "pair_oxrna2_hbond.h"

// oxRNA2 uses the oxDNA base site, only the parameters differ

namespace LAMMPS_NS {

template<class DeviceType>
class PairOxrna2HbondKokkos : public PairOxdnaHbondKokkosT<DeviceType, PairOxrna2Hbond> {
 public:
  PairOxrna2HbondKokkos(class LAMMPS *lmp) :
      PairOxdnaHbondKokkosT<DeviceType, PairOxrna2Hbond>(lmp,
          PairOxdnaHbondKokkosT<DeviceType, PairOxrna2Hbond>::OXDNA) {}
};

}    // namespace LAMMPS_NS

#endif
#endif
