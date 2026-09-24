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

#ifdef BOND_CLASS
// clang-format off
BondStyle(oxdna3/fene/kk,BondOxdna3FENEKokkos<LMPDeviceType>);
BondStyle(oxdna3/fene/kk/device,BondOxdna3FENEKokkos<LMPDeviceType>);
BondStyle(oxdna3/fene/kk/host,BondOxdna3FENEKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_BOND_OXDNA3_FENE_KOKKOS_H
#define LMP_BOND_OXDNA3_FENE_KOKKOS_H

#include "bond_oxdna_fene_kokkos.h"
#include "bond_oxdna3_fene.h"

// oxDNA3 uses the oxDNA2 backbone site, only the parameters and their parsing differ

namespace LAMMPS_NS {

template<class DeviceType>
class BondOxdna3FENEKokkos : public BondOxdnaFENEKokkosT<DeviceType, BondOxdna3Fene> {
 public:
  BondOxdna3FENEKokkos(class LAMMPS *lmp) :
      BondOxdnaFENEKokkosT<DeviceType, BondOxdna3Fene>(lmp,
          BondOxdnaFENEKokkosT<DeviceType, BondOxdna3Fene>::OXDNA2) {}
};

}    // namespace LAMMPS_NS

#endif
#endif
