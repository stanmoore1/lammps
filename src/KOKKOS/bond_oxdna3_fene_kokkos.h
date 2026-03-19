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

namespace LAMMPS_NS {

template<class DeviceType>
class BondOxdna3FENEKokkos : public BondOxdnaFENEKokkos<DeviceType> {
 public:
  BondOxdna3FENEKokkos(class LAMMPS *);
  ~BondOxdna3FENEKokkos() {}
  void coeff(int, char **) override;
};

template<class DeviceType>
BondOxdna3FENEKokkos<DeviceType>::BondOxdna3FENEKokkos(LAMMPS *lmp) : BondOxdnaFENEKokkos<DeviceType>(lmp)
{
   this->oxdnaflag = BondOxdnaFENEKokkos<DeviceType>::EnabledOXDNAFlag::OXDNA2; // oxDNA3 uses same as OXDNA2 here
}

template<class DeviceType>
void BondOxdna3FENEKokkos<DeviceType>::coeff(int narg, char **arg)
{
  this->coeff_oxdna3_common(narg, arg);

  // Unlike vanilla, we don't use the bounds and assert - args have already
  // been parsed.

  int m = this->atom->nbondtypes;
  int n = this->atom->ntypes;
  for (int i = 1; i <= m; i++) {
    this->k_k.view_host()[i] = this->k[i];
    for (int n1 = 0; n1 <= n; n1++) {
      for (int n2 = 0; n2 <= n; n2++) {
        for (int n3 = 0; n3 <= n; n3++) {
          for (int n4 = 0; n4 <= n; n4++) {
            this->k_r0.view_host()(i,n1,n2,n3,n4) = this->r0[i][n1][n2][n3][n4];
            this->k_Delta.view_host()(i,n1,n2,n3,n4) = this->Delta[i][n1][n2][n3][n4];
          }
        }
      }
    }
  }

  this->k_k.template modify<LMPHostType>();
  this->k_r0.template modify<LMPHostType>();
  this->k_Delta.template modify<LMPHostType>();
}

}    // namespace LAMMPS_NS

#endif
#endif