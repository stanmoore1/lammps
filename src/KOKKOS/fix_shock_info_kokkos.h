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

#ifdef FIX_CLASS
// clang-format off
FixStyle(shock/info/kk,FixShockInfoKokkos<LMPDeviceType>);
FixStyle(shock/info/kk/device,FixShockInfoKokkos<LMPDeviceType>);
FixStyle(shock/info/kk/host,FixShockInfoKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_SHOCK_INFO_KOKKOS_H
#define LMP_FIX_SHOCK_INFO_KOKKOS_H

#include "fix_shock_info.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixShockInfoAtomLoop{};

template<class DeviceType>
class FixShockInfoKokkos : public FixShockInfo {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixShockInfoKokkos(class LAMMPS *, int, char **);
  ~FixShockInfoKokkos() override;
  void init() override;
  void end_of_step() override;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixShockInfoAtomLoop, const int &i) const;

 private:
  // Per-call device views; assigned in end_of_step before the kernel
  typename AT::t_kkfloat_1d_3_lr_randomread d_x;
  typename AT::t_kkfloat_1d_3_randomread d_v;
  typename AT::t_kkfloat_1d_randomread d_rmass;
  typename AT::t_kkfloat_1d_randomread d_mass;
  typename AT::t_int_1d_randomread d_type;
  typename AT::t_int_1d_randomread d_mask;

  // Temporary per-atom compute results (reassigned each call)
  Kokkos::View<double*,  Kokkos::LayoutRight, DeviceType> d_pe_atom;
  Kokkos::View<double**, Kokkos::LayoutRight, DeviceType> d_stress_atom;

  // Per-layer accumulators (resized when nlayers grows)
  Kokkos::View<double*,  Kokkos::LayoutRight, DeviceType> d_count_kk;
  Kokkos::View<double**, Kokkos::LayoutRight, DeviceType> d_values_kk;
  int maxlayer_kk;

  // Cached scalars for use inside the Kokkos kernel
  double d_offset, d_invdelta, d_mvv2e;
  int    d_dim, d_nlayers_kk, d_cpnts_all_kk, d_stress_size_kk;
  int    d_has_rmass;
};

}    // namespace LAMMPS_NS

#endif
#endif
