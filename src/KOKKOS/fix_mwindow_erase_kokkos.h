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
FixStyle(mwindow/erase/kk,FixMWindowEraseKokkos<LMPDeviceType>);
FixStyle(mwindow/erase/kk/device,FixMWindowEraseKokkos<LMPDeviceType>);
FixStyle(mwindow/erase/kk/host,FixMWindowEraseKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_MWINDOW_ERASE_KOKKOS_H
#define LMP_FIX_MWINDOW_ERASE_KOKKOS_H

#include "fix_mwindow_erase.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixMWindowEraseMarkAtoms{};
struct TagFixMWindowEraseCountMarked{};

template<class DeviceType>
class FixMWindowEraseKokkos : public FixMWindowErase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixMWindowEraseKokkos(class LAMMPS *, int, char **);
  ~FixMWindowEraseKokkos() override;
  void pre_exchange() override;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixMWindowEraseMarkAtoms, const int &i) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixMWindowEraseCountMarked, const int &i, int &count) const;

 private:
  typename AT::t_kkfloat_1d_3_lr_randomread d_x;
  typename AT::t_int_1d_randomread d_mask;

  Kokkos::DualView<int*, Kokkos::LayoutRight, DeviceType> k_mark;
  typename Kokkos::DualView<int*, Kokkos::LayoutRight, DeviceType>::t_dev  d_mark;
  typename Kokkos::DualView<int*, Kokkos::LayoutRight, DeviceType>::t_host h_mark;

  double d_erase_pos;
  int    d_erase_dim;
  int    d_erase_side;
};

}    // namespace LAMMPS_NS

#endif
#endif
