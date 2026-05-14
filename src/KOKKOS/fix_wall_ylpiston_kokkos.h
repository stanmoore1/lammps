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
FixStyle(wall/ylpiston/kk,FixWallYLPistonKokkos<LMPDeviceType>);
FixStyle(wall/ylpiston/kk/device,FixWallYLPistonKokkos<LMPDeviceType>);
FixStyle(wall/ylpiston/kk/host,FixWallYLPistonKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_WALL_YLPISTON_KOKKOS_H
#define LMP_FIX_WALL_YLPISTON_KOKKOS_H

#include "fix_wall_ylpiston.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixWallYLPistonForce{};

template<class DeviceType>
class FixWallYLPistonKokkos : public FixWallYLPiston {
 public:

  // Reduction accumulator for wall energy and force components
  struct s_EWALL {
    double w[4];
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    s_EWALL() { w[0] = w[1] = w[2] = w[3] = 0.0; }
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    s_EWALL& operator+=(const s_EWALL &rhs) {
      w[0] += rhs.w[0]; w[1] += rhs.w[1];
      w[2] += rhs.w[2]; w[3] += rhs.w[3];
      return *this;
    }
  };
  typedef s_EWALL EWALL;
  typedef EWALL value_type;

  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixWallYLPistonKokkos(class LAMMPS *, int, char **);
  ~FixWallYLPistonKokkos() override;
  void post_force(int) override;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixWallYLPistonForce, const int &i, EWALL &) const;

 private:
  typename AT::t_kkfloat_1d_3_lr_randomread d_x;
  typename AT::t_kkfloat_1d_3 d_f;
  typename AT::t_int_1d_randomread d_mask;

  double d_coord, d_Edeep3, d_cutoff, d_der;
  int    d_dim, d_side, d_ifix_mw;
};

}    // namespace LAMMPS_NS

#endif
#endif
