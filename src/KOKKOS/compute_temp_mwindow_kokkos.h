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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(temp/mwindow/kk,ComputeTempMWindowKokkos<LMPDeviceType>);
ComputeStyle(temp/mwindow/kk/device,ComputeTempMWindowKokkos<LMPDeviceType>);
ComputeStyle(temp/mwindow/kk/host,ComputeTempMWindowKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_COMPUTE_TEMP_MWINDOW_KOKKOS_H
#define LMP_COMPUTE_TEMP_MWINDOW_KOKKOS_H

#include "compute.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<int RMASS>
struct TagComputeTempMWindowScalar{};

template<int RMASS>
struct TagComputeTempMWindowVector{};

struct TagComputeTempMWindowRemoveBias{};
struct TagComputeTempMWindowRestoreBias{};

template<class DeviceType>
class ComputeTempMWindowKokkos : public Compute {
 public:
  struct s_CTEMP {
    double t0, t1, t2, t3, t4, t5;
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    s_CTEMP() { t0 = t1 = t2 = t3 = t4 = t5 = 0.0; }
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    s_CTEMP& operator+=(const s_CTEMP &rhs) {
      t0 += rhs.t0; t1 += rhs.t1; t2 += rhs.t2;
      t3 += rhs.t3; t4 += rhs.t4; t5 += rhs.t5;
      return *this;
    }
  };

  typedef s_CTEMP CTEMP;
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef CTEMP value_type;

  ComputeTempMWindowKokkos(class LAMMPS *, int, char **);
  ~ComputeTempMWindowKokkos() override;
  void init() override;
  double compute_scalar() override;
  void compute_vector() override;
  void remove_bias(int, double *) override;
  void remove_bias_all() override;
  void restore_bias(int, double *) override;
  void restore_bias_all() override;

  template<int RMASS>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeTempMWindowScalar<RMASS>, const int&, CTEMP&) const;

  template<int RMASS>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeTempMWindowVector<RMASS>, const int&, CTEMP&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeTempMWindowRemoveBias, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeTempMWindowRestoreBias, const int&) const;

 protected:
  int fix_dof;
  double tfactor, masstotal;
  double vbias[3];

  void dof_compute();

  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_kkfloat_1d_randomread rmass;
  typename AT::t_kkfloat_1d_randomread mass;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_1d_randomread mask;
};

}    // namespace LAMMPS_NS

#endif
#endif
