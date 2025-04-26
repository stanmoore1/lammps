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

#ifdef ANGLE_CLASS
// clang-format off
AngleStyle(cosine/kk,AngleCosineKokkos<LMPDeviceType>);
AngleStyle(cosine/kk/device,AngleCosineKokkos<LMPDeviceType>);
AngleStyle(cosine/kk/host,AngleCosineKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_ANGLE_COSINE_KOKKOS_H
#define LMP_ANGLE_COSINE_KOKKOS_H

#include "angle_cosine.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<int NEWTON_BOND, int EVFLAG>
struct TagAngleCosineCompute{};

template<class DeviceType>
class AngleCosineKokkos : public AngleCosine {

 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;
  typedef ArrayTypes<DeviceType> AT;

  AngleCosineKokkos(class LAMMPS *);
  ~AngleCosineKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleCosineCompute<NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagAngleCosineCompute<NEWTON_BOND,EVFLAG>, const int&) const;

  //template<int NEWTON_BOND>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int i, const int j, const int k,
                     double &eangle, double *f1, double *f3,
                     const double &delx1, const double &dely1, const double &delz1,
                     const double &delx2, const double &dely2, const double &delz2) const;

  typename AT::tdual_double_1d k_eatom;
  typename AT::tdual_double_1d_6 k_vatom;

 protected:

  class NeighborKokkos *neighborKK;

  typename ArrayTypes<DeviceType>::t_double_1d_3_randomread x;
  typename ArrayTypes<DeviceType>::t_double_1d_3 f;
  typename ArrayTypes<DeviceType>::t_int_2d anglelist;
  typename ArrayTypes<DeviceType>::t_double_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_double_1d_6 d_vatom;

  int nlocal,newton_bond;
  int eflag,vflag;

  typename ArrayTypes<DeviceType>::tdual_double_1d k_k;
  typename ArrayTypes<DeviceType>::t_double_1d d_k;

  void allocate() override;
};

}

#endif
#endif

