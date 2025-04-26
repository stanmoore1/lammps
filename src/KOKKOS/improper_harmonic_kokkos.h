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

#ifdef IMPROPER_CLASS
// clang-format off
ImproperStyle(harmonic/kk,ImproperHarmonicKokkos<LMPDeviceType>);
ImproperStyle(harmonic/kk/device,ImproperHarmonicKokkos<LMPDeviceType>);
ImproperStyle(harmonic/kk/host,ImproperHarmonicKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_IMPROPER_HARMONIC_KOKKOS_H
#define LMP_IMPROPER_HARMONIC_KOKKOS_H

#include "improper_harmonic.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<int NEWTON_BOND, int EVFLAG>
struct TagImproperHarmonicCompute{};

template<class DeviceType>
class ImproperHarmonicKokkos : public ImproperHarmonic {
 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;
  typedef ArrayTypes<DeviceType> AT;

  ImproperHarmonicKokkos(class LAMMPS *);
  ~ImproperHarmonicKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagImproperHarmonicCompute<NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagImproperHarmonicCompute<NEWTON_BOND,EVFLAG>, const int&) const;

  //template<int NEWTON_BOND>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int i1, const int i2, const int i3, const int i4,
                          double &eimproper, double *f1, double *f3, double *f4,
                          const double &vb1x, const double &vb1y, const double &vb1z,
                          const double &vb2x, const double &vb2y, const double &vb2z,
                          const double &vb3x, const double &vb3y, const double &vb3z) const;

  typedef typename KKDevice<DeviceType>::value KKDeviceType;
  Kokkos::DualView<double*,Kokkos::LayoutRight,KKDeviceType> k_eatom;
  Kokkos::DualView<double*[6],Kokkos::LayoutRight,KKDeviceType> k_vatom;

 protected:

  class NeighborKokkos *neighborKK;

  typename AT::t_double_1d_3_randomread x;
  typename Kokkos::View<double*[3],typename AT::t_double_1d_3::array_layout,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > f;
  typename AT::t_int_2d improperlist;
  Kokkos::View<double*,Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_eatom;
  Kokkos::View<double*[6],Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_vatom;

  int nlocal,newton_bond;
  int eflag,vflag;

  Kokkos::DualView<int,DeviceType> k_warning_flag;
  typename Kokkos::DualView<int,DeviceType>::t_dev d_warning_flag;
  typename Kokkos::DualView<int,DeviceType>::t_host h_warning_flag;

  Kokkos::DualView<double*,DeviceType> k_k;
  Kokkos::DualView<double*,DeviceType> k_chi;

  typename Kokkos::DualView<double*,DeviceType>::t_dev d_k;
  typename Kokkos::DualView<double*,DeviceType>::t_dev d_chi;

  void allocate() override;
};

}

#endif
#endif

