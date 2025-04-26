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

#ifdef DIHEDRAL_CLASS
// clang-format off
DihedralStyle(charmmfsw/kk,DihedralCharmmfswKokkos<LMPDeviceType>);
DihedralStyle(charmmfsw/kk/device,DihedralCharmmfswKokkos<LMPDeviceType>);
DihedralStyle(charmmfsw/kk/host,DihedralCharmmfswKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_DIHEDRAL_CHARMMFSW_KOKKOS_H
#define LMP_DIHEDRAL_CHARMMFSW_KOKKOS_H

#include "dihedral_charmmfsw.h"
#include "kokkos_type.h"
#include "dihedral_charmm_kokkos.h" // needed for s_EVM_FLOAT

namespace LAMMPS_NS {

template<int NEWTON_BOND, int EVFLAG>
struct TagDihedralCharmmfswCompute{};

template<class DeviceType>
class DihedralCharmmfswKokkos : public DihedralCharmmfsw {
 public:
  typedef DeviceType device_type;
  typedef EVM_FLOAT value_type;
  typedef ArrayTypes<DeviceType> AT;

  DihedralCharmmfswKokkos(class LAMMPS *);
  ~DihedralCharmmfswKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  void read_restart(FILE *) override;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagDihedralCharmmfswCompute<NEWTON_BOND,EVFLAG>, const int&, EVM_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagDihedralCharmmfswCompute<NEWTON_BOND,EVFLAG>, const int&) const;

  //template<int NEWTON_BOND>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EVM_FLOAT &evm, const int i1, const int i2, const int i3, const int i4,
                          double &edihedral, double *f1, double *f3, double *f4,
                          const double &vb1x, const double &vb1y, const double &vb1z,
                          const double &vb2x, const double &vb2y, const double &vb2z,
                          const double &vb3x, const double &vb3y, const double &vb3z) const;

  KOKKOS_INLINE_FUNCTION
  void ev_tally(EVM_FLOAT &evm, const int i, const int j,
        const double &evdwl, const double &ecoul, const double &fpair, const double &delx,
                const double &dely, const double &delz) const;

  typedef typename KKDevice<DeviceType>::value KKDeviceType;
  Kokkos::DualView<double*,Kokkos::LayoutRight,KKDeviceType> k_eatom;
  Kokkos::DualView<double*[6],Kokkos::LayoutRight,KKDeviceType> k_vatom;

 protected:

  class NeighborKokkos *neighborKK;

  typename AT::t_double_1d_3_randomread x;
  typename AT::t_int_1d_randomread atomtype;
  typename AT::t_double_1d_randomread q;
  typename AT::t_double_1d_3 f;
  typename AT::t_int_2d dihedrallist;
  Kokkos::View<double*,Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_eatom;
  Kokkos::View<double*[6],Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_vatom;

  Kokkos::DualView<double*,Kokkos::LayoutRight,KKDeviceType> k_eatom_pair;
  Kokkos::DualView<double*[6],Kokkos::LayoutRight,KKDeviceType> k_vatom_pair;
  Kokkos::View<double*,Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_eatom_pair;
  Kokkos::View<double*[6],Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_vatom_pair;

  int nlocal,newton_bond;
  int eflag,vflag;
  double qqrd2e;

  Kokkos::DualView<int,DeviceType> k_warning_flag;
  typename Kokkos::DualView<int,DeviceType>::t_dev d_warning_flag;
  typename Kokkos::DualView<int,DeviceType>::t_host h_warning_flag;

  typename AT::t_double_2d d_lj14_1;
  typename AT::t_double_2d d_lj14_2;
  typename AT::t_double_2d d_lj14_3;
  typename AT::t_double_2d d_lj14_4;

  typename AT::t_double_1d d_k;
  typename AT::t_double_1d d_multiplicity;
  typename AT::t_double_1d d_shift;
  typename AT::t_double_1d d_sin_shift;
  typename AT::t_double_1d d_cos_shift;
  typename AT::t_double_1d d_weight;

  void allocate() override;
};

}

#endif
#endif

