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
BondStyle(oxdna/fene/kk,BondOxdnaFENEKokkos<LMPDeviceType>);
BondStyle(oxdna/fene/kk/device,BondOxdnaFENEKokkos<LMPDeviceType>);
BondStyle(oxdna/fene/kk/host,BondOxdnaFENEKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_BOND_OXDNA_FENE_KOKKOS_H
#define LMP_BOND_OXDNA_FENE_KOKKOS_H

#include "bond_oxdna_fene.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagBondOxdnaFENEPrecomputeClosestBond{};

template<int OXDNAFLAG, int NEWTON_BOND, int EVFLAG>
struct TagBondOxdnaFENECompute{};

template<class DeviceType>
class BondOxdnaFENEKokkos : public BondOxdnaFene {
 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;
  typedef ArrayTypes<DeviceType> AT;

  BondOxdnaFENEKokkos(class LAMMPS *);
  ~BondOxdnaFENEKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondOxdnaFENEPrecomputeClosestBond, const int&) const;

  template<int OXDNAFLAG, int NEWTON_BOND, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondOxdnaFENECompute<OXDNAFLAG,NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int OXDNAFLAG, int NEWTON_BOND, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondOxdnaFENECompute<OXDNAFLAG,NEWTON_BOND,EVFLAG>, const int&) const;

// NOLINTNEXTLINE
   KOKKOS_INLINE_FUNCTION
   void ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j, const int &nlocal, const int &newton_bond,\
      const KK_FLOAT &ebond, const KK_FLOAT &fx, const KK_FLOAT &fy, const KK_FLOAT &fz,\
      const KK_FLOAT &delx, const KK_FLOAT &dely, const KK_FLOAT &delz) const;

 protected:
  
  int oxdnaflag;
  enum EnabledOXDNAFlag{OXDNA=1,OXDNA2=2,OXRNA2=4};
  
  class NeighborKokkos *neighborKK;

  typename ArrayTypes<DeviceType>::t_kkfloat_1d_3_lr x;
  typename Kokkos::View<KK_FLOAT*[3],typename AT::t_kkfloat_1d_3::array_layout,\
     typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic> > f;
  typename Kokkos::View<KK_FLOAT*[3],typename AT::t_kkfloat_1d_3::array_layout,\
     typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic> > torque;
  typename ArrayTypes<DeviceType>::t_int_2d_lr bondlist;
  typename AT::t_int_1d_randomread atomtype;
  typename ArrayTypes<DeviceType>::t_tagint_1d tag;
  typename ArrayTypes<DeviceType>::t_tagint_1d id5p;
  typename ArrayTypes<DeviceType>::t_tagint_1d id3p;

  typedef typename KKDevice<DeviceType>::value KKDeviceType;
  TransformView<KK_FLOAT*,double*,Kokkos::LayoutRight,KKDeviceType> k_eatom;
  TransformView<KK_FLOAT*[6],double*[6],LMPDeviceLayout,KKDeviceType> k_vatom;
  Kokkos::View<KK_FLOAT*,Kokkos::LayoutRight,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_eatom;
  Kokkos::View<KK_FLOAT*[6],LMPDeviceLayout,KKDeviceType,Kokkos::MemoryTraits<Kokkos::Atomic> > d_vatom;

  typename AT::t_int_scalar d_flag;
  HAT::t_int_scalar h_flag;

  int nlocal,newton_bond;
  int eflag,vflag;

  DAT::ttransform_kkfloat_1d k_k;
  DAT::ttransform_kkfloat_5d k_r0;
  DAT::ttransform_kkfloat_5d k_Delta;
  typename AT::t_kkfloat_1d d_k;
  typename AT::t_kkfloat_5d d_r0;
  typename AT::t_kkfloat_5d d_Delta;
  // per-atom arrays for local unit vectors
  DAT::tdual_kkfloat_1d_3 k_nx_xtrct, k_ny_xtrct, k_nz_xtrct;
  typename AT::t_kkfloat_1d_3_lr d_nx_xtrct, d_ny_xtrct, d_nz_xtrct;

  void allocate() override;

  // Atom Mapping
  int map_style;
  DAT::tdual_int_1d k_map_array;
  dual_hash_type k_map_hash;
  DAT::tdual_int_1d k_sametag;
  typename AT::t_int_1d d_sametag;
  // Precomputed closest images for bondlist atoms
  // 0-3 : closest images of atom a, atom b, id3p[a], id5p[b] for each bond
  DAT::tdual_int_2d k_closest_bond;
  typename AT::t_int_2d d_closest_bond;

  // NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  int closest_image(const int, int) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
