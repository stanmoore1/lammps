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

template<int NEWTON_BOND, int EVFLAG>
struct TagBondOxdnaFENECompute{};

template<class DeviceType>
class BondOxdnaFENEKokkos : public BondOxdnaFene {
 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;
  typedef ArrayTypes<DeviceType> AT;

  BondOxdnaFENEKokkos(class LAMMPS *);
  //BondOxdnaFENEKokkos(class LAMMPS *lmp) : Bond(lmp) {}
  ~BondOxdnaFENEKokkos() override;
  virtual void compute_interaction_sites(double *, double *, double *, double *) const;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondOxdnaFENECompute<NEWTON_BOND,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEWTON_BOND, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagBondOxdnaFENECompute<NEWTON_BOND,EVFLAG>, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j, const int &nlocal, const int &newton_bond,\
      const F_FLOAT &ebond, const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,\
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const;

 protected:

  class NeighborKokkos *neighborKK;

  typename ArrayTypes<DeviceType>::t_x_array_randomread x;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_f_array torque;
  typename ArrayTypes<DeviceType>::t_int_2d bondlist;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;

  typename AT::t_int_scalar d_flag;
  HAT::t_int_scalar h_flag;

  int nlocal,newton_bond;
  int eflag,vflag;

  //double *k, *Delta, *r0;                       // FENE
  DAT::tdual_ffloat_1d k_k;
  DAT::tdual_ffloat_1d k_r0;
  DAT::tdual_ffloat_1d k_Delta;
  typename AT::t_ffloat_1d d_Delta;
  typename AT::t_ffloat_1d d_k;
  typename AT::t_ffloat_1d d_r0;

  //double **nx_xtrct, **ny_xtrct, **nz_xtrct;    // per-atom arrays for local unit vectors
  //DAT::tdual_ffloat_2d k_nx_xtrct, k_ny_xtrct, k_nz_xtrct;
  //typename AT::t_ffloat_2d d_nx_xtrct, d_ny_xtrct, d_nz_xtrct;

  void allocate() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
