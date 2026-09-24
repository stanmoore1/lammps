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

template<class DeviceType>
class FixOxdnaLRFKokkos;  // forward declaration
template<class DeviceType>
class FixOxdnaPrimeNeighsKokkos;  // forward declaration

template<int OXDNAFLAG, int NEWTON_BOND, int EVFLAG>
struct TagBondOxdnaFENECompute{};

// KOKKOS implementation of the oxDNA/oxRNA FENE bond styles.  The class is
// templated on its CPU parent class (BondOxdnaFene, BondOxdna2Fene, BondOxdna3Fene,
// or BondOxrna2Fene), so that settings, coeff(), and restart parsing are always
// inherited from the matching CPU style, while the device kernels are shared and
// select the model specific code via the oxdnaflag template parameter.

template<class DeviceType, class BondBase>
class BondOxdnaFENEKokkosT : public BondBase {
 public:
  typedef DeviceType device_type;
  typedef EV_FLOAT value_type;
  typedef ArrayTypes<DeviceType> AT;

  enum EnabledOXDNAFlag { OXDNA = 1, OXDNA2 = 2, OXRNA2 = 4 };

  BondOxdnaFENEKokkosT(class LAMMPS *, int oxdnaflag_in);
  ~BondOxdnaFENEKokkosT() override;
  void init_style() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void read_restart(FILE *) override;

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
      const KK_FLOAT &ebond, const KK_ACC_FLOAT &fx, const KK_ACC_FLOAT &fy, const KK_ACC_FLOAT &fz,\
      const KK_FLOAT &delx, const KK_FLOAT &dely, const KK_FLOAT &delz) const;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;

 protected:

  int oxdnaflag;

  // members of the (dependent) CPU base class used in this class
  using BondBase::atom;
  using BondBase::atomKK;
  using BondBase::error;
  using BondBase::force;
  using BondBase::memoryKK;
  using BondBase::modify;
  using BondBase::neighbor;
  using BondBase::update;
  using BondBase::copymode;
  using BondBase::kokkosable;
  using BondBase::execution_space;
  using BondBase::datamask_read;
  using BondBase::datamask_modify;
  using BondBase::ev_init;
  using BondBase::evflag;
  using BondBase::eflag_either;
  using BondBase::eflag_global;
  using BondBase::eflag_atom;
  using BondBase::vflag_either;
  using BondBase::vflag_global;
  using BondBase::vflag_atom;
  using BondBase::energy;
  using BondBase::virial;
  using BondBase::eatom;
  using BondBase::vatom;
  using BondBase::maxeatom;
  using BondBase::maxvatom;
  using BondBase::k;
  using BondBase::r0;
  using BondBase::Delta;
  using BondBase::allocated;
  using BondBase::setflag;

  class NeighborKokkos *neighborKK;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkacc_1d_3 torque;
  typename AT::t_int_2d_lr bondlist;
  typename AT::t_int_1d_randomread atomtype;
  typename AT::t_tagint_1d tag;
  typename AT::t_tagint_1d id5p;
  typename AT::t_tagint_1d id3p;

  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  typename AT::t_int_scalar d_flag;
  HAT::t_int_scalar h_flag;

  int nbondlist;
  int nlocal,newton_bond;
  int eflag,vflag;

  DAT::tdual_kkfloat_1d k_k;
  DAT::tdual_kkfloat_5d k_r0;
  DAT::tdual_kkfloat_5d k_Delta;
  typename AT::t_kkfloat_1d_randomread d_k;
  typename AT::t_kkfloat_5d_randomread d_r0;
  typename AT::t_kkfloat_5d_randomread d_Delta;
  // per-atom arrays for local unit vectors
  DAT::tdual_kkfloat_1d_3 k_nx_xtrct, k_ny_xtrct, k_nz_xtrct;
  typename AT::t_kkfloat_1d_3_randomread d_nx_xtrct, d_ny_xtrct, d_nz_xtrct;

  void allocate() override;

  FixOxdnaLRFKokkos<DeviceType> *fix_oxdna_lrfKK;    // ptr to OXDNA/LRF/kk fix
  FixOxdnaPrimeNeighsKokkos<DeviceType> *fix_oxdna_prime_neighsKK;    // ptr to OXDNA/PRIME_NEIGHS/kk fix

  // Precomputed atom a/b 3'/5' directionality and atom mapping of their 3' and 5' neighbors.
  // 0-3 : atom a, atom b, id3p[a], id5p[b] for each bond.
  bigint last_prime_neighs_bond_ncalls;
  typename AT::t_int_1d_4 d_prime_neighs_bond_own;
  typename AT::t_int_1d_4_randomread d_prime_neighs_bond; // single device-space View suffices
};

template<class DeviceType>
class BondOxdnaFENEKokkos : public BondOxdnaFENEKokkosT<DeviceType, BondOxdnaFene> {
 public:
  BondOxdnaFENEKokkos(class LAMMPS *lmp) :
      BondOxdnaFENEKokkosT<DeviceType, BondOxdnaFene>(lmp,
          BondOxdnaFENEKokkosT<DeviceType, BondOxdnaFene>::OXDNA) {}
};

}    // namespace LAMMPS_NS

#endif
#endif
