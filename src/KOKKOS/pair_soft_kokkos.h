/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(soft/kk,PairSoftKokkos<LMPDeviceType>)
PairStyle(soft/kk/device,PairSoftKokkos<LMPDeviceType>)
PairStyle(soft/kk/host,PairSoftKokkos<LMPHostType>)

#else

#ifndef LMP_PAIR_SOFT_KOKKOS_H
#define LMP_PAIR_SOFT_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_soft.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairSoftKokkos : public PairSoft {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairSoftKokkos(class LAMMPS *);
  ~PairSoftKokkos();

  void compute(int, int);

  void init_style();
  double init_one(int, int);

  struct params_soft{
    KOKKOS_INLINE_FUNCTION
    params_soft(){cutsq=0;prefactor=0;cutinv=0;};
    KOKKOS_INLINE_FUNCTION
    params_soft(int /*i*/){cutsq=0;prefactor=0;cutinv=0;};
    F_FLOAT cutsq,prefactor,cutinv;
  };

 protected:
  void cleanup_copy() {}

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_ecoul(const F_FLOAT& /*rsq*/, const int& /*i*/, const int& /*j*/,
                        const int& /*itype*/, const int& /*jtype*/) const { return 0; }

  Kokkos::DualView<params_soft**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_soft**,Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  params_soft m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];  // hardwired to space for 12 atom types
  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename AT::t_x_array_randomread x;
  typename AT::t_x_array c_x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;
  typename AT::t_tagint_1d tag;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;


  int neighflag;
  int nlocal,nall,eflag,vflag;

  void allocate();
  friend struct PairComputeFunctor<PairSoftKokkos,FULL,true>;
  friend struct PairComputeFunctor<PairSoftKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairSoftKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairSoftKokkos,FULL,false>;
  friend struct PairComputeFunctor<PairSoftKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairSoftKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairSoftKokkos,FULL,void>(PairSoftKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairSoftKokkos,HALF,void>(PairSoftKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairSoftKokkos,HALFTHREAD,void>(PairSoftKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairSoftKokkos,void>(PairSoftKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairSoftKokkos>(PairSoftKokkos*);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use Kokkos pair style with rRESPA inner/middle

Self-explanatory.

E: Cannot use chosen neighbor list style with soft/kk

That style is not supported by Kokkos.

*/
