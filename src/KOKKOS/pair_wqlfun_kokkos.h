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

// clang-format off
PairStyle(wqlfun/kk,PairWQLFunKokkos<LMPDeviceType>);
PairStyle(wqlfun/kk/device,PairWQLFunKokkos<LMPDeviceType>);
PairStyle(wqlfun/kk/host,PairWQLFunKokkos<LMPHostType>);
// clang-format on

#else

#ifndef LMP_PAIR_WQLFUN_KOKKOS_H
#define LMP_PAIR_WQLFUN_KOKKOS_H

#include "pair_wqlfun.h"
#include "kokkos_type.h"
#include "pair_kokkos.h"

#include "stringfunction.h"

namespace LAMMPS_NS {

struct TagPairWQLFunNeigh{};
struct TagPairWQLFunEnergy{};
struct TagPairWQLFunEnergy2{};
struct TagPairWQLFunDeriv{};
struct TagPairWQLFunDeriv2{};

template<int NEIGHFLAG, int EVFLAG>
struct TagPairWQLFunForce{};

template<class DeviceType>
class PairWQLFunKokkos : public PairWQLFun {
 public:
  enum {EnabledNeighFlags=FULL|HALF|HALFTHREAD};
  enum {COUL_FLAG=0};
  using complex = SNAComplex<double>;

  typedef Kokkos::View<int*, DeviceType> t_sna_1i;
  typedef Kokkos::View<double*, DeviceType> t_sna_1d;
  typedef Kokkos::View<double*, DeviceType, Kokkos::MemoryTraits<Kokkos::Atomic> > t_sna_1d_atomic;
  typedef Kokkos::View<int**, Kokkos::LayoutRight, DeviceType> t_sna_2i_lr;
  typedef Kokkos::View<int**, Kokkos::LayoutRight, DeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged> > t_sna_2i_lr_um;
  typedef Kokkos::View<int**, DeviceType> t_sna_2i;
  typedef Kokkos::View<double**, DeviceType> t_sna_2d;
  typedef Kokkos::View<double**, Kokkos::LayoutRight, DeviceType> t_sna_2d_lr;
  typedef Kokkos::DualView<double**, Kokkos::LayoutRight, DeviceType> tdual_sna_2d_lr;
  typedef Kokkos::View<double**, Kokkos::LayoutRight, DeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged> > t_sna_2d_lr_um;
  typedef Kokkos::View<double***, DeviceType> t_sna_3d;
  typedef Kokkos::View<double***, Kokkos::LayoutRight, DeviceType> t_sna_3d_lr;
  typedef Kokkos::View<double***, Kokkos::LayoutRight, DeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged> > t_sna_3d_lr_um;
  typedef Kokkos::View<double***[3], DeviceType> t_sna_4d;
  typedef Kokkos::View<double**[3], DeviceType> t_sna_3d3;
  typedef Kokkos::View<double**[3], Kokkos::LayoutRight, DeviceType> t_sna_3d3_lr;
  typedef Kokkos::View<double**[3], Kokkos::LayoutRight, DeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged> > t_sna_3d3_lr_um;
  typedef Kokkos::View<double*****, DeviceType> t_sna_5d;

  typedef Kokkos::View<complex*, DeviceType> t_sna_1c;
  typedef Kokkos::View<complex*, DeviceType, Kokkos::MemoryTraits<Kokkos::Atomic> > t_sna_1c_atomic;
  typedef Kokkos::View<complex**, DeviceType> t_sna_2c;
  typedef Kokkos::View<complex**, Kokkos::LayoutRight, DeviceType> t_sna_2c_lr;
  typedef Kokkos::View<complex***, DeviceType> t_sna_3c;
  typedef Kokkos::View<complex***[3], DeviceType> t_sna_4c;
  typedef Kokkos::View<complex**[3], DeviceType> t_sna_3c3;
  typedef Kokkos::View<complex*****, DeviceType> t_sna_5c;

  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairWQLFunKokkos(class LAMMPS *);
  virtual ~PairWQLFunKokkos();
  void compute(int, int);
  void settings(int, char **);
  void init();
  void init_style();
  double init_one(int, int);


  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairWQLFunNeigh, const typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunNeigh>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairWQLFunEnergy, const typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunEnergy>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairWQLFunEnergy2, const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairWQLFunDeriv, const typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunDeriv>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairWQLFunDeriv2, const typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunDeriv2>::member_type& team) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairWQLFunForce<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairWQLFunForce<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const;

 protected:
  int inum,maxneigh,chunk_size,chunk_offset,nmax;
  int host_flag,neighflag;
  int eflag,vflag;

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;

  t_sna_3d plm;
  t_sna_3c expi;
  t_sna_3c ylm;
  t_sna_3c zlm;
  t_sna_1d wl;
  t_sna_2c Qlm;
  t_sna_1d rsqrt;
  t_sna_1d sqrtfact;
  t_sna_1d Almvec;
  t_sna_4c gradQlm;
  t_sna_3d3 gradql;
  t_sna_3d3 gradwl;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;
  typename ArrayTypes<DeviceType>::t_int_1d mask;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  t_sna_1d d_nne_scale;
  t_sna_1i d_ncount;
  t_sna_2d d_distsq;
  t_sna_2i d_nearest;
  t_sna_3d d_rlist;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int need_dup;
  Kokkos::Experimental::ScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_rho;
  Kokkos::Experimental::ScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_vatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_rho;
  Kokkos::Experimental::ScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_vatom;

  t_sna_1d d_w3jlist;

  template<class TagStyle>
  void check_team_size_for(int, int&, int);

  template<class TagStyle>
  void check_team_size_reduce(int, int&, int);

  template <typename scratch_type>
  int scratch_size_helper(int values_per_team);

  char *expression,**parms;
  double *parm_vals,(*parm_vals2)[2];
  node *parsetree;

  void allocate();

  KOKKOS_INLINE_FUNCTION
  double fsmooth(double r,double &df) const {
    df = 0.0;
    if (r < rmin) return 1.0;
    else if(r >= rmax) return 0.0;
    else {
      const double
        scale = M_PI/(rmax - rmin),
        c = cos((r-rmin)*scale),
        s = sin((r-rmin)*scale);
      df = -0.5*s*scale;
      return 0.5*(c + 1.0);
    }
  }

  inline double fact(int n) {
    double f = 1;
    for (int i = 1; i<=n; i++)
      f = f*i;
    return f;
  }

  inline double Anm(int n,int m) {
    const double x = 1.0 / sqrt(fact(n-m)*fact(n+m));
    if (((n+m) & 1) == 1)
      return -x;
    else
      return x;
  }

  /* Compute all Plm for L<=lmax, 0<=m<=L */
  KOKKOS_INLINE_FUNCTION
  void plmallcompress(const int ii, const int jj, const int lmax, const double x) const;

  /* Compute all Ylm for L<=lmax, 0<=m<=L, using Greegard's normalization.
     The vector (xhat,yhat,zhat) is assumed to be a point on the unit sphere,
     i.e. sqrt(xhat*xhat + yhat*yhat + zhat*zhat) == 1.
  */
  KOKKOS_INLINE_FUNCTION
  void ylmallcompress(const int ii, const int jj, const int lmax,
                      const double xhat, const double yhat, const double zhat) const;

  KOKKOS_INLINE_FUNCTION
  void ylm2zlmcompress(const int ii, const int jj, const int lmax) const;

  KOKKOS_INLINE_FUNCTION
  void zlmderiv1compress(int ii, int jj, const int l, const int m,
                         const double xn, const double yn, const double zn,
                         complex *DY) const;

  KOKKOS_INLINE_FUNCTION
  double eval_tree_hc(double wl) const;

  KOKKOS_INLINE_FUNCTION
  double eval_tree_deriv_hc(double wl) const;

  friend void pair_virial_fdotr_compute<PairWQLFunKokkos>(PairWQLFunKokkos*);

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
