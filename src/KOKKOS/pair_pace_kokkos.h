/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pace/kk,PairPACEKokkos<LMPDeviceType>);
PairStyle(pace/kk/device,PairPACEKokkos<LMPDeviceType>);
PairStyle(pace/kk/host,PairPACEKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_PACE_KOKKOS_H
#define LMP_PAIR_PACE_KOKKOS_H

#include "pair_pace.h"
#include "ace_radial.h"
#include "kokkos_type.h"
#include "pair_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairPACEKokkos : public PairPACE {
 public:

  struct TagPairPACEComputeNeigh{};
  struct TagPairPACEComputeRadial{};
  struct TagPairPACEComputeYlm{};
  struct TagPairPACEComputeAi{};
  struct TagPairPACEConjugateAi{};
  struct TagPairPACEComputeWeights{};
  struct TagPairPACEComputeRho{};
  struct TagPairPACEComputeFS;

  template<int NEIGHFLAG, int EVFLAG>
  struct TagPairPACEComputeForce{};

  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;
  using complex = SNAComplex<double>;

  PairPACEKokkos(class LAMMPS *);
  ~PairPACEKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeNeigh,const typename Kokkos::TeamPolicy<DeviceType, TagPairPACEComputeNeigh>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeRadial,const typename Kokkos::TeamPolicy<DeviceType, TagPairPACEComputeRadial>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeYlm,const typename Kokkos::TeamPolicy<DeviceType, TagPairPACEComputeYlm>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeAi,const typename Kokkos::TeamPolicy<DeviceType, TagPairPACEComputeAi>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEConjugateAi,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeRho,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeFS,const int& ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeWeights,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPACEComputeForce<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;

 protected:
  int host_flag;
  int nelements, lmax, nradmax, nradbase;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  int need_dup;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> dup_f;
  DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> dup_vatom;

  NonDupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairPACEKokkos>(PairPACEKokkos*);

  void init();
  void grow(int, int);
  void copy_pertype();
  void copy_splines();
  void copy_tilde();

  typedef Kokkos::View<int*, DeviceType> t_ace_1i;
  typedef Kokkos::View<int**, DeviceType> t_ace_2i;
  typedef Kokkos::View<double*, DeviceType> t_ace_1d;
  typedef Kokkos::View<double**, DeviceType> t_ace_2d;
  typedef Kokkos::View<double*[3], DeviceType> t_ace_2d3;
  typedef Kokkos::View<double***, DeviceType> t_ace_3d;
  typedef Kokkos::View<double**[3], DeviceType> t_ace_3d3;
  typedef Kokkos::View<complex*, DeviceType> t_ace_1c;
  typedef Kokkos::View<complex**, DeviceType> t_ace_2c;
  typedef Kokkos::View<complex***, DeviceType> t_ace_3c;
  typedef Kokkos::View<complex**[3], DeviceType> t_ace_3c3;
  typedef Kokkos::View<complex****, DeviceType> t_ace_4c;
  typedef Kokkos::View<complex***[3], DeviceType> t_ace_4c3;

  t_ace_3c A;
  t_ace_3d A_rank1;

  t_ace_2c A_list;
  t_ace_2c A_forward_prod;
  t_ace_2c A_backward_prod;

  t_ace_2c weights;
  t_ace_2d weights_rank1;

  t_ace_1d rhos;
  t_ace_1d dF_drho;

  // hard-core repulsion
  t_ace_1d rho_core;
  t_ace_2c dB_flatten;
  t_ace_2d cr;
  t_ace_2d dcr;

  // radial functions
  t_ace_3d fr;
  t_ace_3d dfr;
  t_ace_3d dgr;

  // Spherical Harmonics

  void pre_compute_harmonics();

  KOKKOS_INLINE_FUNCTION
  void compute_barplm(double rz, int lmaxi);

  KOKKOS_INLINE_FUNCTION
  void compute_ylm(double rx, double ry, double rz, int lmaxi);

  t_ace_1d alm;
  t_ace_1d blm;
  t_ace_1d cl;
  t_ace_1d dl;

  t_ace_3d plm;
  t_ace_3d dplm;

  t_ace_3c ylm;
  t_ace_3c3 dylm;

  // short neigh list
  t_ace_3d3 d_rlist;
  t_ace_2d d_distsq;
  t_ace_2i d_nearest;

  // per-type
  t_ace_1i d_total_basis_size_rank1 = t_ace_1i("total_basis_size_rank1", nelements);
  t_ace_1i d_total_basis_size;
  t_ace_1i d_ndensity;
  t_ace_1i d_npoti;
  t_ace_1d d_rho_core_cutoff;
  t_ace_1d d_drho_core_cutoff;
  t_ace_1d d_E0vals;
  t_ace_2d d_wpre;
  t_ace_2d d_mexp;

  // tilde
  t_ace_2d d_rank;
  t_ace_3d d_mus;
  t_ace_3d d_ns;
  t_ace_3d d_ls;
  t_ace_3d d_ms;

  class SplineInterpolatorKokkos {
   public:
    int ntot, nlut, num_of_functions;
    double cutoff, deltaSplineBins, invrscalelookup, rscalelookup;

    t_ace_1d values, derivatives, second_derivatives;

    typedef Kokkos::View<double**[4], DeviceType> t_ace_3d4;
    t_ace_3d4 lookupTable;

    void operator=(const SplineInterpolator &spline) {
      cutoff = spline.cutoff;
      deltaSplineBins = spline.deltaSplineBins;
      ntot = spline.ntot;
      nlut = spline.nlut;
      invrscalelookup = spline.invrscalelookup;
      rscalelookup = spline.rscalelookup;
      num_of_functions = spline.num_of_functions;

      values = t_ace_1d("values", num_of_functions);
      derivatives = t_ace_1d("derivatives", num_of_functions);
      second_derivatives = t_ace_1d("second_derivatives", num_of_functions);

      lookupTable = t_ace_3d4("lookupTable", ntot+1, num_of_functions);
      auto h_lookupTable = Kokkos::create_mirror_view(lookupTable);
      for (int i = 0; i < ntot+1; i++)
        for (int j = 0; j < num_of_functions; j++)
          for (int k = 0; k < 4; k++)
            h_lookupTable(i, j, k) = spline.lookupTable(i, j, k);
      Kokkos::deep_copy(lookupTable, h_lookupTable);
    }

    void deallocate() {
      values = t_ace_1d();
      derivatives = t_ace_1d();
      second_derivatives = t_ace_1d();
      lookupTable = t_ace_3d4();
    }
  };

  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_gk;
  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_rnl;
  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_hc;

};
}    // namespace LAMMPS_NS

#endif
#endif
