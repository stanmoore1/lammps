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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(oxdna/xstk/kk,PairOxdnaXstkKokkos<LMPDeviceType>);
PairStyle(oxdna/xstk/kk/device,PairOxdnaXstkKokkos<LMPDeviceType>);
PairStyle(oxdna/xstk/kk/host,PairOxdnaXstkKokkos<LMPHostType>);
PairStyle(oxdna2/xstk/kk,PairOxdnaXstkKokkos<LMPDeviceType>);
PairStyle(oxdna2/xstk/kk/device,PairOxdnaXstkKokkos<LMPDeviceType>);
PairStyle(oxdna2/xstk/kk/host,PairOxdnaXstkKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_OXDNA_XSTK_KOKKOS_H
#define LMP_PAIR_OXDNA_XSTK_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "pair_oxdna_xstk.h"
#include "neigh_list_kokkos.h"
#include "oxdna_hbxstk_fused.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixOxdnaLRFKokkos;  // forward declaration

template<class DeviceType>
class FixOxdnaNpairKokkos;  // forward declaration

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairOxdnaXstkCompute{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairOxdnaXstkComputeGPUPair{};

template<class DeviceType>
class PairOxdnaXstkKokkos : public PairOxdnaXstk, public KokkosBase {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairOxdnaXstkKokkos(class LAMMPS *);
  ~PairOxdnaXstkKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  // Prototype hbond+xstk fusion: export the cross-stacking coefficient view
  // handles so the hbond style can evaluate the xstk term in a fused kernel.
  void export_fused_coeffs(OxdnaXstkCoeffs<DeviceType> &out) const;

  // Pure (argument-only) force/torque accumulators, reused by the fused kernel.
  // Declared public so the hbond style can call them on the cross-stacking term.
  // Standard non-GPU Compute Functor(s). 1 with EV_FLOAT, 1 without.

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaXstkCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaXstkCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&) const;

  // GPU ComputeGPUPair Functor(s). 1 with EV_FLOAT, 1 without.

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaXstkComputeGPUPair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdnaXstkComputeGPUPair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const KK_FLOAT &epair, const KK_FLOAT &fx, const KK_FLOAT &fy, const KK_FLOAT &fz, const KK_FLOAT &delx,
                  const KK_FLOAT &dely, const KK_FLOAT &delz) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

 protected:

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkfloat_1d_3 f;
  typename AT::t_kkfloat_1d_3 torque;
  typename AT::t_int_1d_randomread type;

  DAT::ttransform_kkfloat_1d k_eatom;
  DAT::ttransform_kkfloat_1d_6 k_vatom;
  typename AT::t_kkfloat_1d d_eatom;
  typename AT::t_kkfloat_1d_6 d_vatom;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_kkfloat_2d k_cutsq;
  typename AT::t_kkfloat_2d d_cutsq;

  int neighflag;
  int nlocal, eflag, vflag;
  int anum;

  typename AT::t_neighbors_2d_randomread d_neighbors;
  typename AT::t_int_1d_randomread d_alist;
  typename AT::t_int_1d_randomread d_numneigh;
  // Screening takes place on GPUs only
  // These are taken from the generic fix_oxdna_npairKK
  DAT::tdual_uint64_1d k_pairs_screened;
  typename AT::t_uint64_1d d_pairs_screened;
  int screened_pair_count;

  // cross-stacking interaction parameters
  typename AT::tdual_kkfloat_2d k_k_xst, k_cut_xst_0, k_cut_xst_c;
  typename AT::tdual_kkfloat_2d k_cut_xst_lo, k_cut_xst_hi;
  typename AT::tdual_kkfloat_2d k_cut_xst_lc, k_cut_xst_hc, k_b_xst_lo, k_b_xst_hi;
  typename AT::tdual_kkfloat_2d k_cutsq_xst_hc;
  typename AT::tdual_kkfloat_2d k_a_xst1, k_theta_xst1_0, k_dtheta_xst1_ast;
  typename AT::tdual_kkfloat_2d k_b_xst1, k_dtheta_xst1_c;
  typename AT::tdual_kkfloat_2d k_a_xst2, k_theta_xst2_0, k_dtheta_xst2_ast;
  typename AT::tdual_kkfloat_2d k_b_xst2, k_dtheta_xst2_c;
  typename AT::tdual_kkfloat_2d k_a_xst3, k_theta_xst3_0, k_dtheta_xst3_ast;
  typename AT::tdual_kkfloat_2d k_b_xst3, k_dtheta_xst3_c;
  typename AT::tdual_kkfloat_2d k_a_xst4, k_theta_xst4_0, k_dtheta_xst4_ast;
  typename AT::tdual_kkfloat_2d k_b_xst4, k_dtheta_xst4_c;
  typename AT::tdual_kkfloat_2d k_a_xst7, k_theta_xst7_0, k_dtheta_xst7_ast;
  typename AT::tdual_kkfloat_2d k_b_xst7, k_dtheta_xst7_c;
  typename AT::tdual_kkfloat_2d k_a_xst8, k_theta_xst8_0, k_dtheta_xst8_ast;
  typename AT::tdual_kkfloat_2d k_b_xst8, k_dtheta_xst8_c;
  typename AT::t_kkfloat_2d_randomread d_k_xst, d_cut_xst_0, d_cut_xst_c;
  typename AT::t_kkfloat_2d_randomread d_cut_xst_lo, d_cut_xst_hi;
  typename AT::t_kkfloat_2d_randomread d_cut_xst_lc, d_cut_xst_hc, d_b_xst_lo, d_b_xst_hi;
  typename AT::t_kkfloat_2d_randomread d_cutsq_xst_hc;
  typename AT::t_kkfloat_2d_randomread d_a_xst1, d_theta_xst1_0, d_dtheta_xst1_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst1, d_dtheta_xst1_c;
  typename AT::t_kkfloat_2d_randomread d_a_xst2, d_theta_xst2_0, d_dtheta_xst2_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst2, d_dtheta_xst2_c;
  typename AT::t_kkfloat_2d_randomread d_a_xst3, d_theta_xst3_0, d_dtheta_xst3_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst3, d_dtheta_xst3_c;
  typename AT::t_kkfloat_2d_randomread d_a_xst4, d_theta_xst4_0, d_dtheta_xst4_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst4, d_dtheta_xst4_c;
  typename AT::t_kkfloat_2d_randomread d_a_xst7, d_theta_xst7_0, d_dtheta_xst7_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst7, d_dtheta_xst7_c;
  typename AT::t_kkfloat_2d_randomread d_a_xst8, d_theta_xst8_0, d_dtheta_xst8_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst8, d_dtheta_xst8_c;
  // per-atom arrays for local unit vectors
  DAT::tdual_kkfloat_1d_3_lr k_nx_xtrct, k_ny_xtrct, k_nz_xtrct;
  typename AT::t_kkfloat_1d_3_lr_randomread d_nx_xtrct, d_ny_xtrct, d_nz_xtrct;

  int first;
  typename AT::t_int_1d d_sendlist;
  typename AT::t_double_1d_um v_buf;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, \
  KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, \
  KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<KK_FLOAT*[3], typename AT::t_kkfloat_1d_3::array_layout> dup_f;
  DupScatterView<KK_FLOAT*[3], typename AT::t_kkfloat_1d_3::array_layout> dup_torque;
  DupScatterView<KK_FLOAT*, typename DAT::t_kkfloat_1d::array_layout> dup_eatom;
  DupScatterView<KK_FLOAT*[6], typename DAT::t_kkfloat_1d_6::array_layout> dup_vatom;
  NonDupScatterView<KK_FLOAT*[3], typename AT::t_kkfloat_1d_3::array_layout> ndup_f;
  NonDupScatterView<KK_FLOAT*[3], typename AT::t_kkfloat_1d_3::array_layout> ndup_torque;
  NonDupScatterView<KK_FLOAT*, typename DAT::t_kkfloat_1d::array_layout> ndup_eatom;
  NonDupScatterView<KK_FLOAT*[6], typename DAT::t_kkfloat_1d_6::array_layout> ndup_vatom;

  void allocate() override;
 
  friend void pair_virial_fdotr_compute<PairOxdnaXstkKokkos>(PairOxdnaXstkKokkos*);

  FixOxdnaLRFKokkos<DeviceType> *fix_oxdna_lrfKK;    // ptr to oxdna/lrf/kk fix
  FixOxdnaNpairKokkos<DeviceType> *fix_oxdna_npairKK;    // ptr to oxdna/pair/kk fix
  class Pair *fused_hbondKK = nullptr;    // hbond style (fuses xstk) if present

 public:
  // Fused-kernel stash: when the hbond style runs the fused hbond+xstk kernel it
  // writes this style's split global energy/virial here (set before this style's
  // compute() runs); consumed after ev_init() zeroes eng_vdwl/virial.
  double fused_eng_vdwl = 0.0;
  double fused_virial[6] = {0,0,0,0,0,0};

 private:

// The following is totally wild code-readability wise and I don't really like.
// But I was getting a ton of Live Register Pressure and the only way I could
// reduce this was to pull everything out into separate:
// PairOxdnaXstkKokkos<DeviceType>::xstk_* KOKKOS_INLINE_FUNCTIONs.
// The compilers (HIP and CUDA) wouldn't kill off short-lived vars otherwise,
// which really bumped up register usage and bumped up runtime. Simple INLINES
// didn't help.
//
// Compute-wise, it would be nice to calc the derivatives only when they are needed
// after the evdwl. But then I need to have my p_* terms again and the register pressure
// blows through occupancy and runtime goes up again. In these closed-scope areas,
// I've so found it best to just take the FP hit and calc the derivs even if they
// end up not being needed. So far, this is the fastest option.

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_radial_terms(const int &atype, const int &btype, const KK_FLOAT &r_hb,
    KK_FLOAT &f2, KK_FLOAT &df2) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta1_terms(const int &atype, const int &btype,
    const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&b_nx)[3],
    KK_FLOAT &theta1, KK_FLOAT &f4t1, KK_FLOAT &df4t1) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta2_terms(const int &atype, const int &btype,
    const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT &theta2, KK_FLOAT &cost2, KK_FLOAT &f4t2, KK_FLOAT &df4t2) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta3_terms(const int &atype, const int &btype,
    const KK_FLOAT (&b_nx)[3], const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT &theta3, KK_FLOAT &cost3, KK_FLOAT &f4t3, KK_FLOAT &df4t3) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta4_terms(const int &atype, const int &btype,
    const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&b_nz)[3],
    KK_FLOAT &theta4, KK_FLOAT &theta4p, KK_FLOAT &f4t4, KK_FLOAT &df4t4) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta7_terms(const int &atype, const int &btype,
    const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT &theta7, KK_FLOAT &cost7, KK_FLOAT &f4t7, KK_FLOAT &df4t7) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta8_terms(const int &atype, const int &btype,
    const KK_FLOAT (&b_nz)[3], const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT &theta8, KK_FLOAT &cost8, KK_FLOAT &f4t8, KK_FLOAT &df4t8) const;

 public:
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  static void xstk_force_contrib(const KK_FLOAT &f2, const KK_FLOAT &f4t1, const KK_FLOAT &f4t2,
    const KK_FLOAT &f4t3, const KK_FLOAT &f4t4, const KK_FLOAT &f4t7, const KK_FLOAT &f4t8,
    const KK_FLOAT &df2, const KK_FLOAT &df4t2, const KK_FLOAT &df4t3, const KK_FLOAT &df4t7,
    const KK_FLOAT &df4t8, const KK_FLOAT &rinv_hb, const KK_FLOAT &factor_lj,
    const KK_FLOAT &theta2, const KK_FLOAT &theta3, const KK_FLOAT &theta7, const KK_FLOAT &theta8,
    const KK_FLOAT &cost2, const KK_FLOAT &cost3, const KK_FLOAT &cost7, const KK_FLOAT &cost8,
    const KK_FLOAT (&delr_hb)[3], const KK_FLOAT (&delr_hb_norm)[3],
    const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&b_nx)[3],
    const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&b_nz)[3],
    const KK_FLOAT (&ra_chb)[3], const KK_FLOAT (&rb_chb)[3],
    KK_FLOAT (&delf)[3], KK_FLOAT (&delta)[3], KK_FLOAT (&deltb)[3]);

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  static void xstk_torque_contrib(const KK_FLOAT &f2,
    const KK_FLOAT &f4t1, const KK_FLOAT &f4t2, const KK_FLOAT &f4t3,
    const KK_FLOAT &f4t4, const KK_FLOAT &f4t7, const KK_FLOAT &f4t8,
    const KK_FLOAT &df4t1, const KK_FLOAT &df4t2, const KK_FLOAT &df4t3,
    const KK_FLOAT &df4t4, const KK_FLOAT &df4t7, const KK_FLOAT &df4t8,
    const KK_FLOAT &factor_lj,
    const KK_FLOAT &theta1, const KK_FLOAT &theta2, const KK_FLOAT &theta3,
    const KK_FLOAT &theta4, const KK_FLOAT &theta4p,
    const KK_FLOAT &theta7, const KK_FLOAT &theta8,
    const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&b_nx)[3],
    const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&b_nz)[3],
    const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT (&delta)[3], KK_FLOAT (&deltb)[3]);
};

// ---- out-of-class definitions (in header so the fused hbond+xstk kernel
// in pair_oxdna_hbond_kokkos.cpp can reuse these device helpers) ----

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaXstkKokkos<DeviceType>::xstk_force_contrib(const KK_FLOAT &f2,
  const KK_FLOAT &f4t1, const KK_FLOAT &f4t2,
  const KK_FLOAT &f4t3, const KK_FLOAT &f4t4, const KK_FLOAT &f4t7, const KK_FLOAT &f4t8,
  const KK_FLOAT &df2, const KK_FLOAT &df4t2, const KK_FLOAT &df4t3, const KK_FLOAT &df4t7,
  const KK_FLOAT &df4t8, const KK_FLOAT &rinv_hb, const KK_FLOAT &factor_lj,
  const KK_FLOAT &theta2, const KK_FLOAT &theta3, const KK_FLOAT &theta7, const KK_FLOAT &theta8,
  const KK_FLOAT &cost2, const KK_FLOAT &cost3, const KK_FLOAT &cost7, const KK_FLOAT &cost8,
  const KK_FLOAT (&delr_hb)[3], const KK_FLOAT (&delr_hb_norm)[3],
  const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&b_nx)[3],
  const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&b_nz)[3],
  const KK_FLOAT (&ra_chb)[3], const KK_FLOAT (&rb_chb)[3],
  KK_FLOAT (&delf)[3], KK_FLOAT (&delta)[3], KK_FLOAT (&deltb)[3])
{
  KK_FLOAT finc;

  finc  = -df2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;
  delf[0] = fma(delr_hb[0], finc, delf[0]);
  delf[1] = fma(delr_hb[1], finc, delf[1]);
  delf[2] = fma(delr_hb[2], finc, delf[2]);

  if (theta2) {
    finc = -f2 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;
    const KK_FLOAT t2f0 = fma(delr_hb_norm[0], cost2, a_nx[0]);
    const KK_FLOAT t2f1 = fma(delr_hb_norm[1], cost2, a_nx[1]);
    const KK_FLOAT t2f2 = fma(delr_hb_norm[2], cost2, a_nx[2]);
    delf[0] = fma(t2f0, finc, delf[0]);
    delf[1] = fma(t2f1, finc, delf[1]);
    delf[2] = fma(t2f2, finc, delf[2]);
  }

  if (theta3) {
    finc = -f2 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;
    const KK_FLOAT t3f0 = fma(delr_hb_norm[0], cost3, -b_nx[0]);
    const KK_FLOAT t3f1 = fma(delr_hb_norm[1], cost3, -b_nx[1]);
    const KK_FLOAT t3f2 = fma(delr_hb_norm[2], cost3, -b_nx[2]);
    delf[0] = fma(t3f0, finc, delf[0]);
    delf[1] = fma(t3f1, finc, delf[1]);
    delf[2] = fma(t3f2, finc, delf[2]);
  }

  if (theta7) {
    finc = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * rinv_hb * factor_lj;
    const KK_FLOAT t7f0 = fma(delr_hb_norm[0], cost7, a_nz[0]);
    const KK_FLOAT t7f1 = fma(delr_hb_norm[1], cost7, a_nz[1]);
    const KK_FLOAT t7f2 = fma(delr_hb_norm[2], cost7, a_nz[2]);
    delf[0] = fma(t7f0, finc, delf[0]);
    delf[1] = fma(t7f1, finc, delf[1]);
    delf[2] = fma(t7f2, finc, delf[2]);
  }

  if (theta8) {
    finc = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * rinv_hb * factor_lj;
    const KK_FLOAT t8f0 = fma(delr_hb_norm[0], cost8, -b_nz[0]);
    const KK_FLOAT t8f1 = fma(delr_hb_norm[1], cost8, -b_nz[1]);
    const KK_FLOAT t8f2 = fma(delr_hb_norm[2], cost8, -b_nz[2]);
    delf[0] = fma(t8f0, finc, delf[0]);
    delf[1] = fma(t8f1, finc, delf[1]);
    delf[2] = fma(t8f2, finc, delf[2]);
  }

  delta[0] = fma(ra_chb[1], delf[2], -ra_chb[2] * delf[1]);
  delta[1] = fma(ra_chb[2], delf[0], -ra_chb[0] * delf[2]);
  delta[2] = fma(ra_chb[0], delf[1], -ra_chb[1] * delf[0]);

  deltb[0] = fma(rb_chb[1], delf[2], -rb_chb[2] * delf[1]);
  deltb[1] = fma(rb_chb[2], delf[0], -rb_chb[0] * delf[2]);
  deltb[2] = fma(rb_chb[0], delf[1], -rb_chb[1] * delf[0]);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaXstkKokkos<DeviceType>::xstk_torque_contrib(const KK_FLOAT &f2,
  const KK_FLOAT &f4t1, const KK_FLOAT &f4t2, const KK_FLOAT &f4t3,
  const KK_FLOAT &f4t4, const KK_FLOAT &f4t7, const KK_FLOAT &f4t8,
  const KK_FLOAT &df4t1, const KK_FLOAT &df4t2, const KK_FLOAT &df4t3,
  const KK_FLOAT &df4t4, const KK_FLOAT &df4t7, const KK_FLOAT &df4t8,
  const KK_FLOAT &factor_lj,
  const KK_FLOAT &theta1, const KK_FLOAT &theta2, const KK_FLOAT &theta3,
  const KK_FLOAT &theta4, const KK_FLOAT &theta4p,
  const KK_FLOAT &theta7, const KK_FLOAT &theta8,
  const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&b_nx)[3],
  const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&b_nz)[3],
  const KK_FLOAT (&delr_hb_norm)[3],
  KK_FLOAT (&delta)[3], KK_FLOAT (&deltb)[3])
{
  delta[0] = 0.0;
  delta[1] = 0.0;
  delta[2] = 0.0;
  deltb[0] = 0.0;
  deltb[1] = 0.0;
  deltb[2] = 0.0;

  KK_FLOAT tpair;

  if (theta1) {
    tpair = -f2 * df4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj;
    const KK_FLOAT t1dir0 = fma(a_nx[1], b_nx[2], -a_nx[2] * b_nx[1]);
    const KK_FLOAT t1dir1 = fma(a_nx[2], b_nx[0], -a_nx[0] * b_nx[2]);
    const KK_FLOAT t1dir2 = fma(a_nx[0], b_nx[1], -a_nx[1] * b_nx[0]);
    delta[0] += t1dir0 * tpair;
    delta[1] += t1dir1 * tpair;
    delta[2] += t1dir2 * tpair;
    deltb[0] += t1dir0 * tpair;
    deltb[1] += t1dir1 * tpair;
    deltb[2] += t1dir2 * tpair;
  }
  if (theta2) {
    tpair = -f2 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj;
    const KK_FLOAT t2dir0 = fma(a_nx[1], delr_hb_norm[2], -a_nx[2] * delr_hb_norm[1]);
    const KK_FLOAT t2dir1 = fma(a_nx[2], delr_hb_norm[0], -a_nx[0] * delr_hb_norm[2]);
    const KK_FLOAT t2dir2 = fma(a_nx[0], delr_hb_norm[1], -a_nx[1] * delr_hb_norm[0]);
    delta[0] += t2dir0 * tpair;
    delta[1] += t2dir1 * tpair;
    delta[2] += t2dir2 * tpair;
  }
  if (theta3) {
    tpair = -f2 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * factor_lj;
    const KK_FLOAT t3dir0 = fma(b_nx[1], delr_hb_norm[2], -b_nx[2] * delr_hb_norm[1]);
    const KK_FLOAT t3dir1 = fma(b_nx[2], delr_hb_norm[0], -b_nx[0] * delr_hb_norm[2]);
    const KK_FLOAT t3dir2 = fma(b_nx[0], delr_hb_norm[1], -b_nx[1] * delr_hb_norm[0]);
    deltb[0] += t3dir0 * tpair;
    deltb[1] += t3dir1 * tpair;
    deltb[2] += t3dir2 * tpair;
  }
  if (theta4 && theta4p) {
    tpair = -f2 * f4t1 * f4t2 * f4t3 * df4t4 * f4t7 * f4t8 * factor_lj;
    const KK_FLOAT t4dir0 = fma(b_nz[1], a_nz[2], -b_nz[2] * a_nz[1]);
    const KK_FLOAT t4dir1 = fma(b_nz[2], a_nz[0], -b_nz[0] * a_nz[2]);
    const KK_FLOAT t4dir2 = fma(b_nz[0], a_nz[1], -b_nz[1] * a_nz[0]);
    delta[0] += t4dir0 * tpair;
    delta[1] += t4dir1 * tpair;
    delta[2] += t4dir2 * tpair;
    deltb[0] += t4dir0 * tpair;
    deltb[1] += t4dir1 * tpair;
    deltb[2] += t4dir2 * tpair;
  }
  if (theta7) {
    tpair = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * factor_lj;
    const KK_FLOAT t7dir0 = fma(a_nz[1], delr_hb_norm[2], -a_nz[2] * delr_hb_norm[1]);
    const KK_FLOAT t7dir1 = fma(a_nz[2], delr_hb_norm[0], -a_nz[0] * delr_hb_norm[2]);
    const KK_FLOAT t7dir2 = fma(a_nz[0], delr_hb_norm[1], -a_nz[1] * delr_hb_norm[0]);
    delta[0] += t7dir0 * tpair;
    delta[1] += t7dir1 * tpair;
    delta[2] += t7dir2 * tpair;
  }
  if (theta8) {
    tpair = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * factor_lj;
    const KK_FLOAT t8dir0 = fma(b_nz[1], delr_hb_norm[2], -b_nz[2] * delr_hb_norm[1]);
    const KK_FLOAT t8dir1 = fma(b_nz[2], delr_hb_norm[0], -b_nz[0] * delr_hb_norm[2]);
    const KK_FLOAT t8dir2 = fma(b_nz[0], delr_hb_norm[1], -b_nz[1] * delr_hb_norm[0]);
    deltb[0] += t8dir0 * tpair;
    deltb[1] += t8dir1 * tpair;
    deltb[2] += t8dir2 * tpair;
  }
}

}

#endif
#endif
