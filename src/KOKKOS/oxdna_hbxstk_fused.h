/* -*- c++ -*- ----------------------------------------------------------
   Prototype helper for fusing the oxDNA hbond + xstk Kokkos pair styles.

   OxdnaXstkCoeffs bundles the (shallow) device view handles for every
   cross-stacking coefficient so that the hbond pair style can evaluate the
   xstk term inside a single fused kernel. The xstk pair style fills it via
   PairOxdnaXstkKokkos::export_fused_coeffs(); the views alias the xstk style's
   own coefficient arrays (no deep copy), so this is cheap to populate once per
   compute() call.
------------------------------------------------------------------------- */

#ifndef LMP_OXDNA_HBXSTK_FUSED_H
#define LMP_OXDNA_HBXSTK_FUSED_H

#include "kokkos_type.h"

namespace LAMMPS_NS {

// Two-component reduction value for the fused hbond+xstk kernel: it carries the
// hydrogen-bonding and cross-stacking global energy and virial separately so the
// hbond and xstk pair styles can each be credited their own contribution.
struct s_EV_HBXST {
  KK_ACC_FLOAT evdwl_hb, evdwl_xst;
  KK_ACC_FLOAT v_hb[6], v_xst[6];
  KOKKOS_INLINE_FUNCTION
  s_EV_HBXST() {
    evdwl_hb = 0; evdwl_xst = 0;
    for (int i = 0; i < 6; ++i) { v_hb[i] = 0; v_xst[i] = 0; }
  }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_EV_HBXST &r) {
    evdwl_hb += r.evdwl_hb; evdwl_xst += r.evdwl_xst;
    for (int i = 0; i < 6; ++i) { v_hb[i] += r.v_hb[i]; v_xst[i] += r.v_xst[i]; }
  }
};
typedef struct s_EV_HBXST EV_HBXST;

// Reconstruct the nx and nz frame vectors from an orientation quaternion, using
// the exact fma expressions of FixOxdnaLRFKokkos so the result is bit-identical
// to the precomputed d_nx/d_nz. (nx = first body axis a1, nz = third body axis
// a3; the fused hbond+xstk kernel needs only these two.)
KOKKOS_INLINE_FUNCTION
static void oxdna_quat_to_nx_nz(const KK_FLOAT q0, const KK_FLOAT q1,
                               const KK_FLOAT q2, const KK_FLOAT q3,
                               KK_FLOAT (&nx)[3], KK_FLOAT (&nz)[3])
{
  const KK_FLOAT two = 2.0;
  nx[0] = fma(q0, q0, fma(q1, q1, -fma(q2, q2, q3 * q3)));
  nx[1] = two * fma(q1, q2, q0 * q3);
  nx[2] = two * fma(q1, q3, -q0 * q2);
  nz[0] = two * fma(q1, q3, q0 * q2);
  nz[1] = two * fma(q2, q3, -q0 * q1);
  nz[2] = fma(q0, q0, q3 * q3 - fma(q1, q1, q2 * q2));
}

template<class DeviceType>
struct OxdnaXstkCoeffs {
  typedef ArrayTypes<DeviceType> AT;

  // radial F2 coefficients
  typename AT::t_kkfloat_2d_randomread d_k_xst, d_cut_xst_0, d_cut_xst_c;
  typename AT::t_kkfloat_2d_randomread d_cut_xst_lo, d_cut_xst_hi;
  typename AT::t_kkfloat_2d_randomread d_cut_xst_lc, d_cut_xst_hc, d_b_xst_lo, d_b_xst_hi;

  // angular F4 coefficients (theta 1,2,3,4,7,8)
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
};

// Packed per-(atype,btype) cross-stacking coefficients as a single POD struct.
// Stored in a View<OxdnaXstkPackedCoeff**>, the fused kernel loads ONE of these
// per pair: one global base address + fixed field offsets, instead of indexing
// ~39 separate Kokkos Views (each with its own base pointer that must occupy a
// register at the access). This mirrors the standalone's single by-value param
// struct and is the prototype for testing whether the fused kernel's
// view-base-pointer register pressure is what makes fusion slower than the split
// pair in LAMMPS KOKKOS (where every coefficient is its own View).
struct OxdnaXstkPackedCoeff {
  KK_FLOAT k_xst, cut_xst_0, cut_xst_c, cut_xst_lo, cut_xst_hi, cut_xst_lc, cut_xst_hc, b_xst_lo, b_xst_hi;
  KK_FLOAT a_xst1, theta_xst1_0, dtheta_xst1_ast, b_xst1, dtheta_xst1_c;
  KK_FLOAT a_xst2, theta_xst2_0, dtheta_xst2_ast, b_xst2, dtheta_xst2_c;
  KK_FLOAT a_xst3, theta_xst3_0, dtheta_xst3_ast, b_xst3, dtheta_xst3_c;
  KK_FLOAT a_xst4, theta_xst4_0, dtheta_xst4_ast, b_xst4, dtheta_xst4_c;
  KK_FLOAT a_xst7, theta_xst7_0, dtheta_xst7_ast, b_xst7, dtheta_xst7_c;
  KK_FLOAT a_xst8, theta_xst8_0, dtheta_xst8_ast, b_xst8, dtheta_xst8_c;
};

}    // namespace LAMMPS_NS

#endif
