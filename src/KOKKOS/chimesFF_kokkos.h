/*
    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
    Contributing Author: Stan Moore (2025)
*/

#ifndef _chimesFF_KOKKOS_h
#define _chimesFF_KOKKOS_h

#include "chimesFF.h"
#include "kokkos_type.h"
#include "memory_kokkos.h"

#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<algorithm>
#include<cmath>
#include<map>

#define pi 3.14159265359

constexpr int MAX_2B_POLY = 13;
constexpr int MAX_3B_POLY = 9;
constexpr int MAX_4B_POLY = 4;

// PR #4601-style register-blocking factor for the dense/sparse coefficient
// reductions. Each vector lane keeps CHIMES_COEFF_BATCH independent
// Kokkos::Array accumulators (filled from coalesced, ThreadVectorRange-strided
// coefficient reads) to break the single dependent FMA chain into independent
// chains and expose instruction-level parallelism.
//
// 1 = disabled (default; bitwise-unchanged vs the prior reductions). The
// optimal value is hardware dependent, so — exactly like the Kokkos SNAP
// implementation, which defaults to 1 and sets per-architecture values — bump
// this to 2 or 4 for the target GPU and validate/profile before relying on it.
constexpr int CHIMES_COEFF_BATCH = 1;

// Pad the trailing (coefficient) dimension of the 3B/4B parameter tables to a
// multiple of this, so each triplet/quadruplet row starts at an aligned offset.
// A team reads one row at consecutive coefficient indices across its vector
// lanes, so aligned row starts give fully coalesced loads (mirrors SNAP's
// padding_factor). The padding entries are zero-filled and never read (the
// reductions are bounded by the true per-type coefficient count).
constexpr int CHIMES_PARAM_PAD = 32;

using namespace std;

// Notes:
//
// 1. A Morse-style coordinate transformation is hard-coded (see set_cheby_polys)
// 2. Polynomials are hard-coded over the domain [-1,1]
// 3. A cubic style cutoff is assumed, and Tersoff is the only other style considered (see get_fcut)


#define CHDIM 3 // The number of spatial dimensions.
#define USE_DISTANCE_TENSOR 0 // Use tensor of distances in computing stresses.

// Reduction value type used to parallelize the dense 3-body Chebyshev
// evaluation across a Kokkos ThreadVectorRange (energy + 3 pair-force
// derivatives). Modeled on the hierarchical-parallelism reductions used by
// the Kokkos SNAP implementation.

struct s_chimes_poly3 {
  KK_FLOAT e, f0, f1, f2;
  KOKKOS_INLINE_FUNCTION s_chimes_poly3() { e = 0.0; f0 = 0.0; f1 = 0.0; f2 = 0.0; }
  KOKKOS_INLINE_FUNCTION void operator+=(const s_chimes_poly3& rhs) {
    e += rhs.e; f0 += rhs.f0; f1 += rhs.f1; f2 += rhs.f2;
  }
};

// Reduction value type for the dense 4-body evaluation (energy + 6 pair-force
// derivatives), distributed across a Kokkos ThreadVectorRange.

struct s_chimes_poly4 {
  KK_FLOAT e, f0, f1, f2, f3, f4, f5;
  KOKKOS_INLINE_FUNCTION s_chimes_poly4() { e = 0.0; f0 = 0.0; f1 = 0.0; f2 = 0.0; f3 = 0.0; f4 = 0.0; f5 = 0.0; }
  KOKKOS_INLINE_FUNCTION void operator+=(const s_chimes_poly4& rhs) {
    e += rhs.e; f0 += rhs.f0; f1 += rhs.f1; f2 += rhs.f2; f3 += rhs.f3; f4 += rhs.f4; f5 += rhs.f5;
  }
};

namespace Kokkos {
  template<>
  struct reduction_identity<s_chimes_poly3> {
    KOKKOS_FORCEINLINE_FUNCTION static s_chimes_poly3 sum() { return s_chimes_poly3(); }
  };
  template<>
  struct reduction_identity<s_chimes_poly4> {
    KOKKOS_FORCEINLINE_FUNCTION static s_chimes_poly4 sum() { return s_chimes_poly4(); }
  };
}

template<class DeviceType>
class chimesFFKokkos : public chimesFF
{
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef typename Kokkos::TeamPolicy<DeviceType>::member_type t_team;

  ////////////////////////
  // General parameters
  ////////////////////////

  int d_poly_orders[3];    // [bodiedness-1]; i.e. 12 = 2-body only, 12th order; 12 5 = 2+3-body, 0 5 = 3-body only, 5th order

  ////////////////////////
  // Functions
  ////////////////////////

  chimesFFKokkos();
  ~chimesFFKokkos();

  void read_parameters(string paramfile) override;

  KOKKOS_INLINE_FUNCTION
  void compute_1B(const int typ_idx, KK_FLOAT & energy) const;

  // 2+B compute functions overloaded with force_scalar var for compatibility with LAMMPS

  KOKKOS_INLINE_FUNCTION
  void compute_2B(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const;

  KOKKOS_INLINE_FUNCTION
  void compute_2B(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT& force_scalar) const;

  // scratch points at this cluster's 2*npairs*MAX_*B_POLY-float slice of team
  // scratch (allocated once by the caller). The caller owns the get_shmem so a
  // team that evaluates many clusters (fused kernels) does not re-allocate.
  // reuse_lead: skip the lead (ij) pair's set_cheby_polys and read its prebuilt
  // Tn/Tnd from scratch slots 0 / npairs (filled once per typ_k bucket by
  // set_cheby_lead_3b). Default false reproduces the standard per-cluster build.
  KOKKOS_INLINE_FUNCTION
  void compute_3B(const t_team& team, KK_FLOAT* scratch, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, bool reuse_lead = false) const;

  KOKKOS_INLINE_FUNCTION
  void compute_3B(const t_team& team, KK_FLOAT* scratch, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force,KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar, bool reuse_lead = false) const;

  // Fill the lead (ij) pair's Tn/Tnd into scratch slots 0 / npairs for a fixed
  // triplet type (cutoffs from tripidx); team-collective, used by the fused
  // per-2-mer kernel once per typ_k bucket.
  KOKKOS_INLINE_FUNCTION
  void set_cheby_lead_3b(const t_team& team, KK_FLOAT* scratch, KK_FLOAT dist_ij, int typ_i, int typ_j, int type_idx, int tripidx) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B(const t_team& team, KK_FLOAT* scratch, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B(const t_team& team, KK_FLOAT* scratch, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar) const;

  // Per-team scratch byte count for the n_values Chebyshev (Tn/Tnd) entries
  // cached in team scratch by compute_3B/compute_4B. Mirrors the Kokkos SNAP
  // scratch_size_helper idiom.
  static size_t scratch_bytes(int n_values) {
    typedef Kokkos::View<KK_FLOAT*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewType;
    return ScratchViewType::shmem_size(n_values);
  }

  // Functions to aid using ChIMES Calculator for fitting

  void build_pair_int_trip_map() override;
  void build_pair_int_quad_map() override;

  int eflag,vflag;

 private:

  typename AT::t_kkfloat_1d d_morse_var;      // [npairs]; morse_lambda
  typename AT::t_kkfloat_1d_const_um c_morse_var;

  KK_FLOAT d_penalty_params[2]; // [2];  Second dimension: [0] = A_pen, [1] = d_pen
  typename AT::t_kkfloat_1d d_energy_offsets; // [natmtyps]; Single atom ChIMES energies
  typename AT::t_kkfloat_1d_const_um c_energy_offsets;

 public:

  ////////////////////////
  // Definitions for pair, triplet, and quadruplet types
  ////////////////////////

  // 2-body maps

  typename AT::t_int_1d d_atom_int_pair_map; // [nmaps] "fast" maps, based on atom type index
  typename AT::t_int_1d_const_um c_atom_int_pair_map;

  // 3-body maps

  typename AT::t_int_1d d_atom_int_trip_map;    // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of three atoms of arbitrary order
  typename AT::t_int_1d_const_um c_atom_int_trip_map;

  typename AT::t_int_2d d_pair_int_trip_map;  // Gives the atom pair indices for an arbitrary triplet of atom types.
  typename AT::t_int_2d_const_um c_pair_int_trip_map;

  // 4-body maps

  typename AT::t_int_1d d_atom_int_quad_map; // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of four atoms of arbitrary order
  typename AT::t_int_1d_const_um c_atom_int_quad_map;

  typename AT::t_int_2d d_pair_int_quad_map;  // Gives the atom pair indices for an arbitrary quad of atom types.
  typename AT::t_int_2d_const_um c_pair_int_quad_map;

  ////////////////////////
  // Polynomial parameters
  ////////////////////////

  // number of coefficients for the pair/triplet/quadruplet type

  typename AT::t_int_1d d_ncoeffs_2b;        // [npairs]
  typename AT::t_int_1d_const_um c_ncoeffs_2b;

  typename AT::t_int_2d d_chimes_2b_pows;    // [npairs][npowers] power for the coresponding parameter
  typename AT::t_int_2d_const_um c_chimes_2b_pows;

  typename AT::t_kkfloat_2d d_chimes_2b_params;    // [npairs][npowers] 2-body polynomial coefficients
  typename AT::t_kkfloat_2d_const_um c_chimes_2b_params;

  typename AT::t_kkfloat_1d_2 d_chimes_2b_cutoff;  // [npairs][2] inner and outer cutoff for pair
  typename AT::t_kkfloat_1d_2_const_um c_chimes_2b_cutoff;

  typename AT::t_int_1d d_ncoeffs_3b;              // [ntrips]
  typename AT::t_int_1d_const_um c_ncoeffs_3b;

  typename AT::t_int_3d d_chimes_3b_powers;    // [ntrips][constit. pair][nparams] (coeff innermost, padded)
  typename AT::t_int_3d_const_um c_chimes_3b_powers;

  typename AT::t_kkfloat_2d d_chimes_3b_params;    // [ntrips][nparams]
  typename AT::t_kkfloat_2d_const_um c_chimes_3b_params;

  typename AT::t_kkfloat_3d d_chimes_3b_cutoff;    // [ntrips][constit. pair][2] inner and outer cutoff for pair 1
  typename AT::t_kkfloat_3d_const_um c_chimes_3b_cutoff;

  typename AT::t_int_1d d_ncoeffs_4b;          // [nquads]
  typename AT::t_int_1d_const_um c_ncoeffs_4b;

  typename AT::t_int_3d d_chimes_4b_powers;    // [nquads][constit. pair][nparams] (coeff innermost, padded)
  typename AT::t_int_3d_const_um c_chimes_4b_powers;

  typename AT::t_kkfloat_2d d_chimes_4b_params;    // [nquads][nparams]
  typename AT::t_kkfloat_2d_const_um c_chimes_4b_params;

  typename AT::t_kkfloat_3d d_chimes_4b_cutoff;    // [nquads][constit. pair][2] inner and outer cutoff for pair 1
  typename AT::t_kkfloat_3d_const_um c_chimes_4b_cutoff;

 private:

  // Tools for compute functions

  KOKKOS_INLINE_FUNCTION
  void set_cheby_polys(KK_FLOAT* Tn, KK_FLOAT* Tnd, KK_FLOAT dx, const KK_FLOAT morse,
                       const KK_FLOAT inner_cutoff, const KK_FLOAT outer_cutoff, const int order) const;

  KOKKOS_INLINE_FUNCTION
  void poly_2B(KK_FLOAT &e, KK_FLOAT &f0, int ncoeffs_2b, int pair_idx,
               KK_FLOAT* Tn, KK_FLOAT* Tnd) const;

  KOKKOS_INLINE_FUNCTION
  void poly_3B(const t_team& team, KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_3b, int tripidx, int idx,
               KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_jk,
               KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_jk) const;

  // Evaluates the 3-Body chebyshev polynomial in dense format

  KOKKOS_INLINE_FUNCTION
  void poly_3B_dense(const t_team& team, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, int ncoeffs_3b,
                     int tripidx, KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik,
                     KK_FLOAT* Tn_jk, KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik,
                     KK_FLOAT* Tnd_jk) const;

  // Loop evaluators for poly_3B_dense

  KOKKOS_INLINE_FUNCTION
  void poly_3B_dense_loop1(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2,
                           int ncoeffs_3b, int tripidx, KK_FLOAT* Tn_ij,
                           KK_FLOAT* Tn_ik, KK_FLOAT* Tn_jk, KK_FLOAT* Tnd_ij,
                           KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_jk) const;

  KOKKOS_INLINE_FUNCTION
  void poly_3B_dense_loop2(const t_team& team, int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2,
                           int ncoeffs_3b, int tripidx, KK_FLOAT* Tn_ij,
                           KK_FLOAT* Tn_ik, KK_FLOAT* Tn_jk, KK_FLOAT* Tnd_ij,
                           KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_jk) const;

  KOKKOS_INLINE_FUNCTION
  void poly_4B(const t_team& team, KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_4b, int quadidx, int idx,
               KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
               KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
               KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
               KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const;

  // Evaluates the 4-body Chebyshev polynomial in dense format

  KOKKOS_INLINE_FUNCTION
  void poly_4B_dense(const t_team& team, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3, KK_FLOAT &f4,
                     KK_FLOAT &f5, int ncoeffs_4b, int quadidx, KK_FLOAT* Tn_ij,
                     KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il, KK_FLOAT* Tn_jk,
                     KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl, KK_FLOAT* Tnd_ij,
                     KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il, KK_FLOAT* Tnd_jk,
                     KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const;

  // Loop1 uses a flat loop to evaluate a dense 4-body polynomial

  KOKKOS_INLINE_FUNCTION
  void poly_4B_dense_loop1(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3,
                           KK_FLOAT &f4, KK_FLOAT &f5, int ncoeffs_4b, int quadidx,
                           KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
                           KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
                           KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
                           KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const;

  // Innver evaluation loop for dense 4 body poly.  2nd. variant.

  KOKKOS_INLINE_FUNCTION
  void poly_4B_dense_loop2(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3,
                           KK_FLOAT &f4, KK_FLOAT &f5, int ncoeffs_4b, int quadidx,
                           KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
                           KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
                           KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
                           KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const;

  KOKKOS_INLINE_FUNCTION
  void set_polys_out_of_range(KK_FLOAT* d_Tn, KK_FLOAT* d_Tnd, KK_FLOAT dx, KK_FLOAT x,
                              int poly_order, KK_FLOAT inner_cutoff, KK_FLOAT exprlen, KK_FLOAT dx_dr) const;

  KOKKOS_INLINE_FUNCTION
  void get_fcut(const KK_FLOAT dx, const KK_FLOAT outer_cutoff, KK_FLOAT & fcut, KK_FLOAT & fcutderiv) const;

  KOKKOS_INLINE_FUNCTION
  void get_penalty(const KK_FLOAT dx, const int & pair_idx, KK_FLOAT & E_penalty, KK_FLOAT & force_scalar) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT dr2_3B(const KK_FLOAT* dr2, int i, int j, int k, int l) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT dr2_4B(const KK_FLOAT* dr2, int i, int j, int k, int l) const;

  KOKKOS_INLINE_FUNCTION
  void init_distance_tensor(KK_FLOAT* dr2, const KK_FLOAT* dr, int natoms) const;
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::get_fcut(const KK_FLOAT dx, const KK_FLOAT outer_cutoff, KK_FLOAT & fcut, KK_FLOAT & fcutderiv) const
{
  KK_FLOAT fcut0;
  KK_FLOAT fcut0_deriv;

  if (fcut_type == fcutType::CUBIC) {
    fcut0 = (1.0 - dx/outer_cutoff);
    fcut = pow(fcut0,3.0);
    fcutderiv = pow(fcut0,2.0);
    fcutderiv *= -1.0 * 3.0 /outer_cutoff;
  } else if (fcut_type == fcutType::TERSOFF) {

    KK_FLOAT THRESH = outer_cutoff-fcut_var*outer_cutoff;

    if (dx < THRESH)        // Case 1: Our pair distance is less than the fcut kick-in distance
    {
      fcut = 1.0;
      fcutderiv = 0.0;
    }
    else if (dx > outer_cutoff)        // Case 2: Our pair distance is greater than the cutoff
    {
      fcut = 0.0;
      fcutderiv = 0.0;
    }
    else                // Case 3: We'll use our modified sin function
    {
      fcut0 = (dx-THRESH) / (outer_cutoff-THRESH) * pi + pi/2.0;
      fcut0_deriv = pi / (outer_cutoff - THRESH);

      fcut = 0.5 + 0.5 * sin(fcut0);
      fcutderiv  = 0.5 * cos(fcut0) * fcut0_deriv;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::get_penalty(const KK_FLOAT dx, const int& pair_idx, KK_FLOAT& E_penalty, KK_FLOAT& force_scalar) const
{
  KK_FLOAT r_penalty = 0.0;

  E_penalty    = 0.0;
  force_scalar = 1.0;

  if (dx - d_penalty_params[0] < c_chimes_2b_cutoff(pair_idx,0)) { // Then we're within the penalty-enforced region of distance space
    r_penalty = c_chimes_2b_cutoff(pair_idx,0) + d_penalty_params[0] - dx;

    /*if (dx < c_chimes_2b_cutoff(pair_idx,0))
      badness = 2;
    else if (1 > this->badness) // Only update badness if candiate badness is worse than its current value
      badness = 1;*/
  }

  if (r_penalty > 0.0) {
    E_penalty    = r_penalty * r_penalty * r_penalty * d_penalty_params[1];

    force_scalar = -3.0 * r_penalty * r_penalty * d_penalty_params[1];

    //if (rank == 0) // Commenting out - we need all ranks to report if the penalty function has been sampled
    //{
        /*cout << "chimesFFKokkos: " << "Adding penalty in 2B Cheby calc, r < rmin+penalty_dist " << fixed
             << dx << " "
             << d_chimes_2b_cutoff(pair_idx,0) + d_penalty_params[0]
             << " pair type: " << pair_idx << endl;
        cout << "chimesFFKokkos: " << "\t...Penalty potential = "<< E_penalty << endl;*/
    //}
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::set_cheby_polys(KK_FLOAT* Tn, KK_FLOAT* Tnd, KK_FLOAT dx, const KK_FLOAT morse,
                                     const KK_FLOAT inner_cutoff, const KK_FLOAT outer_cutoff, const int order) const
{
  // Currently assumes a Morse-style transformation has been requested

  // Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd).  Tnd is the derivative
  // with respect to the interatomic distance, not the transformed distance (x).

  // Do the Morse transformation

  const KK_FLOAT x_min = exp(-1*inner_cutoff/morse);
  const KK_FLOAT x_max = exp(-1*outer_cutoff/morse);

  const KK_FLOAT x_avg   = 0.5 * (x_max + x_min);
  KK_FLOAT x_diff  = 0.5 * (x_max - x_min);

  x_diff *= -1.0; // Special for Morse style

  bool out_of_range;
  const KK_FLOAT dx_orig = dx;

  // The case dx > outer_cutoff is not treated, because it is assumed that the outer smoothing
  //  function will be zero for dx > outer_cutoff

  if (dx < inner_cutoff) {
    out_of_range = true;
    dx = inner_cutoff;
  } else
    out_of_range = false;

  const KK_FLOAT exprlen = exp(-1*dx/morse);
  const  KK_FLOAT x = (exprlen - x_avg)/x_diff;
  const KK_FLOAT dx_dr = (-exprlen/morse)/x_diff;

  if (!out_of_range) {

    // Generate Chebyshev polynomials by recursion.
    //
    // What we're doing here. Want to fit using Cheby polynomials of the 1st kinD[i]. "T_n(x)."
    // We need to calculate the derivative of these polynomials.
    // Derivatives are defined through use of Cheby polynomials of the 2nd kind "U_n(x)", as:
    //
    // d/dx[ T_n(x) = n * U_n-1(x)]
    //
    // So we need to first set up the 1st-kind polynomials ("Tn[]")
    // Then, to compute the derivatives ("Tnd[]"), first set equal to the 2nd-kind, then multiply by n to get the der's

    // First two 1st-kind Chebys:

    Tn[0] = 1.0;
    Tn[1] = x;

    // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

    Tnd[0] = 1.0;
    Tnd[1] = 2.0 * x;

    // Use recursion to set up the higher n-value Tn and Tnd's

    for (int i = 2; i <= order; i++) {
      Tn[i]  = 2.0 * x *  Tn[i-1] -  Tn[i-2];
      Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
    }

    // Now multiply by n to convert Tnd's to actual derivatives of Tn

    // The following dx_dr compuation assumes a Morse transformation
    // DERIV_CONST is no longer used. (old way: dx_dr = DERIV_CONST*cheby_var_deriv(x_diff, rlen, ff_2body.LAMBDA, ff_2body.CHEBY_TYPE, exprlen);)

    for (int i = order; i >= 1; i--)
      Tnd[i] = i * dx_dr * Tnd[i-1];

    Tnd[0] = 0.0;
  } else { // out_of_range == true
    //////cout << "Warning: An intermolecular distance less than the inner cutoff = " << inner_cutoff << " was found\n ";
    //////cout << "         Distance = " << dx_orig << endl;

    set_polys_out_of_range(Tn, Tnd, dx_orig, x, order, inner_cutoff, exprlen, dx_dr);
  }
}

#include "chimesFF_kokkos_impl.h"
#endif
