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

using namespace std;

// Notes:
//
// 1. A Morse-style coordinate transformation is hard-coded (see set_cheby_polys)
// 2. Polynomials are hard-coded over the domain [-1,1]
// 3. A cubic style cutoff is assumed, and Tersoff is the only other style considered (see get_fcut)


#define CHDIM 3 // The number of spatial dimensions.
#define USE_DISTANCE_TENSOR 1 // Use tensor of distances in computing stresses.

template<class DeviceType>
class chimesFFKokkos : public chimesFF
{
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  // Temporary storage for ChIMES interaction

  // Two-body

  struct Chimes2BKokkos {
    typename AT::t_kkfloat_2d d_Tn, d_Tnd;

    Chimes2BKokkos() {}

    Chimes2BKokkos(int size, int poly_order) {
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn,"chimes:Tn",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd,"chimes:Tnd",size,poly_order+1);
    }

    void resize(int size, int poly_order)
    {
      if (d_Tn.extent(0) < size || d_Tn.extent(1) < poly_order+1) {
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn,"chimes:Tn",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd,"chimes:Tn",size,poly_order+1);
      }
    }
  };

  // Three-body

  struct Chimes3BKokkos
  {
    typename AT::t_kkfloat_2d d_Tn_ij, d_Tn_ik, d_Tn_jk;   // The Chebyshev polymonials
    typename AT::t_kkfloat_2d d_Tnd_ij, d_Tnd_ik, d_Tnd_jk;  // The Chebyshev polymonial derivatives

    Chimes3BKokkos() {}

    Chimes3BKokkos(int size, int poly_order) {
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_ij,"chimes:Tn_ij",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_ik,"chimes:Tn_ik",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_jk,"chimes:Tn_jk",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_ij,"chimes:Tnd_ij",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_ik,"chimes:Tnd_ik",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_jk,"chimes:Tnd_jk",size,poly_order+1);
    }

    void resize(int size, int poly_order)
    {
      if (d_Tn_ij.extent(0) < size || d_Tn_ij.extent(1) < poly_order+1) {
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_ij,"chimes:Tn_ij",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_ik,"chimes:Tn_ik",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_jk,"chimes:Tn_jk",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_ij,"chimes:Tnd_ij",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_ik,"chimes:Tnd_ik",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_jk,"chimes:Tnd_jk",size,poly_order+1);
      }
    }
  };

  // Four-body

  struct Chimes4BKokkos {
    typename AT::t_kkfloat_2d d_Tn_ij, d_Tn_ik, d_Tn_il, d_Tn_jk, d_Tn_jl, d_Tn_kl;   // The Chebyshev polymonials
    typename AT::t_kkfloat_2d d_Tnd_ij, d_Tnd_ik, d_Tnd_il, d_Tnd_jk, d_Tnd_jl, d_Tnd_kl;  // The Chebyshev polymonial derivatives

    Chimes4BKokkos() {}

    Chimes4BKokkos(int size, int poly_order) {
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_ij,"chimes:Tn_ij",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_ik,"chimes:Tn_ik",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_il,"chimes:Tn_il",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_jk,"chimes:Tn_jk",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_jl,"chimes:Tn_jl",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_kl,"chimes:Tn_kl",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_ij,"chimes:Tnd_ij",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_ik,"chimes:Tnd_ik",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_il,"chimes:Tnd_il",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_jk,"chimes:Tnd_jk",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_jl,"chimes:Tnd_jl",size,poly_order+1);
      LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_kl,"chimes:Tnd_kl",size,poly_order+1);
    }

    void resize(int size, int poly_order)
    {
      if (d_Tn_ij.extent(0) < size || d_Tn_ij.extent(1) < poly_order+1) {
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_ij,"chimes:Tn_ij",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_ik,"chimes:Tn_ik",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_il,"chimes:Tn_il",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_jk,"chimes:Tn_jk",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_jl,"chimes:Tn_jl",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tn_kl,"chimes:Tn_kl",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_ij,"chimes:Tnd_ij",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_ik,"chimes:Tnd_ik",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_il,"chimes:Tnd_il",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_jk,"chimes:Tnd_jk",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_jl,"chimes:Tnd_jl",size,poly_order+1);
        LAMMPS_NS::MemKK::realloc_kokkos(d_Tnd_kl,"chimes:Tnd_kl",size,poly_order+1);
      }
    }
  };

  ////////////////////////
  // General parameters
  ////////////////////////

  typename AT::t_int_1d d_poly_orders;    // [bodiedness-1]; i.e. 12 = 2-body only, 12th order; 12 5 = 2+3-body, 0 5 = 3-body only, 5th order

  ////////////////////////
  // Functions
  ////////////////////////

  chimesFFKokkos();
  ~chimesFFKokkos();
  void resize_2B(int size) { chimes2BKK.resize(size,poly_orders[0]); }
  void resize_3B(int size) { chimes3BKK.resize(size,poly_orders[1]); }
  void resize_4B(int size) { chimes4BKK.resize(size,poly_orders[2]); }

  void read_parameters(string paramfile) override;

  KOKKOS_INLINE_FUNCTION
  void compute_1B(const int typ_idx, KK_FLOAT & energy) const;

  // 2+B compute functions overloaded with force_scalar_in var for compatibility with LAMMPS

  KOKKOS_INLINE_FUNCTION
  void compute_2B(const int ii, const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const;

  KOKKOS_INLINE_FUNCTION
  void compute_2B(const int ii, const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT& force_scalar_in) const;

  KOKKOS_INLINE_FUNCTION
  void compute_3B(const int ii, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const;

  KOKKOS_INLINE_FUNCTION
  void compute_3B(const int ii, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force,KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar_in) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B(const int ii, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const;

  KOKKOS_INLINE_FUNCTION
  void compute_4B(const int ii, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar_in) const;

  // Functions to aid using ChIMES Calculator for fitting

  void build_pair_int_trip_map() override;
  void build_pair_int_quad_map() override;

private:

  Chimes2BKokkos chimes2BKK;
  Chimes3BKokkos chimes3BKK;
  Chimes4BKokkos chimes4BKK;

  typename AT::t_kkfloat_1d d_morse_var;      // [npairs]; morse_lambda
  double d_penalty_params[2]; // [2];  Second dimension: [0] = A_pen, [1] = d_pen
  typename AT::t_kkfloat_1d d_energy_offsets; // [natmtyps]; Single atom ChIMES energies

  ////////////////////////
  // Definitions for pair, triplet, and quadruplet types
  ////////////////////////

  // 2-body maps

  typename AT::t_int_1d d_atom_int_pair_map; // [nmaps] "fast" maps, based on atom type index

  // 3-body maps

  typename AT::t_int_1d d_atom_int_trip_map;    // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of three atoms of arbitrary order
  typename AT::t_int_2d d_pair_int_trip_map;  // Gives the atom pair indices for an arbitrary triplet of atom types.

  // 4-body maps

  typename AT::t_int_1d d_atom_int_quad_map; // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of four atoms of arbitrary order
  typename AT::t_int_2d d_pair_int_quad_map;  // Gives the atom pair indices for an arbitrary quad of atom types.

  ////////////////////////
  // Polynomial parameters
  ////////////////////////

  // number of coefficients for the pair/triplet/quadruplet type

  typename AT::t_int_1d d_ncoeffs_2b;        // [npairs]

  typename AT::t_int_2d d_chimes_2b_pows;    // [npairs][npowers] power for the coresponding parameter

  typename AT::t_kkfloat_2d d_chimes_2b_params;    // [npairs][npowers] 2-body polynomial coefficients
  typename AT::t_kkfloat_1d_2 d_chimes_2b_cutoff;  // [npairs][2] inner and outer cutoff for pair

  typename AT::t_int_1d d_ncoeffs_3b;              // [ntrips]
  typename AT::t_kkfloat_3d d_chimes_3b_powers;    // [ntrips][nparams][constit. pair]
  typename AT::t_kkfloat_2d d_chimes_3b_params;    // [ntrips][nparams]
  typename AT::t_kkfloat_3d d_chimes_3b_cutoff;    // [ntrips][2][constit. pair] inner and outer cutoff for pair 1

  typename AT::t_int_1d d_ncoeffs_4b;          // [nquads]
  typename AT::t_int_3d d_chimes_4b_powers;    // [nquads][nparams][constit. pair]
  typename AT::t_kkfloat_2d d_chimes_4b_params;    // [nquads][nparams]
  typename AT::t_kkfloat_3d d_chimes_4b_cutoff;    // [nquads][2][constit. pair] inner and outer cutoff for pair 1

  // Tools for compute functions

  KOKKOS_INLINE_FUNCTION
  void set_cheby_polys(const int ii, typename AT::t_kkfloat_2d &Tn, typename AT::t_kkfloat_2d &Tnd, KK_FLOAT dx, const KK_FLOAT morse,
                       const KK_FLOAT inner_cutoff, const KK_FLOAT outer_cutoff, const int order) const;

  KOKKOS_INLINE_FUNCTION
  void poly_2B(const int ii, KK_FLOAT &e, KK_FLOAT &f0, int ncoeffs_2b, int pair_idx,
               typename AT::t_kkfloat_2d &Tn, typename AT::t_kkfloat_2d &Tnd) const;


  KOKKOS_INLINE_FUNCTION
  void poly_3B(const int ii, KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_3b, int tripidx, int idx,
               typename AT::t_kkfloat_2d &Tn_ij, typename AT::t_kkfloat_2d &Tn_ik, typename AT::t_kkfloat_2d &Tn_jk,
               typename AT::t_kkfloat_2d &Tnd_ij, typename AT::t_kkfloat_2d &Tnd_ik, typename AT::t_kkfloat_2d &Tnd_jk) const;

  KOKKOS_INLINE_FUNCTION
  void poly_4B(const int ii, KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_4b, int quadidx, int idx,
               typename AT::t_kkfloat_2d &Tn_ij, typename AT::t_kkfloat_2d &Tn_ik, typename AT::t_kkfloat_2d &Tn_il,
               typename AT::t_kkfloat_2d &Tn_jk, typename AT::t_kkfloat_2d &Tn_jl, typename AT::t_kkfloat_2d &Tn_kl,
               typename AT::t_kkfloat_2d &Tnd_ij, typename AT::t_kkfloat_2d &Tnd_ik, typename AT::t_kkfloat_2d &Tnd_il,
               typename AT::t_kkfloat_2d &Tnd_jk, typename AT::t_kkfloat_2d &Tnd_jl, typename AT::t_kkfloat_2d &Tnd_kl) const;

  KOKKOS_INLINE_FUNCTION
  void set_polys_out_of_range(const int ii, typename AT::t_kkfloat_2d &d_Tn, typename AT::t_kkfloat_2d &d_Tnd, KK_FLOAT dx, KK_FLOAT x,
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

  if (dx - d_penalty_params[0] < d_chimes_2b_cutoff(pair_idx,0)) { // Then we're within the penalty-enforced region of distance space
    r_penalty = d_chimes_2b_cutoff(pair_idx,0) + d_penalty_params[0] - dx;

    /*if (dx < d_chimes_2b_cutoff(pair_idx,0))
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
void chimesFFKokkos<DeviceType>::set_cheby_polys(const int ii, typename AT::t_kkfloat_2d &Tn, typename AT::t_kkfloat_2d &Tnd, KK_FLOAT dx, const KK_FLOAT morse,
                                     const KK_FLOAT inner_cutoff, const KK_FLOAT outer_cutoff, const int order) const
{
  // Currently assumes a Morse-style transformation has been requested

  // Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd).  Tnd is the derivative
  // with respect to the interatomic distance, not the transformed distance (x).

  // Do the Morse transformation

  KK_FLOAT x_min = exp(-1*inner_cutoff/morse);
  KK_FLOAT x_max = exp(-1*outer_cutoff/morse);

  KK_FLOAT x_avg   = 0.5 * (x_max + x_min);
  KK_FLOAT x_diff  = 0.5 * (x_max - x_min);

  x_diff *= -1.0; // Special for Morse style

  bool out_of_range;
  KK_FLOAT dx_orig = dx;

  // The case dx > outer_cutoff is not treated, because it is assumed that the outer smoothing
  //  function will be zero for dx > outer_cutoff

  if (dx < inner_cutoff) {
    out_of_range = true;
    dx = inner_cutoff;
  } else
    out_of_range = false;

  KK_FLOAT exprlen = exp(-1*dx/morse);
  KK_FLOAT x = (exprlen - x_avg)/x_diff;
  KK_FLOAT dx_dr = (-exprlen/morse)/x_diff;

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

    Tn(ii,0) = 1.0;
    Tn(ii,1) = x;

    // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

    Tnd(ii,0) = 1.0;
    Tnd(ii,1) = 2.0 * x;

    // Use recursion to set up the higher n-value Tn and Tnd's

    for (int i = 2; i <= order; i++) {
      Tn(ii, i)  = 2.0 * x *  Tn(ii, i-1) -  Tn(ii, i-2);
      Tnd(ii, i) = 2.0 * x * Tnd(ii, i-1) - Tnd(ii, i-2);
    }

    // Now multiply by n to convert Tnd's to actual derivatives of Tn

    // The following dx_dr compuation assumes a Morse transformation
    // DERIV_CONST is no longer used. (old way: dx_dr = DERIV_CONST*cheby_var_deriv(x_diff, rlen, ff_2body.LAMBDA, ff_2body.CHEBY_TYPE, exprlen);)

    for (int i = order; i >= 1; i--)
      Tnd(ii, i) = i * dx_dr * Tnd(ii, i-1);

    Tnd(ii,0) = 0.0;
  } else { // out_of_range == true
    //////cout << "Warning: An intermolecular distance less than the inner cutoff = " << inner_cutoff << " was found\n ";
    //////cout << "         Distance = " << dx_orig << endl;

    set_polys_out_of_range(ii, Tn, Tnd, dx_orig, x, order, inner_cutoff, exprlen, dx_dr);
  }
}

#include "chimesFF_kokkos_impl.h"
#endif
