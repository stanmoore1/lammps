/*
  ChIMES Calculator
  Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
  Contributing Author: Stan Moore (2025)
*/

#include<array>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<algorithm>
#include<cmath>
#include<map>

using namespace std;

#include "memory_kokkos.h"

/* ---------------------------------------------------------------------- */

template<class DeviceType>
chimesFFKokkos<DeviceType>::chimesFFKokkos() : chimesFF()
{
 eflag = vflag = 1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
chimesFFKokkos<DeviceType>::~chimesFFKokkos()
{

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void chimesFFKokkos<DeviceType>::read_parameters(string paramfile)
{
  int size, max_j, max_k;

  chimesFF::read_parameters(paramfile);

  // poly_orders

  size = poly_orders.size();

  for (int i = 0; i < size; i++)
    d_poly_orders[i] = poly_orders[i];


  // morse_var

  size = morse_var.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_morse_var,"chimesFF:morse_var",size);

  auto h_morse_var = Kokkos::create_mirror_view(d_morse_var);

  for (int i = 0; i < size; i++)
    h_morse_var[i] = morse_var[i];

  Kokkos::deep_copy(d_morse_var,h_morse_var);
  c_morse_var = d_morse_var;


  // penalty_params

  d_penalty_params[0] = penalty_params[0];
  d_penalty_params[1] = penalty_params[1];


  // ncoeffs_2b

  size = ncoeffs_2b.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_ncoeffs_2b,"chimesFF:ncoeffs_2b",size);

  auto h_ncoeffs_2b = Kokkos::create_mirror_view(d_ncoeffs_2b);

  for (int i = 0; i < size; i++)
    h_ncoeffs_2b[i] = ncoeffs_2b[i];

  Kokkos::deep_copy(d_ncoeffs_2b,h_ncoeffs_2b);
  c_ncoeffs_2b = d_ncoeffs_2b;


  // chimes_2b_pows

  size = chimes_2b_pows.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_2b_pows[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_2b_pows,"chimesFF:chimes_2b_pows",size,max_j);

  auto h_chimes_2b_pows = Kokkos::create_mirror_view(d_chimes_2b_pows);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_2b_pows[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_2b_pows(i,j) = chimes_2b_pows[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_2b_pows,h_chimes_2b_pows);
  c_chimes_2b_pows = d_chimes_2b_pows;

  // chimes_2b_params

  size = chimes_2b_params.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_2b_params[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_2b_params,"chimesFF:chimes_2b_params",size,max_j);

  auto h_chimes_2b_params = Kokkos::create_mirror_view(d_chimes_2b_params);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_2b_params[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_2b_params(i,j) = chimes_2b_params[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_2b_params,h_chimes_2b_params);
  c_chimes_2b_params = d_chimes_2b_params;

  // chimes_2b_cutoff

  size = chimes_2b_cutoff.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_2b_cutoff,"chimesFF:chimes_2b_cutoff",size);

  auto h_chimes_2b_cutoff = Kokkos::create_mirror_view(d_chimes_2b_cutoff);

  for (int i = 0; i < size; i++) {
    h_chimes_2b_cutoff(i,0) = chimes_2b_cutoff[i][0];
    h_chimes_2b_cutoff(i,1) = chimes_2b_cutoff[i][1];
  }

  Kokkos::deep_copy(d_chimes_2b_cutoff,h_chimes_2b_cutoff);
  c_chimes_2b_cutoff = d_chimes_2b_cutoff;

  // ncoeffs_3b

  size = ncoeffs_3b.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_ncoeffs_3b,"chimesFF:ncoeffs_3b",size);

  auto h_ncoeffs_3b = Kokkos::create_mirror_view(d_ncoeffs_3b);

  for (int i = 0; i < size; i++)
    h_ncoeffs_3b[i] = ncoeffs_3b[i];

  Kokkos::deep_copy(d_ncoeffs_3b,h_ncoeffs_3b);
  c_ncoeffs_3b = d_ncoeffs_3b;


  // chimes_3b_powers
  //
  // Stored transposed to [ntrips][constit. pair][nparams] (coefficient index
  // innermost) with the coefficient dimension padded, so the ThreadVectorRange
  // reduction in poly_3B reads consecutive coefficients across lanes as
  // coalesced, aligned loads (mirrors the chimes_3b_params layout).

  size = chimes_3b_powers.size();
  max_j = 0;
  max_k = 0;
  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_powers[i].size();
    max_j = MAX(max_j,size_j);
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_powers[i][j].size();
      max_k = MAX(max_k,size_k);
    }
  }

  // Pad the (innermost) coefficient dimension so each pair row starts aligned
  const int max_j_pad_3b_pow = ((max_j + CHIMES_PARAM_PAD - 1) / CHIMES_PARAM_PAD) * CHIMES_PARAM_PAD;

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_powers,"chimesFF:chimes_3b_powers",size,max_k,max_j_pad_3b_pow);

  auto h_chimes_3b_powers = Kokkos::create_mirror_view(d_chimes_3b_powers);
  Kokkos::deep_copy(h_chimes_3b_powers, 0); // zero-fill the padding

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_powers[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_powers[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_3b_powers(i,k,j) = chimes_3b_powers[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_3b_powers,h_chimes_3b_powers);
  c_chimes_3b_powers = d_chimes_3b_powers;


  // chimes_3b_params

  size = chimes_3b_params.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_3b_params[i].size());

  // Pad the coefficient dimension so each triplet row starts aligned (coalesced reads)
  const int max_j_pad_3b = ((max_j + CHIMES_PARAM_PAD - 1) / CHIMES_PARAM_PAD) * CHIMES_PARAM_PAD;

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_params,"chimesFF:chimes_3b_params",size,max_j_pad_3b);

  auto h_chimes_3b_params = Kokkos::create_mirror_view(d_chimes_3b_params);
  Kokkos::deep_copy(h_chimes_3b_params, 0.0); // zero-fill the padding

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_params[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_3b_params(i,j) = chimes_3b_params[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_3b_params,h_chimes_3b_params);
  c_chimes_3b_params = d_chimes_3b_params;


  // chimes_3b_cutoff

  size = chimes_3b_cutoff.size();
  max_j = 0;
  max_k = 0;
  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_cutoff[i].size();
    max_j = MAX(max_j,size_j);
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_cutoff[i][j].size();
      max_k = MAX(max_k,size_k);
    }
  }

  // last 2 indices are switched from the CPU code

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_cutoff,"chimesFF:chimes_3b_cutoff",size,max_k,max_j);

  auto h_chimes_3b_cutoff = Kokkos::create_mirror_view(d_chimes_3b_cutoff);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_cutoff[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_cutoff[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_3b_cutoff(i,k,j) = chimes_3b_cutoff[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_3b_cutoff,h_chimes_3b_cutoff);
  c_chimes_3b_cutoff = d_chimes_3b_cutoff;


  // ncoeffs_4b

  size = ncoeffs_4b.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_ncoeffs_4b,"chimesFF:ncoeffs_4b",size);

  auto h_ncoeffs_4b = Kokkos::create_mirror_view(d_ncoeffs_4b);

  for (int i = 0; i < size; i++)
    h_ncoeffs_4b[i] = ncoeffs_4b[i];

  Kokkos::deep_copy(d_ncoeffs_4b,h_ncoeffs_4b);
  c_ncoeffs_4b = d_ncoeffs_4b;


  // chimes_4b_powers
  //
  // Stored transposed to [nquads][constit. pair][nparams] (coefficient index
  // innermost) with the coefficient dimension padded, so the ThreadVectorRange
  // reduction in poly_4B reads consecutive coefficients across lanes as
  // coalesced, aligned loads (mirrors the chimes_4b_params layout).

  size = chimes_4b_powers.size();
  max_j = 0;
  max_k = 0;
  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_powers[i].size();
    max_j = MAX(max_j,size_j);
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_powers[i][j].size();
      max_k = MAX(max_k,size_k);
    }
  }

  // Pad the (innermost) coefficient dimension so each pair row starts aligned
  const int max_j_pad_4b_pow = ((max_j + CHIMES_PARAM_PAD - 1) / CHIMES_PARAM_PAD) * CHIMES_PARAM_PAD;

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_powers,"chimesFF:chimes_4b_powers",size,max_k,max_j_pad_4b_pow);

  auto h_chimes_4b_powers = Kokkos::create_mirror_view(d_chimes_4b_powers);
  Kokkos::deep_copy(h_chimes_4b_powers, 0); // zero-fill the padding

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_powers[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_powers[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_4b_powers(i,k,j) = chimes_4b_powers[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_4b_powers,h_chimes_4b_powers);
  c_chimes_4b_powers = d_chimes_4b_powers;


  // chimes_4b_params

  size = chimes_4b_params.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_4b_params[i].size());

  // Pad the coefficient dimension so each quadruplet row starts aligned (coalesced reads)
  const int max_j_pad_4b = ((max_j + CHIMES_PARAM_PAD - 1) / CHIMES_PARAM_PAD) * CHIMES_PARAM_PAD;

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_params,"chimesFF:chimes_4b_params",size,max_j_pad_4b);

  auto h_chimes_4b_params = Kokkos::create_mirror_view(d_chimes_4b_params);
  Kokkos::deep_copy(h_chimes_4b_params, 0.0); // zero-fill the padding

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_params[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_4b_params(i,j) = chimes_4b_params[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_4b_params,h_chimes_4b_params);
  c_chimes_4b_params = d_chimes_4b_params;


  // chimes_4b_cutoff

  size = chimes_4b_cutoff.size();
  max_j = 0;
  max_k = 0;
  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_cutoff[i].size();
    max_j = MAX(max_j,size_j);
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_cutoff[i][j].size();
      max_k = MAX(max_k,size_k);
    }
  }

  // last 2 indices are switched from the CPU code

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_cutoff,"chimesFF:chimes_4b_cutoff",size,max_k,max_j);

  auto h_chimes_4b_cutoff = Kokkos::create_mirror_view(d_chimes_4b_cutoff);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_cutoff[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_cutoff[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_4b_cutoff(i,k,j) = chimes_4b_cutoff[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_4b_cutoff,h_chimes_4b_cutoff);
  c_chimes_4b_cutoff = d_chimes_4b_cutoff;


  // energy_offsets

  size = energy_offsets.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_energy_offsets,"chimesFF:energy_offsets",size);

  auto h_energy_offsets = Kokkos::create_mirror_view(d_energy_offsets);

  for (int i = 0; i < size; i++)
    h_energy_offsets[i] = energy_offsets[i];

  Kokkos::deep_copy(d_energy_offsets,h_energy_offsets);
  c_energy_offsets = d_energy_offsets;


  // atom_int_pair_map

  size = atom_int_pair_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_pair_map,"chimesFF:atom_int_pair_map",size);

  auto h_atom_int_pair_map = Kokkos::create_mirror_view(d_atom_int_pair_map);

  for (int i = 0; i < size; i++)
    h_atom_int_pair_map[i] = atom_int_pair_map[i];

  Kokkos::deep_copy(d_atom_int_pair_map,h_atom_int_pair_map);
  c_atom_int_pair_map = d_atom_int_pair_map;


  // atom_int_trip_map

  size = atom_int_trip_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_trip_map,"chimesFF:",size);

  auto h_atom_int_trip_map = Kokkos::create_mirror_view(d_atom_int_trip_map);

  for (int i = 0; i < size; i++)
    h_atom_int_trip_map[i] = atom_int_trip_map[i];

  Kokkos::deep_copy(d_atom_int_trip_map,h_atom_int_trip_map);
  c_atom_int_trip_map = d_atom_int_trip_map;


  // atom_int_quad_map

  size = atom_int_quad_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_quad_map,"chimesFF:",size);

  auto h_atom_int_quad_map = Kokkos::create_mirror_view(d_atom_int_quad_map);

  for (int i = 0; i < size; i++)
    h_atom_int_quad_map[i] = atom_int_quad_map[i];

  Kokkos::deep_copy(d_atom_int_quad_map,h_atom_int_quad_map);
  c_atom_int_quad_map = d_atom_int_quad_map;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::set_polys_out_of_range(KK_FLOAT* Tn, KK_FLOAT* Tnd, KK_FLOAT dx, KK_FLOAT x, int poly_order, KK_FLOAT inner_cutoff, KK_FLOAT exprlen, KK_FLOAT dx_dr) const
{
  //  Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd) when dx is < inner_cutoff.
  //  Tnd is the derivative with respect to the interatomic distance, not the transformed distance (x).
  //	
  //  The derivative Tnd is continuously set to zero inside the cutoff.
  //  The exponential smoothing distance is set to ChimesFF::inner_smooth_distance.
  //  x, exprlen, and dx_dr are evaluated at the inner cutoff.
  //	
  //  dx is the pair distance, which is assumed to be less than inner_cutoff.
  Tn[0] = 1.0;
  Tn[1] = x;

  // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

  Tnd[0] = 1.0;
  Tnd[1] = 2.0 * x;

  // Use recursion to set up the higher n-value Tn and Tnd's
  for (int i = 2; i <= poly_order; i++) {
    Tn[i] = 2.0 * x * Tn[i-1] - Tn[i-2];
    Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
  }

  // Now multiply by n to convert Tnd's to actual derivatives of Tn

  for (int i = poly_order; i >= 1; i--)
    Tnd[i] = i * dx_dr * Tnd[i-1];

  Tnd[0] = 0.0;

  // Exponential damping of the derivative.
  const KK_FLOAT damp_fac = exp((dx-inner_cutoff) / inner_smooth_distance);

  // Correct Tn outside of the range using the damping factor.
  for (int i = 0 ; i <= poly_order ; i++) {
    Tn[i] += inner_smooth_distance * (damp_fac-1.0)  * Tnd[i];
    Tnd[i] *= damp_fac;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT chimesFFKokkos<DeviceType>::dr2_3B(const KK_FLOAT *dr2, int i, int j, int k, int l) const
{
  // Access the dr2 distance tensor for a 3 body interaction

  return(dr2[i*CHDIM*3*CHDIM + j*3*CHDIM + k*CHDIM + l]);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT chimesFFKokkos<DeviceType>::dr2_4B(const KK_FLOAT *dr2, int i, int j, int k, int l) const
{
  // Access the dr2 distance tensor for a 4 body interaction

  return(dr2[i*CHDIM*6*CHDIM + j*6*CHDIM + k*CHDIM + l]);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::init_distance_tensor(KK_FLOAT *dr2, const KK_FLOAT* dr, int npairs) const
{
  for (int i = 0; i < npairs; i++ )
    for (int j = 0; j < CHDIM; j++ )
      for (int k = 0; k < npairs; k++ )
        for (int l = 0; l < CHDIM; l++ )
          dr2[i* CHDIM * npairs * CHDIM + j * npairs * CHDIM + k * CHDIM + l] = dr[i*CHDIM+j] * dr[k*CHDIM+l];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_1B(const int typ_idx, KK_FLOAT & energy ) const
{
  // Compute 1b (input: a single atom type index... outputs (updates) energy

  energy += c_energy_offsets[typ_idx];
}

/* ---------------------------------------------------------------------- */

// Overload for calls from LAMMPS

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_2B(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar;
  compute_2B(dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_2B(const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT& force_scalar) const
{
  // Compute 2b (input: 2 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx: Scalar (pair distance)
  // dr: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]  *note
  // Energy: Scalar; energy for interaction set
  // Tmp: Temporary storage for calculation.

  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force is a packed array of coordinates.

  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  KK_FLOAT fcut;
  KK_FLOAT fcutderiv;

  // Use references for readability

  KK_FLOAT Tn[MAX_2B_POLY];
  KK_FLOAT Tnd[MAX_2B_POLY];

  const int pair_idx = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);

  //if (dx >= c_chimes_2b_cutoff(pair_idx,1)) return;

  set_cheby_polys(Tn, Tnd, dx, c_morse_var[pair_idx], c_chimes_2b_cutoff(pair_idx,0), c_chimes_2b_cutoff(pair_idx,1), d_poly_orders[0]);

  get_fcut(dx, c_chimes_2b_cutoff(pair_idx,1), fcut, fcutderiv);

  KK_FLOAT poly, dpoly_dx;

  poly_2B(poly, dpoly_dx, c_ncoeffs_2b[pair_idx], pair_idx, Tn, Tnd);

  //const KK_FLOAT dx_inv = (dx > 0.0 ) ? 1.0 / dx : 1e20;

  if (eflag)
    energy += poly * fcut;

  force_scalar = (fcut * dpoly_dx + fcutderiv * poly) / dx;

  force[0] += force_scalar * dr[0];
  force[1] += force_scalar * dr[1];
  force[2] += force_scalar * dr[2];

  force[CHDIM+0] -= force_scalar * dr[0];
  force[CHDIM+1] -= force_scalar * dr[1];
  force[CHDIM+2] -= force_scalar * dr[2];

  // xx xy xz yy yz zz
  // 0  1  2  3  4  5

  // xx xy xz yx yy yz zx zy zz
  // 0  1  2  3  4  5  6  7  8
  // *           *           *

  if (vflag) {
    stress[0] -= force_scalar * dr[0] * dr[0]; // xx tensor component
    stress[1] -= force_scalar * dr[0] * dr[1]; // xy tensor component
    stress[2] -= force_scalar * dr[0] * dr[2]; // xz tensor component
    stress[3] -= force_scalar * dr[1] * dr[1]; // yy tensor component
    stress[4] -= force_scalar * dr[1] * dr[2]; // yz tensor component
    stress[5] -= force_scalar * dr[2] * dr[2]; // zz tensor component
  }

  KK_FLOAT E_penalty = 0.0;
  get_penalty(dx, pair_idx, E_penalty, force_scalar);

  if (E_penalty > 0.0 )
  {
    if (eflag)
      energy += E_penalty;

    force_scalar /= dx;

    // Note: force_scalar is negative (LEF) 7/30/21

    force[0] += force_scalar * dr[0];
    force[1] += force_scalar * dr[1];
    force[2] += force_scalar * dr[2];

    force[CHDIM+0] -= force_scalar * dr[0];
    force[CHDIM+1] -= force_scalar * dr[1];
    force[CHDIM+2] -= force_scalar * dr[2];

    // Update stress according to penalty force. (LEF) 07/30/21

    if (vflag) {
      stress[0] -= force_scalar * dr[0] * dr[0]; // xx tensor component
      stress[1] -= force_scalar * dr[0] * dr[1]; // xy tensor component
      stress[2] -= force_scalar * dr[0] * dr[2]; // xz tensor component
      stress[3] -= force_scalar * dr[1] * dr[1]; // yy tensor component
      stress[4] -= force_scalar * dr[1] * dr[2]; // yz tensor component
      stress[5] -= force_scalar * dr[2] * dr[2]; // zz tensor component
    }
  }
}

/* ---------------------------------------------------------------------- */

// Overload for calls from LAMMPS

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_3B(const t_team& team, KK_FLOAT* scratch, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar[3];
  compute_3B(team, scratch, dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_3B(const t_team& team, KK_FLOAT* scratch, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar) const
{
  // Compute 3b (input: 3 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx_ij: Scalar (pair distance)
  // dr_ij: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]
  // Energy: Scalar; energy for interaction set
  // Tmp: Temporary storage for 3-body interactions.

  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force and dr are packed vectors of coordinates.

  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  constexpr int natoms = 3;                   // Number of atoms in an interaction set
  constexpr int npairs = natoms*(natoms-1)/2; // Number of pairs in an interaction set

  // Cache the Chebyshev arrays (Tn/Tnd) in per-team scratch memory rather than
  // recomputing them redundantly on every vector lane (mirrors the Kokkos SNAP
  // approach of staging recurrence intermediates in scratch). All vector lanes
  // of the team see the same scratch pointers; the arrays are built once (below)
  // and read by the ThreadVectorRange coefficient reduction.

  // scratch (caller-allocated) holds this cluster's 2*npairs*MAX_3B_POLY floats.
  KK_FLOAT* Tn_ij  = scratch + 0 * MAX_3B_POLY;
  KK_FLOAT* Tn_ik  = scratch + 1 * MAX_3B_POLY;
  KK_FLOAT* Tn_jk  = scratch + 2 * MAX_3B_POLY;
  KK_FLOAT* Tnd_ij = scratch + 3 * MAX_3B_POLY;
  KK_FLOAT* Tnd_ik = scratch + 4 * MAX_3B_POLY;
  KK_FLOAT* Tnd_jk = scratch + 5 * MAX_3B_POLY;

  // Avoid allocating vector quantities.  Heap memory allocation is slow on the GPU.
  // fixed-length C arrays are allocated on the stack

  KK_FLOAT fcut[npairs];
  KK_FLOAT fcutderiv[npairs];

  const int type_idx = typ_idxs[0]*natmtyps*natmtyps + typ_idxs[1]*natmtyps + typ_idxs[2];
  const int tripidx = c_atom_int_trip_map[type_idx];

  //if (tripidx < 0) // Skipping an excluded interaction
  //  return;

  // Check whether cutoffs are within allowed ranges
  //auto c_mapped_pair_idx = c_pair_int_trip_map[type_idx];

  const KK_FLOAT cutoff_00 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,0),0);
  const KK_FLOAT cutoff_0 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,0),1);

  //if (dx[0] >= cutoff_0) // ij
  //  return;

  const KK_FLOAT cutoff_01 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,1),0);
  const KK_FLOAT cutoff_1 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,1),1);

  //if (dx[1] >= cutoff_1) // ik
  //  return;

  const KK_FLOAT cutoff_02 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,2),0);
  const KK_FLOAT cutoff_2 = c_chimes_3b_cutoff(tripidx,c_pair_int_trip_map(type_idx,2),1);

  //if (dx[2] >= cutoff_2) // jk
  //  return;

 const int pair_type_1 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);
 const int pair_type_2 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[2]);
 const int pair_type_3 = c_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[2]);
 const int order = d_poly_orders[1];

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

#ifdef USE_DISTANCE_TENSOR
  // Tensor product of displacement vectors

  KK_FLOAT dr2[CHDIM*CHDIM*npairs*npairs];
  if (vflag)
    init_distance_tensor(dr2, dr, npairs);
#endif

  // Set up the polynomials into shared scratch, distributing the npairs
  // independent Chebyshev recurrences across the team's vector lanes (one pair
  // per lane) rather than computing them all on a single lane. Barrier afterward
  // so all vector lanes can read them in the reduction below. On host backends
  // (vector length 1) the lanes collapse and the pairs are built in order,
  // reproducing the original serial setup.

  KK_FLOAT* const Tn_p[npairs]  = { Tn_ij,  Tn_ik,  Tn_jk };
  KK_FLOAT* const Tnd_p[npairs] = { Tnd_ij, Tnd_ik, Tnd_jk };
  const KK_FLOAT morse_p[npairs] = { c_morse_var[pair_type_1], c_morse_var[pair_type_2], c_morse_var[pair_type_3] };
  const KK_FLOAT inner_p[npairs] = { cutoff_00, cutoff_01, cutoff_02 };
  const KK_FLOAT outer_p[npairs] = { cutoff_0,  cutoff_1,  cutoff_2 };

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, npairs), [&] (const int p) {
    set_cheby_polys(Tn_p[p], Tnd_p[p], dx[p], morse_p[p], inner_p[p], outer_p[p], order);
  });
  team.team_barrier();

  // Set up the smoothing functions

  get_fcut(dx[0], cutoff_0, fcut[0], fcutderiv[0]);
  get_fcut(dx[1], cutoff_1, fcut[1], fcutderiv[1]);
  get_fcut(dx[2], cutoff_2, fcut[2], fcutderiv[2]);
  const KK_FLOAT fcut_all = fcut[0] * fcut[1] * fcut[2];

  KK_FLOAT poly, dpoly_dx[npairs];

  // Start the force/stress/energy calculation

  if (!dense_coeffs) {
    poly_3B(team, poly, dpoly_dx, c_ncoeffs_3b[tripidx], tripidx, type_idx,
            Tn_ij, Tn_ik, Tn_jk, Tnd_ij, Tnd_ik, Tnd_jk);
  } else {

    // JIT evaluation of the chebyshev polynomial and its derivatives
    int inv_mapped_pair[npairs];

    for (int j = 0; j < npairs; j++)
      inv_mapped_pair[c_pair_int_trip_map(type_idx,j)] = j;

    KK_FLOAT* Tn[npairs];
    KK_FLOAT* Tnd[npairs];

    for (int j = 0; j < npairs; j++) {
      switch (inv_mapped_pair[j]) {
        case 0:
          Tn[j] = &Tn_ij[0];
          Tnd[j] = &Tnd_ij[0];
          break;
        case 1:
          Tn[j] = &Tn_ik[0];
          Tnd[j] = &Tnd_ik[0];
          break;
        case 2:
          Tn[j] = &Tn_jk[0];
          Tnd[j] = &Tnd_jk[0];
          break;
        default:
          Kokkos::abort("Bad inverse pair mapping found");
      }
    }

    poly_3B_dense(team, poly, dpoly_dx[inv_mapped_pair[0]], dpoly_dx[inv_mapped_pair[1]],
                  dpoly_dx[inv_mapped_pair[2]], c_ncoeffs_3b[tripidx], tripidx,
                  Tn[0], Tn[1], Tn[2], Tnd[0], Tnd[1], Tnd[2]);
  }

  if (eflag)
    energy += poly * fcut_all;

  force_scalar[0] = (fcut_all * dpoly_dx[0] + fcutderiv[0] * fcut[1] * fcut[2] * poly) / dx[0];
  force_scalar[1] = (fcut_all * dpoly_dx[1] + fcutderiv[1] * fcut[0] * fcut[2] * poly) / dx[1];
  force_scalar[2] = (fcut_all * dpoly_dx[2] + fcutderiv[2] * fcut[0] * fcut[1] * poly) / dx[2];

  const KK_FLOAT &fscalar_0 = force_scalar[0];
  const KK_FLOAT &fscalar_1 = force_scalar[1];
  const KK_FLOAT &fscalar_2 = force_scalar[2];

  // Accumulate forces/stresses on/from the ij pair

  force[0] += fscalar_0 * dr[0];
  force[1] += fscalar_0 * dr[1];
  force[2] += fscalar_0 * dr[2];

  force[CHDIM+0] -= fscalar_0 * dr[0];
  force[CHDIM+1] -= fscalar_0 * dr[1];
  force[CHDIM+2] -= fscalar_0 * dr[2];

  // dr2_3B looks like a function call, but the optimizer should remove it entirely
  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    // New stress code

    stress[0] -= fscalar_0 * dr2_3B(dr2,0,0,0,0); // xx tensor component
    stress[1] -= fscalar_0 * dr2_3B(dr2,0,0,0,1); // xy tensor component
    stress[2] -= fscalar_0 * dr2_3B(dr2,0,0,0,2); // xz tensor component
    stress[3] -= fscalar_0 * dr2_3B(dr2,0,1,0,1); // yy tensor component
    stress[4] -= fscalar_0 * dr2_3B(dr2,0,1,0,2); // yz tensor component
    stress[5] -= fscalar_0 * dr2_3B(dr2,0,2,0,2); // zz tensor component

#else
    stress[0] -= fscalar_0 * dr[0] * dr[0]; // xx tensor component
    stress[1] -= fscalar_0 * dr[0] * dr[1]; // xy tensor component
    stress[2] -= fscalar_0 * dr[0] * dr[2]; // xz tensor component
    stress[3] -= fscalar_0 * dr[1] * dr[1]; // yy tensor component
    stress[4] -= fscalar_0 * dr[1] * dr[2]; // yz tensor component
    stress[5] -= fscalar_0 * dr[2] * dr[2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the ik pair

  force[0] += fscalar_1 * dr[CHDIM+0];
  force[1] += fscalar_1 * dr[CHDIM+1];
  force[2] += fscalar_1 * dr[CHDIM+2];

  force[2*CHDIM+0] -= fscalar_1 * dr[CHDIM+0];
  force[2*CHDIM+1] -= fscalar_1 * dr[CHDIM+1];
  force[2*CHDIM+2] -= fscalar_1 * dr[CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_1 * dr2_3B(dr2,1,0,1,0); // xx tensor component
    stress[1] -= fscalar_1 * dr2_3B(dr2,1,0,1,1); // xy tensor component
    stress[2] -= fscalar_1 * dr2_3B(dr2,1,0,1,2); // xz tensor component
    stress[3] -= fscalar_1 * dr2_3B(dr2,1,1,1,1); // yy tensor component
    stress[4] -= fscalar_1 * dr2_3B(dr2,1,1,1,2); // yz tensor component
    stress[5] -= fscalar_1 * dr2_3B(dr2,1,2,1,2); // zz tensor component
#else
    stress[0] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_1 * dr[CHDIM+1] * dr[CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_1 * dr[CHDIM+1] * dr[CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_1 * dr[CHDIM+2] * dr[CHDIM+2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the jk pair

  force[CHDIM+0] += fscalar_2 * dr[2*CHDIM+0];
  force[CHDIM+1] += fscalar_2 * dr[2*CHDIM+1];
  force[CHDIM+2] += fscalar_2 * dr[2*CHDIM+2];

  force[2*CHDIM+0] -= fscalar_2 * dr[2*CHDIM+0];
  force[2*CHDIM+1] -= fscalar_2 * dr[2*CHDIM+1];
  force[2*CHDIM+2] -= fscalar_2 * dr[2*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_2 * dr2_3B(dr2,2,0,2,0); // xx tensor component
    stress[1] -= fscalar_2 * dr2_3B(dr2,2,0,2,1); // xy tensor component
    stress[2] -= fscalar_2 * dr2_3B(dr2,2,0,2,2); // xz tensor component
    stress[3] -= fscalar_2 * dr2_3B(dr2,2,1,2,1); // yy tensor component
    stress[4] -= fscalar_2 * dr2_3B(dr2,2,1,2,2); // yz tensor component
    stress[5] -= fscalar_2 * dr2_3B(dr2,2,2,2,2); // zz tensor component
#else
    stress[0] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_2 * dr[2*CHDIM+1] * dr[2*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_2 * dr[2*CHDIM+1] * dr[2*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_2 * dr[2*CHDIM+2] * dr[2*CHDIM+2]; // zz tensor component
#endif
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B(const t_team& team, KK_FLOAT* scratch, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar[6];
  compute_4B(team, scratch, dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B(const t_team& team, KK_FLOAT* scratch, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar) const
{
  // Compute 3b (input: 3 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx_ij: Scalar (pair distance)
  // dr_ij: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]
  // Energy: Scalar; energy for interaction set
  // Tmp: Structure containing temporary data.
  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force and dr are packed vectors of coordinates.
  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  constexpr int natoms = 4;                     // Number of atoms in an interaction set
  constexpr int npairs = natoms*(natoms-1)/2;    // Number of pairs in an interaction set

  KK_FLOAT fcut[npairs];
  KK_FLOAT fcutderiv[npairs];

  // Cache the Chebyshev arrays (Tn/Tnd) in per-team scratch memory rather than
  // recomputing them redundantly on every vector lane (mirrors Kokkos SNAP).

  // scratch (caller-allocated) holds this cluster's 2*npairs*MAX_4B_POLY floats.
  KK_FLOAT* Tn_ij  = scratch +  0 * MAX_4B_POLY;
  KK_FLOAT* Tn_ik  = scratch +  1 * MAX_4B_POLY;
  KK_FLOAT* Tn_il  = scratch +  2 * MAX_4B_POLY;
  KK_FLOAT* Tn_jk  = scratch +  3 * MAX_4B_POLY;
  KK_FLOAT* Tn_jl  = scratch +  4 * MAX_4B_POLY;
  KK_FLOAT* Tn_kl  = scratch +  5 * MAX_4B_POLY;
  KK_FLOAT* Tnd_ij = scratch +  6 * MAX_4B_POLY;
  KK_FLOAT* Tnd_ik = scratch +  7 * MAX_4B_POLY;
  KK_FLOAT* Tnd_il = scratch +  8 * MAX_4B_POLY;
  KK_FLOAT* Tnd_jk = scratch +  9 * MAX_4B_POLY;
  KK_FLOAT* Tnd_jl = scratch + 10 * MAX_4B_POLY;
  KK_FLOAT* Tnd_kl = scratch + 11 * MAX_4B_POLY;

  const int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
      + typ_idxs[1]*natmtyps*natmtyps + typ_idxs[2]*natmtyps + typ_idxs[3];

  const int quadidx = c_atom_int_quad_map[idx];

  //if (quadidx < 0) // Skipping an excluded interaction
  //  return;

  //auto c_mapped_pair_idx = c_pair_int_quad_map[idx];

  // Check whether cutoffs are within allowed ranges
/*
  for (int i=0; i<npairs; i++)
      if (dx[i] >= c_chimes_4b_cutoff(quadidx,1,c_pair_int_quad_map(idx,i)))
          return;
*/
  // These speed up fcut calculations by a LOT

  const KK_FLOAT cutoff_00 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,0),0);
  const KK_FLOAT cutoff_0 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,0),1);

  //if (dx[0] >= cutoff_0) // ij
  //  return;

  const KK_FLOAT cutoff_01 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,1),0);
  const KK_FLOAT cutoff_1 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,1),1);

  //if (dx[1] >= cutoff_1) // ik
  //  return;

  const KK_FLOAT cutoff_02 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,2),0);
  const KK_FLOAT cutoff_2 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,2),1);

  //if (dx[2] >= cutoff_2) // il
  //  return;

  const KK_FLOAT cutoff_03 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,3),0);
  const KK_FLOAT cutoff_3 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,3),1);

  //if (dx[3] >= cutoff_3) // jk
  //  return;

  const KK_FLOAT cutoff_04 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,4),0);
  const KK_FLOAT cutoff_4 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,4),1);

  //if (dx[4] >= cutoff_4) // jl
  //  return;

  const KK_FLOAT cutoff_05 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,5),0);
  const KK_FLOAT cutoff_5 = c_chimes_4b_cutoff(quadidx,c_pair_int_quad_map(idx,5),1);

  //if (dx[5] >= cutoff_5) // kl
  //  return;

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

  const int pair_type_1 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);
  const int pair_type_2 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[2]);
  const int pair_type_3 = c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[3]);
  const int pair_type_4 = c_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[2]);
  const int pair_type_5 = c_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[3]);
  const int pair_type_6 = c_atom_int_pair_map(typ_idxs[2]*natmtyps + typ_idxs[3]);
  const int order = d_poly_orders[2];

  // Set up the polynomials into shared scratch, distributing the npairs
  // independent Chebyshev recurrences across the team's vector lanes (one pair
  // per lane) rather than computing them all on a single lane. On host backends
  // (vector length 1) the lanes collapse and the pairs are built in order,
  // reproducing the original serial setup.

  KK_FLOAT* const Tn_p[npairs]  = { Tn_ij,  Tn_ik,  Tn_il,  Tn_jk,  Tn_jl,  Tn_kl };
  KK_FLOAT* const Tnd_p[npairs] = { Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl };
  const KK_FLOAT morse_p[npairs] = { c_morse_var[pair_type_1], c_morse_var[pair_type_2], c_morse_var[pair_type_3],
                                     c_morse_var[pair_type_4], c_morse_var[pair_type_5], c_morse_var[pair_type_6] };
  const KK_FLOAT inner_p[npairs] = { cutoff_00, cutoff_01, cutoff_02, cutoff_03, cutoff_04, cutoff_05 };
  const KK_FLOAT outer_p[npairs] = { cutoff_0,  cutoff_1,  cutoff_2,  cutoff_3,  cutoff_4,  cutoff_5 };

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, npairs), [&] (const int p) {
    set_cheby_polys(Tn_p[p], Tnd_p[p], dx[p], morse_p[p], inner_p[p], outer_p[p], order);
  });
  team.team_barrier();

#ifdef USE_DISTANCE_TENSOR
  // Tensor product of displacement vectors

  KK_FLOAT dr2[CHDIM*CHDIM*npairs*npairs];
  if (vflag)
    init_distance_tensor(dr2, dr, npairs);
#endif


  // Set up the smoothing functions
/*
  for (int i=0; i<npairs; i++)
      get_fcut(dx[i], c_chimes_4b_cutoff(quadidx,1,c_pair_int_quad_map(idx,i)], fcut[i], fcutderiv[i));
*/

  get_fcut(dx[0], cutoff_0, fcut[0], fcutderiv[0]);
  get_fcut(dx[1], cutoff_1, fcut[1], fcutderiv[1]);
  get_fcut(dx[2], cutoff_2, fcut[2], fcutderiv[2]);
  get_fcut(dx[3], cutoff_3, fcut[3], fcutderiv[3]);
  get_fcut(dx[4], cutoff_4, fcut[4], fcutderiv[4]);
  get_fcut(dx[5], cutoff_5, fcut[5], fcutderiv[5]);

  // Product of all 6 fcuts, plus the six "leave-one-out" products fcut_5[j]
  // (product of every fcut except j) needed for the per-pair force scalars.
  // Computed with a forward/backward sweep using two running products instead
  // of recomputing a 5-fold product per pair: O(npairs) multiplies and only two
  // extra scalars (vs. the prior ~5*npairs multiplies). Division by fcut[j] is
  // avoided since a cutoff function can be zero at the outer cutoff.

  KK_FLOAT fcut_5[npairs];
  KK_FLOAT run = 1.0;
  for (int j = 0; j < npairs; j++) { fcut_5[j] = run; run *= fcut[j]; }   // prefix product
  const KK_FLOAT fcut_all = run;
  run = 1.0;
  for (int j = npairs - 1; j >= 0; j--) { fcut_5[j] *= run; run *= fcut[j]; } // x suffix product

  // Start the force/stress/energy calculation

  KK_FLOAT poly, dpoly_dx[npairs];

  //if (!dense_coeffs) {
  if (1) {
    poly_4B(team, poly, dpoly_dx, c_ncoeffs_4b[quadidx], quadidx, idx,
            Tn_ij, Tn_ik, Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik,
            Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  } else {
    // Dense evaluation of the chebyshev polynomial and its derivatives
    int inv_mapped_pair[npairs];

    for (int j = 0; j < npairs; j++) { inv_mapped_pair[c_pair_int_quad_map(idx,j)] = j; }

    KK_FLOAT* Tn[npairs];
    KK_FLOAT* Tnd[npairs];

    for (int j = 0; j < npairs; j++) {
      switch (inv_mapped_pair[j]) {
        case 0:
          Tn[j] = &Tn_ij[0];
          Tnd[j] = &Tnd_ij[0];
          break;
        case 1:
          Tn[j] = &Tn_ik[0];
          Tnd[j] = &Tnd_ik[0];
          break;
        case 2:
          Tn[j] = &Tn_il[0];
          Tnd[j] = &Tnd_il[0];
          break;
        case 3:
          Tn[j] = &Tn_jk[0];
          Tnd[j] = &Tnd_jk[0];
          break;
        case 4:
          Tn[j] = &Tn_jl[0];
          Tnd[j] = &Tnd_jl[0];
          break;
        case 5:
          Tn[j] = &Tn_kl[0];
          Tnd[j] = &Tnd_kl[0];
          break;
        default:
          Kokkos::abort("Bad inverse pair mapping found");
      }
    }

    poly_4B_dense(team, poly, dpoly_dx[inv_mapped_pair[0]], dpoly_dx[inv_mapped_pair[1]],
                  dpoly_dx[inv_mapped_pair[2]], dpoly_dx[inv_mapped_pair[3]],
                  dpoly_dx[inv_mapped_pair[4]], dpoly_dx[inv_mapped_pair[5]], d_ncoeffs_4b[quadidx],
                  quadidx, Tn[0], Tn[1], Tn[2], Tn[3], Tn[4], Tn[5],
                  Tnd[0], Tnd[1], Tnd[2], Tnd[3], Tnd[4], Tnd[5]);
  }

  if (eflag)
    energy += poly * fcut_all;

  for (int j = 0; j < npairs; j++)
    force_scalar[j] = (fcut_all * dpoly_dx[j] + fcutderiv[j] * fcut_5[j] * poly) / dx[j];

  const KK_FLOAT &fscalar_0 = force_scalar[0];
  const KK_FLOAT &fscalar_1 = force_scalar[1];
  const KK_FLOAT &fscalar_2 = force_scalar[2];
  const KK_FLOAT &fscalar_3 = force_scalar[3];
  const KK_FLOAT &fscalar_4 = force_scalar[4];
  const KK_FLOAT &fscalar_5 = force_scalar[5];

  // Accumulate forces/stresses on/from the ij pair

  force[0] += fscalar_0 * dr[0];
  force[1] += fscalar_0 * dr[1];
  force[2] += fscalar_0 * dr[2];

  force[CHDIM+0] -= fscalar_0 * dr[0];
  force[CHDIM+1] -= fscalar_0 * dr[1];
  force[CHDIM+2] -= fscalar_0 * dr[2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_0 * dr2_4B(dr2,0,0,0,0); // xx tensor component
    stress[1] -= fscalar_0 * dr2_4B(dr2,0,0,0,1); // xy tensor component
    stress[2] -= fscalar_0 * dr2_4B(dr2,0,0,0,2); // xz tensor component
    stress[3] -= fscalar_0 * dr2_4B(dr2,0,1,0,1); // yy tensor component
    stress[4] -= fscalar_0 * dr2_4B(dr2,0,1,0,2); // yz tensor component
    stress[5] -= fscalar_0 * dr2_4B(dr2,0,2,0,2); // zz tensor component
#else
    stress[0] -= fscalar_0 * dr[0] * dr[0]; // xx tensor component
    stress[1] -= fscalar_0 * dr[0] * dr[1]; // xy tensor component
    stress[2] -= fscalar_0 * dr[0] * dr[2]; // xz tensor component
    stress[3] -= fscalar_0 * dr[1] * dr[1]; // yy tensor component
    stress[4] -= fscalar_0 * dr[1] * dr[2]; // yz tensor component
    stress[5] -= fscalar_0 * dr[2] * dr[2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the ik pair

  force[0] += fscalar_1 * dr[CHDIM+0];
  force[1] += fscalar_1 * dr[CHDIM+1];
  force[2] += fscalar_1 * dr[CHDIM+2];
  force[2*CHDIM+0] -= fscalar_1 * dr[CHDIM+0];
  force[2*CHDIM+1] -= fscalar_1 * dr[CHDIM+1];
  force[2*CHDIM+2] -= fscalar_1 * dr[CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_1 * dr2_4B(dr2,1,0,1,0); // xx tensor component
    stress[1] -= fscalar_1 * dr2_4B(dr2,1,0,1,1); // xy tensor component
    stress[2] -= fscalar_1 * dr2_4B(dr2,1,0,1,2); // xz tensor component
    stress[3] -= fscalar_1 * dr2_4B(dr2,1,1,1,1); // yy tensor component
    stress[4] -= fscalar_1 * dr2_4B(dr2,1,1,1,2); // yz tensor component
    stress[5] -= fscalar_1 * dr2_4B(dr2,1,2,1,2); // zz tensor component
#else
    stress[0] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_1 * dr[CHDIM+0] * dr[CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_1 * dr[CHDIM+1] * dr[CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_1 * dr[CHDIM+1] * dr[CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_1 * dr[CHDIM+2] * dr[CHDIM+2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the il pair

  force[0] += fscalar_2 * dr[2*CHDIM+0];
  force[1] += fscalar_2 * dr[2*CHDIM+1];
  force[2] += fscalar_2 * dr[2*CHDIM+2];
  force[3*CHDIM+0] -= fscalar_2 * dr[2*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_2 * dr[2*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_2 * dr[2*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_2 * dr2_4B(dr2,2,0,2,0); // xx tensor component
    stress[1] -= fscalar_2 * dr2_4B(dr2,2,0,2,1); // xy tensor component
    stress[2] -= fscalar_2 * dr2_4B(dr2,2,0,2,2); // xz tensor component
    stress[3] -= fscalar_2 * dr2_4B(dr2,2,1,2,1); // yy tensor component
    stress[4] -= fscalar_2 * dr2_4B(dr2,2,1,2,2); // yz tensor component
    stress[5] -= fscalar_2 * dr2_4B(dr2,2,2,2,2); // zz tensor component
#else
    stress[0] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_2 * dr[2*CHDIM+0] * dr[2*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_2 * dr[2*CHDIM+1] * dr[2*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_2 * dr[2*CHDIM+1] * dr[2*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_2 * dr[2*CHDIM+2] * dr[2*CHDIM+2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the jk pair

  force[CHDIM+0] += fscalar_3 * dr[3*CHDIM+0];
  force[CHDIM+1] += fscalar_3 * dr[3*CHDIM+1];
  force[CHDIM+2] += fscalar_3 * dr[3*CHDIM+2];

  force[2*CHDIM+0] -= fscalar_3 * dr[3*CHDIM+0];
  force[2*CHDIM+1] -= fscalar_3 * dr[3*CHDIM+1];
  force[2*CHDIM+2] -= fscalar_3 * dr[3*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_3 * dr2_4B(dr2,3,0,3,0); // xx tensor component
    stress[1] -= fscalar_3 * dr2_4B(dr2,3,0,3,1); // xy tensor component
    stress[2] -= fscalar_3 * dr2_4B(dr2,3,0,3,2); // xz tensor component
    stress[3] -= fscalar_3 * dr2_4B(dr2,3,1,3,1); // yy tensor component
    stress[4] -= fscalar_3 * dr2_4B(dr2,3,1,3,2); // yz tensor component
    stress[5] -= fscalar_3 * dr2_4B(dr2,3,2,3,2); // zz tensor component
#else
    stress[0] -= fscalar_3 * dr[3*CHDIM+0] * dr[3*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_3 * dr[3*CHDIM+0] * dr[3*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_3 * dr[3*CHDIM+0] * dr[3*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_3 * dr[3*CHDIM+1] * dr[3*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_3 * dr[3*CHDIM+1] * dr[3*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_3 * dr[3*CHDIM+2] * dr[3*CHDIM+2]; // zz tensor component
#endif
  }

  // Accumulate forces/stresses on/from the jl pair

  force[CHDIM+0] += fscalar_4 * dr[4*CHDIM+0];
  force[CHDIM+1] += fscalar_4 * dr[4*CHDIM+1];
  force[CHDIM+2] += fscalar_4 * dr[4*CHDIM+2];

  force[3*CHDIM+0] -= fscalar_4 * dr[4*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_4 * dr[4*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_4 * dr[4*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_4 * dr2_4B(dr2,4,0,4,0); // xx tensor component
    stress[1] -= fscalar_4 * dr2_4B(dr2,4,0,4,1); // xy tensor component
    stress[2] -= fscalar_4 * dr2_4B(dr2,4,0,4,2); // xz tensor component
    stress[3] -= fscalar_4 * dr2_4B(dr2,4,1,4,1); // yy tensor component
    stress[4] -= fscalar_4 * dr2_4B(dr2,4,1,4,2); // yz tensor component
    stress[5] -= fscalar_4 * dr2_4B(dr2,4,2,4,2); // zz tensor component
#else
    stress[0] -= fscalar_4 * dr[4*CHDIM+0] * dr[4*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_4 * dr[4*CHDIM+0] * dr[4*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_4 * dr[4*CHDIM+0] * dr[4*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_4 * dr[4*CHDIM+1] * dr[4*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_4 * dr[4*CHDIM+1] * dr[4*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_4 * dr[4*CHDIM+2] * dr[4*CHDIM+2]; // zz tensor component
#endif
  }
  // Accumulate forces/stresses on/from the kl pair

  force[2*CHDIM+0] += fscalar_5 * dr[5*CHDIM+0];
  force[2*CHDIM+1] += fscalar_5 * dr[5*CHDIM+1];
  force[2*CHDIM+2] += fscalar_5 * dr[5*CHDIM+2];
  force[3*CHDIM+0] -= fscalar_5 * dr[5*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_5 * dr[5*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_5 * dr[5*CHDIM+2];

  if (vflag) {
#ifdef USE_DISTANCE_TENSOR
    stress[0] -= fscalar_5 * dr2_4B(dr2,5,0,5,0); // xx tensor component
    stress[1] -= fscalar_5 * dr2_4B(dr2,5,0,5,1); // xy tensor component
    stress[2] -= fscalar_5 * dr2_4B(dr2,5,0,5,2); // xz tensor component
    stress[3] -= fscalar_5 * dr2_4B(dr2,5,1,5,1); // yy tensor component
    stress[4] -= fscalar_5 * dr2_4B(dr2,5,1,5,2); // yz tensor component
    stress[5] -= fscalar_5 * dr2_4B(dr2,5,2,5,2); // zz tensor component
#else
    stress[0] -= fscalar_5 * dr[5*CHDIM+0] * dr[5*CHDIM+0]; // xx tensor component
    stress[1] -= fscalar_5 * dr[5*CHDIM+0] * dr[5*CHDIM+1]; // xy tensor component
    stress[2] -= fscalar_5 * dr[5*CHDIM+0] * dr[5*CHDIM+2]; // xz tensor component
    stress[3] -= fscalar_5 * dr[5*CHDIM+1] * dr[5*CHDIM+1]; // yy tensor component
    stress[4] -= fscalar_5 * dr[5*CHDIM+1] * dr[5*CHDIM+2]; // yz tensor component
    stress[5] -= fscalar_5 * dr[5*CHDIM+2] * dr[5*CHDIM+2]; // zz tensor component
#endif
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void chimesFFKokkos<DeviceType>::build_pair_int_quad_map()
// Build the pair maps for all possible quads.  Moved build_atom_and_pair_mappers out of the compute_XX routines
// to support GPU environment without string operations.
// This must be called prior to force evaluation.
{
  chimesFF::build_pair_int_quad_map();

  // pair_int_quad_map

  int size = pair_int_quad_map.size();
  int max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,pair_int_quad_map[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_pair_int_quad_map,"chimesFF:pair_int_quad_map",size,max_j);

  auto h_pair_int_quad_map = Kokkos::create_mirror_view(d_pair_int_quad_map);

  for (int i = 0; i < size; i++) {
    int size_j = pair_int_quad_map[i].size();
    for (int j = 0; j < size_j; j++) {
      h_pair_int_quad_map(i,j) = pair_int_quad_map[i][j];
    }
  }

  Kokkos::deep_copy(d_pair_int_quad_map,h_pair_int_quad_map);
  c_pair_int_quad_map = d_pair_int_quad_map;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void chimesFFKokkos<DeviceType>::build_pair_int_trip_map()
// Build the pair maps for all possible triplets.  Moved build_atom_and_pair_mappers out of the compute_XX routines
// to support GPU environment without string operations.
// This must be called prior to force evaluation.
{
  chimesFF::build_pair_int_trip_map();

  // pair_int_trip_map

  int size = pair_int_trip_map.size();
  int max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,pair_int_trip_map[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_pair_int_trip_map,"chimesFF:pair_int_trip_map",size,max_j);

  auto h_pair_int_trip_map = Kokkos::create_mirror_view(d_pair_int_trip_map);

  for (int i = 0; i < size; i++) {
    int size_j = pair_int_trip_map[i].size();
    for (int j = 0; j < size_j; j++) {
      h_pair_int_trip_map(i,j) = pair_int_trip_map[i][j];
    }
  }

  Kokkos::deep_copy(d_pair_int_trip_map,h_pair_int_trip_map);
  c_pair_int_trip_map = d_pair_int_trip_map;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_2B(KK_FLOAT &e, KK_FLOAT &f0, const int ncoeffs_2b, const int pair_idx,
                                         KK_FLOAT* Tn, KK_FLOAT* Tnd) const
// Compute the 2 body polynomial (e) and derivatives with respect to the pair distance (f0)
// (LEF) 3/11/26
{
  e = 0.0;
  f0 = 0.0;

  auto c_chimes_2b_params_pairidx = Kokkos::subview(c_chimes_2b_params,pair_idx,Kokkos::ALL);
  auto c_chimes_2b_pows_pairidx = Kokkos::subview(c_chimes_2b_pows,pair_idx,Kokkos::ALL);

  #pragma unroll
  for (int coeffs = 0; coeffs < ncoeffs_2b; coeffs++) {
    const KK_FLOAT coeff_val = c_chimes_2b_params_pairidx(coeffs);
    const int powerp1 = c_chimes_2b_pows_pairidx(coeffs) + 1;

    e += coeff_val * Tn[powerp1];

    f0 += coeff_val * Tnd[powerp1];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B(const t_team& team, KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_3b, int tripidx, int idx,
                                         KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_jk,
                                         KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_jk) const
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
//
// Hierarchical-parallelism port (mirrors Kokkos SNAP / poly_4B): the flat
// coefficient loop is distributed across the team's ThreadVectorRange and
// reduced into a single {energy, 3 pair-force derivatives} value. On host
// backends (vector length 1) this reproduces the original serial loop.
{
  const int trip_map_idx[3] = { c_pair_int_trip_map(idx,0),
                                c_pair_int_trip_map(idx,1),
                                c_pair_int_trip_map(idx,2) };

  auto c_chimes_3b_params_tripidx = Kokkos::subview(c_chimes_3b_params,tripidx,Kokkos::ALL);
  auto c_chimes_3b_powers_tripidx = Kokkos::subview(c_chimes_3b_powers,tripidx,Kokkos::ALL,Kokkos::ALL);

  constexpr int coeff_batch = std::is_same<DeviceType, LMPHostType>::value ? 1 : CHIMES_COEFF_BATCH;
  const int n_groups = (ncoeffs_3b + coeff_batch - 1) / coeff_batch;

  s_chimes_poly3 result;

  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, n_groups),
    [&] (const int t, s_chimes_poly3& upd) {
      Kokkos::Array<s_chimes_poly3, coeff_batch> acc;

      #pragma unroll
      for (int b = 0; b < coeff_batch; b++) {
        const int coeffs = t + b * n_groups;   // coalesced across lanes at fixed b
        if (coeffs < ncoeffs_3b) {
          const KK_FLOAT coeff = c_chimes_3b_params_tripidx(coeffs);

          const int powers[3] = { c_chimes_3b_powers_tripidx(trip_map_idx[0],coeffs),
                                  c_chimes_3b_powers_tripidx(trip_map_idx[1],coeffs),
                                  c_chimes_3b_powers_tripidx(trip_map_idx[2],coeffs) };

          acc[b].e  += coeff * Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_jk[powers[2]];
          acc[b].f0 += coeff * Tnd_ij[powers[0]] * Tn_ik[powers[1]] * Tn_jk[powers[2]];
          acc[b].f1 += coeff * Tnd_ik[powers[1]] * Tn_ij[powers[0]] * Tn_jk[powers[2]];
          acc[b].f2 += coeff * Tnd_jk[powers[2]] * Tn_ij[powers[0]] * Tn_ik[powers[1]];
        }
      }

      #pragma unroll
      for (int b = 0; b < coeff_batch; b++) upd += acc[b];
    }, Kokkos::Sum<s_chimes_poly3>(result));

  e = result.e;
  f[0] = result.f0;
  f[1] = result.f1;
  f[2] = result.f2;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B_dense(const t_team& team, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, int ncoeffs_3b, int tripidx,
                                               KK_FLOAT *Tn_ij, KK_FLOAT *Tn_ik, KK_FLOAT *Tn_jk,
                                               KK_FLOAT *Tnd_ij, KK_FLOAT *Tnd_ik, KK_FLOAT *Tnd_jk) const
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f0, f1, f2)
// (LEF) 4/02/26
{
  const int loop_style = 2;

  e = 0.0;
  f0 = 0.0;
  f1 = 0.0;
  f2 = 0.0;

  if (ncoeffs_3b == 0) return;

  int max_poly = 0;
  const int loop_max = 1000;
  int i = 0;
  for (; i < loop_max; i++) {
    if (i * i * i == ncoeffs_3b) {
      max_poly = i;
      break;
    }
  }
  if (i == loop_max) {
    Kokkos::abort("Bad number of 3 body coefficients for dense evaluation");
  }

  if (loop_style == 1) {
    poly_3B_dense_loop1(max_poly, e, f0, f1, f2, ncoeffs_3b, tripidx, Tn_ij, Tn_ik, Tn_jk,
                        Tnd_ij, Tnd_ik, Tnd_jk);
  } else if (loop_style == 2) {
    poly_3B_dense_loop2(team, max_poly, e, f0, f1, f2, ncoeffs_3b, tripidx, Tn_ij, Tn_ik, Tn_jk,
                        Tnd_ij, Tnd_ik, Tnd_jk);
  //} else if (loop_style == 3) {
  //  poly_3B_dense_loop3(max_poly, e, f0, f1, f2, ncoeffs_3b, tripidx, Tn_ij, Tn_ik, Tn_jk,
  //                      Tnd_ij, Tnd_ik, Tnd_jk);
  } else {
    Kokkos::abort("Error: bad 3 body dense loop style");
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B_dense_loop1(int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2,
                                   int ncoeffs_3b, int tripidx,
                                   KK_FLOAT *Tn_ij, KK_FLOAT *Tn_ik,
                                   KK_FLOAT *Tn_jk, KK_FLOAT *Tnd_ij,
                                   KK_FLOAT *Tnd_ik, KK_FLOAT *Tnd_jk) const
{
  auto c_chimes_3b_params_tripidx = Kokkos::subview(c_chimes_3b_params,tripidx,Kokkos::ALL);

  #pragma unroll
  for (int count = 0; count < ncoeffs_3b; count++) {
    int l = count / (max_poly * max_poly);
    //if (l >= max_poly) { cout << "Internal error: l > max_poly: " << l << "\n"; }
    int m = (count / max_poly) % max_poly;
    int n = count % max_poly;

    const KK_FLOAT coeff = c_chimes_3b_params_tripidx[count];
    if (coeff != 0.0) {
      const KK_FLOAT tn_ij = Tn_ij[l];
      const KK_FLOAT tnd_ij = Tnd_ij[l];
      const KK_FLOAT tn_ik = Tn_ik[m];
      const KK_FLOAT tnd_ik = Tnd_ik[m];
      const KK_FLOAT tn_jk = Tn_jk[n];
      const KK_FLOAT tnd_jk = Tnd_jk[n];

      e += coeff * tn_ij * tn_ik * tn_jk;
      f0 += coeff * tnd_ij * tn_ik * tn_jk;
      f1 += coeff * tnd_ik * tn_ij * tn_jk;
      f2 += coeff * tnd_jk * tn_ij * tn_ik;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B_dense_loop2(const t_team& team, int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2,
                                   int ncoeffs_3b, int tripidx,
                                   KK_FLOAT *Tn_ij, KK_FLOAT *Tn_ik,
                                   KK_FLOAT *Tn_jk, KK_FLOAT *Tnd_ij,
                                   KK_FLOAT *Tnd_ik, KK_FLOAT *Tnd_jk) const
{
  // Hierarchical-parallelism port of the dense 3-body Chebyshev reduction
  // (mirrors the Kokkos SNAP approach). The FLAT coefficient index `count`
  // (0..ncoeffs_3b) is distributed across the team's ThreadVectorRange so all
  // vector lanes participate (ncoeffs_3b = max_poly^3 up to 729) and the
  // `c_chimes_3b_params` reads are contiguous/coalesced across lanes. Each lane
  // derives its (l,m,n) polynomial-order indices from `count` exactly as
  // poly_3B_dense_loop1 does. Reduced into a single {energy, 3 pair-force
  // derivatives} value. On host backends (vector length 1) this iterates
  // count = 0,1,2,... in order, reproducing the original serial loop.

  auto c_chimes_3b_params_tripidx = Kokkos::subview(c_chimes_3b_params,tripidx,Kokkos::ALL);

  constexpr int coeff_batch = std::is_same<DeviceType, LMPHostType>::value ? 1 : CHIMES_COEFF_BATCH;
  const int n_groups = (ncoeffs_3b + coeff_batch - 1) / coeff_batch;

  s_chimes_poly3 result;

  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, n_groups),
    [&] (const int t, s_chimes_poly3& upd) {
      Kokkos::Array<s_chimes_poly3, coeff_batch> acc;

      #pragma unroll
      for (int b = 0; b < coeff_batch; b++) {
        const int count = t + b * n_groups;   // coalesced across lanes at fixed b
        if (count < ncoeffs_3b) {
          const KK_FLOAT coeff = c_chimes_3b_params_tripidx[count];
          if (coeff != 0.0) {
            const int l = count / (max_poly * max_poly);
            const int m = (count / max_poly) % max_poly;
            const int n = count % max_poly;

            const KK_FLOAT tn_ij = Tn_ij[l];
            const KK_FLOAT tnd_ij = Tnd_ij[l];
            const KK_FLOAT tn_ik = Tn_ik[m];
            const KK_FLOAT tnd_ik = Tnd_ik[m];
            const KK_FLOAT tn_jk = Tn_jk[n];
            const KK_FLOAT tnd_jk = Tnd_jk[n];
            const KK_FLOAT tn_ij_ik = tn_ij * tn_ik;

            acc[b].e  += coeff * tn_ij_ik * tn_jk;
            acc[b].f0 += coeff * tnd_ij * tn_ik * tn_jk;
            acc[b].f1 += coeff * tnd_ik * tn_ij * tn_jk;
            acc[b].f2 += coeff * tnd_jk * tn_ij_ik;
          }
        }
      }

      #pragma unroll
      for (int b = 0; b < coeff_batch; b++) upd += acc[b];
    }, Kokkos::Sum<s_chimes_poly3>(result));

  e += result.e;
  f0 += result.f0;
  f1 += result.f1;
  f2 += result.f2;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B(const t_team& team, KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_4b, int quadidx, int idx,
                                         KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
                                         KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
                                         KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
                                         KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const
// Compute the 4 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
//
// Hierarchical-parallelism port (mirrors Kokkos SNAP): the flat coefficient
// loop is distributed across the team's ThreadVectorRange and reduced into a
// single {energy, 6 pair-force derivatives} value. On host backends (vector
// length 1) this reproduces the original serial loop.
{
  constexpr int npairs = 6;
  int quad_map_idx[npairs];

  for (int i = 0; i < npairs; i++) quad_map_idx[i] = c_pair_int_quad_map(idx,i);

  auto c_chimes_4b_params_quadidx = Kokkos::subview(c_chimes_4b_params,quadidx,Kokkos::ALL);
  auto c_chimes_4b_powers_quadidx = Kokkos::subview(c_chimes_4b_powers,quadidx,Kokkos::ALL,Kokkos::ALL);

  constexpr int coeff_batch = std::is_same<DeviceType, LMPHostType>::value ? 1 : CHIMES_COEFF_BATCH;
  const int n_groups = (ncoeffs_4b + coeff_batch - 1) / coeff_batch;

  s_chimes_poly4 result;

  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, n_groups),
    [&] (const int t, s_chimes_poly4& upd) {
      Kokkos::Array<s_chimes_poly4, coeff_batch> acc;

      #pragma unroll
      for (int bb = 0; bb < coeff_batch; bb++) {
        const int coeffs = t + bb * n_groups;   // coalesced across lanes at fixed bb
        if (coeffs < ncoeffs_4b) {
          const KK_FLOAT coeff = c_chimes_4b_params_quadidx(coeffs);

          int powers[npairs];
          for (int i = 0; i < npairs; i++) powers[i] = c_chimes_4b_powers_quadidx(quad_map_idx[i],coeffs);

          const KK_FLOAT Tn_ij_ik_il = Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_il[powers[2]];
          const KK_FLOAT Tn_jk_jl = Tn_jk[powers[3]] * Tn_jl[powers[4]];
          const KK_FLOAT Tn_kl_5 = Tn_kl[powers[5]];

          acc[bb].e  += coeff * Tn_ij_ik_il * Tn_jk_jl * Tn_kl_5;

          acc[bb].f0 += coeff * Tnd_ij[powers[0]] * Tn_ik[powers[1]] * Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5;
          acc[bb].f1 += coeff * Tnd_ik[powers[1]] * Tn_ij[powers[0]] * Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5;
          acc[bb].f2 += coeff * Tnd_il[powers[2]] * Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_jk_jl * Tn_kl_5;
          acc[bb].f3 += coeff * Tnd_jk[powers[3]] * Tn_ij_ik_il * Tn_jl[powers[4]] * Tn_kl_5;
          acc[bb].f4 += coeff * Tnd_jl[powers[4]] * Tn_ij_ik_il * Tn_jk[powers[3]] * Tn_kl_5;
          acc[bb].f5 += coeff * Tnd_kl[powers[5]] * Tn_ij_ik_il * Tn_jk_jl;
        }
      }

      #pragma unroll
      for (int bb = 0; bb < coeff_batch; bb++) upd += acc[bb];
    }, Kokkos::Sum<s_chimes_poly4>(result));

  e = result.e;
  f[0] = result.f0;
  f[1] = result.f1;
  f[2] = result.f2;
  f[3] = result.f3;
  f[4] = result.f4;
  f[5] = result.f5;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B_dense(const t_team& team, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3, KK_FLOAT &f4,
                             KK_FLOAT &f5, int ncoeffs_4b, int quadidx,
                             KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik, KK_FLOAT* Tn_il,
                             KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
                             KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il,
                             KK_FLOAT* Tnd_jk, KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const
// Compute the 4 body polynomial (e) and derivatives with respect to each pair distance (f0..f5)
// (LEF) 4/02/26
//
// Hierarchical-parallelism port (mirrors Kokkos SNAP): the flat coefficient
// index is distributed across the team's ThreadVectorRange, with each lane
// deriving its 6 power indices from the flat count (as poly_4B_dense_loop1
// did) and reading params[count] contiguously (coalesced). Reduced into a
// single {energy, 6 pair-force derivatives} value. On host backends (vector
// length 1) this reproduces the original serial loop.
{
  e = 0.0;
  f0 = 0.0;
  f1 = 0.0;
  f2 = 0.0;
  f3 = 0.0;
  f4 = 0.0;
  f5 = 0.0;

  if (ncoeffs_4b == 0) return;

  int max_poly = 0;
  const int loop_max = 100;
  int i = 0;
  for (; i < loop_max; i++) {
    if (i * i * i * i * i * i == ncoeffs_4b) {
      max_poly = i;
      break;
    }
  }
  if (i == loop_max) {
    Kokkos::abort("Bad number of 4 body coefficients for dense evaluation");
  }

  int max_poly_pow[6];
  max_poly_pow[5] = 1;
  for (int l = 4; l >= 0; l--) { max_poly_pow[l] = max_poly_pow[l + 1] * max_poly; }

  auto c_chimes_4b_params_quadidx = Kokkos::subview(c_chimes_4b_params,quadidx,Kokkos::ALL);

  constexpr int coeff_batch = std::is_same<DeviceType, LMPHostType>::value ? 1 : CHIMES_COEFF_BATCH;
  const int n_groups = (ncoeffs_4b + coeff_batch - 1) / coeff_batch;

  s_chimes_poly4 result;

  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, n_groups),
    [&] (const int t, s_chimes_poly4& upd) {
      Kokkos::Array<s_chimes_poly4, coeff_batch> acc;

      #pragma unroll
      for (int bb = 0; bb < coeff_batch; bb++) {
        const int count = t + bb * n_groups;   // coalesced across lanes at fixed bb
        if (count < ncoeffs_4b) {
          const KK_FLOAT coeff = c_chimes_4b_params_quadidx[count];
          if (coeff != 0.0) {
            int index[6];
            for (int n = 0; n < 6; n++) { index[n] = (count / max_poly_pow[n]) % max_poly; }

            const KK_FLOAT tn_ij = Tn_ij[index[0]];
            const KK_FLOAT tnd_ij = Tnd_ij[index[0]];
            const KK_FLOAT tn_ik = Tn_ik[index[1]];
            const KK_FLOAT tnd_ik = Tnd_ik[index[1]];
            const KK_FLOAT tn_il = Tn_il[index[2]];
            const KK_FLOAT tnd_il = Tnd_il[index[2]];
            const KK_FLOAT tn_jk = Tn_jk[index[3]];
            const KK_FLOAT tnd_jk = Tnd_jk[index[3]];
            const KK_FLOAT tn_jl = Tn_jl[index[4]];
            const KK_FLOAT tnd_jl = Tnd_jl[index[4]];
            const KK_FLOAT tn_kl = Tn_kl[index[5]];
            const KK_FLOAT tnd_kl = Tnd_kl[index[5]];

            const KK_FLOAT Tn_jk_jl = tn_jk * tn_jl;
            const KK_FLOAT Tn_ij_ik_il = tn_ij * tn_ik * tn_il;

            acc[bb].e  += coeff * Tn_ij_ik_il * Tn_jk_jl * tn_kl;
            acc[bb].f0 += coeff * tnd_ij * tn_ik * tn_il * Tn_jk_jl * tn_kl;
            acc[bb].f1 += coeff * tnd_ik * tn_ij * tn_il * Tn_jk_jl * tn_kl;
            acc[bb].f2 += coeff * tnd_il * tn_ij * tn_ik * Tn_jk_jl * tn_kl;
            acc[bb].f3 += coeff * tnd_jk * Tn_ij_ik_il * tn_jl * tn_kl;
            acc[bb].f4 += coeff * tnd_jl * Tn_ij_ik_il * tn_jk * tn_kl;
            acc[bb].f5 += coeff * tnd_kl * Tn_ij_ik_il * Tn_jk_jl;
          }
        }
      }

      #pragma unroll
      for (int bb = 0; bb < coeff_batch; bb++) upd += acc[bb];
    }, Kokkos::Sum<s_chimes_poly4>(result));

  e = result.e;
  f0 = result.f0;
  f1 = result.f1;
  f2 = result.f2;
  f3 = result.f3;
  f4 = result.f4;
  f5 = result.f5;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B_dense_loop1(
    int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3, KK_FLOAT &f4, KK_FLOAT &f5,
    int ncoeffs_4b, int quadidx, KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik,
    KK_FLOAT* Tn_il, KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
    KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il, KK_FLOAT* Tnd_jk,
    KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const
{
  auto c_chimes_4b_params_quadidx = Kokkos::subview(c_chimes_4b_params,quadidx,Kokkos::ALL);

  int max_poly_pow[6];
  max_poly_pow[5] = 1;

  for (int l = 4; l >= 0; l--) { max_poly_pow[l] = max_poly_pow[l + 1] * max_poly; }

  #pragma unroll
  for (int count = 0; count < ncoeffs_4b; count++) {
    if (c_chimes_4b_params_quadidx[count] != 0.0) {
      int index[6];
      for (int i = 0; i < 6; i++) { index[i] = (count / max_poly_pow[i]) % max_poly; }
      const KK_FLOAT tn_ij = Tn_ij[index[0]];
      const KK_FLOAT tnd_ij = Tnd_ij[index[0]];
      const KK_FLOAT tn_ik = Tn_ik[index[1]];
      const KK_FLOAT tnd_ik = Tnd_ik[index[1]];
      const KK_FLOAT tn_il = Tn_il[index[2]];
      const KK_FLOAT tnd_il = Tnd_il[index[2]];
      const KK_FLOAT tn_jk = Tn_jk[index[3]];
      const KK_FLOAT tnd_jk = Tnd_jk[index[3]];
      const KK_FLOAT tn_jl = Tn_jl[index[4]];
      const KK_FLOAT tnd_jl = Tnd_jl[index[4]];
      const KK_FLOAT tn_kl = Tn_kl[index[5]];
      const KK_FLOAT tnd_kl = Tnd_kl[index[5]];
      const KK_FLOAT coeff = c_chimes_4b_params_quadidx[count];

      const KK_FLOAT Tn_jk_jl = tn_jk * tn_jl;
      const KK_FLOAT Tn_ij_ik_il = tn_ij * tn_ik * tn_il;

      e += coeff * Tn_ij_ik_il * Tn_jk_jl * tn_kl;

      f0 += coeff * tnd_ij * tn_ik * tn_il * Tn_jk_jl * tn_kl;
      f1 += coeff * tnd_ik * tn_ij * tn_il * Tn_jk_jl * tn_kl;
      f2 += coeff * tnd_il * tn_ij * tn_ik * Tn_jk_jl * tn_kl;
      f3 += coeff * tnd_jk * Tn_ij_ik_il * tn_jl * tn_kl;
      f4 += coeff * tnd_jl * Tn_ij_ik_il * tn_jk * tn_kl;
      f5 += coeff * tnd_kl * Tn_ij_ik_il * Tn_jk_jl;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B_dense_loop2(
    int max_poly, KK_FLOAT &e, KK_FLOAT &f0, KK_FLOAT &f1, KK_FLOAT &f2, KK_FLOAT &f3, KK_FLOAT &f4, KK_FLOAT &f5,
    int ncoeffs_4b, int quadidx, KK_FLOAT* Tn_ij, KK_FLOAT* Tn_ik,
    KK_FLOAT* Tn_il, KK_FLOAT* Tn_jk, KK_FLOAT* Tn_jl, KK_FLOAT* Tn_kl,
    KK_FLOAT* Tnd_ij, KK_FLOAT* Tnd_ik, KK_FLOAT* Tnd_il, KK_FLOAT* Tnd_jk,
    KK_FLOAT* Tnd_jl, KK_FLOAT* Tnd_kl) const
{
  auto c_chimes_4b_params_quadidx = Kokkos::subview(c_chimes_4b_params,quadidx,Kokkos::ALL);

  int count = 0;
  for (int i = 0; i < max_poly; i++) {
    const KK_FLOAT tn_ij = Tn_ij[i];
    const KK_FLOAT tnd_ij = Tnd_ij[i];
    for (int j = 0; j < max_poly; j++) {
      const KK_FLOAT tn_ik = Tn_ik[j];
      const KK_FLOAT tnd_ik = Tnd_ik[j];
      for (int l = 0; l < max_poly; l++) {
        const KK_FLOAT tn_il = Tn_il[l];
        const KK_FLOAT tnd_il = Tnd_il[l];
        const KK_FLOAT Tn_ij_ik_il = tn_ij * tn_ik * tn_il;
        for (int m = 0; m < max_poly; m++) {
          const KK_FLOAT tn_jk = Tn_jk[m];
          const KK_FLOAT tnd_jk = Tnd_jk[m];
          for (int n = 0; n < max_poly; n++) {
            const KK_FLOAT tn_jl = Tn_jl[n];
            const KK_FLOAT tnd_jl = Tnd_jl[n];
            const KK_FLOAT Tn_jk_jl = tn_jk * tn_jl;
            for (int o = 0; o < max_poly; o++) {
              const KK_FLOAT tn_kl = Tn_kl[o];
              const KK_FLOAT tnd_kl = Tnd_kl[o];

              if (c_chimes_4b_params_quadidx[count] != 0.0) {
                const KK_FLOAT coeff = c_chimes_4b_params_quadidx[count];

                e += coeff * Tn_ij_ik_il * Tn_jk_jl * tn_kl;

                f0 += coeff * tnd_ij * tn_ik * tn_il * Tn_jk_jl * tn_kl;
                f1 += coeff * tnd_ik * tn_ij * tn_il * Tn_jk_jl * tn_kl;
                f2 += coeff * tnd_il * tn_ij * tn_ik * Tn_jk_jl * tn_kl;
                f3 += coeff * tnd_jk * Tn_ij_ik_il * tn_jl * tn_kl;
                f4 += coeff * tnd_jl * Tn_ij_ik_il * tn_jk * tn_kl;
                f5 += coeff * tnd_kl * Tn_ij_ik_il * Tn_jk_jl;
              }
              count++;
            }
          }
        }
      }
    }
  }
}
