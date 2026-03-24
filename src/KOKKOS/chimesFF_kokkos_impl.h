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

#include "chimesFF_kokkos.h"
#include "memory_kokkos.h"

/* ---------------------------------------------------------------------- */

template<class DeviceType>
chimesFFKokkos<DeviceType>::chimesFFKokkos() : chimesFF()
{

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
  LAMMPS_NS::MemKK::realloc_kokkos(d_poly_orders,"chimesFF:poly_orders",size);

  auto h_poly_orders = Kokkos::create_mirror_view(d_poly_orders);

  for (int i = 0; i < size; i++)
    h_poly_orders[i] = poly_orders[i];

  Kokkos::deep_copy(d_poly_orders,h_poly_orders);


  // morse_var

  size = morse_var.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_morse_var,"chimesFF:morse_var",size);

  auto h_morse_var = Kokkos::create_mirror_view(d_morse_var);

  for (int i = 0; i < size; i++)
    h_morse_var[i] = morse_var[i];

  Kokkos::deep_copy(d_morse_var,h_morse_var);


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


  // chimes_2b_cutoff

  size = chimes_2b_cutoff.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_2b_cutoff,"chimesFF:chimes_2b_cutoff",size);

  auto h_chimes_2b_cutoff = Kokkos::create_mirror_view(d_chimes_2b_cutoff);

  for (int i = 0; i < size; i++) {
    h_chimes_2b_cutoff(i,0) = chimes_2b_cutoff[i][0];
    h_chimes_2b_cutoff(i,1) = chimes_2b_cutoff[i][1];
  }

  Kokkos::deep_copy(d_chimes_2b_cutoff,h_chimes_2b_cutoff);


  // ncoeffs_3b

  size = ncoeffs_3b.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_ncoeffs_3b,"chimesFF:ncoeffs_3b",size);

  auto h_ncoeffs_3b = Kokkos::create_mirror_view(d_ncoeffs_3b);

  for (int i = 0; i < size; i++)
    h_ncoeffs_3b[i] = ncoeffs_3b[i];

  Kokkos::deep_copy(d_ncoeffs_3b,h_ncoeffs_3b);


  // chimes_3b_powers

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

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_powers,"chimesFF:chimes_3b_powers",size,max_j,max_k);

  auto h_chimes_3b_powers = Kokkos::create_mirror_view(d_chimes_3b_powers);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_powers[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_powers[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_3b_powers(i,j,k) = chimes_3b_powers[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_3b_powers,h_chimes_3b_powers);


  // chimes_3b_params

  size = chimes_3b_params.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_3b_params[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_params,"chimesFF:chimes_3b_params",size,max_j);

  auto h_chimes_3b_params = Kokkos::create_mirror_view(d_chimes_3b_params);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_params[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_3b_params(i,j) = chimes_3b_params[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_3b_params,h_chimes_3b_params);


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

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_3b_cutoff,"chimesFF:chimes_3b_cutoff",size,max_j,max_k);

  auto h_chimes_3b_cutoff = Kokkos::create_mirror_view(d_chimes_3b_cutoff);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_3b_cutoff[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_3b_cutoff[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_3b_cutoff(i,j,k) = chimes_3b_cutoff[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_3b_cutoff,h_chimes_3b_cutoff);


  // ncoeffs_4b

  size = ncoeffs_4b.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_ncoeffs_4b,"chimesFF:ncoeffs_4b",size);

  auto h_ncoeffs_4b = Kokkos::create_mirror_view(d_ncoeffs_4b);

  for (int i = 0; i < size; i++)
    h_ncoeffs_4b[i] = ncoeffs_4b[i];

  Kokkos::deep_copy(d_ncoeffs_4b,h_ncoeffs_4b);


  // chimes_4b_powers

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

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_powers,"chimesFF:chimes_4b_powers",size,max_j,max_k);

  auto h_chimes_4b_powers = Kokkos::create_mirror_view(d_chimes_4b_powers);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_powers[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_powers[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_4b_powers(i,j,k) = chimes_4b_powers[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_4b_powers,h_chimes_4b_powers);


  // chimes_4b_params

  size = chimes_4b_params.size();
  max_j = 0;
  for (int i = 0; i < size; i++)
    max_j = MAX(max_j,chimes_4b_params[i].size());

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_params,"chimesFF:chimes_4b_params",size,max_j);

  auto h_chimes_4b_params = Kokkos::create_mirror_view(d_chimes_4b_params);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_params[i].size();
    for (int j = 0; j < size_j; j++) {
      h_chimes_4b_params(i,j) = chimes_4b_params[i][j];
    }
  }

  Kokkos::deep_copy(d_chimes_4b_params,h_chimes_4b_params);


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

  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_4b_cutoff,"chimesFF:chimes_4b_cutoff",size,max_j,max_k);

  auto h_chimes_4b_cutoff = Kokkos::create_mirror_view(d_chimes_4b_cutoff);

  for (int i = 0; i < size; i++) {
    int size_j = chimes_4b_cutoff[i].size();
    for (int j = 0; j < size_j; j++) {
      int size_k = chimes_4b_cutoff[i][j].size();
      for (int k = 0; k < size_k; k++) {
        h_chimes_4b_cutoff(i,j,k) = chimes_4b_cutoff[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(d_chimes_4b_cutoff,h_chimes_4b_cutoff);


  // energy_offsets

  size = energy_offsets.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_energy_offsets,"chimesFF:energy_offsets",size);

  auto h_energy_offsets = Kokkos::create_mirror_view(d_energy_offsets);

  for (int i = 0; i < size; i++)
    h_energy_offsets[i] = energy_offsets[i];

  Kokkos::deep_copy(d_energy_offsets,h_energy_offsets);


  // atom_int_pair_map

  size = atom_int_pair_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_pair_map,"chimesFF:atom_int_pair_map",size);

  auto h_atom_int_pair_map = Kokkos::create_mirror_view(d_atom_int_pair_map);

  for (int i = 0; i < size; i++)
    h_atom_int_pair_map[i] = atom_int_pair_map[i];

  Kokkos::deep_copy(d_atom_int_pair_map,h_atom_int_pair_map);


  // atom_int_trip_map

  size = atom_int_trip_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_trip_map,"chimesFF:",size);

  auto h_atom_int_trip_map = Kokkos::create_mirror_view(d_atom_int_trip_map);

  for (int i = 0; i < size; i++)
    h_atom_int_trip_map[i] = atom_int_trip_map[i];

  Kokkos::deep_copy(d_atom_int_trip_map,h_atom_int_trip_map);


  // atom_int_quad_map

  size = atom_int_quad_map.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_atom_int_quad_map,"chimesFF:",size);

  auto h_atom_int_quad_map = Kokkos::create_mirror_view(d_atom_int_quad_map);

  for (int i = 0; i < size; i++)
    h_atom_int_quad_map[i] = atom_int_quad_map[i];

  Kokkos::deep_copy(d_atom_int_quad_map,h_atom_int_quad_map);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::set_polys_out_of_range(const int ii, typename AT::t_kkfloat_2d &Tn, typename AT::t_kkfloat_2d &Tnd, KK_FLOAT dx, KK_FLOAT x, int poly_order, KK_FLOAT inner_cutoff, KK_FLOAT exprlen, KK_FLOAT dx_dr) const
{
  //  Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd) when dx is < inner_cutoff.
  //  Tnd is the derivative with respect to the interatomic distance, not the transformed distance (x).
  //	
  //  The derivative Tnd is continuously set to zero inside the cutoff.
  //  The exponential smoothing distance is set to ChimesFF::inner_smooth_distance.
  //  x, exprlen, and dx_dr are evaluated at the inner cutoff.
  //	
  //  dx is the pair distance, which is assumed to be less than inner_cutoff.
  Tn(ii,0) = 1.0;
  Tn(ii,1) = x;

  // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

  Tnd(ii,0) = 1.0;
  Tnd(ii,1) = 2.0 * x;

  // Use recursion to set up the higher n-value Tn and Tnd's
  for (int i = 2; i <= poly_order; i++) {
    Tn(ii,i) = 2.0 * x * Tn(ii,i-1) - Tn(ii,i-2);
    Tnd(ii,i) = 2.0 * x * Tnd(ii,i-1) - Tnd(ii,i-2);
  }

  // Now multiply by n to convert Tnd's to actual derivatives of Tn

  for (int i = poly_order; i >= 1; i--)
    Tnd(ii,i) = i * dx_dr * Tnd(ii,i-1);

  Tnd(ii,0) = 0.0;

  // Exponential damping of the derivative.
  KK_FLOAT damp_fac = exp((dx-inner_cutoff) / inner_smooth_distance);

  // Correct Tn outside of the range using the damping factor.
  for (int i = 0 ; i <= poly_order ; i++) {
    Tn(ii,i) += inner_smooth_distance * (damp_fac-1.0)  * Tnd(ii,i);
    Tnd(ii,i) *= damp_fac;
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

  energy += d_energy_offsets[typ_idx];
}

/* ---------------------------------------------------------------------- */

// Overload for calls from LAMMPS

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_2B(const int ii, const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar;
  compute_2B(ii, dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_2B(const int ii, const KK_FLOAT dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT& force_scalar_in) const
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

  typename AT::t_kkfloat_2d Tn = chimes2BKK.d_Tn;
  typename AT::t_kkfloat_2d Tnd = chimes2BKK.d_Tnd;

  const int pair_idx = d_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);

  //if (dx >= d_chimes_2b_cutoff(pair_idx,1)) return;

  set_cheby_polys(ii, Tn, Tnd, dx, d_morse_var[pair_idx], d_chimes_2b_cutoff(pair_idx,0), d_chimes_2b_cutoff(pair_idx,1), d_poly_orders[0]);

  get_fcut(dx, d_chimes_2b_cutoff(pair_idx,1), fcut, fcutderiv);

  KK_FLOAT poly, dpoly_dx;

  poly_2B(ii, poly, dpoly_dx, d_ncoeffs_2b[pair_idx], pair_idx, Tn, Tnd);

  KK_FLOAT dx_inv = (dx > 0.0 ) ? 1.0 / dx : 1e20;

  energy += poly * fcut;
  KK_FLOAT force_scalar = (fcut * dpoly_dx + fcutderiv * poly) / dx;

  force[0*CHDIM+0] += force_scalar * dr[0];
  force[0*CHDIM+1] += force_scalar * dr[1];
  force[0*CHDIM+2] += force_scalar * dr[2];

  force[1*CHDIM+0] -= force_scalar * dr[0];
  force[1*CHDIM+1] -= force_scalar * dr[1];
  force[1*CHDIM+2] -= force_scalar * dr[2];

  // xx xy xz yy yz zz
  // 0  1  2  3  4  5

  // xx xy xz yx yy yz zx zy zz
  // 0  1  2  3  4  5  6  7  8
  // *           *           *

  stress[0] -= force_scalar * dr[0] * dr[0]; // xx tensor component
  stress[1] -= force_scalar * dr[0] * dr[1]; // xy tensor component
  stress[2] -= force_scalar * dr[0] * dr[2]; // xz tensor component
  stress[3] -= force_scalar * dr[1] * dr[1]; // yy tensor component
  stress[4] -= force_scalar * dr[1] * dr[2]; // yz tensor component
  stress[5] -= force_scalar * dr[2] * dr[2]; // zz tensor component

  KK_FLOAT E_penalty = 0.0;
  get_penalty(dx, pair_idx, E_penalty, force_scalar);

  if (E_penalty > 0.0 )
  {
    energy += E_penalty;

    force_scalar /= dx;

    // Note: force_scalar is negative (LEF) 7/30/21

    force[0*CHDIM+0] += force_scalar * dr[0];
    force[0*CHDIM+1] += force_scalar * dr[1];
    force[0*CHDIM+2] += force_scalar * dr[2];

    force[1*CHDIM+0] -= force_scalar * dr[0];
    force[1*CHDIM+1] -= force_scalar * dr[1];
    force[1*CHDIM+2] -= force_scalar * dr[2];

    // Update stress according to penalty force. (LEF) 07/30/21

    stress[0] -= force_scalar * dr[0] * dr[0]; // xx tensor component
    stress[1] -= force_scalar * dr[0] * dr[1]; // xy tensor component
    stress[2] -= force_scalar * dr[0] * dr[2]; // xz tensor component
    stress[3] -= force_scalar * dr[1] * dr[1]; // yy tensor component
    stress[4] -= force_scalar * dr[1] * dr[2]; // yz tensor component
    stress[5] -= force_scalar * dr[2] * dr[2]; // zz tensor component
  }

  force_scalar_in = force_scalar;
}

/* ---------------------------------------------------------------------- */

// Overload for calls from LAMMPS

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_3B(const int ii, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar[3];
  compute_3B(ii, dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_3B(const int ii, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar_in) const
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

  const int natoms = 3;                   // Number of atoms in an interaction set
  const int npairs = natoms*(natoms-1)/2; // Number of pairs in an interaction set

  typename AT::t_kkfloat_2d Tn_ij = chimes3BKK.d_Tn_ij;
  typename AT::t_kkfloat_2d Tn_ik = chimes3BKK.d_Tn_ik;
  typename AT::t_kkfloat_2d Tn_jk = chimes3BKK.d_Tn_jk;   // The Chebyshev polymonials
  typename AT::t_kkfloat_2d Tnd_ij = chimes3BKK.d_Tnd_ij;
  typename AT::t_kkfloat_2d Tnd_ik = chimes3BKK.d_Tnd_ik;
  typename AT::t_kkfloat_2d Tnd_jk = chimes3BKK.d_Tnd_jk;  // The Chebyshev polymonial derivatives

  // Avoid allocating vector quantities.  Heap memory allocation is slow on the GPU.
  // fixed-length C arrays are allocated on the stack

  KK_FLOAT fcut[npairs];
  KK_FLOAT fcutderiv[npairs];
  KK_FLOAT deriv[npairs];

  int type_idx = typ_idxs[0]*natmtyps*natmtyps + typ_idxs[1]*natmtyps + typ_idxs[2];
  int tripidx = d_atom_int_trip_map[type_idx];

  //if (tripidx < 0) // Skipping an excluded interaction
  //  return;

  // Check whether cutoffs are within allowed ranges
  //auto d_mapped_pair_idx = d_pair_int_trip_map[type_idx];

  KK_FLOAT cutoff_0 = d_chimes_3b_cutoff(tripidx,1,d_pair_int_trip_map(type_idx,0));
  KK_FLOAT cutoff_00 = d_chimes_3b_cutoff(tripidx,0,d_pair_int_trip_map(type_idx,0));

  //if (dx[0] >= cutoff_0) // ij
  //  return;

  KK_FLOAT cutoff_1 = d_chimes_3b_cutoff(tripidx,1,d_pair_int_trip_map(type_idx,1));
  KK_FLOAT cutoff_01 = d_chimes_3b_cutoff(tripidx,0,d_pair_int_trip_map(type_idx,1));

  //if (dx[1] >= cutoff_1) // ik
  //  return;

  KK_FLOAT cutoff_2 = d_chimes_3b_cutoff(tripidx,1,d_pair_int_trip_map(type_idx,2));
  KK_FLOAT cutoff_02 = d_chimes_3b_cutoff(tripidx,0,d_pair_int_trip_map(type_idx,2));

  //if (dx[2] >= cutoff_2) // jk
  //  return;

 int pair_type_1 = d_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);
 int pair_type_2 = d_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[2]);
 int pair_type_3 = d_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[2]);
 int order = d_poly_orders[1];

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

#ifdef USE_DISTANCE_TENSOR
  // Tensor product of displacement vectors

  KK_FLOAT dr2[CHDIM*CHDIM*npairs*npairs];
  init_distance_tensor(dr2, dr, npairs);
#endif

  // Set up the polynomials

  set_cheby_polys(ii, Tn_ij, Tnd_ij, dx[0], d_morse_var[pair_type_1], cutoff_00, cutoff_0, order);
  set_cheby_polys(ii, Tn_ik, Tnd_ik, dx[1], d_morse_var[pair_type_2], cutoff_01, cutoff_1, order);
  set_cheby_polys(ii, Tn_jk, Tnd_jk, dx[2], d_morse_var[pair_type_3], cutoff_02, cutoff_2, order);

  // Set up the smoothing functions

  get_fcut(dx[0], cutoff_0, fcut[0], fcutderiv[0]);
  get_fcut(dx[1], cutoff_1, fcut[1], fcutderiv[1]);
  get_fcut(dx[2], cutoff_2, fcut[2], fcutderiv[2]);
  KK_FLOAT fcut_all = fcut[0] * fcut[1] * fcut[2];

  // Product of 2 fcuts divided by dx. Index i = product of all fcuts except i

  KK_FLOAT fcut_2[npairs];
  fcut_2[0] = fcut[1] * fcut[2] / dx[0];
  fcut_2[1] = fcut[0] * fcut[2] / dx[1];
  fcut_2[2] = fcut[0] * fcut[1] / dx[2];

  KK_FLOAT poly, dpoly_dx[npairs];

  // Start the force/stress/energy calculation

  KK_FLOAT coeff;
  int powers[npairs];
  KK_FLOAT force_scalar[npairs];

  poly_3B(ii, poly, dpoly_dx, d_ncoeffs_3b[tripidx], tripidx, type_idx,
          Tn_ij, Tn_ik, Tn_jk, Tnd_ij, Tnd_ik, Tnd_jk);

  energy += poly * fcut_all;

  force_scalar[0] = (fcut_all * dpoly_dx[0] + fcutderiv[0] * fcut[1] * fcut[2] * poly) / dx[0];
  force_scalar[1] = (fcut_all * dpoly_dx[1] + fcutderiv[1] * fcut[0] * fcut[2] * poly) / dx[1];
  force_scalar[2] = (fcut_all * dpoly_dx[2] + fcutderiv[2] * fcut[0] * fcut[1] * poly) / dx[2];

  const KK_FLOAT fscalar_0 = force_scalar[0];
  const KK_FLOAT fscalar_1 = force_scalar[1];
  const KK_FLOAT fscalar_2 = force_scalar[2];

  // Accumulate forces/stresses on/from the ij pair

  force[0*CHDIM+0] += fscalar_0 * dr[0*CHDIM+0];
  force[0*CHDIM+1] += fscalar_0 * dr[0*CHDIM+1];
  force[0*CHDIM+2] += fscalar_0 * dr[0*CHDIM+2];

  force[1*CHDIM+0] -= fscalar_0 * dr[0*CHDIM+0];
  force[1*CHDIM+1] -= fscalar_0 * dr[0*CHDIM+1];
  force[1*CHDIM+2] -= fscalar_0 * dr[0*CHDIM+2];

  // dr2_3B looks like a function call, but the optimizer should remove it entirely
#ifdef USE_DISTANCE_TENSOR
  // New stress code

  stress[0] -= fscalar_0 * dr2_3B(dr2,0,0,0,0); // xx tensor component
  stress[1] -= fscalar_0 * dr2_3B(dr2,0,0,0,1); // xy tensor component
  stress[2] -= fscalar_0 * dr2_3B(dr2,0,0,0,2); // xz tensor component
  stress[3] -= fscalar_0 * dr2_3B(dr2,0,1,0,1); // yy tensor component
  stress[4] -= fscalar_0 * dr2_3B(dr2,0,1,0,2); // yz tensor component
  stress[5] -= fscalar_0 * dr2_3B(dr2,0,2,0,2); // zz tensor component

#else
  stress[0] -= fscalar_0 * dr[0*CHDIM+0] * dr[0*CHDIM+0]; // xx tensor component
  stress[1] -= fscalar_0 * dr[0*CHDIM+0] * dr[0*CHDIM+1]; // xy tensor component
  stress[2] -= fscalar_0 * dr[0*CHDIM+0] * dr[0*CHDIM+2]; // xz tensor component
  stress[3] -= fscalar_0 * dr[0*CHDIM+1] * dr[0*CHDIM+1]; // yy tensor component
  stress[4] -= fscalar_0 * dr[0*CHDIM+1] * dr[0*CHDIM+2]; // yz tensor component
  stress[5] -= fscalar_0 * dr[0*CHDIM+2] * dr[0*CHDIM+2]; // zz tensor component
#endif

  // Accumulate forces/stresses on/from the ik pair

  force[0*CHDIM+0] += fscalar_1 * dr[1*CHDIM+0];
  force[0*CHDIM+1] += fscalar_1 * dr[1*CHDIM+1];
  force[0*CHDIM+2] += fscalar_1 * dr[1*CHDIM+2];

  force[2*CHDIM+0] -= fscalar_1 * dr[1*CHDIM+0];
  force[2*CHDIM+1] -= fscalar_1 * dr[1*CHDIM+1];
  force[2*CHDIM+2] -= fscalar_1 * dr[1*CHDIM+2];

#ifdef USE_DISTANCE_TENSOR
  stress[0] -= fscalar_1 * dr2_3B(dr2,1,0,1,0); // xx tensor component
  stress[1] -= fscalar_1 * dr2_3B(dr2,1,0,1,1); // xy tensor component
  stress[2] -= fscalar_1 * dr2_3B(dr2,1,0,1,2); // xz tensor component
  stress[3] -= fscalar_1 * dr2_3B(dr2,1,1,1,1); // yy tensor component
  stress[4] -= fscalar_1 * dr2_3B(dr2,1,1,1,2); // yz tensor component
  stress[5] -= fscalar_1 * dr2_3B(dr2,1,2,1,2); // zz tensor component
#else
  stress[0] -= fscalar_1 * dr[1*CHDIM+0] * dr[1*CHDIM+0]; // xx tensor component
  stress[1] -= fscalar_1 * dr[1*CHDIM+0] * dr[1*CHDIM+1]; // xy tensor component
  stress[2] -= fscalar_1 * dr[1*CHDIM+0] * dr[1*CHDIM+2]; // xz tensor component
  stress[3] -= fscalar_1 * dr[1*CHDIM+1] * dr[1*CHDIM+1]; // yy tensor component
  stress[4] -= fscalar_1 * dr[1*CHDIM+1] * dr[1*CHDIM+2]; // yz tensor component
  stress[5] -= fscalar_1 * dr[1*CHDIM+2] * dr[1*CHDIM+2]; // zz tensor component
#endif

  // Accumulate forces/stresses on/from the jk pair

  force[1*CHDIM+0] += fscalar_2 * dr[2*CHDIM+0];
  force[1*CHDIM+1] += fscalar_2 * dr[2*CHDIM+1];
  force[1*CHDIM+2] += fscalar_2 * dr[2*CHDIM+2];

  force[2*CHDIM+0] -= fscalar_2 * dr[2*CHDIM+0];
  force[2*CHDIM+1] -= fscalar_2 * dr[2*CHDIM+1];
  force[2*CHDIM+2] -= fscalar_2 * dr[2*CHDIM+2];

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

  force_scalar_in[0] = force_scalar[0];
  force_scalar_in[1] = force_scalar[1];
  force_scalar_in[2] = force_scalar[2];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B(const int ii, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy) const
{
  KK_FLOAT dummy_force_scalar[6];
  compute_4B(ii, dx, dr, typ_idxs, force, stress, energy, dummy_force_scalar);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::compute_4B(const int ii, const KK_FLOAT* dx, const KK_FLOAT* dr, const int* typ_idxs, KK_FLOAT* force, KK_FLOAT* stress, KK_FLOAT & energy, KK_FLOAT* force_scalar_in) const
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

  const int natoms = 4;                     // Number of atoms in an interaction set
  const int npairs = natoms*(natoms-1)/2;    // Number of pairs in an interaction set

  KK_FLOAT fcut[npairs];
  KK_FLOAT fcutderiv[npairs];
  KK_FLOAT deriv[npairs];

  typename AT::t_kkfloat_2d Tn_ij = chimes4BKK.d_Tn_ij;
  typename AT::t_kkfloat_2d Tn_ik = chimes4BKK.d_Tn_ik;
  typename AT::t_kkfloat_2d Tn_il = chimes4BKK.d_Tn_il;
  typename AT::t_kkfloat_2d Tn_jk = chimes4BKK.d_Tn_jk;
  typename AT::t_kkfloat_2d Tn_jl = chimes4BKK.d_Tn_jl;
  typename AT::t_kkfloat_2d Tn_kl = chimes4BKK.d_Tn_kl;

  typename AT::t_kkfloat_2d Tnd_ij = chimes4BKK.d_Tnd_ij;
  typename AT::t_kkfloat_2d Tnd_ik = chimes4BKK.d_Tnd_ik;
  typename AT::t_kkfloat_2d Tnd_il = chimes4BKK.d_Tnd_il;
  typename AT::t_kkfloat_2d Tnd_jk = chimes4BKK.d_Tnd_jk;
  typename AT::t_kkfloat_2d Tnd_jl = chimes4BKK.d_Tnd_jl;
  typename AT::t_kkfloat_2d Tnd_kl = chimes4BKK.d_Tnd_kl;

  int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
      + typ_idxs[1]*natmtyps*natmtyps + typ_idxs[2]*natmtyps + typ_idxs[3];

  int quadidx = d_atom_int_quad_map[idx];

  //if (quadidx < 0) // Skipping an excluded interaction
  //  return;

  //auto d_mapped_pair_idx = d_pair_int_quad_map[idx];

  // Check whether cutoffs are within allowed ranges
/*
  for (int i=0; i<npairs; i++)
      if (dx[i] >= d_chimes_4b_cutoff(quadidx,1,d_pair_int_quad_map(idx,i)))
          return;
*/
  // These speed up fcut calculations by a LOT

  KK_FLOAT cutoff_0 = d_chimes_4b_cutoff(quadidx,1,d_pair_int_quad_map(idx,0));
  KK_FLOAT cutoff_00 = d_chimes_4b_cutoff(quadidx,0,d_pair_int_quad_map(idx,0));

  //if (dx[0] >= cutoff_0) // ij
  //  return;

  KK_FLOAT cutoff_1 = d_chimes_4b_cutoff(quadidx,1,d_pair_int_quad_map(idx,1));
  KK_FLOAT cutoff_01 = d_chimes_4b_cutoff(quadidx,0,d_pair_int_quad_map(idx,1));

  //if (dx[1] >= cutoff_1) // ik
  //  return;

  KK_FLOAT cutoff_2 = d_chimes_4b_cutoff(quadidx,1,d_pair_int_quad_map(idx,2));
  KK_FLOAT cutoff_02 = d_chimes_4b_cutoff(quadidx,0,d_pair_int_quad_map(idx,2));

  //if (dx[2] >= cutoff_2) // il
  //  return;

  KK_FLOAT cutoff_3 = d_chimes_4b_cutoff(quadidx,1,d_pair_int_quad_map(idx,3));
  KK_FLOAT cutoff_03 = d_chimes_4b_cutoff(quadidx,0,d_pair_int_quad_map(idx,3));

  //if (dx[3] >= cutoff_3) // jk
  //  return;

  KK_FLOAT cutoff_4 = d_chimes_4b_cutoff(quadidx,1,d_pair_int_quad_map(idx,4));
  KK_FLOAT cutoff_04 = d_chimes_4b_cutoff(quadidx,0,d_pair_int_quad_map(idx,4));

  //if (dx[4] >= cutoff_4) // jl
  //  return;

  KK_FLOAT cutoff_5 = d_chimes_4b_cutoff(quadidx,1,d_pair_int_quad_map(idx,5));
  KK_FLOAT cutoff_05 = d_chimes_4b_cutoff(quadidx,0,d_pair_int_quad_map(idx,5));

  //if (dx[5] >= cutoff_5) // kl
  //  return;

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

  int pair_type_1 = d_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);
  int pair_type_2 = d_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[2]);
  int pair_type_3 = d_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[3]);
  int pair_type_4 = d_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[2]);
  int pair_type_5 = d_atom_int_pair_map(typ_idxs[1]*natmtyps + typ_idxs[3]);
  int pair_type_6 = d_atom_int_pair_map(typ_idxs[2]*natmtyps + typ_idxs[3]);
  int order = d_poly_orders[2];

  // Set up the polynomials

  set_cheby_polys(ii, Tn_ij, Tnd_ij, dx[0], d_morse_var[pair_type_1], cutoff_00, cutoff_0, order);
  set_cheby_polys(ii, Tn_ik, Tnd_ik, dx[1], d_morse_var[pair_type_2], cutoff_01, cutoff_1, order);
  set_cheby_polys(ii, Tn_il, Tnd_il, dx[2], d_morse_var[pair_type_3], cutoff_02, cutoff_2, order);
  set_cheby_polys(ii, Tn_jk, Tnd_jk, dx[3], d_morse_var[pair_type_4], cutoff_03, cutoff_3, order);
  set_cheby_polys(ii, Tn_jl, Tnd_jl, dx[4], d_morse_var[pair_type_5], cutoff_04, cutoff_4, order);
  set_cheby_polys(ii, Tn_kl, Tnd_kl, dx[5], d_morse_var[pair_type_6], cutoff_05, cutoff_5, order);

#ifdef USE_DISTANCE_TENSOR
  // Tensor product of displacement vectors

  KK_FLOAT dr2[CHDIM*CHDIM*npairs*npairs];
  init_distance_tensor(dr2, dr, npairs);
#endif


  // Set up the smoothing functions
/*
  for (int i=0; i<npairs; i++)
      get_fcut(dx[i], d_chimes_4b_cutoff(quadidx,1,d_pair_int_quad_map(idx,i)], fcut[i], fcutderiv[i));
*/

  get_fcut(dx[0], cutoff_0, fcut[0], fcutderiv[0]);
  get_fcut(dx[1], cutoff_1, fcut[1], fcutderiv[1]);
  get_fcut(dx[2], cutoff_2, fcut[2], fcutderiv[2]);
  get_fcut(dx[3], cutoff_3, fcut[3], fcutderiv[3]);
  get_fcut(dx[4], cutoff_4, fcut[4], fcutderiv[4]);
  get_fcut(dx[5], cutoff_5, fcut[5], fcutderiv[5]);

  // Product of all 6 fcuts

  KK_FLOAT fcut_all = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[4] * fcut[5];

  // Product of 5 fcuts

  KK_FLOAT fcut_5[npairs];
  fcut_5[0] = fcut[1] * fcut[2] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[1] = fcut[0] * fcut[2] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[2] = fcut[0] * fcut[1] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[3] = fcut[0] * fcut[1] * fcut[2] * fcut[4] * fcut[5];
  fcut_5[4] = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[5];
  fcut_5[5] = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[4];

  // Start the force/stress/energy calculation

  KK_FLOAT force_scalar[npairs]; //// not c++ compliant

  KK_FLOAT poly, dpoly_dx[npairs];

  poly_4B(ii, poly, dpoly_dx, d_ncoeffs_4b[quadidx], quadidx, idx,
          Tn_ij, Tn_ik, Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik,
          Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);

  energy += poly * fcut_all;

  for (int j = 0; j < npairs; j++)
    force_scalar[j] = (fcut_all * dpoly_dx[j] + fcutderiv[j] * fcut_5[j] * poly) / dx[j];

  const KK_FLOAT fscalar_0 = force_scalar[0];
  const KK_FLOAT fscalar_1 = force_scalar[1];
  const KK_FLOAT fscalar_2 = force_scalar[2];
  const KK_FLOAT fscalar_3 = force_scalar[3];
  const KK_FLOAT fscalar_4 = force_scalar[4];
  const KK_FLOAT fscalar_5 = force_scalar[5];

  // Accumulate forces/stresses on/from the ij pair

  force[0*CHDIM+0] += fscalar_0 * dr[0*CHDIM+0];
  force[0*CHDIM+1] += fscalar_0 * dr[0*CHDIM+1];
  force[0*CHDIM+2] += fscalar_0 * dr[0*CHDIM+2];

  force[1*CHDIM+0] -= fscalar_0 * dr[0*CHDIM+0];
  force[1*CHDIM+1] -= fscalar_0 * dr[0*CHDIM+1];
  force[1*CHDIM+2] -= fscalar_0 * dr[0*CHDIM+2];

#ifdef USE_DISTANCE_TENSOR
  stress[0] -= fscalar_0 * dr2_4B(dr2,0,0,0,0); // xx tensor component
  stress[1] -= fscalar_0 * dr2_4B(dr2,0,0,0,1); // xy tensor component
  stress[2] -= fscalar_0 * dr2_4B(dr2,0,0,0,2); // xz tensor component
  stress[3] -= fscalar_0 * dr2_4B(dr2,0,1,0,1); // yy tensor component
  stress[4] -= fscalar_0 * dr2_4B(dr2,0,1,0,2); // yz tensor component
  stress[5] -= fscalar_0 * dr2_4B(dr2,0,2,0,2); // zz tensor component
#else
  stress[0] -= fscalar_0 * dr[0*CHDIM+0] * dr[0*CHDIM+0]; // xx tensor component
  stress[1] -= fscalar_0 * dr[0*CHDIM+0] * dr[0*CHDIM+1]; // xy tensor component
  stress[2] -= fscalar_0 * dr[0*CHDIM+0] * dr[0*CHDIM+2]; // xz tensor component
  stress[3] -= fscalar_0 * dr[0*CHDIM+1] * dr[0*CHDIM+1]; // yy tensor component
  stress[4] -= fscalar_0 * dr[0*CHDIM+1] * dr[0*CHDIM+2]; // yz tensor component
  stress[5] -= fscalar_0 * dr[0*CHDIM+2] * dr[0*CHDIM+2]; // zz tensor component
#endif

  // Accumulate forces/stresses on/from the ik pair

  force[0*CHDIM+0] += fscalar_1 * dr[1*CHDIM+0];
  force[0*CHDIM+1] += fscalar_1 * dr[1*CHDIM+1];
  force[0*CHDIM+2] += fscalar_1 * dr[1*CHDIM+2];
  force[2*CHDIM+0] -= fscalar_1 * dr[1*CHDIM+0];
  force[2*CHDIM+1] -= fscalar_1 * dr[1*CHDIM+1];
  force[2*CHDIM+2] -= fscalar_1 * dr[1*CHDIM+2];

#if USE_DISTANCE_TENSOR
  stress[0] -= fscalar_1 * dr2_4B(dr2,1,0,1,0); // xx tensor component
  stress[1] -= fscalar_1 * dr2_4B(dr2,1,0,1,1); // xy tensor component
  stress[2] -= fscalar_1 * dr2_4B(dr2,1,0,1,2); // xz tensor component
  stress[3] -= fscalar_1 * dr2_4B(dr2,1,1,1,1); // yy tensor component
  stress[4] -= fscalar_1 * dr2_4B(dr2,1,1,1,2); // yz tensor component
  stress[5] -= fscalar_1 * dr2_4B(dr2,1,2,1,2); // zz tensor component
#else
  stress[0] -= fscalar_1 * dr[1*CHDIM+0] * dr[1*CHDIM+0]; // xx tensor component
  stress[1] -= fscalar_1 * dr[1*CHDIM+0] * dr[1*CHDIM+1]; // xy tensor component
  stress[2] -= fscalar_1 * dr[1*CHDIM+0] * dr[1*CHDIM+2]; // xz tensor component
  stress[3] -= fscalar_1 * dr[1*CHDIM+1] * dr[1*CHDIM+1]; // yy tensor component
  stress[4] -= fscalar_1 * dr[1*CHDIM+1] * dr[1*CHDIM+2]; // yz tensor component
  stress[5] -= fscalar_1 * dr[1*CHDIM+2] * dr[1*CHDIM+2]; // zz tensor component
#endif

  // Accumulate forces/stresses on/from the il pair

  force[0*CHDIM+0] += fscalar_2 * dr[2*CHDIM+0];
  force[0*CHDIM+1] += fscalar_2 * dr[2*CHDIM+1];
  force[0*CHDIM+2] += fscalar_2 * dr[2*CHDIM+2];
  force[3*CHDIM+0] -= fscalar_2 * dr[2*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_2 * dr[2*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_2 * dr[2*CHDIM+2];

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

  // Accumulate forces/stresses on/from the jk pair

  force[1*CHDIM+0] += fscalar_3 * dr[3*CHDIM+0];
  force[1*CHDIM+1] += fscalar_3 * dr[3*CHDIM+1];
  force[1*CHDIM+2] += fscalar_3 * dr[3*CHDIM+2];

  force[2*CHDIM+0] -= fscalar_3 * dr[3*CHDIM+0];
  force[2*CHDIM+1] -= fscalar_3 * dr[3*CHDIM+1];
  force[2*CHDIM+2] -= fscalar_3 * dr[3*CHDIM+2];

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

  // Accumulate forces/stresses on/from the jl pair

  force[1*CHDIM+0] += fscalar_4 * dr[4*CHDIM+0];
  force[1*CHDIM+1] += fscalar_4 * dr[4*CHDIM+1];
  force[1*CHDIM+2] += fscalar_4 * dr[4*CHDIM+2];

  force[3*CHDIM+0] -= fscalar_4 * dr[4*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_4 * dr[4*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_4 * dr[4*CHDIM+2];

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
  // Accumulate forces/stresses on/from the kl pair

  force[2*CHDIM+0] += fscalar_5 * dr[5*CHDIM+0];
  force[2*CHDIM+1] += fscalar_5 * dr[5*CHDIM+1];
  force[2*CHDIM+2] += fscalar_5 * dr[5*CHDIM+2];
  force[3*CHDIM+0] -= fscalar_5 * dr[5*CHDIM+0];
  force[3*CHDIM+1] -= fscalar_5 * dr[5*CHDIM+1];
  force[3*CHDIM+2] -= fscalar_5 * dr[5*CHDIM+2];

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

  force_scalar_in[0] = force_scalar[0];
  force_scalar_in[1] = force_scalar[1];
  force_scalar_in[2] = force_scalar[2];
  force_scalar_in[3] = force_scalar[3];
  force_scalar_in[4] = force_scalar[4];
  force_scalar_in[5] = force_scalar[5];
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
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_2B(const int ii, KK_FLOAT &e, KK_FLOAT &f0, const int ncoeffs_2b, const int pair_idx,
                                         typename AT::t_kkfloat_2d &Tn, typename AT::t_kkfloat_2d &Tnd) const
// Compute the 2 body polynomial (e) and derivatives with respect to the pair distance (f0)
// (LEF) 3/11/26
{
  e = 0.0;
  f0 = 0.0;

  for (int coeffs = 0; coeffs < ncoeffs_2b; coeffs++) {
    KK_FLOAT coeff_val = d_chimes_2b_params(pair_idx,coeffs);

    e += coeff_val * Tn(ii,d_chimes_2b_pows(pair_idx,coeffs) + 1);
    f0 += coeff_val * Tnd(ii,d_chimes_2b_pows(pair_idx,coeffs) + 1);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_3B(const int ii, KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_3b, int tripidx, int idx,
                                         typename AT::t_kkfloat_2d &Tn_ij, typename AT::t_kkfloat_2d &Tn_ik, typename AT::t_kkfloat_2d &Tn_jk,
                                         typename AT::t_kkfloat_2d &Tnd_ij, typename AT::t_kkfloat_2d &Tnd_ik, typename AT::t_kkfloat_2d &Tnd_jk) const
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
{
  KK_FLOAT coeff;
  int powers[3];
  KK_FLOAT deriv[3];

  e = 0.0;
  f[0] = 0.0;
  f[1] = 0.0;
  f[2] = 0.0;

  for (int coeffs = 0; coeffs < ncoeffs_3b; coeffs++) {
    coeff = d_chimes_3b_params(tripidx,coeffs);

    powers[0] = d_chimes_3b_powers(tripidx,coeffs,d_pair_int_trip_map(idx,0));
    powers[1] = d_chimes_3b_powers(tripidx,coeffs,d_pair_int_trip_map(idx,1));
    powers[2] = d_chimes_3b_powers(tripidx,coeffs,d_pair_int_trip_map(idx,2));

    e += coeff * Tn_ij(ii,powers[0]) * Tn_ik(ii,powers[1]) * Tn_jk(ii,powers[2]);

    deriv[0] = Tnd_ij(ii,powers[0]);
    deriv[1] = Tnd_ik(ii,powers[1]);
    deriv[2] = Tnd_jk(ii,powers[2]);

    f[0] += coeff * deriv[0] * Tn_ik(ii,powers[1]) * Tn_jk(ii,powers[2]);
    f[1] += coeff * deriv[1] * Tn_ij(ii,powers[0]) * Tn_jk(ii,powers[2]);
    f[2] += coeff * deriv[2] * Tn_ij(ii,powers[0]) * Tn_ik(ii,powers[1]);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void chimesFFKokkos<DeviceType>::poly_4B(const int ii, KK_FLOAT &e, KK_FLOAT *f, int ncoeffs_4b, int quadidx, int idx,
                                         typename AT::t_kkfloat_2d &Tn_ij, typename AT::t_kkfloat_2d &Tn_ik, typename AT::t_kkfloat_2d &Tn_il,
                                         typename AT::t_kkfloat_2d &Tn_jk, typename AT::t_kkfloat_2d &Tn_jl, typename AT::t_kkfloat_2d &Tn_kl,
                                         typename AT::t_kkfloat_2d &Tnd_ij, typename AT::t_kkfloat_2d &Tnd_ik, typename AT::t_kkfloat_2d &Tnd_il,
                                         typename AT::t_kkfloat_2d &Tnd_jk, typename AT::t_kkfloat_2d &Tnd_jl, typename AT::t_kkfloat_2d &Tnd_kl) const
// Compute the 4 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
{
  KK_FLOAT coeff;
  const int npairs = 6;
  int powers[npairs];
  KK_FLOAT deriv[npairs];

  e = 0;
  for (int i = 0; i < npairs; i++) f[i] = 0.0;

  for (int coeffs = 0; coeffs < ncoeffs_4b; coeffs++) {
    coeff = d_chimes_4b_params(quadidx,coeffs);

    for (int i = 0; i < npairs; i++) powers[i] = d_chimes_4b_powers(quadidx,coeffs,d_pair_int_quad_map(idx,i));

    KK_FLOAT Tn_ij_ik_il = Tn_ij(ii,powers[0]) * Tn_ik(ii,powers[1]) * Tn_il(ii,powers[2]);
    KK_FLOAT Tn_jk_jl = Tn_jk(ii,powers[3]) * Tn_jl(ii,powers[4]);
    KK_FLOAT Tn_kl_5 = Tn_kl(ii,powers[5]);

    e += coeff * Tn_ij_ik_il * Tn_jk_jl * Tn_kl_5;

    deriv[0] = Tnd_ij(ii,powers[0]);
    deriv[1] = Tnd_ik(ii,powers[1]);
    deriv[2] = Tnd_il(ii,powers[2]);
    deriv[3] = Tnd_jk(ii,powers[3]);
    deriv[4] = Tnd_jl(ii,powers[4]);
    deriv[5] = Tnd_kl(ii,powers[5]);

    f[0] += coeff * deriv[0] * Tn_ik(ii,powers[1]) * Tn_il(ii,powers[2]) * Tn_jk_jl * Tn_kl_5;

    f[1] += coeff * deriv[1] * Tn_ij(ii,powers[0]) * Tn_il(ii,powers[2]) * Tn_jk_jl * Tn_kl_5;

    f[2] += coeff * deriv[2] * Tn_ij(ii,powers[0]) * Tn_ik(ii,powers[1]) * Tn_jk_jl * Tn_kl_5;

    f[3] += coeff * deriv[3] * Tn_ij_ik_il * Tn_jl(ii,powers[4]) * Tn_kl_5;

    f[4] += coeff * deriv[4] * Tn_ij_ik_il * Tn_jk(ii,powers[3]) * Tn_kl_5;

    f[5] += coeff * deriv[5] * Tn_ij_ik_il * Tn_jk_jl;
  }
}
