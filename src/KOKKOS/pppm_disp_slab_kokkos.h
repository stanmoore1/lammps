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

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/disp/slab/kk,PPPMDispSlabKokkos<LMPDeviceType>);
KSpaceStyle(pppm/disp/slab/kk/device,PPPMDispSlabKokkos<LMPDeviceType>);
KSpaceStyle(pppm/disp/slab/kk/host,PPPMDispSlabKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PPPM_DISP_SLAB_KOKKOS_H
#define LMP_PPPM_DISP_SLAB_KOKKOS_H

#include "pppm_disp_slab.h"
#include "fft3d_kokkos.h"
#include "fftdata_kokkos.h"
#include "kokkos_type.h"
#include "math_const.h"

namespace LAMMPS_NS {

// functor tags for the device kernels
struct TagPPPMDispSlab_make_rho_zero{};
struct TagPPPMDispSlab_make_rho{};
struct TagPPPMDispSlab_dens_to_work{};
struct TagPPPMDispSlab_poisson_energy{};
struct TagPPPMDispSlab_poisson_virial{};
struct TagPPPMDispSlab_poisson_uT_prep{};
struct TagPPPMDispSlab_poisson_uT_copy{};
struct TagPPPMDispSlab_poisson_uN_prep{};
struct TagPPPMDispSlab_poisson_uN_copy{};
struct TagPPPMDispSlab_poisson_fz_prep{};
struct TagPPPMDispSlab_poisson_fz_copy{};
struct TagPPPMDispSlab_poisson_u_prep{};
struct TagPPPMDispSlab_poisson_u_copy{};
struct TagPPPMDispSlab_fieldforce{};
struct TagPPPMDispSlab_fieldforce_peratom{};
struct TagPPPMDispSlab_peatom_zero{};
struct TagPPPMDispSlab_peratom_finalize{};

// (tangential, normal) virial reduction accumulator for the mesh
struct s_PPPMDispSlabVir {
  double vt, vn;
  KOKKOS_INLINE_FUNCTION s_PPPMDispSlabVir() { vt = 0.0; vn = 0.0; }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_PPPMDispSlabVir &rhs) { vt += rhs.vt; vn += rhs.vn; }
};
typedef struct s_PPPMDispSlabVir s_vir;

template<class DeviceType>
class PPPMDispSlabKokkos : public PPPMDispSlab {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef FFTArrayTypes<DeviceType> FFT_AT;

  PPPMDispSlabKokkos(class LAMMPS *);
  ~PPPMDispSlabKokkos() override;
  void init() override;
  void setup() override;
  void compute(int, int) override;
  double memory_usage() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_make_rho_zero, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_make_rho, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_dens_to_work, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_energy, const int&, double&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_virial, const int&, s_vir&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_uT_prep, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_uT_copy, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_uN_prep, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_uN_copy, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_fz_prep, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_fz_copy, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_u_prep, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_poisson_u_copy, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_fieldforce, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_fieldforce_peratom, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_peatom_zero, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_peratom_finalize, const int&) const;

  // assignment weights w[0..order-1] at fractional offset dz (Horner in dz)
  KOKKOS_INLINE_FUNCTION
  void compute_rho1d_kk(const double dz, double *w) const {
    for (int s = 0; s < order_kk; s++) {
      double r = 0.0;
      for (int l = order_kk - 1; l >= 0; l--) r = d_rho_coeff(l, s) + r * dz;
      w[s] = r;
    }
  }

 protected:
  class AtomKokkos *atomKK;

  // local (per-proc) 1d FFTs on the device; the grid is gathered with an
  // MPI_Allreduce and FFT'd locally on each proc (MPI_COMM_SELF plan)
  FFT3dKokkos<DeviceType> *fft_forward;
  FFT3dKokkos<DeviceType> *fft_backward;
  int nz_created;            // nz at last FFT/array allocation

  // interleaved complex work buffers (size 2*nz): even = real, odd = imag
  typename FFT_AT::t_FFT_SCALAR_1d d_work, d_work2;

  // z-grid fields (length nz)
  typename AT::t_double_1d d_dens;        // gathered B-weighted density
  typename AT::t_double_1d d_Gk;          // energy influence function (merged corr)
  typename AT::t_double_1d d_GTk, d_GNk;  // tangential/normal virial influence
  typename AT::t_double_1d d_fz_grid;     // z-force field
  typename AT::t_double_1d d_ugrid;       // per-atom potential field
  typename AT::t_double_1d d_uTgrid, d_uNgrid;   // per-atom T/N virial fields
  Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> h_dens;   // Allreduce staging

  typename AT::t_double_1d d_B;           // per-type amplitude B[ntypes+1]
  typename AT::t_double_2d d_rho_coeff;   // B-spline coefficients [order][order]

  // per-atom reciprocal energy (device, kspace per-atom energy buffer)
  typename AT::t_double_1d d_peatom;
  int nmax_kk;              // current allocation size of d_peatom

  // per-atom output arrays (base KSpace eatom/vatom, aliased via DualViews)
  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  // cached atom views
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d_randomread type;

  // scalar device copies of hot-path parameters
  KK_FLOAT delzinv_kk, zlo_kk, shiftone_kk, zprd_kk;
  int nz_kk, order_kk, nlower_kk, nupper_kk;
  int dim_kk, lat1_kk, lat2_kk;   // inhomogeneous and lateral dim indices

  void allocate_device();
};

}    // namespace LAMMPS_NS

#endif
#endif
