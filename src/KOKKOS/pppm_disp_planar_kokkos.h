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
KSpaceStyle(pppm/disp/planar/kk,PPPMDispPlanarKokkos<LMPDeviceType>);
KSpaceStyle(pppm/disp/planar/kk/device,PPPMDispPlanarKokkos<LMPDeviceType>);
KSpaceStyle(pppm/disp/planar/kk/host,PPPMDispPlanarKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PPPM_DISP_PLANAR_KOKKOS_H
#define LMP_PPPM_DISP_PLANAR_KOKKOS_H

#include "pppm_disp_planar.h"
#include "fft3d_kokkos.h"
#include "fftdata_kokkos.h"
#include "kokkos_type.h"
#include "math_const.h"

namespace LAMMPS_NS {

// functor tags for the device kernels
struct TagPPPMDispPlanar_make_rho_zero{};
struct TagPPPMDispPlanar_make_rho{};
struct TagPPPMDispPlanar_dens_to_work{};
struct TagPPPMDispPlanar_poisson_energy{};
struct TagPPPMDispPlanar_poisson_virial{};
struct TagPPPMDispPlanar_poisson_uT_prep{};
struct TagPPPMDispPlanar_poisson_uT_copy{};
struct TagPPPMDispPlanar_poisson_uN_prep{};
struct TagPPPMDispPlanar_poisson_uN_copy{};
struct TagPPPMDispPlanar_poisson_fz_prep{};
struct TagPPPMDispPlanar_poisson_fz_copy{};
struct TagPPPMDispPlanar_poisson_u_prep{};
struct TagPPPMDispPlanar_poisson_u_copy{};
struct TagPPPMDispPlanar_fieldforce{};
struct TagPPPMDispPlanar_fieldforce_peratom{};
struct TagPPPMDispPlanar_peatom_zero{};
struct TagPPPMDispPlanar_peratom_finalize{};

// (tangential, normal) virial reduction accumulator for the mesh
struct s_PPPMDispPlanarVir {
  double vt, vn;
  KOKKOS_INLINE_FUNCTION s_PPPMDispPlanarVir() { vt = 0.0; vn = 0.0; }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_PPPMDispPlanarVir &rhs) { vt += rhs.vt; vn += rhs.vn; }
};
typedef struct s_PPPMDispPlanarVir s_vir;

template<class DeviceType>
class PPPMDispPlanarKokkos : public PPPMDispPlanar {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef FFTArrayTypes<DeviceType> FFT_AT;

  PPPMDispPlanarKokkos(class LAMMPS *);
  ~PPPMDispPlanarKokkos() override;
  void init() override;
  void setup() override;
  void compute(int, int) override;
  double memory_usage() override;

  // long-range Irving-Kirkwood pressure profile (compute stress/cartesian hook).
  // The host implementation PPPMDispPlanar::pressure_profile_long reads atom->x/type
  // directly, so sync the KK atom data to host first (the profile is a rare
  // diagnostic; no device kernel is warranted).
  int pressure_profile_long(int, int, double, double, double *, double *) override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_make_rho_zero, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_make_rho, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_dens_to_work, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_energy, const int&, double&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_virial, const int&, s_vir&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_uT_prep, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_uT_copy, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_uN_prep, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_uN_copy, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_fz_prep, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_fz_copy, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_u_prep, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_poisson_u_copy, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_fieldforce, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_fieldforce_peratom, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_peatom_zero, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispPlanar_peratom_finalize, const int&) const;

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
