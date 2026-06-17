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
struct TagPPPMDispSlab_poisson_virial_csb{};
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

// damped slab-correction kernels (corr raw + corr bin, all on device)
struct TagPPPMDispSlab_corr_raw{};
struct TagPPPMDispSlab_corr_raw_force{};
struct TagPPPMDispSlab_corr_bin_zero{};
struct TagPPPMDispSlab_corr_bin_spread{};
struct TagPPPMDispSlab_corr_bin_ktable{};
struct TagPPPMDispSlab_corr_bin_conv{};
struct TagPPPMDispSlab_corr_bin_conv_w{};
struct TagPPPMDispSlab_corr_bin_energy{};
struct TagPPPMDispSlab_corr_bin_interp{};
struct TagPPPMDispSlab_corr_bin_interp_force{};
struct TagPPPMDispSlab_corr_calib_err{};
struct TagPPPMDispSlab_peratom_finalize{};

// compact-switch (CSB) shell correction (exact pairwise, device)
struct TagPPPMDispSlab_corr_csb_raw{};

// double-precision (energy, tangential virial) reduction accumulator for corr
struct s_PPPMDispSlabCorr {
  double e, vt;
  KOKKOS_INLINE_FUNCTION s_PPPMDispSlabCorr() { e = 0.0; vt = 0.0; }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_PPPMDispSlabCorr &rhs) { e += rhs.e; vt += rhs.vt; }
};
typedef struct s_PPPMDispSlabCorr s_corr;

// (tangential, normal) virial reduction accumulator for the compact-switch mesh
struct s_PPPMDispSlabVir {
  double vt, vn;
  KOKKOS_INLINE_FUNCTION s_PPPMDispSlabVir() { vt = 0.0; vn = 0.0; }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_PPPMDispSlabVir &rhs) { vt += rhs.vt; vn += rhs.vn; }
};
typedef struct s_PPPMDispSlabVir s_vir;

// (energy, tangential, normal) reduction accumulator for the CSB shell correction
struct s_PPPMDispSlabCsb {
  double e, vt, vn;
  KOKKOS_INLINE_FUNCTION s_PPPMDispSlabCsb() { e = 0.0; vt = 0.0; vn = 0.0; }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_PPPMDispSlabCsb &rhs) { e += rhs.e; vt += rhs.vt; vn += rhs.vn; }
};
typedef struct s_PPPMDispSlabCsb s_csb;

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
  void operator()(TagPPPMDispSlab_poisson_virial_csb, const int&, s_vir&) const;

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
  void operator()(TagPPPMDispSlab_corr_raw, const int&, s_corr&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_raw_force, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_bin_zero, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_bin_spread, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_bin_ktable, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_bin_conv, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_bin_conv_w, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_bin_energy, const int&, s_corr&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_bin_interp, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_bin_interp_force, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_calib_err, const int&, double&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_peratom_finalize, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPPPMDispSlab_corr_csb_raw, const int&, s_csb&) const;

  // assignment weights w[0..order-1] at fractional offset dz (Horner in dz)
  KOKKOS_INLINE_FUNCTION
  void compute_rho1d_kk(const double dz, double *w) const {
    for (int s = 0; s < order_kk; s++) {
      double r = 0.0;
      for (int l = order_kk - 1; l >= 0; l--) r = d_rho_coeff(l, s) + r * dz;
      w[s] = r;
    }
  }

  // d/d(dz) of the assignment weights (energy-conserving B-spline corr force)
  KOKKOS_INLINE_FUNCTION
  void compute_drho1d_kk(const double dz, double *dw) const {
    for (int s = 0; s < order_kk; s++) {
      double r = 0.0;
      for (int l = order_kk - 1; l >= 1; l--) r = l * d_rho_coeff(l, s) + r * dz;
      dw[s] = r;
    }
  }

  // damped slab-correction kernels at squared z-separation x2 = (z_i-z_j)^2
  // (device port of PPPMDispSlab::corr_kernels)
  KOKKOS_INLINE_FUNCTION
  void corr_kernels_kk(const double x2, double &w2, double &f2, double &pt2) const {
    const double g2 = g_ewald_kk * g_ewald_kk;
    const double g4 = g2 * g2, g6 = g4 * g2, g8 = g4 * g4, g10 = g8 * g2, g12 = g10 * g2;
    const double rc4 = rc2_kk * rc2_kk, rc6 = rc4 * rc2_kk;
    const double inv_area = 1.0 / area_kk;

    if (x2 < 1.0e-3) {
      const double x4 = x2 * x2, x6 = x4 * x2;
      w2 = 0.5 * MathConst::MY_PI *
          (0.5 * g4 - x2 * g6 / 3.0 + x4 * g8 / 8.0 - x6 * g10 / 30.0 +
           exp(-rc2_kk * g2) * (1.0 / rc4 + g2 / rc2_kk) - 1.0 / rc4) * inv_area;
      f2 = 2.0 * MathConst::MY_PI * (g6 / 6.0 - x2 * g8 / 8.0 + x4 * g10 / 20.0 - x6 * g12 / 72.0) * inv_area;
      pt2 = 0.5 * MathConst::MY_PI *
          (0.5 * g4 - x2 * g6 / 3.0 + x4 * g8 / 8.0 - x6 * g10 / 30.0 +
           exp(-rc2_kk * g2) *
               (3.0 / rc4 - 2.0 * x2 / rc6 + g4 + 3.0 * g2 / rc2_kk - x2 * g4 / rc2_kk -
                x2 * g2 / rc4) -
           3.0 / rc4 + 2.0 * x2 / rc6) * inv_area;
    } else {
      const double x4 = x2 * x2, x6 = x4 * x2;
      w2 = 0.5 * MathConst::MY_PI *
          (1.0 / x4 - exp(-x2 * g2) * (1.0 / x4 + g2 / x2) +
           exp(-rc2_kk * g2) * (1.0 / rc4 + g2 / rc2_kk) - 1.0 / rc4) * inv_area;
      f2 = 2.0 * MathConst::MY_PI *
          (1.0 / x6 - exp(-x2 * g2) * (1.0 / x6 + g2 / x4 + 0.5 * g4 / x2)) * inv_area;
      pt2 = 0.5 * MathConst::MY_PI *
          (1.0 / x4 - exp(-x2 * g2) * (1.0 / x4 + g2 / x2) +
           exp(-rc2_kk * g2) *
               (3.0 / rc4 - 2.0 * x2 / rc6 + g4 + 3.0 * g2 / rc2_kk - x2 * g4 / rc2_kk -
                x2 * g2 / rc4) -
           3.0 / rc4 + 2.0 * x2 / rc6) * inv_area;
    }
  }

  // CSB shell kernels at |dz| (device port of PPPMDispSlab::shell_vkernel)
  KOKKOS_INLINE_FUNCTION
  void shell_vkernel_kk(const double adz, double &wE, double &wF, double &wT, double &wN) const {
    if (adz >= nwgrid_kk * wdz_kk) { wE = wF = wT = wN = 0.0; return; }
    const double xx = adz / wdz_kk;
    int g = (int) xx;
    if (g >= nwgrid_kk) g = nwgrid_kk - 1;
    const double fr = xx - g;
    wE = d_wEgrid(g) * (1.0 - fr) + d_wEgrid(g + 1) * fr;
    wF = d_wFgrid(g) * (1.0 - fr) + d_wFgrid(g + 1) * fr;
    wT = d_wTgrid(g) * (1.0 - fr) + d_wTgrid(g + 1) * fr;
    wN = d_wNgrid(g) * (1.0 - fr) + d_wNgrid(g + 1) * fr;
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
  typename AT::t_double_1d d_Gk;          // energy influence function
  typename AT::t_double_1d d_GTk, d_GNk;  // tangential/normal virial influence (compact switch)
  typename AT::t_double_1d d_fz_grid;     // z-force field
  typename AT::t_double_1d d_ugrid;       // per-atom potential field
  typename AT::t_double_1d d_uTgrid, d_uNgrid;   // per-atom T/N virial fields (compact switch)
  typename AT::t_double_1d d_wEgrid, d_wFgrid, d_wTgrid, d_wNgrid;   // CSB shell kernel tables
  Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> h_dens;   // Allreduce staging

  typename AT::t_double_1d d_B;           // per-type amplitude B[ntypes+1]
  typename AT::t_double_2d d_rho_coeff;   // B-spline coefficients [order][order]

  // per-atom reciprocal+corr energy (device, kspace per-atom energy buffer)
  typename AT::t_double_1d d_peatom;
  int nmax_kk;              // current allocation size of d_peatom

  // per-atom output arrays (base KSpace eatom/vatom, aliased via DualViews)
  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  // --- corr raw (exact pairwise) device buffers ---
  typename AT::t_double_1d d_zall, d_ball;   // gathered z, B over all procs
  typename AT::t_double_1d d_fzref;          // calibration reference force
  int natoms_all_created;                    // d_zall/d_ball allocation size

  // --- corr bin (z-binned) device buffers ---
  typename AT::t_double_1d d_bdens;          // local binned density
  typename AT::t_double_1d d_dens_all;       // global binned density (after Allreduce)
  typename AT::t_double_1d d_Kw, d_Kpt;      // kernel tables (length nwin+1)
  typename AT::t_double_1d d_phiW, d_phiPT;  // convolved potentials (length nbins)
  typename AT::t_double_1d d_fzbin;          // calibration binned force
  Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace> h_bdens;   // Allreduce staging
  int nbins_created, nwin_created;           // bin-array allocation sizes

  // cached atom views
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d_randomread type;

  // scalar device copies of hot-path parameters
  KK_FLOAT delzinv_kk, zlo_kk, shiftone_kk, zprd_kk;
  int nz_kk, order_kk, nlower_kk, nupper_kk;
  int dim_kk, lat1_kk, lat2_kk;   // inhomogeneous and lateral dim indices

  // scalar device copies for the corr kernels
  double g_ewald_kk, rc2_kk, area_kk;        // damping, cutoff^2, lateral area
  double w2self_kk, pt2self_kk;              // corr raw self terms (x2 = 0)
  double delzc_kk, bindz_kk;                 // corr bin: 1/dz and dz
  int nbins_kk, nwin_kk, myoff_kk, natoms_all_kk;   // corr bin counts; corr raw offsets
  int nwgrid_kk;                             // CSB shell kernel table size
  double wdz_kk;                             // CSB shell kernel table spacing

  void allocate_device();
  void calibrate_bin() override;       // override: dispatch to calibrate_bin_kk
  void corr_kk();              // dispatch raw/bin on the device
  void corr_csb_kk();          // CSB shell correction (exact pairwise, device)
  void corr_raw_kk();          // exact pairwise corr (device)
  void corr_bin_kk();          // z-binned corr incl. O(nbins^2) convolution (device)
  void corr_raw_force_kk();    // force-only pairwise corr -> d_fzref (calibration)
  void corr_bin_force_kk(int nbins);   // force-only binned corr -> d_fzbin (calibration)
  void calibrate_bin_kk();     // size the corr bin grid on the device
  void corr_bin_setup(int nbins);      // compute nwin/dz, (re)allocate bin buffers
  void corr_gather();          // host MPI gather of (z,B) -> d_zall/d_ball
};

}    // namespace LAMMPS_NS

#endif
#endif
