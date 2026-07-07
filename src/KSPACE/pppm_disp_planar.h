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
KSpaceStyle(pppm/disp/planar,PPPMDispPlanar);
// clang-format on
#else

#ifndef LMP_PPPM_DISP_PLANAR_H
#define LMP_PPPM_DISP_PLANAR_H

#include "kspace.h"

namespace LAMMPS_NS {

// Mesh (1-D grid) accelerated smooth-damped planar dispersion Ewald.
// The dispersion-weighted (geometric-mixing) density varies only in the chosen
// inhomogeneous dimension (x, y, or z; default z), so the smooth reciprocal
// part is a 1-D convolution: spread the B-weighted density onto a 1-D grid,
// FFT, apply the influence function, inverse-FFT the force field, and
// interpolate.  The real-space correction is folded into the influence
// function (diagonal in the grid's Fourier basis), so there is no separate
// real-space correction step.  Matched to the lj/disp/planar pair style.

class PPPMDispPlanar : public KSpace {
 public:
  PPPMDispPlanar(class LAMMPS *);
  ~PPPMDispPlanar() override;
  void init() override;
  void setup() override;
  void settings(int, char **) override;
  void compute(int, int) override;
  int modify_param(int, char **) override;
  double memory_usage() override;

  // long-range Irving-Kirkwood pressure profiles P_N(z), P_T(z) on the caller's z
  // grid (compute stress/cartesian supplies the grid and allocates pN/pT).  The
  // merged-damped kspace represents the identical switched tail S(r)*u(r) as the
  // compact-switch method, so the same S*u pressure building blocks apply.
  int pressure_profile_long(int, int, double, double, double *, double *) override;

 protected:
  int dim;               // inhomogeneous dimension: 0=x, 1=y, 2=z (default 2)
  int lat1, lat2;        // lateral dimensions = (dim+1)%3, (dim+2)%3
  int nz;                // # grid points along dim (power of two)
  int order;             // assignment/interpolation stencil order
  double g_ewald_set;    // splitting parameter actually used
  double sw_width;       // dispersion switch width Delta (read from the matched pair)
  int mix_flag;          // C6 cross-term mixing: 0 = geometric, 1 = arithmetic (LB)
  int nchan;             // # dispersion channels: 1 (geometric) or 7 (arithmetic)

  double volume, cutoff, rc2, area, zprd, zlo;
  double delzinv;        // nz/zprd
  double shiftone;       // grid-assignment shift (order parity)
  int nlower, nupper;    // stencil bounds [nlower..nupper]
  double *B;             // per-type dispersion amplitude(s): B[t] geometric, B[7t+j] arith

  // z-grid fields (global, length nz*nchan, channel-major dens[m*nz+g])
  double *dens;                 // spread B-weighted density (real)
  double *fre, *fim;            // FFT workspace (real/imag)
  double *rhat_re, *rhat_im;    // FFT'd density channel spectra (nchan*nz)
  double *Gk;                   // de-convolved energy influence function (per grid mode)
  double *GTk, *GNk;            // de-convolved tangential/normal virial influence
  double *fz_grid;              // z-force field(s) on the grid (nchan channels)
  double *ugrid;                // per-atom potential field(s) (for eatom)
  double *uTgrid, *uNgrid;      // per-atom tangential/normal virial fields

  // smooth switched corr: tabulated plane energy kernel w2(|dz|) of
  // corr_e(r) = u_smooth(r) - [r>rcut] S(r)/r^6 over [0, rcut+Delta], and its
  // 1-D Fourier transforms (corr_tilde) merged per-mode into Gk/GTk/GNk.
  double *cWgrid;
  double *cWraw;    // box-INDEPENDENT kernel integral int r*corr_e dr, precomputed once
  int ncgrid;
  double cwdz;
  double u_smooth(double r);    // smooth (Gaussian-screened) 1/r^6, Taylor near 0
  void build_corr_kernels();    // tabulate w2 by quadrature at setup
  // w2t = 2 int w2 cos(kz) dz;  kw2p = k dW~2/dk = -2k int z w2 sin(kz) dz (exact)
  void corr_tilde(double k, double &w2t, double &kw2p);
  double switch_S(double t);     // C3 septic smoothstep
  double switch_dS(double t);    // dS/dt = 140 t^3 (1-t)^3
  // NPT-proof influence function: W~2(k) = (2*pi/area) times box-independent Fourier
  // transforms of cWraw; tabulate those once and interpolate at the shifted grid modes
  // each setup instead of re-quadraturing (see ewald/disp/planar for the derivation).
  double *Araw_tab, *Braw_tab;    // A(kap)=2 int cWraw cos(kap z);  B=2 int z cWraw sin
  int nkap;                       // table length
  double kap_dk, kap_max;         // wavenumber grid spacing and covered range
  int corr_ft_version;            // bumped whenever the FT tables are (re)built (KK re-upload)
  void build_corr_ft_tables(double kap_need);          // (re)build the FT tables (grow-only)
  void ft_interp(double kap, double &A, double &B);    // cubic-Lagrange interpolation
  // parameters the box-independent corr table cWraw was built with; a change (rcut,
  // Delta, g_ewald) between runs invalidates it (else it is read on a new grid spacing)
  double corr_cut_cached, corr_dz_cached, corr_g_cached;

  double **rho_coeff;           // B-spline assignment polynomial coefficients
  int order_allocated;          // order at last rho_coeff allocation
  int nz_alloc, nchan_alloc;    // nz, nchan at last grid allocation (NPT: skip realloc)

  double estimated_force_accuracy;
  double *peatom;    // per-atom kspace energy buffer
  int nmax;

  void set_grid_params();               // geometry, delzinv, stencil params
  void make_rho();                      // spread density to z grid (global)
  void poisson();                       // FFT, influence fn, energy/force/per-atom field
  void fieldforce();                    // interpolate z-force (and per-atom e/v) to atoms
  virtual void influence_function();    // fill Gk/GTk/GNk (damped, merged corr, de-convolved)

  void fft1d(double *re, double *im, int n, int sign);    // radix-2 in-place FFT
  void compute_rho_coeff();                      // B-spline coefficients (LAMMPS PPPM convention)
  void compute_rho1d(double dz, double *w);      // assignment weights at offset dz
  void compute_drho1d(double dz, double *dw);    // d(assignment weights)/d(dz)
  void estimate_params();                        // choose g_ewald and the z grid size nz
  double compute_qopt(int ngrid, int ord);       // Hockney-Eastwood 1-D qopt (z aliases)
  // per-type self dispersion C6_tt = B[t]^2 (geometric) or (1/16) sum_j B[7t+j]B[7t+6-j]
  // (arithmetic); used only by the RMS/qopt magnitude estimates.
  inline double c6_self(int t) const
  {
    if (nchan == 1) return B[t] * B[t];
    double s = 0.0;
    for (int j = 0; j < 7; j++) s += B[7 * t + j] * B[7 * t + 6 - j];
    return s / 16.0;
  }

  // --- Irving-Kirkwood long-range pressure profile (switched S*u building blocks) ---
  // The kspace effective potential is S(r)*u(r) (S=0 inside rcut, ramps over the
  // shell, =1 beyond), so the profile reuses the compact-switch S*u pressure math:
  // the sharp tail anchored at rcut+Delta plus the switch-shell integral.
  double switch_trans5(double h);                                  // energy shell integral
  void switch_shell_virial(double h, double &sGT, double &sGN);    // shell virial integrals
  double gu_switch(int k);    // GU[k] at mode k for the switched tail (force-accuracy est.)
  void sici_compl_chain(double x, double *Carr, double *Darr);    // C[1..7], D[1..7]
  double ik_phi(double h), ik_psi(double h);    // IK tangential/normal building blocks
  enum { PROF_T, PROF_N, PROF_PHI };
  double prof_integrand(int which, double r, double h);    // potential-form g(r)
  double prof_shell(int which, double h);                  // int_rcut^c W(r) g(r) dr, W=S-S'r/6
  void shell_profile_virial(int nbins, double lo, double dz, double *dens_all, double *shellT,
                            double *shellN);
  // force-accuracy mode cutoff and the IK double-sum assembly
  int profile_kmax();
  int prof_kmax_cached;     // cached force-accuracy mode cutoff (0 = not yet computed)
  int prof_kmax_nz;         // nz at which prof_kmax_cached was computed
  double prof_kmax_zprd;    // zprd at which prof_kmax_cached was computed
  void profile_GTGN_raw(int K, double *GTr, double *GNr);
  void profile_assemble(int K, int nbins, double lo, double width, const double *Sre,
                        const double *Sim, const double *GTr, const double *GNr,
                        const double *shellT, const double *shellN, double *pN, double *pT);
  void cisi(double x, double &si, double &ci);
  void sici_chain(double x, double *Aarr, double *Barr);    // fills A[1..7], B[1..7]
};

}    // namespace LAMMPS_NS

#endif
#endif
