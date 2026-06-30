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

// Mesh (1-D grid) accelerated planar dispersion Ewald (FFT version of
// ewald/disp/planar).  The dispersion-weighted (geometric-mixing) density varies
// only in the chosen inhomogeneous dimension (x, y, or z; default z), so the
// smooth reciprocal part of the C3-switched 1/r^6 is a 1-D convolution: spread
// the B-weighted density onto a 1-D grid, FFT, apply the de-convolved
// compact-switch influence function, inverse-FFT the force field, and
// interpolate.  The shell mean field is removed in real space by corr_shell() so
// the matched pair (lj/cut/dispplanar) supplies the exact 3-D shell interaction.
// The H/IK pressure profiles use the same formulas as ewald/disp/planar.

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

  // long-range Irving-Kirkwood pressure profiles P_T(z), P_N(z) on the caller's z grid
  // (compute stress/cartesian supplies the grid and allocates pN/pT).  (The Harasima
  // contour is the per-atom virial -- compute stress/atom -- not here.)
  int pressure_profile_long(int, int, double, double, double *, double *) override;

 protected:
  int dim;               // inhomogeneous dimension: 0=x, 1=y, 2=z (default 2)
  int lat1, lat2;        // lateral dimensions = (dim+1)%3, (dim+2)%3
  int nz;                // # grid points along dim (power of two)
  int order;             // assignment/interpolation stencil order
  int corr_mode;         // shell correction: 0 = raw pairwise, 1 = binned
  double bin_dz_user;    // user-requested bin width for corr bin (0 => auto)
  double sw_width;       // compact-switch width Delta (read from the matched pair style)
  // dispersion mixing rule for the C6 cross term:
  //   mix_flag 0 = geometric (C6_ij = sqrt(C6_ii C6_jj), single B-amplitude per type)
  //   mix_flag 1 = arithmetic / Lorentz-Berthelot
  //               (C6_ij = 4 sqrt(eps_i eps_j) ((sigma_i+sigma_j)/2)^6, 7-channel)
  // read from the pair style (force->pair->mix_flag); kspace_modify mix/disp overrides
  // it via the base-class KSpace::mixflag (0 = follow pair, 1 = force geometric, 2 = none).
  // Same layout/normalization as ewald/disp/planar.
  int mix_flag;
  int nchan;             // density channels: 1 (geom) or 7 (arith)

  double volume, cutoff, rc2, area, zprd, zlo;
  double delzinv;        // nz/zprd
  double shiftone;       // grid-assignment shift (order parity)
  int nlower, nupper;    // stencil bounds [nlower..nupper]
  // dispersion amplitudes.  geometric (mix_flag 0): B[t] = 2 sqrt(eps_t) sigma_t^3,
  // one per type (size n+1).  arithmetic (mix_flag 1): the 7-channel binomial
  // expansion B[7*t+j] = sigma_t^j sqrt(eps_t) c[j], c={1,sqrt6,sqrt15,sqrt20,
  // sqrt15,sqrt6,1} (size 7*n+7), so sum_j B[7*i+j] B[7*j_type+(6-j)] reproduces
  // 4 sqrt(eps_i eps_j) ((sigma_i+sigma_j)/2)^6.  See init_coeffs().
  double *B;

  void init_coeffs();    // set mix_flag/nchan, build the B amplitude array

  // z-grid fields (global, length nz; arithmetic uses nchan-strided channels)
  double *dens;         // spread B-weighted density (real); nz*nchan for arith
  double *fre, *fim;    // FFT workspace (real/imag)
  double *Gk;           // de-convolved energy influence function (per grid mode)
  double *GTk, *GNk;    // de-convolved tangential/normal virial influence (compact switch)
  double *fz_grid;      // z-force field on the grid (ik differentiation)
  double *ugrid;        // potential field IFFT[2 Gk rho_hat]
                        //   (per-atom e/v always; ad force also reads it)
  double *uTgrid, *uNgrid;    // per-atom tangential/normal virial fields (compact switch)

  // analytic differentiation (kspace_modify diff ad): the z-force is the exact
  // z-gradient of the mesh energy, f_z(i) = B_i*delzinv*sum_s drho1d[s]*ugrid[g_s],
  // minus the spurious self-force.  sf_coeff[0,1] are the 1-D (z-only) self-force
  // amplitudes (analog of pppm_disp's compute_sf_coeff_6); computed in setup()
  // from the z-alias precoefficients and the influence function Gk.  Default
  // (ik) leaves these unused and the ik path bit-identical.
  double sf_coeff[2];
  void compute_sf_coeff();    // fill sf_coeff[] (1-D z self-force calibration)

  // CSB shell correction: tabulated plane (mean-field) energy/z-force/virial kernels
  // of S*u over [rcut, rcut+Delta], subtracted in real space so the matched pair's
  // exact 3-D shell interaction replaces the reciprocal sum's plane mean field.
  double *wEgrid, *wFgrid, *wTgrid, *wNgrid;
  int nwgrid;
  double wdz;

  double **rho_coeff;     // B-spline assignment polynomial coefficients
  int order_allocated;    // order at last rho_coeff allocation

  double e_recip_mesh;    // mesh reciprocal energy
  double corr_energy;     // shell correction energy
  double estimated_force_accuracy;
  double *peatom;    // per-atom kspace energy buffer (zz virial trace)
  int nmax;

  void set_grid_params();       // geometry, delzinv, stencil params
  void make_rho();              // spread density to z grid (global)
  void poisson();               // FFT, influence fn, energy/force/per-atom field
  void fieldforce();            // interpolate z-force (and per-atom e/v) to atoms
  void influence_function();    // fill Gk/GTk/GNk (compact switch, de-convolved)

  // compact-switch reciprocal coefficients (copied from ewald/disp/planar)
  double switch_S(double t);     // C3 septic smoothstep
  double switch_dS(double t);    // dS/dt = 140 t^3 (1-t)^3
  double switch_trans5(double h);                              // energy shell integral
  void switch_shell_virial(double h, double &sGT, double &sGN);    // shell virial integrals
  double gu_switch(int k);     // GU[k] at mesh mode k for the compact switch
  double gu0_switch();         // k=0 energy coefficient
  void sici_compl_chain(double x, double *Carr, double *Darr);    // C[1..7], D[1..7]

  void fft1d(double *re, double *im, int n, int sign);    // radix-2 in-place FFT
  void compute_rho_coeff();                      // B-spline coefficients (LAMMPS PPPM convention)
  void compute_rho1d(double dz, double *w);      // assignment weights at offset dz
  void compute_drho1d(double dz, double *dw);    // d(assignment weights)/d(dz)
  void estimate_params();                        // choose the z grid size nz
  double compute_qopt(int ngrid, int ord);       // Hockney-Eastwood 1-D qopt (z aliases)

  // compact-switch shell correction; shared math with ewald/disp/planar
  void build_shell_vkernels();
  void shell_vkernel(double adz, double &wE, double &wF, double &wT, double &wN);
  void corr_shell();
  void corr_shell_raw();
  void corr_shell_bin();
  // shell-correction virial per profile bin (shellT[g], shellN[g]); dispatches on
  // corr_mode so the contour profile uses the SAME real-space correction as the box
  // average (raw = exact per-atom shell virial binned by z; bin = density convolution).
  void shell_profile_virial(int nbins, double lo, double dz, double *dens_all, double *shellT,
                            double *shellN);

  // pressure-profile scalar building blocks (no per-atom data; shared by the host
  // pressure_profile_long and the Kokkos device override so the reciprocal
  // double-sum / coefficient math is written once):
  //   profile_kmax     : force-accuracy mode cutoff K_prof (<= nz/2-1) for the profile
  //   profile_GTGN_raw : raw per-mode tangential/normal box-pressure coefficients
  //   profile_Bt       : per-type single-channel structure-factor amplitude
  //   profile_assemble : S_n S_m C_{n,m} double sum + bin assembly - shell
  int profile_kmax();
  int prof_kmax_cached;     // cached force-accuracy mode cutoff (0 = not yet computed)
  int prof_kmax_nz;         // nz at which prof_kmax_cached was computed
  double prof_kmax_zprd;    // zprd at which prof_kmax_cached was computed
  void profile_GTGN_raw(int K, double *GTr, double *GNr);
  void profile_Bt(double *Bt);
  void profile_assemble(int K, int nbins, double lo, double width, const double *Sre,
                        const double *Sim, const double *GTr, const double *GNr,
                        const double *shellT, const double *shellN, double *pN, double *pT);

  // pressure-profile building blocks (shared with ewald/disp/planar)
  double ik_phi(double h), ik_psi(double h);
  // compact-switch reweight of the local pressure-profile coefficients: the sharp
  // tail integral int_rcut^inf g(r) dr is made switch-aware by anchoring at the outer
  // cutoff rcut+Delta and adding int_rcut^{rcut+Delta} W(r) g(r) dr, W=S-S'r/6.
  enum { PROF_T, PROF_N, PROF_PHI };
  double prof_integrand(int which, double r, double h);    // potential-form g(r)
  double prof_shell(int which, double h);                  // int_rcut^c W(r) g(r) dr
  void cisi(double x, double &si, double &ci);
  void sici_chain(double x, double *Aarr, double *Barr);
};

}    // namespace LAMMPS_NS

#endif
#endif
