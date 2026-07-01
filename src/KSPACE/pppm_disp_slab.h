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
KSpaceStyle(pppm/disp/slab,PPPMDispSlab);
// clang-format on
#else

#ifndef LMP_PPPM_DISP_SLAB_H
#define LMP_PPPM_DISP_SLAB_H

#include "kspace.h"

namespace LAMMPS_NS {

// Mesh (1-D grid) accelerated damped slab-based dispersion Ewald.
// The dispersion-weighted (geometric-mixing) density varies only in the chosen
// inhomogeneous dimension (x, y, or z; default z), so the smooth reciprocal
// part is a 1-D convolution: spread the B-weighted density onto a 1-D grid,
// FFT, apply the influence function, inverse-FFT the force field, and
// interpolate.  The real-space slab correction corr() and the H/IK pressure
// profiles are shared (identical math) with ewald/disp/slab.  Supports the
// damped (SSB, kspace_modify damp yes) and compact-switch (CSB, damp compact)
// variants; the CSB variant's shell mean field is removed in real space by
// corr_csb() so the matched pair supplies the exact 3-D shell interaction.

class PPPMDispSlab : public KSpace {
 public:
  PPPMDispSlab(class LAMMPS *);
  ~PPPMDispSlab() override;
  void init() override;
  void setup() override;
  void settings(int, char **) override;
  void compute(int, int) override;
  int modify_param(int, char **) override;
  double memory_usage() override;

  // long-range pressure profiles P_T(z), P_N(z) (shared with ewald/disp/slab)
  // contour 0 = Harasima (H), 1 = Irving-Kirkwood (IK).
  int contour_flag, profile_flag, npro;
  double *pt_profile, *pn_profile;

 protected:
  int dim;               // inhomogeneous dimension: 0=x, 1=y, 2=z (default 2)
  int lat1, lat2;        // lateral dimensions = (dim+1)%3, (dim+2)%3
  int nz;                // # grid points along dim (power of two)
  int order;             // assignment/interpolation stencil order
  int damp_flag;         // 0 = damped (SSB); 2 = compact switch (CSB)
  int corr_mode;         // damped correction: 0 = raw pairwise, 1 = binned
  int corr_switch;       // 1 = damped + matched lj/cut/dispswitch pair: the smooth
                         //     switched corr is merged into the influence function
                         //     (no real-space corr step; see influence_function)
  double bin_dz_user;    // user-requested bin width for corr bin (0 => auto)
  int bin_nbins;         // calibrated # corr bins (0 => not calibrated)
  double g_ewald_set;    // splitting parameter actually used
  double sw_width;       // compact-switch width Delta (read from the matched pair style)
  int switch_order;      // smoothstep continuity C^n (n=3 septic default, 5, or 7)

  double volume, cutoff, rc2, area, zprd, zlo;
  double delzinv;        // nz/zprd
  double shiftone;       // grid-assignment shift (order parity)
  int nlower, nupper;    // stencil bounds [nlower..nupper]
  double *B;             // per-type dispersion amplitude B[t] = 2 sqrt(eps) sigma^3

  // z-grid fields (global, length nz)
  double *dens;         // spread B-weighted density (real)
  double *fre, *fim;    // FFT workspace (real/imag)
  double *Gk;           // de-convolved energy influence function (per grid mode)
  double *GTk, *GNk;    // de-convolved tangential/normal virial influence (compact switch)
  double *fz_grid;      // z-force field on the grid
  double *ugrid;        // per-atom potential field (for eatom/vatom)
  double *uTgrid, *uNgrid;    // per-atom tangential/normal virial fields (compact switch)

  // CSB shell correction: tabulated plane (mean-field) energy/z-force/virial kernels
  // of S*u over [rcut, rcut+Delta], subtracted in real space so the matched pair's
  // exact 3-D shell interaction replaces the reciprocal sum's plane mean field.
  double *wEgrid, *wFgrid, *wTgrid, *wNgrid;
  // smooth switched corr (corr_switch): tabulated plane energy kernel w2(|dz|) of
  // corr_e(r) = u_smooth(r) - [r>rcut] S(r)/r^6 over [0, rcut+Delta], and its
  // 1-D Fourier transforms (corr_tilde) merged per-mode into Gk/GTk/GNk.
  double *cWgrid;
  int ncgrid;
  double cwdz;
  double u_smooth(double r);       // smooth (Gaussian-screened) 1/r^6, Taylor near 0
  void build_corr_kernels();       // tabulate w2 by quadrature at setup
  // w2t = 2 int w2 cos(kz) dz;  kw2p = k dW~2/dk = -2k int z w2 sin(kz) dz
  void corr_tilde(double k, double &w2t, double &kw2p);
  int nwgrid;
  double wdz;

  double **rho_coeff;     // B-spline assignment polynomial coefficients
  int order_allocated;    // order at last rho_coeff allocation

  double e_recip_mesh;    // mesh reciprocal energy (for the zz virial trace)
  double corr_energy;     // damped correction energy (for the virial trace)
  double estimated_force_accuracy;
  double *peatom;    // per-atom kspace energy buffer (zz virial trace)
  int nmax;

  void set_grid_params();       // geometry, delzinv, stencil params
  void make_rho();              // spread density to z grid (global)
  void poisson();               // FFT, influence fn, energy/force/per-atom field
  void fieldforce();            // interpolate z-force (and per-atom e/v) to atoms
  void influence_function();    // fill Gk (damped/compact, de-convolved)

  // compact-switch (CSB) reciprocal coefficients (copied from ewald/disp/slab)
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
  void estimate_params();                        // choose g_ewald and the z grid size nz
  double compute_qopt(int ngrid, int ord);       // Hockney-Eastwood 1-D qopt (z aliases)

  // CSB shell correction (compact switch); shared math with ewald/disp/slab
  void build_shell_vkernels();
  void shell_vkernel(double adz, double &wE, double &wF, double &wT, double &wN);
  void corr_csb();
  void corr_csb_raw();
  void corr_csb_bin();

  // shared with ewald/disp/slab (identical formulas)
  void corr();
  void corr_raw();
  void corr_bin();
  void corr_raw_force(double *fzloc);    // lean exact pairwise corr force (calibration ref)
  void corr_bin_force(int nbins, double *fzloc);    // lean force-only binned corr (for calibration)
  virtual void calibrate_bin();                     // tie corr bin count to target accuracy
  void corr_kernels(double x2, double &w2, double &f2, double &pt2);
  void compute_pressure_profile();
  double ik_phi(double h), ik_psi(double h);
  void cisi(double x, double &si, double &ci);
  void sici_chain(double x, double *Aarr, double *Barr);
};

}    // namespace LAMMPS_NS

#endif
#endif
