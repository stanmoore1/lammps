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

// Mesh (1-D grid) accelerated smooth-damped slab-based dispersion Ewald.
// The dispersion-weighted (geometric-mixing) density varies only in the chosen
// inhomogeneous dimension (x, y, or z; default z), so the smooth reciprocal
// part is a 1-D convolution: spread the B-weighted density onto a 1-D grid,
// FFT, apply the influence function, inverse-FFT the force field, and
// interpolate.  The real-space slab correction is folded into the influence
// function (diagonal in the grid's Fourier basis), so there is no separate
// real-space correction step.  Matched to the lj/cut/dispswitch pair style.

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

 protected:
  int dim;               // inhomogeneous dimension: 0=x, 1=y, 2=z (default 2)
  int lat1, lat2;        // lateral dimensions = (dim+1)%3, (dim+2)%3
  int nz;                // # grid points along dim (power of two)
  int order;             // assignment/interpolation stencil order
  double g_ewald_set;    // splitting parameter actually used
  double sw_width;       // dispersion switch width Delta (read from the matched pair)

  double volume, cutoff, rc2, area, zprd, zlo;
  double delzinv;        // nz/zprd
  double shiftone;       // grid-assignment shift (order parity)
  int nlower, nupper;    // stencil bounds [nlower..nupper]
  double *B;             // per-type dispersion amplitude B[t] = 2 sqrt(eps) sigma^3

  // z-grid fields (global, length nz)
  double *dens;         // spread B-weighted density (real)
  double *fre, *fim;    // FFT workspace (real/imag)
  double *Gk;           // de-convolved energy influence function (per grid mode)
  double *GTk, *GNk;    // de-convolved tangential/normal virial influence
  double *fz_grid;      // z-force field on the grid
  double *ugrid;        // per-atom potential field (for eatom)
  double *uTgrid, *uNgrid;    // per-atom tangential/normal virial fields

  // smooth switched corr: tabulated plane energy kernel w2(|dz|) of
  // corr_e(r) = u_smooth(r) - [r>rcut] S(r)/r^6 over [0, rcut+Delta], and its
  // 1-D Fourier transforms (corr_tilde) merged per-mode into Gk/GTk/GNk.
  double *cWgrid;
  int ncgrid;
  double cwdz;
  double u_smooth(double r);       // smooth (Gaussian-screened) 1/r^6, Taylor near 0
  void build_corr_kernels();       // tabulate w2 by quadrature at setup
  // w2t = 2 int w2 cos(kz) dz;  kw2p = k dW~2/dk = -2k int z w2 sin(kz) dz
  void corr_tilde(double k, double &w2t, double &kw2p);
  double switch_S(double t);     // C3 septic smoothstep
  double switch_dS(double t);    // dS/dt = 140 t^3 (1-t)^3

  double **rho_coeff;     // B-spline assignment polynomial coefficients
  int order_allocated;    // order at last rho_coeff allocation

  double estimated_force_accuracy;
  double *peatom;    // per-atom kspace energy buffer
  int nmax;

  void set_grid_params();       // geometry, delzinv, stencil params
  void make_rho();              // spread density to z grid (global)
  void poisson();               // FFT, influence fn, energy/force/per-atom field
  void fieldforce();            // interpolate z-force (and per-atom e/v) to atoms
  void influence_function();    // fill Gk/GTk/GNk (damped, merged corr, de-convolved)

  void fft1d(double *re, double *im, int n, int sign);    // radix-2 in-place FFT
  void compute_rho_coeff();                      // B-spline coefficients (LAMMPS PPPM convention)
  void compute_rho1d(double dz, double *w);      // assignment weights at offset dz
  void compute_drho1d(double dz, double *dw);    // d(assignment weights)/d(dz)
  void estimate_params();                        // choose g_ewald and the z grid size nz
  double compute_qopt(int ngrid, int ord);       // Hockney-Eastwood 1-D qopt (z aliases)
};

}    // namespace LAMMPS_NS

#endif
#endif
