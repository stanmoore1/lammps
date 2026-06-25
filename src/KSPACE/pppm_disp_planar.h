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

  // long-range pressure profiles P_T(z), P_N(z) (shared with ewald/disp/planar)
  // contour 0 = Harasima (H), 1 = Irving-Kirkwood (IK).
  int contour_flag, profile_flag, npro;
  double *pt_profile, *pn_profile;

 protected:
  int dim;               // inhomogeneous dimension: 0=x, 1=y, 2=z (default 2)
  int lat1, lat2;        // lateral dimensions = (dim+1)%3, (dim+2)%3
  int nz;                // # grid points along dim (power of two)
  int order;             // assignment/interpolation stencil order
  int corr_mode;         // shell correction: 0 = raw pairwise, 1 = binned
  double bin_dz_user;    // user-requested bin width for corr bin (0 => auto)
  double sw_width;       // compact-switch width Delta (read from the matched pair style)

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

  // pressure-profile building blocks (shared with ewald/disp/planar)
  void compute_pressure_profile();
  double ik_phi(double h), ik_psi(double h);
  void cisi(double x, double &si, double &ci);
  void sici_chain(double x, double *Aarr, double *Barr);
};

}    // namespace LAMMPS_NS

#endif
#endif
