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

// Mesh (z-grid) accelerated damped slab-based dispersion Ewald.
// The dispersion-weighted (geometric-mixing) density varies only in z, so the
// smooth reciprocal part is a 1-D convolution in z: spread the B-weighted
// density onto a 1-D z grid, FFT in z, apply the damped influence function,
// inverse-FFT the z-force field, and interpolate.  The real-space slab
// correction corr() and the H/IK pressure profiles are shared (identical math)
// with ewald/disp/slab.  Damped (SSB) only.

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
  int nz;                // # z grid points (power of two)
  int order;             // assignment/interpolation stencil order
  int corr_mode;         // damped correction: 0 = raw pairwise, 1 = z-binned
  double bin_dz_user;    // user-requested z-bin width for corr bin (0 => auto)
  int bin_nbins;         // calibrated # corr bins (0 => not calibrated)
  double g_ewald_set;    // splitting parameter actually used

  double volume, cutoff, rc2, area, zprd, zlo;
  double delzinv;        // nz/zprd
  double shiftone;       // grid-assignment shift (order parity)
  int nlower, nupper;    // stencil bounds [nlower..nupper]
  double *B;             // per-type dispersion amplitude B[t] = 2 sqrt(eps) sigma^3

  // z-grid fields (global, length nz)
  double *dens;         // spread B-weighted density (real)
  double *fre, *fim;    // FFT workspace (real/imag)
  double *Gk;           // de-convolved energy influence function (per grid mode)
  double *fz_grid;      // z-force field on the grid
  double *ugrid;        // per-atom potential field (for eatom/vatom)

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
  void influence_function();    // fill Gk (damped, de-convolved)
  void fft1d(double *re, double *im, int n, int sign);    // radix-2 in-place FFT
  void compute_rho_coeff();                      // B-spline coefficients (LAMMPS PPPM convention)
  void compute_rho1d(double dz, double *w);      // assignment weights at offset dz
  void compute_drho1d(double dz, double *dw);    // d(assignment weights)/d(dz)
  void estimate_params();                        // choose g_ewald and the z grid size nz
  double compute_qopt(int ngrid, int ord);       // Hockney-Eastwood 1-D qopt (z aliases)

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
