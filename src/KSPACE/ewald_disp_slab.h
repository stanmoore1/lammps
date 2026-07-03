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
KSpaceStyle(ewald/disp/slab,EwaldDispSlab);
// clang-format on
#else

#ifndef LMP_EWALD_DISP_SLAB_H
#define LMP_EWALD_DISP_SLAB_H

#include "kspace.h"

namespace LAMMPS_NS {

class EwaldDispSlab : public KSpace {
 public:
  EwaldDispSlab(class LAMMPS *);
  ~EwaldDispSlab() override;
  void init() override;
  void setup() override;
  void settings(int, char **) override;
  void compute(int, int) override;
  int modify_param(int, char **) override;
  double memory_usage() override;

 protected:
  int dim;               // inhomogeneous dimension: 0=x, 1=y, 2=z (default 2)
  int lat1, lat2;        // lateral dimensions = (dim+1)%3, (dim+2)%3
  int kmax, kcount;      // # of 1-D wavevectors (modes k=0..kmax-1), kcount=kmax
  int kmax_created;
  int kmax_user;         // user override via kspace_modify kmax (0 if unset)
  double sw_width;       // dispersion switch width Delta (read from the matched pair)
  double volume, cutoff, rc2;
  double unitk;                       // 2*pi/Lz
  double estimated_force_accuracy;    // predicted RMS per-atom force error
  int nmax;                           // size of per-atom arrays

  double *GU, *GF, *GT;    // precomputed coeffs: energy, z-force, tangential pressure
  double *GN;              // normal-pressure coeffs (explicit per-mode strain derivative)
  double *ek;              // per-atom reciprocal z-force accumulator
  double *peatom;          // per-atom kspace energy buffer
  double *sfacrl, *sfacim, *sfacrl_all, *sfacim_all;
  double **cs, **sn;    // per-atom cos/sin of k*unitk*z
  double *B;            // per-type dispersion amplitude, B[i]=sqrt(|lj4[i][i]|)

  void eik_dot_r();
  void init_coeffs();
  void coeffs();
  double gf_of_k(int k);     // force coefficient GF for a single z mode k>=1
  double switch_S(double t);     // C3 septic smoothstep
  double switch_dS(double t);    // dS/dt = 140 t^3 (1-t)^3
  void estimate_params();    // choose g_ewald and kmax from target accuracy
  void allocate();
  void deallocate();

  // smooth (switched-pair) damped correction, folded into the reciprocal
  // coefficients.  With the matched lj/cut/dispswitch pair the 1/r^6 dispersion is
  // faded out by (1-S) over [rcut, rcut+Delta], so the corr potential corr_e(r) =
  // u_smooth(r) - [r>rcut] S(r)/r^6 vanishes smoothly at rcut+Delta.  Its energy
  // kernel w2(z) is tabulated by quadrature (build_corr_kernels); corr_tilde
  // Fourier-transforms it and merge_corr_coeffs folds W~2(k) into GU/GF/GT/GN so
  // the corr is diagonal in the reciprocal basis (E_corr = sum_n [W~2(k_n)/Lz]|S_n|^2)
  // -- no real-space corr step.
  double u_smooth(double r);                 // smooth (Gaussian-screened) 1/r^6, Taylor near 0
  double *cWgrid;                            // tabulated switched corr energy kernel
  int ncgrid;                                // grid points on [0, rcut+Delta]
  double cwdz;                               // grid spacing
  void build_corr_kernels();                 // tabulate the switched corr w2 at setup
  void corr_tilde(double k, double &w2t, double &kw2p);    // W~2(k) and k dW~2/dk
  void merge_corr_coeffs();                  // add the corr to GU/GF/GT/GN
};

}    // namespace LAMMPS_NS

#endif
#endif
