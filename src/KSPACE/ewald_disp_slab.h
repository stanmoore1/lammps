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

  // long-range pressure profiles P_T(z), P_N(z) on a z-grid (npro points), filled
  // by compute_pressure_profile() when kspace_modify pressure/profile is on.
  // contour 0 = Harasima (H), 1 = Irving-Kirkwood (IK).
  int contour_flag, profile_flag, npro;
  double *pt_profile, *pn_profile;

 protected:
  int dim;               // inhomogeneous dimension: 0=x, 1=y, 2=z (default 2)
  int lat1, lat2;        // lateral dimensions = (dim+1)%3, (dim+2)%3
  int kmax, kcount;    // # of 1-D wavevectors (modes k=0..kmax-1), kcount=kmax
  int kmax_created;
  int kmax_user;         // user override via kspace_modify kmax (0 if unset)
  int damp_flag;         // 0 = non-damped (SB), 1 = damped (SSB), 2 = compact switch (CSB)
  int corr_mode;         // damped correction: 0 = raw pairwise, 1 = binned (faster)
  double bin_dz_user;    // requested bin width (0 => default)
  double sw_width;       // compact-switch width Delta (read from the matched pair style)
  double volume, cutoff, rc2;
  double unitk;                       // 2*pi/Lz
  double estimated_force_accuracy;    // predicted RMS per-atom force error
  double corr_energy;                 // damped correction energy (for the virial trace)
  int nmax;                           // size of per-atom arrays

  double *GU, *GF, *GT;    // precomputed coeffs: energy, z-force, tangential pressure
  double *GN;              // normal-pressure coeffs (compact switch; explicit, not via trace)
  double *ek;              // per-atom reciprocal z-force accumulator
  double *peatom;          // per-atom kspace energy buffer (for the zz virial trace)
  double *sfacrl, *sfacim, *sfacrl_all, *sfacim_all;
  double **cs, **sn;    // per-atom cos/sin of k*unitk*z
  double *B;            // per-type dispersion amplitude, B[i]=sqrt(|lj4[i][i]|)

  void eik_dot_r();
  void init_coeffs();
  void coeffs();
  double gf_of_k(int k);     // force coefficient GF for a single z mode k>=1
  // compact-switch (CSB) helpers: smoothed-truncation reciprocal coefficients
  double switch_S(double t);     // C3 septic smoothstep
  double switch_dS(double t);    // dS/dt = 140 t^3 (1-t)^3
  double switch_trans5(double h);    // shell transition integral int S(r) r^-5 sin(h r) dr
  // shell virial integrals int_rcut^{rcut+D} (S'u + S u') A_{T,N}(r,h) dr
  void switch_shell_virial(double h, double &sGT, double &sGN);
  double gu_switch(int k);     // GU[k] for the compact switch
  double gu0_switch();         // k=0 energy coefficient
  void estimate_params();    // choose g_ewald (damped) and kmax from target accuracy
  void allocate();
  void deallocate();
  void corr();                        // damped slab correction dispatcher
  void corr_raw();                    // exact pairwise (global z-gather) correction
  void corr_bin();                    // z-binned (1D particle-mesh, CIC) correction
  // compact-switch shell virial correction: subtract the plane (mean-field) virial
  // of (S u)' over [rcut, rcut+Delta] so the pair's exact full-u' shell virial
  // replaces it (removes the lateral-correlation pressure residual).  Mirrors the
  // damped corr_raw/corr_bin but on the shell slice.  Selected by corr_mode.
  double *wTgrid, *wNgrid;            // tabulated plane virial kernels w_T(dz), w_N(dz)
  int nwgrid;                         // grid points on [0, rcut+Delta]
  double wdz;                         // grid spacing
  void build_shell_vkernels();        // tabulate w_T, w_N at setup
  void shell_vkernel(double adz, double &wT, double &wN);    // interpolate at |dz|
  void corr_csb();                    // dispatcher (compact-switch virial correction)
  void corr_csb_raw();                // global z-gather (N^2)
  void corr_csb_bin();                // z-binned
  void compute_pressure_profile();    // P_T(z), P_N(z) profiles (H or IK contour)
  double ik_phi(double h);            // IK tangential building block Phi(h)
  double ik_psi(double h);            // IK normal building block Psi(h)
  void corr_kernels(double x2, double &w2, double &f2, double &pt2);    // shared kernel

  // generalized sine/cosine integrals via recurrence + continued fraction
  void cisi(double x, double &si, double &ci);
  void sici_chain(double x, double *Aarr, double *Barr);    // fills A[1..7], B[1..7]
};

}    // namespace LAMMPS_NS

#endif
#endif
