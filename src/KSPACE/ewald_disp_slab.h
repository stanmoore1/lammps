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
  int corr_switch;       // 1 = damped corr uses the smooth switched-pair kernel (no rcut
                         //     force discontinuity -> high-order binning); set in init()
                         //     when damp_flag==1 and the pair supplies disp_switch_width
  double bin_dz_user;    // requested bin width (0 => default)
  int bin_nbins;         // calibrated # corr bins (0 => not calibrated)
  double sw_width;       // compact-switch width Delta (read from the matched pair style)
  int switch_order;      // smoothstep continuity C^n (n=3 septic default, 5, or 7)
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
  void corr_raw_force(double *fzloc);          // exact pairwise corr z-force (calibration ref)
  void corr_bin_force(int nbins, double *fzloc);    // binned corr z-force (calibration)
  void calibrate_bin();               // size the corr bin count to the target accuracy
  // compact-switch shell correction: subtract the plane (mean-field) energy, z-force
  // and virial of S*u over [rcut, rcut+Delta] (what the reciprocal sum injects there
  // with a laterally-uniform density) so the matched pair's exact 3-D full-u shell
  // interaction replaces it (removes the lateral-correlation residual in energy AND
  // pressure).  Mirrors the damped corr_raw/corr_bin but on the shell slice.
  double *wEgrid, *wFgrid;            // tabulated plane energy / z-force kernels
  double *wTgrid, *wNgrid;            // tabulated plane virial kernels w_T(dz), w_N(dz)
  int nwgrid;                         // grid points on [0, rcut+Delta]
  double wdz;                         // grid spacing
  void build_shell_vkernels();        // tabulate w_E, w_F, w_T, w_N at setup
  void shell_vkernel(double adz, double &wE, double &wF, double &wT, double &wN);    // interp
  void corr_csb();                    // dispatcher (compact-switch virial correction)
  void corr_csb_raw();                // global z-gather (N^2)
  void corr_csb_bin();                // z-binned
  void compute_pressure_profile();    // P_T(z), P_N(z) profiles (H or IK contour)
  double ik_phi(double h);            // IK tangential building block Phi(h)
  double ik_psi(double h);            // IK normal building block Psi(h)
  void corr_kernels(double x2, double &w2, double &f2, double &pt2);    // shared kernel

  // smooth (switched-pair) damped correction.  With the matched lj/cut/dispswitch
  // pair the 1/r^6 dispersion is faded out by (1-S) over [rcut, rcut+Delta], so the
  // corr potential corr_e(r) = u_smooth(r) - [r>rcut] S(r)/r^6 vanishes smoothly at
  // rcut+Delta (no force discontinuity at rcut) -> the binned corr converges at high
  // order.  f2 is analytic; w2/pt2 are tabulated by quadrature over [0, rcut+Delta].
  double u_smooth(double r);                 // smooth (Gaussian-screened) 1/r^6, Taylor near 0
  double *cWgrid, *cTgrid;                   // tabulated switched corr energy / tangential kernels
  int ncgrid;                                // grid points on [0, rcut+Delta]
  double cwdz;                               // grid spacing
  void build_corr_kernels();                 // tabulate the switched corr w2/pt2 at setup
  void corr_smooth_kernels(double adz, double &w2, double &f2, double &pt2);    // interp + analytic f2

  // generalized sine/cosine integrals via recurrence + continued fraction
  void cisi(double x, double &si, double &ci);
  void sici_chain(double x, double *Aarr, double *Barr);    // fills A[1..7], B[1..7]
  // complementary chain C[m]=A[m]_inf-A[m], D[m]=B[m]_inf-B[m] computed without the
  // pi/2 (Si) constant subtraction; gives the small tail coefficients (pi/48-A[5])
  // etc. directly so the high-k reciprocal coefficients are free of cancellation.
  void sici_compl_chain(double x, double *Carr, double *Darr);    // fills C[1..7], D[1..7]
};

}    // namespace LAMMPS_NS

#endif
#endif
