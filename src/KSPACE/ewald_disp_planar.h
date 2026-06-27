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
KSpaceStyle(ewald/disp/planar,EwaldDispPlanar);
// clang-format on
#else

#ifndef LMP_EWALD_DISP_PLANAR_H
#define LMP_EWALD_DISP_PLANAR_H

#include "kspace.h"

namespace LAMMPS_NS {

class EwaldDispPlanar : public KSpace {
 public:
  EwaldDispPlanar(class LAMMPS *);
  ~EwaldDispPlanar() override;
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
  // dispersion mixing rule for the C6 cross term:
  //   mix_flag 0 = geometric (C6_ij = sqrt(C6_ii C6_jj), single B-amplitude per type)
  //   mix_flag 1 = arithmetic / Lorentz-Berthelot
  //               (C6_ij = 4 sqrt(eps_i eps_j) ((sigma_i+sigma_j)/2)^6, 7-channel)
  // read from the pair style (force->pair->mix_flag); kspace_modify mix/disp overrides
  // it via the base-class KSpace::mixflag (0 = follow pair, 1 = force geometric, 2 = none).
  int mix_flag;
  int nchan;             // structure-factor channels per mode: 1 (geom) or 7 (arith)
  int corr_mode;         // shell correction: 0 = raw pairwise, 1 = binned (faster)
  double bin_dz_user;    // requested bin width (0 => default)
  double sw_width;       // compact-switch width Delta (read from the matched pair style)
  double volume, cutoff, rc2;
  double unitk;                       // 2*pi/Lz
  double estimated_force_accuracy;    // predicted RMS per-atom force error
  double corr_energy;                 // shell correction energy
  int nmax;                           // size of per-atom arrays

  double *GU, *GF, *GT;    // precomputed coeffs: energy, z-force, tangential pressure
  double *GN;              // normal-pressure coeffs (compact switch; explicit, not via trace)
  double *ek;              // per-atom reciprocal z-force accumulator
  double *peatom;          // per-atom kspace energy buffer (for the zz virial trace)
  double *sfacrl, *sfacim, *sfacrl_all, *sfacim_all;
  double **cs, **sn;    // per-atom cos/sin of k*unitk*z
  // dispersion amplitudes.  geometric (mix_flag 0): B[i] = sqrt(|lj4[i][i]|) =
  // 2 sqrt(eps_i) sigma_i^3, one per type (size n+1).  arithmetic (mix_flag 1):
  // the 7-channel binomial expansion B[7*i+j] = sigma_i^j sqrt(eps_i) c[j],
  // c[7]={1,sqrt6,sqrt15,sqrt20,sqrt15,sqrt6,1} (size 7*n+7), so that the cross
  // amplitude sum_j B[7*i+j] B[7*j+(6-j)] reproduces 4 sqrt(eps_i eps_j)
  // ((sigma_i+sigma_j)/2)^6.
  double *B;

  void eik_dot_r();
  void init_coeffs();
  void coeffs();
  double gf_of_k(int k);     // force coefficient GF for a single z mode k>=1
  // compact-switch helpers: smoothed-truncation reciprocal coefficients
  double switch_S(double t);     // C3 septic smoothstep
  double switch_dS(double t);    // dS/dt = 140 t^3 (1-t)^3
  double switch_trans5(double h);    // shell transition integral int S(r) r^-5 sin(h r) dr
  // shell virial integrals int_rcut^{rcut+D} (S'u + S u') A_{T,N}(r,h) dr
  void switch_shell_virial(double h, double &sGT, double &sGN);
  double gu_switch(int k);     // GU[k] for the compact switch
  double gu0_switch();         // k=0 energy coefficient
  void estimate_params();      // choose kmax from the target accuracy
  void allocate();
  void deallocate();
  // compact-switch shell correction: subtract the plane (mean-field) energy, z-force
  // and virial of S*u over [rcut, rcut+Delta] (what the reciprocal sum injects there
  // with a laterally-uniform density) so the matched pair's exact 3-D full-u shell
  // interaction replaces it (removes the lateral-correlation residual in energy AND
  // pressure).
  double *wEgrid, *wFgrid;            // tabulated plane energy / z-force kernels
  double *wTgrid, *wNgrid;            // tabulated plane virial kernels w_T(dz), w_N(dz)
  int nwgrid;                         // grid points on [0, rcut+Delta]
  double wdz;                         // grid spacing
  void build_shell_vkernels();        // tabulate w_E, w_F, w_T, w_N at setup
  void shell_vkernel(double adz, double &wE, double &wF, double &wT, double &wN);    // interp
  void corr_shell();                    // dispatcher (compact-switch shell correction)
  void corr_shell_raw();                // global z-gather (N^2)
  void corr_shell_bin();                // z-binned
  void compute_pressure_profile();    // P_T(z), P_N(z) profiles (H or IK contour)
  // shell-correction virial per profile bin (shellT[g], shellN[g]); dispatches on
  // corr_mode so the contour profile uses the SAME real-space correction as the box
  // average (raw = exact per-atom shell virial binned by z; bin = density convolution).
  void shell_profile_virial(double dz, double *dens_all, double *shellT, double *shellN);
  double ik_phi(double h);            // IK tangential building block Phi(h)
  double ik_psi(double h);            // IK normal building block Psi(h)
  // compact-switch shell correction for the local pressure-profile coefficients.
  // The Harasima (Tn,Nn) and IK (Phi,Psi) coefficients are sharp tail integrals
  // int_rcut^inf g(r) dr of a potential-form integrand g.  Made switch-aware by
  // anchoring the tail at the OUTER cutoff rcut+Delta and adding the shell integral
  // int_rcut^{rcut+Delta} W(r) g(r) dr, W(r) = S(r) - S'(r) r/6 = (S u)'/(6/r^7), the
  // force-reweight that makes the shell term identical to the global switch_shell_virial
  // (reduces to the sharp result as Delta->0).  which selects the integrand g:
  //   PROF_T (combo_GT, used by Tn and Psi), PROF_N (combo_GN, used by Nn),
  //   PROF_PHI (combo_phi, used by Phi).
  enum { PROF_T, PROF_N, PROF_PHI };
  double prof_integrand(int which, double r, double h);    // potential-form g(r)
  double prof_shell(int which, double h);                  // int_rcut^c W(r) g(r) dr

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
