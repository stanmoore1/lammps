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

  // long-range Irving-Kirkwood pressure profiles P_T(z), P_N(z) (compute
  // stress/cartesian hook); the merged-damped kspace represents the same S*u tail.
  int pressure_profile_long(int, int, double, double, double *, double *) override;

 protected:
  int dim;             // inhomogeneous dimension: 0=x, 1=y, 2=z (default 2)
  int lat1, lat2;      // lateral dimensions = (dim+1)%3, (dim+2)%3
  int kmax, kcount;    // # of 1-D wavevectors (modes k=0..kmax-1), kcount=kmax
  int kmax_created;    // kmax at last cs/sn allocation
  int kmax_alloc;      // kmax at last mode-coefficient (GU/GF/GT/GN/sfac) allocation
  int kmax_user;       // user override via kspace_modify kmax (0 if unset)
  int mix_flag;        // C6 cross-term mixing: 0 = geometric, 1 = arithmetic (LB)
  int nchan;           // # dispersion channels: 1 (geometric) or 7 (arithmetic)
  double sw_width;     // dispersion switch width Delta (read from the matched pair)
  int sw_order;        // C^n switch order (read from the matched pair; default 3)
  double volume, cutoff, rc2;
  double unitk;                       // 2*pi/Lz
  double estimated_force_accuracy;    // predicted RMS per-atom force error
  int nmax;                           // size of per-atom arrays

  double *GU, *GF;    // precomputed coeffs: energy, z-force (tangential GT == GU, omitted)
  double *GN;         // normal-pressure coeffs (explicit per-mode strain derivative)
  double *ek;         // per-atom reciprocal z-force accumulator
  double *peatom;     // per-atom kspace energy buffer
  double *sfacrl, *sfacim, *sfacrl_all, *sfacim_all;
  double **cs, **sn;    // per-atom cos/sin of k*unitk*z
  double *B;            // per-type dispersion amplitude, B[i]=sqrt(|lj4[i][i]|)

  void eik_dot_r();
  void init_coeffs();
  void coeffs();
  double gf_of_k(int k);         // force coefficient GF for a single z mode k>=1
  double switch_S(double t);     // C3 septic smoothstep
  double switch_dS(double t);    // dS/dt = 140 t^3 (1-t)^3

  // Irving-Kirkwood pressure-profile building blocks (S*u tail; shared with
  // pppm/disp/planar / pppm/disp/planar).  All self-contained in the switched
  // dispersion potential (cutoff, sw_width, B, volume); independent of the solve.
  enum { PROF_T, PROF_N, PROF_PHI };
  void cisi(double x, double &si, double &ci);
  void sici_chain(double x, double *Aarr, double *Barr);
  void sici_compl_chain(double x, double *Carr, double *Darr);
  double prof_integrand(int which, double r, double h);
  double prof_shell(int which, double h);
  double ik_phi(double h), ik_psi(double h);
  void switch_shell_virial(double h, double &sGT, double &sGN);
  void shell_profile_virial(int nbins, double lo, double dz, double *dens_all, double *shellT,
                            double *shellN);
  void profile_GTGN_raw(int K, double *GTr, double *GNr);
  void profile_assemble(int K, int nbins, double lo, double width, const double *Sre,
                        const double *Sim, const double *GTr, const double *GNr,
                        const double *shellT, const double *shellN, double *pN, double *pT);
  void estimate_params();    // choose g_ewald and kmax from target accuracy
  void allocate();
  void deallocate();

  // smooth (switched-pair) damped correction, folded into the reciprocal
  // coefficients.  With the matched lj/disp/planar pair the 1/r^6 dispersion is
  // faded out by (1-S) over [rcut, rcut+Delta], so the corr potential corr_e(r) =
  // u_smooth(r) - [r>rcut] S(r)/r^6 vanishes smoothly at rcut+Delta.  Its energy
  // kernel w2(z) is tabulated by quadrature (build_corr_kernels); corr_tilde
  // Fourier-transforms it and merge_corr_coeffs folds W~2(k) into GU/GF/GT/GN so
  // the corr is diagonal in the reciprocal basis (E_corr = sum_n [W~2(k_n)/Lz]|S_n|^2)
  // -- no real-space corr step.
  double u_smooth(double r);    // smooth (Gaussian-screened) 1/r^6, Taylor near 0
  double *cWgrid;               // tabulated switched corr energy kernel (= pre*cWraw)
  double *cWraw;                // box-INDEPENDENT kernel integral int r*corr_e dr;
                                //   precomputed once (g_ewald/cutoff/Delta fixed),
                                //   rescaled by pre=2*pi/area each setup (NPT hot loop)
  int ncgrid;                   // grid points on [0, rcut+Delta]
  double cwdz;                  // grid spacing
  void build_corr_kernels();    // tabulate the switched corr w2 at setup
  void corr_tilde(double k, double &w2t, double &kw2p);    // W~2(k) and k dW~2/dk (exact)
  void merge_corr_coeffs();                                // add the corr to GU/GF/GT/GN
  // NPT-proof merge: W~2(k) and k dW~2/dk are (2*pi/area) times BOX-INDEPENDENT Fourier
  // transforms of cWraw.  Tabulate those transforms once on a uniform wavenumber grid,
  // then each setup just interpolate at the shifted modes k_m = m*(2*pi/Lz) and rescale
  // by 2*pi/area -- O(kmax) instead of O(kmax x quadrature) per box change.
  double *Araw_tab, *Braw_tab;    // A(kap)=2 int cWraw cos(kap z) dz;  B=2 int z cWraw sin
  int nkap;                       // table length
  double kap_dk, kap_max;         // wavenumber grid spacing and covered range
  int corr_ft_version;            // bumped whenever the FT tables are (re)built
  // parameters cWraw was built with; a change (rcut, Delta, g_ewald) invalidates it
  double corr_cut_cached, corr_dz_cached, corr_g_cached;
  void build_corr_ft_tables(double kap_need);          // (re)build the FT tables (grow-only)
  void ft_interp(double kap, double &A, double &B);    // cubic-Lagrange interpolation
};

}    // namespace LAMMPS_NS

#endif
#endif
