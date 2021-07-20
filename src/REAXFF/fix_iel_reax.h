/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(iel/reax,FixIELReax)

#else

#ifndef LMP_FIX_IEL_REAX_H
#define LMP_FIX_IEL_REAX_H

#include "fix_qeq_reax.h"

namespace LAMMPS_NS {

class FixIELReax : public FixQEqReax {
 public:
  FixIELReax(class LAMMPS *, int, char **);
  virtual ~FixIELReax();
  void post_constructor();
  int setmask();
  void init();
  void init_storage();
  void initial_integrate(int);
  void setup_pre_force(int);
  void pre_force(int);
  void final_integrate();
  void reset_dt();

 private:

  int setup_flag;

  double *t_EL_Scf, *vt_EL_Scf, *at_EL_Scf, *s_EL_Scf, *vs_EL_Scf, *as_EL_Scf;

  double dtv,dtf,dth;

  double Omega_t;
  double Omega_s;
  double Omega;
  double gamma_t;
  double gamma_s;
  double Energy_s_init;
  double Energy_t_init;
  int thermo_flag;
  double tautemp_aux;
  double kelvin_aux_t;
  double kelvin_aux_s;
  double tgnhaux[4];
  double tvnhaux[4];
  double tnhaux[4];

  double sgnhaux[4];
  double svnhaux[4];
  double snhaux[4];

  double Chi_eq_iEL_Scf,aChi_eq_iEL_Scf,vChi_eq_iEL_Scf,x_last;

  double tolerance_t,tolerance_s;
  double q_last, r_last, d_last, b_last;

  void init_matvec();
  int CG(double*,double*);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  void sparse_matvec(sparse_matrix*,double*,double*,double);
  void calculate_Q();
  void kinaux(double &,double &);
  void Berendersen();
  void Nose_Hoover();
};

}

#endif
#endif
