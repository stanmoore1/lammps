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

FixStyle(acks2/reax,FixACKS2Reax)

#else

#ifndef LMP_FIX_ACKS2_REAX_H
#define LMP_FIX_ACKS2_REAX_H

#include "fix_qeq_reax.h"

namespace LAMMPS_NS {

class FixACKS2Reax : public FixQEqReax {
 public:
  FixACKS2Reax(class LAMMPS *, int, char **);
  virtual ~FixACKS2Reax();
  void init();
  void init_storage();
  void pre_force(int);

 protected:
  double *b_s_acks2,bond_softness,**bcut; // acks2 parameters

  sparse_matrix X;
  double *Xdia_inv;
  double *X_diag;

  //BiCGStab storage
  double *g, *q_hat, *r_hat, *y, *z;

  void pertype_parameters(char*);
  void init_bondcut();
  void allocate_storage();
  void deallocate_storage();
  void allocate_matrix();
  void deallocate_matrix();

  void init_matvec();
  void compute_X();
  double calculate_X(double,double);
  void calculate_Q();

  int BiCGStab(double*,double*);
  void sparse_matvec_acks2(sparse_matrix*,sparse_matrix*,double*,double*);

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void more_forward_comm(double *);
  void more_reverse_comm(double *);
  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

  double parallel_norm( double*, int );
  double parallel_dot( double*, double*, int );
  double parallel_vector_acc( double*, int );

  void vector_sum(double*,double,double*,double,double*,int);
  void vector_add(double*, double, double*,int);
  void vector_copy(double*, double*,int);
};

}

#endif
#endif
