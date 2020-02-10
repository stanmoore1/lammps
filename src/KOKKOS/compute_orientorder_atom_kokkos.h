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

#ifdef COMPUTE_CLASS

ComputeStyle(orientorder/atom/kk,ComputeOrientOrderAtomKokkos<LMPDeviceType>)
ComputeStyle(orientorder/atom/kk/device,ComputeOrientOrderAtomKokkos<LMPDeviceType>)
ComputeStyle(orientorder/atom/kk/host,ComputeOrientOrderAtomKokkos<LMPHostType>)

#else

#ifndef LMP_COMPUTE_ORIENTORDER_ATOM_KOKKOS_H
#define LMP_COMPUTE_ORIENTORDER_ATOM_KOKKOS_H

#include "compute_orientorder_atom.h"

namespace LAMMPS_NS {

struct TagComputeOrientOrderAtom{};

class ComputeOrientOrderAtomKokkos : public ComputeOrientOrderAtom {
 public:
  ComputeOrientOrderAtomKokkos(class LAMMPS *, int, char **);
  ~ComputeOrientOrderAtomKokkos();
  void init();
  void compute_peratom();
  int *qlist;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeOrientOrderAtom, const int&) const;

 private:
  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_float_1d_randomread q;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  class NeighList *list;
  double *distsq;
  int *nearest;
  double **rlist;
  double **qnarray;
  double **qnm_r;
  double **qnm_i;

  KOKKOS_INLINE_FUNCTION
  void select3(int, int, double *, int *, double **);

  KOKKOS_INLINE_FUNCTION
  void calc_boop(double **rlist, int numNeighbors,
                 double qn[], int nlist[], int nnlist);

  KOKKOS_INLINE_FUNCTION
  double dist(const double r[]);

  KOKKOS_INLINE_FUNCTION
  double polar_prefactor(int, int, double);

  KOKKOS_INLINE_FUNCTION
  double associated_legendre(int, int, double);

  void init_clebsch_gordan();
  typedef Kokkos::View<double*, DeviceType> t_sna_1d;
  t_sna_1d cglist;                      // Clebsch-Gordan coeffs
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
