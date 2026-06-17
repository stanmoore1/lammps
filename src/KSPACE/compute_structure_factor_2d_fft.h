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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(structure/factor/2d/fft,ComputeStructureFactor2DFFT);
// clang-format on
#else

#ifndef LMP_COMPUTE_STRUCTURE_FACTOR_2D_FFT_H
#define LMP_COMPUTE_STRUCTURE_FACTOR_2D_FFT_H

#include "compute.h"
#include "lmpfftsettings.h"

namespace LAMMPS_NS {

class ComputeStructureFactor2DFFT : public Compute {
 public:
  ComputeStructureFactor2DFFT(class LAMMPS *, int, char **);
  ~ComputeStructureFactor2DFFT() override;
  void init() override;
  void setup() override;
  void compute_array() override;

 protected:
  int me, nprocs;

  // in-plane k-vector bookkeeping (mirrors ComputeStructureFactor2D)

  int kxmax, kymax, kmax, kcount, kmax2d, ksqmax;
  int nbins;
  double gsqmx, volume;
  double unitk[2];
  int *kxvecs, *kyvecs;
  int kunique;
  int *norms, *ksq2unique;

  void coeffs();
  void allocate_kvecs();
  void deallocate_kvecs();

  // KB particle-mesh machinery (xy only; z stays real-space layers)

  int order;
  double kb_beta, oversample;
  int nx_sf, ny_sf;
  int nlower, nupper;
  double shift, shiftone;
  double delxinv, delyinv;
  double *boxlo;
  int mesh_allocated;

  // z-bin slab decomposition across ranks
  int binlo, binhi, nbins_local;
  int *recvcount, *displs;        // reduce-scatter / allgather bookkeeping

  FFT_SCALAR *meshloc;            // this proc's spread density, all bins (nbins*ny*nx)
  FFT_SCALAR *meshown;            // summed density for this proc's bins
  FFT_SCALAR *work;              // complex 2D-FFT work array
  FFT_SCALAR **rho1d;
  class FFT3d *fft2d;            // serial (MPI_COMM_SELF) 2D transform, per bin

  double *rhohat_re, *rhohat_im;        // owned bins x kcount
  double *rhohat_re_all, *rhohat_im_all; // all bins x kcount (after allgather)

  int **part2grid;
  int *binofatom;
  int *counts, *counts_all;
  int nmax;

  void set_grid();
  void allocate_mesh();
  void deallocate_mesh();
  void atom2bin();
  void particle_map();
  void make_rho();
  void compute_kb1d(const FFT_SCALAR &, const FFT_SCALAR &);
  double kb_window(int, int);
  int factorable(int);
  static double bessel_i0(double);
};

}    // namespace LAMMPS_NS

#endif
#endif
