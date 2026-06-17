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
ComputeStyle(structure/factor/fft,ComputeStructureFactorFFT);
// clang-format on
#else

#ifndef LMP_COMPUTE_STRUCTURE_FACTOR_FFT_H
#define LMP_COMPUTE_STRUCTURE_FACTOR_FFT_H

#include "compute.h"
#include "lmpfftsettings.h"

namespace LAMMPS_NS {

class ComputeStructureFactorFFT : public Compute {
 public:
  ComputeStructureFactorFFT(class LAMMPS *, int, char **);
  ~ComputeStructureFactorFFT() override;
  void init() override;
  void setup() override;
  void compute_array() override;

  // Grid3d ghost-cell callbacks (Grid3d::COMPUTE caller)
  void pack_reverse_grid(int, void *, int, int *) override;
  void unpack_reverse_grid(int, void *, int, int *) override;

 protected:
  int me, nprocs;

  // k-vector bookkeeping (mirrors ComputeStructureFactor so the FFT result is
  // binned over exactly the same shells as the direct-DFT reference compute)

  int kxmax, kymax, kzmax;
  int kcount, kmax, kmax3d, ksqmax;
  double gsqmx, volume;
  double unitk[3];
  int *kxvecs, *kyvecs, *kzvecs;
  int kunique;
  int *norms, *ksq2unique;

  void coeffs();
  void allocate_kvecs();
  void deallocate_kvecs();

  // particle-mesh FFT machinery (distributed grid, modeled on PPPM)
  // spreading kernel = Kaiser-Bessel (NUFFT), deconvolved by its analytic FT

  int order;                       // KB stencil width (grid points / dimension)
  double kb_beta;                  // KB shape parameter (Beatty optimal)
  double oversample;               // FFT-mesh oversampling factor
  int nx_sf, ny_sf, nz_sf;         // global FFT mesh dimensions
  int nlower, nupper;              // stencil extent
  double shift, shiftone;
  double delxinv, delyinv, delzinv;
  double *boxlo;

  // brick + FFT (x-pencil) decompositions
  int nxlo_in, nylo_in, nzlo_in, nxhi_in, nyhi_in, nzhi_in;
  int nxlo_out, nylo_out, nzlo_out, nxhi_out, nyhi_out, nzhi_out;
  int nxlo_fft, nylo_fft, nzlo_fft, nxhi_fft, nyhi_fft, nzhi_fft;
  int ngrid, nfft, nfft_brick, nfft_both;
  int mesh_allocated;

  FFT_SCALAR ***density_brick;
  FFT_SCALAR *density_fft, *work1;
  FFT_SCALAR **rho1d;
  int **part2grid;
  int nmax;
  FFT_SCALAR *gc_buf1, *gc_buf2;
  int ngc_buf1, ngc_buf2, npergrid;

  class FFT3d *fft1;
  class Remap *remap;
  class Grid3d *gc;

  enum { REVERSE_RHO };

  void set_grid();
  void set_grid_local();
  void allocate_mesh();
  void deallocate_mesh();
  void particle_map();
  void make_rho();
  void brick2fft();
  void compute_kb1d(const FFT_SCALAR &, const FFT_SCALAR &, const FFT_SCALAR &);
  double kb_window(int, int);      // analytic KB deconvolution factor for a mode
  int factorable(int);
  void procs2grid2d(int, int, int, int &, int &);
  static double bessel_i0(double);
};

}    // namespace LAMMPS_NS

#endif
#endif
