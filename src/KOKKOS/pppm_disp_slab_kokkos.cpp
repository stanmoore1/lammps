// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Kokkos port of pppm/disp/slab.  The hot, data-parallel parts of the
   mesh reciprocal solve run on the device: the B-weighted density spread
   (make_rho), the per-mode influence/energy/force-field work (poisson),
   the 1d FFTs (via FFT3dKokkos, a local MPI_COMM_SELF nz x 1 x 1 plan),
   the z-force interpolation (fieldforce), and the full damped real-space
   slab correction (corr_raw / corr_bin, including the O(nbins^2)
   convolution and calibrate_bin).  The MPI density gather and the rare
   pressure profiles run on the host via the base class.
------------------------------------------------------------------------- */

#include "pppm_disp_slab_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "pair.h"
#include "utils.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

// MPI_COMM_SELF is provided by any real MPI; the single-proc STUBS build omits it
#ifndef MPI_COMM_SELF
#define MPI_COMM_SELF MPI_COMM_WORLD
#endif

static constexpr int OFFSET = 16384;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PPPMDispSlabKokkos<DeviceType>::PPPMDispSlabKokkos(LAMMPS *lmp) : PPPMDispSlab(lmp)
{
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK;
  datamask_modify = F_MASK;

  fft_forward = nullptr;
  fft_backward = nullptr;
  nz_created = 0;
  nmax_kk = 0;
  natoms_all_created = 0;
  nbins_created = 0;
  nwin_created = 0;

  g_ewald_kk = 0.0;
  rc2_kk = 0.0;
  area_kk = 0.0;
  w2self_kk = 0.0;
  pt2self_kk = 0.0;
  delzc_kk = 0.0;
  bindz_kk = 0.0;
  nbins_kk = 0;
  nwin_kk = 0;
  myoff_kk = 0;
  natoms_all_kk = 0;

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PPPMDispSlabKokkos<DeviceType>::~PPPMDispSlabKokkos()
{
  if (copymode) return;

  delete fft_forward;
  delete fft_backward;

  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::init()
{
  if (domain->triclinic)
    error->all(FLERR, "Cannot (yet) use pppm/disp/slab/kk with triclinic boxes");

  PPPMDispSlab::init();    // estimates params and calls setup() (this override)
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   base setup() fills nz, Gk, rho_coeff, B; allocate/upload device data
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::setup()
{
  PPPMDispSlab::setup();   // also calls calibrate_bin_kk() via override

  allocate_device();

  // upload the per-mode influence function, B-spline coefficients, and B

  auto h_Gk = Kokkos::create_mirror_view(d_Gk);
  for (int m = 0; m < nz; m++) h_Gk(m) = Gk[m];
  Kokkos::deep_copy(d_Gk, h_Gk);

  // compact switch: upload the explicit tangential/normal virial influence too
  if (damp_flag == 2) {
    auto h_GTk = Kokkos::create_mirror_view(d_GTk);
    auto h_GNk = Kokkos::create_mirror_view(d_GNk);
    for (int m = 0; m < nz; m++) { h_GTk(m) = GTk[m]; h_GNk(m) = GNk[m]; }
    Kokkos::deep_copy(d_GTk, h_GTk);
    Kokkos::deep_copy(d_GNk, h_GNk);
  }

  auto h_rho = Kokkos::create_mirror_view(d_rho_coeff);
  for (int l = 0; l < order; l++)
    for (int s = 0; s < order; s++) h_rho(l, s) = rho_coeff[l][s];
  Kokkos::deep_copy(d_rho_coeff, h_rho);

  const int ntypes = atom->ntypes;
  auto h_B = Kokkos::create_mirror_view(d_B);
  for (int t = 0; t <= ntypes; t++) h_B(t) = B[t];
  Kokkos::deep_copy(d_B, h_B);

  // upload corr scalar parameters (NPT-safe values are refreshed each compute)
  g_ewald_kk = g_ewald;
  rc2_kk = cutoff * cutoff;
}

/* ----------------------------------------------------------------------
   (re)allocate device arrays and the local 1d FFT plans when nz changes
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::allocate_device()
{
  const int ntypes = atom->ntypes;
  d_B = typename AT::t_double_1d("pppm/disp/slab/kk:B", ntypes + 1);
  d_rho_coeff = typename AT::t_double_2d("pppm/disp/slab/kk:rho_coeff", order, order);

  if (nz == nz_created) return;
  nz_created = nz;

  d_dens = typename AT::t_double_1d("pppm/disp/slab/kk:dens", nz);
  d_Gk = typename AT::t_double_1d("pppm/disp/slab/kk:Gk", nz);
  d_fz_grid = typename AT::t_double_1d("pppm/disp/slab/kk:fz_grid", nz);
  d_ugrid = typename AT::t_double_1d("pppm/disp/slab/kk:ugrid", nz);
  if (damp_flag == 2) {
    d_GTk = typename AT::t_double_1d("pppm/disp/slab/kk:GTk", nz);
    d_GNk = typename AT::t_double_1d("pppm/disp/slab/kk:GNk", nz);
    d_uTgrid = typename AT::t_double_1d("pppm/disp/slab/kk:uTgrid", nz);
    d_uNgrid = typename AT::t_double_1d("pppm/disp/slab/kk:uNgrid", nz);
  }
  h_dens = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "pppm/disp/slab/kk:h_dens", nz);

  d_work = typename FFT_AT::t_FFT_SCALAR_1d("pppm/disp/slab/kk:work", 2 * nz);
  d_work2 = typename FFT_AT::t_FFT_SCALAR_1d("pppm/disp/slab/kk:work2", 2 * nz);

  // local (per-proc) 1d FFT: nz x 1 x 1 on MPI_COMM_SELF
  delete fft_forward;
  delete fft_backward;
  int nbuf = 0;
  fft_forward = new FFT3dKokkos<DeviceType>(lmp, MPI_COMM_SELF, nz, 1, 1,
      0, nz - 1, 0, 0, 0, 0, 0, nz - 1, 0, 0, 0, 0, 0, 0, &nbuf, 0, 0, 0);
  fft_backward = new FFT3dKokkos<DeviceType>(lmp, MPI_COMM_SELF, nz, 1, 1,
      0, nz - 1, 0, 0, 0, 0, 0, nz - 1, 0, 0, 0, 0, 0, 0, &nbuf, 0, 0, 0);
}

/* ----------------------------------------------------------------------
   (re)allocate corr-bin device buffers when nbins or nwin changes
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::corr_bin_setup(int nbins)
{
  const double dz = zprd / nbins;
  int nwin = (int) (cutoff / dz) + 1;
  if (nwin > nbins / 2) nwin = nbins / 2;

  nbins_kk = nbins;
  nwin_kk = nwin;
  delzc_kk = 1.0 / dz;
  bindz_kk = dz;

  if (nbins != nbins_created) {
    nbins_created = nbins;
    d_bdens    = typename AT::t_double_1d("pppm/disp/slab/kk:bdens",    nbins);
    d_dens_all = typename AT::t_double_1d("pppm/disp/slab/kk:dens_all", nbins);
    d_phiW     = typename AT::t_double_1d("pppm/disp/slab/kk:phiW",     nbins);
    d_phiPT    = typename AT::t_double_1d("pppm/disp/slab/kk:phiPT",    nbins);
    h_bdens    = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>(
        "pppm/disp/slab/kk:h_bdens", nbins);
  }
  if (nwin + 1 > nwin_created) {
    nwin_created = nwin + 1;
    d_Kw  = typename AT::t_double_1d("pppm/disp/slab/kk:Kw",  nwin + 1);
    d_Kpt = typename AT::t_double_1d("pppm/disp/slab/kk:Kpt", nwin + 1);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  // (re)allocate per-atom energy/virial output arrays on device
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pppm/disp/slab/kk:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pppm/disp/slab/kk:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }
  // eatom/vatom managed by memoryKK below; no allocate_peratom() needed

  // grow device d_peatom if needed
  if (atom->nmax > nmax_kk) {
    nmax_kk = atom->nmax;
    d_peatom = typename AT::t_double_1d("pppm/disp/slab/kk:peatom", nmax_kk);
  }

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  // device copies of the hot-path scalars (refresh each step for NPT)
  delzinv_kk = static_cast<KK_FLOAT>(delzinv);
  zlo_kk = static_cast<KK_FLOAT>(zlo);
  shiftone_kk = static_cast<KK_FLOAT>(shiftone);
  zprd_kk = static_cast<KK_FLOAT>(zprd);
  nz_kk = nz;
  order_kk = order;
  nlower_kk = nlower;
  nupper_kk = nupper;
  dim_kk  = dim;
  lat1_kk = lat1;
  lat2_kk = lat2;
  area_kk = domain->prd[lat1] * domain->prd[lat2];   // NPT-safe refresh

  const int nlocal = atomKK->nlocal;

  // --- make_rho: spread the B-weighted density onto the z grid ---

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_make_rho_zero>(0, nz), *this);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_make_rho>(0, nlocal), *this);
  copymode = 0;

  // gather the global density across procs (host Allreduce, in place)
  Kokkos::deep_copy(h_dens, d_dens);
  MPI_Allreduce(MPI_IN_PLACE, h_dens.data(), nz, MPI_DOUBLE, MPI_SUM, world);
  Kokkos::deep_copy(d_dens, h_dens);

  // --- poisson: FFT, influence function, energy, z-force field ---

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_dens_to_work>(0, nz), *this);
  copymode = 0;

  fft_forward->compute1d(d_work, 2 * nz, FFT3dKokkos<DeviceType>::FORWARD);

  double e = 0.0;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_energy>(0, nz),
                          *this, e);
  copymode = 0;
  e_recip_mesh = e;
  if (eflag_global) energy += e;
  if (vflag_global) {
    if (damp_flag == 2) {
      // compact switch: explicit tangential (GTk) and normal (GNk) virial kernels
      s_vir vir;
      copymode = 1;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_virial_csb>(0, nz), *this, vir);
      copymode = 0;
      virial[lat1] += vir.vt;
      virial[lat2] += vir.vt;
      virial[dim] += vir.vn;
    } else {
      virial[lat1] += e;    // tangential (GT = GU)
      virial[lat2] += e;
    }
  }

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_fz_prep>(0, nz), *this);
  copymode = 0;
  fft_backward->compute1d(d_work2, 2 * nz, FFT3dKokkos<DeviceType>::BACKWARD);
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_fz_copy>(0, nz), *this);
  copymode = 0;

  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_u_prep>(0, nz), *this);
    copymode = 0;
    fft_backward->compute1d(d_work2, 2 * nz, FFT3dKokkos<DeviceType>::BACKWARD);
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_u_copy>(0, nz), *this);
    copymode = 0;

    // compact switch: per-atom tangential/normal virial fields (GTk/GNk kernels)
    if (damp_flag == 2) {
      copymode = 1;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_uT_prep>(0, nz), *this);
      copymode = 0;
      fft_backward->compute1d(d_work2, 2 * nz, FFT3dKokkos<DeviceType>::BACKWARD);
      copymode = 1;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_uT_copy>(0, nz), *this);
      copymode = 0;
      copymode = 1;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_uN_prep>(0, nz), *this);
      copymode = 0;
      fft_backward->compute1d(d_work2, 2 * nz, FFT3dKokkos<DeviceType>::BACKWARD);
      copymode = 1;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_poisson_uN_copy>(0, nz), *this);
      copymode = 0;
    }
  }

  // --- fieldforce: interpolate the z-force field to atoms (device) ---

  // zero per-atom accumulators before the reciprocal + corr contributions
  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_peatom_zero>(0, nlocal), *this);
    copymode = 0;
  }

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_fieldforce>(0, nlocal), *this);
  copymode = 0;

  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_fieldforce_peratom>(0, nlocal), *this);
    copymode = 0;
  }

  // --- damped real-space slab correction on the device ---
  // the compact switch needs no real-space correction (S*u vanishes inside rcut)

  corr_energy = 0.0;
  if (damp_flag != 2) corr_kk();

  atomKK->modified(execution_space, F_MASK);

  // --- per-atom energy/virial finalization (device kernel) ---

  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_peratom_finalize>(0, nlocal), *this);
    copymode = 0;
    if (eflag_atom) {
      k_eatom.template modify<DeviceType>();
      k_eatom.sync_host();
    }
    if (vflag_atom) {
      k_vatom.template modify<DeviceType>();
      k_vatom.sync_host();
    }
  }

  // normal virial from the exact 1/r^6 virial trace (the compact switch is
  // non-homogeneous, so its normal virial is the explicit GNk kernel above)
  if (damp_flag != 2 && vflag_global)
    virial[dim] = 6.0 * (e_recip_mesh + corr_energy) - virial[lat1] - virial[lat2];

  if (profile_flag) compute_pressure_profile();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::corr_kk()
{
  if (corr_mode == 1)
    corr_bin_kk();
  else
    corr_raw_kk();
}

/* ----------------------------------------------------------------------
   gather global (z, B) arrays to device via host MPI Allgather
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::corr_gather()
{
  int nlocal = atomKK->nlocal;
  const int nprocs = comm->nprocs;

  // MPI Allgather on host to get global z and B arrays
  auto *recvcounts = new int[nprocs];
  auto *displs = new int[nprocs];
  MPI_Allgather(&nlocal, 1, MPI_INT, recvcounts, 1, MPI_INT, world);
  int natoms_all = 0;
  for (int p = 0; p < nprocs; p++) {
    displs[p] = natoms_all;
    natoms_all += recvcounts[p];
  }
  myoff_kk = displs[comm->me];
  natoms_all_kk = natoms_all;

  // sync atoms to host to read z/type for gather
  atomKK->sync(Host, X_MASK | TYPE_MASK);
  double **x_h = atom->x;
  int *type_h = atom->type;

  auto *zloc = new double[nlocal > 0 ? nlocal : 1];
  auto *bloc = new double[nlocal > 0 ? nlocal : 1];
  for (int i = 0; i < nlocal; i++) {
    zloc[i] = x_h[i][2];
    bloc[i] = B[type_h[i]];
  }

  auto *zall = new double[natoms_all > 0 ? natoms_all : 1];
  auto *ball = new double[natoms_all > 0 ? natoms_all : 1];
  MPI_Allgatherv(zloc, nlocal, MPI_DOUBLE, zall, recvcounts, displs, MPI_DOUBLE, world);
  MPI_Allgatherv(bloc, nlocal, MPI_DOUBLE, ball, recvcounts, displs, MPI_DOUBLE, world);

  // (re)allocate device buffers and upload
  if (natoms_all > natoms_all_created) {
    natoms_all_created = natoms_all;
    d_zall  = typename AT::t_double_1d("pppm/disp/slab/kk:zall", natoms_all);
    d_ball  = typename AT::t_double_1d("pppm/disp/slab/kk:ball", natoms_all);
    d_fzref = typename AT::t_double_1d("pppm/disp/slab/kk:fzref", natoms_all);
  }

  auto h_zall = Kokkos::create_mirror_view(d_zall);
  auto h_ball = Kokkos::create_mirror_view(d_ball);
  for (int i = 0; i < natoms_all; i++) {
    h_zall(i) = zall[i];
    h_ball(i) = ball[i];
  }
  Kokkos::deep_copy(d_zall, h_zall);
  Kokkos::deep_copy(d_ball, h_ball);

  delete[] recvcounts;
  delete[] displs;
  delete[] zloc;
  delete[] bloc;
  delete[] zall;
  delete[] ball;
}

/* ----------------------------------------------------------------------
   exact pairwise corr: gather z/B, then O(N*Nall) device kernel
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::corr_raw_kk()
{
  corr_gather();

  // self-interaction terms (x2 = 0): compute once on host using corr_kernels
  double w0, f0, pt0;
  corr_kernels(0.0, w0, f0, pt0);
  w2self_kk  = 0.5 * w0;
  pt2self_kk = 0.5 * pt0;

  const int nlocal = atomKK->nlocal;

  s_corr ev{};
  copymode = 1;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_raw>(0, nlocal), *this, ev);
  copymode = 0;

  double e_local = ev.e;
  double vt_local = ev.vt;

  double e_all;
  MPI_Allreduce(&e_local, &e_all, 1, MPI_DOUBLE, MPI_SUM, world);
  corr_energy += e_all;
  if (eflag_global) energy += e_all;
  if (vflag_global) {
    double vt_all;
    MPI_Allreduce(&vt_local, &vt_all, 1, MPI_DOUBLE, MPI_SUM, world);
    virial[lat1] += vt_all;
    virial[lat2] += vt_all;
  }
}

/* ----------------------------------------------------------------------
   force-only exact pairwise corr (for calibrate_bin_kk) -> d_fzref
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::corr_raw_force_kk()
{
  corr_gather();
  double w0, f0, pt0;
  corr_kernels(0.0, w0, f0, pt0);
  w2self_kk  = 0.5 * w0;
  pt2self_kk = 0.5 * pt0;

  const int nlocal = atomKK->nlocal;
  if (nlocal > (int)d_fzref.extent(0))
    d_fzref = typename AT::t_double_1d("pppm/disp/slab/kk:fzref", nlocal);

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_raw_force>(0, nlocal), *this);
  copymode = 0;
}

/* ----------------------------------------------------------------------
   z-binned corr: spread -> Allreduce -> kernel table -> O(nbins^2) conv
   -> energy/virial -> B-spline interp forces/per-atom energy
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::corr_bin_kk()
{
  // determine nbins (same logic as base corr_bin)
  int nbins;
  if (bin_dz_user > 0.0)
    nbins = (int) (zprd / bin_dz_user + 0.5);
  else if (bin_nbins > 0)
    nbins = bin_nbins;
  else
    nbins = (int) (zprd / MIN(0.025 / g_ewald, 0.025 * cutoff) + 0.5);
  if (nbins < 4) nbins = 4;

  corr_bin_setup(nbins);

  const int nlocal = atomKK->nlocal;

  // 1. zero bin density
  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_zero>(0, nbins_kk), *this);

  // 2. B-spline spread to corr bins
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_spread>(0, nlocal), *this);
  copymode = 0;

  // 3. MPI Allreduce on host
  Kokkos::deep_copy(h_bdens, d_bdens);
  MPI_Allreduce(MPI_IN_PLACE, h_bdens.data(), nbins_kk, MPI_DOUBLE, MPI_SUM, world);
  Kokkos::deep_copy(d_dens_all, h_bdens);

  // 4. kernel table Kw, Kpt
  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_ktable>(0, nwin_kk + 1), *this);

  // 5. O(nbins^2) convolution: phiW, phiPT
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_conv>(0, nbins_kk), *this);
  copymode = 0;

  // 6. energy/virial (global)
  s_corr ev{};
  copymode = 1;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_energy>(0, nbins_kk), *this, ev);
  copymode = 0;
  corr_energy += 0.5 * ev.e;
  if (eflag_global) energy += 0.5 * ev.e;
  if (vflag_global) {
    virial[lat1] += 0.5 * ev.vt;
    virial[lat2] += 0.5 * ev.vt;
  }

  // 7. force/per-atom interpolation
  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_interp>(0, nlocal), *this);
  copymode = 0;
}

/* ----------------------------------------------------------------------
   force-only binned corr for calibration -> d_fzbin
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::corr_bin_force_kk(int nbins)
{
  corr_bin_setup(nbins);

  const int nlocal = atomKK->nlocal;
  if (nlocal > 0 && (d_fzbin.extent(0) < (size_t)nlocal))
    d_fzbin = typename AT::t_double_1d("pppm/disp/slab/kk:fzbin", nlocal);

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_zero>(0, nbins_kk), *this);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_spread>(0, nlocal), *this);
  copymode = 0;

  Kokkos::deep_copy(h_bdens, d_bdens);
  MPI_Allreduce(MPI_IN_PLACE, h_bdens.data(), nbins_kk, MPI_DOUBLE, MPI_SUM, world);
  Kokkos::deep_copy(d_dens_all, h_bdens);

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_ktable>(0, nwin_kk + 1), *this);
  // conv (phiW only: corr_bin_conv_w)
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_conv_w>(0, nbins_kk), *this);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_bin_interp_force>(0, nlocal), *this);
  copymode = 0;
}

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::calibrate_bin()
{
  calibrate_bin_kk();
}

/* ----------------------------------------------------------------------
   size the corr bin count to target accuracy (fully on device)
   mirrors PPPMDispSlab::calibrate_bin but uses device loops
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::calibrate_bin_kk()
{
  const int nlocal = atomKK->nlocal;
  double natoms = (double) atom->natoms;
  if (natoms < 1.0) natoms = 1.0;

  // update all device scalars needed by corr/bin kernels during calibration
  g_ewald_kk  = g_ewald;
  rc2_kk      = cutoff * cutoff;
  dim_kk      = dim;
  lat1_kk     = lat1;
  lat2_kk     = lat2;
  area_kk     = domain->prd[lat1] * domain->prd[lat2];
  zprd_kk     = static_cast<KK_FLOAT>(zprd);
  zlo_kk      = static_cast<KK_FLOAT>(zlo);
  shiftone_kk = static_cast<KK_FLOAT>(shiftone);
  nlower_kk   = nlower;
  nupper_kk   = nupper;
  order_kk    = order;

  // sync atoms to device (setup may run before first compute)
  atomKK->sync(execution_space, X_MASK | TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  // allocate coefficient tables if not yet done; calibrate_bin is called
  // during PPPMDispSlab::setup() before our allocate_device() runs
  const int ntypes = atom->ntypes;
  if ((int) d_rho_coeff.extent(0) != order)
    d_rho_coeff = typename AT::t_double_2d("pppm/disp/slab/kk:rho_coeff", order, order);
  if ((int) d_B.extent(0) != ntypes + 1)
    d_B = typename AT::t_double_1d("pppm/disp/slab/kk:B", ntypes + 1);

  // upload rho_coeff and B (needed by spread/interp kernels during calibration)
  {
    auto h_rho = Kokkos::create_mirror_view(d_rho_coeff);
    for (int l = 0; l < order; l++)
      for (int s = 0; s < order; s++) h_rho(l, s) = rho_coeff[l][s];
    Kokkos::deep_copy(d_rho_coeff, h_rho);
    auto h_B2 = Kokkos::create_mirror_view(d_B);
    for (int t = 0; t <= ntypes; t++) h_B2(t) = B[t];
    Kokkos::deep_copy(d_B, h_B2);
  }

  // nbins_created etc. need to be reset so corr_bin_setup reallocates
  nbins_created = 0;
  nwin_created  = 0;

  // exact pairwise reference force -> d_fzref
  corr_raw_force_kk();

  const int nb_cap = (int) (zprd / (0.2 * cutoff / 600.0) + 0.5);
  int nb = (int) (zprd / 0.1 + 0.5);
  if (nb < 8) nb = 8;
  int chosen = nb;
  double err = 0.0;

  for (int it = 0; it < 20; it++) {
    corr_bin_force_kk(nb);

    // compute RMS(fzbin - fzref) over nlocal atoms on device
    double s_local = 0.0;
    copymode = 1;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_corr_calib_err>(0, nlocal),
        *this, s_local);
    copymode = 0;

    double sall;
    MPI_Allreduce(&s_local, &sall, 1, MPI_DOUBLE, MPI_SUM, world);
    err = sqrt(sall / natoms);
    chosen = nb;
    if (err < accuracy || nb >= nb_cap) break;
    nb *= 2;
  }
  bin_nbins = chosen;
  if (comm->me == 0) {
    utils::logmesg(lmp, "  corr bin: {} bins (dz = {:.4g}), force error {:.3g} vs target {:.3g}\n",
                   bin_nbins, zprd / bin_nbins, err, accuracy);
    if (err > accuracy)
      error->warning(FLERR,
                     "pppm/disp/slab corr bin did not reach the target force accuracy {:.3g} "
                     "(reached {:.3g}); use kspace_modify corr raw for tighter accuracy",
                     accuracy, err);
  }
}

/* ---------------------------------------------------------------------- */
/* device kernels — reciprocal mesh                                        */
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_make_rho_zero, const int &g) const
{
  d_dens(g) = 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_make_rho, const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzinv_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8];
  compute_rho1d_kk(dz, w);
  const double bi = d_B(type(i));
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nz_kk) + nz_kk) % nz_kk;
    Kokkos::atomic_add(&d_dens(g), bi * w[s]);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_dens_to_work, const int &m) const
{
  d_work(2 * m) = static_cast<FFT_SCALAR>(d_dens(m));
  d_work(2 * m + 1) = static_cast<FFT_SCALAR>(0.0);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_energy, const int &m,
                                                double &esum) const
{
  const double re = static_cast<double>(d_work(2 * m));
  const double im = static_cast<double>(d_work(2 * m + 1));
  esum += d_Gk(m) * (re * re + im * im);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_virial_csb, const int &m,
                                                s_vir &vir) const
{
  const double re = static_cast<double>(d_work(2 * m));
  const double im = static_cast<double>(d_work(2 * m + 1));
  const double uk = re * re + im * im;
  vir.vt += d_GTk(m) * uk;
  vir.vn += d_GNk(m) * uk;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_uT_prep, const int &m) const
{
  const double g2 = 2.0 * d_GTk(m);
  d_work2(2 * m) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m)));
  d_work2(2 * m + 1) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m + 1)));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_uT_copy, const int &m) const
{
  d_uTgrid(m) = static_cast<double>(d_work2(2 * m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_uN_prep, const int &m) const
{
  const double g2 = 2.0 * d_GNk(m);
  d_work2(2 * m) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m)));
  d_work2(2 * m + 1) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m + 1)));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_uN_copy, const int &m) const
{
  d_uNgrid(m) = static_cast<double>(d_work2(2 * m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_fz_prep, const int &m) const
{
  const int mm = (m <= nz_kk / 2) ? m : m - nz_kk;
  const double k = mm * 2.0 * MY_PI / zprd_kk;
  const double g2k = 2.0 * d_Gk(m) * k;
  const double a = static_cast<double>(d_work(2 * m));
  const double b = static_cast<double>(d_work(2 * m + 1));
  d_work2(2 * m) = static_cast<FFT_SCALAR>(g2k * b);
  d_work2(2 * m + 1) = static_cast<FFT_SCALAR>(-g2k * a);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_fz_copy, const int &m) const
{
  d_fz_grid(m) = static_cast<double>(d_work2(2 * m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_u_prep, const int &m) const
{
  const double g2 = 2.0 * d_Gk(m);
  d_work2(2 * m) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m)));
  d_work2(2 * m + 1) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m + 1)));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_poisson_u_copy, const int &m) const
{
  d_ugrid(m) = static_cast<double>(d_work2(2 * m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_fieldforce, const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzinv_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8];
  compute_rho1d_kk(dz, w);
  const double bi = d_B(type(i));

  double fz = 0.0;
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nz_kk) + nz_kk) % nz_kk;
    fz += w[s] * d_fz_grid(g);
  }
  f(i, dim_kk) += static_cast<KK_ACC_FLOAT>(bi * fz);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_peatom_zero, const int &i) const
{
  d_peatom(i) = 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_fieldforce_peratom,
                                                const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzinv_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8];
  compute_rho1d_kk(dz, w);
  const double bi = d_B(type(i));

  double uu = 0.0, uT = 0.0, uN = 0.0;
  const bool csb = (damp_flag == 2);
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nz_kk) + nz_kk) % nz_kk;
    uu += w[s] * d_ugrid(g);
    if (csb) {
      uT += w[s] * d_uTgrid(g);
      uN += w[s] * d_uNgrid(g);
    }
  }
  // reciprocal per-atom energy
  double pe = 0.5 * bi * uu;
  d_peatom(i) += pe;
  if (vflag_atom) {
    if (csb) {
      // explicit tangential (GTk) and normal (GNk) per-atom virial fields
      d_vatom(i, lat1_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bi * uT);
      d_vatom(i, lat2_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bi * uT);
      d_vatom(i, dim_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bi * uN);
    } else {
      d_vatom(i, lat1_kk) += static_cast<KK_ACC_FLOAT>(pe);    // tangential (GT = GU)
      d_vatom(i, lat2_kk) += static_cast<KK_ACC_FLOAT>(pe);
    }
  }
}

/* ---------------------------------------------------------------------- */
/* device kernels — corr raw (exact pairwise)                             */
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_raw, const int &i,
                                                s_corr &ev) const
{
  const double zi = static_cast<double>(x(i, dim_kk));
  const double bi = d_B(type(i));
  const int iglob = myoff_kk + i;

  // self term
  ev.e  += bi * bi * w2self_kk;
  ev.vt += bi * bi * pt2self_kk;
  if (evflag_atom) d_peatom(i) += bi * bi * w2self_kk;
  if (vflag_atom) {
    d_vatom(i, lat1_kk) += static_cast<KK_ACC_FLOAT>(bi * bi * pt2self_kk);
    d_vatom(i, lat2_kk) += static_cast<KK_ACC_FLOAT>(bi * bi * pt2self_kk);
  }

  double fz = 0.0;
  for (int jg = 0; jg < natoms_all_kk; jg++) {
    if (jg == iglob) continue;
    double delz = zi - d_zall(jg);
    delz -= static_cast<double>(zprd_kk) * trunc(2.0 * delz / static_cast<double>(zprd_kk));
    const double x2 = delz * delz;
    if (x2 >= rc2_kk) continue;

    double w2, f2, pt2;
    corr_kernels_kk(x2, w2, f2, pt2);
    const double bij = bi * d_ball(jg);

    ev.e  += 0.5 * bij * w2;
    ev.vt += 0.5 * bij * pt2;
    fz    += delz * bij * f2;

    if (evflag_atom) d_peatom(i) += 0.5 * bij * w2;
    if (vflag_atom) {
      d_vatom(i, lat1_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bij * pt2);
      d_vatom(i, lat2_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bij * pt2);
    }
  }
  f(i, dim_kk) += static_cast<KK_ACC_FLOAT>(fz);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_raw_force, const int &i) const
{
  const double zi = static_cast<double>(x(i, dim_kk));
  const double bi = d_B(type(i));
  const int iglob = myoff_kk + i;
  double fz = 0.0;
  for (int jg = 0; jg < natoms_all_kk; jg++) {
    if (jg == iglob) continue;
    double delz = zi - d_zall(jg);
    delz -= static_cast<double>(zprd_kk) * trunc(2.0 * delz / static_cast<double>(zprd_kk));
    const double x2 = delz * delz;
    if (x2 >= rc2_kk) continue;
    double w2, f2, pt2;
    corr_kernels_kk(x2, w2, f2, pt2);
    fz += delz * bi * d_ball(jg) * f2;
  }
  d_fzref(i) = fz;
}

/* ---------------------------------------------------------------------- */
/* device kernels — corr bin (z-binned)                                   */
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_bin_zero, const int &b) const
{
  d_bdens(b) = 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_bin_spread, const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzc_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8];
  compute_rho1d_kk(dz, w);
  const double bi = d_B(type(i));
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nbins_kk) + nbins_kk) % nbins_kk;
    Kokkos::atomic_add(&d_bdens(g), bi * w[s]);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_bin_ktable, const int &d) const
{
  const double xi = d * bindz_kk;
  const double x2 = xi * xi;
  if (x2 >= rc2_kk) {
    d_Kw(d) = 0.0;
    d_Kpt(d) = 0.0;
  } else {
    double w2, f2, pt2;
    corr_kernels_kk(x2, w2, f2, pt2);
    d_Kw(d) = w2;
    if ((int)d_Kpt.extent(0) > d) d_Kpt(d) = pt2;
  }
}

// full convolution (phiW and phiPT), used in the production per-step path
template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_bin_conv, const int &b) const
{
  double sw = d_Kw(0) * d_dens_all(b);
  double spt = d_Kpt(0) * d_dens_all(b);
  for (int d = 1; d <= nwin_kk; d++) {
    int bp = b + d;
    if (bp >= nbins_kk) bp -= nbins_kk;
    int bm = b - d;
    if (bm < 0) bm += nbins_kk;
    const double s = d_dens_all(bp) + d_dens_all(bm);
    sw  += d_Kw(d) * s;
    spt += d_Kpt(d) * s;
  }
  d_phiW(b)  = sw;
  d_phiPT(b) = spt;
}

// phiW-only convolution for calibrate_bin_kk (no phiPT needed)
template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_bin_conv_w, const int &b) const
{
  double sw = d_Kw(0) * d_dens_all(b);
  for (int d = 1; d <= nwin_kk; d++) {
    int bp = b + d;
    if (bp >= nbins_kk) bp -= nbins_kk;
    int bm = b - d;
    if (bm < 0) bm += nbins_kk;
    sw += d_Kw(d) * (d_dens_all(bp) + d_dens_all(bm));
  }
  d_phiW(b) = sw;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_bin_energy, const int &b,
                                                s_corr &ev) const
{
  ev.e  += d_dens_all(b) * d_phiW(b);
  ev.vt += d_dens_all(b) * d_phiPT(b);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_bin_interp, const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzc_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8], dw[8];
  compute_rho1d_kk(dz, w);
  compute_drho1d_kk(dz, dw);
  const double bi = d_B(type(i));

  double fz = 0.0, pe = 0.0, pt = 0.0;
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nbins_kk) + nbins_kk) % nbins_kk;
    fz += dw[s] * d_phiW(g);
    if (evflag_atom) pe += w[s] * d_phiW(g);
    if (vflag_atom)  pt += w[s] * d_phiPT(g);
  }
  f(i, dim_kk) += static_cast<KK_ACC_FLOAT>(bi * delzc_kk * fz);
  if (evflag_atom) d_peatom(i) += 0.5 * bi * pe;
  if (vflag_atom) {
    d_vatom(i, lat1_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bi * pt);
    d_vatom(i, lat2_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bi * pt);
  }
}

// force-only variant for calibration
template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_bin_interp_force,
                                                const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzc_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double dw[8];
  compute_drho1d_kk(dz, dw);
  const double bi = d_B(type(i));

  double fz = 0.0;
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nbins_kk) + nbins_kk) % nbins_kk;
    fz += dw[s] * d_phiW(g);
  }
  d_fzbin(i) = bi * delzc_kk * fz;
}

/* ---------------------------------------------------------------------- */
/* device kernels — calibration error + per-atom finalization             */
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_corr_calib_err, const int &i,
                                                double &s) const
{
  const double d = d_fzbin(i) - d_fzref(i);
  s += d * d;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispSlabKokkos<DeviceType>::operator()(TagPPPMDispSlab_peratom_finalize,
                                                const int &i) const
{
  // d_peatom holds the full kspace per-atom energy (reciprocal + corr)
  if (eflag_atom) d_eatom(i) += static_cast<KK_ACC_FLOAT>(d_peatom(i));
  // normal per-atom virial from the virial trace: 6*e_i - v_lat1 - v_lat2.
  // The compact switch already set vatom[dim] from its explicit GNk field.
  if (vflag_atom && damp_flag != 2)
    d_vatom(i, dim_kk) += static_cast<KK_ACC_FLOAT>(
        6.0 * d_peatom(i) - static_cast<double>(d_vatom(i, lat1_kk)) -
        static_cast<double>(d_vatom(i, lat2_kk)));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double PPPMDispSlabKokkos<DeviceType>::memory_usage()
{
  double bytes = PPPMDispSlab::memory_usage();
  bytes += (double) 4 * nz * sizeof(double);       // d_dens, d_Gk, d_fz_grid, d_ugrid
  bytes += (double) 4 * nz * sizeof(FFT_SCALAR);   // d_work, d_work2
  if (nbins_created > 0)
    bytes += (double) (2 * nbins_created + (nwin_created)) * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PPPMDispSlabKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PPPMDispSlabKokkos<LMPHostType>;
#endif
}
