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
   Kokkos port of pppm/disp/planar.  The hot, data-parallel parts of the
   mesh reciprocal solve run on the device: the B-weighted density spread
   (make_rho), the per-mode influence/energy/force-field work (poisson),
   the 1d FFTs (via FFT3dKokkos, a local MPI_COMM_SELF nz x 1 x 1 plan),
   the z-force interpolation (fieldforce), and the compact-switch shell
   correction (corr_shell, an O(N*Nall) device pairwise kernel).  The MPI
   (z,B) gather and the rare pressure profiles run on the host via the base.
------------------------------------------------------------------------- */

#include "pppm_disp_planar_kokkos.h"

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
PPPMDispPlanarKokkos<DeviceType>::PPPMDispPlanarKokkos(LAMMPS *lmp) : PPPMDispPlanar(lmp)
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

  myoff_kk = 0;
  natoms_all_kk = 0;
  nwgrid_kk = 0;
  wdz_kk = 0.0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PPPMDispPlanarKokkos<DeviceType>::~PPPMDispPlanarKokkos()
{
  if (copymode) return;

  delete fft_forward;
  delete fft_backward;

  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::init()
{
  if (domain->triclinic)
    error->all(FLERR, "Cannot (yet) use pppm/disp/planar/kk with triclinic boxes");

  PPPMDispPlanar::init();    // estimates params and calls setup() (this override)
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   base setup() fills nz, Gk, rho_coeff, B; allocate/upload device data
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::setup()
{
  PPPMDispPlanar::setup();

  allocate_device();

  // upload the per-mode influence function, B-spline coefficients, and B

  auto h_Gk = Kokkos::create_mirror_view(d_Gk);
  for (int m = 0; m < nz; m++) h_Gk(m) = Gk[m];
  Kokkos::deep_copy(d_Gk, h_Gk);

  // upload the explicit tangential/normal virial influence too, plus the
  // shell-correction kernel tables (built host-side in PPPMDispPlanar::setup)
  {
    auto h_GTk = Kokkos::create_mirror_view(d_GTk);
    auto h_GNk = Kokkos::create_mirror_view(d_GNk);
    for (int m = 0; m < nz; m++) { h_GTk(m) = GTk[m]; h_GNk(m) = GNk[m]; }
    Kokkos::deep_copy(d_GTk, h_GTk);
    Kokkos::deep_copy(d_GNk, h_GNk);

    nwgrid_kk = nwgrid;
    wdz_kk = wdz;
    d_wEgrid = typename AT::t_double_1d("pppm/disp/planar/kk:wEgrid", nwgrid + 1);
    d_wFgrid = typename AT::t_double_1d("pppm/disp/planar/kk:wFgrid", nwgrid + 1);
    d_wTgrid = typename AT::t_double_1d("pppm/disp/planar/kk:wTgrid", nwgrid + 1);
    d_wNgrid = typename AT::t_double_1d("pppm/disp/planar/kk:wNgrid", nwgrid + 1);
    auto h_wE = Kokkos::create_mirror_view(d_wEgrid);
    auto h_wF = Kokkos::create_mirror_view(d_wFgrid);
    auto h_wT = Kokkos::create_mirror_view(d_wTgrid);
    auto h_wN = Kokkos::create_mirror_view(d_wNgrid);
    for (int g = 0; g <= nwgrid; g++) {
      h_wE(g) = wEgrid[g]; h_wF(g) = wFgrid[g]; h_wT(g) = wTgrid[g]; h_wN(g) = wNgrid[g];
    }
    Kokkos::deep_copy(d_wEgrid, h_wE);
    Kokkos::deep_copy(d_wFgrid, h_wF);
    Kokkos::deep_copy(d_wTgrid, h_wT);
    Kokkos::deep_copy(d_wNgrid, h_wN);
  }

  auto h_rho = Kokkos::create_mirror_view(d_rho_coeff);
  for (int l = 0; l < order; l++)
    for (int s = 0; s < order; s++) h_rho(l, s) = rho_coeff[l][s];
  Kokkos::deep_copy(d_rho_coeff, h_rho);

  const int ntypes = atom->ntypes;
  auto h_B = Kokkos::create_mirror_view(d_B);
  for (int t = 0; t <= ntypes; t++) h_B(t) = B[t];
  Kokkos::deep_copy(d_B, h_B);
}

/* ----------------------------------------------------------------------
   (re)allocate device arrays and the local 1d FFT plans when nz changes
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::allocate_device()
{
  const int ntypes = atom->ntypes;
  d_B = typename AT::t_double_1d("pppm/disp/planar/kk:B", ntypes + 1);
  d_rho_coeff = typename AT::t_double_2d("pppm/disp/planar/kk:rho_coeff", order, order);

  if (nz == nz_created) return;
  nz_created = nz;

  d_dens = typename AT::t_double_1d("pppm/disp/planar/kk:dens", nz);
  d_Gk = typename AT::t_double_1d("pppm/disp/planar/kk:Gk", nz);
  d_fz_grid = typename AT::t_double_1d("pppm/disp/planar/kk:fz_grid", nz);
  d_ugrid = typename AT::t_double_1d("pppm/disp/planar/kk:ugrid", nz);
  d_GTk = typename AT::t_double_1d("pppm/disp/planar/kk:GTk", nz);
  d_GNk = typename AT::t_double_1d("pppm/disp/planar/kk:GNk", nz);
  d_uTgrid = typename AT::t_double_1d("pppm/disp/planar/kk:uTgrid", nz);
  d_uNgrid = typename AT::t_double_1d("pppm/disp/planar/kk:uNgrid", nz);
  h_dens = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "pppm/disp/planar/kk:h_dens", nz);

  d_work = typename FFT_AT::t_FFT_SCALAR_1d("pppm/disp/planar/kk:work", 2 * nz);
  d_work2 = typename FFT_AT::t_FFT_SCALAR_1d("pppm/disp/planar/kk:work2", 2 * nz);

  // local (per-proc) 1d FFT: nz x 1 x 1 on MPI_COMM_SELF
  delete fft_forward;
  delete fft_backward;
  int nbuf = 0;
  fft_forward = new FFT3dKokkos<DeviceType>(lmp, MPI_COMM_SELF, nz, 1, 1,
      0, nz - 1, 0, 0, 0, 0, 0, nz - 1, 0, 0, 0, 0, 0, 0, &nbuf, 0, 0, 0);
  fft_backward = new FFT3dKokkos<DeviceType>(lmp, MPI_COMM_SELF, nz, 1, 1,
      0, nz - 1, 0, 0, 0, 0, 0, nz - 1, 0, 0, 0, 0, 0, 0, &nbuf, 0, 0, 0);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  // (re)allocate per-atom energy/virial output arrays on device
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pppm/disp/planar/kk:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pppm/disp/planar/kk:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }
  // eatom/vatom managed by memoryKK below; no allocate_peratom() needed

  // grow device d_peatom if needed
  if (atom->nmax > nmax_kk) {
    nmax_kk = atom->nmax;
    d_peatom = typename AT::t_double_1d("pppm/disp/planar/kk:peatom", nmax_kk);
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

  const int nlocal = atomKK->nlocal;

  // --- make_rho: spread the B-weighted density onto the z grid ---

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_make_rho_zero>(0, nz), *this);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_make_rho>(0, nlocal), *this);
  copymode = 0;

  // gather the global density across procs (host Allreduce, in place)
  Kokkos::deep_copy(h_dens, d_dens);
  MPI_Allreduce(MPI_IN_PLACE, h_dens.data(), nz, MPI_DOUBLE, MPI_SUM, world);
  Kokkos::deep_copy(d_dens, h_dens);

  // --- poisson: FFT, influence function, energy, z-force field ---

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_dens_to_work>(0, nz), *this);
  copymode = 0;

  fft_forward->compute1d(d_work, 2 * nz, FFT3dKokkos<DeviceType>::FORWARD);

  double e = 0.0;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_energy>(0, nz),
                          *this, e);
  copymode = 0;
  e_recip_mesh = e;
  if (eflag_global) energy += e;
  if (vflag_global) {
    // compact switch: explicit tangential (GTk) and normal (GNk) virial kernels
    s_vir vir;
    copymode = 1;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_virial_csb>(0, nz), *this, vir);
    copymode = 0;
    virial[lat1] += vir.vt;
    virial[lat2] += vir.vt;
    virial[dim] += vir.vn;
  }

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_fz_prep>(0, nz), *this);
  copymode = 0;
  fft_backward->compute1d(d_work2, 2 * nz, FFT3dKokkos<DeviceType>::BACKWARD);
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_fz_copy>(0, nz), *this);
  copymode = 0;

  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_u_prep>(0, nz), *this);
    copymode = 0;
    fft_backward->compute1d(d_work2, 2 * nz, FFT3dKokkos<DeviceType>::BACKWARD);
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_u_copy>(0, nz), *this);
    copymode = 0;

    // compact switch: per-atom tangential/normal virial fields (GTk/GNk kernels)
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_uT_prep>(0, nz), *this);
    copymode = 0;
    fft_backward->compute1d(d_work2, 2 * nz, FFT3dKokkos<DeviceType>::BACKWARD);
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_uT_copy>(0, nz), *this);
    copymode = 0;
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_uN_prep>(0, nz), *this);
    copymode = 0;
    fft_backward->compute1d(d_work2, 2 * nz, FFT3dKokkos<DeviceType>::BACKWARD);
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_uN_copy>(0, nz), *this);
    copymode = 0;
  }

  // --- fieldforce: interpolate the z-force field to atoms (device) ---

  // zero per-atom accumulators before the reciprocal + corr contributions
  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_peatom_zero>(0, nlocal), *this);
    copymode = 0;
  }

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_fieldforce>(0, nlocal), *this);
  copymode = 0;

  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_fieldforce_peratom>(0, nlocal), *this);
    copymode = 0;
  }

  // --- compact-switch shell correction on the device ---
  // corr_shell_kk() subtracts the reciprocal sum's plane mean-field S*u over the
  // shell so the matched pair's exact 3-D shell interaction remains

  corr_energy = 0.0;
  corr_shell_kk();

  atomKK->modified(execution_space, F_MASK);

  // --- per-atom energy/virial finalization (device kernel) ---

  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_peratom_finalize>(0, nlocal), *this);
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

  // the normal (zz) virial is the explicit GNk kernel accumulated above (the
  // compact switch is non-homogeneous, so the trace identity does not apply)

  if (profile_flag) compute_pressure_profile();
}

/* ----------------------------------------------------------------------
   compact-switch (CSB) shell correction: gather z/B, then an O(N*Nall) device
   kernel subtracts the reciprocal sum's plane mean-field S*u over the shell
   (energy, z-force, virial) so the matched pair's exact 3-D shell interaction
   remains.  Full ordered double sum incl. self (no 1/2; force carries a factor
   2).  Device port of PPPMDispPlanar::corr_shell_raw.  The exact pairwise form is
   used for both corr raw and corr bin requests on the device.
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::corr_shell_kk()
{
  corr_gather();    // fills d_zall/d_ball, natoms_all_kk (self included below)

  const int nlocal = atomKK->nlocal;

  s_csb ev{};
  copymode = 1;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_corr_shell_raw>(0, nlocal), *this, ev);
  copymode = 0;

  if (eflag_global || vflag_global) {
    double e_all;
    MPI_Allreduce(&ev.e, &e_all, 1, MPI_DOUBLE, MPI_SUM, world);
    corr_energy -= e_all;
    if (eflag_global) energy -= e_all;
  }
  if (vflag_global) {
    double vt_all, vn_all;
    MPI_Allreduce(&ev.vt, &vt_all, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&ev.vn, &vn_all, 1, MPI_DOUBLE, MPI_SUM, world);
    virial[lat1] -= vt_all;
    virial[lat2] -= vt_all;
    virial[dim] -= vn_all;
  }
}

/* ----------------------------------------------------------------------
   gather global (z, B) arrays to device via host MPI Allgather
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::corr_gather()
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
    d_zall  = typename AT::t_double_1d("pppm/disp/planar/kk:zall", natoms_all);
    d_ball  = typename AT::t_double_1d("pppm/disp/planar/kk:ball", natoms_all);
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


/* ---------------------------------------------------------------------- */
/* device kernels — reciprocal mesh                                        */
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_make_rho_zero, const int &g) const
{
  d_dens(g) = 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_make_rho, const int &i) const
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
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_dens_to_work, const int &m) const
{
  d_work(2 * m) = static_cast<FFT_SCALAR>(d_dens(m));
  d_work(2 * m + 1) = static_cast<FFT_SCALAR>(0.0);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_energy, const int &m,
                                                double &esum) const
{
  const double re = static_cast<double>(d_work(2 * m));
  const double im = static_cast<double>(d_work(2 * m + 1));
  esum += d_Gk(m) * (re * re + im * im);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_virial_csb, const int &m,
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
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_uT_prep, const int &m) const
{
  const double g2 = 2.0 * d_GTk(m);
  d_work2(2 * m) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m)));
  d_work2(2 * m + 1) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m + 1)));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_uT_copy, const int &m) const
{
  d_uTgrid(m) = static_cast<double>(d_work2(2 * m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_uN_prep, const int &m) const
{
  const double g2 = 2.0 * d_GNk(m);
  d_work2(2 * m) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m)));
  d_work2(2 * m + 1) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m + 1)));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_uN_copy, const int &m) const
{
  d_uNgrid(m) = static_cast<double>(d_work2(2 * m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_fz_prep, const int &m) const
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
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_fz_copy, const int &m) const
{
  d_fz_grid(m) = static_cast<double>(d_work2(2 * m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_u_prep, const int &m) const
{
  const double g2 = 2.0 * d_Gk(m);
  d_work2(2 * m) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m)));
  d_work2(2 * m + 1) = static_cast<FFT_SCALAR>(g2 * static_cast<double>(d_work(2 * m + 1)));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_u_copy, const int &m) const
{
  d_ugrid(m) = static_cast<double>(d_work2(2 * m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_fieldforce, const int &i) const
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
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_peatom_zero, const int &i) const
{
  d_peatom(i) = 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_fieldforce_peratom,
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
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nz_kk) + nz_kk) % nz_kk;
    uu += w[s] * d_ugrid(g);
    uT += w[s] * d_uTgrid(g);
    uN += w[s] * d_uNgrid(g);
  }
  // reciprocal per-atom energy
  double pe = 0.5 * bi * uu;
  d_peatom(i) += pe;
  if (vflag_atom) {
    // explicit tangential (GTk) and normal (GNk) per-atom virial fields
    d_vatom(i, lat1_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bi * uT);
    d_vatom(i, lat2_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bi * uT);
    d_vatom(i, dim_kk) += static_cast<KK_ACC_FLOAT>(0.5 * bi * uN);
  }
}

// compact-switch (CSB) shell correction: subtract the plane mean-field S*u over
// the shell.  Full ordered sum incl. self; energy/virial carry no 1/2, the
// z-force a factor 2.  Device port of PPPMDispPlanar::corr_shell_raw.
template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_corr_shell_raw, const int &i,
                                                s_csb &ev) const
{
  const double zi = static_cast<double>(x(i, dim_kk));
  const double bi = d_B(type(i));
  const double zprd = static_cast<double>(zprd_kk);
  const double bcut = nwgrid_kk * wdz_kk;
  double e_i = 0.0, fz_i = 0.0, vt_i = 0.0, vn_i = 0.0;
  for (int jg = 0; jg < natoms_all_kk; jg++) {
    double delz = zi - d_zall(jg);
    delz -= zprd * floor(delz / zprd + 0.5);    // nearest image (self -> delz=0)
    const double adz = fabs(delz);
    if (adz >= bcut) continue;
    double wE, wF, wT, wN;
    shell_vkernel_kk(adz, wE, wF, wT, wN);
    const double bij = bi * d_ball(jg);
    e_i  += bij * wE;
    fz_i += 2.0 * delz * bij * wF;    // remove the plane z-force (factor 2)
    vt_i += bij * wT;
    vn_i += bij * wN;
  }
  ev.e  += e_i;
  ev.vt += vt_i;
  ev.vn += vn_i;
  f(i, dim_kk) += static_cast<KK_ACC_FLOAT>(fz_i);
  if (evflag_atom) d_peatom(i) -= e_i;
  if (vflag_atom) {
    d_vatom(i, lat1_kk) -= static_cast<KK_ACC_FLOAT>(vt_i);
    d_vatom(i, lat2_kk) -= static_cast<KK_ACC_FLOAT>(vt_i);
    d_vatom(i, dim_kk)  -= static_cast<KK_ACC_FLOAT>(vn_i);
  }
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_peratom_finalize,
                                                const int &i) const
{
  // d_peatom holds the full kspace per-atom energy (reciprocal + shell corr).
  // The per-atom normal virial is already set from the explicit GNk field in
  // fieldforce_peratom (the compact switch is non-homogeneous, so no trace).
  if (eflag_atom) d_eatom(i) += static_cast<KK_ACC_FLOAT>(d_peatom(i));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double PPPMDispPlanarKokkos<DeviceType>::memory_usage()
{
  double bytes = PPPMDispPlanar::memory_usage();
  bytes += (double) 6 * nz * sizeof(double);       // d_dens, d_Gk, d_GTk, d_GNk, d_fz_grid, d_ugrid
  bytes += (double) 2 * nz * sizeof(double);       // d_uTgrid, d_uNgrid
  bytes += (double) 4 * nz * sizeof(FFT_SCALAR);   // d_work, d_work2
  if (natoms_all_created > 0)
    bytes += (double) 2 * natoms_all_created * sizeof(double);   // d_zall, d_ball
  return bytes;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PPPMDispPlanarKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PPPMDispPlanarKokkos<LMPHostType>;
#endif
}
