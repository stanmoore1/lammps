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
   Kokkos port of pppm/disp/planar.  The hot, data-parallel parts of the mesh
   reciprocal solve run on the device: the B-weighted density spread (make_rho),
   the per-mode influence/energy/force-field work (poisson), the 1d FFTs (via
   FFT3dKokkos, a local MPI_COMM_SELF nz x 1 x 1 plan), and the z-force/per-atom
   interpolation (fieldforce).  The smooth-damped real-space correction is
   folded into the influence function on the host (base class), so there is no
   device correction step.  The MPI density gather runs on the host.
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
  nchan_created = 0;
  nkap_created = 0;
  chan_kk = 0;
  nmax_kk = 0;
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

/* ----------------------------------------------------------------------
   base setup() fills nz, Gk/GTk/GNk (merged corr), rho_coeff, B; allocate and
   upload the device data
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::setup()
{
  // PPPMDispPlanar::setup() builds the corr FT tables (host) and calls the virtual
  // influence_function(), which this class overrides to build Gk/GTk/GNk directly on
  // the device (no host serial loop, no host->device copy) -- NPT-safe.  Device array
  // allocation happens inside the influence_function() override.
  PPPMDispPlanar::setup();

  // upload the (small, box-independent-once-order-is-fixed) B-spline coefficients and
  // the per-type dispersion amplitudes (nchan channels per type)
  auto h_rho = Kokkos::create_mirror_view(d_rho_coeff);
  for (int l = 0; l < order; l++)
    for (int s = 0; s < order; s++) h_rho(l, s) = rho_coeff[l][s];
  Kokkos::deep_copy(d_rho_coeff, h_rho);

  const int ntypes = atom->ntypes;
  auto h_B = Kokkos::create_mirror_view(d_B);
  for (int idx = 0; idx < nchan * (ntypes + 1); idx++) h_B(idx) = B[idx];
  Kokkos::deep_copy(d_B, h_B);
}

/* ----------------------------------------------------------------------
   build the de-convolved influence function Gk/GTk/GNk on the device.  Overrides
   the host PPPMDispPlanar::influence_function (called from PPPMDispPlanar::setup),
   so the per-step NPT rebuild runs entirely on the device: the box-independent
   corr Fourier-transform tables are uploaded once (grow-only) and interpolated in a
   device kernel that also evaluates the reciprocal (erfc) coefficients per mode.
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::influence_function()
{
  allocate_device();    // ensure d_Gk/d_GTk/d_GNk (and the FFT plans) exist for this nz

  // (re)upload the box-independent corr FT tables when they grow (NPT box shrink)
  if (nkap != nkap_created) {
    d_Araw_tab = typename AT::t_double_1d("pppm/disp/planar/kk:Araw_tab", nkap + 1);
    d_Braw_tab = typename AT::t_double_1d("pppm/disp/planar/kk:Braw_tab", nkap + 1);
    auto hA = Kokkos::create_mirror_view(d_Araw_tab);
    auto hB = Kokkos::create_mirror_view(d_Braw_tab);
    for (int j = 0; j <= nkap; j++) { hA(j) = Araw_tab[j]; hB(j) = Braw_tab[j]; }
    Kokkos::deep_copy(d_Araw_tab, hA);
    Kokkos::deep_copy(d_Braw_tab, hB);
    nkap_created = nkap;
  }

  // refresh the device scalars the influence kernel reads (box-dependent under NPT)
  nz_kk = nz;
  order_kk = order;
  zprd_kk = zprd;
  g_ewald_kk = g_ewald;
  volume_kk = volume;
  nkap_kk = nkap;
  kap_dk_kk = kap_dk;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_influence>(0, nz), *this);
  copymode = 0;
}

/* ----------------------------------------------------------------------
   (re)allocate device arrays and the local 1d FFT plans when nz changes
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispPlanarKokkos<DeviceType>::allocate_device()
{
  const int ntypes = atom->ntypes;
  // B carries nchan channels: B[t] geometric, B[7t+j] arithmetic
  d_B = typename AT::t_double_1d("pppm/disp/planar/kk:B", nchan * (ntypes + 1));
  d_rho_coeff = typename AT::t_double_2d("pppm/disp/planar/kk:rho_coeff", order, order);

  if (nz == nz_created && nchan == nchan_created) return;
  nz_created = nz;
  nchan_created = nchan;

  // density and force/potential fields are channel-major [c*nz + g]
  d_dens = typename AT::t_double_1d("pppm/disp/planar/kk:dens", nz * nchan);
  d_Gk = typename AT::t_double_1d("pppm/disp/planar/kk:Gk", nz);
  d_GTk = typename AT::t_double_1d("pppm/disp/planar/kk:GTk", nz);
  d_GNk = typename AT::t_double_1d("pppm/disp/planar/kk:GNk", nz);
  d_fz_grid = typename AT::t_double_1d("pppm/disp/planar/kk:fz_grid", nz * nchan);
  d_ugrid = typename AT::t_double_1d("pppm/disp/planar/kk:ugrid", nz * nchan);
  d_uTgrid = typename AT::t_double_1d("pppm/disp/planar/kk:uTgrid", nz * nchan);
  d_uNgrid = typename AT::t_double_1d("pppm/disp/planar/kk:uNgrid", nz * nchan);
  d_rhat_re = typename AT::t_double_1d("pppm/disp/planar/kk:rhat_re", nz * nchan);
  d_rhat_im = typename AT::t_double_1d("pppm/disp/planar/kk:rhat_im", nz * nchan);
  h_dens = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "pppm/disp/planar/kk:h_dens", nz * nchan);

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
  nchan_kk = nchan;
  chan_kk = 0;

  const int nlocal = atomKK->nlocal;

  // --- make_rho: spread the (nchan-channel) dispersion density onto the z grid ---

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_make_rho_zero>(0, nz * nchan), *this);
  if (nchan == 1)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_make_rho>(0, nlocal), *this);
  else
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_make_rho_arith>(0, nlocal), *this);
  copymode = 0;

  // gather the global (nchan-channel) density across procs (host Allreduce, in place)
  Kokkos::deep_copy(h_dens, d_dens);
  MPI_Allreduce(MPI_IN_PLACE, h_dens.data(), nz * nchan, MPI_DOUBLE, MPI_SUM, world);
  Kokkos::deep_copy(d_dens, h_dens);

  // --- poisson: FFT, influence function, energy, virial, z-force field ---

  if (nchan == 1) {

    // ------- geometric mixing: single density channel -------
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_dens_to_work>(0, nz), *this);
    copymode = 0;
    fft_forward->compute1d(d_work, 2 * nz, FFT3dKokkos<DeviceType>::FORWARD);

    double e = 0.0;
    copymode = 1;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_energy>(0, nz),
                            *this, e);
    copymode = 0;
    if (eflag_global) energy += e;
    if (vflag_global) {
      // explicit tangential (GTk) and normal (GNk) virial kernels (the merged corr
      // makes the kspace share non-homogeneous, so the 6E trace does not apply)
      s_vir vir;
      copymode = 1;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_poisson_virial>(0, nz), *this, vir);
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

  } else {

    // ------- arithmetic (Lorentz-Berthelot): 7 density channels -------
    // forward FFT each channel's density -> d_rhat_re/d_rhat_im[c*nz + mode]
    for (int c = 0; c < nchan; c++) {
      chan_kk = c;
      copymode = 1;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_dens_to_work>(0, nz), *this);
      copymode = 0;
      fft_forward->compute1d(d_work, 2 * nz, FFT3dKokkos<DeviceType>::FORWARD);
      copymode = 1;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_store_rhat>(0, nz), *this);
      copymode = 0;
    }

    // energy/virial from the per-mode channel pairing (as_e = 1/8)
    const double as_e = 0.125;
    double e = 0.0;
    copymode = 1;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_energy_arith>(0, nz),
                            *this, e);
    copymode = 0;
    if (eflag_global) energy += as_e * e;
    if (vflag_global) {
      s_vir vir;
      copymode = 1;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_virial_arith>(0, nz), *this, vir);
      copymode = 0;
      virial[lat1] += as_e * vir.vt;
      virial[lat2] += as_e * vir.vt;
      virial[dim] += as_e * vir.vn;
    }

    // per-channel z-force field (and per-atom potential/virial fields) via inverse FFTs
    for (int c = 0; c < nchan; c++) {
      chan_kk = c;
      copymode = 1;
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_work_from_rhat>(0, nz), *this);
      copymode = 0;

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
    }
    chan_kk = 0;
  }

  // --- fieldforce: interpolate the z-force field(s) to atoms (device) ---

  // zero per-atom accumulators before the reciprocal contributions
  if (evflag_atom) {
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_peatom_zero>(0, nlocal), *this);
    copymode = 0;
  }

  copymode = 1;
  if (nchan == 1)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_fieldforce>(0, nlocal), *this);
  else
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_fieldforce_arith>(0, nlocal), *this);
  copymode = 0;

  if (evflag_atom) {
    copymode = 1;
    if (nchan == 1)
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_fieldforce_peratom>(0, nlocal), *this);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPPPMDispPlanar_fieldforce_peratom_arith>(0, nlocal), *this);
    copymode = 0;
  }

  atomKK->modified(execution_space, F_MASK);

  // --- per-atom energy finalization (device kernel) ---

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
}

/* ---------------------------------------------------------------------- */
/* device kernels — mesh reciprocal solve                                 */
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
  d_work(2 * m) = static_cast<FFT_SCALAR>(d_dens(chan_kk * nz_kk + m));
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
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_poisson_virial, const int &m,
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
  d_uTgrid(chan_kk * nz_kk + m) = static_cast<double>(d_work2(2 * m));
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
  d_uNgrid(chan_kk * nz_kk + m) = static_cast<double>(d_work2(2 * m));
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
  d_fz_grid(chan_kk * nz_kk + m) = static_cast<double>(d_work2(2 * m));
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
  d_ugrid(chan_kk * nz_kk + m) = static_cast<double>(d_work2(2 * m));
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

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_peratom_finalize,
                                                const int &i) const
{
  // d_peatom holds the full kspace per-atom energy; the normal per-atom virial is
  // the explicit GNk field (set in fieldforce_peratom), no trace needed
  if (eflag_atom) d_eatom(i) += static_cast<KK_ACC_FLOAT>(d_peatom(i));
}

/* ----------------------------------------------------------------------
   de-convolved smooth-damped influence function on the device (per grid mode m).
   Exact mirror of PPPMDispPlanar::influence_function; the box-independent corr
   Fourier transform is interpolated from the uploaded tables (ft_interp_kk).
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_influence, const int &m) const
{
  const double sqpi = sqrt(MY_PI);
  const double pre2 = 2.0 * MY_PI / volume_kk;
  const double coef = -2.0 * MY_PI * sqpi / (24.0 * volume_kk);
  const double g = g_ewald_kk;

  if (m == 0) {
    double A0, B0;
    ft_interp_kk(0.0, A0, B0);
    const double ce0 = 0.5 * pre2 * A0;
    const double gk0 = -MY_PI * sqpi * g * g * g / (6.0 * volume_kk) + ce0;
    d_Gk(0) = gk0;
    d_GTk(0) = gk0;
    d_GNk(0) = gk0;    // reciprocal GN(0) = GU(0); corr CN(0) = CE(0) since B(0) = 0
    return;
  }

  const int mm = (m <= nz_kk / 2) ? m : m - nz_kk;
  const double k = mm * 2.0 * MY_PI / zprd_kk;
  const double ak = fabs(k);
  const double b = ak / (2.0 * g), b2 = b * b, b3 = b2 * b;
  const double Bk = ak * ak * ak * (sqpi * erfc(b) + (0.5 / b3 - 1.0 / b) * exp(-b2));
  const double WE = 0.5 * coef * Bk;
  const double sinc = sin(MY_PI * mm / nz_kk) / (MY_PI * mm / nz_kk);
  const double w2 = pow(sinc, 2.0 * order_kk);

  double A, Bv;
  ft_interp_kk(ak, A, Bv);
  const double CE = 0.5 * pre2 * A;
  const double CN = 0.5 * pre2 * (A - ak * Bv);
  const double WN = 0.5 * coef * (4.0 * Bk - 1.5 * ak * ak * ak * exp(-b2) / b3);
  d_Gk(m) = (WE + CE) / w2;
  d_GTk(m) = d_Gk(m);
  d_GNk(m) = (WN + CN) / w2;
}

/* ----------------------------------------------------------------------
   arithmetic (Lorentz-Berthelot) mixing: 7-channel device kernels.  Mirror of the
   nchan==7 branches of PPPMDispPlanar::make_rho/poisson/fieldforce.  Channel m of
   an atom pairs with field channel (6-m); energy/virial use the per-mode channel
   pairing R = sum_{a+b=6} Re(rhat_a conj(rhat_b)), normalized by as_e = 1/8 (energy,
   virial) and as_f = 1/16 (z-force).
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_make_rho_arith,
                                                  const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzinv_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8];
  compute_rho1d_kk(dz, w);
  const int t7 = 7 * type(i);
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nz_kk) + nz_kk) % nz_kk;
    const double ws = w[s];
    for (int c = 0; c < 7; c++) Kokkos::atomic_add(&d_dens(c * nz_kk + g), d_B(t7 + c) * ws);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_store_rhat, const int &m) const
{
  d_rhat_re(chan_kk * nz_kk + m) = static_cast<double>(d_work(2 * m));
  d_rhat_im(chan_kk * nz_kk + m) = static_cast<double>(d_work(2 * m + 1));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_work_from_rhat,
                                                  const int &m) const
{
  d_work(2 * m) = static_cast<FFT_SCALAR>(d_rhat_re(chan_kk * nz_kk + m));
  d_work(2 * m + 1) = static_cast<FFT_SCALAR>(d_rhat_im(chan_kk * nz_kk + m));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_energy_arith, const int &m,
                                                  double &esum) const
{
  const int nz = nz_kk;
  const double r0 = d_rhat_re(m), i0 = d_rhat_im(m);
  const double r1 = d_rhat_re(nz + m), i1 = d_rhat_im(nz + m);
  const double r2 = d_rhat_re(2 * nz + m), i2 = d_rhat_im(2 * nz + m);
  const double r3 = d_rhat_re(3 * nz + m), i3 = d_rhat_im(3 * nz + m);
  const double r4 = d_rhat_re(4 * nz + m), i4 = d_rhat_im(4 * nz + m);
  const double r5 = d_rhat_re(5 * nz + m), i5 = d_rhat_im(5 * nz + m);
  const double r6 = d_rhat_re(6 * nz + m), i6 = d_rhat_im(6 * nz + m);
  const double R = (r0 * r6 + i0 * i6) + (r1 * r5 + i1 * i5) + (r2 * r4 + i2 * i4) +
      0.5 * (r3 * r3 + i3 * i3);
  esum += d_Gk(m) * R;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_virial_arith, const int &m,
                                                  s_vir &vir) const
{
  const int nz = nz_kk;
  const double r0 = d_rhat_re(m), i0 = d_rhat_im(m);
  const double r1 = d_rhat_re(nz + m), i1 = d_rhat_im(nz + m);
  const double r2 = d_rhat_re(2 * nz + m), i2 = d_rhat_im(2 * nz + m);
  const double r3 = d_rhat_re(3 * nz + m), i3 = d_rhat_im(3 * nz + m);
  const double r4 = d_rhat_re(4 * nz + m), i4 = d_rhat_im(4 * nz + m);
  const double r5 = d_rhat_re(5 * nz + m), i5 = d_rhat_im(5 * nz + m);
  const double r6 = d_rhat_re(6 * nz + m), i6 = d_rhat_im(6 * nz + m);
  const double R = (r0 * r6 + i0 * i6) + (r1 * r5 + i1 * i5) + (r2 * r4 + i2 * i4) +
      0.5 * (r3 * r3 + i3 * i3);
  vir.vt += d_GTk(m) * R;
  vir.vn += d_GNk(m) * R;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_fieldforce_arith,
                                                  const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzinv_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8];
  compute_rho1d_kk(dz, w);
  const int t7 = 7 * type(i);

  double fz = 0.0;
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nz_kk) + nz_kk) % nz_kk;
    const double ws = w[s];
    for (int c = 0; c < 7; c++) {
      const double a = d_B(t7 + (6 - c));    // atom channel (6-c) pairs with field channel c
      fz += a * ws * d_fz_grid(c * nz_kk + g);
    }
  }
  f(i, dim_kk) += static_cast<KK_ACC_FLOAT>((1.0 / 16.0) * fz);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMDispPlanarKokkos<DeviceType>::operator()(TagPPPMDispPlanar_fieldforce_peratom_arith,
                                                  const int &i) const
{
  const double u = (static_cast<double>(x(i, dim_kk)) - zlo_kk) * delzinv_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8];
  compute_rho1d_kk(dz, w);
  const int t7 = 7 * type(i);

  double uu = 0.0, uT = 0.0, uN = 0.0;
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nz_kk) + nz_kk) % nz_kk;
    const double ws = w[s];
    for (int c = 0; c < 7; c++) {
      const double a = d_B(t7 + (6 - c));
      uu += a * ws * d_ugrid(c * nz_kk + g);
      uT += a * ws * d_uTgrid(c * nz_kk + g);
      uN += a * ws * d_uNgrid(c * nz_kk + g);
    }
  }
  const double as_e = 0.125;
  d_peatom(i) += 0.25 * as_e * uu;
  if (vflag_atom) {
    d_vatom(i, lat1_kk) += static_cast<KK_ACC_FLOAT>(0.25 * as_e * uT);
    d_vatom(i, lat2_kk) += static_cast<KK_ACC_FLOAT>(0.25 * as_e * uT);
    d_vatom(i, dim_kk) += static_cast<KK_ACC_FLOAT>(0.25 * as_e * uN);
  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   long-range Irving-Kirkwood pressure profile hook (compute stress/cartesian).
   The inherited host implementation reads atom->x/type on the host, so sync the
   KK atom data to host first (mirrors corr_gather); then delegate to the base.
------------------------------------------------------------------------- */

template<class DeviceType>
int PPPMDispPlanarKokkos<DeviceType>::pressure_profile_long(int dir, int nbins, double lo,
                                                          double width, double *pN, double *pT)
{
  atomKK->sync(Host, X_MASK | TYPE_MASK);
  return PPPMDispPlanar::pressure_profile_long(dir, nbins, lo, width, pN, pT);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double PPPMDispPlanarKokkos<DeviceType>::memory_usage()
{
  double bytes = PPPMDispPlanar::memory_usage();
  // channel-major fields: dens, fz_grid, ugrid, uTgrid, uNgrid, rhat_re, rhat_im, h_dens
  bytes += (double) 8 * nz * nchan * sizeof(double);
  bytes += (double) 3 * nz * sizeof(double);       // d_Gk, d_GTk, d_GNk
  bytes += (double) 4 * nz * sizeof(FFT_SCALAR);   // d_work, d_work2
  return bytes;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PPPMDispPlanarKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PPPMDispPlanarKokkos<LMPHostType>;
#endif
}
