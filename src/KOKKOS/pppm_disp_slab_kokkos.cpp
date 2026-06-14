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
   and the z-force interpolation (fieldforce).  The MPI density gather, the
   real-space slab correction corr(), the rare per-atom energy/virial, and
   the pressure profiles run on the host via the base class.
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
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PPPMDispSlabKokkos<DeviceType>::~PPPMDispSlabKokkos()
{
  if (copymode) return;

  delete fft_forward;
  delete fft_backward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::init()
{
  if (domain->triclinic)
    error->all(FLERR, "Cannot (yet) use pppm/disp/slab/kk with triclinic boxes");

  PPPMDispSlab::init();    // estimates params and calls setup() (this override)
}

/* ----------------------------------------------------------------------
   base setup() fills nz, Gk, rho_coeff, B; allocate/upload device data
------------------------------------------------------------------------- */

template<class DeviceType>
void PPPMDispSlabKokkos<DeviceType>::setup()
{
  PPPMDispSlab::setup();

  allocate_device();

  // upload the per-mode influence function, B-spline coefficients, and B

  auto h_Gk = Kokkos::create_mirror_view(d_Gk);
  for (int m = 0; m < nz; m++) h_Gk(m) = Gk[m];
  Kokkos::deep_copy(d_Gk, h_Gk);

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
  h_dens = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>(
      "pppm/disp/slab/kk:h_dens", nz);

  d_work = typename FFT_AT::t_FFT_SCALAR_1d("pppm/disp/slab/kk:work", 2 * nz);
  d_work2 = typename FFT_AT::t_FFT_SCALAR_1d("pppm/disp/slab/kk:work2", 2 * nz);

  // local (per-proc) 1d FFT: nz x 1 x 1 on MPI_COMM_SELF, no remaps; both
  // unnormalized (scaled=0) to match the base-class forward+inverse convention
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
void PPPMDispSlabKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // grow host peatom buffer and device d_peatom if needed

  if (atom->nmax > nmax) {
    memory->destroy(peatom);
    nmax = atom->nmax;
    memory->create(peatom, nmax, "pppm/disp/slab:peatom");
  }
  if (atom->nmax > nmax_kk) {
    nmax_kk = atom->nmax;
    d_peatom = typename AT::t_double_1d("pppm/disp/slab/kk:peatom", nmax_kk);
  }

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  // device copies of the hot-path scalars

  delzinv_kk = static_cast<KK_FLOAT>(delzinv);
  zlo_kk = static_cast<KK_FLOAT>(zlo);
  shiftone_kk = static_cast<KK_FLOAT>(shiftone);
  zprd_kk = static_cast<KK_FLOAT>(zprd);
  nz_kk = nz;
  order_kk = order;
  nlower_kk = nlower;
  nupper_kk = nupper;

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
    virial[0] += e;
    virial[1] += e;
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
  }

  // --- fieldforce: interpolate the z-force field to atoms (device) ---

  if (evflag_atom) {
    // zero d_peatom on device before accumulating the reciprocal contribution
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_peatom_zero>(0, nlocal), *this);
    copymode = 0;
  }

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_fieldforce>(0, nlocal), *this);
  copymode = 0;
  atomKK->modified(execution_space, F_MASK);

  if (evflag_atom) {
    // per-atom reciprocal energy: interpolate d_ugrid → d_peatom (device)
    copymode = 1;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPPPMDispSlab_fieldforce_peratom>(0, nlocal), *this);
    copymode = 0;

    // transfer d_peatom to host peatom[] so corr() can add its contribution
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_pe(peatom, nlocal);
    Kokkos::deep_copy(h_pe, Kokkos::subview(d_peatom, Kokkos::make_pair(0, nlocal)));
  }

  // damped real-space slab correction on the host (modifies host f,
  // corr_energy, peatom, tangential virial); bring forces to host first

  atomKK->sync(Host, X_MASK | F_MASK | TYPE_MASK);
  corr_energy = 0.0;
  corr();
  atomKK->modified(Host, F_MASK);

  // normal (zz) virial from the exact 1/r^6 virial trace

  if (vflag_global) virial[2] = 6.0 * (e_recip_mesh + corr_energy) - virial[0] - virial[1];
  if (vflag_atom)
    for (int i = 0; i < nlocal; i++)
      vatom[i][2] = 6.0 * peatom[i] - vatom[i][0] - vatom[i][1];
  if (eflag_atom)
    for (int i = 0; i < nlocal; i++) eatom[i] += peatom[i];

  if (profile_flag) compute_pressure_profile();
}

/* ----------------------------------------------------------------------
   device kernels: zero d_peatom and compute per-atom reciprocal energy
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   device kernels
------------------------------------------------------------------------- */

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
  const double u = (static_cast<double>(x(i, 2)) - zlo_kk) * delzinv_kk;
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
  const double u = (static_cast<double>(x(i, 2)) - zlo_kk) * delzinv_kk;
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
  f(i, 2) += static_cast<KK_ACC_FLOAT>(bi * fz);
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
  const double u = (static_cast<double>(x(i, 2)) - zlo_kk) * delzinv_kk;
  const double offs = (order_kk % 2) ? OFFSET + 0.5 : (double) OFFSET;
  const int g0 = (int) (u + offs) - OFFSET;
  const double dz = g0 + shiftone_kk - u;
  double w[8];
  compute_rho1d_kk(dz, w);
  const double bi = d_B(type(i));

  double uu = 0.0;
  for (int s = 0; s < order_kk; s++) {
    int g = g0 + nlower_kk + s;
    g = ((g % nz_kk) + nz_kk) % nz_kk;
    uu += w[s] * d_ugrid(g);
  }
  d_peatom(i) += 0.5 * bi * uu;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double PPPMDispSlabKokkos<DeviceType>::memory_usage()
{
  double bytes = PPPMDispSlab::memory_usage();
  bytes += (double) 4 * nz * sizeof(double);       // d_dens, d_Gk, d_fz_grid, d_ugrid
  bytes += (double) 4 * nz * sizeof(FFT_SCALAR);   // d_work, d_work2
  return bytes;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PPPMDispSlabKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PPPMDispSlabKokkos<LMPHostType>;
#endif
}
