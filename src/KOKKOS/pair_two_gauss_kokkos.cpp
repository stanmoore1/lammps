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

#include "pair_two_gauss_kokkos.h"
#include <cmath>
#include <cstring>
#include "kokkos.h"
#include "atom_kokkos.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "memory_kokkos.h"
#include "error.h"
#include "atom_masks.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;


/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairTwoGaussKokkos<DeviceType>::PairTwoGaussKokkos(LAMMPS *lmp) : PairTwoGauss(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairTwoGaussKokkos<DeviceType>::~PairTwoGaussKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom,eatom);
  memoryKK->destroy_kokkos(k_vatom,vatom);
  memoryKK->destroy_kokkos(k_cutsq,cutsq);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairTwoGaussKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

  // loop over neighbors of my atoms

  copymode = 1;

  EV_FLOAT ev = pair_compute<PairTwoGaussKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairTwoGaussKokkos<DeviceType>::
compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;

  F_FLOAT fpair = 0;
  F_FLOAT taper = 0;

  const F_FLOAT r = sqrt(rsq);
  const F_FLOAT cutm0p95 = STACKPARAMS?m_params[itype][jtype].cutm0p95:params(itype,jtype).cutm0p95;

  if (r == cutm0p95) {
    taper = 0.5;
  } else {
    F_FLOAT taperRR = rsq / (cutm0p95 * cutm0p95);
    F_FLOAT taperRR2 = taperRR * taperRR;
    F_FLOAT taperRR4 = taperRR2 * taperRR2;
    F_FLOAT taperRR10 = taperRR2 * taperRR4 * taperRR4;
    F_FLOAT taperRR30 = taperRR10 * taperRR10 * taperRR10;
    F_FLOAT taperRR60 = taperRR30 * taperRR30;
    taper = (1. - taperRR30) / (1 - taperRR60);
  }

  const F_FLOAT rmh = STACKPARAMS?m_params[itype][jtype].rmh:params(itype,jtype).rmh;
  const F_FLOAT invsigmah = STACKPARAMS?m_params[itype][jtype].invsigmah:params(itype,jtype).invsigmah;
  const F_FLOAT pgauss = STACKPARAMS?m_params[itype][jtype].pgauss:params(itype,jtype).pgauss;

  const F_FLOAT rexp = (r - rmh) * invsigmah;
  const F_FLOAT ugauss = pgauss * exp(-0.5 * rexp * rexp);
  fpair = taper * rexp / r * ugauss * invsigmah;

  const F_FLOAT rmh2 = STACKPARAMS?m_params[itype][jtype].rmh2:params(itype,jtype).rmh2;
  const F_FLOAT invsigmah2 = STACKPARAMS?m_params[itype][jtype].invsigmah2:params(itype,jtype).invsigmah2;
  const F_FLOAT pgauss2 = STACKPARAMS?m_params[itype][jtype].pgauss:params(itype,jtype).pgauss2;

  const F_FLOAT rexp2 = (r - rmh2) * invsigmah2;
  const F_FLOAT ugauss2 = pgauss2 * exp(-0.5 * rexp2 * rexp2);
  fpair += taper * rexp2 / r * ugauss2 * invsigmah2;

  return fpair;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairTwoGaussKokkos<DeviceType>::
compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;

  F_FLOAT evdwl = 0;

  F_FLOAT taper = 0;

  const F_FLOAT r = sqrt(rsq);
  const F_FLOAT cutm0p95 = STACKPARAMS?m_params[itype][jtype].cutm0p95:params(itype,jtype).cutm0p95;

  if (r == cutm0p95) {
    taper = 0.5;
  } else {
    F_FLOAT taperRR = rsq / (cutm0p95 * cutm0p95);
    F_FLOAT taperRR2 = taperRR * taperRR;
    F_FLOAT taperRR4 = taperRR2 * taperRR2;
    F_FLOAT taperRR10 = taperRR2 * taperRR4 * taperRR4;
    F_FLOAT taperRR30 = taperRR10 * taperRR10 * taperRR10;
    F_FLOAT taperRR60 = taperRR30 * taperRR30;
    taper = (1. - taperRR30) / (1 - taperRR60);
  }

  const F_FLOAT rmh = STACKPARAMS?m_params[itype][jtype].rmh:params(itype,jtype).rmh;
  const F_FLOAT invsigmah = STACKPARAMS?m_params[itype][jtype].invsigmah:params(itype,jtype).invsigmah;
  const F_FLOAT pgauss = STACKPARAMS?m_params[itype][jtype].pgauss:params(itype,jtype).pgauss;

  const F_FLOAT rexp = (r - rmh) * invsigmah;
  const F_FLOAT ugauss = pgauss * exp(-0.5 * rexp * rexp);

  const F_FLOAT rmh2 = STACKPARAMS?m_params[itype][jtype].rmh2:params(itype,jtype).rmh2;
  const F_FLOAT invsigmah2 = STACKPARAMS?m_params[itype][jtype].invsigmah2:params(itype,jtype).invsigmah2;
  const F_FLOAT pgauss2 = STACKPARAMS?m_params[itype][jtype].pgauss:params(itype,jtype).pgauss2;

  const F_FLOAT rexp2 = (r - rmh2) * invsigmah2;
  const F_FLOAT ugauss2 = pgauss2 * exp(-0.5 * rexp2 * rexp2);

  const F_FLOAT offset = STACKPARAMS?m_params[itype][jtype].offset:params(itype,jtype).offset;
  evdwl = (ugauss + ugauss2) * taper - offset;

  return evdwl;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairTwoGaussKokkos<DeviceType>::allocate()
{
  PairTwoGauss::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  k_params = Kokkos::DualView<params_two_gauss**,Kokkos::LayoutRight,DeviceType>("PairTwoGauss::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairTwoGaussKokkos<DeviceType>::init_style()
{
  PairTwoGauss::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style,"^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa)
      error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // modify neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
  if (neighflag == FULL) request->enable_full();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairTwoGaussKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairTwoGauss::init_one(i,j);

  k_params.h_view(i,j).cutm0p95 = cut[i][j] - 0.95;
  k_params.h_view(i,j).rmh = rmh[i][j];
  k_params.h_view(i,j).pgauss = pgauss[i][j];
  k_params.h_view(i,j).invsigmah = 1. / sigmah[i][j];
  k_params.h_view(i,j).rmh2 = rmh2[i][j];
  k_params.h_view(i,j).pgauss2 = pgauss2[i][j];
  k_params.h_view(i,j).invsigmah2 = 1. / sigmah2[i][j];
  k_params.h_view(i,j).offset = offset[i][j];
  k_params.h_view(i,j).cutsq = cutone*cutone;
  k_params.h_view(j,i) = k_params.h_view(i,j);
  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
  }

  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  return cutone;
}

namespace LAMMPS_NS {
template class PairTwoGaussKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairTwoGaussKokkos<LMPHostType>;
#endif
}

