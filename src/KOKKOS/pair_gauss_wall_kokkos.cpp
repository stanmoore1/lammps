/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "pair_gauss_wall_kokkos.h"
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

using namespace LAMMPS_NS;

#define KOKKOS_CUDA_MAX_THREADS 256
#define KOKKOS_CUDA_MIN_BLOCKS 8

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairGaussWallKokkos<DeviceType>::PairGaussWallKokkos(LAMMPS *lmp) : PairGaussWall(lmp)
{
  respa_enable = 0;

  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  cutsq = nullptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairGaussWallKokkos<DeviceType>::~PairGaussWallKokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    k_cutsq = DAT::tdual_ffloat_2d();
    memory->sfree(cutsq);
    eatom = nullptr;
    vatom = nullptr;
    cutsq = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairGaussWallKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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
  tag = atomKK->k_tag.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

  // loop over neighbors of my atoms

  copymode = 1;
  EV_FLOAT ev = pair_compute<PairGaussWallKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  copymode = 0;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairGaussWallKokkos<DeviceType>::
compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;

  const F_FLOAT cutm0p95_ij = (STACKPARAMS?m_params[itype][jtype].cutm0p95:params(itype,jtype).cutm0p95);
  F_FLOAT taper = 0;

  const F_FLOAT r = sqrt(rsq);
  if (r == cutm0p95_ij) {
    taper = 0.5;
  } else {
    const F_FLOAT taperR = r / cutm0p95_ij;
    const F_FLOAT taperRR = taperR * taperR;
    const F_FLOAT taperRR2 = taperRR * taperRR;
    const F_FLOAT taperRR4 = taperRR2 * taperRR2;
    const F_FLOAT taperRR8 = taperRR4 * taperRR4;
    const F_FLOAT taperRR16 = taperRR8 * taperRR8;
    const F_FLOAT taperRR30 = taperRR16 * taperRR8 * taperRR4 * taperRR2;
    const F_FLOAT taperRR60 = taperRR30 * taperRR30;
    taper = (1. - taperRR30) / (1. - taperRR60);
  }

  const F_FLOAT rmh_ij = STACKPARAMS?m_params[itype][jtype].rmh:params(itype,jtype).rmh;
  const F_FLOAT rmh2_ij = STACKPARAMS?m_params[itype][jtype].rmh2:params(itype,jtype).rmh2;
  const F_FLOAT pgauss_ij = STACKPARAMS?m_params[itype][jtype].pgauss:params(itype,jtype).pgauss;
  const F_FLOAT pgauss2_ij = STACKPARAMS?m_params[itype][jtype].pgauss2:params(itype,jtype).pgauss2;
  const F_FLOAT sigmahinv_ij = STACKPARAMS?m_params[itype][jtype].sigmahinv:params(itype,jtype).sigmahinv;
  const F_FLOAT sigmah2inv_ij = STACKPARAMS?m_params[itype][jtype].sigmah2inv:params(itype,jtype).sigmah2inv;

  const F_FLOAT rexp = (r - rmh_ij) * sigmahinv_ij;
  const F_FLOAT ugauss = pgauss_ij * exp(-0.5 * rexp * rexp);
  const F_FLOAT fpair = taper * rexp / r * ugauss * sigmahinv_ij;

  const F_FLOAT rexp2 = (r - rmh2_ij) * sigmah2inv_ij;
  const F_FLOAT ugauss2 = pgauss2_ij * exp(-0.5 * rexp2 * rexp2);
  const F_FLOAT fpair2 = rexp2 / r * ugauss2 * sigmah2inv_ij;
  // ^ lack of a "taper" factor is NOT a typo

  // delta_x, factor_ij multiplication gets handled separately
  return (fpair + fpair2);

}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairGaussWallKokkos<DeviceType>::
compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;
  const F_FLOAT cutm0p95_ij = (STACKPARAMS?m_params[itype][jtype].cutm0p95:params(itype,jtype).cutm0p95);
  F_FLOAT taper = 0;

  const F_FLOAT r = sqrt(rsq);
  if (r == cutm0p95_ij) {
    taper = 0.5;
  } else {
    const F_FLOAT taperR = r / cutm0p95_ij;
    const F_FLOAT taperRR = taperR * taperR;
    const F_FLOAT taperRR2 = taperRR * taperRR;
    const F_FLOAT taperRR4 = taperRR2 * taperRR2;
    const F_FLOAT taperRR8 = taperRR4 * taperRR4;
    const F_FLOAT taperRR16 = taperRR8 * taperRR8;
    const F_FLOAT taperRR30 = taperRR16 * taperRR8 * taperRR4 * taperRR2;
    const F_FLOAT taperRR60 = taperRR30 * taperRR30;
    taper = (1. - taperRR30) / (1. - taperRR60);
  }

  const F_FLOAT rmh_ij = STACKPARAMS?m_params[itype][jtype].rmh:params(itype,jtype).rmh;
  const F_FLOAT rmh2_ij = STACKPARAMS?m_params[itype][jtype].rmh2:params(itype,jtype).rmh2;
  const F_FLOAT pgauss_ij = STACKPARAMS?m_params[itype][jtype].pgauss:params(itype,jtype).pgauss;
  const F_FLOAT pgauss2_ij = STACKPARAMS?m_params[itype][jtype].pgauss2:params(itype,jtype).pgauss2;
  const F_FLOAT sigmahinv_ij = STACKPARAMS?m_params[itype][jtype].sigmahinv:params(itype,jtype).sigmahinv;
  const F_FLOAT sigmah2inv_ij = STACKPARAMS?m_params[itype][jtype].sigmah2inv:params(itype,jtype).sigmah2inv;

  const F_FLOAT rexp = (r - rmh_ij) * sigmahinv_ij;
  const F_FLOAT ugauss = pgauss_ij * exp(-0.5 * rexp * rexp);

  const F_FLOAT rexp2 = (r - rmh2_ij) * sigmah2inv_ij;
  const F_FLOAT ugauss2 = pgauss2_ij * exp(-0.5 * rexp2 * rexp2);

  const F_FLOAT offset_ij = (STACKPARAMS?m_params[itype][jtype].offset:params(itype,jtype).offset);

  return ugauss * taper + ugauss2 - offset_ij;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairGaussWallKokkos<DeviceType>::allocate()
{
  PairGaussWall::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  k_params = Kokkos::DualView<params_gauss_wall**,Kokkos::LayoutRight,DeviceType>("PairGaussWall::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairGaussWallKokkos<DeviceType>::init_style()
{
  PairGaussWall::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa)
      error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // irequest = neigh request made by parent class

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
double PairGaussWallKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairGaussWall::init_one(i,j);

  k_params.h_view(i,j).cutm0p95 = (cut[i][j] - 0.95); // cut always appears in this form in the potential
  k_params.h_view(i,j).sigmahinv = 1./sigmah[i][j];
  k_params.h_view(i,j).sigmah2inv = 1./sigmah2[i][j];
  k_params.h_view(i,j).rmh = rmh[i][j];
  k_params.h_view(i,j).rmh2 = rmh2[i][j];
  k_params.h_view(i,j).pgauss = pgauss[i][j];
  k_params.h_view(i,j).pgauss2 = pgauss2[i][j];
  k_params.h_view(i,j).offset = offset[i][j];
  k_params.h_view(i,j).cutsq = cutone*cutone;
  k_params.h_view(j,i) = k_params.h_view(i,j);
  if(i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
  }
  k_cutsq.h_view(i,j) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  return cutone;
}

namespace LAMMPS_NS {
template class PairGaussWallKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairGaussWallKokkos<LMPHostType>;
#endif
}

