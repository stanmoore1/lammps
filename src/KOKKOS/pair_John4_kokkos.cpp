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

#include "pair_John4_kokkos.h"
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

#define KOKKOS_CUDA_MAX_THREADS 256
#define KOKKOS_CUDA_MIN_BLOCKS 8

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairJohn4Kokkos<DeviceType>::PairJohn4Kokkos(LAMMPS *lmp) : PairJohn4(lmp)
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
PairJohn4Kokkos<DeviceType>::~PairJohn4Kokkos()
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
void PairJohn4Kokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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
  EV_FLOAT ev = pair_compute<PairJohn4Kokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);

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
F_FLOAT PairJohn4Kokkos<DeviceType>::
compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;

  const F_FLOAT r = sqrt(rsq);
  const F_FLOAT A_ij = STACKPARAMS?m_params[itype][jtype].A:params(itype,jtype).A;
  const F_FLOAT r0_ij = STACKPARAMS?m_params[itype][jtype].r0:params(itype,jtype).r0;
  const F_FLOAT B_ij = STACKPARAMS?m_params[itype][jtype].B:params(itype,jtype).B;
  const F_FLOAT rc_ij = STACKPARAMS?m_params[itype][jtype].rc:params(itype,jtype).rc;

  F_FLOAT fpair = 0.0;
  if (r < r0_ij) {
    fpair = A_ij*cos(MY_PI*r/(2*r0_ij))/r;
  } else {  
    fpair = B_ij*cos(MY_PI*(r-r0_ij)/(rc_ij-r0_ij)+MY_PI/2)/r;
  }

  // delta_x, factor_ij multiplication gets handled separately
  return fpair;

}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairJohn4Kokkos<DeviceType>::
compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const {
  (void) i;
  (void) j;
  
  const F_FLOAT r = sqrt(rsq);
  const F_FLOAT A_ij = STACKPARAMS?m_params[itype][jtype].A:params(itype,jtype).A;
  const F_FLOAT r0_ij = STACKPARAMS?m_params[itype][jtype].r0:params(itype,jtype).r0;
  const F_FLOAT B_ij = STACKPARAMS?m_params[itype][jtype].B:params(itype,jtype).B;
  const F_FLOAT rc_ij = STACKPARAMS?m_params[itype][jtype].rc:params(itype,jtype).rc;

  F_FLOAT evdwl = 0.0;
  if (r < r0_ij) {
    evdwl= -A_ij*2*r0_ij/MY_PI*sin(MY_PI*r/(2*r0_ij)) + 2*A_ij*r0_ij/MY_PI - 2*B_ij*(rc_ij-r0_ij)/MY_PI;
  } else {
    evdwl= -B_ij*(rc_ij-r0_ij)/MY_PI*sin(MY_PI*(r-r0_ij)/(rc_ij-r0_ij)+MY_PI/2) - B_ij*(rc_ij-r0_ij)/MY_PI;
  }

  return evdwl;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairJohn4Kokkos<DeviceType>::allocate()
{
  PairJohn4::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  k_params = Kokkos::DualView<params_john4**,Kokkos::LayoutRight,DeviceType>("PairJohn4::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairJohn4Kokkos<DeviceType>::init_style()
{
  PairJohn4::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
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
double PairJohn4Kokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairJohn4::init_one(i,j);

  k_params.h_view(i,j).A = A_vec[i][j];
  k_params.h_view(i,j).r0 = r0_vec[i][j];
  k_params.h_view(i,j).B = B_vec[i][j];
  k_params.h_view(i,j).rc = rc_vec[i][j];
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
template class PairJohn4Kokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairJohn4Kokkos<LMPHostType>;
#endif
}

