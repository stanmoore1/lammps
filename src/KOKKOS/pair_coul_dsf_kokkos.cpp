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

#include "pair_coul_dsf_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "ewald_const.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace EwaldConst;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairCoulDSFKokkos<DeviceType>::PairCoulDSFKokkos(LAMMPS *lmp) : PairCoulDSF(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | Q_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairCoulDSFKokkos<DeviceType>::~PairCoulDSFKokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCoulDSFKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  special_coul[0] = force->special_coul[0];
  special_coul[1] = force->special_coul[1];
  special_coul[2] = force->special_coul[2];
  special_coul[3] = force->special_coul[3];
  qqrd2e = force->qqrd2e;

  int inum = list->inum;

  copymode = 1;

  // loop over neighbors of my atoms

  EV_FLOAT ev;

  // compute kernel A

  if (evflag) {
    if (neighflag == HALF) {
      if (newton_pair) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<HALF,1,1> >(0,inum),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<HALF,0,1> >(0,inum),*this,ev);
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<HALFTHREAD,1,1> >(0,inum),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<HALFTHREAD,0,1> >(0,inum),*this,ev);
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<FULL,1,1> >(0,inum),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<FULL,0,1> >(0,inum),*this,ev);
      }
    }
  } else {
    if (neighflag == HALF) {
      if (newton_pair) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<HALF,1,0> >(0,inum),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<HALF,0,0> >(0,inum),*this);
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<HALFTHREAD,1,0> >(0,inum),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<HALFTHREAD,0,0> >(0,inum),*this);
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<FULL,1,0> >(0,inum),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairCoulDSFKernelA<FULL,0,0> >(0,inum),*this);
      }
    }
  }

  if (eflag_global) eng_coul += ev.ecoul;
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

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairCoulDSFKokkos<DeviceType>::init_style()
{
  PairCoulDSF::init_style();

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
}

/* ---------------------------------------------------------------------- */

////Specialisation for Neighborlist types Half, HalfThread, Full
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCoulDSFKokkos<DeviceType>::operator()(TagPairCoulDSFKernelA<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii, EV_FLOAT& ev) const {

  // The f array is atomic for Half/Thread neighbor style
  Kokkos::View<double*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_f = f;
  Kokkos::View<double*, typename DAT::t_float_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_eatom = k_eatom.view<DeviceType>();

  const int i = d_ilist[ii];
  const double xtmp = x(i,0);
  const double ytmp = x(i,1);
  const double ztmp = x(i,2);
  const double qtmp = q[i];

  if (eflag) {
    const double e_self = -(e_shift/2.0 + alpha/MY_PIS) * qtmp*qtmp*qqrd2e;
    if (eflag_global)
      ev.ecoul += e_self;
    if (eflag_atom)
      v_eatom[i] += e_self;
  }

  //const AtomNeighborsConst d_neighbors_i = k_list.get_neighbors_const(i);
  const int jnum = d_numneigh[i];

  double fxtmp = 0.0;
  double fytmp = 0.0;
  double fztmp = 0.0;

  for (int jj = 0; jj < jnum; jj++) {
    //int j = d_neighbors_i(jj);
    int j = d_neighbors(i,jj);
    const double factor_coul = special_coul[sbmask(j)];
    j &= NEIGHMASK;
    const double delx = xtmp - x(j,0);
    const double dely = ytmp - x(j,1);
    const double delz = ztmp - x(j,2);
    const double rsq = delx*delx + dely*dely + delz*delz;

    if (rsq < cut_coulsq) {


      const double r2inv = 1.0/rsq;
      const double r = sqrt(rsq);
      const double prefactor = factor_coul * qqrd2e*qtmp*q[j]/r;
      const double erfcd = exp(-alpha*alpha*rsq);
      const double t = 1.0 / (1.0 + EWALD_P*alpha*r);
      const double erfcc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * erfcd;
      const double forcecoul = prefactor * (erfcc/r + 2.0*alpha/MY_PIS * erfcd +
                                 r*f_shift) * r;
      const double fpair = forcecoul * r2inv;

      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;

      if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < nlocal)) {
        a_f(j,0) -= delx*fpair;
        a_f(j,1) -= dely*fpair;
        a_f(j,2) -= delz*fpair;
      }

      if (EVFLAG) {
        double ecoul = 0.0;
        if (eflag) {
          ecoul = prefactor * (erfcc - r*e_shift - rsq*f_shift);
          ev.ecoul += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<nlocal)))?1.0:0.5)*ecoul;
        }

        if (vflag_either || eflag_atom) this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev,i,j,ecoul,fpair,delx,dely,delz);
      }

    }
  }

  a_f(i,0) += fxtmp;
  a_f(i,1) += fytmp;
  a_f(i,2) += fztmp;
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCoulDSFKokkos<DeviceType>::operator()(TagPairCoulDSFKernelA<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(TagPairCoulDSFKernelA<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairCoulDSFKokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const double &epair, const double &fpair, const double &delx,
                const double &dely, const double &delz) const
{
  const int EFLAG = eflag;
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are atomic for Half/Thread neighbor style
  Kokkos::View<double*, typename DAT::t_float_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_eatom = k_eatom.view<DeviceType>();
  Kokkos::View<double*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_vatom = k_vatom.view<DeviceType>();

  if (EFLAG) {
    if (eflag_atom) {
      const double epairhalf = 0.5 * epair;
      if (NEIGHFLAG!=FULL) {
        if (NEWTON_PAIR || i < nlocal) v_eatom[i] += epairhalf;
        if (NEWTON_PAIR || j < nlocal) v_eatom[j] += epairhalf;
      } else {
        v_eatom[i] += epairhalf;
      }
    }
  }

  if (VFLAG) {
    const double v0 = delx*delx*fpair;
    const double v1 = dely*dely*fpair;
    const double v2 = delz*delz*fpair;
    const double v3 = delx*dely*fpair;
    const double v4 = delx*delz*fpair;
    const double v5 = dely*delz*fpair;

    if (vflag_global) {
      if (NEIGHFLAG!=FULL) {
        if (NEWTON_PAIR || i < nlocal) {
          ev.v[0] += 0.5*v0;
          ev.v[1] += 0.5*v1;
          ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3;
          ev.v[4] += 0.5*v4;
          ev.v[5] += 0.5*v5;
        }
        if (NEWTON_PAIR || j < nlocal) {
        ev.v[0] += 0.5*v0;
        ev.v[1] += 0.5*v1;
        ev.v[2] += 0.5*v2;
        ev.v[3] += 0.5*v3;
        ev.v[4] += 0.5*v4;
        ev.v[5] += 0.5*v5;
        }
      } else {
        ev.v[0] += 0.5*v0;
        ev.v[1] += 0.5*v1;
        ev.v[2] += 0.5*v2;
        ev.v[3] += 0.5*v3;
        ev.v[4] += 0.5*v4;
        ev.v[5] += 0.5*v5;
      }
    }

    if (vflag_atom) {
      if (NEIGHFLAG!=FULL) {
        if (NEWTON_PAIR || i < nlocal) {
          v_vatom(i,0) += 0.5*v0;
          v_vatom(i,1) += 0.5*v1;
          v_vatom(i,2) += 0.5*v2;
          v_vatom(i,3) += 0.5*v3;
          v_vatom(i,4) += 0.5*v4;
          v_vatom(i,5) += 0.5*v5;
        }
        if (NEWTON_PAIR || j < nlocal) {
        v_vatom(j,0) += 0.5*v0;
        v_vatom(j,1) += 0.5*v1;
        v_vatom(j,2) += 0.5*v2;
        v_vatom(j,3) += 0.5*v3;
        v_vatom(j,4) += 0.5*v4;
        v_vatom(j,5) += 0.5*v5;
        }
      } else {
        v_vatom(i,0) += 0.5*v0;
        v_vatom(i,1) += 0.5*v1;
        v_vatom(i,2) += 0.5*v2;
        v_vatom(i,3) += 0.5*v3;
        v_vatom(i,4) += 0.5*v4;
        v_vatom(i,5) += 0.5*v5;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int PairCoulDSFKokkos<DeviceType>::sbmask(const int& j) const {
  return j >> SBBITS & 3;
}

namespace LAMMPS_NS {
template class PairCoulDSFKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairCoulDSFKokkos<LMPHostType>;
#endif
}

