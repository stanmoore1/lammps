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

#include "pair_oxdna_excv_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include "mf_oxdna_kokkos.h"

using namespace LAMMPS_NS;
using namespace MFOxdnaKokkos;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairOxdnaExcvKokkos<DeviceType>::PairOxdnaExcvKokkos(LAMMPS *lmp) : PairOxdnaExcv(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | ELLIPSOID_MASK | BONUS_MASK | F_MASK | 
                  TORQUE_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;

  oxdnaflag = EnabledOXDNAFlag::OXDNA;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairOxdnaExcvKokkos<DeviceType>::~PairOxdnaExcvKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaExcvKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.template view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.template view<DeviceType>();
  }

  // reallocate per-atom orientation arrays if atom storage has grown
  if (atom->nmax > static_cast<int>(k_nx.extent(0))) {
    memoryKK->destroy_kokkos(k_nx);
    memoryKK->destroy_kokkos(k_ny);
    memoryKK->destroy_kokkos(k_nz);
    memoryKK->create_kokkos(k_nx,atom->nmax,3,"PairOxdnaExcv:nx");
    memoryKK->create_kokkos(k_ny,atom->nmax,3,"PairOxdnaExcv:ny");
    memoryKK->create_kokkos(k_nz,atom->nmax,3,"PairOxdnaExcv:nz");
    d_nx = k_nx.template view<DeviceType>();
    d_ny = k_ny.template view<DeviceType>();
    d_nz = k_nz.template view<DeviceType>();
    h_nx = k_nx.view_host();
    h_ny = k_ny.view_host();
    h_nz = k_nz.view_host();
  }

  atomKK->sync(execution_space,datamask_read);

  k_epsilon_bkbk.template sync<DeviceType>();
  k_sigma_bkbk.template sync<DeviceType>();
  k_cut_bkbk_ast.template sync<DeviceType>();
  k_b_bkbk.template sync<DeviceType>();
  k_cut_bkbk_c.template sync<DeviceType>();
  k_lj1_bkbk.template sync<DeviceType>();
  k_lj2_bkbk.template sync<DeviceType>();
  k_cutsq_bkbk_ast.template sync<DeviceType>();
  k_cutsq_bkbk_c.template sync<DeviceType>();

  k_epsilon_bkbs.template sync<DeviceType>();
  k_sigma_bkbs.template sync<DeviceType>();
  k_cut_bkbs_ast.template sync<DeviceType>();
  k_b_bkbs.template sync<DeviceType>();
  k_cut_bkbs_c.template sync<DeviceType>();
  k_lj1_bkbs.template sync<DeviceType>();
  k_lj2_bkbs.template sync<DeviceType>();
  k_cutsq_bkbs_ast.template sync<DeviceType>();
  k_cutsq_bkbs_c.template sync<DeviceType>();

  k_epsilon_bsbs.template sync<DeviceType>();
  k_sigma_bsbs.template sync<DeviceType>();
  k_cut_bsbs_ast.template sync<DeviceType>();
  k_b_bsbs.template sync<DeviceType>();
  k_cut_bsbs_c.template sync<DeviceType>();
  k_lj1_bsbs.template sync<DeviceType>();
  k_lj2_bsbs.template sync<DeviceType>();
  k_cutsq_bsbs_ast.template sync<DeviceType>();
  k_cutsq_bsbs_c.template sync<DeviceType>();

  k_sigma4_bsbs.template sync<DeviceType>();
  k_cut4_bsbs_ast.template sync<DeviceType>();
  k_cut4sq_bsbs_ast.template sync<DeviceType>();
  k_lj14_bsbs.template sync<DeviceType>();
  k_lj24_bsbs.template sync<DeviceType>();
  k_b4_bsbs.template sync<DeviceType>();
  k_cut4_bsbs_c.template sync<DeviceType>();
  k_cut4sq_bsbs_c.template sync<DeviceType>();

  k_nx.template sync<DeviceType>();
  k_ny.template sync<DeviceType>();
  k_nz.template sync<DeviceType>();

  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK | TORQUE_MASK);

  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  torque = atomKK->k_torque.template view<DeviceType>();
  type = atomKK->k_type.template view<DeviceType>();

  auto avecEllipKK = dynamic_cast<AtomVecEllipsoidKokkos *>(atom->style_match("ellipsoid"));
  bonus = avecEllipKK->k_bonus.template view<DeviceType>();
  ellipsoid = atomKK->k_ellipsoid.template view<DeviceType>();

  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

  atomtype = atomKK->k_type.template view<DeviceType>();
  id5p = atomKK->k_id5p.template view<DeviceType>();
  id3p = atomKK->k_id3p.template view<DeviceType>();

  map_style = atom->map_style;
  if (map_style == Atom::MAP_ARRAY) {
    k_map_array = atomKK->k_map_array;
    k_map_array.template sync<DeviceType>();
  } else if (map_style == Atom::MAP_HASH) {
    k_map_hash = atomKK->k_map_hash;
    k_map_hash.template sync<DeviceType>();
  }

  // get the neighbor list and neighbors used in operator()

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_neighbors = k_list->d_neighbors;
  anum = list->inum;
  d_alist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;

  int need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, \
    Kokkos::Experimental::ScatterDuplicated>(f);
    dup_torque = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, \
    Kokkos::Experimental::ScatterDuplicated>(torque);
  } else {
    ndup_f = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, \
    Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_torque = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, \
    Kokkos::Experimental::ScatterNonDuplicated>(torque);
  }

  copymode = 1;

  // loop over all local atoms, calculation of local reference frame from quaternions
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagPairOxdnaExcvQuatToXYZ>(0,nlocal),*this);
  k_nx.template modify<DeviceType>();
  k_ny.template modify<DeviceType>();
  k_nz.template modify<DeviceType>();
  comm->forward_comm(this);
  k_nx.template sync<LMPHostType>();
  k_ny.template sync<LMPHostType>();
  k_nz.template sync<LMPHostType>();

  // loop over neighbors of my atoms for compute functors

  EV_FLOAT ev;

  if (evflag) {
    if (neighflag == HALF) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALF,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALF,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALF,1,1> >(0,anum),*this,ev);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALF,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALF,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALF,0,1> >(0,anum),*this,ev);
        }
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALFTHREAD,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALFTHREAD,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALFTHREAD,1,1> >(0,anum),*this,ev);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALFTHREAD,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALFTHREAD,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALFTHREAD,0,1> >(0,anum),*this,ev);
        }
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,FULL,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,FULL,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,FULL,1,1> >(0,anum),*this,ev);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,FULL,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,FULL,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,FULL,0,1> >(0,anum),*this,ev);
        }
      }
    }
  } else {
    if (neighflag == HALF) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALF,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALF,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALF,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALF,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALF,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALF,0,0> >(0,anum),*this);
        }
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALFTHREAD,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALFTHREAD,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALFTHREAD,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALFTHREAD,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALFTHREAD,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALFTHREAD,0,0> >(0,anum),*this);
        }
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,FULL,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,FULL,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,FULL,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,FULL,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,FULL,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,FULL,0,0> >(0,anum),*this);
        }
      }
    }
  }

  if (need_dup) {
    Kokkos::Experimental::contribute(f, dup_f);
    Kokkos::Experimental::contribute(torque, dup_torque);
  }

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
    if (need_dup)
      Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f        = decltype(dup_f)();
    dup_torque   = decltype(dup_torque)();
    dup_eatom    = decltype(dup_eatom)();
    dup_vatom    = decltype(dup_vatom)();
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::operator()(TagPairOxdnaExcvQuatToXYZ, const int &in) const
{
  int n = d_alist(in);
  KK_FLOAT qn[4];
  for (int i = 0; i < 4; i++) {
    qn[i] = bonus(ellipsoid(n)).quat[i];
  }
  d_nx(n,0) = qn[0]*qn[0] + qn[1]*qn[1] - qn[2]*qn[2] - qn[3]*qn[3];
  d_nx(n,1) = 2.0 * (qn[1]*qn[2] + qn[0]*qn[3]);
  d_nx(n,2) = 2.0 * (qn[1]*qn[3] - qn[0]*qn[2]);
  d_ny(n,0) = 2.0 * (qn[1]*qn[2] - qn[0]*qn[3]);
  d_ny(n,1) = qn[0]*qn[0] - qn[1]*qn[1] + qn[2]*qn[2] - qn[3]*qn[3];
  d_ny(n,2) = 2.0 * (qn[2]*qn[3] + qn[0]*qn[1]);
  d_nz(n,0) = 2.0 * (qn[1]*qn[3] + qn[0]*qn[2]);
  d_nz(n,1) = 2.0 * (qn[2]*qn[3] - qn[0]*qn[1]);
  d_nz(n,2) = qn[0]*qn[0] - qn[1]*qn[1] - qn[2]*qn[2] + qn[3]*qn[3];
}

template<class DeviceType>
template<int OXDNAFLAG, int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::operator()(TagPairOxdnaExcvCompute<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>, \
  const int &ia, EV_FLOAT &ev) const
{
  // f and torque array are duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();
  auto v_torque = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,\
    decltype(dup_torque),decltype(ndup_torque)>::get(dup_torque,ndup_torque);
  auto a_torque = v_torque.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const int a = d_alist(ia);
  const int atype = type(a);
  // vectors COM-backbone site in lab frame
  KK_FLOAT ra_cs[3], rb_cs[3];
  KK_FLOAT ra_cb[3], rb_cb[3];
  KK_FLOAT rtmp_s[3], rtmp_b[3];

  KK_FLOAT delf[3], delta[3], deltb[3];    // force, torque increment
  KK_FLOAT evdwl, fpair;                   // energy, force
  KK_FLOAT delr_bkbk[3],rsq_bkbk,delr_bkbs[3],rsq_bkbs;
  KK_FLOAT delr_bs[3],rsq_bs,delr_bsbs[3],rsq_bsbs;

  KK_FLOAT ftmp[3],ttmp[3];  // temporary force, torque to reduce excessive dup/atomic updates.
  //                           might remove these and just use delf, delta, deltb directly.
  //                           still to profile and test.

  // vector COM - backbone and base site a
  if (OXDNAFLAG==OXDNA) {
    constexpr KK_FLOAT d_cs=-0.4;
    ra_cs[0] = d_cs*d_nx(a,0);
    ra_cs[1] = d_cs*d_nx(a,1);
    ra_cs[2] = d_cs*d_nx(a,2);
    ra_cb[0] = -ra_cs[0];
    ra_cb[1] = -ra_cs[1];
    ra_cb[2] = -ra_cs[2];
  } else if (OXDNAFLAG==OXDNA2) {
    constexpr KK_FLOAT d_cs_x = -0.34;
    constexpr KK_FLOAT d_cs_y = +0.3408;
    constexpr KK_FLOAT d_cb = +0.4;
    ra_cs[0] = d_cs_x*d_nx(a,0) + d_cs_y*d_ny(a,0);
    ra_cs[1] = d_cs_x*d_nx(a,1) + d_cs_y*d_ny(a,1);
    ra_cs[2] = d_cs_x*d_nx(a,2) + d_cs_y*d_ny(a,2);
    ra_cb[0] = d_cb*d_nx(a,0);
    ra_cb[1] = d_cb*d_nx(a,1);
    ra_cb[2] = d_cb*d_nx(a,2);
  } else if (OXDNAFLAG==OXRNA2) {
    constexpr KK_FLOAT d_cs_x = -0.4;
    constexpr KK_FLOAT d_cs_z = +0.2;
    constexpr KK_FLOAT d_cb = +0.4;
    ra_cs[0] = d_cs_x*d_nx(a,0) + d_cs_z*d_nz(a,0);
    ra_cs[1] = d_cs_x*d_nx(a,1) + d_cs_z*d_nz(a,1);
    ra_cs[2] = d_cs_x*d_nx(a,2) + d_cs_z*d_nz(a,2);
    ra_cb[0] = d_cb*d_nx(a,0);
    ra_cb[1] = d_cb*d_nx(a,1);
    ra_cb[2] = d_cb*d_nx(a,2);
  }

  rtmp_s[0] = x(a,0)+ra_cs[0];
  rtmp_s[1] = x(a,1)+ra_cs[1];
  rtmp_s[2] = x(a,2)+ra_cs[2];
  rtmp_b[0] = x(a,0)+ra_cb[0];
  rtmp_b[1] = x(a,1)+ra_cb[1];
  rtmp_b[2] = x(a,2)+ra_cb[2];
  
  const int bnum = d_numneigh(a);

  ftmp[0] = 0.0;
  ftmp[1] = 0.0;
  ftmp[2] = 0.0;
  ttmp[0] = 0.0;
  ttmp[1] = 0.0;
  ttmp[2] = 0.0;

  for (int ib = 0; ib < bnum; ib++) {

    int b = d_neighbors(a,ib);
    const KK_FLOAT factor_lj = special_lj[sbmask(b)];
    b &= NEIGHMASK;
    const int btype = type(b);

    // vector COM - backbone and base site b
    if (OXDNAFLAG==OXDNA) {
      constexpr KK_FLOAT d_cs=-0.4;
      rb_cs[0] = d_cs*d_nx(b,0);
      rb_cs[1] = d_cs*d_nx(b,1);
      rb_cs[2] = d_cs*d_nx(b,2);
      rb_cb[0] = -rb_cs[0];
      rb_cb[1] = -rb_cs[1];
      rb_cb[2] = -rb_cs[2];
    } else if (OXDNAFLAG==OXDNA2) {
      constexpr KK_FLOAT d_cs_x = -0.34;
      constexpr KK_FLOAT d_cs_y = +0.3408;
      constexpr KK_FLOAT d_cb = +0.4;
      rb_cs[0] = d_cs_x*d_nx(b,0) + d_cs_y*d_ny(b,0);
      rb_cs[1] = d_cs_x*d_nx(b,1) + d_cs_y*d_ny(b,1);
      rb_cs[2] = d_cs_x*d_nx(b,2) + d_cs_y*d_ny(b,2);
      rb_cb[0] = d_cb*d_nx(b,0);
      rb_cb[1] = d_cb*d_nx(b,1);
      rb_cb[2] = d_cb*d_nx(b,2);
    } else if (OXDNAFLAG==OXRNA2) {
      constexpr KK_FLOAT d_cs_x = -0.4;
      constexpr KK_FLOAT d_cs_z = +0.2;
      constexpr KK_FLOAT d_cb = +0.4;
      rb_cs[0] = d_cs_x*d_nx(b,0) + d_cs_z*d_nz(b,0);
      rb_cs[1] = d_cs_x*d_nx(b,1) + d_cs_z*d_nz(b,1);
      rb_cs[2] = d_cs_x*d_nx(b,2) + d_cs_z*d_nz(b,2);
      rb_cb[0] = d_cb*d_nx(b,0);
      rb_cb[1] = d_cb*d_nx(b,1);
      rb_cb[2] = d_cb*d_nx(b,2);
    }

    // vector backbone site b to a
    delr_bkbk[0] = rtmp_s[0] - (x(b,0)+rb_cs[0]);
    delr_bkbk[1] = rtmp_s[1] - (x(b,1)+rb_cs[1]);
    delr_bkbk[2] = rtmp_s[2] - (x(b,2)+rb_cs[2]);
    rsq_bkbk = delr_bkbk[0]*delr_bkbk[0] + delr_bkbk[1]*delr_bkbk[1] + delr_bkbk[2]*delr_bkbk[2];
    // vector base site b to backbone site a
    delr_bkbs[0] = rtmp_s[0] - (x(b,0)+rb_cb[0]);
    delr_bkbs[1] = rtmp_s[1] - (x(b,1)+rb_cb[1]);
    delr_bkbs[2] = rtmp_s[2] - (x(b,2)+rb_cb[2]);
    rsq_bkbs = delr_bkbs[0]*delr_bkbs[0] + delr_bkbs[1]*delr_bkbs[1] + delr_bkbs[2]*delr_bkbs[2];
    // vector backbone site b to base site a
    delr_bs[0] = rtmp_b[0] - (x(b,0)+rb_cs[0]);
    delr_bs[1] = rtmp_b[1] - (x(b,1)+rb_cs[1]);
    delr_bs[2] = rtmp_b[2] - (x(b,2)+rb_cs[2]);
    rsq_bs = delr_bs[0]*delr_bs[0] + delr_bs[1]*delr_bs[1] + delr_bs[2]*delr_bs[2];
    // vector base site b to a
    delr_bsbs[0] = rtmp_b[0] - (x(b,0)+rb_cb[0]);
    delr_bsbs[1] = rtmp_b[1] - (x(b,1)+rb_cb[1]);
    delr_bsbs[2] = rtmp_b[2] - (x(b,2)+rb_cb[2]);
    rsq_bsbs = delr_bsbs[0]*delr_bsbs[0] + delr_bsbs[1]*delr_bsbs[1] + delr_bsbs[2]*delr_bsbs[2];

    // excluded volume interactions:

    // backbone-backbone
    if (rsq_bkbk < d_cutsq_bkbk_c(atype,btype)) {
      // F3 modulation factor, force and energy calculation
      evdwl = F3_KK(rsq_bkbk,d_cutsq_bkbk_ast(atype,btype),d_cut_bkbk_c(atype,btype),d_lj1_bkbk(atype,btype),
                        d_lj2_bkbk(atype,btype),d_epsilon_bkbk(atype,btype),d_b_bkbk(atype,btype),fpair);
      // knock out nearest-neighbor interaction between ss
      fpair *= factor_lj;
      evdwl *= factor_lj;
      // force and torque increment calculation
      delf[0] = fpair * delr_bkbk[0];
      delf[1] = fpair * delr_bkbk[1];
      delf[2] = fpair * delr_bkbk[2];
      delta[0] = ra_cs[1]*delf[2] - ra_cs[2]*delf[1];
      delta[1] = ra_cs[2]*delf[0] - ra_cs[0]*delf[2];
      delta[2] = ra_cs[0]*delf[1] - ra_cs[1]*delf[0];
      ftmp[0] += delf[0];
      ftmp[1] += delf[1];
      ftmp[2] += delf[2];
      ttmp[0] += delta[0];
      ttmp[1] += delta[1];
      ttmp[2] += delta[2];
      /*a_f(a,0) += delf[0];
      a_f(a,1) += delf[1];
      a_f(a,2) += delf[2];
      a_torque(a,0) += delta[0];
      a_torque(a,1) += delta[1];
      a_torque(a,2) += delta[2];*/
      if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || b < nlocal)) {
        a_f(b,0) -= delf[0];
        a_f(b,1) -= delf[1];
        a_f(b,2) -= delf[2];
        deltb[0] = rb_cs[1]*delf[2] - rb_cs[2]*delf[1];
        deltb[1] = rb_cs[2]*delf[0] - rb_cs[0]*delf[2];
        deltb[2] = rb_cs[0]*delf[1] - rb_cs[1]*delf[0];
        a_torque(b,0) -= deltb[0];
        a_torque(b,1) -= deltb[1];
        a_torque(b,2) -= deltb[2];
      }
      if (EVFLAG) {
        if (eflag) {
          ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // backbone-base
    if (rsq_bkbs < d_cutsq_bkbs_c(atype,btype)) {
      // F3 modulation factor, force and energy calculation
      evdwl = F3_KK(rsq_bkbs,d_cutsq_bkbs_ast(atype,btype),d_cut_bkbs_c(atype,btype),d_lj1_bkbs(atype,btype),
                        d_lj2_bkbs(atype,btype),d_epsilon_bkbs(atype,btype),d_b_bkbs(atype,btype),fpair);
      // force and torque increment calculation
      delf[0] = fpair * delr_bkbs[0];
      delf[1] = fpair * delr_bkbs[1];
      delf[2] = fpair * delr_bkbs[2];
      delta[0] = ra_cs[1]*delf[2] - ra_cs[2]*delf[1];
      delta[1] = ra_cs[2]*delf[0] - ra_cs[0]*delf[2];
      delta[2] = ra_cs[0]*delf[1] - ra_cs[1]*delf[0];
      ftmp[0] += delf[0];
      ftmp[1] += delf[1];
      ftmp[2] += delf[2];
      ttmp[0] += delta[0];
      ttmp[1] += delta[1];
      ttmp[2] += delta[2];
      /*a_f(a,0) += delf[0];
      a_f(a,1) += delf[1];
      a_f(a,2) += delf[2];
      a_torque(a,0) += delta[0];
      a_torque(a,1) += delta[1];
      a_torque(a,2) += delta[2];*/
      if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || b < nlocal)) {
        a_f(b,0) -= delf[0];
        a_f(b,1) -= delf[1];
        a_f(b,2) -= delf[2];
        deltb[0] = rb_cb[1]*delf[2] - rb_cb[2]*delf[1];
        deltb[1] = rb_cb[2]*delf[0] - rb_cb[0]*delf[2];
        deltb[2] = rb_cb[0]*delf[1] - rb_cb[1]*delf[0];
        a_torque(b,0) -= deltb[0];
        a_torque(b,1) -= deltb[1];
        a_torque(b,2) -= deltb[2];
      }
      if (EVFLAG) {
        if (eflag) {
          ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // base-backbone
    if (rsq_bs < d_cutsq_bkbs_c(btype,atype)) {
      // F3 modulation factor, force and energy calculation
      evdwl = F3_KK(rsq_bs,d_cutsq_bkbs_ast(btype,atype),d_cut_bkbs_c(btype,atype),d_lj1_bkbs(btype,atype),
                        d_lj2_bkbs(btype,atype),d_epsilon_bkbs(btype,atype),d_b_bkbs(btype,atype),fpair);
      // force and torque increment calculation
      delf[0] = fpair * delr_bs[0];
      delf[1] = fpair * delr_bs[1];
      delf[2] = fpair * delr_bs[2];
      delta[0] = ra_cb[1]*delf[2] - ra_cb[2]*delf[1];
      delta[1] = ra_cb[2]*delf[0] - ra_cb[0]*delf[2];
      delta[2] = ra_cb[0]*delf[1] - ra_cb[1]*delf[0];
      ftmp[0] += delf[0];
      ftmp[1] += delf[1];
      ftmp[2] += delf[2];
      ttmp[0] += delta[0];
      ttmp[1] += delta[1];
      ttmp[2] += delta[2];
      /*a_f(a,0) += delf[0];
      a_f(a,1) += delf[1];
      a_f(a,2) += delf[2];
      a_torque(a,0) += delta[0];
      a_torque(a,1) += delta[1];
      a_torque(a,2) += delta[2];*/
      if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || b < nlocal)) {
        a_f(b,0) -= delf[0];
        a_f(b,1) -= delf[1];
        a_f(b,2) -= delf[2];
        deltb[0] = rb_cs[1]*delf[2] - rb_cs[2]*delf[1];
        deltb[1] = rb_cs[2]*delf[0] - rb_cs[0]*delf[2];
        deltb[2] = rb_cs[0]*delf[1] - rb_cs[1]*delf[0];
        a_torque(b,0) -= deltb[0];
        a_torque(b,1) -= deltb[1];
        a_torque(b,2) -= deltb[2];
      }
      if (EVFLAG) {
        if (eflag) {
          ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // base-base
    if (rsq_bsbs < d_cutsq_bsbs_c(atype,btype)) {
      // F3 modulation factor, force and energy calculation
      evdwl = F3_KK(rsq_bsbs,d_cutsq_bsbs_ast(atype,btype),d_cut_bsbs_c(atype,btype),d_lj1_bsbs(atype,btype),
                        d_lj2_bsbs(atype,btype),d_epsilon_bsbs(atype,btype),d_b_bsbs(atype,btype),fpair);
      // force and torque increment calculation
      delf[0] = fpair * delr_bsbs[0];
      delf[1] = fpair * delr_bsbs[1];
      delf[2] = fpair * delr_bsbs[2];
      delta[0] = ra_cb[1]*delf[2] - ra_cb[2]*delf[1];
      delta[1] = ra_cb[2]*delf[0] - ra_cb[0]*delf[2];
      delta[2] = ra_cb[0]*delf[1] - ra_cb[1]*delf[0];
      ftmp[0] += delf[0];
      ftmp[1] += delf[1];
      ftmp[2] += delf[2];
      ttmp[0] += delta[0];
      ttmp[1] += delta[1];
      ttmp[2] += delta[2];
      /*a_f(a,0) += delf[0];
      a_f(a,1) += delf[1];
      a_f(a,2) += delf[2];
      a_torque(a,0) += delta[0];
      a_torque(a,1) += delta[1];
      a_torque(a,2) += delta[2];*/
      if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || b < nlocal)) {
        a_f(b,0) -= delf[0];
        a_f(b,1) -= delf[1];
        a_f(b,2) -= delf[2];
        deltb[0] = rb_cb[1]*delf[2] - rb_cb[2]*delf[1];
        deltb[1] = rb_cb[2]*delf[0] - rb_cb[0]*delf[2];
        deltb[2] = rb_cb[0]*delf[1] - rb_cb[1]*delf[0];
        a_torque(b,0) -= deltb[0];
        a_torque(b,1) -= deltb[1];
        a_torque(b,2) -= deltb[2];
      }
      if (EVFLAG) {
        if (eflag) {
          ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }
    // end excluded volume interaction
  }
  a_f(a,0) += ftmp[0];
  a_f(a,1) += ftmp[1];
  a_f(a,2) += ftmp[2];
  a_torque(a,0) += ttmp[0];
  a_torque(a,1) += ttmp[1];
  a_torque(a,2) += ttmp[2];
}

template<class DeviceType>
template<int OXDNAFLAG, int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::operator()(TagPairOxdnaExcvCompute<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>, \
  const int &ia) const
{
  EV_FLOAT ev;
  this->template operator()<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>\
  (TagPairOxdnaExcvCompute<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>(),ia,ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairOxdnaExcvKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                        DAT::tdual_double_1d &buf,
                                                        int /*pbc_flag*/, int * /*pbc*/)
{
  d_sendlist = k_sendlist.template view<DeviceType>();
  v_buf = buf.template view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvPackForwardComm>(0,n),*this);
  return n*9;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::operator()(TagPairOxdnaExcvPackForwardComm, const int &i) const {
  int j = d_sendlist(i);
  v_buf[i*9] = d_nx(j,0);
  v_buf[i*9+1] = d_nx(j,1);
  v_buf[i*9+2] = d_nx(j,2);
  v_buf[i*9+3] = d_ny(j,0);
  v_buf[i*9+4] = d_ny(j,1);
  v_buf[i*9+5] = d_ny(j,2);
  v_buf[i*9+6] = d_nz(j,0);
  v_buf[i*9+7] = d_nz(j,1);
  v_buf[i*9+8] = d_nz(j,2);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaExcvKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_double_1d &buf)
{
  first = first_in;
  v_buf = buf.template view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvUnpackForwardComm>(0,n),*this);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::operator()(TagPairOxdnaExcvUnpackForwardComm, const int &i) const {
  d_nx(i+first,0) = v_buf[i*9];
  d_nx(i+first,1) = v_buf[i*9+1];
  d_nx(i+first,2) = v_buf[i*9+2];
  d_ny(i+first,0) = v_buf[i*9+3];
  d_ny(i+first,1) = v_buf[i*9+4];
  d_ny(i+first,2) = v_buf[i*9+5];
  d_nz(i+first,0) = v_buf[i*9+6];
  d_nz(i+first,1) = v_buf[i*9+7];
  d_nz(i+first,2) = v_buf[i*9+8];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairOxdnaExcvKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
                                                 int /*pbc_flag*/, int * /*pbc*/)
{
  k_nx.sync_host();
  k_ny.sync_host();
  k_nz.sync_host();

  int i,j,m;
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = h_nx(j,0);
    buf[m++] = h_nx(j,1);
    buf[m++] = h_nx(j,2);
    buf[m++] = h_ny(j,0);
    buf[m++] = h_ny(j,1);
    buf[m++] = h_ny(j,2);
    buf[m++] = h_nz(j,0);
    buf[m++] = h_nz(j,1);
    buf[m++] = h_nz(j,2);
  }
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaExcvKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  k_nx.sync_host();
  k_ny.sync_host();
  k_nz.sync_host();

  int m = 0;
  for (int i = 0; i < n; i++) {
    h_nx(i+first,0) = buf[m++];
    h_nx(i+first,1) = buf[m++];
    h_nx(i+first,2) = buf[m++];
    h_ny(i+first,0) = buf[m++];
    h_ny(i+first,1) = buf[m++];
    h_ny(i+first,2) = buf[m++];
    h_nz(i+first,0) = buf[m++];
    h_nz(i+first,1) = buf[m++];
    h_nz(i+first,2) = buf[m++];
  }

  k_nx.modify_host();
  k_ny.modify_host();
  k_nz.modify_host();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void *PairOxdnaExcvKokkos<DeviceType>::extract(const char *str, int &dim)
{
  PairOxdnaExcv::extract(str,dim);

  if (strcmp(str,"d_nx") == 0) return (void *) d_nx.data();
  if (strcmp(str,"d_ny") == 0) return (void *) d_ny.data();
  if (strcmp(str,"d_nz") == 0) return (void *) d_nz.data();


  return nullptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaExcvKokkos<DeviceType>::allocate()
{
  PairOxdnaExcv::allocate();

  int n = atom->ntypes;

  memoryKK->create_kokkos(k_epsilon_bkbk,n+1,n+1,"PairOxdnaExcv:epsilon_bkbk");
  memoryKK->create_kokkos(k_sigma_bkbk,n+1,n+1,"PairOxdnaExcv:sigma_bkbk");
  memoryKK->create_kokkos(k_cut_bkbk_ast,n+1,n+1,"PairOxdnaExcv:cut_bkbk_ast");
  memoryKK->create_kokkos(k_b_bkbk,n+1,n+1,"PairOxdnaExcv:b_bkbk");
  memoryKK->create_kokkos(k_cut_bkbk_c,n+1,n+1,"PairOxdnaExcv:cut_bkbk_c");
  memoryKK->create_kokkos(k_lj1_bkbk,n+1,n+1,"PairOxdnaExcv:lj1_bkbk");
  memoryKK->create_kokkos(k_lj2_bkbk,n+1,n+1,"PairOxdnaExcv:lj2_bkbk");
  memoryKK->create_kokkos(k_cutsq_bkbk_ast,n+1,n+1,"PairOxdnaExcv:cutsq_bkbk_ast");
  memoryKK->create_kokkos(k_cutsq_bkbk_c,n+1,n+1,"PairOxdnaExcv:cutsq_bkbk_c");

  memoryKK->create_kokkos(k_epsilon_bkbs,n+1,n+1,"PairOxdnaExcv:epsilon_bkbs");
  memoryKK->create_kokkos(k_sigma_bkbs,n+1,n+1,"PairOxdnaExcv:sigma_bkbs");
  memoryKK->create_kokkos(k_cut_bkbs_ast,n+1,n+1,"PairOxdnaExcv:cut_bkbs_ast");
  memoryKK->create_kokkos(k_b_bkbs,n+1,n+1,"PairOxdnaExcv:b_bkbs");
  memoryKK->create_kokkos(k_cut_bkbs_c,n+1,n+1,"PairOxdnaExcv:cut_bkbs_c");
  memoryKK->create_kokkos(k_lj1_bkbs,n+1,n+1,"PairOxdnaExcv:lj1_bkbs");
  memoryKK->create_kokkos(k_lj2_bkbs,n+1,n+1,"PairOxdnaExcv:lj2_bkbs");
  memoryKK->create_kokkos(k_cutsq_bkbs_ast,n+1,n+1,"PairOxdnaExcv:cutsq_bkbs_ast");
  memoryKK->create_kokkos(k_cutsq_bkbs_c,n+1,n+1,"PairOxdnaExcv:cutsq_bkbs_c");

  memoryKK->create_kokkos(k_epsilon_bsbs,n+1,n+1,"PairOxdnaExcv:epsilon_bsbs");
  memoryKK->create_kokkos(k_sigma_bsbs,n+1,n+1,"PairOxdnaExcv:sigma_bsbs");
  memoryKK->create_kokkos(k_cut_bsbs_ast,n+1,n+1,"PairOxdnaExcv:cut_bsbs_ast");
  memoryKK->create_kokkos(k_b_bsbs,n+1,n+1,"PairOxdnaExcv:b_bsbs");
  memoryKK->create_kokkos(k_cut_bsbs_c,n+1,n+1,"PairOxdnaExcv:cut_bsbs_c");
  memoryKK->create_kokkos(k_lj1_bsbs,n+1,n+1,"PairOxdnaExcv:lj1_bsbs");
  memoryKK->create_kokkos(k_lj2_bsbs,n+1,n+1,"PairOxdnaExcv:lj2_bsbs");
  memoryKK->create_kokkos(k_cutsq_bsbs_ast,n+1,n+1,"PairOxdnaExcv:cutsq_bsbs_ast");
  memoryKK->create_kokkos(k_cutsq_bsbs_c,n+1,n+1,"PairOxdnaExcv:cutsq_bsbs_c");

  memoryKK->create_kokkos(k_sigma4_bsbs,n+1,n+1,n+1,n+1,"PairOxdnaExcv:sigma4_bsbs");
  memoryKK->create_kokkos(k_cut4_bsbs_ast,n+1,n+1,n+1,n+1,"PairOxdnaExcv:cut4_bsbs_ast");
  memoryKK->create_kokkos(k_cut4sq_bsbs_ast,n+1,n+1,n+1,n+1,"PairOxdnaExcv:cut4sq_bsbs_ast");
  memoryKK->create_kokkos(k_lj14_bsbs,n+1,n+1,n+1,n+1,"PairOxdnaExcv:lj14_bsbs");
  memoryKK->create_kokkos(k_lj24_bsbs,n+1,n+1,n+1,n+1,"PairOxdnaExcv:lj24_bsbs");
  memoryKK->create_kokkos(k_b4_bsbs,n+1,n+1,n+1,n+1,"PairOxdnaExcv:b4_bsbs");
  memoryKK->create_kokkos(k_cut4_bsbs_c,n+1,n+1,n+1,n+1,"PairOxdnaExcv:cut4_bsbs_c");
  memoryKK->create_kokkos(k_cut4sq_bsbs_c,n+1,n+1,n+1,n+1,"PairOxdnaExcv:cut4sq_bsbs_c");

  memoryKK->create_kokkos(k_nx,atom->nmax,3,"PairOxdnaExcv:nx");
  memoryKK->create_kokkos(k_ny,atom->nmax,3,"PairOxdnaExcv:ny");
  memoryKK->create_kokkos(k_nz,atom->nmax,3,"PairOxdnaExcv:nz");

  d_epsilon_bkbk = k_epsilon_bkbk.template view<DeviceType>();
  d_sigma_bkbk = k_sigma_bkbk.template view<DeviceType>();
  d_cut_bkbk_ast = k_cut_bkbk_ast.template view<DeviceType>();
  d_b_bkbk = k_b_bkbk.template view<DeviceType>();
  d_cut_bkbk_c = k_cut_bkbk_c.template view<DeviceType>();
  d_lj1_bkbk = k_lj1_bkbk.template view<DeviceType>();
  d_lj2_bkbk = k_lj2_bkbk.template view<DeviceType>();
  d_cutsq_bkbk_ast = k_cutsq_bkbk_ast.template view<DeviceType>();
  d_cutsq_bkbk_c = k_cutsq_bkbk_c.template view<DeviceType>();

  d_epsilon_bkbs = k_epsilon_bkbs.template view<DeviceType>();
  d_sigma_bkbs = k_sigma_bkbs.template view<DeviceType>();
  d_cut_bkbs_ast = k_cut_bkbs_ast.template view<DeviceType>();
  d_b_bkbs = k_b_bkbs.template view<DeviceType>();
  d_cut_bkbs_c = k_cut_bkbs_c.template view<DeviceType>();
  d_lj1_bkbs = k_lj1_bkbs.template view<DeviceType>();
  d_lj2_bkbs = k_lj2_bkbs.template view<DeviceType>();
  d_cutsq_bkbs_ast = k_cutsq_bkbs_ast.template view<DeviceType>();
  d_cutsq_bkbs_c = k_cutsq_bkbs_c.template view<DeviceType>();

  d_epsilon_bsbs = k_epsilon_bsbs.template view<DeviceType>();
  d_sigma_bsbs = k_sigma_bsbs.template view<DeviceType>();
  d_cut_bsbs_ast = k_cut_bsbs_ast.template view<DeviceType>();
  d_b_bsbs = k_b_bsbs.template view<DeviceType>();
  d_cut_bsbs_c = k_cut_bsbs_c.template view<DeviceType>();
  d_lj1_bsbs = k_lj1_bsbs.template view<DeviceType>();
  d_lj2_bsbs = k_lj2_bsbs.template view<DeviceType>();
  d_cutsq_bsbs_ast = k_cutsq_bsbs_ast.template view<DeviceType>();
  d_cutsq_bsbs_c = k_cutsq_bsbs_c.template view<DeviceType>();

  d_sigma4_bsbs = k_sigma4_bsbs.template view<DeviceType>();
  d_cut4_bsbs_ast = k_cut4_bsbs_ast.template view<DeviceType>();
  d_cut4sq_bsbs_ast = k_cut4sq_bsbs_ast.template view<DeviceType>();
  d_lj14_bsbs = k_lj14_bsbs.template view<DeviceType>();
  d_lj24_bsbs = k_lj24_bsbs.template view<DeviceType>();
  d_b4_bsbs = k_b4_bsbs.template view<DeviceType>();
  d_cut4_bsbs_c = k_cut4_bsbs_c.template view<DeviceType>();
  d_cut4sq_bsbs_c = k_cut4sq_bsbs_c.template view<DeviceType>();

  d_nx = k_nx.template view<DeviceType>();
  d_ny = k_ny.template view<DeviceType>();
  d_nz = k_nz.template view<DeviceType>();
  h_nx = k_nx.view_host();
  h_ny = k_ny.view_host();
  h_nz = k_nz.view_host();

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaExcvKokkos<DeviceType>::settings(int narg, char **/*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaExcvKokkos<DeviceType>::init_style() 
{
  neighbor->add_request(this);
  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double PairOxdnaExcvKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairOxdnaExcv::init_one(i,j);
  
  // All non-tetramer Kokkos views are set here within ::init_one, and
  // the tetramer Kokkos views are set within ::coeff

  // Assign directionally: [i][j] gets [i][j], [j][i] gets [j][i]
  k_epsilon_bkbk.view_host()(i,j) = epsilon_bkbk[i][j];
  k_epsilon_bkbk.view_host()(j,i) = epsilon_bkbk[j][i];
  k_sigma_bkbk.view_host()(i,j) = sigma_bkbk[i][j];
  k_sigma_bkbk.view_host()(j,i) = sigma_bkbk[j][i];
  k_cut_bkbk_ast.view_host()(i,j) = cut_bkbk_ast[i][j];
  k_cut_bkbk_ast.view_host()(j,i) = cut_bkbk_ast[j][i];
  k_b_bkbk.view_host()(i,j) = b_bkbk[i][j];
  k_b_bkbk.view_host()(j,i) = b_bkbk[j][i];
  k_cut_bkbk_c.view_host()(i,j) = cut_bkbk_c[i][j];
  k_cut_bkbk_c.view_host()(j,i) = cut_bkbk_c[j][i];
  k_lj1_bkbk.view_host()(i,j) = lj1_bkbk[i][j];
  k_lj1_bkbk.view_host()(j,i) = lj1_bkbk[j][i];
  k_lj2_bkbk.view_host()(i,j) = lj2_bkbk[i][j];
  k_lj2_bkbk.view_host()(j,i) = lj2_bkbk[j][i];
  k_cutsq_bkbk_ast.view_host()(i,j) = cutsq_bkbk_ast[i][j];
  k_cutsq_bkbk_ast.view_host()(j,i) = cutsq_bkbk_ast[j][i];
  k_cutsq_bkbk_c.view_host()(i,j) = cutsq_bkbk_c[i][j];
  k_cutsq_bkbk_c.view_host()(j,i) = cutsq_bkbk_c[j][i];

  k_epsilon_bkbs.view_host()(i,j) = epsilon_bkbs[i][j];
  k_epsilon_bkbs.view_host()(j,i) = epsilon_bkbs[j][i];
  k_sigma_bkbs.view_host()(i,j) = sigma_bkbs[i][j];
  k_sigma_bkbs.view_host()(j,i) = sigma_bkbs[j][i];
  k_cut_bkbs_ast.view_host()(i,j) = cut_bkbs_ast[i][j];
  k_cut_bkbs_ast.view_host()(j,i) = cut_bkbs_ast[j][i];
  k_b_bkbs.view_host()(i,j) = b_bkbs[i][j];
  k_b_bkbs.view_host()(j,i) = b_bkbs[j][i];
  k_cut_bkbs_c.view_host()(i,j) = cut_bkbs_c[i][j];
  k_cut_bkbs_c.view_host()(j,i) = cut_bkbs_c[j][i];
  k_lj1_bkbs.view_host()(i,j) = lj1_bkbs[i][j];
  k_lj1_bkbs.view_host()(j,i) = lj1_bkbs[j][i];
  k_lj2_bkbs.view_host()(i,j) = lj2_bkbs[i][j];
  k_lj2_bkbs.view_host()(j,i) = lj2_bkbs[j][i];
  k_cutsq_bkbs_ast.view_host()(i,j) = cutsq_bkbs_ast[i][j];
  k_cutsq_bkbs_ast.view_host()(j,i) = cutsq_bkbs_ast[j][i];
  k_cutsq_bkbs_c.view_host()(i,j) = cutsq_bkbs_c[i][j];
  k_cutsq_bkbs_c.view_host()(j,i) = cutsq_bkbs_c[j][i];

  k_epsilon_bsbs.view_host()(i,j) = epsilon_bsbs[i][j];
  k_epsilon_bsbs.view_host()(j,i) = epsilon_bsbs[j][i];
  k_sigma_bsbs.view_host()(i,j) = sigma_bsbs[i][j];
  k_sigma_bsbs.view_host()(j,i) = sigma_bsbs[j][i];
  k_cut_bsbs_ast.view_host()(i,j) = cut_bsbs_ast[i][j];
  k_cut_bsbs_ast.view_host()(j,i) = cut_bsbs_ast[j][i];
  k_b_bsbs.view_host()(i,j) = b_bsbs[i][j];
  k_b_bsbs.view_host()(j,i) = b_bsbs[j][i];
  k_cut_bsbs_c.view_host()(i,j) = cut_bsbs_c[i][j];
  k_cut_bsbs_c.view_host()(j,i) = cut_bsbs_c[j][i];
  k_lj1_bsbs.view_host()(i,j) = lj1_bsbs[i][j];
  k_lj1_bsbs.view_host()(j,i) = lj1_bsbs[j][i];
  k_lj2_bsbs.view_host()(i,j) = lj2_bsbs[i][j];
  k_lj2_bsbs.view_host()(j,i) = lj2_bsbs[j][i];
  k_cutsq_bsbs_ast.view_host()(i,j) = cutsq_bsbs_ast[i][j];
  k_cutsq_bsbs_ast.view_host()(j,i) = cutsq_bsbs_ast[j][i];
  k_cutsq_bsbs_c.view_host()(i,j) = cutsq_bsbs_c[i][j];
  k_cutsq_bsbs_c.view_host()(j,i) = cutsq_bsbs_c[j][i];

  k_epsilon_bkbk.template modify<LMPHostType>();
  k_sigma_bkbk.template modify<LMPHostType>();
  k_cut_bkbk_ast.template modify<LMPHostType>();
  k_b_bkbk.template modify<LMPHostType>();
  k_cut_bkbk_c.template modify<LMPHostType>();
  k_lj1_bkbk.template modify<LMPHostType>();
  k_lj2_bkbk.template modify<LMPHostType>();
  k_cutsq_bkbk_ast.template modify<LMPHostType>();
  k_cutsq_bkbk_c.template modify<LMPHostType>();

  k_epsilon_bkbs.template modify<LMPHostType>();
  k_sigma_bkbs.template modify<LMPHostType>();
  k_cut_bkbs_ast.template modify<LMPHostType>();
  k_b_bkbs.template modify<LMPHostType>();
  k_cut_bkbs_c.template modify<LMPHostType>();
  k_lj1_bkbs.template modify<LMPHostType>();
  k_lj2_bkbs.template modify<LMPHostType>();
  k_cutsq_bkbs_ast.template modify<LMPHostType>();
  k_cutsq_bkbs_c.template modify<LMPHostType>();

  k_epsilon_bsbs.template modify<LMPHostType>();
  k_sigma_bsbs.template modify<LMPHostType>();
  k_cut_bsbs_ast.template modify<LMPHostType>();
  k_b_bsbs.template modify<LMPHostType>();
  k_cut_bsbs_c.template modify<LMPHostType>();
  k_lj1_bsbs.template modify<LMPHostType>();
  k_lj2_bsbs.template modify<LMPHostType>();
  k_cutsq_bsbs_ast.template modify<LMPHostType>();
  k_cutsq_bsbs_c.template modify<LMPHostType>();

  k_sigma4_bsbs.template modify<LMPHostType>();
  k_cut4_bsbs_ast.template modify<LMPHostType>();
  k_cut4sq_bsbs_ast.template modify<LMPHostType>();
  k_lj14_bsbs.template modify<LMPHostType>();
  k_lj24_bsbs.template modify<LMPHostType>();
  k_b4_bsbs.template modify<LMPHostType>();
  k_cut4_bsbs_c.template modify<LMPHostType>();
  k_cut4sq_bsbs_c.template modify<LMPHostType>();

  // "cutone" is "cut_bkbk_c[i][j]", sets the master list distance cutoff
  return cutone;

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaExcvKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairOxdnaExcv::coeff(narg,arg);

  // The tetramer Kokkos views are set here within ::coeff, and the
  // non-tetramer Kokkos views are set within ::init_one

  int ilo,ihi,jlo,jhi,nlo,nhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  assert((ilo == jlo) & (ihi == jhi));
  nlo = ilo;
  nhi = ihi;

  for (int i = 0; i <= nhi; i++) { // type 0 for terminal j
    for (int j = nlo; j <= nhi; j++) {
      for (int k = nlo; k <= nhi; k++) {
        for (int l = 0; l <= nhi; l++) { // type 0 for terminal k
          k_sigma4_bsbs.view_host()(i,j,k,l) = sigma4_bsbs[i][j][k][l];
          k_cut4_bsbs_ast.view_host()(i,j,k,l) = cut4_bsbs_ast[i][j][k][l];
          k_cut4sq_bsbs_ast.view_host()(i,j,k,l) = cut4sq_bsbs_ast[i][j][k][l];
          k_lj14_bsbs.view_host()(i,j,k,l) = lj14_bsbs[i][j][k][l];
          k_lj24_bsbs.view_host()(i,j,k,l) = lj24_bsbs[i][j][k][l];
          k_b4_bsbs.view_host()(i,j,k,l) = b4_bsbs[i][j][k][l];
          k_cut4_bsbs_c.view_host()(i,j,k,l) = cut4_bsbs_c[i][j][k][l];
          k_cut4sq_bsbs_c.view_host()(i,j,k,l) = cut4sq_bsbs_c[i][j][k][l];
        }
      }
    }
  }

  k_sigma4_bsbs.template modify<LMPHostType>();
  k_cut4_bsbs_ast.template modify<LMPHostType>();
  k_cut4sq_bsbs_ast.template modify<LMPHostType>();
  k_lj14_bsbs.template modify<LMPHostType>();
  k_lj24_bsbs.template modify<LMPHostType>();
  k_b4_bsbs.template modify<LMPHostType>();
  k_cut4_bsbs_c.template modify<LMPHostType>();
  k_cut4sq_bsbs_c.template modify<LMPHostType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const KK_FLOAT &epair, const KK_FLOAT &fx, const KK_FLOAT &fy, const KK_FLOAT &fz, const KK_FLOAT &delx,
                const KK_FLOAT &dely, const KK_FLOAT &delz) const
{
  const int EFLAG = eflag;
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_eatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,\
    decltype(dup_eatom),decltype(ndup_eatom)>::get(dup_eatom,ndup_eatom);
  auto a_eatom = v_eatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,\
    decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  if (EFLAG) {
    if (eflag_atom) {
      const KK_FLOAT epairhalf = 0.5 * epair;
      if (NEIGHFLAG!=FULL) {
        if (NEWTON_PAIR || i < nlocal) a_eatom[i] += epairhalf;
        if (NEWTON_PAIR || j < nlocal) a_eatom[j] += epairhalf;
      } else {
        a_eatom[i] += epairhalf;
      }
    }
  }

  if (VFLAG) {
    const KK_FLOAT v0 = delx*fx;
    const KK_FLOAT v1 = dely*fy;
    const KK_FLOAT v2 = delz*fz;
    const KK_FLOAT v3 = delx*fy;
    const KK_FLOAT v4 = delx*fz;
    const KK_FLOAT v5 = dely*fz;

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
          a_vatom(i,0) += 0.5*v0;
          a_vatom(i,1) += 0.5*v1;
          a_vatom(i,2) += 0.5*v2;
          a_vatom(i,3) += 0.5*v3;
          a_vatom(i,4) += 0.5*v4;
          a_vatom(i,5) += 0.5*v5;
        }
        if (NEWTON_PAIR || j < nlocal) {
        a_vatom(j,0) += 0.5*v0;
        a_vatom(j,1) += 0.5*v1;
        a_vatom(j,2) += 0.5*v2;
        a_vatom(j,3) += 0.5*v3;
        a_vatom(j,4) += 0.5*v4;
        a_vatom(j,5) += 0.5*v5;
        }
      } else {
        a_vatom(i,0) += 0.5*v0;
        a_vatom(i,1) += 0.5*v1;
        a_vatom(i,2) += 0.5*v2;
        a_vatom(i,3) += 0.5*v3;
        a_vatom(i,4) += 0.5*v4;
        a_vatom(i,5) += 0.5*v5;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int PairOxdnaExcvKokkos<DeviceType>::sbmask(const int& j) const {
  return j >> SBBITS & 3;
}

namespace LAMMPS_NS {
template class PairOxdnaExcvKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairOxdnaExcvKokkos<LMPHostType>;
#endif
}