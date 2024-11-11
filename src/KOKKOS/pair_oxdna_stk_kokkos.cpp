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

#include "pair_oxdna_stk_kokkos.h"

#include "atom_kokkos.h"
//#include "atom_vec_ellipsoid_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairOxdnaStkKokkos<DeviceType>::PairOxdnaStkKokkos(LAMMPS *lmp) : PairOxdnaStk(lmp)
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
PairOxdnaStkKokkos<DeviceType>::~PairOxdnaStkKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);

    memoryKK->destroy_kokkos(k_epsilon_ss,epsilon_ss);
    memoryKK->destroy_kokkos(k_sigma_ss,sigma_ss);
    memoryKK->destroy_kokkos(k_cut_ss_ast,cut_ss_ast);
    memoryKK->destroy_kokkos(k_b_ss,b_ss);
    memoryKK->destroy_kokkos(k_cut_ss_c,cut_ss_c);
    memoryKK->destroy_kokkos(k_lj1_ss,lj1_ss);
    memoryKK->destroy_kokkos(k_lj2_ss,lj2_ss);
    memoryKK->destroy_kokkos(k_cutsq_ss_ast,cutsq_ss_ast);
    memoryKK->destroy_kokkos(k_cutsq_ss_c,cutsq_ss_c);

    memoryKK->destroy_kokkos(k_epsilon_sb,epsilon_sb);
    memoryKK->destroy_kokkos(k_sigma_sb,sigma_sb);
    memoryKK->destroy_kokkos(k_cut_sb_ast,cut_sb_ast);
    memoryKK->destroy_kokkos(k_b_sb,b_sb);
    memoryKK->destroy_kokkos(k_cut_sb_c,cut_sb_c);
    memoryKK->destroy_kokkos(k_lj1_sb,lj1_sb);
    memoryKK->destroy_kokkos(k_lj2_sb,lj2_sb);
    memoryKK->destroy_kokkos(k_cutsq_sb_ast,cutsq_sb_ast);
    memoryKK->destroy_kokkos(k_cutsq_sb_c,cutsq_sb_c);

    memoryKK->destroy_kokkos(k_epsilon_bb,epsilon_bb);
    memoryKK->destroy_kokkos(k_sigma_bb,sigma_bb);
    memoryKK->destroy_kokkos(k_cut_bb_ast,cut_bb_ast);
    memoryKK->destroy_kokkos(k_b_bb,b_bb);
    memoryKK->destroy_kokkos(k_cut_bb_c,cut_bb_c);
    memoryKK->destroy_kokkos(k_lj1_bb,lj1_bb);
    memoryKK->destroy_kokkos(k_lj2_bb,lj2_bb);
    memoryKK->destroy_kokkos(k_cutsq_bb_ast,cutsq_bb_ast);
    memoryKK->destroy_kokkos(k_cutsq_bb_c,cutsq_bb_c);

    memoryKK->destroy_kokkos(k_nx,nx);
    memoryKK->destroy_kokkos(k_ny,ny);
    memoryKK->destroy_kokkos(k_nz,nz);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaStkKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  //printf("neighflag, newton_pair, evflag : %d %d %d\n",neighflag,newton_pair,evflag);

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

  atomKK->sync(execution_space,datamask_read); //need or not need? same for fene

  k_epsilon_ss.template sync<DeviceType>();
  k_sigma_ss.template sync<DeviceType>();
  k_cut_ss_ast.template sync<DeviceType>();
  k_b_ss.template sync<DeviceType>();
  k_cut_ss_c.template sync<DeviceType>();
  k_lj1_ss.template sync<DeviceType>();
  k_lj2_ss.template sync<DeviceType>();
  k_cutsq_ss_ast.template sync<DeviceType>();
  k_cutsq_ss_c.template sync<DeviceType>();

  k_epsilon_sb.template sync<DeviceType>();
  k_sigma_sb.template sync<DeviceType>();
  k_cut_sb_ast.template sync<DeviceType>();
  k_b_sb.template sync<DeviceType>();
  k_cut_sb_c.template sync<DeviceType>();
  k_lj1_sb.template sync<DeviceType>();
  k_lj2_sb.template sync<DeviceType>();
  k_cutsq_sb_ast.template sync<DeviceType>();
  k_cutsq_sb_c.template sync<DeviceType>();

  k_epsilon_bb.template sync<DeviceType>();
  k_sigma_bb.template sync<DeviceType>();
  k_cut_bb_ast.template sync<DeviceType>();
  k_b_bb.template sync<DeviceType>();
  k_cut_bb_c.template sync<DeviceType>();
  k_lj1_bb.template sync<DeviceType>();
  k_lj2_bb.template sync<DeviceType>();
  k_cutsq_bb_ast.template sync<DeviceType>();
  k_cutsq_bb_c.template sync<DeviceType>();

  k_nx.template sync<DeviceType>();
  k_ny.template sync<DeviceType>();
  k_nz.template sync<DeviceType>();

  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK | TORQUE_MASK); // TODO: need or not need? same for fene, also add TORQUE_MASK later

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  auto avecEllipKK = dynamic_cast<AtomVecEllipsoidKokkos *>(atom->style_match("ellipsoid")); // TODO: check if this is correct, may ask Stan at some point
  bonus = avecEllipKK->k_bonus.view<DeviceType>();
  ellipsoid = atomKK->k_ellipsoid.view<DeviceType>();

  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

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
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagPairOxdnaStkQuatToXYZ>(0,nlocal),*this);
  k_nx.template modify<DeviceType>();
  k_ny.template modify<DeviceType>();
  k_nz.template modify<DeviceType>();
  comm->forward_comm(this);
  k_nx.template sync<DeviceType>();
  k_ny.template sync<DeviceType>();
  k_nz.template sync<DeviceType>();

  // loop over neighbors of my atoms for compute functors

  EV_FLOAT ev;

  if (evflag) {
    if (neighflag == HALF) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,HALF,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,HALF,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,HALF,1,1> >(0,anum),*this,ev);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,HALF,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,HALF,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,HALF,0,1> >(0,anum),*this,ev);
        }
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,HALFTHREAD,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,HALFTHREAD,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,HALFTHREAD,1,1> >(0,anum),*this,ev);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,HALFTHREAD,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,HALFTHREAD,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,HALFTHREAD,0,1> >(0,anum),*this,ev);
        }
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,FULL,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,FULL,1,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,FULL,1,1> >(0,anum),*this,ev);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,FULL,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,FULL,0,1> >(0,anum),*this,ev);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,FULL,0,1> >(0,anum),*this,ev);
        }
      }
    }
  } else {
    if (neighflag == HALF) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,HALF,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,HALF,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,HALF,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,HALF,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,HALF,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,HALF,0,0> >(0,anum),*this);
        }
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,HALFTHREAD,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,HALFTHREAD,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,HALFTHREAD,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,HALFTHREAD,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,HALFTHREAD,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,HALFTHREAD,0,0> >(0,anum),*this);
        }
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,FULL,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,FULL,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,FULL,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag==OXDNA) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA,FULL,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXDNA2,FULL,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<OXRNA2,FULL,0,0> >(0,anum),*this);
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
void PairOxdnaStkKokkos<DeviceType>::operator()(TagPairOxdnaStkQuatToXYZ, const int &in) const
{
  int n = d_alist(in);
  // TODO: confirm in testing this implementation of quaternion to Cartesian unit vectors in lab frame actually works
  F_FLOAT qn[4];
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

  /*d_nx(n,0) = 0.0;
  d_nx(n,1) = 0.0;
  d_nx(n,2) = 0.0;
  d_ny(n,0) = 0.0;
  d_ny(n,1) = 0.0;
  d_ny(n,2) = 0.0;
  d_nz(n,0) = 0.0;
  d_nz(n,1) = 0.0;
  d_nz(n,2) = 0.0;*/
}

template<class DeviceType>
template<int OXDNAFLAG, int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaStkKokkos<DeviceType>::operator()(TagPairOxdnaStkCompute<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>, \
  const int &ia, EV_FLOAT &ev) const
{
  //TODO: figure out evdwl in context of ev_tally_xyz and ev.evdwl

  // f and torque array are duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();
  auto v_torque = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,\
    decltype(dup_torque),decltype(ndup_torque)>::get(dup_torque,ndup_torque);
  auto a_torque = v_torque.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  //printf("NEIGHFLAG, NEWTON_PAIR, EVFLAG: %d %d %d\n",NEIGHFLAG,NEWTON_PAIR,EVFLAG);
  //printf("vflag_either,eflag_atom: %d %d\n",vflag_either,eflag_atom);

  const int a = d_alist(ia);
  const int atype = type(a);
  // vectors COM-backbone site in lab frame
  F_FLOAT ra_cs[3], rb_cs[3];
  F_FLOAT ra_cb[3], rb_cb[3];
  F_FLOAT rtmp_s[3], rtmp_b[3];

  F_FLOAT delf[3], delta[3], deltb[3];    // force, torque increment
  F_FLOAT evdwl, fpair;                   // energy, force
  F_FLOAT delr_ss[3],rsq_ss,delr_sb[3],rsq_sb;
  F_FLOAT delr_bs[3],rsq_bs,delr_bb[3],rsq_bb;

  F_FLOAT ftmp[3],ttmp[3];  // temporary force, torque to reduce excessive dup/atomic updates
  // f/t/tmp can probably be removed actually and += del* directly? not sure why I did this, perhaps to avoid potential race conditions?

  // vector COM - backbone and base site a
  if (OXDNAFLAG==OXDNA) {
    constexpr F_FLOAT d_cs=-0.4;
    ra_cs[0] = d_cs*d_nx(a,0);
    ra_cs[1] = d_cs*d_nx(a,1);
    ra_cs[2] = d_cs*d_nx(a,2);
    ra_cb[0] = -ra_cs[0];
    ra_cb[1] = -ra_cs[1];
    ra_cb[2] = -ra_cs[2];
  } else if (OXDNAFLAG==OXDNA2) {
    constexpr F_FLOAT d_cs_x = -0.34;
    constexpr F_FLOAT d_cs_y = +0.3408;
    constexpr F_FLOAT d_cb = +0.4;
    ra_cs[0] = d_cs_x*d_nx(a,0) + d_cs_y*d_ny(a,0);
    ra_cs[1] = d_cs_x*d_nx(a,1) + d_cs_y*d_ny(a,1);
    ra_cs[2] = d_cs_x*d_nx(a,2) + d_cs_y*d_ny(a,2);
    ra_cb[0] = d_cb*d_nx(a,0);
    ra_cb[1] = d_cb*d_nx(a,1);
    ra_cb[2] = d_cb*d_nx(a,2);
  } else if (OXDNAFLAG==OXRNA2) {
    constexpr F_FLOAT d_cs_x = -0.4;
    constexpr F_FLOAT d_cs_z = +0.2;
    constexpr F_FLOAT d_cb = +0.4;
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
    const F_FLOAT factor_lj = special_lj[sbmask(b)];
    b &= NEIGHMASK;
    const int btype = type(b);

    // vector COM - backbone and base site b
    if (OXDNAFLAG==OXDNA) {
      constexpr F_FLOAT d_cs=-0.4;
      rb_cs[0] = d_cs*d_nx(b,0);
      rb_cs[1] = d_cs*d_nx(b,1);
      rb_cs[2] = d_cs*d_nx(b,2);
      rb_cb[0] = -rb_cs[0];
      rb_cb[1] = -rb_cs[1];
      rb_cb[2] = -rb_cs[2];
    } else if (OXDNAFLAG==OXDNA2) {
      constexpr F_FLOAT d_cs_x = -0.34;
      constexpr F_FLOAT d_cs_y = +0.3408;
      constexpr F_FLOAT d_cb = +0.4;
      rb_cs[0] = d_cs_x*d_nx(b,0) + d_cs_y*d_ny(b,0);
      rb_cs[1] = d_cs_x*d_nx(b,1) + d_cs_y*d_ny(b,1);
      rb_cs[2] = d_cs_x*d_nx(b,2) + d_cs_y*d_ny(b,2);
      rb_cb[0] = d_cb*d_nx(b,0);
      rb_cb[1] = d_cb*d_nx(b,1);
      rb_cb[2] = d_cb*d_nx(b,2);
    } else if (OXDNAFLAG==OXRNA2) {
      constexpr F_FLOAT d_cs_x = -0.4;
      constexpr F_FLOAT d_cs_z = +0.2;
      constexpr F_FLOAT d_cb = +0.4;
      rb_cs[0] = d_cs_x*d_nx(b,0) + d_cs_z*d_nz(b,0);
      rb_cs[1] = d_cs_x*d_nx(b,1) + d_cs_z*d_nz(b,1);
      rb_cs[2] = d_cs_x*d_nx(b,2) + d_cs_z*d_nz(b,2);
      rb_cb[0] = d_cb*d_nx(b,0);
      rb_cb[1] = d_cb*d_nx(b,1);
      rb_cb[2] = d_cb*d_nx(b,2);
    }

    // vector backbone site b to a
    delr_ss[0] = rtmp_s[0] - (x(b,0)+rb_cs[0]);
    delr_ss[1] = rtmp_s[1] - (x(b,1)+rb_cs[1]);
    delr_ss[2] = rtmp_s[2] - (x(b,2)+rb_cs[2]);
    rsq_ss = delr_ss[0]*delr_ss[0] + delr_ss[1]*delr_ss[1] + delr_ss[2]*delr_ss[2];
    // vector base site b to backbone site a
    delr_sb[0] = rtmp_s[0] - (x(b,0)+rb_cb[0]);
    delr_sb[1] = rtmp_s[1] - (x(b,1)+rb_cb[1]);
    delr_sb[2] = rtmp_s[2] - (x(b,2)+rb_cb[2]);
    rsq_sb = delr_sb[0]*delr_sb[0] + delr_sb[1]*delr_sb[1] + delr_sb[2]*delr_sb[2];
    // vector backbone site b to base site a
    delr_bs[0] = rtmp_b[0] - (x(b,0)+rb_cs[0]);
    delr_bs[1] = rtmp_b[1] - (x(b,1)+rb_cs[1]);
    delr_bs[2] = rtmp_b[2] - (x(b,2)+rb_cs[2]);
    rsq_bs = delr_bs[0]*delr_bs[0] + delr_bs[1]*delr_bs[1] + delr_bs[2]*delr_bs[2];
    // vector base site b to a
    delr_bb[0] = rtmp_b[0] - (x(b,0)+rb_cb[0]);
    delr_bb[1] = rtmp_b[1] - (x(b,1)+rb_cb[1]);
    delr_bb[2] = rtmp_b[2] - (x(b,2)+rb_cb[2]);
    rsq_bb = delr_bb[0]*delr_bb[0] + delr_bb[1]*delr_bb[1] + delr_bb[2]*delr_bb[2];

    // excluded volume interactions:
    //printf("rsq_ss: %f\n",rsq_ss);
    //printf("d_cut_ss_c: %f\n",d_cut_ss_c(atype,btype));

    // backbone-backbone
    if (rsq_ss < d_cutsq_ss_c(atype,btype)) {
      // F3 modulation factor, force and energy calculation
      if (rsq_ss < d_cutsq_ss_ast(atype,btype)) {
        const F_FLOAT r2inv = 1.0 / rsq_ss;
        const F_FLOAT r6inv = r2inv * r2inv * r2inv;
        fpair = r2inv * r6inv * \
          (12 * d_lj1_ss(atype,btype) * r6inv - 6 * d_lj2_ss(atype,btype));
        evdwl = r6inv * (d_lj1_ss(atype,btype) * r6inv - d_lj2_ss(atype,btype));
      } else {
        const F_FLOAT r = sqrt(rsq_ss);
        const F_FLOAT rinv = 1.0 / r;
        fpair = 2 * d_epsilon_ss(atype,btype) * d_b_ss(atype,btype) * \
          (d_cut_ss_c(atype,btype)  * rinv - 1);
        evdwl = d_epsilon_ss(atype,btype) * d_b_ss(atype,btype) * \
          (d_cut_ss_c(atype,btype) - r) * (d_cut_ss_c(atype,btype) - r);
      }
      // knock out nearest-neighbor interaction between ss
      fpair *= factor_lj;
      evdwl *= factor_lj;
      // force and torque increment calculation
      delf[0] = fpair * delr_ss[0];
      delf[1] = fpair * delr_ss[1];
      delf[2] = fpair * delr_ss[2];
      delta[0] = ra_cs[1]*delf[2] - ra_cs[2]*delf[1];
      delta[1] = ra_cs[2]*delf[0] - ra_cs[0]*delf[2];
      delta[2] = ra_cs[0]*delf[1] - ra_cs[1]*delf[0];
      ftmp[0] += delf[0];
      ftmp[1] += delf[1];
      ftmp[2] += delf[2];
      ttmp[0] += delta[0];
      ttmp[1] += delta[1];
      ttmp[2] += delta[2];
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
          //ev.evdwl += evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // backbone-base
    if (rsq_sb < d_cutsq_sb_c(atype,btype)) {
      // F3 modulation factor, force and energy calculation
      if (rsq_sb < d_cutsq_sb_ast(atype,btype)) {
        const F_FLOAT r2inv = 1.0 / rsq_sb;
        const F_FLOAT r6inv = r2inv * r2inv * r2inv;
        fpair = r2inv * r6inv * \
          (12 * d_lj1_sb(atype,btype) * r6inv - 6 * d_lj2_sb(atype,btype));
        evdwl = r6inv * (d_lj1_sb(atype,btype) * r6inv - d_lj2_sb(atype,btype));
      } else {
        const F_FLOAT r = sqrt(rsq_sb);
        const F_FLOAT rinv = 1.0 / r;
        fpair = 2 * d_epsilon_sb(atype,btype) * d_b_sb(atype,btype) * \
          (d_cut_sb_c(atype,btype)  * rinv - 1);
        evdwl = d_epsilon_sb(atype,btype) * d_b_sb(atype,btype) * \
          (d_cut_sb_c(atype,btype) - r) * (d_cut_sb_c(atype,btype) - r);
      }
      // force and torque increment calculation
      delf[0] = fpair * delr_sb[0];
      delf[1] = fpair * delr_sb[1];
      delf[2] = fpair * delr_sb[2];
      delta[0] = ra_cs[1]*delf[2] - ra_cs[2]*delf[1];
      delta[1] = ra_cs[2]*delf[0] - ra_cs[0]*delf[2];
      delta[2] = ra_cs[0]*delf[1] - ra_cs[1]*delf[0];
      ftmp[0] += delf[0];
      ftmp[1] += delf[1];
      ftmp[2] += delf[2];
      ttmp[0] += delta[0];
      ttmp[1] += delta[1];
      ttmp[2] += delta[2];
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
          //ev.evdwl += evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // base-backbone
    if (rsq_bs < d_cutsq_sb_c(btype,atype)) {
      // F3 modulation factor, force and energy calculation
      if (rsq_bs < d_cutsq_sb_ast(btype,atype)) {
        const F_FLOAT r2inv = 1.0 / rsq_bs;
        const F_FLOAT r6inv = r2inv * r2inv * r2inv;
        fpair = r2inv * r6inv * \
          (12 * d_lj1_sb(btype,atype) * r6inv - 6 * d_lj2_sb(btype,atype));
        evdwl = r6inv * (d_lj1_sb(btype,atype) * r6inv - d_lj2_sb(btype,atype));
      } else {
        const F_FLOAT r = sqrt(rsq_bs);
        const F_FLOAT rinv = 1.0 / r;
        fpair = 2 * d_epsilon_sb(btype,atype) * d_b_sb(btype,atype) * \
          (d_cut_sb_c(btype,atype)  * rinv - 1);
        evdwl = d_epsilon_sb(btype,atype) * d_b_sb(btype,atype) * \
          (d_cut_sb_c(btype,atype) - r) * (d_cut_sb_c(btype,atype) - r);
      }
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
          //ev.evdwl += evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // base-base
    if (rsq_bb < d_cutsq_bb_c(atype,btype)) {
      // F3 modulation factor, force and energy calculation
      if (rsq_bb < d_cutsq_bb_ast(atype,btype)) {
        const F_FLOAT r2inv = 1.0 / rsq_bb;
        const F_FLOAT r6inv = r2inv * r2inv * r2inv;
        fpair = r2inv * r6inv * \
          (12 * d_lj1_bb(atype,btype) * r6inv - 6 * d_lj2_bb(atype,btype));
        evdwl = r6inv * (d_lj1_bb(atype,btype) * r6inv - d_lj2_bb(atype,btype));
      } else {
        const F_FLOAT r = sqrt(rsq_bb);
        const F_FLOAT rinv = 1.0 / r;
        fpair = 2 * d_epsilon_bb(atype,btype) * d_b_bb(atype,btype) * \
          (d_cut_bb_c(atype,btype)  * rinv - 1);
        evdwl = d_epsilon_bb(atype,btype) * d_b_bb(atype,btype) * \
          (d_cut_bb_c(atype,btype) - r) * (d_cut_bb_c(atype,btype) - r);
      }
      // force and torque increment calculation
      delf[0] = fpair * delr_bb[0];
      delf[1] = fpair * delr_bb[1];
      delf[2] = fpair * delr_bb[2];
      delta[0] = ra_cb[1]*delf[2] - ra_cb[2]*delf[1];
      delta[1] = ra_cb[2]*delf[0] - ra_cb[0]*delf[2];
      delta[2] = ra_cb[0]*delf[1] - ra_cb[1]*delf[0];
      ftmp[0] += delf[0];
      ftmp[1] += delf[1];
      ftmp[2] += delf[2];
      ttmp[0] += delta[0];
      ttmp[1] += delta[1];
      ttmp[2] += delta[2];
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
          //ev.evdwl += evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }
    //printf("INDEX ia, a, ib, b: %d %d %d %d\n",ia,a,ib,b);
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
void PairOxdnaStkKokkos<DeviceType>::operator()(TagPairOxdnaStkCompute<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>, \
  const int &ia) const
{
  EV_FLOAT ev;
  this->template operator()<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>\
  (TagPairOxdnaStkCompute<OXDNAFLAG,NEIGHFLAG,NEWTON_PAIR,EVFLAG>(),ia,ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairOxdnaStkKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                        DAT::tdual_xfloat_1d &buf,
                                                        int /*pbc_flag*/, int * /*pbc*/)
{
  d_sendlist = k_sendlist.view<DeviceType>();
  v_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkPackForwardComm>(0,n),*this);
  return n*9;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaStkKokkos<DeviceType>::operator()(TagPairOxdnaStkPackForwardComm, const int &i) const {
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
void PairOxdnaStkKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_xfloat_1d &buf)
{
  first = first_in;
  v_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkUnpackForwardComm>(0,n),*this);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaStkKokkos<DeviceType>::operator()(TagPairOxdnaStkUnpackForwardComm, const int &i) const {
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
int PairOxdnaStkKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
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
void PairOxdnaStkKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
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
void *PairOxdnaStkKokkos<DeviceType>::extract(const char *str, int &dim)
{
  PairOxdnaStk::extract(str,dim);

  if (strcmp(str,"d_nx") == 0) return (void *) d_nx.data();
  if (strcmp(str,"d_ny") == 0) return (void *) d_ny.data();
  if (strcmp(str,"d_nz") == 0) return (void *) d_nz.data();


  return nullptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaStkKokkos<DeviceType>::allocate()
{
  PairOxdnaStk::allocate();

  int n = atom->ntypes;

  //memory->destroy(setflag);

  // sequence-specific stacking strength
  // A:0 C:1 G:2 T:3, 3'- [i][j] -5'
  memory->destroy(eta_st);
  memoryKK->create_kokkos(k_eta_st,eta_st,4,4,"PairOxdnaStk:eta_st");
  d_eta_st = k_eta_st.view<DeviceType>();
  // TODO: should eta_st be here?
  d_eta_st(0,0) = 1.11960;
  d_eta_st(1,0) = 1.00852;
  d_eta_st(2,0) = 0.96950;
  d_eta_st(3,0) = 0.99632;
  d_eta_st(0,1) = 1.01889;
  d_eta_st(1,1) = 0.97804;
  d_eta_st(2,1) = 1.02681;
  d_eta_st(3,1) = 0.96950;
  d_eta_st(0,2) = 0.98169;
  d_eta_st(1,2) = 1.05913;
  d_eta_st(2,2) = 0.97804;
  d_eta_st(3,2) = 1.00852;
  d_eta_st(0,3) = 0.94694;
  d_eta_st(1,3) = 0.98169;
  d_eta_st(2,3) = 1.01889;
  d_eta_st(3,3) = 0.96383;
  
  memory->destroy(epsilon_st);
  memory->destroy(a_st);
  memory->destroy(cut_st_0);
  memory->destroy(cut_st_c);
  memory->destroy(cut_st_lo);
  memory->destroy(cut_st_hi);
  memory->destroy(cut_st_lc);
  memory->destroy(cut_st_hc);
  memory->destroy(b_st_lo);
  memory->destroy(b_st_hi);
  memory->destroy(shift_st);
  memory->destroy(cutsq_st_hc);

  memory->destroy(a_st4);
  memory->destroy(theta_st4_0);
  memory->destroy(dtheta_st4_ast);
  memory->destroy(b_st4);
  memory->destroy(dtheta_st4_c);

  memory->destroy(a_st5);
  memory->destroy(theta_st5_0);
  memory->destroy(dtheta_st5_ast);
  memory->destroy(b_st5);
  memory->destroy(dtheta_st5_c);

  memory->destroy(a_st6);
  memory->destroy(theta_st6_0);
  memory->destroy(dtheta_st6_ast);
  memory->destroy(b_st6);
  memory->destroy(dtheta_st6_c);

  memory->destroy(a_st1);
  memory->destroy(cosphi_st1_ast);
  memory->destroy(b_st1);
  memory->destroy(cosphi_st1_c);
  memory->destroy(a_st2);
  memory->destroy(cosphi_st2_ast);
  memory->destroy(b_st2);
  memory->destroy(cosphi_st2_c);

  memoryKK->create_kokkos(k_epsilon_st,epsilon_st,n+1,n+1,"PairOxdnaStk:epsilon_st");
  memoryKK->create_kokkos(k_a_st,a_st,n+1,n+1,"PairOxdnaStk:a_st");
  memoryKK->create_kokkos(k_cut_st_0,cut_st_0,n+1,n+1,"PairOxdnaStk:cut_st_0");
  memoryKK->create_kokkos(k_cut_st_c,cut_st_c,n+1,n+1,"PairOxdnaStk:cut_st_c");
  memoryKK->create_kokkos(k_cut_st_lo,cut_st_lo,n+1,n+1,"PairOxdnaStk:cut_st_lo");
  memoryKK->create_kokkos(k_cut_st_hi,cut_st_hi,n+1,n+1,"PairOxdnaStk:cut_st_hi");
  memoryKK->create_kokkos(k_cut_st_lc,cut_st_lc,n+1,n+1,"PairOxdnaStk:cut_st_lc");
  memoryKK->create_kokkos(k_cut_st_hc,cut_st_hc,n+1,n+1,"PairOxdnaStk:cut_st_hc");
  memoryKK->create_kokkos(k_b_st_lo,b_st_lo,n+1,n+1,"PairOxdnaStk:b_st_lo");
  memoryKK->create_kokkos(k_b_st_hi,b_st_hi,n+1,n+1,"PairOxdnaStk:b_st_hi");
  memoryKK->create_kokkos(k_shift_st,shift_st,n+1,n+1,"PairOxdnaStk:shift_st");
  memoryKK->create_kokkos(k_cutsq_st_hc,cutsq_st_hc,n+1,n+1,"PairOxdnaStk:cutsq_st_hc");

  memoryKK->create_kokkos(k_a_st4,a_st4,n+1,n+1,"PairOxdnaStk:a_st4");
  memoryKK->create_kokkos(k_theta_st4_0,theta_st4_0,n+1,n+1,"PairOxdnaStk:theta_st4_0");
  memoryKK->create_kokkos(k_dtheta_st4_ast,dtheta_st4_ast,n+1,n+1,"PairOxdnaStk:dtheta_st4_ast");
  memoryKK->create_kokkos(k_b_st4,b_st4,n+1,n+1,"PairOxdnaStk:b_st4");
  memoryKK->create_kokkos(k_dtheta_st4_c,dtheta_st4_c,n+1,n+1,"PairOxdnaStk:dtheta_st4_c");

  memoryKK->create_kokkos(k_a_st5,a_st5,n+1,n+1,"PairOxdnaStk:a_st5");
  memoryKK->create_kokkos(k_theta_st5_0,theta_st5_0,n+1,n+1,"PairOxdnaStk:theta_st5_0");
  memoryKK->create_kokkos(k_dtheta_st5_ast,dtheta_st5_ast,n+1,n+1,"PairOxdnaStk:dtheta_st5_ast");
  memoryKK->create_kokkos(k_b_st5,b_st5,n+1,n+1,"PairOxdnaStk:b_st5");
  memoryKK->create_kokkos(k_dtheta_st5_c,dtheta_st5_c,n+1,n+1,"PairOxdnaStk:dtheta_st5_c");

  memoryKK->create_kokkos(k_a_st6,a_st6,n+1,n+1,"PairOxdnaStk:a_st6");
  memoryKK->create_kokkos(k_theta_st6_0,theta_st6_0,n+1,n+1,"PairOxdnaStk:theta_st6_0");
  memoryKK->create_kokkos(k_dtheta_st6_ast,dtheta_st6_ast,n+1,n+1,"PairOxdnaStk:dtheta_st6_ast");
  memoryKK->create_kokkos(k_b_st6,b_st6,n+1,n+1,"PairOxdnaStk:b_st6");
  memoryKK->create_kokkos(k_dtheta_st6_c,dtheta_st6_c,n+1,n+1,"PairOxdnaStk:dtheta_st6_c");

  memoryKK->create_kokkos(k_a_st1,a_st1,n+1,n+1,"PairOxdnaStk:a_st1");
  memoryKK->create_kokkos(k_cosphi_st1_ast,cosphi_st1_ast,n+1,n+1,"PairOxdnaStk:cosphi_st1_ast");
  memoryKK->create_kokkos(k_b_st1,b_st1,n+1,n+1,"PairOxdnaStk:b_st1");
  memoryKK->create_kokkos(k_cosphi_st1_c,cosphi_st1_c,n+1,n+1,"PairOxdnaStk:cosphi_st1_c");
  memoryKK->create_kokkos(k_a_st2,a_st2,n+1,n+1,"PairOxdnaStk:a_st2");
  memoryKK->create_kokkos(k_cosphi_st2_ast,cosphi_st2_ast,n+1,n+1,"PairOxdnaStk:cosphi_st2_ast");
  memoryKK->create_kokkos(k_b_st2,b_st2,n+1,n+1,"PairOxdnaStk:b_st2");
  memoryKK->create_kokkos(k_cosphi_st2_c,cosphi_st2_c,n+1,n+1,"PairOxdnaStk:cosphi_st2_c");

  d_epsilon_st = k_epsilon_st.template view<DeviceType>();
  d_a_st = k_a_st.template view<DeviceType>();
  d_cut_st_0 = k_cut_st_0.template view<DeviceType>();
  d_cut_st_c = k_cut_st_c.template view<DeviceType>();
  d_cut_st_lo = k_cut_st_lo.template view<DeviceType>();
  d_cut_st_hi = k_cut_st_hi.template view<DeviceType>();
  d_cut_st_lc = k_cut_st_lc.template view<DeviceType>();
  d_cut_st_hc = k_cut_st_hc.template view<DeviceType>();
  d_b_st_lo = k_b_st_lo.template view<DeviceType>();
  d_b_st_hi = k_b_st_hi.template view<DeviceType>();
  d_shift_st = k_shift_st.template view<DeviceType>();
  d_cutsq_st_hc = k_cutsq_st_hc.template view<DeviceType>();

  d_a_st4 = k_a_st4.template view<DeviceType>();
  d_theta_st4_0 = k_theta_st4_0.template view<DeviceType>();
  d_dtheta_st4_ast = k_dtheta_st4_ast.template view<DeviceType>();
  d_b_st4 = k_b_st4.template view<DeviceType>();
  d_dtheta_st4_c = k_dtheta_st4_c.template view<DeviceType>();

  d_a_st5 = k_a_st5.template view<DeviceType>();
  d_theta_st5_0 = k_theta_st5_0.template view<DeviceType>();
  d_dtheta_st5_ast = k_dtheta_st5_ast.template view<DeviceType>();
  d_b_st5 = k_b_st5.template view<DeviceType>();
  d_dtheta_st5_c = k_dtheta_st5_c.template view<DeviceType>();

  d_a_st6 = k_a_st6.template view<DeviceType>();
  d_theta_st6_0 = k_theta_st6_0.template view<DeviceType>();
  d_dtheta_st6_ast = k_dtheta_st6_ast.template view<DeviceType>();
  d_b_st6 = k_b_st6.template view<DeviceType>();
  d_dtheta_st6_c = k_dtheta_st6_c.template view<DeviceType>();

  d_a_st1 = k_a_st1.template view<DeviceType>();
  d_cosphi_st1_ast = k_cosphi_st1_ast.template view<DeviceType>();
  d_b_st1 = k_b_st1.template view<DeviceType>();
  d_cosphi_st1_c = k_cosphi_st1_c.template view<DeviceType>();
  d_a_st2 = k_a_st2.template view<DeviceType>();
  d_cosphi_st2_ast = k_cosphi_st2_ast.template view<DeviceType>();
  d_b_st2 = k_b_st2.template view<DeviceType>();
  d_cosphi_st2_c = k_cosphi_st2_c.template view<DeviceType>();

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaStkKokkos<DeviceType>::settings(int narg, char **/*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaStkKokkos<DeviceType>::init_style() 
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
double PairOxdnaStkKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairOxdnaStk::init_one(i,j);

  k_epsilon_st.h_view(i,j) = k_epsilon_st.h_view(j,i) = epsilon_st[i][j];
  k_a_st.h_view(i,j) = k_a_st.h_view(j,i) = a_st[i][j];
  k_cut_st_0.h_view(i,j) = k_cut_st_0.h_view(j,i) = cut_st_0[i][j];
  k_cut_st_c.h_view(i,j) = k_cut_st_c.h_view(j,i) = cut_st_c[i][j];
  k_cut_st_lo.h_view(i,j) = k_cut_st_lo.h_view(j,i) = cut_st_lo[i][j];
  k_cut_st_hi.h_view(i,j) = k_cut_st_hi.h_view(j,i) = cut_st_hi[i][j];
  k_cut_st_lc.h_view(i,j) = k_cut_st_lc.h_view(j,i) = cut_st_lc[i][j];
  k_cut_st_hc.h_view(i,j) = k_cut_st_hc.h_view(j,i) = cut_st_hc[i][j];
  k_b_st_lo.h_view(i,j) = k_b_st_lo.h_view(j,i) = b_st_lo[i][j];
  k_b_st_hi.h_view(i,j) = k_b_st_hi.h_view(j,i) = b_st_hi[i][j];
  k_shift_st.h_view(i,j) = k_shift_st.h_view(j,i) = shift_st[i][j];
  k_cutsq_st_hc.h_view(i,j) = k_cutsq_st_hc.h_view(j,i) = cutsq_st_hc[i][j];

  k_a_st4.h_view(i,j) = k_a_st4.h_view(j,i) = a_st4[i][j];
  k_theta_st4_0.h_view(i,j) = k_theta_st4_0.h_view(j,i) = theta_st4_0[i][j];
  k_dtheta_st4_ast.h_view(i,j) = k_dtheta_st4_ast.h_view(j,i) = dtheta_st4_ast[i][j];
  k_b_st4.h_view(i,j) = k_b_st4.h_view(j,i) = b_st4[i][j];
  k_dtheta_st4_c.h_view(i,j) = k_dtheta_st4_c.h_view(j,i) = dtheta_st4_c[i][j];

  k_a_st5.h_view(i,j) = k_a_st5.h_view(j,i) = a_st5[i][j];
  k_theta_st5_0.h_view(i,j) = k_theta_st5_0.h_view(j,i) = theta_st5_0[i][j];
  k_dtheta_st5_ast.h_view(i,j) = k_dtheta_st5_ast.h_view(j,i) = dtheta_st5_ast[i][j];
  k_b_st5.h_view(i,j) = k_b_st5.h_view(j,i) = b_st5[i][j];
  k_dtheta_st5_c.h_view(i,j) = k_dtheta_st5_c.h_view(j,i) = dtheta_st5_c[i][j];

  k_a_st6.h_view(i,j) = k_a_st6.h_view(j,i) = a_st6[i][j];
  k_theta_st6_0.h_view(i,j) = k_theta_st6_0.h_view(j,i) = theta_st6_0[i][j];
  k_dtheta_st6_ast.h_view(i,j) = k_dtheta_st6_ast.h_view(j,i) = dtheta_st6_ast[i][j];
  k_b_st6.h_view(i,j) = k_b_st6.h_view(j,i) = b_st6[i][j];
  k_dtheta_st6_c.h_view(i,j) = k_dtheta_st6_c.h_view(j,i) = dtheta_st6_c[i][j];

  k_a_st1.h_view(i,j) = k_a_st1.h_view(j,i) = a_st1[i][j];
  k_cosphi_st1_ast.h_view(i,j) = k_cosphi_st1_ast.h_view(j,i) = cosphi_st1_ast[i][j];
  k_b_st1.h_view(i,j) = k_b_st1.h_view(j,i) = b_st1[i][j];
  k_cosphi_st1_c.h_view(i,j) = k_cosphi_st1_c.h_view(j,i) = cosphi_st1_c[i][j];
  k_a_st2.h_view(i,j) = k_a_st2.h_view(j,i) = a_st2[i][j];
  k_cosphi_st2_ast.h_view(i,j) = k_cosphi_st2_ast.h_view(j,i) = cosphi_st2_ast[i][j];
  k_b_st2.h_view(i,j) = k_b_st2.h_view(j,i) = b_st2[i][j];
  k_cosphi_st2_c.h_view(i,j) = k_cosphi_st2_c.h_view(j,i) = cosphi_st2_c[i][j];

  // "cutone" is "cut_st_hc[i][j]", sets the master list distance cutoff
  return cutone;

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairOxdnaStkKokkos<DeviceType>::ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
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
      const E_FLOAT epairhalf = 0.5 * epair;
      if (NEIGHFLAG!=FULL) {
        if (NEWTON_PAIR || i < nlocal) a_eatom[i] += epairhalf;
        if (NEWTON_PAIR || j < nlocal) a_eatom[j] += epairhalf;
      } else {
        a_eatom[i] += epairhalf;
      }
    }
  }

  if (VFLAG) {
    const E_FLOAT v0 = delx*fx;
    const E_FLOAT v1 = dely*fy;
    const E_FLOAT v2 = delz*fz;
    const E_FLOAT v3 = delx*fy;
    const E_FLOAT v4 = delx*fz;
    const E_FLOAT v5 = dely*fz;

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
int PairOxdnaStkKokkos<DeviceType>::sbmask(const int& j) const {
  return j >> SBBITS & 3;
}


namespace LAMMPS_NS {
template class PairOxdnaStkKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairOxdnaStkKokkos<LMPHostType>;
#endif
}