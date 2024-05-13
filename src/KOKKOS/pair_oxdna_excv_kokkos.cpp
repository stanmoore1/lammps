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
//#include "atom_vec_ellipsoid_kokkos.h" ???
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
//using namespace MFOxdna;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairOxdnaExcvKokkos<DeviceType>::PairOxdnaExcvKokkos(LAMMPS *lmp) : PairOxdnaExcv(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  nmax = 0;
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
    memoryKK->destroy_kokkos(k_cutsq,cutsq);

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
  else atomKK->modified(execution_space,F_MASK); //need or not need? same for fene, also add TORQUE_MASK later

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  //torque = atomKK->k_torque.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

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
    //dup_torque = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, \
    //Kokkos::Experimental::ScatterDuplicated>(torque);
  } else {
    ndup_f = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, \
    Kokkos::Experimental::ScatterNonDuplicated>(f);
    //ndup_torque = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, \
    //Kokkos::Experimental::ScatterNonDuplicated>(torque);
  }

  copymode = 1;

  // loop over all local atoms, calculation of local reference frame from quaternions
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagPairOxdnaExcvQuatToXYZ>(0,nlocal),*this);
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
        if (oxdnaflag) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALF,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALF,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALF,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALF,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALF,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALF,0,0> >(0,anum),*this);
        }
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        if (oxdnaflag) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALFTHREAD,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALFTHREAD,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALFTHREAD,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,HALFTHREAD,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,HALFTHREAD,0,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,HALFTHREAD,0,0> >(0,anum),*this);
        }
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        if (oxdnaflag) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA,FULL,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXDNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXDNA2,FULL,1,0> >(0,anum),*this);
        } else if (oxdnaflag==OXRNA2) {
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaExcvCompute<OXRNA2,FULL,1,0> >(0,anum),*this);
        }
      } else {
        if (oxdnaflag) {
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
    //Kokkos::Experimental::contribute(torque, dup_torque);
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
    //dup_torque   = decltype(dup_torque)();
    dup_eatom    = decltype(dup_eatom)();
    dup_vatom    = decltype(dup_vatom)();
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::operator()(TagPairOxdnaExcvQuatToXYZ, const int &in) const
{
  int n = d_alist(in);
  // TODO: implement quaternion to Cartesian unit vectors in lab frame
  d_nx(n,0) = 0.0;
  d_nx(n,1) = 0.0;
  d_nx(n,2) = 0.0;
  d_ny(n,0) = 0.0;
  d_ny(n,1) = 0.0;
  d_ny(n,2) = 0.0;
  d_nz(n,0) = 0.0;
  d_nz(n,1) = 0.0;
  d_nz(n,2) = 0.0;
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
  //auto v_torque = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,\
  //  decltype(dup_torque),decltype(ndup_torque)>::get(dup_torque,ndup_torque);
  //auto a_torque = v_torque.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

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
      // f3 modulation factor, force and energy calculation
      if (rsq_ss < d_cutsq_ss_ast(atype,btype)) {
        const F_FLOAT r2inv = 1.0 / rsq_ss;
        const F_FLOAT r6inv = r2inv * r2inv * r2inv;
        fpair = factor_lj * r2inv * r6inv * \
          (12 * d_lj1_ss(atype,btype) * r6inv - 6 * d_lj2_ss(atype,btype));
        ev.evdwl = factor_lj * r6inv * (d_lj1_ss(atype,btype) * r6inv - d_lj2_ss(atype,btype));
      } else {
        const F_FLOAT r = sqrt(rsq_ss);
        const F_FLOAT rinv = 1.0 / r;
        fpair = factor_lj * 2 * d_epsilon_ss(atype,btype) * d_b_ss(atype,btype) * \
          (d_cut_ss_c(atype,btype)  * rinv - 1);
        ev.evdwl = d_epsilon_ss(atype,btype) * d_b_ss(atype,btype) * \
          (d_cut_ss_c(atype,btype) - r) * (d_cut_ss_c(atype,btype) - r);
      }
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
        //a_torque(b,0) -= delta[0];
        //a_torque(b,1) -= delta[1];
        //a_torque(b,2) -= delta[2];
      }
      if (EVFLAG) {
        if (eflag) {
          ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*ev.evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // backbone-base
    if (rsq_sb < d_cutsq_sb_c(atype,btype)) {
      // f3 modulation factor, force and energy calculation
      if (rsq_sb < d_cutsq_sb_ast(atype,btype)) {
        const F_FLOAT r2inv = 1.0 / rsq_sb;
        const F_FLOAT r6inv = r2inv * r2inv * r2inv;
        fpair = factor_lj * r2inv * r6inv * \
          (12 * d_lj1_sb(atype,btype) * r6inv - 6 * d_lj2_sb(atype,btype));
        ev.evdwl = factor_lj * r6inv * (d_lj1_sb(atype,btype) * r6inv - d_lj2_sb(atype,btype));
      } else {
        const F_FLOAT r = sqrt(rsq_sb);
        const F_FLOAT rinv = 1.0 / r;
        fpair = factor_lj * 2 * d_epsilon_sb(atype,btype) * d_b_sb(atype,btype) * \
          (d_cut_sb_c(atype,btype)  * rinv - 1);
        ev.evdwl = d_epsilon_sb(atype,btype) * d_b_sb(atype,btype) * \
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
        //a_torque(b,0) -= delta[0];
        //a_torque(b,1) -= delta[1];
        //a_torque(b,2) -= delta[2];
      }
      if (EVFLAG) {
        if (eflag) {
          ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*ev.evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // base-backbone
    if (rsq_bs < d_cutsq_sb_c(btype,atype)) {
      // f3 modulation factor, force and energy calculation
      if (rsq_bs < d_cutsq_sb_ast(btype,atype)) {
        const F_FLOAT r2inv = 1.0 / rsq_bs;
        const F_FLOAT r6inv = r2inv * r2inv * r2inv;
        fpair = factor_lj * r2inv * r6inv * \
          (12 * d_lj1_sb(btype,atype) * r6inv - 6 * d_lj2_sb(btype,atype));
        ev.evdwl = factor_lj * r6inv * (d_lj1_sb(btype,atype) * r6inv - d_lj2_sb(btype,atype));
      } else {
        const F_FLOAT r = sqrt(rsq_bs);
        const F_FLOAT rinv = 1.0 / r;
        fpair = factor_lj * 2 * d_epsilon_sb(btype,atype) * d_b_sb(btype,atype) * \
          (d_cut_sb_c(btype,atype)  * rinv - 1);
        ev.evdwl = d_epsilon_sb(btype,atype) * d_b_sb(btype,atype) * \
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
        //a_torque(b,0) -= delta[0];
        //a_torque(b,1) -= delta[1];
        //a_torque(b,2) -= delta[2];
      }
      if (EVFLAG) {
        if (eflag) {
          ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*ev.evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }

    // base-base
    if (rsq_bb < d_cutsq_bb_c(atype,btype)) {
      // f3 modulation factor, force and energy calculation
      if (rsq_bb < d_cutsq_bb_ast(atype,btype)) {
        const F_FLOAT r2inv = 1.0 / rsq_bb;
        const F_FLOAT r6inv = r2inv * r2inv * r2inv;
        fpair = factor_lj * r2inv * r6inv * \
          (12 * d_lj1_bb(atype,btype) * r6inv - 6 * d_lj2_bb(atype,btype));
        ev.evdwl = factor_lj * r6inv * (d_lj1_bb(atype,btype) * r6inv - d_lj2_bb(atype,btype));
      } else {
        const F_FLOAT r = sqrt(rsq_bb);
        const F_FLOAT rinv = 1.0 / r;
        fpair = factor_lj * 2 * d_epsilon_bb(atype,btype) * d_b_bb(atype,btype) * \
          (d_cut_bb_c(atype,btype)  * rinv - 1);
        ev.evdwl = d_epsilon_bb(atype,btype) * d_b_bb(atype,btype) * \
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
        //a_torque(b,0) -= delta[0];
        //a_torque(b,1) -= delta[1];
        //a_torque(b,2) -= delta[2];
      }
      if (EVFLAG) {
        if (eflag) {
          ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*ev.evdwl;
        }

        if (vflag_either || eflag_atom) {
          this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
          delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
        }
      }
    }
  }
  a_f(a,0) += ftmp[0];
  a_f(a,1) += ftmp[1];
  a_f(a,2) += ftmp[2];
  //a_torque(a,0) += ttmp[0];
  //a_torque(a,1) += ttmp[1];
  //a_torque(a,2) += ttmp[2];
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
                                                        DAT::tdual_xfloat_1d &buf,
                                                        int /*pbc_flag*/, int * /*pbc*/)
{
  d_sendlist = k_sendlist.view<DeviceType>();
  v_buf = buf.view<DeviceType>();
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
void PairOxdnaExcvKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_xfloat_1d &buf)
{
  first = first_in;
  v_buf = buf.view<DeviceType>();
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
void PairOxdnaExcvKokkos<DeviceType>::allocate()
{
  PairOxdnaExcv::allocate();

  int n = atom->ntypes;

  //memory->destroy(setflag);
  //memory->destroy(cutsq);
  
  memory->destroy(epsilon_ss);
  memory->destroy(sigma_ss);
  memory->destroy(cut_ss_ast);
  memory->destroy(b_ss);
  memory->destroy(cut_ss_c);
  memory->destroy(lj1_ss);
  memory->destroy(lj2_ss);
  memory->destroy(cutsq_ss_ast);
  memory->destroy(cutsq_ss_c);

  memory->destroy(epsilon_sb);
  memory->destroy(sigma_sb);
  memory->destroy(cut_sb_ast);
  memory->destroy(b_sb);
  memory->destroy(cut_sb_c);
  memory->destroy(lj1_sb);
  memory->destroy(lj2_sb);
  memory->destroy(cutsq_sb_ast);
  memory->destroy(cutsq_sb_c);

  memory->destroy(epsilon_bb);
  memory->destroy(sigma_bb);
  memory->destroy(cut_bb_ast);
  memory->destroy(b_bb);
  memory->destroy(cut_bb_c);
  memory->destroy(lj1_bb);
  memory->destroy(lj2_bb);
  memory->destroy(cutsq_bb_ast);
  memory->destroy(cutsq_bb_c);

  memory->destroy(nx);
  memory->destroy(ny);
  memory->destroy(nz);

  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"PairOxdnaExcv:cutsq");

  memoryKK->create_kokkos(k_epsilon_ss,epsilon_ss,n+1,n+1,"PairOxdnaExcv:epsilon_ss");
  memoryKK->create_kokkos(k_sigma_ss,sigma_ss,n+1,n+1,"PairOxdnaExcv:sigma_ss");
  memoryKK->create_kokkos(k_cut_ss_ast,cut_ss_ast,n+1,n+1,"PairOxdnaExcv:cut_ss_ast");
  memoryKK->create_kokkos(k_b_ss,b_ss,n+1,n+1,"PairOxdnaExcv:b_ss");
  memoryKK->create_kokkos(k_cut_ss_c,cut_ss_c,n+1,n+1,"PairOxdnaExcv:cut_ss_c");
  memoryKK->create_kokkos(k_lj1_ss,lj1_ss,n+1,n+1,"PairOxdnaExcv:lj1_ss");
  memoryKK->create_kokkos(k_lj2_ss,lj2_ss,n+1,n+1,"PairOxdnaExcv:lj2_ss");
  memoryKK->create_kokkos(k_cutsq_ss_ast,cutsq_ss_ast,n+1,n+1,"PairOxdnaExcv:cutsq_ss_ast");
  memoryKK->create_kokkos(k_cutsq_ss_c,cutsq_ss_c,n+1,n+1,"PairOxdnaExcv:cutsq_ss_c");

  memoryKK->create_kokkos(k_epsilon_sb,epsilon_sb,n+1,n+1,"PairOxdnaExcv:epsilon_sb");
  memoryKK->create_kokkos(k_sigma_sb,sigma_sb,n+1,n+1,"PairOxdnaExcv:sigma_sb");
  memoryKK->create_kokkos(k_cut_sb_ast,cut_sb_ast,n+1,n+1,"PairOxdnaExcv:cut_sb_ast");
  memoryKK->create_kokkos(k_b_sb,b_sb,n+1,n+1,"PairOxdnaExcv:b_sb");
  memoryKK->create_kokkos(k_cut_sb_c,cut_sb_c,n+1,n+1,"PairOxdnaExcv:cut_sb_c");
  memoryKK->create_kokkos(k_lj1_sb,lj1_sb,n+1,n+1,"PairOxdnaExcv:lj1_sb");
  memoryKK->create_kokkos(k_lj2_sb,lj2_sb,n+1,n+1,"PairOxdnaExcv:lj2_sb");
  memoryKK->create_kokkos(k_cutsq_sb_ast,cutsq_sb_ast,n+1,n+1,"PairOxdnaExcv:cutsq_sb_ast");
  memoryKK->create_kokkos(k_cutsq_sb_c,cutsq_sb_c,n+1,n+1,"PairOxdnaExcv:cutsq_sb_c");

  memoryKK->create_kokkos(k_epsilon_bb,epsilon_bb,n+1,n+1,"PairOxdnaExcv:epsilon_bb");
  memoryKK->create_kokkos(k_sigma_bb,sigma_bb,n+1,n+1,"PairOxdnaExcv:sigma_bb");
  memoryKK->create_kokkos(k_cut_bb_ast,cut_bb_ast,n+1,n+1,"PairOxdnaExcv:cut_bb_ast");
  memoryKK->create_kokkos(k_b_bb,b_bb,n+1,n+1,"PairOxdnaExcv:b_bb");
  memoryKK->create_kokkos(k_cut_bb_c,cut_bb_c,n+1,n+1,"PairOxdnaExcv:cut_bb_c");
  memoryKK->create_kokkos(k_lj1_bb,lj1_bb,n+1,n+1,"PairOxdnaExcv:lj1_bb");
  memoryKK->create_kokkos(k_lj2_bb,lj2_bb,n+1,n+1,"PairOxdnaExcv:lj2_bb");
  memoryKK->create_kokkos(k_cutsq_bb_ast,cutsq_bb_ast,n+1,n+1,"PairOxdnaExcv:cutsq_bb_ast");
  memoryKK->create_kokkos(k_cutsq_bb_c,cutsq_bb_c,n+1,n+1,"PairOxdnaExcv:cutsq_bb_c");

  memoryKK->create_kokkos(k_nx,nx,atom->nmax,3,"PairOxdnaExcv:nx");
  memoryKK->create_kokkos(k_ny,ny,atom->nmax,3,"PairOxdnaExcv:ny");
  memoryKK->create_kokkos(k_nz,nz,atom->nmax,3,"PairOxdnaExcv:nz");

  d_cutsq = k_cutsq.template view<DeviceType>();

  d_epsilon_ss = k_epsilon_ss.template view<DeviceType>();
  d_sigma_ss = k_sigma_ss.template view<DeviceType>();
  d_cut_ss_ast = k_cut_ss_ast.template view<DeviceType>();
  d_b_ss = k_b_ss.template view<DeviceType>();
  d_cut_ss_c = k_cut_ss_c.template view<DeviceType>();
  d_lj1_ss = k_lj1_ss.template view<DeviceType>();
  d_lj2_ss = k_lj2_ss.template view<DeviceType>();
  d_cutsq_ss_ast = k_cutsq_ss_ast.template view<DeviceType>();
  d_cutsq_ss_c = k_cutsq_ss_c.template view<DeviceType>();

  d_epsilon_sb = k_epsilon_sb.template view<DeviceType>();
  d_sigma_sb = k_sigma_sb.template view<DeviceType>();
  d_cut_sb_ast = k_cut_sb_ast.template view<DeviceType>();
  d_b_sb = k_b_sb.template view<DeviceType>();
  d_cut_sb_c = k_cut_sb_c.template view<DeviceType>();
  d_lj1_sb = k_lj1_sb.template view<DeviceType>();
  d_lj2_sb = k_lj2_sb.template view<DeviceType>();
  d_cutsq_sb_ast = k_cutsq_sb_ast.template view<DeviceType>();
  d_cutsq_sb_c = k_cutsq_sb_c.template view<DeviceType>();

  d_epsilon_bb = k_epsilon_bb.template view<DeviceType>();
  d_sigma_bb = k_sigma_bb.template view<DeviceType>();
  d_cut_bb_ast = k_cut_bb_ast.template view<DeviceType>();
  d_b_bb = k_b_bb.template view<DeviceType>();
  d_cut_bb_c = k_cut_bb_c.template view<DeviceType>();
  d_lj1_bb = k_lj1_bb.template view<DeviceType>();
  d_lj2_bb = k_lj2_bb.template view<DeviceType>();
  d_cutsq_bb_ast = k_cutsq_bb_ast.template view<DeviceType>();
  d_cutsq_bb_c = k_cutsq_bb_c.template view<DeviceType>();

  d_nx = k_nx.template view<DeviceType>();
  d_ny = k_ny.template view<DeviceType>();
  d_nz = k_nz.template view<DeviceType>();
  h_nx = k_nx.h_view;
  h_ny = k_ny.h_view;
  h_nz = k_nz.h_view;

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
  //PairOxdnaExcv::init_style();
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

  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;

  k_epsilon_ss.h_view(i,j) = k_epsilon_ss.h_view(j,i) = epsilon_ss[i][j];
  k_sigma_ss.h_view(i,j) = k_sigma_ss.h_view(j,i) = sigma_ss[i][j];
  k_cut_ss_ast.h_view(i,j) = k_cut_ss_ast.h_view(j,i) = cut_ss_ast[i][j];
  k_b_ss.h_view(i,j) = k_b_ss.h_view(j,i) = b_ss[i][j];
  k_cut_ss_c.h_view(i,j) = k_cut_ss_c.h_view(j,i) = cut_ss_c[i][j];
  k_lj1_ss.h_view(i,j) = k_lj1_ss.h_view(j,i) = lj1_ss[i][j];
  k_lj2_ss.h_view(i,j) = k_lj2_ss.h_view(j,i) = lj2_ss[i][j];
  k_cutsq_ss_ast.h_view(i,j) = k_cutsq_ss_ast.h_view(j,i) = cutsq_ss_ast[i][j];
  k_cutsq_ss_c.h_view(i,j) = k_cutsq_ss_c.h_view(j,i) = cutsq_ss_c[i][j];

  k_epsilon_sb.h_view(i,j) = k_epsilon_sb.h_view(j,i) = epsilon_sb[i][j];
  k_sigma_sb.h_view(i,j) = k_sigma_sb.h_view(j,i) = sigma_sb[i][j];
  k_cut_sb_ast.h_view(i,j) = k_cut_sb_ast.h_view(j,i) = cut_sb_ast[i][j];
  k_b_sb.h_view(i,j) = k_b_sb.h_view(j,i) = b_sb[i][j];
  k_cut_sb_c.h_view(i,j) = k_cut_sb_c.h_view(j,i) = cut_sb_c[i][j];
  k_lj1_sb.h_view(i,j) = k_lj1_sb.h_view(j,i) = lj1_sb[i][j];
  k_lj2_sb.h_view(i,j) = k_lj2_sb.h_view(j,i) = lj2_sb[i][j];
  k_cutsq_sb_ast.h_view(i,j) = k_cutsq_sb_ast.h_view(j,i) = cutsq_sb_ast[i][j];
  k_cutsq_sb_c.h_view(i,j) = k_cutsq_sb_c.h_view(j,i) = cutsq_sb_c[i][j];

  k_epsilon_bb.h_view(i,j) = k_epsilon_bb.h_view(j,i) = epsilon_bb[i][j];
  k_sigma_bb.h_view(i,j) = k_sigma_bb.h_view(j,i) = sigma_bb[i][j];
  k_cut_bb_ast.h_view(i,j) = k_cut_bb_ast.h_view(j,i) = cut_bb_ast[i][j];
  k_b_bb.h_view(i,j) = k_b_bb.h_view(j,i) = b_bb[i][j];
  k_cut_bb_c.h_view(i,j) = k_cut_bb_c.h_view(j,i) = cut_bb_c[i][j];
  k_lj1_bb.h_view(i,j) = k_lj1_bb.h_view(j,i) = lj1_bb[i][j];
  k_lj2_bb.h_view(i,j) = k_lj2_bb.h_view(j,i) = lj2_bb[i][j];
  k_cutsq_bb_ast.h_view(i,j) = k_cutsq_bb_ast.h_view(j,i) = cutsq_bb_ast[i][j];
  k_cutsq_bb_c.h_view(i,j) = k_cutsq_bb_c.h_view(j,i) = cutsq_bb_c[i][j];

  return cutone;

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
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
int PairOxdnaExcvKokkos<DeviceType>::sbmask(const int& j) const {
  return j >> SBBITS & 3;
}


namespace LAMMPS_NS {
template class PairOxdnaExcvKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairOxdnaExcvKokkos<LMPHostType>;
#endif
}