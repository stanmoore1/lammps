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
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include "pair_oxdna_excv_kokkos.h"

using namespace LAMMPS_NS;

// TODO: remove NEIGHFLAG from stk_kokkos - not needed due to bondlist

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairOxdnaStkKokkos<DeviceType>::PairOxdnaStkKokkos(LAMMPS *lmp) : PairOxdnaStk(lmp) , mfOxdnaKokkos<DeviceType>(lmp)
{
  mfOxdnaKokkos<DeviceType> instance(lmp);
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  neighborKK = (NeighborKokkos *) neighbor;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | ELLIPSOID_MASK | BONUS_MASK | F_MASK | 
                  TORQUE_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairOxdnaStkKokkos<DeviceType>::~PairOxdnaStkKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);

    memoryKK->destroy_kokkos(k_epsilon_st,epsilon_st);
    memoryKK->destroy_kokkos(k_a_st,a_st);
    memoryKK->destroy_kokkos(k_cut_st_0,cut_st_0);
    memoryKK->destroy_kokkos(k_cut_st_c,cut_st_c);
    memoryKK->destroy_kokkos(k_cut_st_lo,cut_st_lo);
    memoryKK->destroy_kokkos(k_cut_st_hi,cut_st_hi);
    memoryKK->destroy_kokkos(k_cut_st_lc,cut_st_lc);
    memoryKK->destroy_kokkos(k_cut_st_hc,cut_st_hc);
    memoryKK->destroy_kokkos(k_b_st_lo,b_st_lo);
    memoryKK->destroy_kokkos(k_b_st_hi,b_st_hi);
    memoryKK->destroy_kokkos(k_shift_st,shift_st);
    memoryKK->destroy_kokkos(k_cutsq_st_hc,cutsq_st_hc);

    memoryKK->destroy_kokkos(k_a_st4,a_st4);
    memoryKK->destroy_kokkos(k_theta_st4_0,theta_st4_0);
    memoryKK->destroy_kokkos(k_dtheta_st4_ast,dtheta_st4_ast);
    memoryKK->destroy_kokkos(k_b_st4,b_st4);
    memoryKK->destroy_kokkos(k_dtheta_st4_c,dtheta_st4_c);

    memoryKK->destroy_kokkos(k_a_st5,a_st5);
    memoryKK->destroy_kokkos(k_theta_st5_0,theta_st5_0);
    memoryKK->destroy_kokkos(k_dtheta_st5_ast,dtheta_st5_ast);
    memoryKK->destroy_kokkos(k_b_st5,b_st5);
    memoryKK->destroy_kokkos(k_dtheta_st5_c,dtheta_st5_c);

    memoryKK->destroy_kokkos(k_a_st6,a_st6);
    memoryKK->destroy_kokkos(k_theta_st6_0,theta_st6_0);
    memoryKK->destroy_kokkos(k_dtheta_st6_ast,dtheta_st6_ast);
    memoryKK->destroy_kokkos(k_b_st6,b_st6);
    memoryKK->destroy_kokkos(k_dtheta_st6_c,dtheta_st6_c);

    memoryKK->destroy_kokkos(k_a_st1,a_st1);
    memoryKK->destroy_kokkos(k_cosphi_st1_ast,cosphi_st1_ast);
    memoryKK->destroy_kokkos(k_b_st1,b_st1);
    memoryKK->destroy_kokkos(k_cosphi_st1_c,cosphi_st1_c);
    memoryKK->destroy_kokkos(k_a_st2,a_st2);
    memoryKK->destroy_kokkos(k_cosphi_st2_ast,cosphi_st2_ast);
    memoryKK->destroy_kokkos(k_b_st2,b_st2);
    memoryKK->destroy_kokkos(k_cosphi_st2_c,cosphi_st2_c);

    memoryKK->destroy_kokkos(k_nx_xtrct,nx_xtrct);
    memoryKK->destroy_kokkos(k_ny_xtrct,ny_xtrct);
    memoryKK->destroy_kokkos(k_nz_xtrct,nz_xtrct);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaStkKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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

  k_epsilon_st.template sync<DeviceType>();
  k_a_st.template sync<DeviceType>();
  k_cut_st_0.template sync<DeviceType>();
  k_cut_st_c.template sync<DeviceType>();
  k_cut_st_lo.template sync<DeviceType>();
  k_cut_st_hi.template sync<DeviceType>();
  k_cut_st_lc.template sync<DeviceType>();
  k_cut_st_hc.template sync<DeviceType>();
  k_b_st_lo.template sync<DeviceType>();
  k_b_st_hi.template sync<DeviceType>();
  k_shift_st.template sync<DeviceType>();
  k_cutsq_st_hc.template sync<DeviceType>();

  k_a_st4.template sync<DeviceType>();
  k_theta_st4_0.template sync<DeviceType>();
  k_dtheta_st4_ast.template sync<DeviceType>();
  k_b_st4.template sync<DeviceType>();
  k_dtheta_st4_c.template sync<DeviceType>();

  k_a_st5.template sync<DeviceType>();
  k_theta_st5_0.template sync<DeviceType>();
  k_dtheta_st5_ast.template sync<DeviceType>();
  k_b_st5.template sync<DeviceType>();
  k_dtheta_st5_c.template sync<DeviceType>();

  k_a_st6.template sync<DeviceType>();
  k_theta_st6_0.template sync<DeviceType>();
  k_dtheta_st6_ast.template sync<DeviceType>();
  k_b_st6.template sync<DeviceType>();
  k_dtheta_st6_c.template sync<DeviceType>();

  k_a_st1.template sync<DeviceType>();
  k_cosphi_st1_ast.template sync<DeviceType>();
  k_b_st1.template sync<DeviceType>();
  k_cosphi_st1_c.template sync<DeviceType>();
  k_a_st2.template sync<DeviceType>();
  k_cosphi_st2_ast.template sync<DeviceType>();
  k_b_st2.template sync<DeviceType>();
  k_cosphi_st2_c.template sync<DeviceType>();

  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK | TORQUE_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  bondlist = neighborKK->k_bondlist.view<DeviceType>();
  id5p = atomKK->k_id5p.view<DeviceType>();

  nlocal = atom->nlocal;
  newton_bond = force->newton_bond;
  neighborKK->k_bondlist.template sync<DeviceType>();
  nbondlist = neighborKK->nbondlist;

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

  // d_n(x/y/z)_xtrct = extracted local unit vectors in lab frame from oxdna_excv/kk
  auto oxdna_excvKK = dynamic_cast<PairOxdnaExcvKokkos<DeviceType> *>(force->pair_match("oxdna/excv/kk", 1, 1));
  d_nx_xtrct = oxdna_excvKK->k_nx.template view<DeviceType>();
  d_ny_xtrct = oxdna_excvKK->k_ny.template view<DeviceType>();
  d_nz_xtrct = oxdna_excvKK->k_nz.template view<DeviceType>();

  // loop over neighbors of my atoms for compute functors

  EV_FLOAT ev;

  if (evflag) {
    if (neighflag == HALF) {
      if (newton_bond) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<HALF,1,1> >(0,nbondlist),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<HALF,0,1> >(0,nbondlist),*this,ev);
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_bond) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<HALFTHREAD,1,1> >(0,nbondlist),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<HALFTHREAD,0,1> >(0,nbondlist),*this,ev);
      }
    } else if (neighflag == FULL) {
      if (newton_bond) {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<FULL,1,1> >(0,nbondlist),*this,ev);
      } else {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<FULL,0,1> >(0,nbondlist),*this,ev);
      }
    }
  } else {
    if (neighflag == HALF) {
      if (newton_bond) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<HALF,1,0> >(0,nbondlist),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<HALF,0,0> >(0,nbondlist),*this);
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_bond) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<HALFTHREAD,1,0> >(0,nbondlist),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<HALFTHREAD,0,0> >(0,nbondlist),*this);
      }
    } else if (neighflag == FULL) {
      if (newton_bond) {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<FULL,1,0> >(0,nbondlist),*this);
      } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaStkCompute<FULL,0,0> >(0,nbondlist),*this);
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
template<int NEIGHFLAG, int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaStkKokkos<DeviceType>::operator()(TagPairOxdnaStkCompute<NEIGHFLAG,NEWTON_BOND,EVFLAG>, \
  const int &in, EV_FLOAT &ev) const
{
  // f and torque array are duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();
  auto v_torque = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,\
    decltype(dup_torque),decltype(ndup_torque)>::get(dup_torque,ndup_torque);
  auto a_torque = v_torque.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  int b = bondlist(in,0);
  int a = bondlist(in,1);
  int btemp, atype, btype;

  // directionality test: a -> b is 3' -> 5'
  if ( tag(b) != id5p(a) ) {
    btemp = b;
    b = a;
    a = btemp;  
  }

  F_FLOAT ra_cst[3], rb_cst[3];           // vectors COM-stacking sites in lab frame
  F_FLOAT ra_cs[3], rb_cs[3];             // vectors COM-backbone sites in lab frame

  F_FLOAT delf[3], delta[3], deltb[3];    // force, torque increment
  F_FLOAT evdwl,finc,tpair;               
  F_FLOAT delr_ss[3],delr_ss_norm[3],rsq_ss,r_ss,rinv_ss;
  F_FLOAT delr_st[3],delr_st_norm[3],rsq_st,r_st,rinv_st;
  F_FLOAT theta4,t4dir[3],cost4;
  F_FLOAT theta5p,t5pdir[3],cost5p;
  F_FLOAT theta6p,t6pdir[3],cost6p;
  F_FLOAT cosphi1,cosphi2,cosphi1dir[3],cosphi2dir[3];

  F_FLOAT f1,f4t4,f4t5,f4t6,f5c1,f5c2;
  F_FLOAT df1,df4t4,df4t5,df4t6,df5c1,df5c2;

  // vector COM [a/b] - stacking site [a/b]
  constexpr F_FLOAT d_cst = +0.34;
  ra_cst[0] = d_cst * d_nx_xtrct(a,0);
  ra_cst[1] = d_cst * d_nx_xtrct(a,1);
  ra_cst[2] = d_cst * d_nx_xtrct(a,2);
  rb_cst[0] = d_cst * d_nx_xtrct(b,0);
  rb_cst[1] = d_cst * d_nx_xtrct(b,1);
  rb_cst[2] = d_cst * d_nx_xtrct(b,2);

  // vector stacking site a to b
  delr_st[0] = x(b,0) + rb_cst[0] - x(a,0) - ra_cst[0];
  delr_st[1] = x(b,1) + rb_cst[1] - x(a,1) - ra_cst[1];
  delr_st[2] = x(b,2) + rb_cst[2] - x(a,2) - ra_cst[2];

  atype = type(a);
  btype = type(b);

  rsq_st = delr_st[0]*delr_st[0] + delr_st[1]*delr_st[1] + delr_st[2]*delr_st[2];
  r_st = sqrt(rsq_st);
  rinv_st = 1.0/r_st;

  delr_st_norm[0] = delr_st[0] * rinv_st;
  delr_st_norm[1] = delr_st[1] * rinv_st;
  delr_st_norm[2] = delr_st[2] * rinv_st;

  // vector COM [a/b] - backbone site [a/b]
  constexpr F_FLOAT d_cs = -0.4;
  ra_cs[0] = d_cs * d_nx_xtrct(a,0);
  ra_cs[1] = d_cs * d_nx_xtrct(a,1);
  ra_cs[2] = d_cs * d_nx_xtrct(a,2);
  rb_cs[0] = d_cs * d_nx_xtrct(b,0);
  rb_cs[1] = d_cs * d_nx_xtrct(b,1);
  rb_cs[2] = d_cs * d_nx_xtrct(b,2);

  // vector backbone site a to b
  delr_ss[0] = x(b,0) + rb_cs[0] - x(a,0) - ra_cs[0];
  delr_ss[1] = x(b,1) + rb_cs[1] - x(a,1) - ra_cs[1];
  delr_ss[2] = x(b,2) + rb_cs[2] - x(a,2) - ra_cs[2];

  rsq_ss = delr_ss[0]*delr_ss[0] + delr_ss[1]*delr_ss[1] + delr_ss[2]*delr_ss[2];
  r_ss = sqrt(rsq_ss);
  rinv_ss = 1.0/r_ss;

  delr_ss_norm[0] = delr_ss[0] * rinv_ss;
  delr_ss_norm[1] = delr_ss[1] * rinv_ss;
  delr_ss_norm[2] = delr_ss[2] * rinv_ss;

  // beginning of modulation factors

  // f1 = F1
  // if (r_st > d_cut_st_hc(atype, btype)) {
  //   f1 = 0.0;
  // } else if (r_st > d_cut_st_hi(atype, btype)) {
  //   f1 = d_epsilon_st(atype, btype) * d_b_st_hi(atype, btype) * (r_st - d_cut_st_hc(atype, btype)) * (r_st - d_cut_st_hc(atype, btype));
  // } else if (r_st > d_cut_st_lo(atype, btype)) {
  //   double tmp = 1 - exp(-(r_st - d_cut_st_0(atype, btype)) * d_a_st(atype, btype));
  //   f1 = d_epsilon_st(atype, btype) * tmp * tmp - d_shift_st(atype, btype);
  // } else if (r_st > d_cut_st_lc(atype, btype)) {
  //   f1 = d_epsilon_st(atype, btype) * d_b_st_lo(atype, btype) * (r_st - d_cut_st_lc(atype, btype)) * (r_st - d_cut_st_lc(atype, btype));
  // } else {
  //   f1 = 0.0;
  // }
  this->oxDNA_F1_KK(r_st, d_epsilon_st(atype, btype), d_a_st(atype, btype), d_cut_st_0(atype, btype),
          d_cut_st_lc(atype, btype), d_cut_st_hc(atype, btype), d_cut_st_lo(atype, btype), d_cut_st_hi(atype, btype),
          d_b_st_lo(atype, btype), d_b_st_hi(atype, btype), d_shift_st(atype, btype), f1);

  // start early rejection criterium
  if (f1) {
    // theta4 angle and correction
    cost4 = d_nz_xtrct(b,0) * d_nz_xtrct(a,0) + 
            d_nz_xtrct(b,1) * d_nz_xtrct(a,1) + 
            d_nz_xtrct(b,2) * d_nz_xtrct(a,2);
    if (cost4 > 1.0) cost4 = 1.0;
    if (cost4 < -1.0) cost4 = -1.0;
    theta4 = acos(cost4);
    // f4t4 = F4 
    double dtheta = theta4 - d_theta_st4_0(atype, btype);
    if (fabs(dtheta) > d_dtheta_st4_c(atype, btype)) {
      f4t4 = 0.0;
    } else if (dtheta > d_dtheta_st4_ast(atype, btype)) {
      f4t4 = d_b_st4(atype, btype) * (dtheta - d_dtheta_st4_c(atype, btype)) * (dtheta - d_dtheta_st4_c(atype, btype));
    } else if (dtheta > -d_dtheta_st4_ast(atype, btype)) {
      f4t4 = 1 - d_a_st4(atype, btype) * dtheta * dtheta;
    } else {
      f4t4 = d_b_st4(atype, btype) * (dtheta + d_dtheta_st4_c(atype, btype)) * (dtheta + d_dtheta_st4_c(atype, btype));
    }

  // early rejection criterium
  if (f4t4) {
    // theta5 angle and correction
    cost5p = d_nz_xtrct(b,0) * delr_st_norm[0] + 
             d_nz_xtrct(b,1) * delr_st_norm[1] + 
             d_nz_xtrct(b,2) * delr_st_norm[2];
    if (cost5p > 1.0) cost5p = 1.0;
    if (cost5p < -1.0) cost5p = -1.0;
    theta5p = acos(cost5p);
    // f4t5 = F4
    dtheta = theta5p - d_theta_st5_0(atype, btype);
    if (fabs(dtheta) > d_dtheta_st5_c(atype, btype)) {
      f4t5 = 0.0;
    } else if (dtheta > d_dtheta_st5_ast(atype, btype)) {
      f4t5 = d_b_st5(atype, btype) * (dtheta - d_dtheta_st5_c(atype, btype)) * (dtheta - d_dtheta_st5_c(atype, btype));
    } else if (dtheta > -d_dtheta_st5_ast(atype, btype)) {
      f4t5 = 1 - d_a_st5(atype, btype) * dtheta * dtheta;
    } else {
      f4t5 = d_b_st5(atype, btype) * (dtheta + d_dtheta_st5_c(atype, btype)) * (dtheta + d_dtheta_st5_c(atype, btype));
    }

  // early rejection criterium
  if (f4t5) {
    // theta6 angle and correction
    cost6p = delr_st_norm[0] * d_nz_xtrct(a,0) + 
             delr_st_norm[1] * d_nz_xtrct(a,1) + 
             delr_st_norm[2] * d_nz_xtrct(a,2);
    if (cost6p > 1.0) cost6p = 1.0;
    if (cost6p < -1.0) cost6p = -1.0;
    theta6p = acos(cost6p);
    // cosphi1 and cosphi2 angles
    cosphi1 = delr_ss_norm[0] * d_ny_xtrct(b,0) + 
              delr_ss_norm[1] * d_ny_xtrct(b,1) + 
              delr_ss_norm[2] * d_ny_xtrct(b,2);
    cosphi2 = delr_ss_norm[0] * d_ny_xtrct(a,0) +
              delr_ss_norm[1] * d_ny_xtrct(a,1) +
              delr_ss_norm[2] * d_ny_xtrct(a,2);
    if (cosphi1 > 1.0) cosphi1 = 1.0;
    if (cosphi1 < -1.0) cosphi1 = -1.0;
    if (cosphi2 > 1.0) cosphi2 = 1.0;
    if (cosphi2 < -1.0) cosphi2 = -1.0;
    // f4t6 = F4
    dtheta = theta6p - d_theta_st6_0(atype, btype);
    if (fabs(dtheta) > d_dtheta_st6_c(atype, btype)) {
      f4t6 = 0.0;
    } else if (dtheta > d_dtheta_st6_ast(atype, btype)) {
      f4t6 = d_b_st6(atype, btype) * (dtheta - d_dtheta_st6_c(atype, btype)) * (dtheta - d_dtheta_st6_c(atype, btype));
    } else if (dtheta > -d_dtheta_st6_ast(atype, btype)) {
      f4t6 = 1 - d_a_st6(atype, btype) * dtheta * dtheta;
    } else {
      f4t6 = d_b_st6(atype, btype) * (dtheta + d_dtheta_st6_c(atype, btype)) * (dtheta + d_dtheta_st6_c(atype, btype));
    }
    // f5c1 = F5
    if (-cosphi1 >= 0) {
      f5c1 = 1.0;
    } else if (-cosphi1 > d_cosphi_st1_ast(atype, btype)) {
      f5c1 = 1 - d_a_st1(atype, btype) * (-cosphi1) * (-cosphi1);
    } else if (-cosphi1 > d_cosphi_st1_c(atype, btype)) {
      f5c1 = d_b_st1(atype, btype) * (-cosphi1 - d_cosphi_st1_c(atype, btype)) * (-cosphi1 - d_cosphi_st1_c(atype, btype));
    } else {
      f5c1 = 0.0;
    }
    // f5c2 = F5
    if (-cosphi2 >= 0) {
      f5c2 = 1.0;
    } else if (-cosphi2 > d_cosphi_st2_ast(atype, btype)) {
      f5c2 = 1 - d_a_st2(atype, btype) * (-cosphi2) * (-cosphi2);
    } else if (-cosphi2 > d_cosphi_st2_c(atype, btype)) {
      f5c2 = d_b_st2(atype, btype) * (-cosphi2 - d_cosphi_st2_c(atype, btype)) * (-cosphi2 - d_cosphi_st2_c(atype, btype));
    } else {
      f5c2 = 0.0;
    }
    evdwl = f1 * f4t4 * f4t5 * f4t6 * f5c1 * f5c2;
  
  // early rejection criterium
  if (evdwl) {
    // df1 = DF1
    if (r_st > d_cut_st_hc(atype, btype)) {
      df1 = 0.0;
    } else if (r_st > d_cut_st_hi(atype, btype)) {
      df1 = 2 * d_epsilon_st(atype, btype) * d_b_st_hi(atype, btype) * (1 - d_cut_st_hc(atype, btype) / r_st);
    } else if (r_st > d_cut_st_lo(atype, btype)) {
      double tmp = exp(-(r_st - d_cut_st_0(atype, btype)) * d_a_st(atype, btype));
      df1 = 2 * d_epsilon_st(atype, btype) * (1 - tmp) * tmp * d_a_st(atype, btype) / r_st;
    } else if (r_st > d_cut_st_lc(atype, btype)) {
      df1 = 2 * d_epsilon_st(atype, btype) * d_b_st_lo(atype, btype) * (1 - d_cut_st_lc(atype, btype) / r_st);
    } else {
      df1 = 0.0;
    }
    // df4t4 = DF4
    double dtheta = theta4 - d_theta_st4_0(atype, btype);
    if (fabs(dtheta) > d_dtheta_st4_c(atype, btype)) {
      df4t4 = 0.0;
    } else if (dtheta > d_dtheta_st4_ast(atype, btype)) {
      df4t4 = 2 * d_b_st4(atype, btype) * (dtheta - d_dtheta_st4_c(atype, btype));
    } else if (dtheta > -d_dtheta_st4_ast(atype, btype)) {
      df4t4 = -2 * d_a_st4(atype, btype) * dtheta;
    } else {
      df4t4 = 2 * d_b_st4(atype, btype) * (dtheta + d_dtheta_st4_c(atype, btype));
    }
    df4t4 /= sin(theta4); // TODO: check if this is correct (see original code in mf_oxdna.h)
    // df4t5 = DF4
    dtheta = theta5p - d_theta_st5_0(atype, btype);
    if (fabs(dtheta) > d_dtheta_st5_c(atype, btype)) {
      df4t5 = 0.0;
    } else if (dtheta > d_dtheta_st5_ast(atype, btype)) {
      df4t5 = 2 * d_b_st5(atype, btype) * (dtheta - d_dtheta_st5_c(atype, btype));
    } else if (dtheta > -d_dtheta_st5_ast(atype, btype)) {
      df4t5 = -2 * d_a_st5(atype, btype) * dtheta;
    } else {
      df4t5 = 2 * d_b_st5(atype, btype) * (dtheta + d_dtheta_st5_c(atype, btype));
    }
    df4t5 /= sin(theta5p); // TODO: check if this is correct (see original code in mf_oxdna.h)
    // df4t6 = DF4
    dtheta = theta6p - d_theta_st6_0(atype, btype);
    if (fabs(dtheta) > d_dtheta_st6_c(atype, btype)) {
      df4t6 = 0.0;
    } else if (dtheta > d_dtheta_st6_ast(atype, btype)) {
      df4t6 = 2 * d_b_st6(atype, btype) * (dtheta - d_dtheta_st6_c(atype, btype));
    } else if (dtheta > -d_dtheta_st6_ast(atype, btype)) {
      df4t6 = -2 * d_a_st6(atype, btype) * dtheta;
    } else {
      df4t6 = 2 * d_b_st6(atype, btype) * (dtheta + d_dtheta_st6_c(atype, btype));
    }
    df4t6 /= sin(theta6p); // TODO: check if this is correct (see original code in mf_oxdna.h)
    // df5c1 = DF5
    if (-cosphi1 >= 0) {
      df5c1 = 0.0;
    } else if (-cosphi1 > d_cosphi_st1_ast(atype, btype)) {
      df5c1 = -2 * d_a_st1(atype, btype) * (-cosphi1);
    } else if (-cosphi1 > d_cosphi_st1_c(atype, btype)) {
      df5c1 = 2 * d_b_st1(atype, btype) * (-cosphi1 - d_cosphi_st1_c(atype, btype));
    } else {
      df5c1 = 0.0;
    }
    // df5c2 = DF5
    if (-cosphi2 >= 0) {
      df5c2 = 0.0;
    } else if (-cosphi2 > d_cosphi_st2_ast(atype, btype)) {
      df5c2 = -2 * d_a_st2(atype, btype) * (-cosphi2);
    } else if (-cosphi2 > d_cosphi_st2_c(atype, btype)) {
      df5c2 = 2 * d_b_st2(atype, btype) * (-cosphi2 - d_cosphi_st2_c(atype, btype));
    } else {
      df5c2 = 0.0;
    }

    // force, torque and virial contribution for forces between stacking sites
    delf[0] = 0.0;
    delf[1] = 0.0;
    delf[2] = 0.0;
    delta[0] = 0.0;
    delta[1] = 0.0;
    delta[2] = 0.0;
    deltb[0] = 0.0;
    deltb[1] = 0.0;
    deltb[2] = 0.0;

    // radial force
    finc = -df1 * f4t4 * f4t5 * f4t6 * f5c1 * f5c2;

    delf[0] += delr_st[0] * finc;
    delf[1] += delr_st[1] * finc;
    delf[2] += delr_st[2] * finc;

    // theta5p force
    if (theta5p) {
      finc = -f1 * f4t4 * df4t5 * f4t6 * f5c1 * f5c2 * rinv_st;

      delf[0] += (delr_st_norm[0]*cost5p - d_nz_xtrct(b,0)) * finc;
      delf[1] += (delr_st_norm[1]*cost5p - d_nz_xtrct(b,1)) * finc;
      delf[2] += (delr_st_norm[2]*cost5p - d_nz_xtrct(b,2)) * finc;
    }

    // theta6p force
    if (theta6p) {
      finc = -f1 * f4t4 * f4t5 * df4t6 * f5c1 * f5c2 * rinv_st;

      delf[0] += (delr_st_norm[0]*cost6p - d_nz_xtrct(a,0)) * finc;
      delf[1] += (delr_st_norm[1]*cost6p - d_nz_xtrct(a,1)) * finc;
      delf[2] += (delr_st_norm[2]*cost6p - d_nz_xtrct(a,2)) * finc;
    }

    // increment forces and torques
    if ( NEWTON_BOND || a < nlocal ) {
      a_f(a,0) -= delf[0];
      a_f(a,1) -= delf[1];
      a_f(a,2) -= delf[2];
      delta[0] = ra_cst[1]*delf[2] - ra_cst[2]*delf[1];
      delta[1] = ra_cst[2]*delf[0] - ra_cst[0]*delf[2];
      delta[2] = ra_cst[0]*delf[1] - ra_cst[1]*delf[0];
      a_torque(a,0) -= delta[0];
      a_torque(a,1) -= delta[1];
      a_torque(a,2) -= delta[2];
    }
    if ( NEWTON_BOND || b < nlocal ) {
      a_f(b,0) += delf[0];
      a_f(b,1) += delf[1];
      a_f(b,2) += delf[2];
      deltb[0] = rb_cst[1]*delf[2] - rb_cst[2]*delf[1];
      deltb[1] = rb_cst[2]*delf[0] - rb_cst[0]*delf[2];
      deltb[2] = rb_cst[0]*delf[1] - rb_cst[1]*delf[0];
      a_torque(b,0) += deltb[0];
      a_torque(b,1) += deltb[1];
      a_torque(b,2) += deltb[2];
    }

    if (EVFLAG) { ev_tally_xyz(ev, a, b, nlocal, NEWTON_BOND, evdwl, delf[0], delf[1], delf[2], \
      x(b,0)-x(a,0), x(b,1)-x(a,1), x(b,2)-x(a,2)); }

    // force, torque and virial contribution for forces between backbone sites
    delf[0] = 0.0;
    delf[1] = 0.0;
    delf[2] = 0.0;
    delta[0] = 0.0;
    delta[1] = 0.0;
    delta[2] = 0.0;
    deltb[0] = 0.0;
    deltb[1] = 0.0;
    deltb[2] = 0.0;

    // cosphi1 force
    if (cosphi1) {
      finc = -f1 * f4t4 * f4t5 * f4t6 * df5c1 * f5c2 * rinv_ss;

      delf[0] += (delr_ss_norm[0]*cosphi1 - d_ny_xtrct(b,0)) * finc;
      delf[1] += (delr_ss_norm[1]*cosphi1 - d_ny_xtrct(b,1)) * finc;
      delf[2] += (delr_ss_norm[2]*cosphi1 - d_ny_xtrct(b,2)) * finc;
    }

    // cosphi2 force
    if (cosphi2) {
      finc = -f1 * f4t4 * f4t5 * f4t6 * f5c1 * df5c2 * rinv_ss;

      delf[0] += (delr_ss_norm[0]*cosphi2 - d_ny_xtrct(a,0)) * finc;
      delf[1] += (delr_ss_norm[1]*cosphi2 - d_ny_xtrct(a,1)) * finc;
      delf[2] += (delr_ss_norm[2]*cosphi2 - d_ny_xtrct(a,2)) * finc;
    }

    // increment forces and torques
    if ( NEWTON_BOND || a < nlocal ) {
      a_f(a,0) -= delf[0];
      a_f(a,1) -= delf[1];
      a_f(a,2) -= delf[2];
      delta[0] = ra_cs[1]*delf[2] - ra_cs[2]*delf[1];
      delta[1] = ra_cs[2]*delf[0] - ra_cs[0]*delf[2];
      delta[2] = ra_cs[0]*delf[1] - ra_cs[1]*delf[0];
      a_torque(a,0) -= delta[0];
      a_torque(a,1) -= delta[1];
      a_torque(a,2) -= delta[2];
    }
    if ( NEWTON_BOND || b < nlocal ) {
      a_f(b,0) += delf[0];
      a_f(b,1) += delf[1];
      a_f(b,2) += delf[2];
      deltb[0] = rb_cs[1]*delf[2] - rb_cs[2]*delf[1];
      deltb[1] = rb_cs[2]*delf[0] - rb_cs[0]*delf[2];
      deltb[2] = rb_cs[0]*delf[1] - rb_cs[1]*delf[0];
      a_torque(b,0) += deltb[0];
      a_torque(b,1) += deltb[1];
      a_torque(b,2) += deltb[2];
    }
    
    // increment viral only
    if (EVFLAG) { ev_tally_xyz(ev, a, b, nlocal, NEWTON_BOND, 0.0, delf[0], delf[1], delf[2], \
      x(b,0)-x(a,0), x(b,1)-x(a,1), x(b,2)-x(a,2)); }

    // pure torques not expressible as r x f

    delta[0] = 0.0;
    delta[1] = 0.0;
    delta[2] = 0.0;
    deltb[0] = 0.0;
    deltb[1] = 0.0;
    deltb[2] = 0.0;

    // theta4 torque
    if (theta4) {
      tpair = -f1 * df4t4 * f4t5 * f4t6 * f5c1 * f5c2;
      t4dir[0] = d_nz_xtrct(a,1) * d_nz_xtrct(b,2) - d_nz_xtrct(a,2) * d_nz_xtrct(b,1);
      t4dir[1] = d_nz_xtrct(a,2) * d_nz_xtrct(b,0) - d_nz_xtrct(a,0) * d_nz_xtrct(b,2);
      t4dir[2] = d_nz_xtrct(a,0) * d_nz_xtrct(b,1) - d_nz_xtrct(a,1) * d_nz_xtrct(b,0);
      delta[0] += t4dir[0] * tpair;
      delta[1] += t4dir[1] * tpair;
      delta[2] += t4dir[2] * tpair;
      deltb[0] += t4dir[0] * tpair;
      deltb[1] += t4dir[1] * tpair;
      deltb[2] += t4dir[2] * tpair;
    }

    // theta5p torque
    if (theta5p) {
      tpair = -f1 * f4t4 * df4t5 * f4t6 * f5c1 * f5c2;
      t5pdir[0] = delr_st_norm[1] * d_nz_xtrct(b,2) - delr_st_norm[2] * d_nz_xtrct(b,1);
      t5pdir[1] = delr_st_norm[2] * d_nz_xtrct(b,0) - delr_st_norm[0] * d_nz_xtrct(b,2);
      t5pdir[2] = delr_st_norm[0] * d_nz_xtrct(b,1) - delr_st_norm[1] * d_nz_xtrct(b,0);
      deltb[0] += t5pdir[0] * tpair;
      deltb[1] += t5pdir[1] * tpair;
      deltb[2] += t5pdir[2] * tpair;
    }

    // theta6p torque
    if (theta6p) {
      tpair = -f1 * f4t4 * f4t5 * df4t6 * f5c1 * f5c2;
      t6pdir[0] = delr_st_norm[1] * d_nz_xtrct(a,2) - delr_st_norm[2] * d_nz_xtrct(a,1);
      t6pdir[1] = delr_st_norm[2] * d_nz_xtrct(a,0) - delr_st_norm[0] * d_nz_xtrct(a,2);
      t6pdir[2] = delr_st_norm[0] * d_nz_xtrct(a,1) - delr_st_norm[1] * d_nz_xtrct(a,0);
      delta[0] -= t6pdir[0] * tpair;
      delta[1] -= t6pdir[1] * tpair;
      delta[2] -= t6pdir[2] * tpair;
    }

    // cosphi1 torque
    if (cosphi1) {
      tpair = -f1 * f4t4 * f4t5 * f4t6 * df5c1 * f5c2;
      cosphi1dir[0] = delr_ss_norm[1] * d_ny_xtrct(b,2) - delr_ss_norm[2] * d_ny_xtrct(b,1);
      cosphi1dir[1] = delr_ss_norm[2] * d_ny_xtrct(b,0) - delr_ss_norm[0] * d_ny_xtrct(b,2);
      cosphi1dir[2] = delr_ss_norm[0] * d_ny_xtrct(b,1) - delr_ss_norm[1] * d_ny_xtrct(b,0);
      deltb[0] += cosphi1dir[0] * tpair;
      deltb[1] += cosphi1dir[1] * tpair;
      deltb[2] += cosphi1dir[2] * tpair;
    }

    // cosphi2 torque
    if (cosphi2) {
      tpair = -f1 * f4t4 * f4t5 * f4t6 * f5c1 * df5c2;
      cosphi2dir[0] = delr_ss_norm[1] * d_ny_xtrct(a,2) - delr_ss_norm[2] * d_ny_xtrct(a,1);
      cosphi2dir[1] = delr_ss_norm[2] * d_ny_xtrct(a,0) - delr_ss_norm[0] * d_ny_xtrct(a,2);
      cosphi2dir[2] = delr_ss_norm[0] * d_ny_xtrct(a,1) - delr_ss_norm[1] * d_ny_xtrct(a,0);
      delta[0] -= cosphi2dir[0] * tpair;
      delta[1] -= cosphi2dir[1] * tpair;
      delta[2] -= cosphi2dir[2] * tpair;
    }

    // increment torques
    if ( NEWTON_BOND || a < nlocal ) {
      a_torque(a,0) -= delta[0];
      a_torque(a,1) -= delta[1];
      a_torque(a,2) -= delta[2];
    }
    if ( NEWTON_BOND || b < nlocal ) {
      a_torque(b,0) += deltb[0];
      a_torque(b,1) += deltb[1];
      a_torque(b,2) += deltb[2];
    }
  // end of early rejection criterium:
  }    // evdwl
  }    // f4t5
  }    // f4t4
  }    // f1
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaStkKokkos<DeviceType>::operator()(TagPairOxdnaStkCompute<NEIGHFLAG,NEWTON_BOND,EVFLAG>, \
  const int &in) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,NEWTON_BOND,EVFLAG>\
  (TagPairOxdnaStkCompute<NEIGHFLAG,NEWTON_BOND,EVFLAG>(),in,ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaStkKokkos<DeviceType>::allocate()
{
  PairOxdnaStk::allocate();

  int n = atom->ntypes;
  
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

  k_epsilon_st.template modify<LMPHostType>();
  k_a_st.template modify<LMPHostType>();
  k_cut_st_0.template modify<LMPHostType>();
  k_cut_st_c.template modify<LMPHostType>();
  k_cut_st_lo.template modify<LMPHostType>();
  k_cut_st_hi.template modify<LMPHostType>();
  k_cut_st_lc.template modify<LMPHostType>();
  k_cut_st_hc.template modify<LMPHostType>();
  k_b_st_lo.template modify<LMPHostType>();
  k_b_st_hi.template modify<LMPHostType>();
  k_shift_st.template modify<LMPHostType>();
  k_cutsq_st_hc.template modify<LMPHostType>();

  k_a_st4.template modify<LMPHostType>();
  k_theta_st4_0.template modify<LMPHostType>();
  k_dtheta_st4_ast.template modify<LMPHostType>();
  k_b_st4.template modify<LMPHostType>();
  k_dtheta_st4_c.template modify<LMPHostType>();

  k_a_st5.template modify<LMPHostType>();
  k_theta_st5_0.template modify<LMPHostType>();
  k_dtheta_st5_ast.template modify<LMPHostType>();
  k_b_st5.template modify<LMPHostType>();
  k_dtheta_st5_c.template modify<LMPHostType>();

  k_a_st6.template modify<LMPHostType>();
  k_theta_st6_0.template modify<LMPHostType>();
  k_dtheta_st6_ast.template modify<LMPHostType>();
  k_b_st6.template modify<LMPHostType>();
  k_dtheta_st6_c.template modify<LMPHostType>();

  k_a_st1.template modify<LMPHostType>();
  k_cosphi_st1_ast.template modify<LMPHostType>();
  k_b_st1.template modify<LMPHostType>();
  k_cosphi_st1_c.template modify<LMPHostType>();
  k_a_st2.template modify<LMPHostType>();
  k_cosphi_st2_ast.template modify<LMPHostType>();
  k_b_st2.template modify<LMPHostType>();
  k_cosphi_st2_c.template modify<LMPHostType>();

  // "cutone" is "cut_st_hc[i][j]", sets the master list distance cutoff
  return cutone;

}

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators

   NOTE: Although this is a pair style interaction, the algorithm below
   follows the virial incrementation of the bond style. This is because
   the bond topology is used in the main compute loop.
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaStkKokkos<DeviceType>::ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,\
      const int &nlocal, const int &newton_bond, const F_FLOAT &evdwl,\
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,\
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
{
  E_FLOAT evdwlhalf;
  F_FLOAT v[6];

  // The eatom and vatom arrays are atomic
  Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,\
    typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > \
    v_eatom = k_eatom.view<DeviceType>();
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,\
    typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > \
    v_vatom = k_vatom.view<DeviceType>();

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) ev.evdwl += evdwl;
      else {
        evdwlhalf = 0.5*evdwl;
        if (i < nlocal) ev.evdwl += evdwlhalf;
        if (j < nlocal) ev.evdwl += evdwlhalf;
      }
    }
    if (eflag_atom) {
      evdwlhalf = 0.5*evdwl;
      if (newton_bond || i < nlocal) v_eatom[i] += evdwlhalf;
      if (newton_bond || j < nlocal) v_eatom[j] += evdwlhalf;
    }
  }

  if (vflag_either) {
    v[0] = delx * fx;
    v[1] = dely * fy;
    v[2] = delz * fz;
    v[3] = delx * fy;
    v[4] = delx * fz;
    v[5] = dely * fz;

    if (vflag_global) {
      if (newton_bond) {
        ev.v[0] += v[0];
        ev.v[1] += v[1];
        ev.v[2] += v[2];
        ev.v[3] += v[3];
        ev.v[4] += v[4];
        ev.v[5] += v[5];
      } else {
        if (i < nlocal) {
          ev.v[0] += 0.5*v[0];
          ev.v[1] += 0.5*v[1];
          ev.v[2] += 0.5*v[2];
          ev.v[3] += 0.5*v[3];
          ev.v[4] += 0.5*v[4];
          ev.v[5] += 0.5*v[5];
        }
        if (j < nlocal) {
          ev.v[0] += 0.5*v[0];
          ev.v[1] += 0.5*v[1];
          ev.v[2] += 0.5*v[2];
          ev.v[3] += 0.5*v[3];
          ev.v[4] += 0.5*v[4];
          ev.v[5] += 0.5*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i < nlocal) {
        v_vatom(i,0) += 0.5*v[0];
        v_vatom(i,1) += 0.5*v[1];
        v_vatom(i,2) += 0.5*v[2];
        v_vatom(i,3) += 0.5*v[3];
        v_vatom(i,4) += 0.5*v[4];
        v_vatom(i,5) += 0.5*v[5];
      }
      if (newton_bond || j < nlocal) {
        v_vatom(j,0) += 0.5*v[0];
        v_vatom(j,1) += 0.5*v[1];
        v_vatom(j,2) += 0.5*v[2];
        v_vatom(j,3) += 0.5*v[3];
        v_vatom(j,4) += 0.5*v[4];
        v_vatom(j,5) += 0.5*v[5];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairOxdnaStkKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairOxdnaStkKokkos<LMPHostType>;
#endif
}