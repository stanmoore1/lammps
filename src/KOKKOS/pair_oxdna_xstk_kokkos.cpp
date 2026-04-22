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

#include "pair_oxdna_xstk_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include "pair_oxdna_excv_kokkos.h"
#include "mf_oxdna_kokkos.h"

using namespace LAMMPS_NS;
using namespace MFOxdnaKokkos;
using MathConst::MY_PI;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairOxdnaXstkKokkos<DeviceType>::PairOxdnaXstkKokkos(LAMMPS *lmp) : PairOxdnaXstk(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | ELLIPSOID_MASK | BONUS_MASK | F_MASK | 
                  TORQUE_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
  k_screened_pair_count = DAT::tdual_int_scalar("PairOxdnaXstk:screened_pair_count");
  screened_max_atoms = 0;
  screened_max_neigh = 0;
  screened_pair_count = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairOxdnaXstkKokkos<DeviceType>::~PairOxdnaXstkKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaXstkKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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

  atomKK->sync(execution_space,datamask_read);

  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK | TORQUE_MASK);

  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  torque = atomKK->k_torque.template view<DeviceType>();
  type = atomKK->k_type.template view<DeviceType>();

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

  // d_n(x/y/z)_xtrct = extracted local unit vectors in lab frame from oxdna/excv/kk or oxdna2/excv/kk
  auto oxdna_excvKK = dynamic_cast<PairOxdnaExcvKokkos<DeviceType> *>(force->pair_match("oxdna.*excv.*", 0, 1));
  if (!oxdna_excvKK) {
    error->all(FLERR, "Failed to cast to PairOxdnaExcvKokkos");
  }
  d_nx_xtrct = oxdna_excvKK->k_nx.template view<DeviceType>();
  d_ny_xtrct = oxdna_excvKK->k_ny.template view<DeviceType>();
  d_nz_xtrct = oxdna_excvKK->k_nz.template view<DeviceType>();

  // If we're on a GPU, screen pairs and create screened neighbor list
  if (execution_space != HostKK) {
    screened_pair_count = 0;
    const int max_atoms = atom->nmax;
    const int max_neigh = d_neighbors.extent(1);
    if (max_atoms > screened_max_atoms || max_neigh > screened_max_neigh) {
      screened_max_atoms = max_atoms;
      screened_max_neigh = max_neigh;
      MemKK::realloc_kokkos(k_neighbors_screened, "PairOxdnaXstk:neighbors_screened",
                            screened_max_atoms, screened_max_neigh);
      MemKK::realloc_kokkos(k_numneigh_screened, "PairOxdnaXstk:numneigh_screened",
                            screened_max_atoms);
      MemKK::realloc_kokkos(k_screened_offsets, "PairOxdnaXstk:screened_offsets",
                            screened_max_atoms + 1);
      d_neighbors_screened = k_neighbors_screened.template view<DeviceType>();
      d_numneigh_screened = k_numneigh_screened.template view<DeviceType>();
      d_screened_offsets = k_screened_offsets.template view<DeviceType>();
    }

    // Pretty simple first pass via "TagPairOxdnaXstkScreen". We just loop through each atom a
    // and its neighbors, run 'screen_pair_fast' for each a-neighbor pair which runs up
    // to f2 and return bool. If true, we add the neighbor to the d_neighbors_screened neighbor
    // list and increment the screened neighbor count.
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkScreen>(0, anum), *this);

    // Perhaps "_local" suffixes are a little deceiving - these are shallow copies and point
    // to the same data as the non "_local" views. They're just for use in the lambdas below
    // to avoid "this->" captures which the compiler would not like. 
    const auto d_alist_local = d_alist;
    const auto d_numneigh_screened_local = d_numneigh_screened;
    const auto d_screened_offsets_local = d_screened_offsets;
    const int anum_local = anum;

    // Final/Second pass is a little more conceptually complex. ComputeGPUPair takes one flat
    // "ipair" index which runs from 0 to screened_pair_count - 1, and needs to map that
    // back to the corresponding (a,b) pair. The parallel_scan is building a prefix sum
    // over the screened neighbor counts per atom, which gives us the starting index in
    // the screened neighbor list for each atom.
    // So for example if atom 0 has 2 screened neighbors, atom 1 has 0 screened neighbors,
    // and atom 2 has 3 screened neighbors, the scanned screened_offsets would be [0, 2, 2, 5].
    // The Kokkos documentation/wiki explains parallel_scan, prefix sum, "update", "final", etc
    // in more detail.
    Kokkos::parallel_scan(
      Kokkos::RangePolicy<DeviceType>(0, anum + 1),
      KOKKOS_LAMBDA(const int i, int &update, const bool final) {
        if (i < anum_local) {
          if (final) d_screened_offsets_local(i) = update;
          const int a = d_alist_local(i);
          update += d_numneigh_screened_local(a);
        } else if (final) {
          d_screened_offsets_local(anum_local) = update;
        }
      });

    // After the parallel_scan, the subview just gives us the value at the last offset,
    // i.e. the total screened pair count.
    // screened_pair_count is a host-side int, so we can't just directly read the value
    // from the device-side d_screened_offsets view (would trigger a seg-fault).
    Kokkos::deep_copy(
      k_screened_pair_count.view_host(), Kokkos::subview(d_screened_offsets_local, anum_local));
    screened_pair_count = k_screened_pair_count.view_host()();
  }

  // loop over neighbors of my atoms for compute functors

  EV_FLOAT ev;

  if (evflag) {
    if (neighflag == HALF) {
      if (newton_pair) {
        if (execution_space == HostKK)
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<HALF,1,1> >(0,anum),*this,ev);
        else
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<HALF,1,1> >(0,screened_pair_count),*this,ev);
      } else {
        if (execution_space == HostKK)
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<HALF,0,1> >(0,anum),*this,ev);
        else
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<HALF,0,1> >(0,screened_pair_count),*this,ev);
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        if (execution_space == HostKK)
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<HALFTHREAD,1,1> >(0,anum),*this,ev);
        else
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<HALFTHREAD,1,1> >(0,screened_pair_count),*this,ev);
      } else {
        if (execution_space == HostKK)
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<HALFTHREAD,0,1> >(0,anum),*this,ev);
        else
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<HALFTHREAD,0,1> >(0,screened_pair_count),*this,ev);
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        if (execution_space == HostKK)
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<FULL,1,1> >(0,anum),*this,ev);
        else
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<FULL,1,1> >(0,screened_pair_count),*this,ev);
      } else {
        if (execution_space == HostKK)
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<FULL,0,1> >(0,anum),*this,ev);
        else
          Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<FULL,0,1> >(0,screened_pair_count),*this,ev);
      }
    }
  } else {
    if (neighflag == HALF) {
      if (newton_pair) {
        if (execution_space == HostKK)
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<HALF,1,0> >(0,anum),*this);
        else
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<HALF,1,0> >(0,screened_pair_count),*this);
      } else {
        if (execution_space == HostKK)
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<HALF,0,0> >(0,anum),*this);
        else
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<HALF,0,0> >(0,screened_pair_count),*this);
      }
    } else if (neighflag == HALFTHREAD) {
      if (newton_pair) {
        if (execution_space == HostKK)
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<HALFTHREAD,1,0> >(0,anum),*this);
        else
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<HALFTHREAD,1,0> >(0,screened_pair_count),*this);
      } else {
        if (execution_space == HostKK)
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<HALFTHREAD,0,0> >(0,anum),*this);
        else
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<HALFTHREAD,0,0> >(0,screened_pair_count),*this);
      }
    } else if (neighflag == FULL) {
      if (newton_pair) {
        if (execution_space == HostKK)
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<FULL,1,0> >(0,anum),*this);
        else
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<FULL,1,0> >(0,screened_pair_count),*this);
      } else {
        if (execution_space == HostKK)
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkCompute<FULL,0,0> >(0,anum),*this);
        else
          Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairOxdnaXstkComputeGPUPair<FULL,0,0> >(0,screened_pair_count),*this);
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

/* ----------------------------------------------------------------------
   Standard non-GPU Compute Functor(s)
-------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaXstkKokkos<DeviceType>::operator()(TagPairOxdnaXstkCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, \
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
  const KK_FLOAT a_nx0 = d_nx_xtrct(a,0);
  const KK_FLOAT a_nx1 = d_nx_xtrct(a,1);
  const KK_FLOAT a_nx2 = d_nx_xtrct(a,2);
  const KK_FLOAT a_nz0 = d_nz_xtrct(a,0);
  const KK_FLOAT a_nz1 = d_nz_xtrct(a,1);
  const KK_FLOAT a_nz2 = d_nz_xtrct(a,2);
  // vectors COM-hbond site in lab frame
  KK_FLOAT ra_chb[3], rb_chb[3];

  KK_FLOAT delf[3], delta[3], deltb[3];    // force, torque increment
  KK_FLOAT evdwl, finc, tpair;             // energy, force, torque
  KK_FLOAT delr_hb[3],delr_hb_norm[3],rsq_hb,r_hb,rinv_hb;
  KK_FLOAT theta1,t1dir[3],cost1;
  KK_FLOAT theta2,t2dir[3],cost2;
  KK_FLOAT theta3,t3dir[3],cost3;
  KK_FLOAT theta4,theta4p,t4dir[3],cost4;
  KK_FLOAT theta7,theta7p,t7dir[3],cost7;
  KK_FLOAT theta8,theta8p,t8dir[3],cost8;

  KK_FLOAT f2,f4t1,f4t4,f4t2,f4t3,f4t7,f4t8;
  KK_FLOAT df2,df4t1,df4t4,df4t2,df4t3,df4t7,df4t8,rsint;

  // vector COM-hbond site a
  constexpr KK_FLOAT d_chb=+0.4;
  ra_chb[0] = d_chb*a_nx0;
  ra_chb[1] = d_chb*a_nx1;
  ra_chb[2] = d_chb*a_nx2;
  
  const int bnum = d_numneigh(a);

  for (int ib = 0; ib < bnum; ib++) {

    int b = d_neighbors(a,ib);
    const KK_FLOAT factor_lj = special_lj[sbmask(b)];
    if (!factor_lj) continue;
    b &= NEIGHMASK;
    const int btype = type(b);

    // "p_" for per-pair scalar. Reduces repeated global reads.
    const KK_FLOAT p_k_xst = d_k_xst(atype,btype);
    const KK_FLOAT p_cut_xst_0 = d_cut_xst_0(atype,btype);
    const KK_FLOAT p_cut_xst_lc = d_cut_xst_lc(atype,btype);
    const KK_FLOAT p_cut_xst_hc = d_cut_xst_hc(atype,btype);
    const KK_FLOAT p_cut_xst_lo = d_cut_xst_lo(atype,btype);
    const KK_FLOAT p_cut_xst_hi = d_cut_xst_hi(atype,btype);
    const KK_FLOAT p_b_xst_lo = d_b_xst_lo(atype,btype);
    const KK_FLOAT p_b_xst_hi = d_b_xst_hi(atype,btype);
    const KK_FLOAT p_cut_xst_c = d_cut_xst_c(atype,btype);

    const KK_FLOAT p_a_xst1 = d_a_xst1(atype, btype);
    const KK_FLOAT p_theta_xst1_0 = d_theta_xst1_0(atype, btype);
    const KK_FLOAT p_dtheta_xst1_ast = d_dtheta_xst1_ast(atype, btype);
    const KK_FLOAT p_b_xst1 = d_b_xst1(atype, btype);
    const KK_FLOAT p_dtheta_xst1_c = d_dtheta_xst1_c(atype, btype);

    const KK_FLOAT p_a_xst2 = d_a_xst2(atype, btype);
    const KK_FLOAT p_theta_xst2_0 = d_theta_xst2_0(atype, btype);
    const KK_FLOAT p_dtheta_xst2_ast = d_dtheta_xst2_ast(atype, btype);
    const KK_FLOAT p_b_xst2 = d_b_xst2(atype, btype);
    const KK_FLOAT p_dtheta_xst2_c = d_dtheta_xst2_c(atype, btype);

    const KK_FLOAT p_a_xst3 = d_a_xst3(atype, btype);
    const KK_FLOAT p_theta_xst3_0 = d_theta_xst3_0(atype, btype);
    const KK_FLOAT p_dtheta_xst3_ast = d_dtheta_xst3_ast(atype, btype);
    const KK_FLOAT p_b_xst3 = d_b_xst3(atype, btype);
    const KK_FLOAT p_dtheta_xst3_c = d_dtheta_xst3_c(atype, btype);

    const KK_FLOAT p_a_xst4 = d_a_xst4(atype, btype);
    const KK_FLOAT p_theta_xst4_0 = d_theta_xst4_0(atype, btype);
    const KK_FLOAT p_dtheta_xst4_ast = d_dtheta_xst4_ast(atype, btype);
    const KK_FLOAT p_b_xst4 = d_b_xst4(atype, btype);
    const KK_FLOAT p_dtheta_xst4_c = d_dtheta_xst4_c(atype, btype);

    const KK_FLOAT p_a_xst7 = d_a_xst7(atype, btype);
    const KK_FLOAT p_theta_xst7_0 = d_theta_xst7_0(atype, btype);
    const KK_FLOAT p_dtheta_xst7_ast = d_dtheta_xst7_ast(atype, btype);
    const KK_FLOAT p_b_xst7 = d_b_xst7(atype, btype);
    const KK_FLOAT p_dtheta_xst7_c = d_dtheta_xst7_c(atype, btype);

    const KK_FLOAT p_a_xst8 = d_a_xst8(atype, btype);
    const KK_FLOAT p_theta_xst8_0 = d_theta_xst8_0(atype, btype);
    const KK_FLOAT p_dtheta_xst8_ast = d_dtheta_xst8_ast(atype, btype);
    const KK_FLOAT p_b_xst8 = d_b_xst8(atype, btype);
    const KK_FLOAT p_dtheta_xst8_c = d_dtheta_xst8_c(atype, btype);

    const KK_FLOAT b_nx0 = d_nx_xtrct(b,0);
    const KK_FLOAT b_nx1 = d_nx_xtrct(b,1);
    const KK_FLOAT b_nx2 = d_nx_xtrct(b,2);
    const KK_FLOAT b_nz0 = d_nz_xtrct(b,0);
    const KK_FLOAT b_nz1 = d_nz_xtrct(b,1);
    const KK_FLOAT b_nz2 = d_nz_xtrct(b,2);

    // vector COM-hbond site b
    rb_chb[0] = d_chb*b_nx0;
    rb_chb[1] = d_chb*b_nx1;
    rb_chb[2] = d_chb*b_nx2;

    // vector h-bonding site b-a
    delr_hb[0] = x(a,0) + ra_chb[0] - x(b,0) - rb_chb[0];
    delr_hb[1] = x(a,1) + ra_chb[1] - x(b,1) - rb_chb[1];
    delr_hb[2] = x(a,2) + ra_chb[2] - x(b,2) - rb_chb[2];

    rsq_hb = delr_hb[0]*delr_hb[0] + delr_hb[1]*delr_hb[1] + delr_hb[2]*delr_hb[2];
    r_hb = sqrt(rsq_hb);
    rinv_hb = 1.0 / r_hb;

    delr_hb_norm[0] = delr_hb[0] * rinv_hb;
    delr_hb_norm[1] = delr_hb[1] * rinv_hb;
    delr_hb_norm[2] = delr_hb[2] * rinv_hb;

    f2 = F2_KK(r_hb, p_k_xst, p_cut_xst_0,
             p_cut_xst_lc, p_cut_xst_hc, p_cut_xst_lo, p_cut_xst_hi,
             p_b_xst_lo, p_b_xst_hi, p_cut_xst_c);
    if (!f2) continue;

    // theta1 calculation
    cost1 = - (a_nx0*b_nx0 + a_nx1*b_nx1 + a_nx2*b_nx2);
    if (cost1 > 1.0) cost1 = 1.0;
    if (cost1 < -1.0) cost1 = -1.0;
    theta1 = acos(cost1);
    // F4 modulation factor
    f4t1 = F4_KK(theta1, p_a_xst1, p_theta_xst1_0, p_dtheta_xst1_ast,
      p_b_xst1, p_dtheta_xst1_c);
    if (!f4t1) continue;

    // theta2 calculation
    cost2 = - (a_nx0*delr_hb_norm[0] + a_nx1*delr_hb_norm[1] + a_nx2*delr_hb_norm[2]);
    if (cost2 > 1.0) cost2 = 1.0;
    if (cost2 < -1.0) cost2 = -1.0;
    theta2 = acos(cost2);
    // F4 modulation factor
    f4t2 = F4_KK(theta2, p_a_xst2, p_theta_xst2_0, p_dtheta_xst2_ast,
      p_b_xst2, p_dtheta_xst2_c);
    if (!f4t2) continue;

    // theta3 calculation
    cost3 = b_nx0*delr_hb_norm[0] + b_nx1*delr_hb_norm[1] + b_nx2*delr_hb_norm[2];
    if (cost3 > 1.0) cost3 = 1.0;
    if (cost3 < -1.0) cost3 = -1.0;
    theta3 = acos(cost3);
    // F4 modulation factor
    f4t3 = F4_KK(theta3, p_a_xst3, p_theta_xst3_0, p_dtheta_xst3_ast,
      p_b_xst3, p_dtheta_xst3_c);
    if (!f4t3) continue;

    // theta4 and theta4p calculation
    cost4 = a_nz0*b_nz0 + a_nz1*b_nz1 + a_nz2*b_nz2;
    if (cost4 > 1.0) cost4 = 1.0;
    if (cost4 < -1.0) cost4 = -1.0;
    theta4 = acos(cost4);
    theta4p = MY_PI - theta4;
    // F4 modulation factor
    f4t4 = F4_KK(theta4, p_a_xst4, p_theta_xst4_0, p_dtheta_xst4_ast,
      p_b_xst4, p_dtheta_xst4_c) +
      F4_KK(theta4p, p_a_xst4, p_theta_xst4_0, p_dtheta_xst4_ast,
      p_b_xst4, p_dtheta_xst4_c);
    if (!f4t4) continue;

    // theta7 and theta7p calculation
    cost7 = - (a_nz0*delr_hb_norm[0] + a_nz1*delr_hb_norm[1] + a_nz2*delr_hb_norm[2]);
    if (cost7 > 1.0) cost7 = 1.0;
    if (cost7 < -1.0) cost7 = -1.0;
    theta7 = acos(cost7);
    theta7p = MY_PI - theta7;
    // F4 modulation factor
    f4t7 = F4_KK(theta7, p_a_xst7, p_theta_xst7_0, p_dtheta_xst7_ast,
      p_b_xst7, p_dtheta_xst7_c) +
      F4_KK(theta7p, p_a_xst7, p_theta_xst7_0, p_dtheta_xst7_ast,
      p_b_xst7, p_dtheta_xst7_c);
    if (!f4t7) continue;

    // theta8 and theta8p calculation
    cost8 = b_nz0*delr_hb_norm[0] + b_nz1*delr_hb_norm[1] + b_nz2*delr_hb_norm[2];
    if (cost8 > 1.0) cost8 = 1.0;
    if (cost8 < -1.0) cost8 = -1.0;
    theta8 = acos(cost8);
    theta8p = MY_PI - theta8;
    // F4 modulation factor
    f4t8 = F4_KK(theta8, p_a_xst8, p_theta_xst8_0, p_dtheta_xst8_ast,
      p_b_xst8, p_dtheta_xst8_c) +
      F4_KK(theta8p, p_a_xst8, p_theta_xst8_0, p_dtheta_xst8_ast,
      p_b_xst8, p_dtheta_xst8_c);

    evdwl = f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj;
    if (!evdwl) continue;

    // df2 = DF2 modulation factor
    df2 = DF2_KK(r_hb, p_k_xst, p_cut_xst_0,
            p_cut_xst_lc, p_cut_xst_hc, p_cut_xst_lo, p_cut_xst_hi,
            p_b_xst_lo, p_b_xst_hi);
    // df4t1 = DF4 modulation factor
        df4t1 = DF4_KK(theta1, p_a_xst1, p_theta_xst1_0, p_dtheta_xst1_ast,
          p_b_xst1, p_dtheta_xst1_c)/sin(theta1);
    // df4t2 = DF4 modulation factor
        df4t2 = DF4_KK(theta2, p_a_xst2, p_theta_xst2_0, p_dtheta_xst2_ast,
          p_b_xst2, p_dtheta_xst2_c)/sin(theta2);
    // df4t3 = DF4 modulation factor
        df4t3 = DF4_KK(theta3, p_a_xst3, p_theta_xst3_0, p_dtheta_xst3_ast,
          p_b_xst3, p_dtheta_xst3_c)/sin(theta3);
    // df4t4 = DF4 modulation factor
    rsint = 1.0 / sin(theta4);
        df4t4 = DF4_KK(theta4, p_a_xst4, p_theta_xst4_0, p_dtheta_xst4_ast,
          p_b_xst4, p_dtheta_xst4_c) -
          DF4_KK(theta4p, p_a_xst4, p_theta_xst4_0, p_dtheta_xst4_ast,
          p_b_xst4, p_dtheta_xst4_c);
    df4t4 *= rsint;
    // df4t7 = DF4 modulation factor
    rsint = 1.0 / sin(theta7);
        df4t7 = DF4_KK(theta7, p_a_xst7, p_theta_xst7_0, p_dtheta_xst7_ast,
          p_b_xst7, p_dtheta_xst7_c) -
          DF4_KK(theta7p, p_a_xst7, p_theta_xst7_0, p_dtheta_xst7_ast,
          p_b_xst7, p_dtheta_xst7_c);
    df4t7 *= rsint;
    // df4t8 = DF4 modulation factor
    rsint = 1.0 / sin(theta8);
        df4t8 = DF4_KK(theta8, p_a_xst8, p_theta_xst8_0, p_dtheta_xst8_ast,
          p_b_xst8, p_dtheta_xst8_c) -
          DF4_KK(theta8p, p_a_xst8, p_theta_xst8_0, p_dtheta_xst8_ast,
          p_b_xst8, p_dtheta_xst8_c);
    df4t8 *= rsint;

    // force, torque, and viral contributions for forces between h-bonding sites

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
    finc  = -df2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;

    delf[0] += delr_hb[0] * finc;
    delf[1] += delr_hb[1] * finc;
    delf[2] += delr_hb[2] * finc;

    // theta2 force
    if (theta2) {

      finc  = -f2 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;

      delf[0] += (delr_hb_norm[0]*cost2 + a_nx0) * finc;
      delf[1] += (delr_hb_norm[1]*cost2 + a_nx1) * finc;
      delf[2] += (delr_hb_norm[2]*cost2 + a_nx2) * finc;
    }

    // theta3 force
    if (theta3) {

      finc  = -f2 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;

      delf[0] += (delr_hb_norm[0]*cost3 - b_nx0) * finc;
      delf[1] += (delr_hb_norm[1]*cost3 - b_nx1) * finc;
      delf[2] += (delr_hb_norm[2]*cost3 - b_nx2) * finc;
    }

    // theta7 force
    if (theta7) {
      
      finc  = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * rinv_hb * factor_lj;

      delf[0] += (delr_hb_norm[0]*cost7 + a_nz0) * finc;
      delf[1] += (delr_hb_norm[1]*cost7 + a_nz1) * finc;
      delf[2] += (delr_hb_norm[2]*cost7 + a_nz2) * finc;

    }

    // theta8 force
    if (theta8) {

      finc  = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * rinv_hb * factor_lj;

      delf[0] += (delr_hb_norm[0]*cost8 - b_nz0) * finc;
      delf[1] += (delr_hb_norm[1]*cost8 - b_nz1) * finc;
      delf[2] += (delr_hb_norm[2]*cost8 - b_nz2) * finc;

    }
    
    // increment forces and torques

    a_f(a,0) += delf[0];
    a_f(a,1) += delf[1];
    a_f(a,2) += delf[2];
    delta[0] = ra_chb[1]*delf[2] - ra_chb[2]*delf[1];
    delta[1] = ra_chb[2]*delf[0] - ra_chb[0]*delf[2];
    delta[2] = ra_chb[0]*delf[1] - ra_chb[1]*delf[0];
    a_torque(a,0) += delta[0];
    a_torque(a,1) += delta[1];
    a_torque(a,2) += delta[2];

    if ( (NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || b < nlocal) ) {
      a_f(b,0) -= delf[0];
      a_f(b,1) -= delf[1];
      a_f(b,2) -= delf[2];
      deltb[0] = rb_chb[1]*delf[2] - rb_chb[2]*delf[1];
      deltb[1] = rb_chb[2]*delf[0] - rb_chb[0]*delf[2];
      deltb[2] = rb_chb[0]*delf[1] - rb_chb[1]*delf[0];
      a_torque(b,0) -= deltb[0];
      a_torque(b,1) -= deltb[1];
      a_torque(b,2) -= deltb[2];
    }

    // increment energy and virial
    // NOTE: The virial is calculated on the 'molecular' basis.
    // (see G. Ciccotti and J.P. Ryckaert, Comp. Phys. Rep. 4, 345-392 (1986))

    if (EVFLAG) {
      ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*evdwl;

      if (vflag_either || eflag_atom) {
        this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
        delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
      }
    }

    // pure torques not expressible as r x f

    delta[0] = 0.0;
    delta[1] = 0.0;
    delta[2] = 0.0;
    deltb[0] = 0.0;
    deltb[1] = 0.0;
    deltb[2] = 0.0;

    // theta1 torque
    if (theta1) {

      tpair = -f2 * df4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj;

      t1dir[0] = a_nx1 * b_nx2 - a_nx2 * b_nx1;
      t1dir[1] = a_nx2 * b_nx0 - a_nx0 * b_nx2;
      t1dir[2] = a_nx0 * b_nx1 - a_nx1 * b_nx0;
      delta[0] += t1dir[0] * tpair;
      delta[1] += t1dir[1] * tpair;
      delta[2] += t1dir[2] * tpair;
      deltb[0] += t1dir[0] * tpair;
      deltb[1] += t1dir[1] * tpair;
      deltb[2] += t1dir[2] * tpair;
    }
    //theta2 torque
    if (theta2) {

      tpair = -f2 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj;

      t2dir[0] = a_nx1 * delr_hb_norm[2] - a_nx2 * delr_hb_norm[1];
      t2dir[1] = a_nx2 * delr_hb_norm[0] - a_nx0 * delr_hb_norm[2];
      t2dir[2] = a_nx0 * delr_hb_norm[1] - a_nx1 * delr_hb_norm[0];
      delta[0] += t2dir[0] * tpair;
      delta[1] += t2dir[1] * tpair;
      delta[2] += t2dir[2] * tpair;
    }
    //theta3 torque
    if (theta3) {

      tpair = -f2 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * factor_lj;

      t3dir[0] = b_nx1 * delr_hb_norm[2] - b_nx2 * delr_hb_norm[1];
      t3dir[1] = b_nx2 * delr_hb_norm[0] - b_nx0 * delr_hb_norm[2];
      t3dir[2] = b_nx0 * delr_hb_norm[1] - b_nx1 * delr_hb_norm[0];
      deltb[0] += t3dir[0] * tpair;
      deltb[1] += t3dir[1] * tpair;
      deltb[2] += t3dir[2] * tpair;
    }
    //theta4 torque
    if (theta4 && theta4p) {

      tpair = -f2 * f4t1 * f4t2 * f4t3 * df4t4 * f4t7 * f4t8 * factor_lj;

      t4dir[0] = b_nz1 * a_nz2 - b_nz2 * a_nz1;
      t4dir[1] = b_nz2 * a_nz0 - b_nz0 * a_nz2;
      t4dir[2] = b_nz0 * a_nz1 - b_nz1 * a_nz0;
      delta[0] += t4dir[0] * tpair;
      delta[1] += t4dir[1] * tpair;
      delta[2] += t4dir[2] * tpair;
      deltb[0] += t4dir[0] * tpair;
      deltb[1] += t4dir[1] * tpair;
      deltb[2] += t4dir[2] * tpair;
    }
    //theta7 torque
    if (theta7) {

      tpair = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * factor_lj;

      t7dir[0] = a_nz1 * delr_hb_norm[2] - a_nz2 * delr_hb_norm[1];
      t7dir[1] = a_nz2 * delr_hb_norm[0] - a_nz0 * delr_hb_norm[2];
      t7dir[2] = a_nz0 * delr_hb_norm[1] - a_nz1 * delr_hb_norm[0];
      delta[0] += t7dir[0] * tpair;
      delta[1] += t7dir[1] * tpair;
      delta[2] += t7dir[2] * tpair;
    }
    //theta8 torque
    if (theta8) {

      tpair = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * factor_lj;

      t8dir[0] = b_nz1 * delr_hb_norm[2] - b_nz2 * delr_hb_norm[1];
      t8dir[1] = b_nz2 * delr_hb_norm[0] - b_nz0 * delr_hb_norm[2];
      t8dir[2] = b_nz0 * delr_hb_norm[1] - b_nz1 * delr_hb_norm[0];
      deltb[0] += t8dir[0] * tpair;
      deltb[1] += t8dir[1] * tpair;
      deltb[2] += t8dir[2] * tpair;
    }
    
    // increment torques

    a_torque(a,0) += delta[0];
    a_torque(a,1) += delta[1];
    a_torque(a,2) += delta[2];

    if ( (NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || b < nlocal) ) {
      a_torque(b,0) -= deltb[0];
      a_torque(b,1) -= deltb[1];
      a_torque(b,2) -= deltb[2];
    }
  }
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaXstkKokkos<DeviceType>::operator()(TagPairOxdnaXstkCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, \
  const int &ia) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG>\
  (TagPairOxdnaXstkCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(),ia,ev);
}

/* ----------------------------------------------------------------------
   ComputeGPUPair Functor(s)
-------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
bool PairOxdnaXstkKokkos<DeviceType>::screen_pair_fast(const int &atype,
                                                         const int &braw,
                                                         const KK_FLOAT &a_hb0,
                                                         const KK_FLOAT &a_hb1,
                                                         const KK_FLOAT &a_hb2) const
{
  const KK_FLOAT factor_lj = special_lj[sbmask(braw)];
  if (factor_lj == 0.0) return false;

  const int b = braw & NEIGHMASK;
  const int btype = type(b);

  constexpr KK_FLOAT d_chb = +0.4;

  const KK_FLOAT b_nx0 = d_nx_xtrct(b,0);
  const KK_FLOAT b_nx1 = d_nx_xtrct(b,1);
  const KK_FLOAT b_nx2 = d_nx_xtrct(b,2);
  const KK_FLOAT rb0 = d_chb * b_nx0;
  const KK_FLOAT rb1 = d_chb * b_nx1;
  const KK_FLOAT rb2 = d_chb * b_nx2;

  KK_FLOAT delr_hb[3];
  delr_hb[0] = a_hb0 - x(b,0) - rb0;
  delr_hb[1] = a_hb1 - x(b,1) - rb1;
  delr_hb[2] = a_hb2 - x(b,2) - rb2;

  // fma is fused-multipy-add op
  const KK_FLOAT rsq_hb = fma(delr_hb[2], delr_hb[2],
                          fma(delr_hb[1], delr_hb[1], delr_hb[0] * delr_hb[0]));

  // Fast boolean screen: F2_KK is nonzero if r is in [cut_lc, cut_hc]
  // Use squared-distance comparison instead of usual oxDNA F2 to avoid
  // expensive sqrt(). It's cheaper to do two squares than one sqrt.
  const KK_FLOAT cut_lc = d_cut_xst_lc(atype,btype);
  const KK_FLOAT cut_hc = d_cut_xst_hc(atype,btype);
  return (rsq_hb >= cut_lc*cut_lc && rsq_hb <= cut_hc*cut_hc);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairOxdnaXstkKokkos<DeviceType>::operator()(TagPairOxdnaXstkScreen, const int &ia) const
{
  const int a = d_alist(ia);
  const int atype = type(a);
  const int bnum = d_numneigh(a);
  const KK_FLOAT a_nx0 = d_nx_xtrct(a,0);
  const KK_FLOAT a_nx1 = d_nx_xtrct(a,1);
  const KK_FLOAT a_nx2 = d_nx_xtrct(a,2);
  constexpr KK_FLOAT d_chb = +0.4;
  const KK_FLOAT a_hb0 = x(a,0) + d_chb * a_nx0;
  const KK_FLOAT a_hb1 = x(a,1) + d_chb * a_nx1;
  const KK_FLOAT a_hb2 = x(a,2) + d_chb * a_nx2;

  int nscreen = 0;
  for (int ib = 0; ib < bnum; ib++) {
    const int braw = d_neighbors(a,ib);
    if (screen_pair_fast(atype, braw, a_hb0, a_hb1, a_hb2)) {
      d_neighbors_screened(a, nscreen++) = braw;
    }
  }
  d_numneigh_screened(a) = nscreen;
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaXstkKokkos<DeviceType>::operator()(TagPairOxdnaXstkComputeGPUPair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, \
  const int &ipair, EV_FLOAT &ev) const
{
  auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();
  auto v_torque = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,\
    decltype(dup_torque),decltype(ndup_torque)>::get(dup_torque,ndup_torque);
  auto a_torque = v_torque.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  // Firstly, we need to find which atom block owns this ipair.
  // We can do this via binary search of the screened neighbor list offsets.
  int lo = 0;
  int hi = anum;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (d_screened_offsets(mid) <= ipair) lo = mid;
    else hi = mid;
  }

  const int ia = lo;
  const int a = d_alist(ia);
  const int atype = type(a);
  // Secondly, we need to find which neighbor b this ipair corresponds to.
  const int ib = ipair - d_screened_offsets(ia);

  // And finally, we get our desired (a,b) pair.
  int b = d_neighbors_screened(a, ib);
  const KK_FLOAT factor_lj = special_lj[sbmask(b)];
  // No need for factor_lj early exit check here since we already screened the neighbor list.
  b &= NEIGHMASK;
  const int btype = type(b);

  const KK_FLOAT a_nx0 = d_nx_xtrct(a,0);
  const KK_FLOAT a_nx1 = d_nx_xtrct(a,1);
  const KK_FLOAT a_nx2 = d_nx_xtrct(a,2);
  const KK_FLOAT a_nz0 = d_nz_xtrct(a,0);
  const KK_FLOAT a_nz1 = d_nz_xtrct(a,1);
  const KK_FLOAT a_nz2 = d_nz_xtrct(a,2);

  KK_FLOAT ra_chb[3], rb_chb[3];
  KK_FLOAT delf[3], delta[3], deltb[3];
  KK_FLOAT evdwl, finc, tpair;
  KK_FLOAT delr_hb[3],delr_hb_norm[3],rsq_hb,r_hb,rinv_hb;
  KK_FLOAT theta1,cost1;
  KK_FLOAT theta2,cost2;
  KK_FLOAT theta3,cost3;
  KK_FLOAT theta4,theta4p,cost4;
  KK_FLOAT theta7,theta7p,cost7;
  KK_FLOAT theta8,theta8p,cost8;

  KK_FLOAT f2,f4t1,f4t4,f4t2,f4t3,f4t7,f4t8;
  KK_FLOAT df2,df4t1,df4t4,df4t2,df4t3,df4t7,df4t8,rsint;

  // "p_" for per-pair scalar. Reduces repeated global reads.
  const KK_FLOAT p_k_xst = d_k_xst(atype,btype);
  const KK_FLOAT p_cut_xst_0 = d_cut_xst_0(atype,btype);
  const KK_FLOAT p_cut_xst_lc = d_cut_xst_lc(atype,btype);
  const KK_FLOAT p_cut_xst_hc = d_cut_xst_hc(atype,btype);
  const KK_FLOAT p_cut_xst_lo = d_cut_xst_lo(atype,btype);
  const KK_FLOAT p_cut_xst_hi = d_cut_xst_hi(atype,btype);
  const KK_FLOAT p_b_xst_lo = d_b_xst_lo(atype,btype);
  const KK_FLOAT p_b_xst_hi = d_b_xst_hi(atype,btype);
  const KK_FLOAT p_cut_xst_c = d_cut_xst_c(atype,btype);

  const KK_FLOAT p_a_xst1 = d_a_xst1(atype, btype);
  const KK_FLOAT p_theta_xst1_0 = d_theta_xst1_0(atype, btype);
  const KK_FLOAT p_dtheta_xst1_ast = d_dtheta_xst1_ast(atype, btype);
  const KK_FLOAT p_b_xst1 = d_b_xst1(atype, btype);
  const KK_FLOAT p_dtheta_xst1_c = d_dtheta_xst1_c(atype, btype);

  const KK_FLOAT p_a_xst2 = d_a_xst2(atype, btype);
  const KK_FLOAT p_theta_xst2_0 = d_theta_xst2_0(atype, btype);
  const KK_FLOAT p_dtheta_xst2_ast = d_dtheta_xst2_ast(atype, btype);
  const KK_FLOAT p_b_xst2 = d_b_xst2(atype, btype);
  const KK_FLOAT p_dtheta_xst2_c = d_dtheta_xst2_c(atype, btype);

  const KK_FLOAT p_a_xst3 = d_a_xst3(atype, btype);
  const KK_FLOAT p_theta_xst3_0 = d_theta_xst3_0(atype, btype);
  const KK_FLOAT p_dtheta_xst3_ast = d_dtheta_xst3_ast(atype, btype);
  const KK_FLOAT p_b_xst3 = d_b_xst3(atype, btype);
  const KK_FLOAT p_dtheta_xst3_c = d_dtheta_xst3_c(atype, btype);

  const KK_FLOAT p_a_xst4 = d_a_xst4(atype, btype);
  const KK_FLOAT p_theta_xst4_0 = d_theta_xst4_0(atype, btype);
  const KK_FLOAT p_dtheta_xst4_ast = d_dtheta_xst4_ast(atype, btype);
  const KK_FLOAT p_b_xst4 = d_b_xst4(atype, btype);
  const KK_FLOAT p_dtheta_xst4_c = d_dtheta_xst4_c(atype, btype);

  const KK_FLOAT p_a_xst7 = d_a_xst7(atype, btype);
  const KK_FLOAT p_theta_xst7_0 = d_theta_xst7_0(atype, btype);
  const KK_FLOAT p_dtheta_xst7_ast = d_dtheta_xst7_ast(atype, btype);
  const KK_FLOAT p_b_xst7 = d_b_xst7(atype, btype);
  const KK_FLOAT p_dtheta_xst7_c = d_dtheta_xst7_c(atype, btype);

  const KK_FLOAT p_a_xst8 = d_a_xst8(atype, btype);
  const KK_FLOAT p_theta_xst8_0 = d_theta_xst8_0(atype, btype);
  const KK_FLOAT p_dtheta_xst8_ast = d_dtheta_xst8_ast(atype, btype);
  const KK_FLOAT p_b_xst8 = d_b_xst8(atype, btype);
  const KK_FLOAT p_dtheta_xst8_c = d_dtheta_xst8_c(atype, btype);

  constexpr KK_FLOAT d_chb=+0.4;
  ra_chb[0] = d_chb*a_nx0;
  ra_chb[1] = d_chb*a_nx1;
  ra_chb[2] = d_chb*a_nx2;

  const KK_FLOAT b_nx0 = d_nx_xtrct(b,0);
  const KK_FLOAT b_nx1 = d_nx_xtrct(b,1);
  const KK_FLOAT b_nx2 = d_nx_xtrct(b,2);
  const KK_FLOAT b_nz0 = d_nz_xtrct(b,0);
  const KK_FLOAT b_nz1 = d_nz_xtrct(b,1);
  const KK_FLOAT b_nz2 = d_nz_xtrct(b,2);

  rb_chb[0] = d_chb*b_nx0;
  rb_chb[1] = d_chb*b_nx1;
  rb_chb[2] = d_chb*b_nx2;

  delr_hb[0] = x(a,0) + ra_chb[0] - x(b,0) - rb_chb[0];
  delr_hb[1] = x(a,1) + ra_chb[1] - x(b,1) - rb_chb[1];
  delr_hb[2] = x(a,2) + ra_chb[2] - x(b,2) - rb_chb[2];

  // fma (Kokkos::fma) fuses multiply-add operations: fma(x,y,z) = x*y + z,
  // but with only one FP rounding error and one instruction instead of two.
  rsq_hb = fma(delr_hb[2], delr_hb[2],
           fma(delr_hb[1], delr_hb[1], delr_hb[0] * delr_hb[0]));
  r_hb = sqrt(rsq_hb);
  rinv_hb = 1.0 / r_hb;

  delr_hb_norm[0] = delr_hb[0] * rinv_hb;
  delr_hb_norm[1] = delr_hb[1] * rinv_hb;
  delr_hb_norm[2] = delr_hb[2] * rinv_hb;

  f2 = F2_KK(r_hb, p_k_xst, p_cut_xst_0,
         p_cut_xst_lc, p_cut_xst_hc, p_cut_xst_lo, p_cut_xst_hi,
         p_b_xst_lo, p_b_xst_hi, p_cut_xst_c);
  // No need for f2 early exit check here since we already screened the neighbor list.

  cost1 = -fma(a_nx2, b_nx2,
           fma(a_nx1, b_nx1, a_nx0 * b_nx0));
  if (cost1 > 1.0) cost1 = 1.0;
  if (cost1 < -1.0) cost1 = -1.0;
  theta1 = acos(cost1);
  f4t1 = F4_KK(theta1, p_a_xst1, p_theta_xst1_0, p_dtheta_xst1_ast,
    p_b_xst1, p_dtheta_xst1_c);
  if (!f4t1) return;

  cost2 = -fma(a_nx2, delr_hb_norm[2],
           fma(a_nx1, delr_hb_norm[1], a_nx0 * delr_hb_norm[0]));
  if (cost2 > 1.0) cost2 = 1.0;
  if (cost2 < -1.0) cost2 = -1.0;
  theta2 = acos(cost2);
  f4t2 = F4_KK(theta2, p_a_xst2, p_theta_xst2_0, p_dtheta_xst2_ast,
    p_b_xst2, p_dtheta_xst2_c);
  if (!f4t2) return;

  cost3 = fma(b_nx2, delr_hb_norm[2],
          fma(b_nx1, delr_hb_norm[1], b_nx0 * delr_hb_norm[0]));
  if (cost3 > 1.0) cost3 = 1.0;
  if (cost3 < -1.0) cost3 = -1.0;
  theta3 = acos(cost3);
  f4t3 = F4_KK(theta3, p_a_xst3, p_theta_xst3_0, p_dtheta_xst3_ast,
    p_b_xst3, p_dtheta_xst3_c);
  if (!f4t3) return;

  cost4 = fma(a_nz2, b_nz2,
          fma(a_nz1, b_nz1, a_nz0 * b_nz0));
  if (cost4 > 1.0) cost4 = 1.0;
  if (cost4 < -1.0) cost4 = -1.0;
  theta4 = acos(cost4);
  theta4p = MY_PI - theta4;
  f4t4 = F4_KK(theta4, p_a_xst4, p_theta_xst4_0, p_dtheta_xst4_ast,
    p_b_xst4, p_dtheta_xst4_c) +
    F4_KK(theta4p, p_a_xst4, p_theta_xst4_0, p_dtheta_xst4_ast,
    p_b_xst4, p_dtheta_xst4_c);
  if (!f4t4) return;

  cost7 = -fma(a_nz2, delr_hb_norm[2],
           fma(a_nz1, delr_hb_norm[1], a_nz0 * delr_hb_norm[0]));
  if (cost7 > 1.0) cost7 = 1.0;
  if (cost7 < -1.0) cost7 = -1.0;
  theta7 = acos(cost7);
  theta7p = MY_PI - theta7;
  f4t7 = F4_KK(theta7, p_a_xst7, p_theta_xst7_0, p_dtheta_xst7_ast,
    p_b_xst7, p_dtheta_xst7_c) +
    F4_KK(theta7p, p_a_xst7, p_theta_xst7_0, p_dtheta_xst7_ast,
    p_b_xst7, p_dtheta_xst7_c);
  if (!f4t7) return;

  cost8 = fma(b_nz2, delr_hb_norm[2],
          fma(b_nz1, delr_hb_norm[1], b_nz0 * delr_hb_norm[0]));
  if (cost8 > 1.0) cost8 = 1.0;
  if (cost8 < -1.0) cost8 = -1.0;
  theta8 = acos(cost8);
  theta8p = MY_PI - theta8;
  f4t8 = F4_KK(theta8, p_a_xst8, p_theta_xst8_0, p_dtheta_xst8_ast,
    p_b_xst8, p_dtheta_xst8_c) +
    F4_KK(theta8p, p_a_xst8, p_theta_xst8_0, p_dtheta_xst8_ast,
    p_b_xst8, p_dtheta_xst8_c);

  evdwl = f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj;
  if (!evdwl) return;

  df2 = DF2_KK(r_hb, p_k_xst, p_cut_xst_0,
          p_cut_xst_lc, p_cut_xst_hc, p_cut_xst_lo, p_cut_xst_hi,
          p_b_xst_lo, p_b_xst_hi);
  df4t1 = DF4_KK(theta1, p_a_xst1, p_theta_xst1_0, p_dtheta_xst1_ast,
        p_b_xst1, p_dtheta_xst1_c)/sin(theta1);
  df4t2 = DF4_KK(theta2, p_a_xst2, p_theta_xst2_0, p_dtheta_xst2_ast,
        p_b_xst2, p_dtheta_xst2_c)/sin(theta2);
  df4t3 = DF4_KK(theta3, p_a_xst3, p_theta_xst3_0, p_dtheta_xst3_ast,
        p_b_xst3, p_dtheta_xst3_c)/sin(theta3);
  rsint = 1.0 / sin(theta4);
  df4t4 = DF4_KK(theta4, p_a_xst4, p_theta_xst4_0, p_dtheta_xst4_ast,
        p_b_xst4, p_dtheta_xst4_c) -
        DF4_KK(theta4p, p_a_xst4, p_theta_xst4_0, p_dtheta_xst4_ast,
        p_b_xst4, p_dtheta_xst4_c);
  df4t4 *= rsint;
  rsint = 1.0 / sin(theta7);
  df4t7 = DF4_KK(theta7, p_a_xst7, p_theta_xst7_0, p_dtheta_xst7_ast,
        p_b_xst7, p_dtheta_xst7_c) -
        DF4_KK(theta7p, p_a_xst7, p_theta_xst7_0, p_dtheta_xst7_ast,
        p_b_xst7, p_dtheta_xst7_c);
  df4t7 *= rsint;
  rsint = 1.0 / sin(theta8);
  df4t8 = DF4_KK(theta8, p_a_xst8, p_theta_xst8_0, p_dtheta_xst8_ast,
        p_b_xst8, p_dtheta_xst8_c) -
        DF4_KK(theta8p, p_a_xst8, p_theta_xst8_0, p_dtheta_xst8_ast,
        p_b_xst8, p_dtheta_xst8_c);
  df4t8 *= rsint;

  delf[0] = 0.0;
  delf[1] = 0.0;
  delf[2] = 0.0;

  delta[0] = 0.0;
  delta[1] = 0.0;
  delta[2] = 0.0;

  deltb[0] = 0.0;
  deltb[1] = 0.0;
  deltb[2] = 0.0;

  finc  = -df2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;

  delf[0] = fma(delr_hb[0], finc, delf[0]);
  delf[1] = fma(delr_hb[1], finc, delf[1]);
  delf[2] = fma(delr_hb[2], finc, delf[2]);

  if (theta2) {
    finc  = -f2 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;
    const KK_FLOAT t2f0 = fma(delr_hb_norm[0], cost2, a_nx0);
    const KK_FLOAT t2f1 = fma(delr_hb_norm[1], cost2, a_nx1);
    const KK_FLOAT t2f2 = fma(delr_hb_norm[2], cost2, a_nx2);
    delf[0] = fma(t2f0, finc, delf[0]);
    delf[1] = fma(t2f1, finc, delf[1]);
    delf[2] = fma(t2f2, finc, delf[2]);
  }

  if (theta3) {
    finc  = -f2 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * rinv_hb * factor_lj;
    const KK_FLOAT t3f0 = fma(delr_hb_norm[0], cost3, -b_nx0);
    const KK_FLOAT t3f1 = fma(delr_hb_norm[1], cost3, -b_nx1);
    const KK_FLOAT t3f2 = fma(delr_hb_norm[2], cost3, -b_nx2);
    delf[0] = fma(t3f0, finc, delf[0]);
    delf[1] = fma(t3f1, finc, delf[1]);
    delf[2] = fma(t3f2, finc, delf[2]);
  }

  if (theta7) {
    finc  = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * rinv_hb * factor_lj;
    const KK_FLOAT t7f0 = fma(delr_hb_norm[0], cost7, a_nz0);
    const KK_FLOAT t7f1 = fma(delr_hb_norm[1], cost7, a_nz1);
    const KK_FLOAT t7f2 = fma(delr_hb_norm[2], cost7, a_nz2);
    delf[0] = fma(t7f0, finc, delf[0]);
    delf[1] = fma(t7f1, finc, delf[1]);
    delf[2] = fma(t7f2, finc, delf[2]);
  }

  if (theta8) {
    finc  = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * rinv_hb * factor_lj;
    const KK_FLOAT t8f0 = fma(delr_hb_norm[0], cost8, -b_nz0);
    const KK_FLOAT t8f1 = fma(delr_hb_norm[1], cost8, -b_nz1);
    const KK_FLOAT t8f2 = fma(delr_hb_norm[2], cost8, -b_nz2);
    delf[0] = fma(t8f0, finc, delf[0]);
    delf[1] = fma(t8f1, finc, delf[1]);
    delf[2] = fma(t8f2, finc, delf[2]);
  }

  a_f(a,0) += delf[0];
  a_f(a,1) += delf[1];
  a_f(a,2) += delf[2];
  delta[0] = fma(ra_chb[1], delf[2], -ra_chb[2] * delf[1]);
  delta[1] = fma(ra_chb[2], delf[0], -ra_chb[0] * delf[2]);
  delta[2] = fma(ra_chb[0], delf[1], -ra_chb[1] * delf[0]);
  a_torque(a,0) += delta[0];
  a_torque(a,1) += delta[1];
  a_torque(a,2) += delta[2];

  if ( (NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || b < nlocal) ) {
    a_f(b,0) -= delf[0];
    a_f(b,1) -= delf[1];
    a_f(b,2) -= delf[2];
    deltb[0] = fma(rb_chb[1], delf[2], -rb_chb[2] * delf[1]);
    deltb[1] = fma(rb_chb[2], delf[0], -rb_chb[0] * delf[2]);
    deltb[2] = fma(rb_chb[0], delf[1], -rb_chb[1] * delf[0]);
    a_torque(b,0) -= deltb[0];
    a_torque(b,1) -= deltb[1];
    a_torque(b,2) -= deltb[2];
  }

  if (EVFLAG) {
    ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(b<nlocal)))?1.0:0.5)*evdwl;

    if (vflag_either || eflag_atom) {
      this->template ev_tally_xyz<NEIGHFLAG,NEWTON_PAIR>(ev,a,b,ev.evdwl,\
      delf[0],delf[1],delf[2],x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2));
    }
  }

  delta[0] = 0.0;
  delta[1] = 0.0;
  delta[2] = 0.0;
  deltb[0] = 0.0;
  deltb[1] = 0.0;
  deltb[2] = 0.0;

  if (theta1) {
    tpair = -f2 * df4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj;
    const KK_FLOAT t1dir0 = fma(a_nx1, b_nx2, -a_nx2 * b_nx1);
    const KK_FLOAT t1dir1 = fma(a_nx2, b_nx0, -a_nx0 * b_nx2);
    const KK_FLOAT t1dir2 = fma(a_nx0, b_nx1, -a_nx1 * b_nx0);
    delta[0] += t1dir0 * tpair;
    delta[1] += t1dir1 * tpair;
    delta[2] += t1dir2 * tpair;
    deltb[0] += t1dir0 * tpair;
    deltb[1] += t1dir1 * tpair;
    deltb[2] += t1dir2 * tpair;
  }
  if (theta2) {
    tpair = -f2 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factor_lj;
    const KK_FLOAT t2dir0 = fma(a_nx1, delr_hb_norm[2], -a_nx2 * delr_hb_norm[1]);
    const KK_FLOAT t2dir1 = fma(a_nx2, delr_hb_norm[0], -a_nx0 * delr_hb_norm[2]);
    const KK_FLOAT t2dir2 = fma(a_nx0, delr_hb_norm[1], -a_nx1 * delr_hb_norm[0]);
    delta[0] += t2dir0 * tpair;
    delta[1] += t2dir1 * tpair;
    delta[2] += t2dir2 * tpair;
  }
  if (theta3) {
    tpair = -f2 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * factor_lj;
    const KK_FLOAT t3dir0 = fma(b_nx1, delr_hb_norm[2], -b_nx2 * delr_hb_norm[1]);
    const KK_FLOAT t3dir1 = fma(b_nx2, delr_hb_norm[0], -b_nx0 * delr_hb_norm[2]);
    const KK_FLOAT t3dir2 = fma(b_nx0, delr_hb_norm[1], -b_nx1 * delr_hb_norm[0]);
    deltb[0] += t3dir0 * tpair;
    deltb[1] += t3dir1 * tpair;
    deltb[2] += t3dir2 * tpair;
  }
  if (theta4 && theta4p) {
    tpair = -f2 * f4t1 * f4t2 * f4t3 * df4t4 * f4t7 * f4t8 * factor_lj;
    const KK_FLOAT t4dir0 = fma(b_nz1, a_nz2, -b_nz2 * a_nz1);
    const KK_FLOAT t4dir1 = fma(b_nz2, a_nz0, -b_nz0 * a_nz2);
    const KK_FLOAT t4dir2 = fma(b_nz0, a_nz1, -b_nz1 * a_nz0);
    delta[0] += t4dir0 * tpair;
    delta[1] += t4dir1 * tpair;
    delta[2] += t4dir2 * tpair;
    deltb[0] += t4dir0 * tpair;
    deltb[1] += t4dir1 * tpair;
    deltb[2] += t4dir2 * tpair;
  }
  if (theta7) {
    tpair = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * factor_lj;
    const KK_FLOAT t7dir0 = fma(a_nz1, delr_hb_norm[2], -a_nz2 * delr_hb_norm[1]);
    const KK_FLOAT t7dir1 = fma(a_nz2, delr_hb_norm[0], -a_nz0 * delr_hb_norm[2]);
    const KK_FLOAT t7dir2 = fma(a_nz0, delr_hb_norm[1], -a_nz1 * delr_hb_norm[0]);
    delta[0] += t7dir0 * tpair;
    delta[1] += t7dir1 * tpair;
    delta[2] += t7dir2 * tpair;
  }
  if (theta8) {
    tpair = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * factor_lj;
    const KK_FLOAT t8dir0 = fma(b_nz1, delr_hb_norm[2], -b_nz2 * delr_hb_norm[1]);
    const KK_FLOAT t8dir1 = fma(b_nz2, delr_hb_norm[0], -b_nz0 * delr_hb_norm[2]);
    const KK_FLOAT t8dir2 = fma(b_nz0, delr_hb_norm[1], -b_nz1 * delr_hb_norm[0]);
    deltb[0] += t8dir0 * tpair;
    deltb[1] += t8dir1 * tpair;
    deltb[2] += t8dir2 * tpair;
  }

  a_torque(a,0) += delta[0];
  a_torque(a,1) += delta[1];
  a_torque(a,2) += delta[2];

  if ( (NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || b < nlocal) ) {
    a_torque(b,0) -= deltb[0];
    a_torque(b,1) -= deltb[1];
    a_torque(b,2) -= deltb[2];
  }
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaXstkKokkos<DeviceType>::operator()(TagPairOxdnaXstkComputeGPUPair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>,
  const int &ipair) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG>
    (TagPairOxdnaXstkComputeGPUPair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(),ipair,ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaXstkKokkos<DeviceType>::allocate()
{
  PairOxdnaXstk::allocate();

  int n = atom->ntypes;

  memoryKK->create_kokkos(k_k_xst,n+1,n+1,"PairOxdnaXstk:xst");
  memoryKK->create_kokkos(k_cut_xst_0,n+1,n+1,"PairOxdnaXstk:cut_xst_0");
  memoryKK->create_kokkos(k_cut_xst_c,n+1,n+1,"PairOxdnaXstk:cut_xst_c");
  memoryKK->create_kokkos(k_cut_xst_lo,n+1,n+1,"PairOxdnaXstk:cut_xst_lo");
  memoryKK->create_kokkos(k_cut_xst_hi,n+1,n+1,"PairOxdnaXstk:cut_xst_hi");
  memoryKK->create_kokkos(k_cut_xst_lc,n+1,n+1,"PairOxdnaXstk:cut_xst_lc");
  memoryKK->create_kokkos(k_cut_xst_hc,n+1,n+1,"PairOxdnaXstk:cut_xst_hc");
  memoryKK->create_kokkos(k_b_xst_lo,n+1,n+1,"PairOxdnaXstk:b_xst_lo");
  memoryKK->create_kokkos(k_b_xst_hi,n+1,n+1,"PairOxdnaXstk:b_xst_hi");
  memoryKK->create_kokkos(k_cutsq_xst_hc,n+1,n+1,"PairOxdnaXstk:cutsq_xst_hc");

  memoryKK->create_kokkos(k_a_xst1,n+1,n+1,"PairOxdnaXstk:a_xst1");
  memoryKK->create_kokkos(k_theta_xst1_0,n+1,n+1,"PairOxdnaXstk:theta_xst1_0");
  memoryKK->create_kokkos(k_dtheta_xst1_ast,n+1,n+1,"PairOxdnaXstk:dtheta_xst1_ast");
  memoryKK->create_kokkos(k_b_xst1,n+1,n+1,"PairOxdnaXstk:b_xst1");
  memoryKK->create_kokkos(k_dtheta_xst1_c,n+1,n+1,"PairOxdnaXstk:dtheta_xst1_c");

  memoryKK->create_kokkos(k_a_xst2,n+1,n+1,"PairOxdnaXstk:a_xst2");
  memoryKK->create_kokkos(k_theta_xst2_0,n+1,n+1,"PairOxdnaXstk:theta_xst2_0");
  memoryKK->create_kokkos(k_dtheta_xst2_ast,n+1,n+1,"PairOxdnaXstk:dtheta_xst2_ast");
  memoryKK->create_kokkos(k_b_xst2,n+1,n+1,"PairOxdnaXstk:b_xst2");
  memoryKK->create_kokkos(k_dtheta_xst2_c,n+1,n+1,"PairOxdnaXstk:dtheta_xst2_c");

  memoryKK->create_kokkos(k_a_xst3,n+1,n+1,"PairOxdnaXstk:a_xst3");
  memoryKK->create_kokkos(k_theta_xst3_0,n+1,n+1,"PairOxdnaXstk:theta_xst3_0");
  memoryKK->create_kokkos(k_dtheta_xst3_ast,n+1,n+1,"PairOxdnaXstk:dtheta_xst3_ast");
  memoryKK->create_kokkos(k_b_xst3,n+1,n+1,"PairOxdnaXstk:b_xst3");
  memoryKK->create_kokkos(k_dtheta_xst3_c,n+1,n+1,"PairOxdnaXstk:dtheta_xst3_c");

  memoryKK->create_kokkos(k_a_xst4,n+1,n+1,"PairOxdnaXstk:a_xst4");
  memoryKK->create_kokkos(k_theta_xst4_0,n+1,n+1,"PairOxdnaXstk:theta_xst4_0");
  memoryKK->create_kokkos(k_dtheta_xst4_ast,n+1,n+1,"PairOxdnaXstk:dtheta_xst4_ast");
  memoryKK->create_kokkos(k_b_xst4,n+1,n+1,"PairOxdnaXstk:b_xst4");
  memoryKK->create_kokkos(k_dtheta_xst4_c,n+1,n+1,"PairOxdnaXstk:dtheta_xst4_c");

  memoryKK->create_kokkos(k_a_xst7,n+1,n+1,"PairOxdnaXstk:a_xst7");
  memoryKK->create_kokkos(k_theta_xst7_0,n+1,n+1,"PairOxdnaXstk:theta_xst7_0");
  memoryKK->create_kokkos(k_dtheta_xst7_ast,n+1,n+1,"PairOxdnaXstk:dtheta_xst7_ast");
  memoryKK->create_kokkos(k_b_xst7,n+1,n+1,"PairOxdnaXstk:b_xst7");
  memoryKK->create_kokkos(k_dtheta_xst7_c,n+1,n+1,"PairOxdnaXstk:dtheta_xst7_c");

  memoryKK->create_kokkos(k_a_xst8,n+1,n+1,"PairOxdnaXstk:a_xst8");
  memoryKK->create_kokkos(k_theta_xst8_0,n+1,n+1,"PairOxdnaXstk:theta_xst8_0");
  memoryKK->create_kokkos(k_dtheta_xst8_ast,n+1,n+1,"PairOxdnaXstk:dtheta_xst8_ast");
  memoryKK->create_kokkos(k_b_xst8,n+1,n+1,"PairOxdnaXstk:b_xst8");
  memoryKK->create_kokkos(k_dtheta_xst8_c,n+1,n+1,"PairOxdnaXstk:dtheta_xst8_c");

  d_k_xst = k_k_xst.template view<DeviceType>();
  d_cut_xst_0 = k_cut_xst_0.template view<DeviceType>();
  d_cut_xst_c = k_cut_xst_c.template view<DeviceType>();
  d_cut_xst_lo = k_cut_xst_lo.template view<DeviceType>();
  d_cut_xst_hi = k_cut_xst_hi.template view<DeviceType>();
  d_cut_xst_lc = k_cut_xst_lc.template view<DeviceType>();
  d_cut_xst_hc = k_cut_xst_hc.template view<DeviceType>();
  d_b_xst_lo = k_b_xst_lo.template view<DeviceType>();
  d_b_xst_hi = k_b_xst_hi.template view<DeviceType>();
  d_cutsq_xst_hc = k_cutsq_xst_hc.template view<DeviceType>();

  d_a_xst1 = k_a_xst1.template view<DeviceType>();
  d_theta_xst1_0 = k_theta_xst1_0.template view<DeviceType>();
  d_dtheta_xst1_ast = k_dtheta_xst1_ast.template view<DeviceType>();
  d_b_xst1 = k_b_xst1.template view<DeviceType>();
  d_dtheta_xst1_c = k_dtheta_xst1_c.template view<DeviceType>();

  d_a_xst2 = k_a_xst2.template view<DeviceType>();
  d_theta_xst2_0 = k_theta_xst2_0.template view<DeviceType>();
  d_dtheta_xst2_ast = k_dtheta_xst2_ast.template view<DeviceType>();
  d_b_xst2 = k_b_xst2.template view<DeviceType>();
  d_dtheta_xst2_c = k_dtheta_xst2_c.template view<DeviceType>();

  d_a_xst3 = k_a_xst3.template view<DeviceType>();
  d_theta_xst3_0 = k_theta_xst3_0.template view<DeviceType>();
  d_dtheta_xst3_ast = k_dtheta_xst3_ast.template view<DeviceType>();
  d_b_xst3 = k_b_xst3.template view<DeviceType>();
  d_dtheta_xst3_c = k_dtheta_xst3_c.template view<DeviceType>();

  d_a_xst4 = k_a_xst4.template view<DeviceType>();
  d_theta_xst4_0 = k_theta_xst4_0.template view<DeviceType>();
  d_dtheta_xst4_ast = k_dtheta_xst4_ast.template view<DeviceType>();
  d_b_xst4 = k_b_xst4.template view<DeviceType>();
  d_dtheta_xst4_c = k_dtheta_xst4_c.template view<DeviceType>();

  d_a_xst7 = k_a_xst7.template view<DeviceType>();
  d_theta_xst7_0 = k_theta_xst7_0.template view<DeviceType>();
  d_dtheta_xst7_ast = k_dtheta_xst7_ast.template view<DeviceType>();
  d_b_xst7 = k_b_xst7.template view<DeviceType>();
  d_dtheta_xst7_c = k_dtheta_xst7_c.template view<DeviceType>();

  d_a_xst8 = k_a_xst8.template view<DeviceType>();
  d_theta_xst8_0 = k_theta_xst8_0.template view<DeviceType>();
  d_dtheta_xst8_ast = k_dtheta_xst8_ast.template view<DeviceType>();
  d_b_xst8 = k_b_xst8.template view<DeviceType>();
  d_dtheta_xst8_c = k_dtheta_xst8_c.template view<DeviceType>();

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaXstkKokkos<DeviceType>::settings(int narg, char **/*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaXstkKokkos<DeviceType>::init_style() 
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
double PairOxdnaXstkKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairOxdnaXstk::init_one(i,j);

  // Assign directionally: [i][j] gets [i][j], [j][i] gets [j][i]
  k_k_xst.view_host()(i,j) = k_xst[i][j]; k_k_xst.view_host()(j,i) = k_xst[j][i];
  k_cut_xst_0.view_host()(i,j) = cut_xst_0[i][j]; k_cut_xst_0.view_host()(j,i) = cut_xst_0[j][i];
  k_cut_xst_c.view_host()(i,j) = cut_xst_c[i][j]; k_cut_xst_c.view_host()(j,i) = cut_xst_c[j][i];
  k_cut_xst_lo.view_host()(i,j) = cut_xst_lo[i][j]; k_cut_xst_lo.view_host()(j,i) = cut_xst_lo[j][i];
  k_cut_xst_hi.view_host()(i,j) = cut_xst_hi[i][j]; k_cut_xst_hi.view_host()(j,i) = cut_xst_hi[j][i];
  k_cut_xst_lc.view_host()(i,j) = cut_xst_lc[i][j]; k_cut_xst_lc.view_host()(j,i) = cut_xst_lc[j][i];
  k_cut_xst_hc.view_host()(i,j) = cut_xst_hc[i][j]; k_cut_xst_hc.view_host()(j,i) = cut_xst_hc[j][i];
  k_b_xst_lo.view_host()(i,j) = b_xst_lo[i][j]; k_b_xst_lo.view_host()(j,i) = b_xst_lo[j][i];
  k_b_xst_hi.view_host()(i,j) = b_xst_hi[i][j]; k_b_xst_hi.view_host()(j,i) = b_xst_hi[j][i];
  k_cutsq_xst_hc.view_host()(i,j) = cutsq_xst_hc[i][j]; k_cutsq_xst_hc.view_host()(j,i) = cutsq_xst_hc[j][i];

  k_a_xst1.view_host()(i,j) = a_xst1[i][j]; k_a_xst1.view_host()(j,i) = a_xst1[j][i];
  k_theta_xst1_0.view_host()(i,j) = theta_xst1_0[i][j]; k_theta_xst1_0.view_host()(j,i) = theta_xst1_0[j][i];
  k_dtheta_xst1_ast.view_host()(i,j) = dtheta_xst1_ast[i][j]; k_dtheta_xst1_ast.view_host()(j,i) = dtheta_xst1_ast[j][i];
  k_b_xst1.view_host()(i,j) = b_xst1[i][j]; k_b_xst1.view_host()(j,i) = b_xst1[j][i];
  k_dtheta_xst1_c.view_host()(i,j) = dtheta_xst1_c[i][j]; k_dtheta_xst1_c.view_host()(j,i) = dtheta_xst1_c[j][i];

  k_a_xst2.view_host()(i,j) = a_xst2[i][j]; k_a_xst2.view_host()(j,i) = a_xst2[j][i];
  k_theta_xst2_0.view_host()(i,j) = theta_xst2_0[i][j]; k_theta_xst2_0.view_host()(j,i) = theta_xst2_0[j][i];
  k_dtheta_xst2_ast.view_host()(i,j) = dtheta_xst2_ast[i][j]; k_dtheta_xst2_ast.view_host()(j,i) = dtheta_xst2_ast[j][i];
  k_b_xst2.view_host()(i,j) = b_xst2[i][j]; k_b_xst2.view_host()(j,i) = b_xst2[j][i];
  k_dtheta_xst2_c.view_host()(i,j) = dtheta_xst2_c[i][j]; k_dtheta_xst2_c.view_host()(j,i) = dtheta_xst2_c[j][i];

  k_a_xst3.view_host()(i,j) = a_xst3[i][j]; k_a_xst3.view_host()(j,i) = a_xst3[j][i];
  k_theta_xst3_0.view_host()(i,j) = theta_xst3_0[i][j]; k_theta_xst3_0.view_host()(j,i) = theta_xst3_0[j][i];
  k_dtheta_xst3_ast.view_host()(i,j) = dtheta_xst3_ast[i][j]; k_dtheta_xst3_ast.view_host()(j,i) = dtheta_xst3_ast[j][i];
  k_b_xst3.view_host()(i,j) = b_xst3[i][j]; k_b_xst3.view_host()(j,i) = b_xst3[j][i];
  k_dtheta_xst3_c.view_host()(i,j) = dtheta_xst3_c[i][j]; k_dtheta_xst3_c.view_host()(j,i) = dtheta_xst3_c[j][i];

  k_a_xst4.view_host()(i,j) = a_xst4[i][j]; k_a_xst4.view_host()(j,i) = a_xst4[j][i];
  k_theta_xst4_0.view_host()(i,j) = theta_xst4_0[i][j]; k_theta_xst4_0.view_host()(j,i) = theta_xst4_0[j][i];
  k_dtheta_xst4_ast.view_host()(i,j) = dtheta_xst4_ast[i][j]; k_dtheta_xst4_ast.view_host()(j,i) = dtheta_xst4_ast[j][i];
  k_b_xst4.view_host()(i,j) = b_xst4[i][j]; k_b_xst4.view_host()(j,i) = b_xst4[j][i];
  k_dtheta_xst4_c.view_host()(i,j) = dtheta_xst4_c[i][j]; k_dtheta_xst4_c.view_host()(j,i) = dtheta_xst4_c[j][i];

  k_a_xst7.view_host()(i,j) = a_xst7[i][j]; k_a_xst7.view_host()(j,i) = a_xst7[j][i];
  k_theta_xst7_0.view_host()(i,j) = theta_xst7_0[i][j]; k_theta_xst7_0.view_host()(j,i) = theta_xst7_0[j][i];
  k_dtheta_xst7_ast.view_host()(i,j) = dtheta_xst7_ast[i][j]; k_dtheta_xst7_ast.view_host()(j,i) = dtheta_xst7_ast[j][i];
  k_b_xst7.view_host()(i,j) = b_xst7[i][j]; k_b_xst7.view_host()(j,i) = b_xst7[j][i];
  k_dtheta_xst7_c.view_host()(i,j) = dtheta_xst7_c[i][j]; k_dtheta_xst7_c.view_host()(j,i) = dtheta_xst7_c[j][i];

  k_a_xst8.view_host()(i,j) = a_xst8[i][j]; k_a_xst8.view_host()(j,i) = a_xst8[j][i];
  k_theta_xst8_0.view_host()(i,j) = theta_xst8_0[i][j]; k_theta_xst8_0.view_host()(j,i) = theta_xst8_0[j][i];
  k_dtheta_xst8_ast.view_host()(i,j) = dtheta_xst8_ast[i][j]; k_dtheta_xst8_ast.view_host()(j,i) = dtheta_xst8_ast[j][i];
  k_b_xst8.view_host()(i,j) = b_xst8[i][j]; k_b_xst8.view_host()(j,i) = b_xst8[j][i];
  k_dtheta_xst8_c.view_host()(i,j) = dtheta_xst8_c[i][j]; k_dtheta_xst8_c.view_host()(j,i) = dtheta_xst8_c[j][i];

  k_k_xst.template modify<LMPHostType>();
  k_cut_xst_0.template modify<LMPHostType>();
  k_cut_xst_c.template modify<LMPHostType>();
  k_cut_xst_lo.template modify<LMPHostType>();
  k_cut_xst_hi.template modify<LMPHostType>();
  k_cut_xst_lc.template modify<LMPHostType>();
  k_cut_xst_hc.template modify<LMPHostType>();
  k_b_xst_lo.template modify<LMPHostType>();
  k_b_xst_hi.template modify<LMPHostType>();
  k_cutsq_xst_hc.template modify<LMPHostType>();

  k_a_xst1.template modify<LMPHostType>();
  k_theta_xst1_0.template modify<LMPHostType>();
  k_dtheta_xst1_ast.template modify<LMPHostType>();
  k_b_xst1.template modify<LMPHostType>();
  k_dtheta_xst1_c.template modify<LMPHostType>();

  k_a_xst2.template modify<LMPHostType>();
  k_theta_xst2_0.template modify<LMPHostType>();
  k_dtheta_xst2_ast.template modify<LMPHostType>();
  k_b_xst2.template modify<LMPHostType>();
  k_dtheta_xst2_c.template modify<LMPHostType>();

  k_a_xst3.template modify<LMPHostType>();
  k_theta_xst3_0.template modify<LMPHostType>();
  k_dtheta_xst3_ast.template modify<LMPHostType>();
  k_b_xst3.template modify<LMPHostType>();
  k_dtheta_xst3_c.template modify<LMPHostType>();

  k_a_xst4.template modify<LMPHostType>();
  k_theta_xst4_0.template modify<LMPHostType>();
  k_dtheta_xst4_ast.template modify<LMPHostType>();
  k_b_xst4.template modify<LMPHostType>();
  k_dtheta_xst4_c.template modify<LMPHostType>();

  k_a_xst7.template modify<LMPHostType>();
  k_theta_xst7_0.template modify<LMPHostType>();
  k_dtheta_xst7_ast.template modify<LMPHostType>();
  k_b_xst7.template modify<LMPHostType>();
  k_dtheta_xst7_c.template modify<LMPHostType>();

  k_a_xst8.template modify<LMPHostType>();
  k_theta_xst8_0.template modify<LMPHostType>();
  k_dtheta_xst8_ast.template modify<LMPHostType>();
  k_b_xst8.template modify<LMPHostType>();
  k_dtheta_xst8_c.template modify<LMPHostType>();

  // Sync to device
  k_k_xst.template sync<DeviceType>();
  k_cut_xst_0.template sync<DeviceType>();
  k_cut_xst_c.template sync<DeviceType>();
  k_cut_xst_lo.template sync<DeviceType>();
  k_cut_xst_hi.template sync<DeviceType>();
  k_cut_xst_lc.template sync<DeviceType>();
  k_cut_xst_hc.template sync<DeviceType>();
  k_b_xst_lo.template sync<DeviceType>();
  k_b_xst_hi.template sync<DeviceType>();
  k_cutsq_xst_hc.template sync<DeviceType>();

  k_a_xst1.template sync<DeviceType>();
  k_theta_xst1_0.template sync<DeviceType>();
  k_dtheta_xst1_ast.template sync<DeviceType>();
  k_b_xst1.template sync<DeviceType>();
  k_dtheta_xst1_c.template sync<DeviceType>();

  k_a_xst2.template sync<DeviceType>();
  k_theta_xst2_0.template sync<DeviceType>();
  k_dtheta_xst2_ast.template sync<DeviceType>();
  k_b_xst2.template sync<DeviceType>();
  k_dtheta_xst2_c.template sync<DeviceType>();

  k_a_xst3.template sync<DeviceType>();
  k_theta_xst3_0.template sync<DeviceType>();
  k_dtheta_xst3_ast.template sync<DeviceType>();
  k_b_xst3.template sync<DeviceType>();
  k_dtheta_xst3_c.template sync<DeviceType>();

  k_a_xst4.template sync<DeviceType>();
  k_theta_xst4_0.template sync<DeviceType>();
  k_dtheta_xst4_ast.template sync<DeviceType>();
  k_b_xst4.template sync<DeviceType>();
  k_dtheta_xst4_c.template sync<DeviceType>();

  k_a_xst7.template sync<DeviceType>();
  k_theta_xst7_0.template sync<DeviceType>();
  k_dtheta_xst7_ast.template sync<DeviceType>();
  k_b_xst7.template sync<DeviceType>();
  k_dtheta_xst7_c.template sync<DeviceType>();

  k_a_xst8.template sync<DeviceType>();
  k_theta_xst8_0.template sync<DeviceType>();
  k_dtheta_xst8_ast.template sync<DeviceType>();
  k_b_xst8.template sync<DeviceType>();
  k_dtheta_xst8_c.template sync<DeviceType>();

  // "cutone" is "cut_xst_hc[i][j]", sets the master list distance cutoff
  return cutone;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairOxdnaXstkKokkos<DeviceType>::ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
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
int PairOxdnaXstkKokkos<DeviceType>::sbmask(const int& j) const {
  return j >> SBBITS & 3;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairOxdnaXstkKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairOxdnaXstkKokkos<LMPHostType>;
#endif
}