// clang-format off
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

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mpi.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "force.h"
#include "kokkos.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "my_page.h"
#include "math_const.h"
#include "math_special.h"
#include "memory_kokkos.h"
#include "error.h"
#include "pair_chimes_kokkos.h"
#include "group.h"
#include "update.h" // Needed for mb neighlist updates and info printing for fitting
#include "output.h" // Needed for infor printing for fitting -- dump 1 must be the "main" dump file used for fitting
#include "utils.h"  // Needed for infor printing for fitting
//#include <iostream>
//#include <sstream>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
PairCHIMESKokkos<DeviceType, vector_length>::PairCHIMESKokkos(LAMMPS *lmp) : PairCHIMES(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  host_flag = (execution_space == HostKK);
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

#ifdef TABULATION
 error->all(FLERR,"Cannot (yet) use tabulation with pair_style chimes/kk");
#endif

#ifdef FINGERPRINT
 error->all(FLERR,"Cannot (yet) use fingerprint with pair_style chimes/kk");
#endif

  chimes_calculatorKK.init(comm->me);   // chimesFF instance

  delete chimes_calculator;
  chimes_calculator = (chimesFF*) (&chimes_calculatorKK);

  d_size_4mers = typename AT::t_int_scalar("pair:size_4mers");

  max_2mers = 1;
  max_3mers = 1;
  max_4mers = 1;

  // Number of owned atoms processed per chunk. The many-body (2/3/4-mer)
  // neighbor lists are built and consumed one chunk at a time so their peak
  // memory is bounded by the per-chunk cluster counts rather than the whole
  // system (mirrors the Kokkos SNAP chunksize mechanism). Results are
  // independent of chunksize; lower it if the 3/4-body lists exhaust GPU memory.
  chunksize = 32768;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
PairCHIMESKokkos<DeviceType, vector_length>::~PairCHIMESKokkos()
{
  if (copymode) return;

  chimes_calculator = nullptr;

  /*if (allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }*/

  memoryKK->destroy_kokkos(k_eatom,eatom);
  memoryKK->destroy_kokkos(k_vatom,vatom);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
void PairCHIMESKokkos<DeviceType, vector_length>::allocate()
{
  PairCHIMES::allocate();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
void PairCHIMESKokkos<DeviceType, vector_length>::init_style()
{
  PairCHIMES::init_style();

  // adjust neighbor list request for KOKKOS

  auto request = neighbor->find_request(this);

  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);

  neighflag = lmp->kokkos->neighflag;

  if (neighflag == FULL)
    error->all(FLERR,"Must use half neighbor list style with pair chimes/kk");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
void PairCHIMESKokkos<DeviceType, vector_length>::settings(int narg, char **arg)
{
  // Optional "chunksize N" keyword bounds the per-chunk many-body list memory.
  // With no arguments the default chunksize is kept (matching the base style).

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "chunksize") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal pair_style chimesFF/kk command: chunksize requires a value");
      chunksize = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (chunksize <= 0)
        error->all(FLERR, "pair_style chimesFF/kk chunksize must be > 0");
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal pair_style chimesFF/kk argument: {}", arg[iarg]);
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
void PairCHIMESKokkos<DeviceType, vector_length>::coeff(int narg, char **arg)
{
  PairCHIMES::coeff(narg,arg);

  if (chimes_calculatorKK.poly_orders[0]+1 > MAX_2B_POLY)
    lmp->error->all(FLERR,"Exceeded maximum poly order for 2-body interactions, "
                    "increase value of MAX_2B_POLY in src/KOKKOS/chimes_kokkos.h "
                    "and recompile");

  if (chimes_calculatorKK.poly_orders[1]+1 > MAX_3B_POLY)
    lmp->error->all(FLERR,"Exceeded maximum poly order for 3-body interactions, "
                    "increase value of MAX_3B_POLY in src/KOKKOS/chimes_kokkos.h "
                    "and recompile");

  if (chimes_calculatorKK.poly_orders[2]+1 > MAX_4B_POLY)
    lmp->error->all(FLERR,"Exceeded maximum poly order for 4-body interactions, "
                    "increase value of MAX_4B_POLY in src/KOKKOS/chimes_kokkos.h "
                    "and recompile");

  // chimes_type

  int size = chimes_type.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_type,"chimes:chimes_type",size);

  auto h_chimes_type = Kokkos::create_mirror_view(d_chimes_type);

  for (int i = 0; i < size; i++)
    h_chimes_type[i] = chimes_type[i];

  Kokkos::deep_copy(d_chimes_type,h_chimes_type);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairCHIMESKokkos<DeviceType, vector_length>::get_dist(int i, int j, KK_FLOAT *dr) const
{
  dr[0] = x(j,0) - x(i,0);
  dr[1] = x(j,1) - x(i,1);
  dr[2] = x(j,2) - x(i,2);

  return sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairCHIMESKokkos<DeviceType, vector_length>::get_dist(int i, int j) const
{
  KK_FLOAT dummy_dr[3];

  return get_dist(i,j, dummy_dr);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESComputeShortNeigh, const int& ii) const
{
  // Build a per-owned-atom compacted list of neighbors within maxcut_3b, so the
  // 3-/4-body enumeration scans only nearby candidates (maxcut_4b <= maxcut_3b
  // <= maxcut_2b) instead of the full 2-body neighbor list. Stored at the lead
  // atom's local index i; neighbors are pre-masked and self is removed. The
  // accepted-neighbor order matches the full list, so downstream cluster offsets
  // (and results) are unchanged.

  const int i = d_ilist[ii + chunk_offset];
  const int jnum = d_numneigh[i];

  int inside = 0;
  if constexpr (CHIMES_FUSED_LEAD_REUSE) {
    // Emit neighbors grouped by chimes type (stable counting sort over the small
    // natmtyps buckets) so the fused per-2-mer kernel can reuse the lead pair's
    // Chebyshev arrays across each contiguous typ_k bucket.
    const int natmtyps = chimes_calculatorKK.natmtyps;
    for (int t = 0; t < natmtyps; t++) {
      for (int jj = 0; jj < jnum; jj++) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;
        if (j == i) continue;
        if (d_chimes_type[type[j]-1] != t) continue;
        if (get_dist(i,j) >= maxcut_3b) continue;
        d_neighbors_short(i,inside) = j;
        inside++;
      }
    }
  } else {
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      if (j == i) continue;
      if (get_dist(i,j) >= maxcut_3b) continue;
      d_neighbors_short(i,inside) = j;
      inside++;
    }
  }
  d_numneigh_short(i) = inside;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
void PairCHIMESKokkos<DeviceType, vector_length>::build_mb_neighlists()
{
  if (maxcut_3b > maxcut_2b)
    error->all(FLERR,"KOKKOS ChIMES assumes 2-body cutoffs >= 3-body cutoffs");

  if (maxcut_4b > maxcut_3b)
    error->all(FLERR,"KOKKOS ChIMES assumes 3-body cutoffs >= 4-body cutoffs");

  // List gets built based on atoms owned by calling proc

  if (d_neighborlist_2mers.extent(0) < max_2mers)
    LAMMPS_NS::MemKK::realloc_kokkos(d_neighborlist_2mers,"chimes:neighborlist_2mers",max_2mers);

  if (d_neighborlist_3mers.extent(0) < max_3mers)
    LAMMPS_NS::MemKK::realloc_kokkos(d_neighborlist_3mers,"chimes:neighborlist_3mers",max_3mers);

  if (d_neighborlist_4mers.extent(0) < max_4mers)
    LAMMPS_NS::MemKK::realloc_kokkos(d_neighborlist_4mers,"chimes:neighborlist_4mers",max_4mers);

  // try building 2-body list, resize if necessary

  int resize = 1;
  while (resize) {
    resize = 0;

    PairCHIMESComputeNeigh2BodyFunctor<DeviceType, vector_length> neigh_2B_functor(this);
    Kokkos::parallel_scan("ComputeNeigh2Body", chunk_size, neigh_2B_functor,size_2mers);

    resize = size_2mers > max_2mers;
    if (resize) {
      max_2mers = MAX(max_2mers+MAX(1,max_2mers*0.1),size_2mers);
      LAMMPS_NS::MemKK::realloc_kokkos(d_neighborlist_2mers,"chimes:neighborlist_2mers",max_2mers);
    }
  }

  // build the per-atom short neighbor list (neighbors within maxcut_3b) that the
  // 3-/4-body enumeration below iterates instead of the full 2-body list

  if (chimes_calculatorKK.poly_orders[1] > 0 || chimes_calculatorKK.poly_orders[2] > 0) {
    const int nlocal = atom->nlocal;
    const int maxn = d_neighbors.extent(1);
    if ((int)d_neighbors_short.extent(0) < nlocal || (int)d_neighbors_short.extent(1) < maxn) {
      LAMMPS_NS::MemKK::realloc_kokkos(d_neighbors_short,"chimes:neighbors_short",nlocal,maxn);
      LAMMPS_NS::MemKK::realloc_kokkos(d_numneigh_short,"chimes:numneigh_short",nlocal);
    }
    typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESComputeShortNeigh> policy_short(0,chunk_size);
    Kokkos::parallel_for("ComputeShortNeigh",policy_short,*this);
  }

  // PROTOTYPE: the fused 3B/4B kernels enumerate clusters on the fly from the
  // 2-mer list + short neighbor list, so the (memory-heavy) 3-mer and 4-mer
  // lists are NOT materialized here. The materialized neigh_3B_item /
  // ComputeNeigh4Body builders are intentionally left unused on this branch.
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::neigh_2B_item(const int& ii, int &offset, const bool &final) const
{
  const int i = d_ilist[ii + chunk_offset];
  const tagint itag = tag[i];
  const int jnum = d_numneigh[i];

  int typ_idxs[2];
  typ_idxs[0] = d_chimes_type[type[i]-1]; // Type (index) of the current atom

  const int natmtyps = chimes_calculatorKK.natmtyps;

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i,jj);
    j &= NEIGHMASK;
    const tagint jtag = tag[j];

    if (j == i) continue;
    if (jtag <= itag) continue; // only allow calculation for j<i, since we've requested a f

    // Check ij distance

    const KK_FLOAT dist_ij = get_dist(i,j);

    typ_idxs[1] = d_chimes_type[type[j]-1];

    const int pair_idx = chimes_calculatorKK.c_atom_int_pair_map(typ_idxs[0]*natmtyps + typ_idxs[1]);

    if (dist_ij >= chimes_calculatorKK.c_chimes_2b_cutoff(pair_idx,1)) continue;

    if (final) {
      if (offset < max_2mers) {
        d_neighborlist_2mers(offset,0) = i;
        d_neighborlist_2mers(offset,1) = j;
      }
    }

    offset++;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::neigh_3B_item(const int& ii, int &offset, const bool &final) const
{
  const int i = d_neighborlist_2mers(ii,0);
  const int j = d_neighborlist_2mers(ii,1);

  const KK_FLOAT dist_ij = get_dist(i,j);
  if (dist_ij >= maxcut_3b) return;

  const tagint itag = tag[i];
  const tagint jtag = tag[j];

  const int natmtyps = chimes_calculatorKK.natmtyps;
  const auto &d_atom_int_trip_map = chimes_calculatorKK.c_atom_int_trip_map;
  const auto &d_pair_int_trip_map = chimes_calculatorKK.c_pair_int_trip_map;
  const auto &d_chimes_3b_cutoff = chimes_calculatorKK.c_chimes_3b_cutoff;

  int typ_idxs[3];
  typ_idxs[0] = d_chimes_type[type[i]-1];
  typ_idxs[1] = d_chimes_type[type[j]-1];

  // ChIMES assumes all atoms must be within cutoff of each other for a valid
  // interaction. Iterate the short neighbor list (already within maxcut_3b of i
  // and pre-masked, self removed).
  const int knum = d_numneigh_short[i];

  for (int kk = 0; kk < knum; kk++) {
    int k = d_neighbors_short(i,kk);
    const tagint ktag = tag[k];

    if (k == j) continue;

    if ((ktag < itag) || (ktag < jtag)) continue;

    // Check ik distance

    const KK_FLOAT dist_ik = get_dist(i,k);

    if (dist_ik >= maxcut_3b) continue;

    // Check jk distance

    const KK_FLOAT dist_jk = get_dist(j,k);

    if ((dist_ij < maxcut_3b) && (dist_ik < maxcut_3b) && (dist_jk < maxcut_3b))
    {
      typ_idxs[2] = d_chimes_type[type[k]-1];

      const int type_idx = typ_idxs[0]*natmtyps*natmtyps + typ_idxs[1]*natmtyps + typ_idxs[2];
      const int tripidx = d_atom_int_trip_map[type_idx];

      if (tripidx < 0) // Skipping an excluded interaction
        continue;

      // Check whether cutoffs are within allowed ranges
      //auto d_mapped_pair_idx = d_pair_int_trip_map[type_idx];

      const KK_FLOAT cutoff_0 = d_chimes_3b_cutoff(tripidx,d_pair_int_trip_map(type_idx,0),1);
      if (dist_ij >= cutoff_0) // ij
        continue;

      const KK_FLOAT cutoff_1 = d_chimes_3b_cutoff(tripidx,d_pair_int_trip_map(type_idx,1),1);
      if (dist_ik >= cutoff_1) // ik
        continue;

      const KK_FLOAT cutoff_2 = d_chimes_3b_cutoff(tripidx,d_pair_int_trip_map(type_idx,2),1);
      if (dist_jk >= cutoff_2) // jk
        continue;

      // If we're here and valid_3mer == true, then add the triplet to the chimes neigh list

      if (final) {
        if (offset < max_3mers) {
          d_neighborlist_3mers(offset,0) = i;
          d_neighborlist_3mers(offset,1) = j;
          d_neighborlist_3mers(offset,2) = k;
        }
      }

      offset++;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESComputeNeigh4Body, const int& ii) const
//void PairCHIMESKokkos<DeviceType, vector_length>::neigh_4B_item(const int& ii, int &offset, const bool &final) const
{
  const int i = d_neighborlist_3mers(ii,0);
  const int j = d_neighborlist_3mers(ii,1);
  const int k = d_neighborlist_3mers(ii,2);

  const KK_FLOAT dist_ij = get_dist(i,j);
  const KK_FLOAT dist_ik = get_dist(i,k);
  const KK_FLOAT dist_jk = get_dist(j,k);

  if ((dist_ij >= maxcut_4b) || (dist_ik >= maxcut_4b) || (dist_jk >= maxcut_4b))
    return;

  const tagint itag = tag[i];
  const tagint jtag = tag[j];
  const tagint ktag = tag[k];

  const int natmtyps = chimes_calculatorKK.natmtyps;
  const auto &d_atom_int_quad_map = chimes_calculatorKK.c_atom_int_quad_map;
  const auto &d_pair_int_quad_map = chimes_calculatorKK.c_pair_int_quad_map;
  const auto &d_chimes_4b_cutoff = chimes_calculatorKK.c_chimes_4b_cutoff;

  int typ_idxs[4];
  typ_idxs[0] = d_chimes_type[type[i]-1];
  typ_idxs[1] = d_chimes_type[type[j]-1];
  typ_idxs[2] = d_chimes_type[type[k]-1];

  // Now decide if we should continue on to 4-body neighbor list construction

  const int lnum = d_numneigh_short[i];

  for (int ll = 0; ll < lnum; ll++)
  {
    int l = d_neighbors_short(i,ll);
    const tagint ltag = tag[l];

    if ((l == j) || (l == k)) continue;

    if ((ltag < itag) || (ltag < jtag) || (ltag < ktag))
      continue;

    // Check il distance

    const KK_FLOAT dist_il = get_dist(i,l);
    if (dist_il >= maxcut_4b) continue;

    // Check jl distance

    const KK_FLOAT dist_jl = get_dist(j,l);
    if (dist_jl >= maxcut_4b) continue;

    // Check kl distance

    const KK_FLOAT dist_kl = get_dist(k,l);
    if (dist_kl >= maxcut_4b) continue;

    typ_idxs[3] = d_chimes_type[type[l]-1];

    const int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
        + typ_idxs[1]*natmtyps*natmtyps + typ_idxs[2]*natmtyps + typ_idxs[3];

    const int quadidx = d_atom_int_quad_map[idx];

    if (quadidx < 0) continue; // Skipping an excluded interaction

    const KK_FLOAT cutoff_0 = d_chimes_4b_cutoff(quadidx,d_pair_int_quad_map(idx,0),1);
    if (dist_ij >= cutoff_0) continue; // ij

    const KK_FLOAT cutoff_1 = d_chimes_4b_cutoff(quadidx,d_pair_int_quad_map(idx,1),1);
    if (dist_ik >= cutoff_1) continue; // ik

    const KK_FLOAT cutoff_2 = d_chimes_4b_cutoff(quadidx,d_pair_int_quad_map(idx,2),1);
    if (dist_il >= cutoff_2) continue; // il

    const KK_FLOAT cutoff_3 = d_chimes_4b_cutoff(quadidx,d_pair_int_quad_map(idx,3),1);
    if (dist_jk >= cutoff_3) continue; // jk

    const KK_FLOAT cutoff_4 = d_chimes_4b_cutoff(quadidx,d_pair_int_quad_map(idx,4),1);
    if (dist_jl >= cutoff_4) continue; // jl

    const KK_FLOAT cutoff_5 = d_chimes_4b_cutoff(quadidx,d_pair_int_quad_map(idx,5),1);
    if (dist_kl >= cutoff_5) continue; // kl

    // If we're here and valid_4mer == true, then add the quadruplet to the chimes neigh list

    const int offset = Kokkos::atomic_fetch_add(&d_size_4mers(),1);

    //if (final) {
      if (offset < max_4mers) {
        d_neighborlist_4mers(offset,0) = i;
        d_neighborlist_4mers(offset,1) = j;
        d_neighborlist_4mers(offset,2) = k;
        d_neighborlist_4mers(offset,3) = l;
      }
    //}

    //offset++;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
void PairCHIMESKokkos<DeviceType, vector_length>::compute(int eflag_in, int vflag_in)
{
  copymode = 1;

  // Vars for access to chimesFF compute_XB functions

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();

  // Set up vars controlling if energy/pressure (virial) contributions are computed

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

  if (eflag || vflag) {
    ev_setup(eflag,vflag);
  } else {
    evflag = 0;
    vflag_fdotr = 0;
    vflag_atom = 0;
  }

  chimes_calculatorKK.eflag = eflag_either;
  chimes_calculatorKK.vflag = vflag_either;

  ////////////////////////////////////////
  // Access to (2-body) neighbor list vars
  ////////////////////////////////////////

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }

  EV_FLOAT ev;

  // Build and consume the ChIMES many-body neighbor lists one atom-chunk at a
  // time so their peak memory is bounded by the per-chunk cluster counts
  // (mirrors the Kokkos SNAP chunksize mechanism). Forces and energy/virial
  // accumulate across chunks; results are independent of chunksize.

  for (chunk_offset = 0; chunk_offset < inum; chunk_offset += chunk_size) {
    chunk_size = MIN(chunksize, inum - chunk_offset);

    build_mb_neighlists();

    //Compute1Body
    if (eflag_either) {
      EV_FLOAT ev_tmp;
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute1Body<HALF> > policy_1body(0,chunk_size);
        Kokkos::parallel_reduce("Compute1Body", policy_1body, *this, ev_tmp);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute1Body<HALFTHREAD> > policy_1body(0,chunk_size);
        Kokkos::parallel_reduce("Compute1Body", policy_1body, *this, ev_tmp);
      }
      ev += ev_tmp;
    }

    //Compute2Body
    if (evflag) {
      EV_FLOAT ev_tmp;
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALF,1> > policy_2body(0,size_2mers);
        Kokkos::parallel_reduce("Compute2Body", policy_2body, *this, ev_tmp);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALFTHREAD,1> > policy_2body(0,size_2mers);
        Kokkos::parallel_reduce("Compute2Body", policy_2body, *this, ev_tmp);
      }
      ev += ev_tmp;
    } else {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALF,0> > policy_2body(0,size_2mers);
        Kokkos::parallel_for("Compute2Body", policy_2body, *this);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALFTHREAD,0> > policy_2body(0,size_2mers);
        Kokkos::parallel_for("Compute2Body", policy_2body, *this);
      }
    }

    // Fused3Body (PROTOTYPE): enumerate 3-body clusters on the fly and evaluate
    // them with no materialized 3-mer list. One team per lead unit (owned atom
    // for CHIMES_FUSED_GRANULARITY==0, 2-mer for ==1); the per-cluster Chebyshev
    // reduction runs over the team's ThreadVectorRange exactly as before.
    if (chimes_calculatorKK.poly_orders[1] > 0)
    {
      using LB3 = Kokkos::LaunchBounds<vector_length,chimes_min_blocks_3b>;
      const auto scratch_req_3b = Kokkos::PerTeam(chimes_calculatorKK.scratch_bytes(2*3*MAX_3B_POLY));
      const int league3 = (CHIMES_FUSED_GRANULARITY == 0) ? chunk_size : size_2mers;
      if (evflag) {
        EV_FLOAT ev_tmp;
        if (neighflag == HALF) {
          typename Kokkos::TeamPolicy<DeviceType,LB3,TagPairCHIMESFused3Body<HALF,1> > p(league3,1,vector_length);
          Kokkos::parallel_reduce("Fused3Body", p.set_scratch_size(0,scratch_req_3b), *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::TeamPolicy<DeviceType,LB3,TagPairCHIMESFused3Body<HALFTHREAD,1> > p(league3,1,vector_length);
          Kokkos::parallel_reduce("Fused3Body", p.set_scratch_size(0,scratch_req_3b), *this, ev_tmp);
        }
        ev += ev_tmp;
      } else {
        if (neighflag == HALF) {
          typename Kokkos::TeamPolicy<DeviceType,LB3,TagPairCHIMESFused3Body<HALF,0> > p(league3,1,vector_length);
          Kokkos::parallel_for("Fused3Body", p.set_scratch_size(0,scratch_req_3b), *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::TeamPolicy<DeviceType,LB3,TagPairCHIMESFused3Body<HALFTHREAD,0> > p(league3,1,vector_length);
          Kokkos::parallel_for("Fused3Body", p.set_scratch_size(0,scratch_req_3b), *this);
        }
      }
    }

    // Fused4Body (PROTOTYPE): same scheme for 4-body clusters.
    if (chimes_calculatorKK.poly_orders[2] > 0)
    {
      using LB4 = Kokkos::LaunchBounds<vector_length,chimes_min_blocks_4b>;
      const auto scratch_req_4b = Kokkos::PerTeam(chimes_calculatorKK.scratch_bytes(2*6*MAX_4B_POLY));
      const int league4 = (CHIMES_FUSED_GRANULARITY == 0) ? chunk_size : size_2mers;
      if (evflag) {
        EV_FLOAT ev_tmp;
        if (neighflag == HALF) {
          typename Kokkos::TeamPolicy<DeviceType,LB4,TagPairCHIMESFused4Body<HALF,1> > p(league4,1,vector_length);
          Kokkos::parallel_reduce("Fused4Body", p.set_scratch_size(0,scratch_req_4b), *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::TeamPolicy<DeviceType,LB4,TagPairCHIMESFused4Body<HALFTHREAD,1> > p(league4,1,vector_length);
          Kokkos::parallel_reduce("Fused4Body", p.set_scratch_size(0,scratch_req_4b), *this, ev_tmp);
        }
        ev += ev_tmp;
      } else {
        if (neighflag == HALF) {
          typename Kokkos::TeamPolicy<DeviceType,LB4,TagPairCHIMESFused4Body<HALF,0> > p(league4,1,vector_length);
          Kokkos::parallel_for("Fused4Body", p.set_scratch_size(0,scratch_req_4b), *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::TeamPolicy<DeviceType,LB4,TagPairCHIMESFused4Body<HALFTHREAD,0> > p(league4,1,vector_length);
          Kokkos::parallel_for("Fused4Body", p.set_scratch_size(0,scratch_req_4b), *this);
        }
      }
    }
  }

  if (need_dup)
    Kokkos::Experimental::contribute(f, dup_f);

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
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  atomKK->modified(execution_space,F_MASK);

  copymode = 0;

  // free duplicated memory

  if (need_dup) {
    dup_f     = {};
    dup_vatom = {};
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESCompute1Body<NEIGHFLAG>, const int& ii, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 1-body interactions
  ////////////////////////////////////////

  // First, get the single-atom energy contribution

  const int i = d_ilist[ii + chunk_offset];

  KK_FLOAT energy = 0.0;
  KK_FLOAT stensor[6];
  for (int n = 0; n < 6; n++) stensor[n] = 0.0;

  chimes_calculatorKK.compute_1B(type[i]-1, energy);

  int atmidxlst[6][2];
  atmidxlst[0][0] = i;

  ev_tally_mb<NEIGHFLAG>(1, 0, atmidxlst, energy, stensor, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 2-body interactions
  ////////////////////////////////////////

  // Now move on to two-body force, stress, and energy

  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const int i = d_neighborlist_2mers(ii,0);
  const int j = d_neighborlist_2mers(ii,1);

  const int itype = type(i);
  const tagint itag = tag(i);

  const tagint jtag = tag[j]; // Get j's global atom index (sort of like its "parent")

  //if (jtag <= itag) // only allow calculation for j<i, since we've requested a full neighbor list
  //  return;

  // Get distance using ghost atoms... don't need MIC since we're using ghost atoms

  KK_FLOAT dr[3];
  const KK_FLOAT dist = get_dist(i,j,&dr[0]);

  int typ_idxs_2b[2];
  typ_idxs_2b[0] = d_chimes_type[type[i]-1]; // Type (index) of the current atom... subtract 1 to account for chimesFF vs LAMMPS numbering convention
  typ_idxs_2b[1] = d_chimes_type[type[j]-1];

  KK_FLOAT energy = 0.0;

  KK_FLOAT force_2b[2*CHDIM];
  for (int idx = 0; idx < 3; idx++) {
    force_2b[idx] = 0.0;
    force_2b[CHDIM+idx] = 0.0;
  }

  KK_FLOAT stensor[6];
  for (int n = 0; n < 6; n++) stensor[n] = 0.0;

  chimes_calculatorKK.compute_2B(dist, dr, typ_idxs_2b, force_2b, stensor, energy);      // Auto-updates badness

  for (int idx = 0; idx < 3; idx++) {
    a_f(i,idx) += force_2b[idx];
    a_f(j,idx) += force_2b[CHDIM+idx];
  }

  // "Save"/tally up the energy and stresses to the global virial/energy data objects (see pair.cpp ~ line 1000)
  // Compute pressure, (in contrast to chimes_md) AFTER penalty has been added

  int atmidxlst[6][2];
  if (EVFLAG && vflag_atom)
  {
    atmidxlst[0][0] = i;
    atmidxlst[0][1] = j;
  }

  if (EVFLAG)
    ev_tally_mb<NEIGHFLAG>(2, 1, atmidxlst, energy, stensor, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>, const t_team& team, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 3-body interactions
  ////////////////////////////////////////

  // One team-thread handles one 3-body cluster. The expensive dense Chebyshev
  // coefficient reduction inside compute_3B is parallelized over the team's
  // ThreadVectorRange; all vector lanes redundantly compute the cheap setup and
  // receive the (broadcast) reduced energy/forces. Global force scatter and the
  // energy/virial tally are therefore done exactly once per cluster, guarded by
  // Kokkos::single, to avoid multi-counting across lanes.
  //
  // With CHIMES_CLUSTERS_PER_TEAM_3B == 1 (default) cpt==1, slot==0 and ii is
  // exactly team.league_rank(). When packing clusters per team (force-only
  // launches), each team-thread (team_rank) takes one cluster and its own
  // scratch slot. Tail team-threads past the end are clamped to a valid cluster
  // so they still reach compute_3B's collective team_barrier, but skip the
  // scatter (valid == false).

  constexpr int cpt = EVFLAG ? 1 : CHIMES_CLUSTERS_PER_TEAM_3B;
  const int slot = (cpt == 1) ? 0 : team.team_rank();
  const int ii_raw = team.league_rank() * cpt + slot;
  const bool valid = ii_raw < size_3mers;
  const int ii = valid ? ii_raw : (size_3mers - 1);

  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const int i = d_neighborlist_3mers(ii,0);
  const int j = d_neighborlist_3mers(ii,1);
  const int k = d_neighborlist_3mers(ii,2);

  KK_FLOAT dist_3b[3], dr_3b[3*CHDIM];
  dist_3b[0] = get_dist(i,j,&dr_3b[0*CHDIM]);
  dist_3b[1] = get_dist(i,k,&dr_3b[CHDIM]);
  dist_3b[2] = get_dist(j,k,&dr_3b[2*CHDIM]);

  int typ_idxs_3b[3];
  typ_idxs_3b[0] = d_chimes_type[type[i]-1];
  typ_idxs_3b[1] = d_chimes_type[type[j]-1];
  typ_idxs_3b[2] = d_chimes_type[type[k]-1];

  KK_FLOAT energy = 0.0;

  KK_FLOAT force_3b[3*CHDIM];
  for (int idx = 0; idx < 3; idx++)
  {
    force_3b[idx] = 0.0;
    force_3b[CHDIM+idx] = 0.0;
    force_3b[2*CHDIM+idx] = 0.0;
  }

  KK_FLOAT stensor[6];
  for (int n = 0; n < 6; n++) stensor[n] = 0.0;

  const int slot_floats_3b = 2*3*MAX_3B_POLY;
  KK_FLOAT* scratch_3b = ((KK_FLOAT*) team.team_shmem().get_shmem(cpt * slot_floats_3b * sizeof(KK_FLOAT), 0)) + slot * slot_floats_3b;
  chimes_calculatorKK.compute_3B(team, scratch_3b, dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy);

  if (valid) {
    auto scatter_and_tally = [&] () {
      for (int idx = 0; idx < 3; idx++)
      {
        a_f(i,idx) += force_3b[idx];
        a_f(j,idx) += force_3b[CHDIM+idx];
        a_f(k,idx) += force_3b[2*CHDIM+idx];
      }

      int atmidxlst[6][2];

      if (EVFLAG && vflag_atom)
      {
        atmidxlst[0][0] = i;
        atmidxlst[0][1] = j;
        atmidxlst[1][0] = i;
        atmidxlst[1][1] = k;
        atmidxlst[2][0] = j;
        atmidxlst[2][1] = k;
      }

      if (EVFLAG)
        ev_tally_mb<NEIGHFLAG>(3, 3, atmidxlst, energy, stensor, ev);
    };

    // cpt==1 keeps the original PerTeam single; packed teams scatter once per
    // cluster (PerThread), since each team-thread owns a distinct cluster.
    if constexpr (cpt == 1)
      Kokkos::single(Kokkos::PerTeam(team), scatter_and_tally);
    else
      Kokkos::single(Kokkos::PerThread(team), scatter_and_tally);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>,const t_team& team) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>(), team, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>, const t_team& team, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 4-body interactions
  ////////////////////////////////////////

  // One team-thread handles one 4-body cluster. The dense Chebyshev coefficient
  // reduction inside compute_4B is parallelized over the team's
  // ThreadVectorRange; the global force scatter and energy/virial tally are
  // done once per cluster, guarded by Kokkos::single. See the 3-body operator
  // for the cluster-packing scheme (cpt/slot/clamp); cpt==1 is the default and
  // recovers the original one-cluster-per-team behavior.

  constexpr int cpt = EVFLAG ? 1 : CHIMES_CLUSTERS_PER_TEAM_4B;
  const int slot = (cpt == 1) ? 0 : team.team_rank();
  const int ii_raw = team.league_rank() * cpt + slot;
  const bool valid = ii_raw < size_4mers;
  const int ii = valid ? ii_raw : (size_4mers - 1);

  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const int i = d_neighborlist_4mers(ii,0);
  const int j = d_neighborlist_4mers(ii,1);
  const int k = d_neighborlist_4mers(ii,2);
  const int l = d_neighborlist_4mers(ii,3);

  KK_FLOAT dist_4b[6], dr_4b[6*CHDIM];
  dist_4b[0] = get_dist(i,j,&dr_4b[0*CHDIM]);
  dist_4b[1] = get_dist(i,k,&dr_4b[CHDIM]);
  dist_4b[2] = get_dist(i,l,&dr_4b[2*CHDIM]);
  dist_4b[3] = get_dist(j,k,&dr_4b[3*CHDIM]);
  dist_4b[4] = get_dist(j,l,&dr_4b[4*CHDIM]);
  dist_4b[5] = get_dist(k,l,&dr_4b[5*CHDIM]);

  int typ_idxs_4b[4];
  typ_idxs_4b[0] = d_chimes_type[type[i]-1];
  typ_idxs_4b[1] = d_chimes_type[type[j]-1];
  typ_idxs_4b[2] = d_chimes_type[type[k]-1];
  typ_idxs_4b[3] = d_chimes_type[type[l]-1];

  KK_FLOAT energy = 0.0;

  KK_FLOAT force_4b[4*CHDIM];
  for (int idx = 0; idx < 3; idx++) {
    force_4b[idx] = 0.0;
    force_4b[CHDIM+idx] = 0.0;
    force_4b[2*CHDIM+idx] = 0.0;
    force_4b[3*CHDIM+idx] = 0.0;
  }

  KK_FLOAT stensor[6];
  for (int n = 0; n < 6; n++) stensor[n] = 0.0;

  const int slot_floats_4b = 2*6*MAX_4B_POLY;
  KK_FLOAT* scratch_4b = ((KK_FLOAT*) team.team_shmem().get_shmem(cpt * slot_floats_4b * sizeof(KK_FLOAT), 0)) + slot * slot_floats_4b;
  chimes_calculatorKK.compute_4B(team, scratch_4b, dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy);

  if (valid) {
    auto scatter_and_tally = [&] () {
      for (int idx = 0; idx < 3; idx++) {
        a_f(i,idx) += force_4b[idx];
        a_f(j,idx) += force_4b[CHDIM+idx];
        a_f(k,idx) += force_4b[2*CHDIM+idx];
        a_f(l,idx) += force_4b[3*CHDIM+idx];
      }

      int atmidxlst[6][2];

      if (EVFLAG && vflag_atom) {
        atmidxlst[0][0] = i;
        atmidxlst[0][1] = j;
        atmidxlst[1][0] = i;
        atmidxlst[1][1] = k;
        atmidxlst[2][0] = i;
        atmidxlst[2][1] = l;
        atmidxlst[3][0] = j;
        atmidxlst[3][1] = k;
        atmidxlst[4][0] = j;
        atmidxlst[4][1] = l;
        atmidxlst[5][0] = k;
        atmidxlst[5][1] = l;
      }

      if (EVFLAG)
        ev_tally_mb<NEIGHFLAG>(4, 6, atmidxlst, energy, stensor, ev);
    };

    if constexpr (cpt == 1)
      Kokkos::single(Kokkos::PerTeam(team), scatter_and_tally);
    else
      Kokkos::single(Kokkos::PerThread(team), scatter_and_tally);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>,const t_team& team) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>(), team, ev);
}

/* ----------------------------------------------------------------------
   PROTOTYPE fused (list-free) 3B/4B evaluation. Per-cluster evaluators below
   merge the validity checks of the neighbor-list build (neigh_3B_item / the
   4-body build) with the force/energy evaluation (Compute3Body/Compute4Body),
   so no 3-/4-mer list is ever materialized. All vector lanes of a team evaluate
   the same cluster, so the early-return validity checks are uniform across the
   team and never diverge at compute_3B/compute_4B's team_barrier.
 ------------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::eval_fused_3body(const t_team& team, KK_FLOAT* scratch, bool reuse_lead, int i, int j, int k, EV_FLOAT& ev) const
{
  KK_FLOAT dr_3b[3*CHDIM], dist_3b[3];
  dist_3b[0] = get_dist(i,j,&dr_3b[0*CHDIM]);
  dist_3b[1] = get_dist(i,k,&dr_3b[1*CHDIM]);
  dist_3b[2] = get_dist(j,k,&dr_3b[2*CHDIM]);

  if (dist_3b[2] >= maxcut_3b) return;   // ij, ik are < maxcut_3b by construction

  const int natmtyps = chimes_calculatorKK.natmtyps;
  int typ[3];
  typ[0] = d_chimes_type[type[i]-1];
  typ[1] = d_chimes_type[type[j]-1];
  typ[2] = d_chimes_type[type[k]-1];

  const int type_idx = typ[0]*natmtyps*natmtyps + typ[1]*natmtyps + typ[2];
  const int tripidx = chimes_calculatorKK.c_atom_int_trip_map[type_idx];
  if (tripidx < 0) return;

  const auto &c_cut = chimes_calculatorKK.c_chimes_3b_cutoff;
  const auto &c_map = chimes_calculatorKK.c_pair_int_trip_map;
  if (dist_3b[0] >= c_cut(tripidx,c_map(type_idx,0),1)) return;
  if (dist_3b[1] >= c_cut(tripidx,c_map(type_idx,1),1)) return;
  if (dist_3b[2] >= c_cut(tripidx,c_map(type_idx,2),1)) return;

  KK_FLOAT energy = 0.0;
  KK_FLOAT force_3b[3*CHDIM];
  for (int idx = 0; idx < 3; idx++) {
    force_3b[idx] = 0.0; force_3b[CHDIM+idx] = 0.0; force_3b[2*CHDIM+idx] = 0.0;
  }
  KK_FLOAT stensor[6];
  for (int n = 0; n < 6; n++) stensor[n] = 0.0;

  chimes_calculatorKK.compute_3B(team, scratch, dist_3b, dr_3b, typ, force_3b, stensor, energy, reuse_lead);

  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  Kokkos::single(Kokkos::PerTeam(team), [&] () {
    for (int idx = 0; idx < 3; idx++) {
      a_f(i,idx) += force_3b[idx];
      a_f(j,idx) += force_3b[CHDIM+idx];
      a_f(k,idx) += force_3b[2*CHDIM+idx];
    }
    int atmidxlst[6][2];
    if (EVFLAG && vflag_atom) {
      atmidxlst[0][0] = i; atmidxlst[0][1] = j;
      atmidxlst[1][0] = i; atmidxlst[1][1] = k;
      atmidxlst[2][0] = j; atmidxlst[2][1] = k;
    }
    if (EVFLAG) ev_tally_mb<NEIGHFLAG>(3, 3, atmidxlst, energy, stensor, ev);
  });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::eval_fused_4body(const t_team& team, KK_FLOAT* scratch, int i, int j, int k, int l, EV_FLOAT& ev) const
{
  KK_FLOAT dr_4b[6*CHDIM], dist_4b[6];
  dist_4b[0] = get_dist(i,j,&dr_4b[0*CHDIM]);
  dist_4b[1] = get_dist(i,k,&dr_4b[1*CHDIM]);
  dist_4b[2] = get_dist(i,l,&dr_4b[2*CHDIM]);
  dist_4b[3] = get_dist(j,k,&dr_4b[3*CHDIM]);
  dist_4b[4] = get_dist(j,l,&dr_4b[4*CHDIM]);
  dist_4b[5] = get_dist(k,l,&dr_4b[5*CHDIM]);

  for (int p = 0; p < 6; p++) if (dist_4b[p] >= maxcut_4b) return;

  const int natmtyps = chimes_calculatorKK.natmtyps;
  int typ[4];
  typ[0] = d_chimes_type[type[i]-1];
  typ[1] = d_chimes_type[type[j]-1];
  typ[2] = d_chimes_type[type[k]-1];
  typ[3] = d_chimes_type[type[l]-1];

  const int idx = typ[0]*natmtyps*natmtyps*natmtyps + typ[1]*natmtyps*natmtyps + typ[2]*natmtyps + typ[3];
  const int quadidx = chimes_calculatorKK.c_atom_int_quad_map[idx];
  if (quadidx < 0) return;

  const auto &c_cut = chimes_calculatorKK.c_chimes_4b_cutoff;
  const auto &c_map = chimes_calculatorKK.c_pair_int_quad_map;
  for (int p = 0; p < 6; p++) if (dist_4b[p] >= c_cut(quadidx,c_map(idx,p),1)) return;

  KK_FLOAT energy = 0.0;
  KK_FLOAT force_4b[4*CHDIM];
  for (int idx2 = 0; idx2 < 3; idx2++) {
    force_4b[idx2] = 0.0; force_4b[CHDIM+idx2] = 0.0;
    force_4b[2*CHDIM+idx2] = 0.0; force_4b[3*CHDIM+idx2] = 0.0;
  }
  KK_FLOAT stensor[6];
  for (int n = 0; n < 6; n++) stensor[n] = 0.0;

  chimes_calculatorKK.compute_4B(team, scratch, dist_4b, dr_4b, typ, force_4b, stensor, energy);

  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  Kokkos::single(Kokkos::PerTeam(team), [&] () {
    for (int idx2 = 0; idx2 < 3; idx2++) {
      a_f(i,idx2) += force_4b[idx2];
      a_f(j,idx2) += force_4b[CHDIM+idx2];
      a_f(k,idx2) += force_4b[2*CHDIM+idx2];
      a_f(l,idx2) += force_4b[3*CHDIM+idx2];
    }
    int atmidxlst[6][2];
    if (EVFLAG && vflag_atom) {
      atmidxlst[0][0] = i; atmidxlst[0][1] = j;
      atmidxlst[1][0] = i; atmidxlst[1][1] = k;
      atmidxlst[2][0] = i; atmidxlst[2][1] = l;
      atmidxlst[3][0] = j; atmidxlst[3][1] = k;
      atmidxlst[4][0] = j; atmidxlst[4][1] = l;
      atmidxlst[5][0] = k; atmidxlst[5][1] = l;
    }
    if (EVFLAG) ev_tally_mb<NEIGHFLAG>(4, 6, atmidxlst, energy, stensor, ev);
  });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESFused3Body<NEIGHFLAG,EVFLAG>, const t_team& team, EV_FLOAT& ev) const
{
  // Enumerate 3-body clusters on the fly with i = min-tag lead (itag<jtag<ktag),
  // exactly the ordering the materialized build used, from the short neighbor
  // list. compute_3B's per-cluster vector reduction runs once per triplet.
  // One scratch slice is allocated per team and reused for every cluster the
  // team evaluates (compute_3B no longer self-allocates).
  KK_FLOAT* scratch = (KK_FLOAT*) team.team_shmem().get_shmem(2*3*MAX_3B_POLY * sizeof(KK_FLOAT), 0);
  if constexpr (CHIMES_FUSED_GRANULARITY == 0) {
    const int i = d_ilist[team.league_rank() + chunk_offset];
    const tagint itag = tag[i];
    const int n = d_numneigh_short[i];
    for (int jj = 0; jj < n; jj++) {
      const int j = d_neighbors_short(i,jj);
      const tagint jtag = tag[j];
      if (jtag <= itag) continue;
      for (int kk = 0; kk < n; kk++) {
        const int k = d_neighbors_short(i,kk);
        if (tag[k] <= jtag) continue;
        eval_fused_3body<NEIGHFLAG,EVFLAG>(team, scratch, false, i, j, k, ev);
      }
    }
  } else {
    const int m = team.league_rank();
    const int i = d_neighborlist_2mers(m,0);
    const int j = d_neighborlist_2mers(m,1);
    if (get_dist(i,j) >= maxcut_3b) return;
    const tagint jtag = tag[j];
    const int n = d_numneigh_short[i];
    if constexpr (CHIMES_FUSED_LEAD_REUSE) {
      // The short list is type-sorted, so same-typ_k candidates are contiguous.
      // Within a typ_k bucket the triplet type (and thus the lead pair's cutoffs)
      // is fixed, so Tn(r_ij)/Tnd(r_ij) are reusable: fill them once per bucket
      // with set_cheby_lead_3b, then evaluate each cluster with reuse_lead=true
      // (compute_3B skips the lead pair's set_cheby_polys).
      const int nt = chimes_calculatorKK.natmtyps;
      const int typ_i = d_chimes_type[type[i]-1];
      const int typ_j = d_chimes_type[type[j]-1];
      const KK_FLOAT dist_ij = get_dist(i,j);
      int prev_typ_k = -1;
      int cur_tripidx = -1;
      for (int kk = 0; kk < n; kk++) {
        const int k = d_neighbors_short(i,kk);
        if (tag[k] <= jtag) continue;
        const int typ_k = d_chimes_type[type[k]-1];
        if (typ_k != prev_typ_k) {                    // new bucket (uniform across the team)
          prev_typ_k = typ_k;
          const int type_idx = typ_i*nt*nt + typ_j*nt + typ_k;
          cur_tripidx = chimes_calculatorKK.c_atom_int_trip_map[type_idx];
          if (cur_tripidx >= 0)
            chimes_calculatorKK.set_cheby_lead_3b(team, scratch, dist_ij, typ_i, typ_j, type_idx, cur_tripidx);
        }
        if (cur_tripidx < 0) continue;
        eval_fused_3body<NEIGHFLAG,EVFLAG>(team, scratch, true, i, j, k, ev);
      }
    } else {
      for (int kk = 0; kk < n; kk++) {
        const int k = d_neighbors_short(i,kk);
        if (tag[k] <= jtag) continue;
        eval_fused_3body<NEIGHFLAG,EVFLAG>(team, scratch, false, i, j, k, ev);
      }
    }
  }
}

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESFused3Body<NEIGHFLAG,EVFLAG>, const t_team& team) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESFused3Body<NEIGHFLAG,EVFLAG>(), team, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESFused4Body<NEIGHFLAG,EVFLAG>, const t_team& team, EV_FLOAT& ev) const
{
  // Enumerate 4-body clusters on the fly with i = min-tag lead (itag<jtag<ktag<ltag).
  // One team scratch slice, reused for every cluster the team evaluates.
  KK_FLOAT* scratch = (KK_FLOAT*) team.team_shmem().get_shmem(2*6*MAX_4B_POLY * sizeof(KK_FLOAT), 0);
  if constexpr (CHIMES_FUSED_GRANULARITY == 0) {
    const int i = d_ilist[team.league_rank() + chunk_offset];
    const tagint itag = tag[i];
    const int n = d_numneigh_short[i];
    for (int jj = 0; jj < n; jj++) {
      const int j = d_neighbors_short(i,jj);
      const tagint jtag = tag[j];
      if (jtag <= itag) continue;
      for (int kk = 0; kk < n; kk++) {
        const int k = d_neighbors_short(i,kk);
        const tagint ktag = tag[k];
        if (ktag <= jtag) continue;
        for (int ll = 0; ll < n; ll++) {
          const int l = d_neighbors_short(i,ll);
          if (tag[l] <= ktag) continue;
          eval_fused_4body<NEIGHFLAG,EVFLAG>(team, scratch, i, j, k, l, ev);
        }
      }
    }
  } else {
    const int m = team.league_rank();
    const int i = d_neighborlist_2mers(m,0);
    const int j = d_neighborlist_2mers(m,1);
    if (get_dist(i,j) >= maxcut_4b) return;
    const tagint jtag = tag[j];
    const int n = d_numneigh_short[i];
    for (int kk = 0; kk < n; kk++) {
      const int k = d_neighbors_short(i,kk);
      const tagint ktag = tag[k];
      if (ktag <= jtag) continue;
      for (int ll = 0; ll < n; ll++) {
        const int l = d_neighbors_short(i,ll);
        if (tag[l] <= ktag) continue;
        eval_fused_4body<NEIGHFLAG,EVFLAG>(team, scratch, i, j, k, l, ev);
      }
    }
  }
}

template<class DeviceType, int vector_length>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::operator() (TagPairCHIMESFused4Body<NEIGHFLAG,EVFLAG>, const t_team& team) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESFused4Body<NEIGHFLAG,EVFLAG>(), team, ev);
}

/* ----------------------------------------------------------------------
   general ev tally function for many-body models where per-atom assignments
   do not make sense. Expects newton_pair = 1.
 ------------------------------------------------------------------------- */

template<class DeviceType, int vector_length>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType, vector_length>::ev_tally_mb(int ninteractionatoms, int npairs,
                                               int atmpairidxlst[6][2],
                                               KK_FLOAT evdwl, KK_FLOAT stress[6],
                                               EV_FLOAT &ev) const
{
  // Assumes newton pair is always true
  // Assumes a full neighbor list is always true (hard coded in pair_chimes.cpp)
  // Modeled after ev_tally_full and ev_tally3 (to get MB handling)
  // force and distance vector are flattened 2d vectors, e.g., atom_idx*3 + [0,1,2 == x,y,z dims]

  // The vatom array is duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_eatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_eatom),decltype(ndup_eatom)>::get(dup_eatom,ndup_eatom);
  auto a_eatom = v_eatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  int atmlist[4];

  atmlist[0] = atmpairidxlst[0][0];       // i

  if (ninteractionatoms > 1) // 2, 3, and/or 4b
    atmlist[1] = atmpairidxlst[0][1];   // j

  if (ninteractionatoms > 2) // 3 and/or 4b
    atmlist[2] = atmpairidxlst[1][1];   // k

  if (ninteractionatoms > 3) // 4b only
    atmlist[3] = atmpairidxlst[2][1];   // l

  if (eflag_global)
    ev.evdwl += evdwl;

  if (eflag_atom)
    for (int atm = 0; atm < ninteractionatoms; atm++)
      a_eatom[atmlist[atm]] += evdwl/ninteractionatoms;

  if (ninteractionatoms < 2)
    return;

  if (!vflag_either)
    return;

  // FYI, stress calculations follow strategy described here: https://docs.lammps.org/compute_stress_atom.html

  if (vflag_global) {
    ev.v[0] += stress[0];
    ev.v[1] += stress[3];
    ev.v[2] += stress[5];
    ev.v[3] += stress[1];
    ev.v[4] += stress[2];
    ev.v[5] += stress[4];
  }

  if (vflag_atom) {
    for (int a = 0; a < ninteractionatoms; a++) {
      a_vatom(atmlist[a],0) += stress[0]/ninteractionatoms;
      a_vatom(atmlist[a],1) += stress[3]/ninteractionatoms;
      a_vatom(atmlist[a],2) += stress[5]/ninteractionatoms;
      a_vatom(atmlist[a],3) += stress[1]/ninteractionatoms;
      a_vatom(atmlist[a],4) += stress[2]/ninteractionatoms;
      a_vatom(atmlist[a],5) += stress[4]/ninteractionatoms;
    }
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairCHIMESKokkosDevice<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairCHIMESKokkosHost<LMPHostType>;
#endif
}
