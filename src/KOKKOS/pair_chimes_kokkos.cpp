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

template<class DeviceType>
PairCHIMESKokkos<DeviceType>::PairCHIMESKokkos(LAMMPS *lmp) : PairCHIMES(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
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
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairCHIMESKokkos<DeviceType>::~PairCHIMESKokkos()
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

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::allocate()
{
  PairCHIMES::allocate();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::init_style()
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

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairCHIMES::coeff(narg,arg);

  // The device kernels hold the Chebyshev values in fixed-size registers, so
  // they cap the polynomial order.  The host path sizes its scratch from the
  // parameter file and has no such limit, so models that the device cannot
  // run are still usable there.

  if constexpr (!host_flag) {
    if (chimes_calculatorKK.poly_orders[0]+1 > MAX_2B_POLY)
      lmp->error->all(FLERR,"Exceeded maximum poly order for 2-body interactions, "
                      "increase value of MAX_2B_POLY in src/KOKKOS/chimesFF_kokkos.h "
                      "and recompile");

    if (chimes_calculatorKK.poly_orders[1]+1 > MAX_3B_POLY)
      lmp->error->all(FLERR,"Exceeded maximum poly order for 3-body interactions, "
                      "increase value of MAX_3B_POLY in src/KOKKOS/chimesFF_kokkos.h "
                      "and recompile");

    if (chimes_calculatorKK.poly_orders[2]+1 > MAX_4B_POLY)
      lmp->error->all(FLERR,"Exceeded maximum poly order for 4-body interactions, "
                      "increase value of MAX_4B_POLY in src/KOKKOS/chimesFF_kokkos.h "
                      "and recompile");
  }

  // chimes_type

  int size = chimes_type.size();
  LAMMPS_NS::MemKK::realloc_kokkos(d_chimes_type,"chimes:chimes_type",size);

  auto h_chimes_type = Kokkos::create_mirror_view(d_chimes_type);

  for (int i = 0; i < size; i++)
    h_chimes_type[i] = chimes_type[i];

  Kokkos::deep_copy(d_chimes_type,h_chimes_type);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairCHIMESKokkos<DeviceType>::get_dist(int i, int j, KK_FLOAT *dr) const
{
  dr[0] = x(j,0) - x(i,0);
  dr[1] = x(j,1) - x(i,1);
  dr[2] = x(j,2) - x(i,2);

  return sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairCHIMESKokkos<DeviceType>::get_dist(int i, int j) const
{
  KK_FLOAT dummy_dr[3];

  return get_dist(i,j, dummy_dr);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::build_mb_neighlists()
{
  if constexpr (host_flag) {
    host_build_mb_neighlists();
    return;
  }

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

    PairCHIMESComputeNeigh2BodyFunctor<DeviceType> neigh_2B_functor(this);
    Kokkos::parallel_scan("ComputeNeigh2Body", inum, neigh_2B_functor,size_2mers);

    resize = size_2mers > max_2mers;
    if (resize) {
      max_2mers = MAX(max_2mers+MAX(1,max_2mers*0.1),size_2mers);
      LAMMPS_NS::MemKK::realloc_kokkos(d_neighborlist_2mers,"chimes:neighborlist_2mers",max_2mers);
    }
  }

  // try building 3-body list, resize if necessary

  if (chimes_calculatorKK.poly_orders[1] > 0 || chimes_calculatorKK.poly_orders[2] > 0) {
    resize = 1;
    while (resize) {
      resize = 0;

      PairCHIMESComputeNeigh3BodyFunctor<DeviceType> neigh_3B_functor(this);
      Kokkos::parallel_scan("ComputeNeigh3Body", size_2mers, neigh_3B_functor, size_3mers);

      resize = size_3mers > max_3mers;
      if (resize) {
        max_3mers = MAX(max_3mers+MAX(1,max_3mers*0.1),size_3mers);
        LAMMPS_NS::MemKK::realloc_kokkos(d_neighborlist_3mers,"chimes:neighborlist_3mers",max_3mers);
      }
    }
  }

  // try building 4-body list, resize if necessary

  if (chimes_calculatorKK.poly_orders[2] > 0) {
    resize = 1;
    while (resize) {
      resize = 0;

      Kokkos::deep_copy(d_size_4mers,0.0);

      typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESComputeNeigh4Body> policy_neigh(0,size_3mers);
      Kokkos::parallel_for("ComputeNeigh4Body",policy_neigh,*this);
      //PairCHIMESComputeNeigh4BodyFunctor<DeviceType> neigh_4B_functor(this);
      //Kokkos::parallel_scan("ComputeNeigh4Body", size_3mers, neigh_4B_functor), size_4mers;

      auto h_size_4mers = Kokkos::create_mirror_view_and_copy(LMPHostType(),d_size_4mers);

      size_4mers = h_size_4mers();
      resize = size_4mers > max_4mers;
      if (resize) {
        max_4mers = MAX(max_4mers+MAX(1,max_4mers*0.1),size_4mers);
        LAMMPS_NS::MemKK::realloc_kokkos(d_neighborlist_4mers,"chimes:neighborlist_4mers",max_4mers);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::neigh_2B_item(const int& ii, int &offset, const bool &final) const
{
  const int i = d_ilist[ii];
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

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::neigh_3B_item(const int& ii, int &offset, const bool &final) const
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

  // ChIMES assumes all atoms must be within cutoff of each other for a valid interaction
  const int knum = d_numneigh[i];

  for (int kk = 0; kk < knum; kk++) {
    int k = d_neighbors(i,kk);
    k &= NEIGHMASK;
    const tagint ktag = tag[k];

    if ((k == i) || (k == j)) continue;

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

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESComputeNeigh4Body, const int& ii) const
//void PairCHIMESKokkos<DeviceType>::neigh_4B_item(const int& ii, int &offset, const bool &final) const
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

  const int lnum = d_numneigh[i];

  for (int ll = 0; ll < lnum; ll++)
  {
    int l = d_neighbors(i,ll);
    const tagint ltag = tag[l];
    l &= NEIGHMASK;

    if ((l == i) || (l == j) || (l == k)) continue;

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

/* ----------------------------------------------------------------------
   Host execution path.

   On the host the batched chimesFF evaluators are available: clusters of one
   type are handed to the polynomial a lane group at a time, the Morse
   transform runs through a vectorized exponential, and the coefficient tree
   is walked once per group rather than once per cluster.  That machinery is
   plain C++ operating on flat double arrays, so the /kk variant runs it
   directly instead of the one-cluster-at-a-time device kernels below, and
   threads it by handing each work item a contiguous chunk of the type-sorted
   cluster list.
------------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::setup_neighlist_ptrs()
{
  if (!host_flag) {
    PairCHIMES::setup_neighlist_ptrs();
    return;
  }

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType>*>(list);

  nl_inum = list->inum;
  nl_ilist = k_list->d_ilist.data();
  nl_numneigh = k_list->d_numneigh.data();

  const int nrows = k_list->d_neighbors.extent(0);
  const int ncols = k_list->d_neighbors.extent(1);

  host_firstneigh.resize(nrows);

  // A neighbor row is contiguous when the inner stride is one, which is the
  // layout every host-only build produces; then the rows are handed out as
  // they lie.  A device-ordered list reaching the host variant is strided
  // instead, so those rows are compacted once per rebuild.

  const bool contiguous = (ncols < 2) ||
      ((&k_list->d_neighbors(0,1) - &k_list->d_neighbors(0,0)) == 1);

  if (contiguous) {
    for (int i = 0; i < nrows; i++) host_firstneigh[i] = &k_list->d_neighbors(i,0);
  } else {
    host_neigh_buf.resize((size_t) nrows * ncols);

    for (int i = 0; i < nrows; i++) {
      int *const row = &host_neigh_buf[(size_t) i * ncols];
      const int n = k_list->d_numneigh(i);

      for (int jj = 0; jj < n; jj++) row[jj] = k_list->d_neighbors(i,jj);

      host_firstneigh[i] = row;
    }
  }

  nl_firstneigh = host_firstneigh.data();
}

/* ----------------------------------------------------------------------
   Cluster enumeration for one block of owned atoms.  Each block writes to its
   own output buffers, so the blocks are independent; they are concatenated in
   block order afterwards, which reproduces the order a single pass would have
   produced.
------------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::host_build_chunk(const int chunk) const
{
  const int sz = (nl_inum + host_nchunk_build - 1) / host_nchunk_build;
  const int lo = chunk * sz;
  const int hi = MIN(lo + sz, nl_inum);

  auto &out3 = const_cast<std::vector<int> &>(host_out3[chunk]);
  auto &out4 = const_cast<std::vector<int> &>(host_out4[chunk]);
  auto &scr = const_cast<MBScratch &>(host_mb_scratch[chunk]);

  out3.resize(0);
  out4.resize(0);

  for (int ii = lo; ii < hi; ii++)
    mb_clusters_for_atom(nl_ilist[ii], mb_ctx, scr, out3, out4);
}

/* ----------------------------------------------------------------------
   Counting sort of a cluster list by packed atom-type index, threaded.  Each
   block histograms its own clusters, the per-block starting offsets are formed
   by one pass over the (key, block) table, and the blocks then scatter in
   parallel.  Walking the table key-major and block-minor is what makes the
   result identical to the serial stable sort: within a key the clusters keep
   their build order.
------------------------------------------------------------------------- */

template<class DeviceType>
template<int WIDTH>
void PairCHIMESKokkos<DeviceType>::host_sort_count(const int chunk) const
{
  const int sz = (sort_nmers + host_nchunk_sort - 1) / host_nchunk_sort;
  const int lo = chunk * sz;
  const int hi = MIN(lo + sz, sort_nmers);

  const int nt = chimes_calculator->natmtyps;
  const int *atype = atom->type;
  const int *mers = sort_mers->data();

  int *const hist = const_cast<int *>(&host_sort_hist[(size_t) chunk * sort_nkey]);
  int *const key_of = const_cast<int *>(host_sort_key.data());

  for (int k = 0; k < sort_nkey; k++) hist[k] = 0;

  for (int c = lo; c < hi; c++) {
    const int *m = &mers[(size_t) WIDTH * c];
    int key = 0;

    for (int w = 0; w < WIDTH; w++) key = key * nt + chimes_type[atype[m[w]] - 1];

    key_of[c] = key;
    hist[key]++;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int WIDTH>
void PairCHIMESKokkos<DeviceType>::host_sort_scatter(const int chunk) const
{
  const int sz = (sort_nmers + host_nchunk_sort - 1) / host_nchunk_sort;
  const int lo = chunk * sz;
  const int hi = MIN(lo + sz, sort_nmers);

  const int *mers = sort_mers->data();
  int *const dst = const_cast<int *>(host_sort_scratch.data());
  int *const type_out = const_cast<int *>(sort_type->data());
  const int *key_of = host_sort_key.data();

  int *const off = const_cast<int *>(&host_sort_hist[(size_t) chunk * sort_nkey]);

  for (int c = lo; c < hi; c++) {
    const int key = key_of[c];
    const int d = off[key]++;

    type_out[d] = key;

    for (int w = 0; w < WIDTH; w++) dst[(size_t) WIDTH * d + w] = mers[(size_t) WIDTH * c + w];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int WIDTH>
void PairCHIMESKokkos<DeviceType>::host_sort_mers(std::vector<int> &mers, int nmers,
                                                  std::vector<int> &mer_type)
{
  if (nmers == 0) {
    mer_type.resize(0);
    return;
  }

  const int nt = chimes_calculator->natmtyps;

  sort_nkey = 1;

  for (int w = 0; w < WIDTH; w++) sort_nkey *= nt;

  const int nthreads = lmp->kokkos->nthreads;

  host_nchunk_sort = MAX(MIN(nthreads * 8, (nmers + 4095) / 4096), 1);

  // The offset table is one int per (block, key).  A model with many atom types
  // makes it the larger of the two allocations, so past a few megabytes the
  // block count is cut back rather than the table grown.

  while ((host_nchunk_sort > 1) && ((double) host_nchunk_sort * sort_nkey > 4.0e6))
    host_nchunk_sort /= 2;

  sort_mers = &mers;
  sort_type = &mer_type;
  sort_nmers = nmers;

  host_sort_hist.resize((size_t) host_nchunk_sort * sort_nkey);
  host_sort_key.resize(nmers);
  host_sort_scratch.resize((size_t) nmers * WIDTH);
  mer_type.resize(nmers);

  using policy_t = Kokkos::RangePolicy<DeviceType, Kokkos::Schedule<Kokkos::Dynamic>>;

  PairCHIMESHostBuildFunctor<DeviceType,1,WIDTH> fcount(this);
  Kokkos::parallel_for("CHIMESHostSortCount", policy_t(0,host_nchunk_sort), fcount);

  // Key-major, block-minor running total: block b's clusters of key k land
  // after every cluster of key k from the blocks before it.

  int running = 0;

  for (int k = 0; k < sort_nkey; k++)
    for (int b = 0; b < host_nchunk_sort; b++) {
      const int n = host_sort_hist[(size_t) b * sort_nkey + k];

      host_sort_hist[(size_t) b * sort_nkey + k] = running;
      running += n;
    }

  PairCHIMESHostBuildFunctor<DeviceType,2,WIDTH> fscat(this);
  Kokkos::parallel_for("CHIMESHostSortScatter", policy_t(0,host_nchunk_sort), fscat);

  mers.swap(host_sort_scratch);
}

/* ----------------------------------------------------------------------
   Threaded replacement for PairCHIMES::build_mb_neighlists on the host.
------------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::host_build_mb_neighlists()
{
  if ((chimes_calculator->poly_orders[1] == 0) && (chimes_calculator->poly_orders[2] == 0)) return;

  setup_neighlist_ptrs();

  mb_ctx = mb_context();

  const int nthreads = lmp->kokkos->nthreads;

  host_nchunk_build = MAX(MIN(nthreads * 16, (nl_inum + 7) / 8), 1);

  if ((int) host_out3.size() < host_nchunk_build) {
    host_out3.resize(host_nchunk_build);
    host_out4.resize(host_nchunk_build);
    host_mb_scratch.resize(host_nchunk_build);
  }

  using policy_t = Kokkos::RangePolicy<DeviceType, Kokkos::Schedule<Kokkos::Dynamic>>;

  PairCHIMESHostBuildFunctor<DeviceType,0,0> fbuild(this);
  Kokkos::parallel_for("CHIMESHostBuild", policy_t(0,host_nchunk_build), fbuild);

  size_t n3 = 0, n4 = 0;

  for (int c = 0; c < host_nchunk_build; c++) {
    n3 += host_out3[c].size();
    n4 += host_out4[c].size();
  }

  neighborlist_3mers.resize(n3);
  neighborlist_4mers.resize(n4);

  size_t o3 = 0, o4 = 0;

  for (int c = 0; c < host_nchunk_build; c++) {
    if (!host_out3[c].empty())
      memcpy(&neighborlist_3mers[o3], host_out3[c].data(), host_out3[c].size() * sizeof(int));

    if (!host_out4[c].empty())
      memcpy(&neighborlist_4mers[o4], host_out4[c].data(), host_out4[c].size() * sizeof(int));

    o3 += host_out3[c].size();
    o4 += host_out4[c].size();
  }

  n_3mers = n3 / 3;
  n_4mers = n4 / 4;

  if (mb_ctx.do_3b) host_sort_mers<3>(neighborlist_3mers, n_3mers, mer_type_3b);

  if (mb_ctx.do_4b) host_sort_mers<4>(neighborlist_4mers, n_4mers, mer_type_4b);
}

/* ----------------------------------------------------------------------
   Split the cluster lists into work items.  Clusters cost the same to
   evaluate, so equal-sized chunks balance; several per thread absorb the
   fraction of clusters that fall outside the cutoffs and are skipped.  The
   floor keeps a chunk long enough to fill lane groups -- a chunk shorter than
   a lane group would evaluate partly empty batches.
------------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::host_setup_chunks()
{
  const int nthreads = lmp->kokkos->nthreads;
  const int per_thread = 32;
  const int min_chunk = 64;

  // Several chunks per thread rather than one: the list is grouped by cluster
  // type, and the cost of a cluster is set by how many coefficients its type
  // carries, which varies by more than an order of magnitude across types.  An
  // equal split by cluster count is therefore not an equal split of work, so
  // the chunks are made small enough for dynamic scheduling to even out.

  auto nchunk = [&](const int nitem) {
    if (nitem <= 0) return 0;

    int n = nthreads * per_thread;

    if (n > (nitem + min_chunk - 1) / min_chunk) n = (nitem + min_chunk - 1) / min_chunk;

    return MAX(n,1);
  };

  host_nchunk_2b = nchunk(nl_inum);
  host_nchunk_3b = nchunk(n_3mers);
  host_nchunk_4b = nchunk(n_4mers);

  // Per-chunk scratch.  The batch objects hold the Chebyshev arrays for one
  // lane group, sized from the polynomial order, so they are built once and
  // reused for every batch a chunk evaluates.

  const int order3 = chimes_calculator->poly_orders[1];
  const int order4 = chimes_calculator->poly_orders[2];

  while ((int) host_batch3.size() < host_nchunk_3b) host_batch3.emplace_back(order3);
  while ((int) host_batch4.size() < host_nchunk_4b) host_batch4.emplace_back(order4);

  const int nchem = chimes_calculator->natmtyps;
  const size_t nkey = (size_t) nchem * nchem;

  // Each chunk's staging is padded out to whole cache lines.  With one
  // chemical species nkey is 1, and unpadded the per-chunk lane counters are
  // four bytes apart: every thread then writes the same line once per pair,
  // and the two-body kernel ran slower on four threads than on one.

  host_b2_cnt_stride = ((nkey + 15) / 16) * 16;

  const size_t data_stride = ((nkey * CHIMES_VLEN + 15) / 16) * 16;

  host_b2_stride = data_stride;

  host_b2_cnt.assign((size_t) host_nchunk_2b * host_b2_cnt_stride, 0);
  host_b2_i.resize((size_t) host_nchunk_2b * data_stride);
  host_b2_j.resize((size_t) host_nchunk_2b * data_stride);
  host_b2_dist.resize((size_t) host_nchunk_2b * data_stride);
  host_b2_dr.resize((size_t) host_nchunk_2b * data_stride * CHDIM);
}

/* ----------------------------------------------------------------------
   Energy and virial tally for one cluster.  Mirrors PairCHIMES::ev_tally_mb,
   but accumulates the globals into the reduction value and the per-atom
   arrays through the scatter views, so it is safe to call from several
   threads at once.
------------------------------------------------------------------------- */

template<class DeviceType>
template<class EAtomAccess, class VAtomAccess>
void PairCHIMESKokkos<DeviceType>::host_tally(int ninteractionatoms, const int *atmlist,
                                              double evdwl, const double *stress,
                                              EV_FLOAT &ev, const EAtomAccess &a_eatom,
                                              const VAtomAccess &a_vatom) const
{
  if (eflag_global) ev.evdwl += evdwl;

  if (eflag_atom) {
    const double share = evdwl/ninteractionatoms;

    for (int a = 0; a < ninteractionatoms; a++) a_eatom[atmlist[a]] += share;
  }

  if (ninteractionatoms < 2) return;

  if (!vflag_either) return;

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
      const int ga = atmlist[a];

      a_vatom(ga,0) += stress[0]/ninteractionatoms;
      a_vatom(ga,1) += stress[3]/ninteractionatoms;
      a_vatom(ga,2) += stress[5]/ninteractionatoms;
      a_vatom(ga,3) += stress[1]/ninteractionatoms;
      a_vatom(ga,4) += stress[2]/ninteractionatoms;
      a_vatom(ga,5) += stress[4]/ninteractionatoms;
    }
  }
}

/* ----------------------------------------------------------------------
   1- and 2-body interactions for one block of owned atoms.  Pairs that sit in
   the plain middle of the potential are staged per chemical-pair key and
   evaluated CHIMES_VLEN at a time; everything else takes the scalar path.
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
void PairCHIMESKokkos<DeviceType>::host_2body_chunk(const int chunk, EV_FLOAT &ev) const
{
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  // Built once per chunk, not once per cluster: a duplicated scatter view's
  // accessor takes a unique token, and acquiring one is an atomic on a pool
  // shared by every thread.  Per cluster that atomic dominated the tally.

  auto v_eatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_eatom),decltype(ndup_eatom)>::get(dup_eatom,ndup_eatom);
  auto a_eatom = v_eatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  double **xx = atom->x;
  int *atype = atom->type;
  tagint *atag = atom->tag;

  const int nchem = chimes_calculator->natmtyps;
  const int nkey = nchem * nchem;

  const int sz = (nl_inum + host_nchunk_2b - 1) / host_nchunk_2b;
  const int lo = chunk * sz;
  const int hi = MIN(lo + sz, nl_inum);

  // Staging for this chunk only, so no two threads share a key's lanes.

  int *const cnt = const_cast<int *>(&host_b2_cnt[(size_t) chunk * host_b2_cnt_stride]);
  int *const b_i = const_cast<int *>(&host_b2_i[(size_t) chunk * host_b2_stride]);
  int *const b_j = const_cast<int *>(&host_b2_j[(size_t) chunk * host_b2_stride]);
  double *const b_d = const_cast<double *>(&host_b2_dist[(size_t) chunk * host_b2_stride]);
  double *const b_dr = const_cast<double *>(&host_b2_dr[(size_t) chunk * host_b2_stride * CHDIM]);

  chimes2BTmp tmp(chimes_calculator->poly_orders[0]);

  std::vector<double> lforce(2*CHDIM), lstress(6), ldr(CHDIM);
  std::vector<int> ltyp(2);

  auto flush = [&](const int key) {
    const int nb = cnt[key];
    double bd[CHIMES_VLEN], be[CHIMES_VLEN], bfs[CHIMES_VLEN];

    for (int l = 0; l < nb; l++) bd[l] = b_d[key*CHIMES_VLEN + l];

    for (int l = nb; l < CHIMES_VLEN; l++) bd[l] = bd[0];

    chimes_calculator->compute_2B_batch(key, bd, be, bfs);

    for (int l = 0; l < nb; l++) {
      const int ai = b_i[key*CHIMES_VLEN + l], aj = b_j[key*CHIMES_VLEN + l];
      const double *const pdr = &b_dr[(key*CHIMES_VLEN + l)*CHDIM];

      for (int idx = 0; idx < CHDIM; idx++) {
        const double fc = bfs[l] * pdr[idx];

        a_f(ai,idx) += fc;
        a_f(aj,idx) -= fc;
      }

      if (evflag) {
        const int alist[2] = {ai, aj};
        double st[6];

        st[0] = -bfs[l] * pdr[0] * pdr[0];
        st[1] = -bfs[l] * pdr[0] * pdr[1];
        st[2] = -bfs[l] * pdr[0] * pdr[2];
        st[3] = -bfs[l] * pdr[1] * pdr[1];
        st[4] = -bfs[l] * pdr[1] * pdr[2];
        st[5] = -bfs[l] * pdr[2] * pdr[2];

        host_tally(2, alist, be[l], st, ev, a_eatom, a_vatom);
      }
    }

    cnt[key] = 0;
  };

  for (int ii = lo; ii < hi; ii++) {
    const int i = nl_ilist[ii];
    const tagint itag = atag[i];
    const int *const jlist = nl_firstneigh[i];
    const int jnum = nl_numneigh[i];

    const int ichem = chimes_type[atype[i] - 1];

    double energy = 0.0;

    chimes_calculator->compute_1B(atype[i] - 1, energy);

    if (evflag) {
      const int alist[1] = {i};

      host_tally(1, alist, energy, nullptr, ev, a_eatom, a_vatom);
    }

    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;

      if (atag[j] <= itag) continue;

      const double dxv = xx[j][0] - xx[i][0];
      const double dyv = xx[j][1] - xx[i][1];
      const double dzv = xx[j][2] - xx[i][2];
      const double dist = sqrt(dxv*dxv + dyv*dyv + dzv*dzv);

      const int jchem = chimes_type[atype[j] - 1];
      const int key = ichem * nchem + jchem;

      if (chimes_calculator->fast_2b(key, dist)) {
        const int nb = cnt[key];

        b_i[key*CHIMES_VLEN + nb] = i;
        b_j[key*CHIMES_VLEN + nb] = j;
        b_d[key*CHIMES_VLEN + nb] = dist;

        double *const pdr = &b_dr[(key*CHIMES_VLEN + nb)*CHDIM];

        pdr[0] = dxv;
        pdr[1] = dyv;
        pdr[2] = dzv;

        if (++cnt[key] == CHIMES_VLEN) flush(key);

        continue;
      }

      ldr[0] = dxv;
      ldr[1] = dyv;
      ldr[2] = dzv;

      ltyp[0] = ichem;
      ltyp[1] = jchem;

      std::fill(lforce.begin(), lforce.end(), 0.0);

      if (vflag_either) std::fill(lstress.begin(), lstress.end(), 0.0);

      energy = 0.0;

      chimes_calculator->compute_2B(dist, ldr, ltyp, lforce, lstress, energy, tmp, vflag_either);

      for (int idx = 0; idx < CHDIM; idx++) {
        a_f(i,idx) += lforce[idx];
        a_f(j,idx) += lforce[CHDIM + idx];
      }

      if (evflag) {
        const int alist[2] = {i, j};

        host_tally(2, alist, energy, lstress.data(), ev, a_eatom, a_vatom);
      }
    }
  }

  for (int key = 0; key < nkey; key++)
    if (cnt[key]) flush(key);
}

/* ----------------------------------------------------------------------
   3-body interactions for one block of the type-sorted triplet list.
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
void PairCHIMESKokkos<DeviceType>::host_3body_chunk(const int chunk, EV_FLOAT &ev) const
{
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  // Built once per chunk, not once per cluster: a duplicated scatter view's
  // accessor takes a unique token, and acquiring one is an atomic on a pool
  // shared by every thread.  Per cluster that atomic dominated the tally.

  auto v_eatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_eatom),decltype(ndup_eatom)>::get(dup_eatom,ndup_eatom);
  auto a_eatom = v_eatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  double **xx = atom->x;

  const int sz = (n_3mers + host_nchunk_3b - 1) / host_nchunk_3b;
  const int lo = chunk * sz;
  const int hi = MIN(lo + sz, n_3mers);

  chimes3BBatch &b = const_cast<chimes3BBatch &>(host_batch3[chunk]);

  double bdx[3][CHIMES_VLEN];
  double bdr[3][CHDIM][CHIMES_VLEN];
  int batom[3][CHIMES_VLEN];
  double dist_l[3], dr_l[3*CHDIM];
  double stensor[6];

  int nb = 0;
  int batch_type = -1;

  static const int pa[3] = {0, 0, 1}, pb[3] = {1, 2, 2};

  for (int ii = lo; ii <= hi; ii++) {
    int this_type = -1;
    int i = 0, j = 0, k = 0;

    if (ii < hi) {
      const int *mer = &neighborlist_3mers[3 * ii];

      i = mer[0];
      j = mer[1];
      k = mer[2];

      const int cand_type = mer_type_3b[ii];
      const chimesSlotConst *sc3 = chimes_calculator->slots_3B_idx(cand_type);

      if (sc3) {
        if (within(xx, i, j, sc3[0].outer_sq, &dr_l[0*CHDIM], dist_l[0]) &&
            within(xx, i, k, sc3[1].outer_sq, &dr_l[1*CHDIM], dist_l[1]) &&
            within(xx, j, k, sc3[2].outer_sq, &dr_l[2*CHDIM], dist_l[2]))
          this_type = cand_type;
      }
    }

    const bool at_end = (ii == hi);

    if (!at_end && (this_type < 0)) continue;

    if ((nb > 0) && (at_end || (this_type != batch_type) || (nb == CHIMES_VLEN))) {
      for (int p = 0; p < 3; p++)
        for (int l = nb; l < CHIMES_VLEN; l++) bdx[p][l] = bdx[p][0];

      chimes_calculator->compute_3B_batch(nb, batch_type, bdx, b);

      for (int l = 0; l < nb; l++) {
        const double fc0 = b.fcut[0][l];
        const double fc1 = b.fcut[1][l];
        const double fc2 = b.fcut[2][l];
        const double fcut_all = fc0 * fc1 * fc2;
        const double poly = b.poly[l];

        double fs[3];

        fs[0] = (fcut_all * b.dpoly[0][l] + b.fcutderiv[0][l] * fc1 * fc2 * poly) * b.inv_dx[0][l];
        fs[1] = (fcut_all * b.dpoly[1][l] + b.fcutderiv[1][l] * fc0 * fc2 * poly) * b.inv_dx[1][l];
        fs[2] = (fcut_all * b.dpoly[2][l] + b.fcutderiv[2][l] * fc0 * fc1 * poly) * b.inv_dx[2][l];

        if (vflag_either)
          for (int n = 0; n < 6; n++) stensor[n] = 0.0;

        double fatom[3][CHDIM] = {{0.0}};

        for (int p = 0; p < 3; p++) {
          for (int idx = 0; idx < CHDIM; idx++) {
            const double fpair = fs[p] * bdr[p][idx][l];

            fatom[pa[p]][idx] += fpair;
            fatom[pb[p]][idx] -= fpair;
          }

          if (vflag_either) {
            stensor[0] -= fs[p] * bdr[p][0][l] * bdr[p][0][l];
            stensor[1] -= fs[p] * bdr[p][0][l] * bdr[p][1][l];
            stensor[2] -= fs[p] * bdr[p][0][l] * bdr[p][2][l];
            stensor[3] -= fs[p] * bdr[p][1][l] * bdr[p][1][l];
            stensor[4] -= fs[p] * bdr[p][1][l] * bdr[p][2][l];
            stensor[5] -= fs[p] * bdr[p][2][l] * bdr[p][2][l];
          }
        }

        for (int a = 0; a < 3; a++) {
          const int ga = batom[a][l];

          for (int idx = 0; idx < CHDIM; idx++) a_f(ga,idx) += fatom[a][idx];
        }

        if (evflag) {
          const int alist[3] = {batom[0][l], batom[1][l], batom[2][l]};

          host_tally(3, alist, poly * fcut_all, stensor, ev, a_eatom, a_vatom);
        }
      }

      nb = 0;
    }

    if (at_end) break;

    batch_type = this_type;
    batom[0][nb] = i;
    batom[1][nb] = j;
    batom[2][nb] = k;

    for (int p = 0; p < 3; p++) {
      bdx[p][nb] = dist_l[p];

      for (int idx = 0; idx < CHDIM; idx++) bdr[p][idx][nb] = dr_l[p*CHDIM + idx];
    }

    nb++;
  }
}

/* ----------------------------------------------------------------------
   4-body interactions for one block of the type-sorted quadruplet list.
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
void PairCHIMESKokkos<DeviceType>::host_4body_chunk(const int chunk, EV_FLOAT &ev) const
{
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  // Built once per chunk, not once per cluster: a duplicated scatter view's
  // accessor takes a unique token, and acquiring one is an atomic on a pool
  // shared by every thread.  Per cluster that atomic dominated the tally.

  auto v_eatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_eatom),decltype(ndup_eatom)>::get(dup_eatom,ndup_eatom);
  auto a_eatom = v_eatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  double **xx = atom->x;

  const int sz = (n_4mers + host_nchunk_4b - 1) / host_nchunk_4b;
  const int lo = chunk * sz;
  const int hi = MIN(lo + sz, n_4mers);

  chimes4BBatch &b = const_cast<chimes4BBatch &>(host_batch4[chunk]);

  double bdx[6][CHIMES_VLEN];
  double bdr[6][CHDIM][CHIMES_VLEN];
  int batom[4][CHIMES_VLEN];
  double dist_l[6], dr_l[6*CHDIM];
  double stensor[6];

  int nb = 0;
  int batch_type = -1;

  static const int pa[6] = {0, 0, 0, 1, 1, 2}, pb[6] = {1, 2, 3, 2, 3, 3};

  for (int ii = lo; ii <= hi; ii++) {
    int this_type = -1;
    int i = 0, j = 0, k = 0, l = 0;

    if (ii < hi) {
      const int *mer = &neighborlist_4mers[4 * ii];

      i = mer[0];
      j = mer[1];
      k = mer[2];
      l = mer[3];

      const int cand_type = mer_type_4b[ii];
      const chimesSlotConst *sc4 = chimes_calculator->slots_4B_idx(cand_type);

      if (sc4) {
        if (within(xx, i, j, sc4[0].outer_sq, &dr_l[0*CHDIM], dist_l[0]) &&
            within(xx, i, k, sc4[1].outer_sq, &dr_l[1*CHDIM], dist_l[1]) &&
            within(xx, i, l, sc4[2].outer_sq, &dr_l[2*CHDIM], dist_l[2]) &&
            within(xx, j, k, sc4[3].outer_sq, &dr_l[3*CHDIM], dist_l[3]) &&
            within(xx, j, l, sc4[4].outer_sq, &dr_l[4*CHDIM], dist_l[4]) &&
            within(xx, k, l, sc4[5].outer_sq, &dr_l[5*CHDIM], dist_l[5]))
          this_type = cand_type;
      }
    }

    const bool at_end = (ii == hi);

    if (!at_end && (this_type < 0)) continue;

    if ((nb > 0) && (at_end || (this_type != batch_type) || (nb == CHIMES_VLEN))) {
      for (int p = 0; p < 6; p++)
        for (int lane = nb; lane < CHIMES_VLEN; lane++) bdx[p][lane] = bdx[p][0];

      chimes_calculator->compute_4B_batch(nb, batch_type, bdx, b);

      for (int lane = 0; lane < nb; lane++) {
        double fc[6], fcut_5[6];

        for (int p = 0; p < 6; p++) fc[p] = b.fcut[p][lane];

        const double fcut_all = fc[0] * fc[1] * fc[2] * fc[3] * fc[4] * fc[5];
        const double poly = b.poly[lane];

        double pre[6], suf[6];

        pre[0] = 1.0;
        suf[5] = 1.0;

        for (int p = 1; p < 6; p++) pre[p] = pre[p-1] * fc[p-1];

        for (int p = 4; p >= 0; p--) suf[p] = suf[p+1] * fc[p+1];

        for (int p = 0; p < 6; p++) fcut_5[p] = pre[p] * suf[p];

        if (vflag_either)
          for (int n = 0; n < 6; n++) stensor[n] = 0.0;

        double fatom[4][CHDIM] = {{0.0}};

        for (int p = 0; p < 6; p++) {
          const double fs = (fcut_all * b.dpoly[p][lane] +
                             b.fcutderiv[p][lane] * fcut_5[p] * poly) * b.inv_dx[p][lane];

          for (int idx = 0; idx < CHDIM; idx++) {
            const double fpair = fs * bdr[p][idx][lane];

            fatom[pa[p]][idx] += fpair;
            fatom[pb[p]][idx] -= fpair;
          }

          if (vflag_either) {
            stensor[0] -= fs * bdr[p][0][lane] * bdr[p][0][lane];
            stensor[1] -= fs * bdr[p][0][lane] * bdr[p][1][lane];
            stensor[2] -= fs * bdr[p][0][lane] * bdr[p][2][lane];
            stensor[3] -= fs * bdr[p][1][lane] * bdr[p][1][lane];
            stensor[4] -= fs * bdr[p][1][lane] * bdr[p][2][lane];
            stensor[5] -= fs * bdr[p][2][lane] * bdr[p][2][lane];
          }
        }

        for (int a = 0; a < 4; a++) {
          const int ga = batom[a][lane];

          for (int idx = 0; idx < CHDIM; idx++) a_f(ga,idx) += fatom[a][idx];
        }

        if (evflag) {
          const int alist[4] = {batom[0][lane], batom[1][lane], batom[2][lane], batom[3][lane]};

          host_tally(4, alist, poly * fcut_all, stensor, ev, a_eatom, a_vatom);
        }
      }

      nb = 0;
    }

    if (at_end) break;

    batch_type = this_type;
    batom[0][nb] = i;
    batom[1][nb] = j;
    batom[2][nb] = k;
    batom[3][nb] = l;

    for (int p = 0; p < 6; p++) {
      bdx[p][nb] = dist_l[p];

      for (int idx = 0; idx < CHDIM; idx++) bdr[p][idx][nb] = dr_l[p*CHDIM + idx];
    }

    nb++;
  }
}

/* ----------------------------------------------------------------------
   Launch the three host cluster kernels and sum their reductions.
------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
void PairCHIMESKokkos<DeviceType>::host_launch(EV_FLOAT &ev)
{
  EV_FLOAT ev_tmp;

  using policy_t = Kokkos::RangePolicy<DeviceType, Kokkos::Schedule<Kokkos::Dynamic>>;

  PairCHIMESHostClusterFunctor<DeviceType,2,NEIGHFLAG> fn2(this);
  Kokkos::parallel_reduce("CHIMESHost2Body",
      policy_t(0,host_nchunk_2b), fn2, ev_tmp);
  ev += ev_tmp;

  if (chimes_calculator->poly_orders[1] > 0) {
    PairCHIMESHostClusterFunctor<DeviceType,3,NEIGHFLAG> fn3(this);
    Kokkos::parallel_reduce("CHIMESHost3Body",
        policy_t(0,host_nchunk_3b), fn3, ev_tmp);
    ev += ev_tmp;
  }

  if (chimes_calculator->poly_orders[2] > 0) {
    PairCHIMESHostClusterFunctor<DeviceType,4,NEIGHFLAG> fn4(this);
    Kokkos::parallel_reduce("CHIMESHost4Body",
        policy_t(0,host_nchunk_4b), fn4, ev_tmp);
    ev += ev_tmp;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::compute_host(int eflag_in, int vflag_in)
{
  copymode = 1;

  eflag = eflag_in;
  vflag = vflag_in;

  ev_init(eflag,vflag,0);

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

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();

  inum = list->inum;

  setup_neighlist_ptrs();

  if (neighbor->ago == 0) host_build_mb_neighlists();

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

  host_setup_chunks();



  // Kokkos overwrites the reduction target on each launch rather than adding
  // to it, so each kernel reduces into its own value and the three are summed.

  EV_FLOAT ev;

  if (neighflag == HALF)
    host_launch<HALF>(ev);
  else
    host_launch<HALFTHREAD>(ev);

  if (need_dup) Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev.evdwl;

  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  atomKK->modified(execution_space,F_MASK);

  if (eflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);


  copymode = 0;

  if (need_dup) {
    dup_f     = {};
    dup_eatom = {};
    dup_vatom = {};
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  if constexpr (host_flag) {
    compute_host(eflag_in, vflag_in);
    return;
  }

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

  // Build the ChIMES many-body neighbor lists.. only do so when LAMMPS neighborlist has been updated

  if (0 && chimes_calculatorKK.rank == 0)
    std::cout << "Updating chimesFF neighbor lists..." << std::endl;

  build_mb_neighlists();

  if (0 && chimes_calculatorKK.rank == 0) {
    std::cout << "      Rank " << comm->me << " 2-body list size: " << size_2mers << std::endl;
    std::cout << "      Rank " << comm->me << " 3-body list size: " << size_3mers << std::endl;
    std::cout << "      Rank " << comm->me << " 4-body list size: " << size_4mers << std::endl;
    std::cout << "      ...update complete" << std::endl;
  }

  EV_FLOAT ev, ev_tmp;

  //Compute1Body
  {
    if (eflag_either) {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute1Body<HALF> > policy_2body(0,inum);
        Kokkos::parallel_reduce("Compute1Body", policy_2body, *this, ev_tmp);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute1Body<HALFTHREAD> > policy_2body(0,inum);
        Kokkos::parallel_reduce("Compute1Body", policy_2body, *this, ev_tmp);
      }
    }
  }
  ev += ev_tmp;

  //Compute2Body
  {
    if (evflag) {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALF,1> > policy_2body(0,size_2mers);
        Kokkos::parallel_reduce("Compute2Body", policy_2body, *this, ev_tmp);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALFTHREAD,1> > policy_2body(0,size_2mers);
        Kokkos::parallel_reduce("Compute2Body", policy_2body, *this, ev_tmp);
      }
    } else {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALF,0> > policy_2body(0,size_2mers);
        Kokkos::parallel_for("Compute2Body", policy_2body, *this);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALFTHREAD,0> > policy_2body(0,size_2mers);
        Kokkos::parallel_for("Compute2Body", policy_2body, *this);
      }
    }
  }
  ev += ev_tmp;

  //Compute3Body
  if (chimes_calculatorKK.poly_orders[1] > 0)
  {
    if (evflag) {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute3Body<HALF,1> > policy_3body(0,size_3mers);
        Kokkos::parallel_reduce("Compute3Body", policy_3body, *this, ev_tmp);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute3Body<HALFTHREAD,1> > policy_3body(0,size_3mers);
        Kokkos::parallel_reduce("Compute3Body", policy_3body, *this, ev_tmp);
      }
    } else {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute3Body<HALF,0> > policy_3body(0,size_3mers);
        Kokkos::parallel_for("Compute3Body", policy_3body, *this);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute3Body<HALFTHREAD,0> > policy_3body(0,size_3mers);
        Kokkos::parallel_for("Compute3Body", policy_3body, *this);
      }
    }
  }
  ev += ev_tmp;

  //Compute4Body
  if (chimes_calculatorKK.poly_orders[2] > 0)
  {
    if (evflag) {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute4Body<HALF,1> > policy_4body(0,size_4mers);
        Kokkos::parallel_reduce("Compute4Body", policy_4body, *this, ev_tmp);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute4Body<HALFTHREAD,1> > policy_4body(0,size_4mers);
        Kokkos::parallel_reduce("Compute4Body",policy_4body, *this, ev_tmp);
      }
    } else {
      if (neighflag == HALF) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute4Body<HALF,0> > policy_4body(0,size_4mers);
        Kokkos::parallel_for("Compute4Body", policy_4body, *this);
      } else if (neighflag == HALFTHREAD) {
        typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute4Body<HALFTHREAD,0> > policy_4body(0,size_4mers);
        Kokkos::parallel_for("Compute4Body", policy_4body, *this);
      }
    }
  }
  ev += ev_tmp;

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

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute1Body<NEIGHFLAG>, const int& ii, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 1-body interactions
  ////////////////////////////////////////

  // First, get the single-atom energy contribution

  const int i = d_ilist[ii /*+ chunk_offset*/];

  KK_FLOAT energy = 0.0;
  KK_FLOAT stensor[6];
  for (int n = 0; n < 6; n++) stensor[n] = 0.0;

  chimes_calculatorKK.compute_1B(type[i]-1, energy);

  int atmidxlst[6][2];
  atmidxlst[0][0] = i;

  ev_tally_mb<NEIGHFLAG>(1, 0, atmidxlst, energy, stensor, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
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

  const int ncount = d_numneigh[i];

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

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 3-body interactions
  ////////////////////////////////////////

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

  chimes_calculatorKK.compute_3B(dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy);

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
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 4-body interactions
  ////////////////////////////////////////

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

  chimes_calculatorKK.compute_4B(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy);

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
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ----------------------------------------------------------------------
   general ev tally function for many-body models where per-atom assignments
   do not make sense. Expects newton_pair = 1.
 ------------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::ev_tally_mb(int ninteractionatoms, int npairs,
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
template class PairCHIMESKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairCHIMESKokkos<LMPHostType>;
#endif
}
