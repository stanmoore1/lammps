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

#include "fix_oxdna_npair_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixOxdnaNpairKokkos<DeviceType>::FixOxdnaNpairKokkos(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | TYPE_MASK;
  datamask_modify = EMPTY_MASK;

  k_screened_pair_count = DAT::tdual_int_scalar("FixOxdnaNpair:screened_pair_count");
  screened_max_atoms = 0;
  screened_max_neigh = 0;
  screened_pair_count = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixOxdnaNpairKokkos<DeviceType>::~FixOxdnaNpairKokkos() = default;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixOxdnaNpairKokkos<DeviceType>::init()
{
  // adjust neighbor list request for KOKKOS
  neighbor->add_request(this);
  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();

  last_allocate = -1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixOxdnaNpairKokkos<DeviceType>::setmask()
{
  int mask = 0;
  mask |= MIN_PRE_FORCE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixOxdnaNpairKokkos<DeviceType>::init_list(int, class NeighList* ptr)
{
  this->list = ptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixOxdnaNpairKokkos<DeviceType>::min_setup_pre_force(int vflag)
{
  min_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixOxdnaNpairKokkos<DeviceType>::min_pre_force(int /*vflag*/)
{
  // TODO: Have a think about how often we need to do this?
  // Option 1: Every timestep (current implementation)
  if (execution_space != HostKK) compute_neigh_screen_to_npair();
  // Option 2: Only when neighbor list updates
  // if (execution_space != HostKK && last_allocate < neighbor->lastcall) {
  //   compute_neigh_screen_to_npair();
  //   last_allocate = update->ntimestep;
  // }
  // Option 3: Only every N timesteps
  // if (execution_space != HostKK && update->ntimestep > last_allocate + N) {
  //   compute_neigh_screen_to_npair();
  //   last_allocate = update->ntimestep;
  // }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixOxdnaNpairKokkos<DeviceType>::setup_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixOxdnaNpairKokkos<DeviceType>::pre_force(int /*vflag*/)
{
  // TODO: Have a think about how often we need to do this?
  // Option 1: Every timestep (current implementation)
  if (execution_space != HostKK) compute_neigh_screen_to_npair();
  // Option 2: Only when neighbor list updates
  // if (execution_space != HostKK && last_allocate < neighbor->lastcall) {
  //   compute_neigh_screen_to_npair();
  //   last_allocate = update->ntimestep;
  // }
  // Option 3: Only every N timesteps
  // if (execution_space != HostKK && update->ntimestep > last_allocate + N) {
  //   compute_neigh_screen_to_npair();
  //   last_allocate = update->ntimestep;
  // }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixOxdnaNpairKokkos<DeviceType>::compute_neigh_screen_to_npair()
{
  // get the neighbor list and neighbors used in operator()
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(this->list);
  d_neighbors = k_list->d_neighbors;
  anum = this->list->inum;
  d_alist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;

  // reallocate screened neighbor arrays if necessary
  screened_pair_count = 0;
  const int max_atoms = atom->nmax;
  const int max_neigh = d_neighbors.extent(1);
  if (max_atoms > screened_max_atoms || max_neigh > screened_max_neigh) {
    screened_max_atoms = max_atoms;
    screened_max_neigh = max_neigh;
    MemKK::realloc_kokkos(k_neighbors_screened, "FixOxdnaNpair:neighbors_screened",
                          screened_max_atoms, screened_max_neigh);
    MemKK::realloc_kokkos(k_numneigh_screened, "FixOxdnaNpair:numneigh_screened",
                          screened_max_atoms);
    MemKK::realloc_kokkos(k_screened_offsets, "FixOxdnaNpair:screened_offsets",
                          screened_max_atoms + 1);
    MemKK::realloc_kokkos(k_pairs_screened, "FixOxdnaNpair:pairs_screened",
              screened_max_atoms * screened_max_neigh);
    d_neighbors_screened = k_neighbors_screened.template view<DeviceType>();
    d_numneigh_screened = k_numneigh_screened.template view<DeviceType>();
    d_screened_offsets = k_screened_offsets.template view<DeviceType>();
    d_pairs_screened = k_pairs_screened.template view<DeviceType>();
  }

  atomKK->sync(execution_space, datamask_read);
  x = atomKK->k_x.view<DeviceType>();

  // Pretty simple first pass via "TagFixOxdnaNpairNeighScreen". We just loop through each atom a
  // and its neighbors, run 'screen_pair_fast' for each a-neighbor and its b-neighs which runs up
  // to a simple CoM bool. If true, we add the neighbor to the d_neighbors_screened neighbor
  // list and increment the screened neighbor count.
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixOxdnaNpairNeighScreen>(0, anum), *this);
  copymode = 0;

  // Perhaps "_local" suffixes are a little deceiving - these are shallow copies and point
  // to the same data as the non "_local" views. They're just for use in the lambdas below
  // to avoid "this->" captures which the compiler would not like. 
  const auto d_alist_local = d_alist;
  const auto d_numneigh_screened_local = d_numneigh_screened;
  const auto d_screened_offsets_local = d_screened_offsets;
  const auto d_neighbors_screened_local = d_neighbors_screened;
  const int anum_local = anum;

  // Final/Second pass is a little more conceptually complex. ComputeGPUPair (in our bond/pair styles)
  // takes two flat "pairs_screened_" a,b index(es) which runs from 0 to screened_pair_count - 1,
  // and does a global lookup to get the corresponding a,b for that pair index.
  // The parallel_scan is building a prefix sum
  // over the screened neighbor counts per atom, which gives us the starting index in
  // the screened neighbor list for each atom.
  // So for example if atom 0 has 2 screened neighbors, atom 1 has 0 screened neighbors,
  // and atom 2 has 3 screened neighbors, the scanned screened_offsets would be [0, 2, 2, 5].
  // The Kokkos documentation/wiki explains parallel_scan, prefix sum, "update", "final", etc
  // in more detail.
  // Pretty much populating d_pairs_screened with the corresponding a,b for each
  // screened pair index.
  const auto d_pairs_screened_local = d_pairs_screened;
  
  Kokkos::parallel_scan(
    Kokkos::RangePolicy<DeviceType>(0, anum + 1),
    KOKKOS_LAMBDA(const int i, int &update, const bool final) {
      if (i < anum_local) {
        if (final) d_screened_offsets_local(i) = update;
        const int a = d_alist_local(i);
        const int num_screened = d_numneigh_screened_local(a);
        if (final) {
          for (int ib = 0; ib < num_screened; ib++) {
            const int ipair = update + ib;
            const int b = d_neighbors_screened_local(a, ib);
            // So this utin64_t packing stores atom-a in the upper 32 bits and
            // atom-b in the lower 32 bits. We can unpack these in the ComputeGPUPair functors,
            // Meaning we only have to store one index and have one global load there.
            d_pairs_screened_local(ipair) =
              (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
              static_cast<uint64_t>(static_cast<uint32_t>(b));
          }
        }
        update += num_screened;
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

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
bool FixOxdnaNpairKokkos<DeviceType>::screen_pair_fast(const int &atype,
                                                         const int &braw,
                                                         const KK_FLOAT &a_com0,
                                                         const KK_FLOAT &a_com1,
                                                         const KK_FLOAT &a_com2) const
{
  const int b = braw & NEIGHMASK;
  const int btype = type(b);

  const KK_FLOAT b_com0 = x(b,0);
  const KK_FLOAT b_com1 = x(b,1);
  const KK_FLOAT b_com2 = x(b,2);

  KK_FLOAT delr_com[3];
  delr_com[0] = a_com0 - b_com0;
  delr_com[1] = a_com1 - b_com1;
  delr_com[2] = a_com2 - b_com2;

  // fma is fused-multipy-add op
  const KK_FLOAT rsq_com = fma(delr_com[2], delr_com[2],
                           fma(delr_com[1], delr_com[1], delr_com[0] * delr_com[0]));

  // Fast boolean screen: TODO: I've totally made up this cutoff - need a proper one...
  constexpr KK_FLOAT cutoffsq = 4;  // 1.69=1.3^2, 2.25=1.5^2, 4=2^2
  return (rsq_com < cutoffsq);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixOxdnaNpairKokkos<DeviceType>::operator()(TagFixOxdnaNpairNeighScreen, const int &ia) const
{
  const int a = d_alist(ia);
  const int atype = type(a);
  const int bnum = d_numneigh(a);
  const KK_FLOAT a_com0 = x(a,0);
  const KK_FLOAT a_com1 = x(a,1);
  const KK_FLOAT a_com2 = x(a,2);

  int nscreen = 0;
  for (int ib = 0; ib < bnum; ib++) {
    const int braw = d_neighbors(a,ib);
    if (screen_pair_fast(atype, braw, a_com0, a_com1, a_com2)) {
      d_neighbors_screened(a, nscreen++) = braw;
    }
  }
  d_numneigh_screened(a) = nscreen;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixOxdnaNpairKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixOxdnaNpairKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
