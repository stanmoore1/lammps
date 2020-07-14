/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "fix_acks2_reax_kokkos.h"
#include <cmath>
#include "kokkos.h"
#include "atom.h"
#include "atom_masks.h"
#include "atom_kokkos.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "memory_kokkos.h"
#include "error.h"
#include "pair_reaxc_kokkos.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define SMALL 0.0001
#define EV_TO_KCAL_PER_MOL 14.4

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixACKS2ReaxKokkos<DeviceType>::
FixACKS2ReaxKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixACKS2Reax(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | V_MASK | F_MASK | MASK_MASK | Q_MASK | TYPE_MASK | TAG_MASK;
  datamask_modify = Q_MASK | X_MASK;

  nmax = m_cap = 0;
  allocated_flag = 0;
  nprev = 4;

  memory->destroy(s_hist);
  memory->destroy(s_hist_X);
  memory->destroy(s_hist_last);
  grow_arrays(atom->nmax);

  d_mfill_offset = typename AT::t_int_scalar("acks2/kk:mfill_offset");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixACKS2ReaxKokkos<DeviceType>::~FixACKS2ReaxKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_s_hist,s_hist);
  memoryKK->destroy_kokkos(k_s_hist_X,s_hist_X);
  memoryKK->destroy_kokkos(k_s_hist_last,s_hist_last);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::init()
{
  atomKK->k_q.modify<LMPHostType>();
  atomKK->k_q.sync<DeviceType>();

  FixACKS2Reax::init();

  neighflag = lmp->kokkos->neighflag_qeq;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = std::is_same<DeviceType,LMPHostType>::value &&
    !std::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = std::is_same<DeviceType,LMPDeviceType>::value;

  if (neighflag == FULL) {
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
  } else { //if (neighflag == HALF || neighflag == HALFTHREAD)
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->full = 0;
    neighbor->requests[irequest]->half = 1;
    neighbor->requests[irequest]->ghost = 1;
  }

  int ntypes = atom->ntypes;
  k_params = Kokkos::DualView<params_acks2*,Kokkos::LayoutRight,DeviceType>
    ("FixACKS2Reax::params",ntypes+1);
  params = k_params.template view<DeviceType>();

  for (int n = 1; n <= ntypes; n++) {
    k_params.h_view(n).chi = chi[n];
    k_params.h_view(n).eta = eta[n];
    k_params.h_view(n).gamma = gamma[n];
    k_params.h_view(n).b_s_acks2 = b_s_acks2[n];
    k_params.h_view(n).refcharge = refcharge[n];
  }
  k_params.template modify<LMPHostType>();

  cutsq = swb * swb;

  init_shielding_k();
  init_hist();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::init_shielding_k()
{
  int i,j;
  int ntypes = atom->ntypes;

  k_shield = DAT::tdual_ffloat_2d("acks2/kk:shield",ntypes+1,ntypes+1);
  d_shield = k_shield.template view<DeviceType>();

  for( i = 1; i <= ntypes; ++i )
    for( j = 1; j <= ntypes; ++j )
      k_shield.h_view(i,j) = pow( gamma[i] * gamma[j], -1.5 );

  k_shield.template modify<LMPHostType>();
  k_shield.template sync<DeviceType>();

  k_bondcut = DAT::tdual_ffloat_2d("acks2/kk:bondcut",ntypes+1,ntypes+1);
  d_bondcut = k_bondcut.template view<DeviceType>();

  for( i = 1; i <= ntypes; ++i )
    for( j = 1; j <= ntypes; ++j )
      k_bondcut.h_view(i,j) = 0.5*(b_s_acks2[i] + b_s_acks2[j];

  k_bondcut.template modify<LMPHostType>();
  k_bondcut.template sync<DeviceType>();

  k_tap = DAT::tdual_ffloat_1d("acks2/kk:tap",8);
  d_tap = k_tap.template view<DeviceType>();

  for (i = 0; i < 8; i ++)
    k_tap.h_view(i) = Tap[i];

  k_tap.template modify<LMPHostType>();
  k_tap.template sync<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::init_hist()
{
  k_s_hist.clear_sync_state();
  k_s_hist_X.clear_sync_state();
  k_s_hist_last.clear_sync_state();

  Kokkos::deep_copy(d_s_hist,0.0);
  Kokkos::deep_copy(d_s_hist_X,0.0);
  Kokkos::deep_copy(d_s_hist_last,0.0);

  k_s_hist.template modify<DeviceType>();
  k_s_hist_X.template modify<DeviceType>();
  k_s_hist_last.template modify<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::setup_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::pre_force(int vflag)
{
  if (update->ntimestep % nevery) return;

  atomKK->sync(execution_space,datamask_read);

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  nlocal = atomKK->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;

  k_params.template sync<DeviceType>();
  k_shield.template sync<DeviceType>();
  k_bondcut.template sync<DeviceType>();
  k_tap.template sync<DeviceType>();

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  nn = list->inum;
  NN = list->inum + list->gnum;

  copymode = 1;

  // allocate

  allocate_array();

  // get max number of neighbor

  if (!allocated_flag || update->ntimestep == neighbor->lastcall)
    allocate_matrix();

  // compute_H

  if (execution_space == Host) { // CPU
    if (neighflag == FULL) {
      FixACKS2ReaxKokkosComputeHFunctor<DeviceType, FULL> computeH_functor(this);
      Kokkos::parallel_scan(nn,computeH_functor);
    } else { // HALF and HALFTHREAD are the same
      FixACKS2ReaxKokkosComputeHFunctor<DeviceType, HALF> computeH_functor(this);
      Kokkos::parallel_scan(nn,computeH_functor);
    }
  } else { // GPU, use teams
    Kokkos::deep_copy(d_mfill_offset,0);

    int vector_length = 32;
    int atoms_per_team = 4;
    int num_teams = nn / atoms_per_team + (nn % atoms_per_team ? 1 : 0);

    Kokkos::TeamPolicy<DeviceType> policy(num_teams, atoms_per_team,
                                          vector_length);
    if (neighflag == FULL) {
      FixACKS2ReaxKokkosComputeHFunctor<DeviceType, FULL> computeH_functor(
          this, atoms_per_team, vector_length);
      Kokkos::parallel_for(policy, computeH_functor);
    } else { // HALF and HALFTHREAD are the same
      FixACKS2ReaxKokkosComputeHFunctor<DeviceType, HALF> computeH_functor(
          this, atoms_per_team, vector_length);
      Kokkos::parallel_for(policy, computeH_functor);
    }
  }

  // compute_X

  Kokkos::deep_copy(d_X_diag,0.0);

  if (execution_space == Host) { // CPU
    if (neighflag == FULL) {
      FixACKS2ReaxKokkosComputeXFunctor<DeviceType, FULL> computeX_functor(this);
      Kokkos::parallel_scan(nn,computeX_functor);
    } else { // HALF and HALFTHREAD are the same
      FixACKS2ReaxKokkosComputeXFunctor<DeviceType, HALF> computeX_functor(this);
      Kokkos::parallel_scan(nn,computeX_functor);
    }
  } else { // GPU, use teams
    Kokkos::deep_copy(d_mfill_offset,0);

    int vector_length = 32;
    int atoms_per_team = 4;
    int num_teams = nn / atoms_per_team + (nn % atoms_per_team ? 1 : 0);

    Kokkos::TeamPolicy<DeviceType> policy(num_teams, atoms_per_team,
                                          vector_length);
    if (neighflag == FULL) {
      FixACKS2ReaxKokkosComputeHFunctor<DeviceType, FULL> computeX_functor(
          this, atoms_per_team, vector_length);
      Kokkos::parallel_for(policy, computeX_functor);
    } else { // HALF and HALFTHREAD are the same
      FixACKS2ReaxKokkosComputeHFunctor<DeviceType, HALF> computeX_functor(
          this, atoms_per_team, vector_length);
      Kokkos::parallel_for(policy, computeX_functor);
    }
  }

  // init_matvec

  k_s_hist.template sync<DeviceType>();
  k_s_hist_X.template sync<DeviceType>();
  k_s_hist_last.template sync<DeviceType>();
  FixACKS2ReaxKokkosMatVecFunctor<DeviceType> matvec_functor(this);
  Kokkos::parallel_for(nn,matvec_functor);

  // comm->forward_comm_fix(this); //Dist_vector( s );
  pack_flag = 2;
  k_s.template modify<DeviceType>();
  k_s.template sync<LMPHostType>();
  comm->forward_comm_fix(this);
  k_s.template modify<LMPHostType>();
  k_s.template sync<DeviceType>();

  need_dup = lmp->kokkos->need_dup<DeviceType>();

  if (need_dup)
    dup_o = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated> (d_o); // allocate duplicated memory
  else
    ndup_o = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated> (d_o);

  // bicgstab solve over b_s, s

  bicgstab_solve();

  // calculate_Q();

  calculate_q1();

  pack_flag = 2;
  comm->forward_comm_fix(this); //Dist_vector( s );

  calculate_q2();

  k_s_hist.template modify<DeviceType>();
  k_s_hist_X.template modify<DeviceType>();
  k_s_hist_last.template modify<DeviceType>();

  copymode = 0;

  if (!allocated_flag)
    allocated_flag = 1;

  // free duplicated memory

  if (need_dup)
    dup_o = decltype(dup_o)();

  atomKK->modified(execution_space,datamask_modify);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::num_neigh_item(int ii, int &maxneigh) const
{
  const int i = d_ilist[ii];
  maxneigh += d_numneigh[i];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::allocate_matrix()
{
  nmax = atom->nmax;

  // determine the total space for the H matrix

  m_cap = 0;
  FixACKS2ReaxKokkosNumNeighFunctor<DeviceType> neigh_functor(this);
  Kokkos::parallel_reduce(nn,neigh_functor,m_cap);

  // H matrix

  d_firstnbr_H = typename AT::t_int_1d("acks2/kk:firstnbr_H",nmax);
  d_numnbrs_H = typename AT::t_int_1d("acks2/kk:numnbrs_H",nmax);
  d_jlist_H = typename AT::t_int_1d("acks2/kk:jlist_H",m_cap);
  d_val_H = typename AT::t_ffloat_1d("acks2/kk:val_H",m_cap);

  // X matrix

  d_firstnbr_X = typename AT::t_int_1d("acks2/kk:firstnbr_X",nmax);
  d_numnbrs_X = typename AT::t_int_1d("acks2/kk:numnbrs_X",nmax);
  d_jlist_X = typename AT::t_int_1d("acks2/kk:jlist_X",m_cap);
  d_val_X = typename AT::t_ffloat_1d("acks2/kk:val_X",m_cap);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::allocate_array()
{
  // 0 to nn-1: owned atoms related to H matrix
  // nn to NN-1: ghost atoms related to H matrix
  // NN to NN+nn-1: owned atoms related to X matrix
  // NN+nn to 2*NN-1: ghost atoms related X matrix
  // 2*NN to 2*NN+1: last two rows, owned by proc 0

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    int size = nmax*2 + 2;

    k_o = DAT::tdual_ffloat_1d("acks2/kk:o",nmax);
    d_o = k_o.template view<DeviceType>();
    h_o = k_o.h_view;

    k_s = DAT::tdual_ffloat_1d("acks2/kk:s",size);
    d_s = k_s.template view<DeviceType>();
    h_s = k_s.h_view;

    d_b_s = typename AT::t_ffloat_1d("acks2/kk:b_s",size);

    d_Hdia_inv = typename AT::t_ffloat_1d("acks2/kk:Hdia_inv",nmax);
    d_chi_field = typename AT::t_ffloat_1d("acks2/kk:chi_field",nmax);

    d_X_diag = typename AT::t_ffloat_1d("acks2/kk:X_diag",nmax);
    d_Xdia_inv = typename AT::t_ffloat_1d("acks2/kk:Xdia_inv",nmax);


    d_p = typename AT::t_ffloat_1d("acks2/kk:p",size);
    d_r = typename AT::t_ffloat_1d("acks2/kk:r",size);

    k_d = DAT::tdual_ffloat_1d("acks2/kk:d",size);
    d_d = k_d.template view<DeviceType>();
    h_d = k_d.h_view;

    d_g = typename AT::t_ffloat_1d("acks2/kk:g",size);
    d_q_hat = typename AT::t_ffloat_1d("acks2/kk:q_hat",size);
    d_r_hat = typename AT::t_ffloat_1d("acks2/kk:r_hat",size);
    d_y = typename AT::t_ffloat_1d("acks2/kk:y",size);
    d_z = typename AT::t_ffloat_1d("acks2/kk:z",size);
  }

  // init_storage
  const int ignum = atom->nlocal + atom->nghost;
  FixACKS2ReaxKokkosZeroFunctor<DeviceType> zero_functor(this);
  Kokkos::parallel_for(ignum,zero_functor); ///////////

}
/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::zero_item(int ii) const
{
  const int i = d_ilist[ii];
  const int itype = type(i);

  if (mask[i] & groupbit) {
    d_Hdia_inv[i] = 1.0 / params(itype).eta;
    d_b_s[i] = -params(itype).chi - d_chi_field[i];
    d_b_t[i] = -1.0;
    d_s[i] = 0.0;
    d_t[i] = 0.0;
    d_p[i] = 0.0;
    d_o[i] = 0.0;
    d_r[i] = 0.0;
    d_d[i] = 0.0;
  }

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::compute_h_item(int ii, int &m_fill, const bool &final) const
{
  const int i = d_ilist[ii];
  int j,jj,jtype;

  if (mask[i] & groupbit) {

    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);
    const int itype = type(i);
    const tagint itag = tag(i);
    const int jnum = d_numneigh[i];
    if (final)
      d_firstnbr_H[i] = m_fill;

    for (jj = 0; jj < jnum; jj++) {
      j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      jtype = type(j);

      const X_FLOAT delx = x(j,0) - xtmp;
      const X_FLOAT dely = x(j,1) - ytmp;
      const X_FLOAT delz = x(j,2) - ztmp;

      if (NEIGHFLAG != FULL) {
        // skip half of the interactions
        const tagint jtag = tag(j);
        if (j >= nlocal) {
          if (itag > jtag) {
            if ((itag+jtag) % 2 == 0) continue;
          } else if (itag < jtag) {
            if ((itag+jtag) % 2 == 1) continue;
          } else {
            if (x(j,2) < ztmp) continue;
            if (x(j,2) == ztmp && x(j,1)  < ytmp) continue;
            if (x(j,2) == ztmp && x(j,1) == ytmp && x(j,0) < xtmp) continue;
          }
        }
      }

      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      if (rsq > cutsq) continue;

      if (final) {
        const F_FLOAT r = sqrt(rsq);
        d_jlist_H(m_fill) = j;
        const F_FLOAT shldij = d_shield(itype,jtype);
        d_val_H(m_fill) = calculate_H_k(r,shldij);
      }
      m_fill++;
    }
    if (final)
      d_numnbrs_H[i] = m_fill - d_firstnbr_H[i];
  }
}

/* ---------------------------------------------------------------------- */

// Calculate Qeq matrix H where H is a sparse matrix and H[i][j] represents the electrostatic interaction coefficients on atom-i with atom-j
// d_val     - contains the non-zero entries of sparse matrix H
// d_numnbrs - d_numnbrs[i] contains the # of non-zero entries in the i-th row of H (which also represents the # of neighbor atoms with electrostatic interaction coefficients with atom-i)
// d_firstnbr- d_firstnbr[i] contains the beginning index from where the H matrix entries corresponding to row-i is stored in d_val
// d_jlist   - contains the column index corresponding to each entry in d_val

template <class DeviceType>
template <int NEIGHFLAG>
void FixACKS2ReaxKokkos<DeviceType>::compute_h_team(
    const typename Kokkos::TeamPolicy<DeviceType>::member_type &team,
    int atoms_per_team, int vector_length) const {

  // scratch space setup
  Kokkos::View<int *, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_ilist(team.team_shmem(), atoms_per_team);
  Kokkos::View<int *, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_numnbrs(team.team_shmem(), atoms_per_team);
  Kokkos::View<int *, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_firstnbr(team.team_shmem(), atoms_per_team);

  Kokkos::View<int **, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_jtype(team.team_shmem(), atoms_per_team, vector_length);
  Kokkos::View<int **, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_jlist(team.team_shmem(), atoms_per_team, vector_length);
  Kokkos::View<F_FLOAT **, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_r(team.team_shmem(), atoms_per_team, vector_length);

  // team of threads work on atoms with index in [firstatom, lastatom)
  int firstatom = team.league_rank() * atoms_per_team;
  int lastatom =
      (firstatom + atoms_per_team < nn) ? (firstatom + atoms_per_team) : nn;

  // kokkos-thread-0 is used to load info from global memory into scratch space
  if (team.team_rank() == 0) {

    // copy atom indices from d_ilist[firstatom:lastatom] to scratch space s_ilist[0:atoms_per_team]
    // copy # of neighbor atoms for all the atoms with indices in d_ilist[firstatom:lastatom] from d_numneigh to scratch space s_numneigh[0:atoms_per_team]
    // calculate total number of neighbor atoms for all atoms assigned to the current team of threads (Note - Total # of neighbor atoms here provides the
    // upper bound space requirement to store the H matrix values corresponding to the atoms with indices in d_ilist[firstatom:lastatom])

    Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team, atoms_per_team),
                          [&](const int &idx, int &totalnbrs, bool final) {
                            int ii = firstatom + idx;

                            if (ii < nn) {
                              const int i = d_ilist[ii];
                              int jnum = d_numneigh[i];

                              if (final) {
                                s_ilist[idx] = i;
                                s_numnbrs[idx] = jnum;
                                s_firstnbr[idx] = totalnbrs;
                              }
                              totalnbrs += jnum;
                            } else {
                              s_numnbrs[idx] = 0;
                            }
                          });
  }

  // barrier ensures that the data moved to scratch space is visible to all the
  // threads of the corresponding team
  team.team_barrier();

  // calculate the global memory offset from where the H matrix values to be
  // calculated by the current team will be stored in d_val
  int team_firstnbr_idx = 0;
  Kokkos::single(Kokkos::PerTeam(team),
                 [=](int &val) {
                   int totalnbrs = s_firstnbr[lastatom - firstatom - 1] +
                                   s_numnbrs[lastatom - firstatom - 1];
                   val = Kokkos::atomic_fetch_add(&d_mfill_offset(), totalnbrs);
                 },
                 team_firstnbr_idx);

  // map the H matrix computation of each atom to kokkos-thread (one atom per
  // kokkos-thread) neighbor computation for each atom is assigned to vector
  // lanes of the corresponding thread
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, atoms_per_team), [&](const int &idx) {
        int ii = firstatom + idx;

        if (ii < nn) {
          const int i = s_ilist[idx];

          if (mask[i] & groupbit) {
            const X_FLOAT xtmp = x(i, 0);
            const X_FLOAT ytmp = x(i, 1);
            const X_FLOAT ztmp = x(i, 2);
            const int itype = type(i);
            const tagint itag = tag(i);
            const int jnum = s_numnbrs[idx];

            // calculate the write-offset for atom-i's first neighbor
            int atomi_firstnbr_idx = team_firstnbr_idx + s_firstnbr[idx];
            Kokkos::single(Kokkos::PerThread(team),
                           [&]() { d_firstnbr_H[i] = atomi_firstnbr_idx; });

            // current # of neighbor atoms with non-zero electrostatic
            // interaction coefficients with atom-i which represents the # of
            // non-zero elements in row-i of H matrix
            int atomi_nbrs_inH = 0;

            // calculate H matrix values corresponding to atom-i where neighbors
            // are processed in batches and the batch size is vector_length
            for (int jj_start = 0; jj_start < jnum; jj_start += vector_length) {

              int atomi_nbr_writeIdx = atomi_firstnbr_idx + atomi_nbrs_inH;

              // count the # of neighbor atoms with non-zero electrostatic
              // interaction coefficients with atom-i in the current batch
              int atomi_nbrs_curbatch = 0;

              // compute rsq, jtype, j and store in scratch space which is
              // reused later
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team, vector_length),
                  [&](const int &idx, int &m_fill) {
                    const int jj = jj_start + idx;

                    // initialize: -1 represents no interaction with atom-j
                    // where j = d_neighbors(i,jj)
                    s_jlist(team.team_rank(), idx) = -1;

                    if (jj < jnum) {
                      int j = d_neighbors(i, jj);
                      j &= NEIGHMASK;
                      const int jtype = type(j);

                      const X_FLOAT delx = x(j, 0) - xtmp;
                      const X_FLOAT dely = x(j, 1) - ytmp;
                      const X_FLOAT delz = x(j, 2) - ztmp;

                      // valid nbr interaction
                      bool valid = true;
                      if (NEIGHFLAG != FULL) {
                        // skip half of the interactions
                        const tagint jtag = tag(j);
                        if (j >= nlocal) {
                          if (itag > jtag) {
                            if ((itag + jtag) % 2 == 0)
                              valid = false;
                          } else if (itag < jtag) {
                            if ((itag + jtag) % 2 == 1)
                              valid = false;
                          } else {
                            if (x(j, 2) < ztmp)
                              valid = false;
                            if (x(j, 2) == ztmp && x(j, 1) < ytmp)
                              valid = false;
                            if (x(j, 2) == ztmp && x(j, 1) == ytmp &&
                                x(j, 0) < xtmp)
                              valid = false;
                          }
                        }
                      }

                      const F_FLOAT rsq =
                          delx * delx + dely * dely + delz * delz;
                      if (rsq > cutsq)
                        valid = false;

                      if (valid) {
                        s_jlist(team.team_rank(), idx) = j;
                        s_jtype(team.team_rank(), idx) = jtype;
                        s_r(team.team_rank(), idx) = sqrt(rsq);
                        m_fill++;
                      }
                    }
                  },
                  atomi_nbrs_curbatch);

              // write non-zero entries of H to global memory
              Kokkos::parallel_scan(
                  Kokkos::ThreadVectorRange(team, vector_length),
                  [&](const int &idx, int &m_fill, bool final) {
                    int j = s_jlist(team.team_rank(), idx);
                    if (final) {
                      if (j != -1) {
                        const int jtype = s_jtype(team.team_rank(), idx);
                        const F_FLOAT r = s_r(team.team_rank(), idx);
                        const F_FLOAT shldij = d_shield(itype, jtype);

                        d_jlist_H[atomi_nbr_writeIdx + m_fill] = j;
                        d_val_H[atomi_nbr_writeIdx + m_fill] =
                            calculate_H_k(r, shldij);
                      }
                    }

                    if (j != -1) {
                      m_fill++;
                    }
                  });
              atomi_nbrs_inH += atomi_nbrs_curbatch;
            }

            Kokkos::single(Kokkos::PerThread(team),
                           [&]() { d_numnbrs_H[i] = atomi_nbrs_inH; });
          }
        }
      });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double FixACKS2ReaxKokkos<DeviceType>::calculate_H_k(const F_FLOAT &r, const F_FLOAT &shld) const
{
  F_FLOAT taper, denom;

  taper = d_tap[7] * r + d_tap[6];
  taper = taper * r + d_tap[5];
  taper = taper * r + d_tap[4];
  taper = taper * r + d_tap[3];
  taper = taper * r + d_tap[2];
  taper = taper * r + d_tap[1];
  taper = taper * r + d_tap[0];

  denom = r * r * r + shld;
  denom = pow(denom,0.3333333333333);

  return taper * EV_TO_KCAL_PER_MOL / denom;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::compute_x_item(int ii, int &m_fill, const bool &final) const
{
  const int i = d_ilist[ii];
  int j,jj,jtype;

  if (mask[i] & groupbit) {

    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);
    const int itype = type(i);
    const tagint itag = tag(i);
    const int jnum = d_numneigh[i];
    if (final)
      d_firstnbr_X[i] = m_fill;

    for (jj = 0; jj < jnum; jj++) {
      j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      jtype = type(j);

      const X_FLOAT delx = x(j,0) - xtmp;
      const X_FLOAT dely = x(j,1) - ytmp;
      const X_FLOAT delz = x(j,2) - ztmp;

      if (NEIGHFLAG != FULL) {
        // skip half of the interactions
        const tagint jtag = tag(j);
        if (j >= nlocal) {
          if (itag > jtag) {
            if ((itag+jtag) % 2 == 0) continue;
          } else if (itag < jtag) {
            if ((itag+jtag) % 2 == 1) continue;
          } else {
            if (x(j,2) < ztmp) continue;
            if (x(j,2) == ztmp && x(j,1)  < ytmp) continue;
            if (x(j,2) == ztmp && x(j,1) == ytmp && x(j,0) < xtmp) continue;
          }
        }
      }

      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      if (rsq > cutsq) continue;

      const F_FLOAT bcutoff = d_bcut(itype,jtype);
      const F_FLOAT bcutoff2 = bcutoff*bcutoff;
      if (r_sqr > bcutoff2) continue;

      if (final) {
        const F_FLOAT r = sqrt(rsq);
        d_jlist_X(m_fill) = j;
        const F_FLOAT X_val = calculate_X_k(r,bcutoff);
        d_val_X(m_fill) = X_val;
        d_X_diag[i] -= X_val;
        d_X_diag[j] -= X_val;
      }
      m_fill++;
    }
    if (final)
      d_numnbrs_X[i] = m_fill - d_firstnbr_X[i];
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG>
void FixACKS2ReaxKokkos<DeviceType>::compute_x_team(
    const typename Kokkos::TeamPolicy<DeviceType>::member_type &team,
    int atoms_per_team, int vector_length) const {

  // scratch space setup
  Kokkos::View<int *, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_ilist(team.team_shmem(), atoms_per_team);
  Kokkos::View<int *, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_numnbrs(team.team_shmem(), atoms_per_team);
  Kokkos::View<int *, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_firstnbr(team.team_shmem(), atoms_per_team);

  Kokkos::View<int **, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_jtype(team.team_shmem(), atoms_per_team, vector_length);
  Kokkos::View<int **, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_jlist(team.team_shmem(), atoms_per_team, vector_length);
  Kokkos::View<F_FLOAT **, Kokkos::ScratchMemorySpace<DeviceType>,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      s_r(team.team_shmem(), atoms_per_team, vector_length);

  // team of threads work on atoms with index in [firstatom, lastatom)
  int firstatom = team.league_rank() * atoms_per_team;
  int lastatom =
      (firstatom + atoms_per_team < nn) ? (firstatom + atoms_per_team) : nn;

  // kokkos-thread-0 is used to load info from global memory into scratch space
  if (team.team_rank() == 0) {

   // copy atom indices from d_ilist[firstatom:lastatom] to scratch space s_ilist[0:atoms_per_team]
    // copy # of neighbor atoms for all the atoms with indices in d_ilist[firstatom:lastatom] from d_numneigh to scratch space s_numneigh[0:atoms_per_team]
    // calculate total number of neighbor atoms for all atoms assigned to the current team of threads (Note - Total # of neighbor atoms here provides the
    // upper bound space requirement to store the H matrix values corresponding to the atoms with indices in d_ilist[firstatom:lastatom])

    Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team, atoms_per_team),
                          [&](const int &idx, int &totalnbrs, bool final) {
                            int ii = firstatom + idx;

                            if (ii < nn) {
                              const int i = d_ilist[ii];
                              int jnum = d_numneigh[i];

                              if (final) {
                                s_ilist[idx] = i;
                                s_numnbrs[idx] = jnum;
                                s_firstnbr[idx] = totalnbrs;
                              }
                              totalnbrs += jnum;
                            } else {
                              s_numnbrs[idx] = 0;
                            }
                          });
  }

  // barrier ensures that the data moved to scratch space is visible to all the
  // threads of the corresponding team
  team.team_barrier();

  // calculate the global memory offset from where the H matrix values to be
  // calculated by the current team will be stored in d_val_X
  int team_firstnbr_idx = 0;
  Kokkos::single(Kokkos::PerTeam(team),
                 [=](int &val) {
                   int totalnbrs = s_firstnbr[lastatom - firstatom - 1] +
                                   s_numnbrs[lastatom - firstatom - 1];
                   val = Kokkos::atomic_fetch_add(&d_mfill_offset(), totalnbrs);
                 },
                 team_firstnbr_idx);

  // map the H matrix computation of each atom to kokkos-thread (one atom per
  // kokkos-thread) neighbor computation for each atom is assigned to vector
  // lanes of the corresponding thread
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, atoms_per_team), [&](const int &idx) {
        int ii = firstatom + idx;

        if (ii < nn) {
          const int i = s_ilist[idx];

          if (mask[i] & groupbit) {
            const X_FLOAT xtmp = x(i, 0);
            const X_FLOAT ytmp = x(i, 1);
            const X_FLOAT ztmp = x(i, 2);
            const int itype = type(i);
            const tagint itag = tag(i);
            const int jnum = s_numnbrs[idx];

            // calculate the write-offset for atom-i's first neighbor
            int atomi_firstnbr_idx = team_firstnbr_idx + s_firstnbr[idx];
            Kokkos::single(Kokkos::PerThread(team),
                           [&]() { d_firstnbr_X[i] = atomi_firstnbr_idx; });

            // current # of neighbor atoms with non-zero electrostatic
            // interaction coefficients with atom-i which represents the # of
            // non-zero elements in row-i of H matrix
            int atomi_nbrs_inH = 0;

            // calculate H matrix values corresponding to atom-i where neighbors
            // are processed in batches and the batch size is vector_length
            for (int jj_start = 0; jj_start < jnum; jj_start += vector_length) {

              int atomi_nbr_writeIdx = atomi_firstnbr_idx + atomi_nbrs_inH;

              // count the # of neighbor atoms with non-zero electrostatic
              // interaction coefficients with atom-i in the current batch
              int atomi_nbrs_curbatch = 0;

              // compute rsq, jtype, j and store in scratch space which is
              // reused later
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team, vector_length),
                  [&](const int &idx, int &m_fill) {
                    const int jj = jj_start + idx;

                    // initialize: -1 represents no interaction with atom-j
                    // where j = d_neighbors(i,jj)
                    s_jlist(team.team_rank(), idx) = -1;

                    if (jj < jnum) {
                      int j = d_neighbors(i, jj);
                      j &= NEIGHMASK;
                      const int jtype = type(j);

                      const X_FLOAT delx = x(j, 0) - xtmp;
                      const X_FLOAT dely = x(j, 1) - ytmp;
                      const X_FLOAT delz = x(j, 2) - ztmp;

                      // valid nbr interaction
                      bool valid = true;
                      if (NEIGHFLAG != FULL) {
                        // skip half of the interactions
                        const tagint jtag = tag(j);
                        if (j >= nlocal) {
                          if (itag > jtag) {
                            if ((itag + jtag) % 2 == 0)
                              valid = false;
                          } else if (itag < jtag) {
                            if ((itag + jtag) % 2 == 1)
                              valid = false;
                          } else {
                            if (x(j, 2) < ztmp)
                              valid = false;
                            if (x(j, 2) == ztmp && x(j, 1) < ytmp)
                              valid = false;
                            if (x(j, 2) == ztmp && x(j, 1) == ytmp &&
                                x(j, 0) < xtmp)
                              valid = false;
                          }
                        }
                      }

                      const F_FLOAT rsq =
                          delx * delx + dely * dely + delz * delz;
                      if (rsq > cutsq)
                        valid = false;

                       const F_FLOAT bcutoff = d_bcut(itype,jtype];
                       const F_FLOAT bcutoff2 = bcutoff*bcutoff;
                       if (rsq > bcutoff2)
                         valid = false;

                      if (valid) {
                        s_jlist(team.team_rank(), idx) = j;
                        s_jtype(team.team_rank(), idx) = jtype;
                        s_r(team.team_rank(), idx) = sqrt(rsq);
                        m_fill++;
                      }
                    }
                  },
                  atomi_nbrs_curbatch);

              // write non-zero entries of H to global memory
              Kokkos::parallel_scan(
                  Kokkos::ThreadVectorRange(team, vector_length),
                  [&](const int &idx, int &m_fill, bool final) {
                    int j = s_jlist(team.team_rank(), idx);
                    if (final) {
                      if (j != -1) {
                        const int jtype = s_jtype(team.team_rank(), idx);
                        const F_FLOAT r = s_r(team.team_rank(), idx);
                        const F_FLOAT bcutoff = d_bcut(itype, jtype);

                        d_jlist_X[atomi_nbr_writeIdx + m_fill] = j;
                        const F_FLOAT X_val = calculate_X_k(r, bcutoff);
                        d_val_X[atomi_nbr_writeIdx + m_fill] =
                            X_val;
                        X_diag[i] -= X_val;
                        X_diag[j] -= X_val;
                      }
                    }

                    if (j != -1) {
                      m_fill++;
                    }
                  });
              atomi_nbrs_inH += atomi_nbrs_curbatch;
            }

            Kokkos::single(Kokkos::PerThread(team),
                           [&]() { d_numnbrs_X[i] = atomi_nbrs_inH; });
          }
        }
      });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::calculate_X( double r, double bcut)
{
  const F_FLOAT d = r/bcut;
  const F_FLOAT d3 = d*d*d;
  const F_FLOAT omd = 1.0 - d;
  const F_FLOAT omd2 = omd*omd;
  const F_FLOAT omd6 = omd2*omd2*omd2;

  return bond_softness*d3*omd6;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::matvec_item(int ii) const
{
  const int i = d_ilist[ii];
  const int itype = type(i);

  if (mask[i] & groupbit) {
    d_Hdia_inv[i] = 1.0 / params(itype).eta;
    d_b_s[i] = -params(itype).chi - d_chi_field[i];
    d_b_s[NN+i] = d_refcharge(itype);

    d_s[i] = 4*(d_s_hist(i,0)+d_s_hist(i,2))-(6*d_s_hist(i,1)+d_s_hist(i,3));
    d_s[NN+i] = 4*(d_s_hist_X(i,0)+d_s_hist_X(i,2))-(6*d_s_hist_X(i,1)+d_s_hist_X(i,3));
  }

  // last two rows
  if (proc_0_flag && ii == 0) {
    for (i = 0; i < 2; i++) {
      d_b_s[2*NN+i] = 0.0;
      d_s[2*NN+i] = 4*(d_s_hist_last[i][0]+d_s_hist_last[i][2])-(6*d_s_hist_last[i][1]+d_s_hist_last[i][3]);
    }

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::bicgstab_solve()
{
  F_FLOAT my_norm,norm_sqr,my_dot,dot_sqr;

  int teamsize;
  if (execution_space == Host) teamsize = 1;
  else teamsize = 128;

  // sparse_matvec( &H, &X, x, d );
  if (neighflag != FULL) {
    sparse_matvec_acks2_half(d_s, d_d);

    k_d.template modify<DeviceType>();
    k_d.template sync<LMPHostType>();
    pack_flag = 1;
    comm->reverse_comm_fix(this); //Coll_vector( d );
    more_reverse_comm(k_d.h_view.data());
    k_d.template modify<LMPHostType>();
    k_d.template sync<DeviceType>();
  } else
    sparse_matvec_acks2_full(d_s, d_d);

  // vector_sum( r , 1.,  b, -1., d, nn );
  // b_norm = parallel_norm( b, nn );
  my_norm = 0.0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,TagNorm1>(0,nn),*this,my_norm);
  norm_sqr = 0.0;
  MPI_Allreduce( &my_norm, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world );
  b_norm = sqrt(norm_sqr);

  // rnorm = parallel_norm( r, nn);
  my_norm = 0.0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,TagNorm2>(0,nn),*this,my_norm);
  norm_sqr = 0.0;
  MPI_Allreduce( &my_norm, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world );
  r_norm = sqrt(norm_sqr);

  if (bnorm == 0.0 ) bnorm = 1.0;
  deep_copy(d_r_hat,d_r);
  omega = 1.0;
  rho = 1.0;

  for (loop = 1; loop < imax && rnorm / b_norm > tolerance; ++loop) {
    // rho = parallel_dot( r_hat, r, nn);
    my_dot = 0.0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,TagDot1>(0,nn),*this,my_dot);
    dot_sqr = 0.0;
    MPI_Allreduce( &my_dot, &dot_sqr, 1, MPI_DOUBLE, MPI_SUM, world );
    rho = dot_sqr;
    if (rho == 0.0) break;

    if (loop > 1)
      beta = (rho / rho_old) * (alpha / omega);

    // vector_sum( p , 1., r, beta, q, nn);
    // vector_sum( q , 1., p, -omega, z, nn);
    // pre-conditioning
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagPrecon1>(0,nn),*this);

    // comm->forward_comm_fix(this); //Dist_vector( d );
    pack_flag = 1;
    k_d.template modify<DeviceType>();
    k_d.template sync<LMPHostType>();
    comm->forward_comm_fix(this);
    more_forward_comm(k_d.h_view.data());
    k_d.template modify<LMPHostType>();
    k_d.template sync<DeviceType>();

    // sparse_matvec( &H, &X, d, z );
    if (neighflag != FULL) {
      sparse_matvec_acks2_half(d_d, d_z);

      k_d.template modify<DeviceType>();
      k_d.template sync<LMPHostType>();
      pack_flag = 2;
      comm->reverse_comm_fix(this); //Coll_vector( z );
      more_reverse_comm(k_z.h_view.data());
      k_d.template modify<LMPHostType>();
      k_d.template sync<DeviceType>();
    } else
      sparse_matvec_acks2_full(d_d, d_z);

    // tmp = parallel_dot( r_hat, z, nn);
    my_dot = dot_sqr = 0.0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,TagDot2>(0,nn),*this,my_dot);
    MPI_Allreduce( &my_dot, &dot_sqr, 1, MPI_DOUBLE, MPI_SUM, world );
    tmp = dot_sqr;
    alpha = rho / tmp;

    // vector_sum( q, 1., r, -alpha, z, nn);
    // tmp = parallel_dot( q, q, nn);
    my_dot = dot_sqr = 0.0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,TagDot3>(0,nn),*this,my_dot);
    MPI_Allreduce( &my_dot, &dot_sqr, 1, MPI_DOUBLE, MPI_SUM, world );
    tmp = dot_sqr;

    // early convergence check
    if (tmp < tolerance) {
      // vector_add( x, alpha, d, nn);
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagAdd>(0,nn),*this);
      break;
    }

    // pre-conditioning
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagPrecon2>(0,nn),*this);

    // comm->forward_comm_fix(this); //Dist_vector( q_hat );
    pack_flag = 3;
    k_q_hat.template modify<DeviceType>();
    k_q_hat.template sync<LMPHostType>();
    comm->forward_comm_fix(this);
    more_forward_comm(k_q_hat.h_view.data());
    k_q_hat.template modify<LMPHostType>();
    k_q_hat.template sync<DeviceType>();

    sparse_matvec_acks2( &H, &X, q_hat, y );
    pack_flag = 3;
    comm->reverse_comm_fix(this); //Dist_vector( y );
    more_reverse_comm(y);

    // sparse_matvec( &H, &X, q_hat, y );
    if (neighflag != FULL) {
      sparse_matvec_acks2_half(d_q_hat, d_y);

      k_d.template modify<DeviceType>();
      k_d.template sync<LMPHostType>();
      pack_flag = 2;
      comm->reverse_comm_fix(this); //Coll_vector( z );
      more_reverse_comm(k_z.h_view.data());
      k_d.template modify<LMPHostType>();
      k_d.template sync<DeviceType>();
    } else
      sparse_matvec_acks2_full(d_d, d_z);

    sigma = parallel_dot( y, q, nn);
    tmp = parallel_dot( y, y, nn);
    omega = sigma / tmp;

    // vector_sum( g , alpha, d, omega, q_hat, nn);
    // vector_add( x, 1., g, nn);
    // vector_sum( r , 1., q, -omega, y, nn);
    // rnorm = parallel_norm( r, nn);
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagNorm2>(0,nn),*this);

    if (omega == 0) break;
    rho_old = rho;
  }

  if (comm->me == 0) {
    if (omega == 0 || rho == 0) {
      char str[128];
      sprintf(str,"Fix acks2/reax/kk BiCGStab numerical breakdown, omega = %g, rho = %g",omega,rho);
      error->warning(FLERR,str);
    } else if (loop >= imax) {
      char str[128];
      sprintf(str,"Fix acks2/reax/kk BiCGStab convergence failed after %d iterations "
              "at " BIGINT_FORMAT " step",i,update->ntimestep);
      error->warning(FLERR,str);
    }
  }

  return loop;
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::calculate_q()
{
  // q[i] = s[i];
  FixACKS2ReaxKokkosCalculateQFunctor<DeviceType> calculateQ_functor(this);
  Kokkos::parallel_for(nn,calculateQ_functor);

  pack_flag = 4;
  //comm->forward_comm_fix( this ); //Dist_vector( atom->q );
  atomKK->k_q.modify<DeviceType>();
  atomKK->k_q.sync<LMPHostType>();
  comm->forward_comm_fix(this);
  atomKK->k_q.modify<LMPHostType>();
  atomKK->k_q.sync<DeviceType>();

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::sparse_matvec_acks2_half(k_bb, k_xx)
{
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagSparseMatvec1>(0,nn),*this);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagSparseMatvec2>(nn,NN),*this);

  if (neighflag == HALF)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagSparseMatvec3_half<HALF> >(0,nn),*this);
  else if (neighflag == HALFTHREAD)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagSparseMatvec3_half<HALFTHREAD> >(0,nn),*this);

  if (need_dup)
    Kokkos::Experimental::contribute(d_bb, dup_bb);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::sparse_matvec_acks2_full(k_bb, k_xx)
{
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagSparseMatvec1>(0,nn),*this);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagSparseMatvec3_full>(0,nn),*this);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::operator() (TagSparseMatvec1, const int &ii) const
{
  const int i = d_ilist[ii];
  const int itype = type(i);
  if (mask[i] & groupbit) {
    d_bb[i] = params(itype).eta * d_xx[i];
    d_bb[NN + i] = d_X_diag[i] * d_xx[NN + i];
  }

  // last two rows
  if (ii == nn-1) {
    d_bb[2*NN] = 0.0;
    d_bb[2*NN + 1] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::operator() (TagSparseMatvec2, const int &ii) const
{
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    d_bb[i] = 0.0;
    d_bb[NN + i] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::operator() (TagSparseMatvec3_Half, const int &ii) const
{
  // The bb array is duplicated for OpenMP, atomic for CUDA, and neither for Serial
  auto v_bb = ScatterViewHelper<NeedDup<NEIGHFLAG,DeviceType>::value,decltype(dup_bb),decltype(ndup_bb)>::get(dup_bb,ndup_bb);
  auto a_bb = v_bb.template access<AtomicDup<NEIGHFLAG,DeviceType>::value>();

  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    F_FLOAT tmp = 0.0;

    // H Matrix
    for(int jj = d_firstnbr_H[i]; jj < d_firstnbr_H[i] + d_numnbrs_H[i]; jj++) {
      const int j = d_jlist_H(jj);
      tmp += d_val_H(jj) * d_xx[j];
      a_bb[j] += d_val_H(jj) * d_xx[i];
    }
    a_bb[i] += tmp;

    // X Matrix
    tmp = 0.0;
    for(int jj = d_firstnbr_X[i]; jj < d_firstnbr_X[i] + d_numnbrs_X[i]; jj++) {
      const int j = d_jlist_X(jj);
      tmp += d_val_X(jj) * d_xx[NN + j];
      a_bb[NN + j] += d_val_X(jj) * d_xx[NN + i];
    }
    a_bb[NN + i] += tmp;

    // Identity Matrix
    a_bb[NN + i] += d_xx[i];
    a_bb[i] += d_xx[NN + i];

    // Second-to-last row/column
    a_bb[2*NN] += d_xx[NN + i];
    a_bb[NN + i] += d_xx[2*NN];

    // Last row/column
    a_bb[2*NN + 1] += d_xx[i];
    a_bb[i] += d_xx[2*NN + 1];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::operator() (TagSparseMatvec3_Full, const membertype1 &team) const
{
  const int i = d_ilist[team.league_rank()];
  if (mask[i] & groupbit) {
    F_FLOAT doitmp;

    // H Matrix
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, d_firstnbr_H[i], d_firstnbr_H[i] + d_numnbrs_H[i]), [&] (const int &jj, F_FLOAT &doi) {
      const int j = d_jlist_H(jj);
      doi += d_val_H(jj) * d_xx[j];
    }, doitmp);
    Kokkos::single(Kokkos::PerTeam(team), [&] () {d_bb[i] += doitmp;});

    // X Matrix
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, d_firstnbr_X[i], d_firstnbr_X[i] + d_numnbrs_X[i]), [&] (const int &jj, F_FLOAT &doi) {
      const int j = d_jlist_X(jj);
      doi += d_val_X(jj) * d_xx[j];
    }, doitmp);

    Kokkos::single(Kokkos::PerTeam(team), [&] () {
      d_bb[i] += doitmp;

      // Identity Matrix
      d_bb[NN + i] += d_xx[i];
      d_bb[i] += d_xx[NN + i];

      // Second-to-last row/column
      d_bb[2*NN] += d_xx[NN + i];
      d_bb[NN + i] += d_xx[2*NN];

      // Last row/column
      d_bb[2*NN + 1] += d_xx[i];
      d_bb[i] += d_xx[2*NN + 1];
    });

  }
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::operator() (TagNorm1, const int &i) const
{
  F_FLOAT tmp = 0;
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    d_r[i] = 1.0*d_b_s[i] + -1.0*d_d[i];
    tmp = d_b_s[i] * d_b_s[i];
  }
  return tmp;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::operator() (TagNorm2, const int &i) const
{
  F_FLOAT tmp = 0;
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    tmp = d_r[i] * d_r[i];
  }
  return tmp;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::operator() (TagDot1, const int &i) const
{
  F_FLOAT tmp = 0.0;
  const int i = d_ilist[ii];
  if (mask[i] & groupbit)
    tmp = d_r_hat[i] * d_r[i];
  return tmp;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::operator() (TagPrecon1, const int &i) const
{
  F_FLOAT tmp = 0.0;
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    if (loop > 1) {
      d_q[i] = 1.0*d_p[i] - omega*d_z[i];
      d_p[i] = 1.0*d_r[i] + beta*d_q[i];
  }
  return tmp;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double FixACKS2ReaxKokkos<DeviceType>::dot2_item(int ii) const
{
  double tmp = 0.0;
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    tmp = d_d[i] * d_o[i];
  }
  return tmp;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::precon1_item(int ii) const
{
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    d_s[i] += alpha * d_d[i];
    d_r[i] += -alpha * d_o[i];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::precon2_item(int ii) const
{
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    d_t[i] += alpha * d_d[i];
    d_r[i] += -alpha * d_o[i];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double FixACKS2ReaxKokkos<DeviceType>::precon_item(int ii) const
{
  F_FLOAT tmp = 0.0;
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    d_p[i] = d_r[i] * d_Hdia_inv[i];
    tmp = d_r[i] * d_p[i];
  }
  return tmp;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::calculate_q_item1(int ii) const
{
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {

    for (int k = nprev-1; k > 0; --k) {
      d_s_hist(i,k) = d_s_hist(i,k-1);
      d_s_hist_X(i,k) = d_s_hist_X(i,k-1);
    }
    d_s_hist(i,0) = d_s[i];
    d_s_hist_X(i,0) = d_s[NN+i];
  }
  // last two rows
  if (comm->me == 0) {
    for (int i = 0; i < 2; ++i) {
      for (k = nprev-1; k > 0; --k)
        d_s_hist_last(i,k) = d_s_hist_last(i,k-1);
      d_s_hist_last(i,0) = d_s(2*NN+i);
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixACKS2ReaxKokkos<DeviceType>::calculate_q_item2(int ii) const
{
  const int i = d_ilist[ii];
  if (mask[i] & groupbit)
    q(i) = d_s(i);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::cleanup_copy()
{
  id = style = NULL;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

template<class DeviceType>
double FixACKS2ReaxKokkos<DeviceType>::memory_usage()
{
  double bytes;

  int size = 2*nmax + 2;

  bytes = size*nprev * sizeof(double); // s_hist
  bytes += nmax*4 * sizeof(double); // storage
  bytes += size*11 * sizeof(double); // storage
  bytes += n_cap*4 * sizeof(int); // matrix...
  bytes += m_cap*2 * sizeof(int);
  bytes += m_cap*2 * sizeof(double);

  return bytes;
}

/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::grow_arrays(int nmax)
{
  k_s_hist.template sync<LMPHostType>();
  k_s_hist_X.template sync<LMPHostType>();

  k_s_hist.template modify<LMPHostType>(); // force reallocation on host
  k_s_hist_X.template modify<LMPHostType>();

  memoryKK->grow_kokkos(k_s_hist,s_hist,nmax,nprev,"acks2:s_hist");
  memoryKK->grow_kokkos(k_s_hist_X,s_hist_X,nmax,nprev,"acks2:s_hist_X");

  d_s_hist = k_s_hist.template view<DeviceType>();
  d_s_hist_X = k_s_hist_X.template view<DeviceType>();

  k_s_hist.template modify<LMPHostType>();
  k_s_hist_X.template modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void FixACKS2ReaxKokkos<DeviceType>::copy_arrays(int i, int j, int delflag)
{
  k_s_hist.template sync<LMPHostType>();
  k_s_hist_X.template sync<LMPHostType>();

  FixACKS2Reax::copy_arrays(i,j,delflag);

  k_s_hist.template modify<LMPHostType>();
  k_s_hist_X.template modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

template<class DeviceType>
int FixACKS2ReaxKokkos<DeviceType>::pack_exchange(int i, double *buf)
{
  k_s_hist.template sync<LMPHostType>();
  k_s_hist_X.template sync<LMPHostType>();

  return FixACKS2Reax::pack_exchange(i,buf);
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

template<class DeviceType>
int FixACKS2ReaxKokkos<DeviceType>::unpack_exchange(int nlocal, double *buf)
{
  int n = FixACKS2Reax::unpack_exchange(nlocal,buf);

  k_s_hist.template modify<LMPHostType>();
  k_s_hist_X.template modify<LMPHostType>();

  return n;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixACKS2ReaxKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class FixACKS2ReaxKokkos<LMPHostType>;
#endif
}
