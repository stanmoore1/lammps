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
   Contributing author:  Stan Moore (SNL), Tomas Oppelstrup (LLNL)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "pair_wqlfun_kokkos.h"
#include "atom_kokkos.h"
#include "error.h"
#include "force.h"
#include "atom_masks.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"
#include "kokkos.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairWQLFunKokkos<DeviceType>::PairWQLFunKokkos(LAMMPS *lmp) : PairWQLFun(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairWQLFunKokkos<DeviceType>::~PairWQLFunKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairWQLFunKokkos<DeviceType>::init_style()
{
  PairWQLFun::init_style();

  // irequest = neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;

  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = std::is_same<DeviceType,LMPHostType>::value &&
    !std::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = std::is_same<DeviceType,LMPDeviceType>::value;

  if (neighflag == HALF || neighflag == HALFTHREAD) { // still need atomics, even though using a full neigh list
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
  } else {
    error->all(FLERR,"Must use half neighbor list style with pair wqlfun/kk");
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType>* nl): k_list(*nl) {}
  ~FindMaxNumNeighs() {k_list.copymode = 1;}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii, int& maxneigh) const {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (maxneigh < num_neighs) maxneigh = num_neighs;
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairWQLFunKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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

  copymode = 1;
  int newton_pair = force->newton_pair;
  if (newton_pair == false)
    error->all(FLERR,"PairWQLFunKokkos requires 'newton on'");

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  k_cutsq.template sync<DeviceType>();

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }

  maxneigh = 0;
  Kokkos::parallel_reduce("PairWQLFunKokkos::find_maxneigh",inum, FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(maxneigh));

  int vector_length_default = 1;
  int team_size_default = 1;
  if (!host_flag)
    team_size_default = 32;//maxneigh;

  chunk_size = MIN(chunksize,inum); // "chunksize" variable is set by user
  chunk_offset = 0;

  // Wl relevant variables and tables

  if (chunk_size > (int)d_ncount.extent(0)) {
    wl = t_sna_1d("wqlfun:wl",chunk_size);
    Qlm = t_sna_2c("wqlfun:Qlm",chunk_size,lmax+1);
    d_ncount = t_sna_1i("wqlfun:ncount",chunk_size);
    d_nne_scale = t_sna_1d("wqlfun:nne_scale",chunk_size);
  }

  // insure distsq and nearest arrays are long enough

  if (chunk_size > (int)d_distsq.extent(0) || maxneigh > (int)d_distsq.extent(1)) {
    plm = t_sna_3d("wqlfun:plm",chunk_size,maxneigh,(lmax+2)*(lmax+3)/2);
    expi = t_sna_3c("wqlfun:expi",chunk_size,maxneigh,lmax+2);
    ylm = t_sna_3c("wqlfun:ylm",chunk_size,maxneigh,(lmax+2)*(lmax+3)/2);
    zlm = t_sna_3c("wqlfun:zlm",chunk_size,maxneigh,(lmax+2)*(lmax+3)/2);
    gradQlm = t_sna_4c("wqlfun:gradQlm",chunk_size,maxneigh,lmax+1);
    gradql = t_sna_3d3("wqlfun:gradql",chunk_size,maxneigh);
    gradwl = t_sna_3d3("wqlfun:gradwl",chunk_size,maxneigh);
    d_rlist = t_sna_3d3("wqlfun:rlist",chunk_size,maxneigh);
    d_distsq = t_sna_2d("wqlfun:distsq",chunk_size,maxneigh);
    d_nearest = t_sna_2i("wqlfun:nearest",chunk_size,maxneigh);
  }

  EV_FLOAT ev;

  while (chunk_offset < inum) { // chunk up loop to prevent running out of memory

    EV_FLOAT ev_tmp;

    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    //Neigh
    {
      int vector_length = vector_length_default;
      int team_size = team_size_default;
      check_team_size_for<TagPairWQLFunNeigh>(chunk_size,team_size,vector_length);
      int scratch_size = scratch_size_helper<int>(team_size * maxneigh);
      typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunNeigh> policy_neigh(chunk_size,team_size,vector_length);
      policy_neigh = policy_neigh.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
      Kokkos::parallel_for("PairWQLFunNeigh",policy_neigh,*this);
    }

    //Energy
    {
      int vector_length = vector_length_default;
      int team_size = team_size_default;
      check_team_size_for<TagPairWQLFunEnergy>(((chunk_size+team_size-1)/team_size)*maxneigh,team_size,vector_length);
      typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunEnergy> policy_energy(((chunk_size+team_size-1)/team_size)*maxneigh,team_size,vector_length);
      Kokkos::parallel_for("PairWQLFunEnergy",policy_energy,*this);
    }

    //Energy2
    typename Kokkos::RangePolicy<DeviceType, TagPairWQLFunEnergy2> policy_energy2(0,chunk_size);
    Kokkos::parallel_for("PairWQLFunEnergy2",policy_energy2,*this);

    //Derivative
    {
      int vector_length = vector_length_default;
      int team_size = team_size_default;
      check_team_size_for<TagPairWQLFunDeriv>(((chunk_size+team_size-1)/team_size)*maxneigh,team_size,vector_length);
      typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunDeriv> policy_deriv(((chunk_size+team_size-1)/team_size)*maxneigh,team_size,vector_length);
      Kokkos::parallel_for("PairWQLFunDeriv",policy_deriv,*this);
    }

    //Derivative2
    {
      int vector_length = vector_length_default;
      int team_size = team_size_default;
      check_team_size_for<TagPairWQLFunDeriv2>(((chunk_size+team_size-1)/team_size)*maxneigh,team_size,vector_length);
      typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunDeriv2> policy_deriv2(((chunk_size+team_size-1)/team_size)*maxneigh,team_size,vector_length);
      Kokkos::parallel_for("PairWQLFunDeriv2",policy_deriv2,*this);
    }

    //ComputeForce
    {
      if (evflag) {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairWQLFunForce<HALF,1> > policy_force(0,chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairWQLFunForce<HALFTHREAD,1> > policy_force(0,chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        }
      } else {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairWQLFunForce<HALF,0> > policy_force(0,chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairWQLFunForce<HALFTHREAD,0> > policy_force(0,chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        }
      }
    }
    ev += ev_tmp;
    chunk_offset += chunk_size;

  } // end while

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
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  atomKK->modified(execution_space,F_MASK);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f     = decltype(dup_f)();
    dup_vatom = decltype(dup_vatom)();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::operator() (TagPairWQLFunNeigh,const typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunNeigh>::member_type& team) const
{
  const int ii = team.league_rank();
  const int i = d_ilist[ii + chunk_offset];
  const int itype = type[i];
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);
  const int jnum = d_numneigh[i];

  // get a pointer to scratch memory
  // This is used to cache whether or not an atom is within the cutoff
  // If it is, inside is assigned to 1, otherwise -1
  const int team_rank = team.team_rank();
  const int scratch_shift = team_rank * maxneigh; // offset into pointer for entire team
  int* inside = (int*)team.team_shmem().get_shmem(team.team_size() * maxneigh * sizeof(int), 0) + scratch_shift;

  // loop over list of all neighbors within force cutoff
  // distsq[] = distance sq to each
  // rlist[] = distance vector to each
  // nearest[] = atom indices of neighbors

  int ncount = 0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,jnum),
      [&] (const int jj, int& count) {
    int j = d_neighbors(i,jj);
    j &= NEIGHMASK;
    const int jtype = type(j);
    const F_FLOAT delx = xtmp - x(j,0);
    const F_FLOAT dely = ytmp - x(j,1);
    const F_FLOAT delz = ztmp - x(j,2);
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    inside[jj] = -1;
    if (rsq < d_cutsq(itype,jtype)) {
     inside[jj] = 1;
     count++;
    }
  },ncount);

  d_ncount(ii) = ncount;

  for (int m = 0; m<=lmax; m++)
    Qlm(ii,m) = complex::zero();

  // nne_scale = how many effective neighbors, used for scaling ql
  //  Set to 1 if zero neighbors within cutoff,
  //  otherwise to: sum fsmooth(r_ij), over j

  d_nne_scale(ii) = 0.0;

  Kokkos::parallel_scan(Kokkos::TeamThreadRange(team,jnum),
      [&] (const int jj, int& offset, bool final) {

    if (inside[jj] < 0) return;

    if (final) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      const F_FLOAT delx = xtmp - x(j,0);
      const F_FLOAT dely = ytmp - x(j,1);
      const F_FLOAT delz = ztmp - x(j,2);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      d_distsq(ii,offset) = rsq;
      d_rlist(ii,offset,0) = delx;
      d_rlist(ii,offset,1) = dely;
      d_rlist(ii,offset,2) = delz;
      d_nearest(ii,offset) = j;
    }
    offset++;
  });
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::operator() (TagPairWQLFunEnergy,const typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunEnergy>::member_type& team) const
{
  // Extract the atom number
  int ii = team.team_rank() + team.team_size() * (team.league_rank() %
           ((chunk_size+team.team_size()-1)/team.team_size()));
  if (ii >= chunk_size) return;

  // Extract the neighbor number
  const int jj = team.league_rank() / ((chunk_size+team.team_size()-1)/team.team_size());
  const int ncount = d_ncount(ii);
  if (jj >= ncount) return;

  const double delx = d_rlist(ii,jj,0);
  const double dely = d_rlist(ii,jj,1);
  const double delz = d_rlist(ii,jj,2);
  const double rsq = d_distsq(ii,jj);

  double df;
  const double rinv = 1.0/sqrt(rsq);
  const double r = rsq*rinv;
  const double xn = delx*rinv;
  const double yn = dely*rinv;
  const double zn = delz*rinv;
  const double fij = fsmooth(r,df);
  Kokkos::atomic_add(&(d_nne_scale(ii)), fij);
  const int lmax1 = lmax + 1; // Need higher degree to enable differentiation
  ylmallcompress(ii,jj,lmax1,xn,yn,zn);
  const int idx = lmax*(lmax+1)/2;
  for (int m = 0; m<=lmax; m++) {
    Kokkos::atomic_add(&(Qlm(ii,m).re), fij*ylm(ii,jj,idx+m).re);
    Kokkos::atomic_add(&(Qlm(ii,m).im), fij*ylm(ii,jj,idx+m).im);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::operator() (TagPairWQLFunEnergy2,const int& ii) const
{
  const int i = d_ilist[ii + chunk_offset];

  if (d_nne_scale(ii) <= 0.0) d_nne_scale(ii) = 1.0;

  // Wl ('energy') section

  // Compute Ql for this atom
  double ql = Qlm(ii,0).re*Qlm(ii,0).re + Qlm(ii,0).im*Qlm(ii,0).im;
  for (int m = 1; m<=lmax; m++)
    ql += 2*(Qlm(ii,m).re*Qlm(ii,m).re +
	      Qlm(ii,m).im*Qlm(ii,m).im);

  double wlscale = 0.0;
  if (ql > 0)
    wlscale = 1 / sqrt(ql*ql*ql);

  // Compute Wl
  int widx = 0;
  wl(ii) = 0.0;
  for (int m1 = -lmax; m1<=0; m1++) {
    for (int m2 = 0; m2<=((-m1)>>1); m2++) {
      const int m3 = -(m1 + m2);
      // Loop enforces -lmax<=m1<=0<=m2<=m3<=lmax, and m1+m2+m3=0

      // For even lmax, W3j is invariant under permutation of
      // (m1,m2,m3) and (m1,m2,m3)->(-m1,-m2,-m3). The loop
      // structure enforces visiting only one member of each
      // such symmetry (invariance) group.

      const int sgn = 1 - 2*(m1&1);
      complex Q1Q2[3],Q1Q2Q3;
      // m1 <= 0, and Qlm[-m] = (-1)^m*conjg(Qlm[m]). sgn = (-1)^m.
      Q1Q2[0].re = (Qlm(ii,-m1).re*Qlm(ii,m2).re + Qlm(ii,-m1).im*Qlm(ii,m2).im)*sgn;
      Q1Q2[0].im = (Qlm(ii,-m1).re*Qlm(ii,m2).im - Qlm(ii,-m1).im*Qlm(ii,m2).re)*sgn;
      Q1Q2Q3.re = Q1Q2[0].re*Qlm(ii,m3).re - Q1Q2[0].im*Qlm(ii,m3).im;

      const double c = d_w3jlist[widx++];
      wl(ii) += Q1Q2Q3.re*c;
    }
  }
  wl(ii) *= wlscale; // wl is really wlhat
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::operator() (TagPairWQLFunDeriv,const typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunDeriv>::member_type& team) const
{
  // Extract the atom number
  int ii = team.team_rank() + team.team_size() * (team.league_rank() %
           ((chunk_size+team.team_size()-1)/team.team_size()));
  if (ii >= chunk_size) return;

  // Extract the neighbor number
  const int jj = team.league_rank() / ((chunk_size+team.team_size()-1)/team.team_size());
  const int ncount = d_ncount(ii);
  if (jj >= ncount) return;

  const double delx = d_rlist(ii,jj,0);
  const double dely = d_rlist(ii,jj,1);
  const double delz = d_rlist(ii,jj,2);
  const double rsq = d_distsq(ii,jj);

  double df;
  const double rinv = 1.0/sqrt(rsq);
  const double r = rsq*rinv;
  const double xn = delx*rinv;
  const double yn = dely*rinv;
  const double zn = delz*rinv;
  const double fij = fsmooth(r,df);

  const int lmax1 = lmax + 1; // Need higher degree to enable differentiation
  ylm2zlmcompress(ii,jj,lmax1); // Renormalization to help with differentiation

  for (int k = 0; k<3; k++)
    gradql(ii,jj,k) = 0.0;

  for (int m = 0; m<=lmax; m++) {
    const double Alm = Almvec[m];

    // Derivatives of Qlm w.r.t (dx,dy,dz) = r(i) - x(j)
    complex yval,dy[3];

    yval.re = zlm(ii,jj,lmax*(lmax+1)/2 + m).re;
    yval.im = zlm(ii,jj,lmax*(lmax+1)/2 + m).im;

    zlmderiv1compress(ii,jj,lmax,m,xn,yn,zn,dy); // Differentiate re-normalized Ylm (i.e. Zlm)
    const double dQlmdx0 = Alm*(df*xn*yval.re + fij*dy[0].re*rinv); // Alm re-re-normalizes
    const double dQlmdx1 = Alm*(df*xn*yval.im + fij*dy[0].im*rinv); // Zlm to Ylm again...

    const double dQlmdy0 = Alm*(df*yn*yval.re + fij*dy[1].re*rinv);
    const double dQlmdy1 = Alm*(df*yn*yval.im + fij*dy[1].im*rinv);

    const double dQlmdz0 = Alm*(df*zn*yval.re + fij*dy[2].re*rinv);
    const double dQlmdz1 = Alm*(df*zn*yval.im + fij*dy[2].im*rinv);
    gradQlm(ii,jj,m,0).re = dQlmdx0;  // Re(d Qlm/dx)
    gradQlm(ii,jj,m,0).im = dQlmdx1;  // Im(d Qlm/dx)
    gradQlm(ii,jj,m,1).re = dQlmdy0;  // Re(d Qlm/dy)
    gradQlm(ii,jj,m,1).im = dQlmdy1;  // Im(d Qlm/dy)
    gradQlm(ii,jj,m,2).re = dQlmdz0;  // Re(d Qlm/dz)
    gradQlm(ii,jj,m,2).im = dQlmdz1;  // Im(d Qlm/dz)
  }

  // Compute derivatives of wlscale (one over WQLFun expression denominator) w.r.t atom jj
  for (int k = 0; k<3; k++)
    gradql(ii,jj,k) += 2*(Qlm(ii,0).re*gradQlm(ii,jj,0,k).re +
		     Qlm(ii,0).im*gradQlm(ii,jj,0,k).im);
  for (int m = 1; m<=lmax; m++)
    for (int k = 0; k<3; k++)
      gradql(ii,jj,k) += 4*(Qlm(ii,m).re*gradQlm(ii,jj,m,k).re +
		       Qlm(ii,m).im*gradQlm(ii,jj,m,k).im);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::operator() (TagPairWQLFunDeriv2,const typename Kokkos::TeamPolicy<DeviceType, TagPairWQLFunDeriv2>::member_type& team) const
{
  // Extract the atom number
  int ii = team.team_rank() + team.team_size() * (team.league_rank() %
           ((chunk_size+team.team_size()-1)/team.team_size()));
  if (ii >= chunk_size) return;

  // Extract the neighbor number
  const int jj = team.league_rank() / ((chunk_size+team.team_size()-1)/team.team_size());
  const int ncount = d_ncount(ii);
  if (jj >= ncount) return;

  // Forces
  // Need to accumulate everything in inner-most loop so we can
  //  calculate contribution to forces on 'j' particles...

  // Compute Ql for this atom
  double ql = Qlm(ii,0).re*Qlm(ii,0).re + Qlm(ii,0).im*Qlm(ii,0).im;
  for (int m = 1; m<=lmax; m++)
    ql += 2*(Qlm(ii,m).re*Qlm(ii,m).re +
              Qlm(ii,m).im*Qlm(ii,m).im);

  double wlscale = 0.0;
  if (ql > 0)
    wlscale = 1 / sqrt(ql*ql*ql);

  double gradwlscale[3] = {0,0,0};
  if (wlscale > 0)
    for (int k = 0; k<3; k++)
      gradwlscale[k] = (-3.0/2)*wlscale/ql * gradql(ii,jj,k);
  else
    for (int k = 0; k<3; k++)
      gradwlscale[k] = 0;

  for (int k = 0; k<3; k++)
    gradwl(ii,jj,k) = 0.0;

  int widx = 0;
  for (int m1 = -lmax; m1<=0; m1++)
    for (int m2 = 0; m2<=((-m1)>>1); m2++) {
      const int m3 = -(m1 + m2);
      // Loop enforces -lmax<=m1<=0<=m2<=m3<=lmax, and m1+m2+m3=0

      // For even lmax, W3j is invariant under permutation of
      // (m1,m2,m3) and (m1,m2,m3)->(-m1,-m2,-m3). The loop
      // structure enforces visiting only one member of each
      // such symmetry (invariance) group.

      const int sgn = 1 - 2*(m1&1);
      complex Q1Q2[3],Q1Q2Q3;
      Q1Q2[0].re =  (Qlm(ii,-m1).re*Qlm(ii, m2).re + Qlm(ii,-m1).im*Qlm(ii, m2).im)*sgn; // m1<0 -> conjugation
      Q1Q2[0].im =  (Qlm(ii,-m1).re*Qlm(ii, m2).im - Qlm(ii,-m1).im*Qlm(ii, m2).re)*sgn;
      Q1Q2[1].re =   Qlm(ii, m2).re*Qlm(ii, m3).re - Qlm(ii, m2).im*Qlm(ii, m3).im;
      Q1Q2[1].im =   Qlm(ii, m2).re*Qlm(ii, m3).im + Qlm(ii, m2).im*Qlm(ii, m3).re;
      Q1Q2[2].re =  (Qlm(ii, m3).re*Qlm(ii,-m1).re + Qlm(ii, m3).im*Qlm(ii,-m1).im)*sgn;
      Q1Q2[2].im = (-Qlm(ii, m3).re*Qlm(ii,-m1).im + Qlm(ii, m3).im*Qlm(ii,-m1).re)*sgn;
      Q1Q2Q3.re = Q1Q2[0].re*Qlm(ii,m3).re - Q1Q2[0].im*Qlm(ii,m3).im;

      const double c = d_w3jlist[widx++];
      for (int k = 0; k<3; k++) {
	// Since wl is real by summation symmetry, (d/dx) wl
	// and thus grad wl must be also.
	gradwl(ii,jj,k) +=
	  ((Q1Q2[0].re*gradQlm(ii,jj,m3,k).re - Q1Q2[0].im*gradQlm(ii,jj,m3,k).im +
	      (Q1Q2[1].re*gradQlm(ii,jj,-m1,k).re + Q1Q2[1].im*gradQlm(ii,jj,-m1,k).im)*sgn +
	       Q1Q2[2].re*gradQlm(ii,jj,m2,k).re - Q1Q2[2].im*gradQlm(ii,jj,m2,k).im) * wlscale +
	    Q1Q2Q3.re * gradwlscale[k]) * c;
      }
    } // closing m1,m2 loops.
}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::operator() (TagPairWQLFunForce<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  // The f array is duplicated for OpenMP, atomic for CUDA, and neither for Serial
  auto v_f = ScatterViewHelper<typename NeedDup<NEIGHFLAG,DeviceType>::value,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

  const int i = d_ilist[ii + chunk_offset];

  const int ncount = d_ncount(ii);

  F_FLOAT fitmp[3] = {0.0,0.0,0.0};
  for (int jj = 0; jj < ncount; jj++) {
    int j = d_nearest(ii,jj);

    F_FLOAT fij[3];
    for (int k = 0; k<3; k++) { //// unroll??
      const double du = eval_tree_deriv_hc(wl(ii));
      const double dudx = du * gradwl(ii,jj,k);

      fij[k] = -dudx;

      fitmp[k] += fij[k];
      a_f(j,k) -= fij[k];
    }

    // tally global and per-atom virial contribution
    if (EVFLAG) {
      if (vflag_either) {
        v_tally_xyz<NEIGHFLAG>(ev,i,j,
          fij[0],fij[1],fij[2],
          d_rlist(ii,jj,0),d_rlist(ii,jj,1),
          d_rlist(ii,jj,2));
      }
    }
  }
  a_f(i,0) += fitmp[0];
  a_f(i,1) += fitmp[1];
  a_f(i,2) += fitmp[2];

  // tally energy contribution

  const double u = eval_tree_hc(wl(ii));

  if (EVFLAG) {
    if (eflag_either) {

      if (eflag_global) ev.evdwl += u;
      if (eflag_atom) d_eatom[i] += u;
    }
  }

}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::operator() (TagPairWQLFunForce<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairWQLFunForce<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairWQLFunKokkos<DeviceType>::allocate()
{
  PairWQLFun::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairWQLFunKokkos<DeviceType>::settings(int narg, char **arg)
{
  PairWQLFun::settings(narg,arg);

  rsqrt = t_sna_1d("wqlfun:rsqrt",2*lmax+3);
  sqrtfact = t_sna_1d("wqlfun:sqrtfac",2*(lmax+1)+3);
  Almvec = t_sna_1d("wqlfun:Almvec",lmax+1);

  auto h_rsqrt = Kokkos::create_mirror_view(rsqrt);
  auto h_sqrtfact = Kokkos::create_mirror_view(sqrtfact);
  auto h_Almvec = Kokkos::create_mirror_view(Almvec);

  for (int m=0; m<=lmax; m++)
    h_Almvec[m] = Anm(lmax,m);

  h_rsqrt[0] = 1.0;
  for (int m=1; m<=lmax+1; m++) {
    h_rsqrt[2*m-1] = sqrt(1.0/(2*m-1));
    h_rsqrt[2*m]   = sqrt(1.0/(2*m));
  }

  h_sqrtfact[0] = 1.0;
  for (int m=0; m<=lmax+1; m++) {
    h_sqrtfact[2*m+1] = h_sqrtfact[2*m]   * sqrt((double) (2*m+1));
    h_sqrtfact[2*m+2] = h_sqrtfact[2*m+1] * sqrt((double) (2*m+2));
  }

  Kokkos::deep_copy(rsqrt,h_rsqrt);
  Kokkos::deep_copy(sqrtfact,h_sqrtfact);
  Kokkos::deep_copy(Almvec,h_Almvec);

  d_w3jlist = t_sna_1d("wqlfun:w3jlist",w3jlist.size());

  // copy w3jlist values

  auto h_w3jlist = Kokkos::create_mirror_view(d_w3jlist);

  for (int i = 0; i < w3jlist.size(); i++)
    h_w3jlist(i) = w3jlist[i];

  Kokkos::deep_copy(d_w3jlist,h_w3jlist);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairWQLFunKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairWQLFun::init_one(i,j);
  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();

  return cutone;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::plmallcompress(const int ii, const int jj, const int lmax, const double x) const {

  int idx1,idx2,idx3;

  // Numerically stable computation (instead of sqrt(1-x*x)), when x is close to 1.
  const double sine = sqrt((1+x)*(1-x));
  plm(ii,jj,0) = 1;
  for (int m=0; m<=lmax; m++) {
    // P(m,m)
    idx1 = m*(m+1)/2 + m;
    if (m > 0) plm(ii,jj,idx1) = -plm(ii,jj,(m-1)*m/2+(m-1))*(2*m-1)*sine;

    // P(m+1,m)
    idx2 = (m+1)*(m+2)/2 + m;
    if (m < lmax) plm(ii,jj,idx2) = x*(2*m+1)*plm(ii,jj,idx1);

    for (int l=m+2; l<=lmax; l++) {
      // P(l,m)
      idx3 = l*(l+1)/2 + m;
      plm(ii,jj,idx3) = (x*(2*l-1)*plm(ii,jj,idx2) - (l+m-1)*plm(ii,jj,idx1))/(l-m);

      idx1 = idx2;
      idx2 = idx3;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::ylmallcompress(const int ii, const int jj, const int lmax,
                                   const double xhat,const double yhat,const double zhat) const {
  double c,s;

  const double rxy2 = xhat*xhat + yhat*yhat;
  if (rxy2 > 0.0) {
    const double rxyinv = 1.0/sqrt(rxy2);
    c = xhat*rxyinv;
    s = yhat*rxyinv;
  } else {
    c = 1.0;
    s = 0.0;
  }
  plmallcompress(ii,jj,lmax,zhat);

  ylm(ii,jj,0).re = 1.0;
  ylm(ii,jj,0).im = 0.0;

  expi(ii,jj,0).re = 1.0;
  expi(ii,jj,0).im = 0.0;

  for (int l=1; l<=lmax; l++) {
    //expi[l] = expi[l-1] * (c + im*s)
    expi(ii,jj,l).re = expi(ii,jj,l-1).re*c - expi(ii,jj,l-1).im*s;
    expi(ii,jj,l).im = expi(ii,jj,l-1).im*c + expi(ii,jj,l-1).re*s;

    double f = 1.0;
    const int idx = l*(l+1)/2;
    ylm(ii,jj,idx).re = f * plm(ii,jj,idx);
    ylm(ii,jj,idx).im = 0.0;

    for (int m=1; m<=l; m++) {
      f = f * rsqrt[l-m+1]*rsqrt[l+m];
      ylm(ii,jj,idx + m).re = f * plm(ii,jj,idx + m) * expi(ii,jj,m).re;
      ylm(ii,jj,idx + m).im = f * plm(ii,jj,idx + m) * expi(ii,jj,m).im;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::ylm2zlmcompress(const int ii, const int jj, const int lmax) const {
  int k = 0;
  double lsign = 1;
  for (int l=0; l<=lmax; l++) {
    double sign = lsign;
    lsign = -lsign;
    for (int m=0; m<=l; m++) {
      //const double ainv = 1.0/Anm(l,m);
      const double ainv = sign*sqrtfact[l-m]*sqrtfact[l+m];
      zlm(ii,jj,k).re = ylm(ii,jj,k).re*ainv;
      zlm(ii,jj,k).im = ylm(ii,jj,k).im*ainv;
      k++;
      sign = -sign;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::zlmderiv1compress(int ii, int jj, const int lmax, const int m,
                                   const double xn, const double yn, const double zn, complex *DY) const {
  const double half = 0.5;
#define Yr(lmax,m) zlm(ii,jj,(lmax)*((lmax)+1)/2 + (m)).re
#define Yi(lmax,m) zlm(ii,jj,(lmax)*((lmax)+1)/2 + (m)).im

  if (m == 0) {
    // DY(1) = (lmax+1)*xn*Y(lmax,m) + realpart(Y(lmax+1,1))
    DY[0].re = (lmax+1)*xn*Yr(lmax,m) + Yr(lmax+1,1);
    DY[0].im = (lmax+1)*xn*Yi(lmax,m);

    // DY(2) = (lmax+1)*yn*Y(l,M) + imagpart(Y(lmax+1,1))
    DY[1].re = (lmax+1)*yn*Yr(lmax,m) + Yi(lmax+1,1);
    DY[1].im = (lmax+1)*yn*Yi(lmax,m);
  } else {
    DY[0].re = (lmax+1)*xn*Yr(lmax,m) +
      (Yr(lmax+1,m+1) - Yr(lmax+1,m-1))*half;
    DY[0].im = (lmax+1)*xn*Yi(lmax,m) +
      (Yi(lmax+1,m+1) - Yi(lmax+1,m-1))*half;
    DY[1].re = (lmax+1)*yn*Yr(lmax,m) +
      (Yi(lmax+1,m+1) + Yi(lmax+1,m-1))*half;
    DY[1].im = (lmax+1)*yn*Yi(lmax,m) -
      (Yr(lmax+1,m+1) + Yr(lmax+1,m-1))*half;
  }
  // DY(3) = (lmax+1)*zn*Y(lmax,m) + Y(lmax+1,m)
  DY[2].re = (lmax+1)*zn*Yr(lmax,m) + Yr(lmax+1,m);
  DY[2].im = (lmax+1)*zn*Yi(lmax,m) + Yi(lmax+1,m);
#undef Yr
#undef Yi
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairWQLFunKokkos<DeviceType>::eval_tree_hc(double wl) const {
  // attractive form
  //return kappa*0.5*(wl - wl0)*(wl - wl0);

  // repulsive form
  return kappa/(renorm + (wl - wl0)*(wl - wl0));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double PairWQLFunKokkos<DeviceType>::eval_tree_deriv_hc(double wl) const {
  // attractive form
  //return kappa*(wl - wl0);

  // repulsive form
  const double denom = renorm + (wl - wl0)*(wl - wl0);
  return -2.0*kappa*(wl - wl0)/(denom*denom);
}

/* ----------------------------------------------------------------------
   check max team size for parallel_for
------------------------------------------------------------------------- */

template<class DeviceType>
template<class TagStyle>
void PairWQLFunKokkos<DeviceType>::check_team_size_for(int inum, int &team_size, int vector_length) {
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelForTag());

  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
}

/* ----------------------------------------------------------------------
   check max team size for parallel_reduce
------------------------------------------------------------------------- */

template<class DeviceType>
template<class TagStyle>
void PairWQLFunKokkos<DeviceType>::check_team_size_reduce(int inum, int &team_size, int vector_length) {
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelReduceTag());

  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<typename scratch_type>
int PairWQLFunKokkos<DeviceType>::scratch_size_helper(int values_per_team) {
  typedef Kokkos::View<scratch_type*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewType;

  return ScratchViewType::shmem_size(values_per_team);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairWQLFunKokkos<DeviceType>::v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
{
  // The vatom array is duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto v_vatom = ScatterViewHelper<typename NeedDup<NEIGHFLAG,DeviceType>::value,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

  const E_FLOAT v0 = delx*fx;
  const E_FLOAT v1 = dely*fy;
  const E_FLOAT v2 = delz*fz;
  const E_FLOAT v3 = delx*fy;
  const E_FLOAT v4 = delx*fz;
  const E_FLOAT v5 = dely*fz;

  if (vflag_global) {
    ev.v[0] += v0;
    ev.v[1] += v1;
    ev.v[2] += v2;
    ev.v[3] += v3;
    ev.v[4] += v4;
    ev.v[5] += v5;
  }

  if (vflag_atom) {
    a_vatom(i,0) += 0.5*v0;
    a_vatom(i,1) += 0.5*v1;
    a_vatom(i,2) += 0.5*v2;
    a_vatom(i,3) += 0.5*v3;
    a_vatom(i,4) += 0.5*v4;
    a_vatom(i,5) += 0.5*v5;
    a_vatom(j,0) += 0.5*v0;
    a_vatom(j,1) += 0.5*v1;
    a_vatom(j,2) += 0.5*v2;
    a_vatom(j,3) += 0.5*v3;
    a_vatom(j,4) += 0.5*v4;
    a_vatom(j,5) += 0.5*v5;
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairWQLFunKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairWQLFunKokkos<LMPHostType>;
#endif
}
