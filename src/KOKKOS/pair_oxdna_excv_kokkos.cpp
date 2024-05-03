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
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
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

  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK); //need or not need? same for fene

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
  firstneigh = list->firstneigh;

  copymode = 1;

  // loop over neighbors of my atoms

  EV_FLOAT ev;

  // deal with all the compute operators() and ev_tally_xyz
  //EV_FLOAT ev = pair_compute<PairOxdnaExcvKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

template<class DeviceType>
template<int OXDNAFLAG, int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::operator()(TagPairOxdnaExcvCompute<OXDNAFLAG,NEWTON_BOND,EVFLAG>, \
  const int &in, EV_FLOAT &ev) const
{
  
}

template<class DeviceType>
template<int OXDNAFLAG, int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairOxdnaExcvKokkos<DeviceType>::operator()(TagPairOxdnaExcvCompute<OXDNAFLAG,NEWTON_BOND,EVFLAG>, \
  const int &in) const
{
  EV_FLOAT ev;
  this->template operator()<OXDNAFLAG,NEWTON_BOND,EVFLAG>(TagPairOxdnaExcvCompute<OXDNAFLAG,NEWTON_BOND,EVFLAG>(),in,ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairOxdnaExcvKokkos<DeviceType>::allocate()
{
  PairOxdnaExcv::allocate();

  int n = atom->ntypes;

  /*memory->destroy(setflag);
  memory->destroy(cutsq);
  
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
  memory->destroy(nz);*/

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


namespace LAMMPS_NS {
template class PairOxdnaExcvKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairOxdnaExcvKokkos<LMPHostType>;
#endif
}