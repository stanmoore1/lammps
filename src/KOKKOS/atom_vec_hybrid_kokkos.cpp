// clang-format off
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

#include "atom_vec_hybrid_kokkos.h"

#include "atom_kokkos.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecHybridKokkos::AtomVecHybridKokkos(LAMMPS *lmp) : AtomVec(lmp),
AtomVecKokkos(lmp), AtomVecHybrid(lmp)
{
  no_comm_vel_flag = 1;
  no_border_vel_flag = 1;
}

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::process_args(int narg, char **arg)
{
  AtomVecHybrid::process_args(narg,arg);

  nstyles_cast = new AtomVecKokkos*[nstyles];
  for (int k = 0; k < nstyles; k++) {
    nstyles_cast[k] = dynamic_cast<AtomVecKokkos*>(styles[k]);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::grow(int n)
{
  for (int k = 0; k < nstyles; k++) nstyles_cast[k]->grow(n);
  nmax = atomKK->k_x.h_view.extent(0);
}

/* ----------------------------------------------------------------------
   sort atom arrays on device
------------------------------------------------------------------------- */

void AtomVecHybridKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  for (int k = 0; k < nstyles; k++) nstyles_cast[k]->sort_kokkos(Sorter);
}

/* ---------------------------------------------------------------------- */

int AtomVecHybridKokkos::pack_comm_kokkos(const int &n, const DAT::tdual_int_1d &k_sendlist,
                                          const DAT::tdual_double_2d_lr &buf,
                                          const int &pbc_flag, const int pbc[])
{
  // TODO: figure out how to sum parameters of all styles?
  int ntot = 0; // sum of "n*size_forward" from all styles of pack_comm_kokkos
  for (int k = 0; k < nstyles; k++) {
    ntot += nstyles_cast[k]->pack_comm_kokkos(n,k_sendlist,buf,pbc_flag,pbc);
  }
  return ntot;
}

void AtomVecHybridKokkos::unpack_comm_kokkos(const int &n, const int &nfirst,
                                             const DAT::tdual_double_2d_lr &buf)
{
  for (int k = 0; k < nstyles; k++) {
    nstyles_cast[k]->unpack_comm_kokkos(n,nfirst,buf);
  }
}

int AtomVecHybridKokkos::pack_comm_self(const int &n, const DAT::tdual_int_1d &list,
                                        const int nfirst,
                                        const int &pbc_flag, const int pbc[])
{
  int ntot = 0; // sum of "n*size_forward" from all styles of pack_comm_self
  for (int k = 0; k < nstyles; k++) {
    ntot += nstyles_cast[k]->pack_comm_self(n,list,nfirst,pbc_flag,pbc);
  }
  return ntot;
}

int AtomVecHybridKokkos::pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                            DAT::tdual_double_2d_lr buf,
                                            int pbc_flag, int * pbc, ExecutionSpace space)
{
  int ntot = 0; // sum of "n*size_border" from all styles of pack_border_kokkos
  for (int k = 0; k < nstyles; k++) {
    ntot += nstyles_cast[k]->pack_border_kokkos(n, k_sendlist, buf, pbc_flag, pbc, space);
  }
  return ntot;
}

void AtomVecHybridKokkos::unpack_border_kokkos(const int &n, const int &nfirst,
                                               const DAT::tdual_double_2d_lr &buf,
                                               ExecutionSpace space)
{
  for (int k = 0; k < nstyles; k++) {
    nstyles_cast[k]->unpack_border_kokkos(n,nfirst,buf,space);
  }
}

int AtomVecHybridKokkos::pack_exchange_kokkos(const int &nsend,DAT::tdual_double_2d_lr &buf,
                                              DAT::tdual_int_1d k_sendlist,
                                              DAT::tdual_int_1d k_copylist,
                                              DAT::tdual_int_1d k_sendlist_exchange,
                                              DAT::tdual_int_1d k_copylist_exchange,
                                              ExecutionSpace space)
{
  int ntot = 0; // sum of "nsend*size_exchange" from all styles of pack_exchange_kokkos
  for (int k = 0; k < nstyles; k++) {
    ntot += nstyles_cast[k]->pack_exchange_kokkos(nsend,buf,k_sendlist,k_copylist,k_sendlist_exchange,k_copylist_exchange,space);
  }
  return ntot;
}

int AtomVecHybridKokkos::unpack_exchange_kokkos(DAT::tdual_double_2d_lr & k_buf, int nrecv,
                                                int nlocal, int dim, double lo,
                                                double hi, ExecutionSpace space,
                                                DAT::tdual_int_1d &k_indices) 
{
  int ntot = 0; // sums to new atom->nlocal after all styles of unpack_exchange_kokkos
  for (int k = 0; k < nstyles; k++) {
    nstyles_cast[k]->unpack_exchange_kokkos(k_buf,nrecv,nlocal,dim,lo,hi,space,k_indices);
  }
  return ntot;
}

// TODO: move dynamic_cast into init

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::sync(ExecutionSpace space, unsigned int h_mask)
{
  for (int k = 0; k < nstyles; k++) nstyles_cast[k]->sync(space,h_mask);
}

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::sync_overlapping_device(ExecutionSpace space, unsigned int h_mask)
{
  for (int k = 0; k < nstyles; k++) nstyles_cast[k]->sync_overlapping_device(space,h_mask);
}

/* ---------------------------------------------------------------------- */

void AtomVecHybridKokkos::modified(ExecutionSpace space, unsigned int h_mask)
{
  for (int k = 0; k < nstyles; k++) nstyles_cast[k]->modified(space,h_mask);
}
