// clang-format off
/* -e--------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "comm_tiled_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "atom_vec_kokkos.h"

using namespace LAMMPS_NS;

#define BUFFACTOR 1.5
#define BUFFACTOR 1.5
#define BUFMIN 1024
#define EPSILON 1.0e-6

#define DELTA_PROCS 16

/* ---------------------------------------------------------------------- */

CommTiledKokkos::CommTiledKokkos(LAMMPS *_lmp) : CommTiled(_lmp) {}

/* ---------------------------------------------------------------------- */
//IMPORTANT: we *MUST* pass "*oldcomm" to the Comm initializer here, as
//           the code below *requires* that the (implicit) copy constructor
//           for Comm is run and thus creating a shallow copy of "oldcomm".
//           The call to Comm::copy_arrays() then converts the shallow copy
//           into a deep copy of the class with the new layout.

CommTiledKokkos::CommTiledKokkos(LAMMPS *_lmp, Comm *oldcomm) : CommTiled(_lmp,oldcomm) {}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   forward communication of atom coords every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

void CommTiledKokkos::forward_comm(int dummy)
{
  if (!forward_comm_classic) {
    if (forward_comm_on_host) forward_comm_device<LMPHostType>(dummy);
    else forward_comm_device<LMPDeviceType>(dummy);
    return;
  }

  k_sendlist.sync<LMPHostType>();

  if (comm_x_only) {
    atomKK->sync(Host,X_MASK);
    atomKK->modified(Host,X_MASK);
  } else if (ghost_velocity) {
    atomKK->sync(Host,X_MASK | V_MASK);
    atomKK->modified(Host,X_MASK | V_MASK);
  } else {
    atomKK->sync(Host,ALL_MASK);
    atomKK->modified(Host,ALL_MASK);
  }

  CommTiled::forward_comm(dummy);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommTiledKokkos::forward_comm_device(int)
{
  int i,irecv,n,nsend,nrecv;
  AtomVecKokkos *avec = (AtomVecKokkos *) atom->avec;
  double *buf;

  // exchange data with another set of procs in each swap
  // post recvs from all procs except self
  // send data to all procs except self
  // copy data to self if sendself is set
  // wait on all procs except self and unpack received data
  // if comm_x_only set, exchange or copy directly to x, don't unpack

  k_sendlist.sync<DeviceType>();
  atomKK->sync(ExecutionSpaceFromDevice<DeviceType>::space,X_MASK);

  for (int iswap = 0; iswap < nswap; iswap++) {
    nsend = nsendproc[iswap] - sendself[iswap];
    nrecv = nrecvproc[iswap] - sendself[iswap];

    if (comm_x_only) {
      if (recvother[iswap]) {
        for (i = 0; i < nrecv; i++) {
          buf = atomKK->k_x.view<DeviceType>().data() +
            firstrecv[iswap][i]*atomKK->k_x.view<DeviceType>().extent(1);
          MPI_Irecv(buf,size_forward_recv[iswap][i],
                    MPI_DOUBLE,recvproc[iswap][i],0,world,&requests[i]);
        }
      }
      if (sendother[iswap]) {
        for (i = 0; i < nsend; i++) {
          k_sendlist_small = Kokkos::subview(k_sendlist,iswap,i,Kokkos::ALL);
          n = atomKK->avecKK->pack_comm_kokkos(sendnum[iswap][i],k_sendlist_small,
                              iswap,k_buf_send,pbc_flag[iswap][i],pbc[iswap][i]);
          DeviceType().fence();
          MPI_Send(k_buf_send.view<DeviceType>.data(),n,MPI_DOUBLE,sendproc[iswap][i],0,world);
        }
      }
      if (sendself[iswap]) {
        k_sendlist_small = Kokkos::subview(k_sendlist,iswap,nsend,Kokkos::ALL);
        atomKK->avecKK->pack_comm_self(sendnum[iswap][nsend],k_sendlist_small,
                        firstrecv[iswap][nrecv],pbc_flag[iswap][nsend],pbc[iswap][nsend]);
        DeviceType().fence();
      }
      if (recvother[iswap]) {
        atomKK->modified(ExecutionSpaceFromDevice<DeviceType>::
                         space,X_MASK);
        MPI_Waitall(nrecv,requests,MPI_STATUS_IGNORE);
      }

    } else if (ghost_velocity) {
      if (recvother[iswap]) {
        for (i = 0; i < nrecv; i++)
          MPI_Irecv(&buf_recv[size_forward*forward_recv_offset[iswap][i]],
                    size_forward_recv[iswap][i],MPI_DOUBLE,recvproc[iswap][i],0,world,&requests[i]);
      }
      if (sendother[iswap]) {
        for (i = 0; i < nsend; i++) {
          k_sendlist_small = Kokkos::subview(k_sendlist,iswap,i,Kokkos::ALL);
          n = atomKK->avecKK->pack_comm_vel(sendnum[iswap][i],k_sendlist_small,iswap,
                                  k_buf_send,pbc_flag[iswap][i],pbc[iswap][i]);
          DeviceType().fence();
          MPI_Send(k_buf_send.view<DeviceType>().data(),n,
                   MPI_DOUBLE,sendproc[iswap][i],0,world);
        }
      }
      if (sendself[iswap]) {
        k_sendlist_small = Kokkos::subview(k_sendlist,iswap,nsend,Kokkos::ALL);
        atomKK->avecKK->pack_comm_vel_kokkos(sendnum[iswap][nsend],k_sendlist_small,iswap,
                            k_buf_send,pbc_flag[iswap][nsend],pbc[iswap][nsend]);
        atomKK->avecKK->unpack_comm_vel_kokkos(recvnum[iswap][nrecv],firstrecv[iswap][nrecv],buf_send);
      }
      if (recvother[iswap]) {
        for (i = 0; i < nrecv; i++) {
          MPI_Waitany(nrecv,requests,&irecv,MPI_STATUS_IGNORE);
          atomKK->avecKK->unpack_comm_vel_kokkos(recvnum[iswap][irecv],firstrecv[iswap][irecv],
                                &buf_recv[size_forward*forward_recv_offset[iswap][irecv]]);
        }
      }

    } else {
      if (recvother[iswap]) {
        for (i = 0; i < nrecv; i++) {
          buf = atomKK->k_buf_recv.view<DeviceType>().data() +
            size_forward*forward_recv_offset[iswap][i]*atomKK->k_recv.view<DeviceType>().extent(1);
          MPI_Irecv(buf,
                    size_forward_recv[iswap][i],MPI_DOUBLE,recvproc[iswap][i],0,world,&requests[i]);
        }
      }
      if (sendother[iswap]) {
        for (i = 0; i < nsend; i++) {
          n = atomKK->avecKK->pack_comm_kokkos(sendnum[iswap][i],k_sendlist_small,iswap
                              k_buf_send,pbc_flag[iswap][i],pbc[iswap][i]);
          MPI_Send(k_buf_send,n,MPI_DOUBLE,sendproc[iswap][i],0,world);
        }
      }
      if (sendself[iswap]) {
        atomKK->avecKK->pack_comm_self(sendnum[iswap][nsend],k_sendlist_small,
                        k_buf_send,pbc_flag[iswap][nsend],pbc[iswap][nsend]);
        DeviceType().fence();
      }
      if (recvother[iswap]) {
        for (i = 0; i < nrecv; i++) {
          MPI_Waitany(nrecv,requests,&irecv,MPI_STATUS_IGNORE);
          atomKK->avecKK->unpack_comm_kokkos(recvnum[iswap][irecv],firstrecv[iswap][irecv],
                                   k_buf_recv.view<DeviceType>.data() + size_forward*forward_recv_offset[iswap][irecv]);
          DeviceType().fence();
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   reverse communication of forces on atoms every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

void CommTiledKokkos::reverse_comm()
{
  if (comm_f_only)
    atomKK->sync(Host,F_MASK);
  else
    atomKK->sync(Host,ALL_MASK);
  CommTiled::reverse_comm();
  if (comm_f_only)
    atomKK->modified(Host,F_MASK);
  else
    atomKK->modified(Host,ALL_MASK);
  atomKK->sync(Device,ALL_MASK);
}

/* ----------------------------------------------------------------------
   exchange: move atoms to correct processors
   atoms exchanged with procs that touch sub-box in each of 3 dims
   send out atoms that have left my box, receive ones entering my box
   atoms will be lost if not inside a touching proc's box
     can happen if atom moves outside of non-periodic boundary
     or if atom moves more than one proc away
   this routine called before every reneighboring
   for triclinic, atoms must be in lamda coords (0-1) before exchange is called
------------------------------------------------------------------------- */

void CommTiledKokkos::exchange()
{
  atomKK->sync(Host,ALL_MASK);
  CommTiled::exchange();
  atomKK->modified(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   borders: list nearby atoms to send to neighboring procs at every timestep
   one list is created per swap/proc that will be made
   as list is made, actually do communication
   this does equivalent of a forward_comm(), so don't need to explicitly
     call forward_comm() on reneighboring timestep
   this routine is called before every reneighboring
   for triclinic, atoms must be in lamda coords (0-1) before borders is called
------------------------------------------------------------------------- */

void CommTiledKokkos::borders()
{
  atomKK->sync(Host,ALL_MASK);
  CommTiled::borders();
  atomKK->modified(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommTiledKokkos::forward_comm(Pair *pair)
{
  CommTiled::forward_comm(pair);
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommTiledKokkos::reverse_comm(Pair *pair)
{
  CommTiled::reverse_comm(pair);
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Fix
   size/nsize used only to set recv buffer limit
   size = 0 (default) -> use comm_forward from Fix
   size > 0 -> Fix passes max size per atom
   the latter is only useful if Fix does several comm modes,
     some are smaller than max stored in its comm_forward
------------------------------------------------------------------------- */

void CommTiledKokkos::forward_comm(Fix *fix, int size)
{
  CommTiled::forward_comm(fix,size);
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Fix
   size/nsize used only to set recv buffer limit
   size = 0 (default) -> use comm_forward from Fix
   size > 0 -> Fix passes max size per atom
   the latter is only useful if Fix does several comm modes,
     some are smaller than max stored in its comm_forward
------------------------------------------------------------------------- */

void CommTiledKokkos::reverse_comm(Fix *fix, int size)
{
  CommTiled::reverse_comm(fix,size);
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Fix with variable size data
   query fix for all pack sizes to ensure buf_send is big enough
   handshake sizes before irregular comm to ensure buf_recv is big enough
   NOTE: how to setup one big buf recv with correct offsets ??
------------------------------------------------------------------------- */

void CommTiledKokkos::reverse_comm_variable(Fix *fix)
{
  CommTiled::reverse_comm_variable(fix);
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommTiledKokkos::forward_comm(Compute *compute)
{
  CommTiled::forward_comm(compute);
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommTiledKokkos::reverse_comm(Compute *compute)
{
  CommTiled::reverse_comm(compute);
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommTiledKokkos::forward_comm(Dump *dump)
{
  CommTiled::forward_comm(dump);
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommTiledKokkos::reverse_comm(Dump *dump)
{
  CommTiled::reverse_comm(dump);
}

/* ----------------------------------------------------------------------
   forward communication of Nsize values in per-atom array
------------------------------------------------------------------------- */

void CommTiledKokkos::forward_comm_array(int nsize, double **array)
{
  CommTiled::forward_comm_array(nsize,array);
}

/* ----------------------------------------------------------------------
   realloc the size of the send buffer as needed with BUFFACTOR and bufextra
   if flag = 1, realloc
   if flag = 0, don't need to realloc with copy, just free/malloc
------------------------------------------------------------------------- */

void CommTiledKokkos::grow_send(int n, int flag)
{
  grow_send_kokkos(n,flag,Host);
}

/* ----------------------------------------------------------------------
   free/malloc the size of the recv buffer as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommTiledKokkos::grow_recv(int n)
{
  grow_recv_kokkos(n,Host);
}

/* ----------------------------------------------------------------------
   realloc the size of the send buffer as needed with BUFFACTOR & BUFEXTRA
   if flag = 1, realloc
   if flag = 0, don't need to realloc with copy, just free/malloc
------------------------------------------------------------------------- */

void CommTiledKokkos::grow_send_kokkos(int n, int flag, ExecutionSpace space)
{

  maxsend = static_cast<int> (BUFFACTOR * n);
  int maxsend_border = (maxsend+BUFEXTRA)/atomKK->avecKK->size_border;
  if (flag) {
    if (space == Device)
      k_buf_send.modify<LMPDeviceType>();
    else
      k_buf_send.modify<LMPHostType>();

    if (ghost_velocity)
      k_buf_send.resize(maxsend_border,
                        atomKK->avecKK->size_border + atomKK->avecKK->size_velocity);
    else
      k_buf_send.resize(maxsend_border,atomKK->avecKK->size_border);
    buf_send = k_buf_send.view<LMPHostType>().data();
  } else {
    if (ghost_velocity)
      MemoryKokkos::realloc_kokkos(k_buf_send,"comm:k_buf_send",maxsend_border,
                        atomKK->avecKK->size_border + atomKK->avecKK->size_velocity);
    else
      MemoryKokkos::realloc_kokkos(k_buf_send,"comm:k_buf_send",maxsend_border,
                        atomKK->avecKK->size_border);
    buf_send = k_buf_send.view<LMPHostType>().data();
  }
}

/* ----------------------------------------------------------------------
   free/malloc the size of the recv buffer as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommTiledKokkos::grow_recv_kokkos(int n, ExecutionSpace /*space*/)
{
  maxrecv = static_cast<int> (BUFFACTOR * n);
  int maxrecv_border = (maxrecv+BUFEXTRA)/atomKK->avecKK->size_border;

  MemoryKokkos::realloc_kokkos(k_buf_recv,"comm:k_buf_recv",maxrecv_border,
    atomKK->avecKK->size_border);
  buf_recv = k_buf_recv.view<LMPHostType>().data();
}

/* ----------------------------------------------------------------------
    realloc the size of the iswap sendlist as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommTiled::grow_list(int iswap, int iwhich, int n)
{
  maxsendlist[iswap][iwhich] = static_cast<int> (BUFFACTOR * n);
  memory->grow(sendlist[iswap][iwhich],maxsendlist[iswap][iwhich],
               "comm:sendlist[i][j]");
}

/* ----------------------------------------------------------------------
   realloc the size of the iswap sendlist as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommTiledKokkos::grow_list(int /*iswap*/, int /*iwhich*/, int n)
{
  int size = static_cast<int> (BUFFACTOR * n);

  if (exchange_comm_classic) { // force realloc on Host
    k_sendlist.sync<LMPHostType>();
    k_sendlist.modify<LMPHostType>();
  }

  memoryKK->grow_kokkos(k_sendlist,sendlist,maxswap,size,"comm:sendlist");

  for (int i=0;i<maxswap;i++) {
    maxsendlist[i]=size; sendlist[i]=&k_sendlist.view<LMPHostType>()(i,0);
  }
}

/* ----------------------------------------------------------------------
   realloc the buffers needed for swaps
------------------------------------------------------------------------- */

void CommTiledKokkos::grow_swap(int n)
{
  free_swap();
  allocate_swap(n);
  if (mode == Comm::MULTI) {
    free_multi();
    allocate_multi(n);
  }

  maxswap = n;
  int size = MAX(k_sendlist.d_view.extent(1),BUFMIN);

  if (exchange_comm_classic) { // force realloc on Host
    k_sendlist.sync<LMPHostType>();
    k_sendlist.modify<LMPHostType>();
  }

  memoryKK->grow_kokkos(k_sendlist,sendlist,maxswap,size,"comm:sendlist");

  memory->grow(maxsendlist,n,"comm:maxsendlist");
  for (int i = 0; i < maxswap; i++) maxsendlist[i] = size;
}
