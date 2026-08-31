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

#include "comm_brick_direct_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "atom_vec_kokkos.h"
#include "kokkos.h"
#include "memory_kokkos.h"

using namespace LAMMPS_NS;

static constexpr double BUFFACTOR = 1.5;
static constexpr int BUFMIN = 1024;

/* ---------------------------------------------------------------------- */

CommBrickDirectKokkos::CommBrickDirectKokkos(LAMMPS *lmp) : CommBrickDirect(lmp)
{
  totalsend = 0;
}

/* ---------------------------------------------------------------------- */

CommBrickDirectKokkos::~CommBrickDirectKokkos()
{
  // the buffers are owned by the dual views, not by the base class

  buf_send_direct = nullptr;
  buf_recv_direct = nullptr;

  memoryKK->destroy_kokkos(k_sendatoms_list,sendatoms_list);
  sendatoms_list = nullptr;
}

/* ---------------------------------------------------------------------- */
//IMPORTANT: we *MUST* pass "*oldcomm" to the Comm initializer here, as
//           the code below *requires* that the (implicit) copy constructor
//           for Comm is run and thus creating a shallow copy of "oldcomm".
//           The call to Comm::copy_arrays() then converts the shallow copy
//           into a deep copy of the class with the new layout.

CommBrickDirectKokkos::CommBrickDirectKokkos(LAMMPS *lmp, Comm *oldcomm) :
  CommBrickDirect(lmp, oldcomm)
{
  totalsend = 0;
}

/* ----------------------------------------------------------------------
   replace the base class's per-list arrays with the host side of a dual view
   so the lists built on the device are the same memory the host routines read
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::lists_to_kokkos()
{
  if (sendatoms_list) {
    for (int ilist = 0; ilist < maxlist; ilist++)
      memory->destroy(sendatoms_list[ilist]);
    memory->sfree(sendatoms_list);
    sendatoms_list = nullptr;
  }

  k_sendatoms_list = DAT::tdual_int_2d_lr();
  memoryKK->create_kokkos(k_sendatoms_list,sendatoms_list,maxlist,BUFMIN,
                          "comm_direct:sendatoms_list");

  for (int ilist = 0; ilist < maxlist; ilist++) {
    maxsendatoms_list[ilist] = BUFMIN;
    sendatoms_list[ilist] = &k_sendatoms_list.view_host()(ilist,0);
  }
}

/* ---------------------------------------------------------------------- */

void CommBrickDirectKokkos::allocate_lists()
{
  CommBrickDirect::allocate_lists();
  lists_to_kokkos();
}

/* ---------------------------------------------------------------------- */

void CommBrickDirectKokkos::deallocate_lists(int nlist)
{
  memoryKK->destroy_kokkos(k_sendatoms_list,sendatoms_list);
  sendatoms_list = nullptr;
  CommBrickDirect::deallocate_lists(nlist);
}

/* ----------------------------------------------------------------------
   the dual view is rectangular, so every list grows together
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_list_direct(int /*ilist*/, int n)
{
  const int size = static_cast<int> (BUFFACTOR * n);

  memoryKK->grow_kokkos(k_sendatoms_list,sendatoms_list,maxlist,size,
                        "comm_direct:sendatoms_list");

  for (int i = 0; i < maxlist; i++) {
    maxsendatoms_list[i] = size;
    sendatoms_list[i] = &k_sendatoms_list.view_host()(i,0);
  }
}

/* ----------------------------------------------------------------------
   create stencil of direct swaps this proc makes with each proc in stencil
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::setup()
{
  CommBrickDirect::setup();

  MemKK::realloc_kokkos(k_swap2list,"comm_direct:swap2list",ndirect);
  MemKK::realloc_kokkos(k_pbc_flag_direct,"comm_direct:pbc_flag",ndirect);
  MemKK::realloc_kokkos(k_pbc_direct,"comm_direct:pbc",ndirect,6);
  MemKK::realloc_kokkos(k_self_flag,"comm_direct:self_flag",ndirect);

  for (int iswap = 0; iswap < ndirect; iswap++) {
    k_swap2list.view_host()[iswap] = swap2list[iswap];
    k_pbc_flag_direct.view_host()[iswap] = pbc_flag_direct[iswap];
    for (int m = 0; m < 6; m++)
      k_pbc_direct.view_host()(iswap,m) = pbc_direct[iswap][m];
    k_self_flag.view_host()(iswap) = proc_direct[iswap] == me;
  }

  k_swap2list.modify_host();
  k_pbc_flag_direct.modify_host();
  k_pbc_direct.modify_host();
  k_self_flag.modify_host();
}

/* ----------------------------------------------------------------------
   forward communication of atom coords every timestep
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::forward_comm(int dummy)
{
  // the device path below packs coords only.  ghost velocities and atom
  // styles with extra forward-comm fields both clear comm_x_only and are
  // not handled there, so they use the host path instead

  if (comm_x_only && !atomKK->k_x.NEED_TRANSFORM) {
    if (lmp->kokkos->forward_comm_on_host) forward_comm_device<LMPHostType>();
    else forward_comm_device<LMPDeviceType>();
    return;
  }

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

  CommBrickDirect::forward_comm(dummy);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::forward_comm_device()
{
  // post all receives for ghost atoms, except for swaps with self
  // comm_x_only, so receive straight into the ghost region of x

  int npost = 0;
  double *xdata = (double *) atomKK->k_x.view<DeviceType>().data();
  const int xcols = atomKK->k_x.view<DeviceType>().extent(1);

  DeviceType().fence();

  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (size_forward_recv_direct[iswap]) {
      MPI_Irecv(xdata + firstrecv_direct[iswap]*xcols,
                size_forward_recv_direct[iswap],MPI_DOUBLE,
                proc_direct[iswap],recvtag[iswap],world,&requests[npost++]);
    }
  }

  // pack every swap in one kernel, including copies to self

  k_sendatoms_list.sync<DeviceType>();
  k_swap2list.sync<DeviceType>();
  k_pbc_flag_direct.sync<DeviceType>();
  k_pbc_direct.sync<DeviceType>();
  k_self_flag.sync<DeviceType>();
  k_sendnum_scan_direct.sync<DeviceType>();
  k_firstrecv_direct.sync<DeviceType>();

  if (totalsend)
    atomKK->avecKK->pack_comm_direct(totalsend,k_sendatoms_list,
                                     k_sendnum_scan_direct,k_firstrecv_direct,
                                     k_pbc_flag_direct,k_pbc_direct,
                                     k_swap2list,k_buf_send_direct,k_self_flag);

  DeviceType().fence();

  // each swap already occupies its own rows of the send buffer,
  // so post every message at once with non-blocking sends

  int nsendpost = 0;
  int offset = 0;
  double *sbuf = k_buf_send_direct.view<DeviceType>().data();

  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (sendnum_direct[iswap]) {
      int n = sendnum_direct[iswap]*atomKK->avecKK->size_forward;
      if (proc_direct[iswap] != me)
        MPI_Isend(sbuf + offset,n,MPI_DOUBLE,proc_direct[iswap],
                  sendtag[iswap],world,&send_requests[nsendpost++]);
      offset += n;
    }
  }

  if (npost) MPI_Waitall(npost,requests,MPI_STATUS_IGNORE);
  if (nsendpost) MPI_Waitall(nsendpost,send_requests,MPI_STATUS_IGNORE);

  // MPI wrote the received coords straight into x on this side

  atomKK->modified(ExecutionSpaceFromDevice<DeviceType>::space,X_MASK);
  DeviceType().fence();
}

/* ----------------------------------------------------------------------
   reverse communication of forces on atoms every timestep
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::reverse_comm()
{
  if (lmp->kokkos->reverse_comm_on_host) reverse_comm_device<LMPHostType>();
  else reverse_comm_device<LMPDeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::reverse_comm_device()
{
  // buffer offsets are in atoms; reverse comm moves size_reverse per atom
  // recv_offset_reverse_atoms scans sendnum over swaps, so it indexes the
  //   region each swap's owned-atom contributions arrive in

  const int nrev = size_reverse;

  // post all receives for owned atoms, except for swaps with self

  int npost = 0;

  DeviceType().fence();

  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (sendnum_direct[iswap] == 0) continue;
    MPI_Irecv(k_buf_recv_reverse.view<DeviceType>().data() +
              nrev*recv_offset_reverse_atoms[iswap],
              nrev*sendnum_direct[iswap],MPI_DOUBLE,
              proc_direct[iswap],sendtag[iswap],world,&requests[npost++]);
  }

  // copy/sum to self on device
  // reads the ghost region and sums into owned atoms, so it is disjoint from
  //   the sends below and can run before them

  k_sendatoms_list.sync<DeviceType>();

  for (int iself = 0; iself < nself_direct; iself++) {
    const int iswap = self_indices_direct[iself];
    if (sendnum_direct[iswap] == 0) continue;
    auto k_list = Kokkos::subview(k_sendatoms_list,swap2list[iswap],Kokkos::ALL);
    atomKK->avecKK->pack_reverse_self_kokkos(sendnum_direct[iswap],k_list,
                                             firstrecv_direct[iswap]);
  }

  DeviceType().fence();

  // send ghost contributions to the procs that own those atoms
  // comm_f_only sends straight out of the ghost region of f, which the
  //   unpack below never touches, so the sends can stay in flight

  int nsendpost = 0;

  if (comm_f_only && !atomKK->k_f.NEED_TRANSFORM) {

    atomKK->sync(ExecutionSpaceFromDevice<DeviceType>::space,F_MASK);
    DeviceType().fence();

    double *fdata = (double *) atomKK->k_f.view<DeviceType>().data();
    const int fcols = atomKK->k_f.view<DeviceType>().extent(1);

    for (int iswap = 0; iswap < ndirect; iswap++) {
      if (proc_direct[iswap] == me) continue;
      if (recvnum_direct[iswap] == 0) continue;
      MPI_Isend(fdata + firstrecv_direct[iswap]*fcols,
                nrev*recvnum_direct[iswap],MPI_DOUBLE,proc_direct[iswap],
                recvtag[iswap],world,&send_requests[nsendpost++]);
    }

  } else {

    int send_offset = 0;

    for (int iswap = 0; iswap < ndirect; iswap++) {
      if (proc_direct[iswap] == me) continue;
      if (recvnum_direct[iswap] == 0) continue;
      auto k_sub = Kokkos::subview(k_buf_send_reverse,
                                   Kokkos::make_pair(send_offset,
                                     send_offset+recvnum_direct[iswap]),
                                   Kokkos::ALL);
      const int n = atomKK->avecKK->pack_reverse_kokkos(recvnum_direct[iswap],
                                                        firstrecv_direct[iswap],k_sub);
      if (n) {
        DeviceType().fence();
        MPI_Isend(k_buf_send_reverse.view<DeviceType>().data() + nrev*send_offset,
                  n,MPI_DOUBLE,proc_direct[iswap],recvtag[iswap],world,
                  &send_requests[nsendpost++]);
        send_offset += recvnum_direct[iswap];
      }
    }
  }

  // wait on incoming messages, summing each into the owned atoms as it lands

  for (int i = 0; i < npost; i++) {
    int irecv;
    MPI_Waitany(npost,requests,&irecv,MPI_STATUS_IGNORE);
    const int iswap = send_indices_direct[irecv];
    auto k_list = Kokkos::subview(k_sendatoms_list,swap2list[iswap],Kokkos::ALL);
    auto k_sub = Kokkos::subview(k_buf_recv_reverse,
                                 Kokkos::make_pair(recv_offset_reverse_atoms[iswap],
                                   recv_offset_reverse_atoms[iswap]+sendnum_direct[iswap]),
                                 Kokkos::ALL);
    atomKK->avecKK->unpack_reverse_kokkos(sendnum_direct[iswap],k_list,k_sub);
  }

  if (nsendpost) MPI_Waitall(nsendpost,send_requests,MPI_STATUS_IGNORE);

  DeviceType().fence();
}

/* ----------------------------------------------------------------------
   exchange: move atoms to correct processors
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::exchange()
{
  atomKK->sync(Host,ALL_MASK);
  CommBrickDirect::exchange();
  atomKK->modified(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   borders: list nearby atoms to send to neighboring procs at every timestep
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::borders()
{
  atomKK->sync(Host,ALL_MASK);
  int prev_auto_sync = lmp->kokkos->auto_sync;
  lmp->kokkos->auto_sync = 1;
  CommBrickDirect::borders();
  lmp->kokkos->auto_sync = prev_auto_sync;
  atomKK->modified(Host,ALL_MASK);

  // sendatoms_list is the host side of k_sendatoms_list, so the lists the
  //   build above wrote are already the device view's host data; it is grown
  //   only through grow_list_direct(), which keeps the row pointers in step

  if ((int) k_sendnum_scan_direct.extent(0) < ndirect) {
    MemKK::realloc_kokkos(k_sendnum_scan_direct,"comm_direct:sendnum_scan",ndirect);
    MemKK::realloc_kokkos(k_firstrecv_direct,"comm_direct:firstrecv",ndirect);
  }

  int scan = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    scan += sendnum_direct[iswap];
    k_sendnum_scan_direct.view_host()[iswap] = scan;
    k_firstrecv_direct.view_host()[iswap] = firstrecv_direct[iswap];
  }
  totalsend = scan;

  if (totalsend > (int) k_buf_send_direct.view_device().extent(0))
    grow_send_direct(totalsend*size_forward,0);

  // reverse comm buffers, indexed in atoms with size_reverse per atom
  // sends carry the ghosts this proc received (rsum), receives carry the
  //   owned-atom contributions coming back from every swap (ssum)

  if ((rsum_direct > (int) k_buf_send_reverse.view_device().extent(0)) ||
      (size_reverse != (int) k_buf_send_reverse.view_device().extent(1)))
    MemKK::realloc_kokkos(k_buf_send_reverse,"comm:buf_send_reverse",
                          rsum_direct+1,size_reverse);

  if ((ssum_direct > (int) k_buf_recv_reverse.view_device().extent(0)) ||
      (size_reverse != (int) k_buf_recv_reverse.view_device().extent(1)))
    MemKK::realloc_kokkos(k_buf_recv_reverse,"comm:buf_recv_reverse",
                          ssum_direct+1,size_reverse);

  k_sendatoms_list.modify_host();
  k_sendnum_scan_direct.modify_host();
  k_firstrecv_direct.modify_host();
}

/* ---------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_send_direct(int n, int flag)
{
  const int nrow = static_cast<int> (BUFFACTOR * n) / 3 + 1;

  if (flag == 1) {
    k_buf_send_direct.resize(nrow,3);
    k_buf_send_direct.clear_sync_state();
  } else {
    MemKK::realloc_kokkos(k_buf_send_direct,"comm:buf_send_direct",nrow,3);
  }
  maxsend_direct = nrow*3;
  buf_send_direct = k_buf_send_direct.view_host().data();
}

/* ---------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_recv_direct(int n)
{
  const int nrow = static_cast<int> (BUFFACTOR * n) / 3 + 1;

  MemKK::realloc_kokkos(k_buf_recv_direct,"comm:buf_recv_direct",nrow,3);
  maxrecv_direct = nrow*3;
  buf_recv_direct = k_buf_recv_direct.view_host().data();
}
