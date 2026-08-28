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
  if (comm_f_only)
    atomKK->sync(Host,F_MASK);
  else
    atomKK->sync(Host,ALL_MASK);

  CommBrickDirect::reverse_comm();

  if (comm_f_only)
    atomKK->modified(Host,F_MASK);
  else
    atomKK->modified(Host,ALL_MASK);
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

  // mirror the per-swap send lists and scans onto the device
  // for the fused pack in forward_comm_device()

  int maxsend = 0;
  for (int ilist = 0; ilist < maxlist; ilist++)
    maxsend = MAX(maxsend,maxsendatoms_list[ilist]);

  if ((int) k_sendatoms_list.view_device().extent(1) < maxsend)
    MemKK::realloc_kokkos(k_sendatoms_list,"comm_direct:sendatoms_list",maxlist,maxsend);

  if ((int) k_sendnum_scan_direct.extent(0) < ndirect) {
    MemKK::realloc_kokkos(k_sendnum_scan_direct,"comm_direct:sendnum_scan",ndirect);
    MemKK::realloc_kokkos(k_firstrecv_direct,"comm_direct:firstrecv",ndirect);
  }

  for (int ilist = 0; ilist < maxlist; ilist++) {
    if (!active_list[ilist]) continue;
    const int nsend = sendnum_list[ilist];
    for (int i = 0; i < nsend; i++)
      k_sendatoms_list.view_host()(ilist,i) = sendatoms_list[ilist][i];
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
