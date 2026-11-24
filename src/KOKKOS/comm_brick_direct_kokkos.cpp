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
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "kokkos.h"
#include "kokkos_base.h"
#include "memory_kokkos.h"
#include "pair.h"

#include <vector>

using namespace LAMMPS_NS;

static constexpr double BUFFACTOR = 1.5;
static constexpr int BUFMIN = 1024;


/* ----------------------------------------------------------------------
   build every send list on the device in a single pass over the owned atoms
   a parallel scan carries one running count per active list, so each list
     comes out in increasing atom order, exactly as the host build does, and
     the result does not depend on thread scheduling
   a list is stored only while it fits; the scan total gives the needed size
------------------------------------------------------------------------- */

namespace {
  static constexpr int MAXACTIVE = 27;

  struct DirectListCounts {
    int c[MAXACTIVE];
    KOKKOS_INLINE_FUNCTION DirectListCounts() {
      for (int a = 0; a < MAXACTIVE; a++) c[a] = 0;
    }
  };
}

template<class DeviceType>
struct CommBrickDirectKokkos_BuildLists {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef DirectListCounts value_type;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_int_2d_lr _list;
  int _nactive,_maxsend,_center_active;
  int _alist[MAXACTIVE];
  int _astate[MAXACTIVE][3];
  int _dcheck[3][3];
  double _dlo[3][3],_dhi[3][3];

  CommBrickDirectKokkos_BuildLists(const DAT::ttransform_kkfloat_1d_3_lr &x,
                                   const DAT::tdual_int_2d_lr &list,
                                   int nactive, int maxsend, int center_active,
                                   const int *alist, const int astate[][3],
                                   const int dcheck[][3], const double dlo[][3],
                                   const double dhi[][3]) :
    _x(x.template view<DeviceType>()),
    _list(list.template view<DeviceType>()),
    _nactive(nactive),_maxsend(maxsend),_center_active(center_active)
  {
    for (int a = 0; a < MAXACTIVE; a++) {
      _alist[a] = (a < nactive) ? alist[a] : 0;
      for (int d = 0; d < 3; d++) _astate[a][d] = (a < nactive) ? astate[a][d] : 1;
    }
    for (int d = 0; d < 3; d++)
      for (int k = 0; k < 3; k++) {
        _dcheck[d][k] = dcheck[d][k];
        _dlo[d][k] = dlo[d][k];
        _dhi[d][k] = dhi[d][k];
      }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type &v) const {
    for (int a = 0; a < MAXACTIVE; a++) v.c[a] = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    for (int a = 0; a < MAXACTIVE; a++) dst.c[a] += src.c[a];
  }

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int i, value_type &update, const bool final) const {
    int q[3][3];
    for (int d = 0; d < 3; d++) {
      const double v = static_cast<double>(_x(i,d));
      q[d][0] = !_dcheck[d][0] || ((v >= _dlo[d][0]) && (v <= _dhi[d][0]));
      q[d][1] = 1;
      q[d][2] = !_dcheck[d][2] || ((v >= _dlo[d][2]) && (v <= _dhi[d][2]));
    }

    // an atom in no lo/hi slab in any dim can only belong to the unrestricted
    //   list, so skip it outright unless that list is in use

    if (!_center_active && !q[0][0] && !q[0][2] && !q[1][0] &&
        !q[1][2] && !q[2][0] && !q[2][2]) return;

    for (int a = 0; a < _nactive; a++) {
      if (!q[0][_astate[a][0]]) continue;
      if (!q[1][_astate[a][1]]) continue;
      if (!q[2][_astate[a][2]]) continue;
      if (final && (update.c[a] < _maxsend)) _list(_alist[a],update.c[a]) = i;
      update.c[a]++;
    }
  }
};

/* ---------------------------------------------------------------------- */

CommBrickDirectKokkos::CommBrickDirectKokkos(LAMMPS *lmp) : CommBrickDirect(lmp)
{
  totalsend = 0;
  border_device_flag = 0;
  k_total_send = DAT::tdual_int_scalar("comm_direct:total_send");
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
  border_device_flag = 0;
  k_total_send = DAT::tdual_int_scalar("comm_direct:total_send");
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
   ilist < 0: called from the device build, which rewrites every list after
     growing, so the contents need not be kept and no copy is made
   ilist >= 0: called from the host build in the middle of filling the lists;
     DualView::resize() keeps only the side flagged as last modified and
     replaces the other with an uninitialized mirror, so the host side has to
     be flagged first or the entries already written are lost on a GPU build
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_list_direct(int ilist, int n)
{
  const int size = static_cast<int> (BUFFACTOR * n) + 1;

  if (ilist < 0) {
    MemKK::realloc_kokkos(k_sendatoms_list,"comm_direct:sendatoms_list",maxlist,size);
  } else {
    k_sendatoms_list.modify_host();
    k_sendatoms_list.resize(maxlist,size);
  }

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

  // forward_comm_legacy (the default without GPUs) selects the classic host
  //   pack/unpack, which is faster than the device kernels on a CPU

  if (comm_x_only && !atomKK->k_x.NEED_TRANSFORM && !lmp->kokkos->forward_comm_legacy) {
    if (lmp->kokkos->forward_comm_on_host) forward_comm_device<LMPHostType>();
    else forward_comm_device<LMPDeviceType>();
    return;
  }

  // sync and flag only the fields the pack/unpack routines touch;
  //   flagging ALL_MASK would make the next device kernel copy every
  //   per-atom array back to the device

  const uint64_t mask = ghost_velocity ? atomKK->avecKK->datamask_comm_vel
                                           : atomKK->avecKK->datamask_comm;
  atomKK->sync(Host,mask);
  CommBrickDirect::forward_comm(dummy);
  atomKK->modified(Host,mask);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::forward_comm_device()
{
  // post all receives for ghost atoms, except for swaps with self
  // comm_x_only, so receive straight into the ghost region of x

  // bring x up to date in this space before any receive is posted into it:
  //   the pack below syncs x too, and if that sync copied the whole array
  //   while messages were already landing in the ghost region it would
  //   overwrite them

  atomKK->sync(lmp->kokkos->forward_comm_on_host ? HostKK : Device,X_MASK);

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
  // reverse_comm_legacy (the default without GPUs) selects the classic host
  //   pack/unpack, as in CommBrickKokkos

  if (lmp->kokkos->reverse_comm_legacy) {
    atomKK->sync(Host,atomKK->avecKK->datamask_reverse);
    CommBrickDirect::reverse_comm();
    atomKK->modified(Host,atomKK->avecKK->datamask_reverse);
    return;
  }

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

    // pack every swap, fence once, then post all the sends

    int send_offset = 0;
    std::vector<int> nsend_swap(ndirect,0), offset_swap(ndirect,0);

    for (int iswap = 0; iswap < ndirect; iswap++) {
      if (proc_direct[iswap] == me) continue;
      if (recvnum_direct[iswap] == 0) continue;
      auto k_sub = Kokkos::subview(k_buf_send_reverse,
                                   Kokkos::make_pair(send_offset,
                                     send_offset+recvnum_direct[iswap]),
                                   Kokkos::ALL);
      nsend_swap[iswap] = atomKK->avecKK->pack_reverse_kokkos(recvnum_direct[iswap],
                                                              firstrecv_direct[iswap],k_sub);
      offset_swap[iswap] = send_offset;
      send_offset += recvnum_direct[iswap];
    }

    DeviceType().fence();

    for (int iswap = 0; iswap < ndirect; iswap++) {
      if (proc_direct[iswap] == me) continue;
      if (recvnum_direct[iswap] == 0) continue;
      MPI_Isend(k_buf_send_reverse.view<DeviceType>().data() + nrev*offset_swap[iswap],
                nsend_swap[iswap],MPI_DOUBLE,proc_direct[iswap],recvtag[iswap],world,
                &send_requests[nsendpost++]);
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
   CommBrickDirect does not define its own exchange, it uses CommBrick's
     dimension-by-dimension one, and the device version of that lives in
     CommBrickKokkos::exchange_device().  CommBrickKokkos is a sibling of
     this class rather than a parent, so that code cannot be reached from
     here; running it on the device would mean either duplicating it and the
     dual views it owns, or rearranging the class hierarchy so the direct
     stencil and the KOKKOS exchange can be composed.  Until then the atoms
     have to be synced to the host for the inherited routine.
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::exchange()
{
  atomKK->sync(Host,ALL_MASK);
  CommBrickDirect::exchange();
  atomKK->modified(Host,ALL_MASK);
}


/* ----------------------------------------------------------------------
   build the per-swap lists of owned atoms, one kernel per active list
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::build_lists()
{
  if (!border_device_flag) {
    CommBrickDirect::build_lists();
    return;
  }

  if (lmp->kokkos->exchange_comm_on_host) build_lists_device<LMPHostType>();
  else build_lists_device<LMPDeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::build_lists_device()
{
  const ExecutionSpace exec = ExecutionSpaceFromDevice<DeviceType>::space;
  const int nlocal = atom->nlocal;
  const int dim = domain->dimension;

  atomKK->sync(exec,X_MASK);

  // per-dim slab bounds, identical to the host build in CommBrickDirect

  int dcheck[3][3];
  double dlo[3][3],dhi[3][3];
  for (int d = 0; d < 3; d++)
    for (int k = 0; k < 3; k++) {
      dcheck[d][k] = 0;
      dlo[d][k] = dhi[d][k] = 0.0;
    }

  int nactive = 0;
  int alist[27],astate[27][3];
  int center_active = 0;

  for (int ilist = 0; ilist < maxlist; ilist++) {
    sendnum_list[ilist] = 0;
    if (!active_list[ilist]) continue;
    int state[3];
    state[0] = ilist % 3;
    state[1] = (ilist / 3) % 3;
    state[2] = (dim == 3) ? ilist / 9 : 1;
    alist[nactive] = ilist;
    for (int d = 0; d < 3; d++) {
      astate[nactive][d] = state[d];
      dcheck[d][state[d]] = check_list[ilist][d];
      dlo[d][state[d]] = bounds_list[ilist][d][0];
      dhi[d][state[d]] = bounds_list[ilist][d][1];
    }
    if ((state[0] == 1) && (state[1] == 1) && (state[2] == 1)) center_active = 1;
    nactive++;
  }

  // one scan builds every list; on overflow grow once and repeat the scan
  // the lists are fully rewritten, so the grow does not need to keep them

  DirectListCounts total;
  for (int attempt = 0; attempt < 2; attempt++) {
    const int maxsend = maxsendatoms_list[0];
    CommBrickDirectKokkos_BuildLists<DeviceType>
      f(atomKK->k_x,k_sendatoms_list,nactive,maxsend,center_active,
        alist,astate,dcheck,dlo,dhi);
    Kokkos::parallel_scan(Kokkos::RangePolicy<DeviceType>(0,nlocal),f,total);

    int maxcount = 0;
    for (int a = 0; a < nactive; a++) maxcount = MAX(maxcount,total.c[a]);
    if (maxcount <= maxsend) break;

    grow_list_direct(-1,maxcount);
  }

  for (int a = 0; a < nactive; a++) sendnum_list[alist[a]] = total.c[a];

  k_sendatoms_list.template modify<DeviceType>();

  // the host routines for non-Kokkos styles read sendatoms_list directly,
  //   so the host side has to be current

  k_sendatoms_list.sync_host();
}

/* ----------------------------------------------------------------------
   exchange border data for every swap on the device
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::borders_comm()
{
  if (!border_device_flag) {
    CommBrickDirect::borders_comm();
    return;
  }

  if (lmp->kokkos->exchange_comm_on_host) borders_comm_device<LMPHostType>();
  else borders_comm_device<LMPDeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::borders_comm_device()
{
  const ExecutionSpace exec = ExecutionSpaceFromDevice<DeviceType>::space;
  const int ncol = ghost_velocity ?
    size_border + atomKK->avecKK->size_velocity : size_border;

  // buffers are indexed in atoms: sends carry sendnum per swap (ssum in all),
  //   receives carry recvnum per swap (rsum in all)

  if ((ssum_direct + 1 > (int) k_buf_send_border.view_device().extent(0)) ||
      (ncol != (int) k_buf_send_border.view_device().extent(1)))
    MemKK::realloc_kokkos(k_buf_send_border,"comm:buf_send_border",ssum_direct+1,ncol);

  if ((rsum_direct + 1 > (int) k_buf_recv_border.view_device().extent(0)) ||
      (ncol != (int) k_buf_recv_border.view_device().extent(1)))
    MemKK::realloc_kokkos(k_buf_recv_border,"comm:buf_recv_border",rsum_direct+1,ncol);

  k_sendatoms_list.sync<DeviceType>();

  // post all receives for ghost atoms, except for swaps with self
  // recv_offset_forward_atoms scans recvnum, so it indexes each swap's region

  int npost = 0;

  DeviceType().fence();

  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (recvnum_direct[iswap] == 0) continue;
    MPI_Irecv(k_buf_recv_border.view<DeviceType>().data() +
              ncol*recv_offset_forward_atoms[iswap],
              ncol*recvnum_direct[iswap],MPI_DOUBLE,
              proc_direct[iswap],recvtag[iswap],world,&requests[npost++]);
  }

  // copies to self go through the send buffer, so do them before the sends
  //   claim it

  for (int iself = 0; iself < nself_direct; iself++) {
    const int iswap = self_indices_direct[iself];
    if (sendnum_direct[iswap] == 0) continue;
    auto k_list = Kokkos::subview(k_sendatoms_list,swap2list[iswap],Kokkos::ALL);
    if (ghost_velocity) {
      atomKK->avecKK->pack_border_vel_kokkos(sendnum_direct[iswap],k_list,
                                             k_buf_send_border,pbc_flag_direct[iswap],
                                             pbc_direct[iswap],exec);
      atomKK->avecKK->unpack_border_vel_kokkos(recvnum_direct[iswap],
                                               firstrecv_direct[iswap],
                                               k_buf_send_border,exec);
    } else {
      atomKK->avecKK->pack_border_kokkos(sendnum_direct[iswap],k_list,
                                         k_buf_send_border,pbc_flag_direct[iswap],
                                         pbc_direct[iswap],exec);
      atomKK->avecKK->unpack_border_kokkos(recvnum_direct[iswap],
                                           firstrecv_direct[iswap],
                                           k_buf_send_border,exec);
    }
  }

  // pack each remaining swap into its own region and send it

  // launch the packs for every swap back to back into their own regions,
  //   then fence once and post all the sends; a fence per swap would
  //   serialize kernel launch and MPI for every neighbor

  int send_offset = 0;
  std::vector<int> nsend_swap(ndirect,0), offset_swap(ndirect,0);

  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (sendnum_direct[iswap] == 0) continue;

    auto k_list = Kokkos::subview(k_sendatoms_list,swap2list[iswap],Kokkos::ALL);
    auto k_sub = Kokkos::subview(k_buf_send_border,
                                 Kokkos::make_pair(send_offset,
                                   send_offset+sendnum_direct[iswap]),
                                 Kokkos::ALL);
    if (ghost_velocity)
      nsend_swap[iswap] =
        atomKK->avecKK->pack_border_vel_kokkos(sendnum_direct[iswap],k_list,k_sub,
                                               pbc_flag_direct[iswap],
                                               pbc_direct[iswap],exec);
    else
      nsend_swap[iswap] =
        atomKK->avecKK->pack_border_kokkos(sendnum_direct[iswap],k_list,k_sub,
                                           pbc_flag_direct[iswap],
                                           pbc_direct[iswap],exec);
    offset_swap[iswap] = send_offset;
    send_offset += sendnum_direct[iswap];
  }

  DeviceType().fence();

  int nsendpost = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (sendnum_direct[iswap] == 0) continue;
    MPI_Isend(k_buf_send_border.view<DeviceType>().data() + ncol*offset_swap[iswap],
              nsend_swap[iswap],MPI_DOUBLE,proc_direct[iswap],sendtag[iswap],world,
              &send_requests[nsendpost++]);
  }

  // unpack each message as it arrives

  for (int ipost = 0; ipost < npost; ipost++) {
    int irecv;
    MPI_Waitany(npost,requests,&irecv,MPI_STATUS_IGNORE);
    const int iswap = recv_indices_direct[irecv];
    auto k_sub = Kokkos::subview(k_buf_recv_border,
                                 Kokkos::make_pair(recv_offset_forward_atoms[iswap],
                                   recv_offset_forward_atoms[iswap]+recvnum_direct[iswap]),
                                 Kokkos::ALL);
    if (ghost_velocity)
      atomKK->avecKK->unpack_border_vel_kokkos(recvnum_direct[iswap],
                                               firstrecv_direct[iswap],k_sub,exec);
    else
      atomKK->avecKK->unpack_border_kokkos(recvnum_direct[iswap],
                                           firstrecv_direct[iswap],k_sub,exec);
  }

  if (nsendpost) MPI_Waitall(nsendpost,send_requests,MPI_STATUS_IGNORE);

  DeviceType().fence();
}

/* ----------------------------------------------------------------------
   borders: list nearby atoms to send to neighboring procs at every timestep
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::borders()
{
  // decide once whether the device path can handle this configuration
  // extra border data, multi mode and a border group are not implemented there

  if (!lmp->kokkos->exchange_comm_legacy) {
    if (atom->nextra_border || mode != Comm::SINGLE || bordergroup) {
      if (me == 0)
        error->warning(FLERR,"Required border comm not yet implemented in Kokkos "
                       "communication, switching to legacy exchange/border communication");
      lmp->kokkos->exchange_comm_legacy = 1;
    }
  }

  border_device_flag = !lmp->kokkos->exchange_comm_legacy;

  if (border_device_flag) {

    // build_lists() and borders_comm() run on the device

    CommBrickDirect::borders();

  } else {

    if (ghost_velocity)
      atomKK->sync(Host,atomKK->avecKK->datamask_border_vel);
    else
      atomKK->sync(Host,atomKK->avecKK->datamask_border);
    k_sendatoms_list.sync_host();
    int prev_auto_sync = lmp->kokkos->auto_sync;
    lmp->kokkos->auto_sync = 1;
    CommBrickDirect::borders();
    lmp->kokkos->auto_sync = prev_auto_sync;
    k_sendatoms_list.modify_host();
    if (ghost_velocity)
      atomKK->modified(Host,atomKK->avecKK->datamask_border_vel);
    else
      atomKK->modified(Host,atomKK->avecKK->datamask_border);
  }

  // sendatoms_list is the host side of k_sendatoms_list, so the lists the
  //   build above wrote are already the device view's host data; it is grown
  //   only through grow_list_direct(), which keeps the row pointers in step

  // exactly ndirect entries: the fused pack binary-searches the whole scan

  if ((int) k_sendnum_scan_direct.extent(0) != ndirect) {
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

  // the send lists were flagged where they were built (device build or the
  //   legacy host branch above), so they are not flagged again here; doing so
  //   would force a full host-to-device copy of every list on the next comm

  k_sendnum_scan_direct.modify_host();
  k_firstrecv_direct.modify_host();
}

/* ----------------------------------------------------------------------
   forward/reverse comm for Pair, Fix, Compute styles
   Kokkos styles that ask for device comm get it through the KokkosBase
     pack/unpack routines, everything else uses the host routines of
     CommBrickDirect, mirroring the dispatch in CommBrickKokkos
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::forward_comm(Pair *pair, int size)
{
  if (pair->execution_space == Host || pair->execution_space == HostKK ||
      lmp->kokkos->forward_pair_comm_legacy) {
    k_sendatoms_list.sync_host();
    CommBrickDirect::forward_comm(pair,size);
  } else forward_comm_device<LMPDeviceType>(pair,size);
}

void CommBrickDirectKokkos::reverse_comm(Pair *pair, int size)
{
  if (pair->execution_space == Host || pair->execution_space == HostKK ||
      !pair->reverse_comm_device || lmp->kokkos->reverse_pair_comm_legacy) {
    k_sendatoms_list.sync_host();
    CommBrickDirect::reverse_comm(pair,size);
  } else reverse_comm_device<LMPDeviceType>(pair,size);
}

void CommBrickDirectKokkos::forward_comm(Fix *fix, int size)
{
  if (fix->execution_space == Host || fix->execution_space == HostKK ||
      !fix->forward_comm_device || lmp->kokkos->forward_fix_comm_legacy) {
    k_sendatoms_list.sync_host();
    CommBrickDirect::forward_comm(fix,size);
  } else forward_comm_device<LMPDeviceType>(fix,size);
}

void CommBrickDirectKokkos::reverse_comm(Fix *fix, int size)
{
  if (fix->execution_space == Host || fix->execution_space == HostKK ||
      !fix->reverse_comm_device || lmp->kokkos->reverse_fix_comm_legacy) {
    k_sendatoms_list.sync_host();
    CommBrickDirect::reverse_comm(fix,size);
  } else reverse_comm_device<LMPDeviceType>(fix,size);
}

void CommBrickDirectKokkos::forward_comm(Compute *compute, int size)
{
  if (compute->execution_space == Host || compute->execution_space == HostKK ||
      !compute->forward_comm_device || lmp->kokkos->forward_compute_comm_legacy) {
    k_sendatoms_list.sync_host();
    CommBrickDirect::forward_comm(compute,size);
  } else forward_comm_device<LMPDeviceType>(compute,size);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::forward_comm_device(Pair *pair, int size)
{
  const int nsize = size ? size : pair->comm_forward;
  forward_comm_object<DeviceType>(dynamic_cast<KokkosBase *>(pair),nsize);
}

template<class DeviceType>
void CommBrickDirectKokkos::reverse_comm_device(Pair *pair, int size)
{
  const int nsize = size ? size : MAX(pair->comm_reverse,pair->comm_reverse_off);
  reverse_comm_object<DeviceType>(dynamic_cast<KokkosBase *>(pair),nsize);
}

template<class DeviceType>
void CommBrickDirectKokkos::forward_comm_device(Fix *fix, int size)
{
  const int nsize = size ? size : fix->comm_forward;
  forward_comm_object<DeviceType>(dynamic_cast<KokkosBase *>(fix),nsize);
}

template<class DeviceType>
void CommBrickDirectKokkos::reverse_comm_device(Fix *fix, int size)
{
  const int nsize = size ? size : fix->comm_reverse;
  reverse_comm_object<DeviceType>(dynamic_cast<KokkosBase *>(fix),nsize);
}

template<class DeviceType>
void CommBrickDirectKokkos::forward_comm_device(Compute *compute, int size)
{
  const int nsize = size ? size : compute->comm_forward;
  forward_comm_object<DeviceType>(dynamic_cast<KokkosBase *>(compute),nsize);
}

/* ----------------------------------------------------------------------
   device forward comm for a Kokkos style
   buffer layout, in atoms times nsize:
     send buffer: swap regions at the exclusive scan of sendnum (owned atoms)
     recv buffer: swap regions at the exclusive scan of recvnum (ghost atoms)
   every swap is packed first, swaps with self are unpacked straight from
     the send buffer, one fence, then all messages are posted at once
   a style may pack fewer than nsize values per atom (e.g. fix rigid/small),
     possibly none, but a message is still sent for every swap whose
     receiver posted a receive, so the receive always completes
------------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::forward_comm_object(KokkosBase *obj, int nsize)
{
  if (!obj) error->all(FLERR,"Style requested Kokkos device communication without support");

  grow_buf_obj(nsize*MAX(ssum_direct,rsum_direct) + 1);
  k_sendatoms_list.sync<DeviceType>();

  // MPI uses device buffers with GPU-aware MPI, otherwise the host copies,
  //   staged per message; on a host-only build the two are the same memory

  const bool stage = !lmp->kokkos->gpu_aware_flag &&
    (k_buf_send_obj.view<DeviceType>().data() != k_buf_send_obj.view_host().data());
  double *sbuf = stage ? k_buf_send_obj.view_host().data() :
    k_buf_send_obj.view<DeviceType>().data();
  double *rbuf = stage ? k_buf_recv_obj.view_host().data() :
    k_buf_recv_obj.view<DeviceType>().data();

  // post all receives, except for swaps with self

  int npost = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (recvnum_direct[iswap] == 0) continue;
    MPI_Irecv(rbuf + nsize*recv_offset_forward_atoms[iswap],nsize*recvnum_direct[iswap],
              MPI_DOUBLE,proc_direct[iswap],recvtag[iswap],world,&requests[npost++]);
  }

  // pack every swap into its own region; unpack swaps with self right away

  std::vector<int> nsend_swap(ndirect,0);
  int nsendtotal = 0;

  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (sendnum_direct[iswap] == 0) continue;
    const int first = nsize*recv_offset_reverse_atoms[iswap];
    DAT::tdual_double_1d k_sub =
      Kokkos::subview(k_buf_send_obj,std::make_pair(first,first+nsize*sendnum_direct[iswap]));
    auto k_list = Kokkos::subview(k_sendatoms_list,swap2list[iswap],Kokkos::ALL);
    nsend_swap[iswap] = obj->pack_forward_comm_kokkos(sendnum_direct[iswap],k_list,k_sub,
                                                      pbc_flag_direct[iswap],
                                                      pbc_direct[iswap]);
    if (proc_direct[iswap] == me)
      obj->unpack_forward_comm_kokkos(recvnum_direct[iswap],firstrecv_direct[iswap],k_sub);
    else nsendtotal = first + nsize*sendnum_direct[iswap];
  }

  DeviceType().fence();

  if (stage && nsendtotal) {
    auto d_s = Kokkos::subview(k_buf_send_obj.view<DeviceType>(),std::make_pair(0,nsendtotal));
    auto h_s = Kokkos::subview(k_buf_send_obj.view_host(),std::make_pair(0,nsendtotal));
    Kokkos::deep_copy(h_s,d_s);
  }

  int nsendpost = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (sendnum_direct[iswap] == 0) continue;
    MPI_Isend(sbuf + nsize*recv_offset_reverse_atoms[iswap],nsend_swap[iswap],MPI_DOUBLE,
              proc_direct[iswap],sendtag[iswap],world,&send_requests[nsendpost++]);
  }

  // unpack each message as it arrives

  for (int ipost = 0; ipost < npost; ipost++) {
    int irecv;
    MPI_Waitany(npost,requests,&irecv,MPI_STATUS_IGNORE);
    const int iswap = recv_indices_direct[irecv];
    const int first = nsize*recv_offset_forward_atoms[iswap];
    const int last = first + nsize*recvnum_direct[iswap];
    if (stage) {
      auto d_r = Kokkos::subview(k_buf_recv_obj.view<DeviceType>(),std::make_pair(first,last));
      auto h_r = Kokkos::subview(k_buf_recv_obj.view_host(),std::make_pair(first,last));
      Kokkos::deep_copy(d_r,h_r);
    }
    DAT::tdual_double_1d k_sub = Kokkos::subview(k_buf_recv_obj,std::make_pair(first,last));
    obj->unpack_forward_comm_kokkos(recvnum_direct[iswap],firstrecv_direct[iswap],k_sub);
  }

  if (nsendpost) MPI_Waitall(nsendpost,send_requests,MPI_STATUS_IGNORE);

  // the buffers are reused by the next call, so the unpacks must be done

  DeviceType().fence();
}

/* ----------------------------------------------------------------------
   device reverse comm for a Kokkos style
   buffer layout, in atoms times nsize:
     send buffer: swap regions at the exclusive scan of recvnum (ghost atoms)
     recv buffer: swap regions at the exclusive scan of sendnum (owned atoms)
------------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::reverse_comm_object(KokkosBase *obj, int nsize)
{
  if (!obj) error->all(FLERR,"Style requested Kokkos device communication without support");

  grow_buf_obj(nsize*MAX(ssum_direct,rsum_direct) + 1);
  k_sendatoms_list.sync<DeviceType>();

  const bool stage = !lmp->kokkos->gpu_aware_flag &&
    (k_buf_send_obj.view<DeviceType>().data() != k_buf_send_obj.view_host().data());
  double *sbuf = stage ? k_buf_send_obj.view_host().data() :
    k_buf_send_obj.view<DeviceType>().data();
  double *rbuf = stage ? k_buf_recv_obj.view_host().data() :
    k_buf_recv_obj.view<DeviceType>().data();

  int npost = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (sendnum_direct[iswap] == 0) continue;
    MPI_Irecv(rbuf + nsize*recv_offset_reverse_atoms[iswap],nsize*sendnum_direct[iswap],
              MPI_DOUBLE,proc_direct[iswap],sendtag[iswap],world,&requests[npost++]);
  }

  // pack the ghost contributions of every swap; sum swaps with self right away

  std::vector<int> nsend_swap(ndirect,0);
  int nsendtotal = 0;

  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (recvnum_direct[iswap] == 0) continue;
    const int first = nsize*recv_offset_forward_atoms[iswap];
    DAT::tdual_double_1d k_sub =
      Kokkos::subview(k_buf_send_obj,std::make_pair(first,first+nsize*recvnum_direct[iswap]));
    nsend_swap[iswap] = obj->pack_reverse_comm_kokkos(recvnum_direct[iswap],
                                                      firstrecv_direct[iswap],k_sub);
    if (proc_direct[iswap] == me) {
      auto k_list = Kokkos::subview(k_sendatoms_list,swap2list[iswap],Kokkos::ALL);
      obj->unpack_reverse_comm_kokkos(sendnum_direct[iswap],k_list,k_sub);
    } else nsendtotal = first + nsize*recvnum_direct[iswap];
  }

  DeviceType().fence();

  if (stage && nsendtotal) {
    auto d_s = Kokkos::subview(k_buf_send_obj.view<DeviceType>(),std::make_pair(0,nsendtotal));
    auto h_s = Kokkos::subview(k_buf_send_obj.view_host(),std::make_pair(0,nsendtotal));
    Kokkos::deep_copy(h_s,d_s);
  }

  int nsendpost = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (recvnum_direct[iswap] == 0) continue;
    MPI_Isend(sbuf + nsize*recv_offset_forward_atoms[iswap],nsend_swap[iswap],MPI_DOUBLE,
              proc_direct[iswap],recvtag[iswap],world,&send_requests[nsendpost++]);
  }

  for (int ipost = 0; ipost < npost; ipost++) {
    int irecv;
    MPI_Waitany(npost,requests,&irecv,MPI_STATUS_IGNORE);
    const int iswap = send_indices_direct[irecv];
    const int first = nsize*recv_offset_reverse_atoms[iswap];
    const int last = first + nsize*sendnum_direct[iswap];
    if (stage) {
      auto d_r = Kokkos::subview(k_buf_recv_obj.view<DeviceType>(),std::make_pair(first,last));
      auto h_r = Kokkos::subview(k_buf_recv_obj.view_host(),std::make_pair(first,last));
      Kokkos::deep_copy(d_r,h_r);
    }
    DAT::tdual_double_1d k_sub = Kokkos::subview(k_buf_recv_obj,std::make_pair(first,last));
    auto k_list = Kokkos::subview(k_sendatoms_list,swap2list[iswap],Kokkos::ALL);
    obj->unpack_reverse_comm_kokkos(sendnum_direct[iswap],k_list,k_sub);
  }

  if (nsendpost) MPI_Waitall(nsendpost,send_requests,MPI_STATUS_IGNORE);

  DeviceType().fence();
}

/* ---------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_buf_obj(int n)
{
  if (n <= (int) k_buf_send_obj.extent(0)) return;
  const int nnew = static_cast<int> (BUFFACTOR * n);

  // scratch buffers, always filled before they are read, so no copy

  MemKK::realloc_kokkos(k_buf_send_obj,"comm_direct:buf_send_obj",nnew);
  MemKK::realloc_kokkos(k_buf_recv_obj,"comm_direct:buf_recv_obj",nnew);
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

/* ---------------------------------------------------------------------- */

template void CommBrickDirectKokkos::forward_comm_device<LMPHostType>(Fix *, int);
template void CommBrickDirectKokkos::reverse_comm_device<LMPHostType>(Fix *, int);
#if defined(LMP_KOKKOS_GPU)
template void CommBrickDirectKokkos::forward_comm_device<LMPDeviceType>(Fix *, int);
template void CommBrickDirectKokkos::reverse_comm_device<LMPDeviceType>(Fix *, int);
#endif
