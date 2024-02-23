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

#include "atom_vec_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "error.h"
#include "domain.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecKokkos::AtomVecKokkos(LAMMPS *lmp) : AtomVec(lmp)
{
  kokkosable = 1;
  buffer = nullptr;
  buffer_size = 0;

  no_comm_vel_flag = 0;
  no_border_vel_flag = 1;
  unpack_exchange_indices_flag = 0;
  size_exchange = 0;

  k_count = DAT::tdual_int_1d("atom:k_count",1);
  atomKK = (AtomKokkos *) atom;
}

/* ---------------------------------------------------------------------- */

AtomVecKokkos::~AtomVecKokkos()
{
  // Kokkos already deallocated host memory

  ngrow = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecKokkos_PackComm {
  typedef DeviceType device_type;

  typename AT::t_x_array_randomread _x;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d_const _list;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecKokkos_PackComm(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_xfloat_2d &k_buf,
    const typename DAT::tdual_int_1d &k_list,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
    _x(atomKK->k_x.view<DeviceType>()),_list(k_list.view<DeviceType>()),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz) {
      const size_t elements = 3;
      const size_t maxsend = (k_buf.h_view.extent(0)*
                              k_buf.h_view.extent(1))/elements;
      buffer_view<DeviceType>(_buf,k_buf,maxsend,elements);
      _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
      _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _buf(i,0) = _x(j,0) + _pbc[0]*_xprd;
        _buf(i,1) = _x(j,1) + _pbc[1]*_yprd;
        _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _buf(i,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _buf(i,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int AtomVecKokkos::pack_comm_kokkos(const int &n,
                                          const DAT::tdual_int_1d &k_list,
                                          const DAT::tdual_xfloat_2d &k_buf,
                                          const int &pbc_flag,
                                          const int* const pbc)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,X_MASK);
  if (pbc_flag) {
    if (domain->triclinic) {
      struct AtomVecKokkos_PackComm<DeviceType,1,1> f(atomKK,k_buf,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_PackComm<DeviceType,1,0> f(atomKK,k_buf,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
    }
  } else {
    if (domain->triclinic) {
      struct AtomVecKokkos_PackComm<DeviceType,0,1> f(atomKK,k_buf,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_PackComm<DeviceType,0,0> f(atomKK,k_buf,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
    }
  }

  return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecKokkos_PackCommSelf {
  typedef DeviceType device_type;

  typename AT::t_x_array_randomread _x;
  typename AT::t_x_array _xw;
  int _nfirst;
  typename AT::t_int_1d_const _list;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecKokkos_PackCommSelf(
    const AtomKokkos* atomKK,
    const int &nfirst,
    const typename DAT::tdual_int_1d &k_list,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
    _x(atomKK->k_x.view<DeviceType>()),_xw(atomKK->k_x.view<DeviceType>()),
    _nfirst(nfirst),_list(k_list.view<DeviceType>()),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz)
  {
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    if (PBC_FLAG == 0) {
      _xw(i+_nfirst,0) = _x(j,0);
      _xw(i+_nfirst,1) = _x(j,1);
      _xw(i+_nfirst,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int AtomVecKokkos::pack_comm_self(const int &n, const DAT::tdual_int_1d &k_list,
                                  const int nfirst, const int &pbc_flag,
                                  const int* const pbc)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,X_MASK);

  if (pbc_flag) {
    if (domain->triclinic) {
      struct AtomVecKokkos_PackCommSelf<DeviceType,1,1> f(atomKK,nfirst,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_PackCommSelf<DeviceType,1,0> f(atomKK,nfirst,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
    }
  } else {
    if (domain->triclinic) {
      struct AtomVecKokkos_PackCommSelf<DeviceType,0,1> f(atomKK,nfirst,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_PackCommSelf<DeviceType,0,0> f(atomKK,nfirst,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc);
      Kokkos::parallel_for(n,f);
    }
  }

  atomKK->modified(space,X_MASK);

  return n*3;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int TRICLINIC>
struct AtomVecKokkos_PackCommSelfFused {
  typedef DeviceType device_type;

  typename AT::t_x_array_randomread _x;
  typename AT::t_x_array _xw;
  typename AT::t_int_2d_const _list;
  typename AT::t_int_2d_const _pbc;
  typename AT::t_int_1d_const _pbc_flag;
  typename AT::t_int_1d_const _firstrecv;
  typename AT::t_int_1d_const _sendnum_scan;
  typename AT::t_int_1d_const _g2l;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;

  AtomVecKokkos_PackCommSelfFused(
      const AtomKokkos* atomKK,
      const typename DAT::tdual_int_2d &k_list,
      const typename DAT::tdual_int_2d &pbc,
      const typename DAT::tdual_int_1d &pbc_flag,
      const typename DAT::tdual_int_1d &firstrecv,
      const typename DAT::tdual_int_1d &sendnum_scan,
      const typename DAT::tdual_int_1d &g2l,
      const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
      const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz):
      _x(atomKK->k_x.view<DeviceType>()),_xw(atomKK->k_x.view<DeviceType>()),
      _list(k_list.view<DeviceType>()),
      _pbc(pbc.view<DeviceType>()),
      _pbc_flag(pbc_flag.view<DeviceType>()),
      _firstrecv(firstrecv.view<DeviceType>()),
      _sendnum_scan(sendnum_scan.view<DeviceType>()),
      _g2l(g2l.view<DeviceType>()),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii) const {

    int iswap = 0;
    while (ii >= _sendnum_scan[iswap]) iswap++;
    int i = ii;
    if (iswap > 0)
      i = ii - _sendnum_scan[iswap-1];

    const int _nfirst = _firstrecv[iswap];
    const int nlocal = _firstrecv[0];

    int j = _list(iswap,i);
    if (j >= nlocal)
      j = _g2l(j-nlocal);

    if (_pbc_flag(ii) == 0) {
      _xw(i+_nfirst,0) = _x(j,0);
      _xw(i+_nfirst,1) = _x(j,1);
      _xw(i+_nfirst,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc(ii,0)*_xprd;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc(ii,1)*_yprd;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc(ii,2)*_zprd;
      } else {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc(ii,0)*_xprd + _pbc(ii,5)*_xy + _pbc(ii,4)*_xz;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc(ii,1)*_yprd + _pbc(ii,3)*_yz;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc(ii,2)*_zprd;
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int AtomVecKokkos::pack_comm_self_fused(const int &n, const DAT::tdual_int_2d &k_list,
                                        const DAT::tdual_int_1d &sendnum_scan,
                                        const DAT::tdual_int_1d &firstrecv,
                                        const DAT::tdual_int_1d &pbc_flag,
                                        const DAT::tdual_int_2d &pbc,
                                        const DAT::tdual_int_1d &g2l)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,X_MASK);

  if (domain->triclinic) {
    struct AtomVecKokkos_PackCommSelfFused<DeviceType,1> f(
      atomKK,k_list,pbc,pbc_flag,
      firstrecv,sendnum_scan,g2l,
      domain->xprd,domain->yprd,domain->zprd,
      domain->xy,domain->xz,domain->yz);
    Kokkos::parallel_for(n,f);
  } else {
    struct AtomVecKokkos_PackCommSelfFused<DeviceType,0> f(
      atomKK,k_list,pbc,pbc_flag,
      firstrecv,sendnum_scan,g2l,
      domain->xprd,domain->yprd,domain->zprd,
      domain->xy,domain->xz,domain->yz);
    Kokkos::parallel_for(n,f);
  }

  atomKK->modified(Device,X_MASK);

  return n*3;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_UnpackComm {
  typedef DeviceType device_type;

  typename AT::t_x_array _x;
  typename AT::t_xfloat_2d_const _buf;
  int _first;

  AtomVecKokkos_UnpackComm(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_xfloat_2d &k_buf,
    const int& first):
    _x(atomKK->k_x.view<DeviceType>()),
    _buf(k_buf.view<DeviceType>()),
    _first(first) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void AtomVecKokkos::unpack_comm_kokkos(const int &n, const int &first,
                                       const DAT::tdual_xfloat_2d &k_buf)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,X_MASK);
  struct AtomVecKokkos_UnpackComm<DeviceType> f(atomKK,buf,first);
  Kokkos::parallel_for(n,f);

  atomKK->modified(Device,X_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC,int DEFORM_VREMAP>
struct AtomVecKokkos_PackCommVel {
  typedef DeviceType device_type;

  typename AT::t_x_array_randomread _x;
  typename AT::t_int_1d _mask;
  typename AT::t_v_array _v;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d_const _list;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];
  X_FLOAT _h_rate[6];
  const int _deform_vremap;

  AtomVecKokkos_PackCommVel(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_xfloat_2d &k_buf,
    const typename DAT::tdual_int_1d &k_list,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc,
    const double * const h_rate,
    const int &deform_vremap):
    _x(atomKK->k_x.view<DeviceType>()),
    _mask(atomKK->k_mask.view<DeviceType>()),
    _v(atomKK->k_v.view<DeviceType>()),
    _list(k_list.view<DeviceType>()),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz),
    _deform_vremap(deform_vremap)
  {
    const size_t elements = 6;
    const int maxsend = (buf.template view<DeviceType>().extent(0)*
                         buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,k_buf,maxsend,elements);
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
    _h_rate[0] = h_rate[0]; _h_rate[1] = h_rate[1]; _h_rate[2] = h_rate[2];
    _h_rate[3] = h_rate[3]; _h_rate[4] = h_rate[4]; _h_rate[5] = h_rate[5];
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
      _buf(i,3) = _v(j,0);
      _buf(i,4) = _v(j,1);
      _buf(i,5) = _v(j,2);
    } else {
      if (TRICLINIC == 0) {
      _buf(i,0) = _x(j,0) + _pbc[0]*_xprd;
      _buf(i,1) = _x(j,1) + _pbc[1]*_yprd;
      _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
           } else {
      _buf(i,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
      _buf(i,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
      _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
      }

      if (DEFORM_VREMAP == 0) {
        _buf(i,3) = _v(j,0);
        _buf(i,4) = _v(j,1);
        _buf(i,5) = _v(j,2);
      } else {
        if (_mask(i) & _deform_vremap) {
          _buf(i,3) = _v(j,0) + _pbc[0]*_h_rate[0] + _pbc[5]*_h_rate[5] + _pbc[4]*_h_rate[4];
          _buf(i,4) = _v(j,1) + _pbc[1]*_h_rate[1] + _pbc[3]*_h_rate[3];
          _buf(i,5) = _v(j,2) + _pbc[2]*_h_rate[2];
        } else {
          _buf(i,3) = _v(j,0);
          _buf(i,4) = _v(j,1);
          _buf(i,5) = _v(j,2);
        }
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int AtomVecKokkos::pack_comm_vel_kokkos(const int &n, const DAT::tdual_int_1d &k_list,
                                        const DAT::tdual_xfloat_2d &k_buf, const int &pbc_flag,
                                        const int* const pbc)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,X_MASK|V_MASK|MASK_MASK);

  if (pbc_flag) {
    if (deform_vremap) {
      if (domain->triclinic) {
        struct AtomVecKokkos_PackCommVel<DeviceType,1,1,1> f(
          atomKK,k_buf,k_list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackCommVel<DeviceType,1,0,1> f(
          atomKK,k_buf,k_list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecKokkos_PackCommVel<DeviceType,1,1,0> f(
          atomKK,k_buf,k_list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackCommVel<DeviceType,1,0,0> f(
          atomKK,k_buf,k_list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    if (domain->triclinic) {
      struct AtomVecKokkos_PackCommVel<DeviceType,0,1,0> f(
        atomKK,k_buf,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_PackCommVel<DeviceType,0,0,0> f(
        atomKK,k_buf,k_list,
        domain->xprd,domain->yprd,domain->zprd,
        domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
      Kokkos::parallel_for(n,f);
    }
  }

  return n*6;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_UnpackCommVel {
  typedef DeviceType device_type;

  typename AT::t_x_array _x;
  typename AT::t_v_array _v;
  typename AT::t_xfloat_2d_const _buf;
  int _first;

  AtomVecKokkos_UnpackCommVel(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_xfloat_2d &k_buf,
    const int& first):
    _x(atomKK->k_x.view<DeviceType>()),
    _v(atomKK->k_v.view<DeviceType>()),
    _first(first)
  {
    const size_t elements = 6;
    const int maxsend = (buf.template view<DeviceType>().extent(0)*
                         buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,k_buf,maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _v(i+_first,0) = _buf(i,3);
    _v(i+_first,1) = _buf(i,4);
    _v(i+_first,2) = _buf(i,5);
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void AtomVecKokkos::unpack_comm_vel_kokkos(const int &n, const int &first,
                                           const DAT::tdual_xfloat_2d &k_buf)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,X_MASK|V_MASK);

  struct AtomVecKokkos_UnpackCommVel<DeviceType> f(atomKK,k_buf,first);
  Kokkos::parallel_for(n,f);

  atomKK->modified(space,X_MASK|V_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_PackReverse {
  typedef DeviceType device_type;

  typename AT::t_f_array_randomread _f;
  typename AT::t_ffloat_2d _buf;
  int _first;

  AtomVecKokkos_PackReverse(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_ffloat_2d &k_buf,
    const int& first):
    _f(atomKK->k_f.view<DeviceType>()),
    _buf(k_buf.view<DeviceType>()),_first(first) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _buf(i,0) = _f(i+_first,0);
    _buf(i,1) = _f(i+_first,1);
    _buf(i,2) = _f(i+_first,2);
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int AtomVecKokkos::pack_reverse_kokkos(const int &n, const int &first,
                                       const DAT::tdual_ffloat_2d &k_buf)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,F_MASK);

  struct AtomVecKokkos_PackReverse<DeviceType> f(atomKK,k_buf,first);
  Kokkos::parallel_for(n,f);

  return n*size_reverse;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_UnPackReverseSelf {
  typedef DeviceType device_type;

  typename AT::t_f_array_randomread _f;
  typename AT::t_f_array _fw;
  int _nfirst;
  typename AT::t_int_1d_const _list;

  AtomVecKokkos_UnPackReverseSelf(
    const AtomKokkos* atomKK,
    const int &nfirst,
    const typename DAT::tdual_int_1d &k_list):
    _f(atomKK->k_f.view<DeviceType>()),_fw(atomKK->k_f.view<DeviceType>()),
    _nfirst(nfirst),_list(k_list.view<DeviceType>()) {
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    _fw(j,0) += _f(i+_nfirst,0);
    _fw(j,1) += _f(i+_nfirst,1);
    _fw(j,2) += _f(i+_nfirst,2);
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int AtomVecKokkos::unpack_reverse_self(const int &n, const DAT::tdual_int_1d &k_list,
                                       const int nfirst)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,F_MASK);

  struct AtomVecKokkos_UnPackReverseSelf<DeviceType> f(atomKK,nfirst,k_list);
  Kokkos::parallel_for(n,f);

  atomKK->modified(space,F_MASK);

  return n*3;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecKokkos_UnPackReverse {
  typedef DeviceType device_type;

  typename AT::t_f_array _f;
  typename AT::t_ffloat_2d_const _buf;
  typename AT::t_int_1d_const _list;

  AtomVecKokkos_UnPackReverse(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_ffloat_2d &k_buf,
    const typename DAT::tdual_int_1d &k_list):
    _f(atomKK->k_f.view<DeviceType>()),
    _buf(k_buf.view<DeviceType>())
    _list(k_list.view<DeviceType>()) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    _f(j,0) += _buf(i,0);
    _f(j,1) += _buf(i,1);
    _f(j,2) += _buf(i,2);
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void AtomVecKokkos::unpack_reverse_kokkos(const int &n,
                                          const DAT::tdual_int_1d &k_list,
                                          const DAT::tdual_ffloat_2d &k_buf)
{
  auto space = ExecutionSpaceFromDevice<DeviceType>::space;

  atomKK->sync(space,F_MASK);

  struct AtomVecKokkos_UnPackReverse<DeviceType> f(atomKK,k_buf,k_list);
  Kokkos::parallel_for(n,f);

  atomKK->modified(space,F_MASK);
}
