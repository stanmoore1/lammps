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
#include "domain.h"
#include "error.h"
#include "kokkos.h"
#include "memory_kokkos.h"

#include <utility>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecKokkos::AtomVecKokkos(LAMMPS *lmp) : AtomVec(lmp)
{
  if (!lmp->kokkos || !lmp->kokkos->kokkos_exists)
    error->all(FLERR, Error::NOLASTLINE, "Cannot use KOKKOS styles without enabling KOKKOS");

  kokkosable = 1;
  buffer = nullptr;
  buffer_size = 0;
  size_exchange = size_exchange_default = size_exchange_bonus = 0;

  datamask_grow = datamask_comm = datamask_comm_vel = datamask_reverse =
    datamask_border = datamask_border_vel = datamask_exchange =
    datamask_bonus = EMPTY_MASK;

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

template<class DeviceType,int PBC_FLAG,int TRICLINIC,int DEFAULT>
struct AtomVecKokkos_PackComm {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr_randomread _x;
  typename AT::t_kkfloat_1d_4_randomread _mu;
  typename AT::t_kkfloat_1d_4_randomread _sp;
  typename AT::t_kkfloat_1d_randomread _dpdTheta,_uCond,_uMech,_uChem;
  typename AT::t_double_2d_lr_um _buf;
  typename AT::t_int_1d_const _list;
  double _xprd,_yprd,_zprd,_xy,_xz,_yz;
  double _pbc[6];
  uint64_t _datamask;

  AtomVecKokkos_PackComm(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_double_2d_lr &buf,
    const typename DAT::tdual_int_1d &list,
    const double &xprd, const double &yprd, const double &zprd,
    const double &xy, const double &xz, const double &yz, const int* const pbc,
    const uint64_t &datamask):
    _x(atomKK->k_x.view<DeviceType>()),
    _mu(atomKK->k_mu.view<DeviceType>()),
    _sp(atomKK->k_sp.view<DeviceType>()),
    _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
    _uCond(atomKK->k_uCond.view<DeviceType>()),
    _uMech(atomKK->k_uMech.view<DeviceType>()),
    _uChem(atomKK->k_uChem.view<DeviceType>()),
    _list(list.view<DeviceType>()),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz),_datamask(datamask) {
      const int size_forward = atomKK->avecKK->size_forward;
      const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/size_forward;
      const size_t elements = size_forward;
      buffer_view<DeviceType>(_buf,buf,maxsend,elements);
      _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
      _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
    };

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    int m = 0;
    if constexpr (PBC_FLAG == 0) {
      _buf(i,m++) = _x(j,0);
      _buf(i,m++) = _x(j,1);
      _buf(i,m++) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _buf(i,m++) = _x(j,0) + _pbc[0]*_xprd;
        _buf(i,m++) = _x(j,1) + _pbc[1]*_yprd;
        _buf(i,m++) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _buf(i,m++) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _buf(i,m++) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _buf(i,m++) = _x(j,2) + _pbc[2]*_zprd;
      }
    }

    if constexpr (!DEFAULT) {

      // DIPOLE package

      if (_datamask & MU_MASK) {
        _buf(i,m++) = _mu(j,0);
        _buf(i,m++) = _mu(j,1);
        _buf(i,m++) = _mu(j,2);
      }

      // SPIN package

      if (_datamask & SP_MASK) {
        _buf(i,m++) = _sp(j,0);
        _buf(i,m++) = _sp(j,1);
        _buf(i,m++) = _sp(j,2);
        _buf(i,m++) = _sp(j,3);
      }

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _buf(i,m++) = _dpdTheta(j);
        _buf(i,m++) = _uCond(j);
        _buf(i,m++) = _uMech(j);
        _buf(i,m++) = _uChem(j);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_comm_kokkos(const int &n,
                                          const DAT::tdual_int_1d &list,
                                          const DAT::tdual_double_2d_lr &buf,
                                          const int &pbc_flag,
                                          const int* const pbc)
{
  // Check whether to always run forward communication on the host
  // Choose correct forward PackComm kernel

  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,datamask_comm);
    if (pbc_flag) {
      if (domain->triclinic) {
        if (comm_x_only) {
          struct AtomVecKokkos_PackComm<LMPHostType,1,1,1> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackComm<LMPHostType,1,1,0> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (comm_x_only) {
          struct AtomVecKokkos_PackComm<LMPHostType,1,0,1> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackComm<LMPHostType,1,0,0> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
        if (comm_x_only) {
          struct AtomVecKokkos_PackComm<LMPHostType,0,1,1> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackComm<LMPHostType,0,1,0> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (comm_x_only) {
          struct AtomVecKokkos_PackComm<LMPHostType,0,0,1> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackComm<LMPHostType,0,0,0> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      }
    }
  } else {
    atomKK->sync(Device,datamask_comm);
    if (pbc_flag) {
      if (domain->triclinic) {
        if (comm_x_only) {
          struct AtomVecKokkos_PackComm<LMPDeviceType,1,1,1> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackComm<LMPDeviceType,1,1,0> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (comm_x_only) {
          struct AtomVecKokkos_PackComm<LMPDeviceType,1,0,1> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackComm<LMPDeviceType,1,0,0> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
        if (comm_x_only) {
          struct AtomVecKokkos_PackComm<LMPDeviceType,0,1,1> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackComm<LMPDeviceType,0,1,0> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (comm_x_only) {
          struct AtomVecKokkos_PackComm<LMPDeviceType,0,0,1> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackComm<LMPDeviceType,0,0,0> f(atomKK,buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      }
    }
  }

  if (bonus_flag) pack_comm_bonus_kokkos(n, list, buf);

  return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int DEFAULT>
struct AtomVecKokkos_UnpackComm {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_kkfloat_1d_4 _mu;
  typename AT::t_kkfloat_1d_4 _sp;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem;
  typename AT::t_double_2d_lr_const _buf;
  int _first;
  uint64_t _datamask;

  AtomVecKokkos_UnpackComm(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_double_2d_lr &buf,
    const int &first, const uint64_t &datamask):
      _x(atomKK->k_x.view<DeviceType>()),
      _mu(atomKK->k_mu.view<DeviceType>()),
      _sp(atomKK->k_sp.view<DeviceType>()),
      _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
      _uCond(atomKK->k_uCond.view<DeviceType>()),
      _uMech(atomKK->k_uMech.view<DeviceType>()),
      _uChem(atomKK->k_uChem.view<DeviceType>()),
      _first(first),_datamask(datamask) {
        const int size_forward = atomKK->avecKK->size_forward;
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/size_forward;
        const size_t elements = size_forward;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
      };

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    int m = 0;
    _x(i+_first,0) = _buf(i,m++);
    _x(i+_first,1) = _buf(i,m++);
    _x(i+_first,2) = _buf(i,m++);

    if constexpr (!DEFAULT) {

      // DIPOLE package

      if (_datamask & MU_MASK) {
        _mu(i+_first,0) = _buf(i,m++);
        _mu(i+_first,1) = _buf(i,m++);
        _mu(i+_first,2) = _buf(i,m++);
      }

      // SPIN package

      if (_datamask & SP_MASK) {
        _sp(i+_first,0) = _buf(i,m++);
        _sp(i+_first,1) = _buf(i,m++);
        _sp(i+_first,2) = _buf(i,m++);
        _sp(i+_first,3) = _buf(i,m++);
      }

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _dpdTheta(i+_first) = _buf(i,m++);
        _uCond(i+_first) = _buf(i,m++);
        _uMech(i+_first) = _buf(i,m++);
        _uChem(i+_first) = _buf(i,m++);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_comm_kokkos(const int &n, const int &first,
    const DAT::tdual_double_2d_lr &buf) {
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,datamask_comm);
    if (comm_x_only) {
      struct AtomVecKokkos_UnpackComm<LMPHostType,1> f(atomKK,buf,first,datamask_comm);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnpackComm<LMPHostType,0> f(atomKK,buf,first,datamask_comm);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(HostKK,datamask_comm);
  } else {
    atomKK->sync(Device,datamask_comm);
    if (comm_x_only) {
      struct AtomVecKokkos_UnpackComm<LMPDeviceType,1> f(atomKK,buf,first,datamask_comm);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnpackComm<LMPDeviceType,0> f(atomKK,buf,first,datamask_comm);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(Device,datamask_comm);
  }

  if (bonus_flag) unpack_comm_bonus_kokkos(n, first, buf);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC,int DEFAULT>
struct AtomVecKokkos_PackCommSelf {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_kkfloat_1d_4 _mu;
  typename AT::t_kkfloat_1d_4 _sp;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem;
  int _nfirst;
  typename AT::t_int_1d_const _list;
  double _xprd,_yprd,_zprd,_xy,_xz,_yz;
  double _pbc[6];
  uint64_t _datamask;

  AtomVecKokkos_PackCommSelf(
    const AtomKokkos* atomKK,
    const int &nfirst,
    const typename DAT::tdual_int_1d &list,
    const double &xprd, const double &yprd, const double &zprd,
    const double &xy, const double &xz, const double &yz, const int* const pbc,
    const uint64_t datamask):
    _x(atomKK->k_x.view<DeviceType>()),
    _mu(atomKK->k_mu.view<DeviceType>()),
    _sp(atomKK->k_sp.view<DeviceType>()),
    _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
    _uCond(atomKK->k_uCond.view<DeviceType>()),
    _uMech(atomKK->k_uMech.view<DeviceType>()),
    _uChem(atomKK->k_uChem.view<DeviceType>()),
    _nfirst(nfirst),_list(list.view<DeviceType>()),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz),_datamask(datamask) {
      _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
      _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    if constexpr (PBC_FLAG == 0) {
      _x(i+_nfirst,0) = _x(j,0);
      _x(i+_nfirst,1) = _x(j,1);
      _x(i+_nfirst,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _x(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd;
        _x(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd;
        _x(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _x(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _x(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _x(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
      }
    }

    if constexpr (!DEFAULT) {

      // DIPOLE package

      if (_datamask & MU_MASK) {
        _mu(i+_nfirst,0) = _mu(j,0);
        _mu(i+_nfirst,1) = _mu(j,1);
        _mu(i+_nfirst,2) = _mu(j,2);
      }

      // SPIN package

      if (_datamask & SP_MASK) {
        _sp(i+_nfirst,0) = _sp(j,0);
        _sp(i+_nfirst,1) = _sp(j,1);
        _sp(i+_nfirst,2) = _sp(j,2);
        _sp(i+_nfirst,3) = _sp(j,3);
      }

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _dpdTheta(i+_nfirst) = _dpdTheta(j);
        _uCond(i+_nfirst) = _uCond(j);
        _uMech(i+_nfirst) = _uMech(j);
        _uChem(i+_nfirst) = _uChem(j);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_comm_self_kokkos(const int &n, const DAT::tdual_int_1d &list,
                                         const int nfirst, const int &pbc_flag, const int* const pbc) {
  // Check whether to always run forward communication on the host
  // Choose correct forward PackComm kernel

  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,datamask_comm);
    if (pbc_flag) {
      if (domain->triclinic) {
        if (comm_x_only) {
          struct AtomVecKokkos_PackCommSelf<LMPHostType,1,1,1> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommSelf<LMPHostType,1,1,0> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (comm_x_only) {
          struct AtomVecKokkos_PackCommSelf<LMPHostType,1,0,1> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommSelf<LMPHostType,1,0,0> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
        if (comm_x_only) {
          struct AtomVecKokkos_PackCommSelf<LMPHostType,0,1,1> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommSelf<LMPHostType,0,1,0> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (comm_x_only) {
          struct AtomVecKokkos_PackCommSelf<LMPHostType,0,0,1> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommSelf<LMPHostType,0,0,0> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      }
    }
    atomKK->modified(HostKK,datamask_comm);
  } else {
    atomKK->sync(Device,datamask_comm);
    if (pbc_flag) {
      if (domain->triclinic) {
        if (comm_x_only) {
          struct AtomVecKokkos_PackCommSelf<LMPDeviceType,1,1,1> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommSelf<LMPDeviceType,1,1,0> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (comm_x_only) {
          struct AtomVecKokkos_PackCommSelf<LMPDeviceType,1,0,1> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommSelf<LMPDeviceType,1,0,0> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
        if (comm_x_only) {
          struct AtomVecKokkos_PackCommSelf<LMPDeviceType,0,1,1> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommSelf<LMPDeviceType,0,1,0> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (comm_x_only) {
          struct AtomVecKokkos_PackCommSelf<LMPDeviceType,0,0,1> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommSelf<LMPDeviceType,0,0,0> f(atomKK,nfirst,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,datamask_comm);
          Kokkos::parallel_for(n,f);
        }
      }
    }
    atomKK->modified(Device,datamask_comm);
  }

  if (bonus_flag) pack_comm_self_bonus_kokkos(n, list, nfirst);

  return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int TRICLINIC,int DEFAULT>
struct AtomVecKokkos_PackCommSelfFused {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_kkfloat_1d_4 _mu;
  typename AT::t_kkfloat_1d_4 _sp;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem;
  typename AT::t_int_2d_lr_const _list;
  typename AT::t_int_2d_const _pbc;
  typename AT::t_int_1d_const _pbc_flag;
  typename AT::t_int_1d_const _firstrecv;
  typename AT::t_int_1d_const _sendnum_scan;
  typename AT::t_int_1d_const _g2l;
  double _xprd,_yprd,_zprd,_xy,_xz,_yz;
  uint64_t _datamask;

  AtomVecKokkos_PackCommSelfFused(
      const AtomKokkos* atomKK,
      const typename DAT::tdual_int_2d_lr &list,
      const typename DAT::tdual_int_2d &pbc,
      const typename DAT::tdual_int_1d &pbc_flag,
      const typename DAT::tdual_int_1d &firstrecv,
      const typename DAT::tdual_int_1d &sendnum_scan,
      const typename DAT::tdual_int_1d &g2l,
      const double &xprd, const double &yprd, const double &zprd,
      const double &xy, const double &xz, const double &yz,
      const uint64_t datamask):
      _x(atomKK->k_x.view<DeviceType>()),
      _mu(atomKK->k_mu.view<DeviceType>()),
      _sp(atomKK->k_sp.view<DeviceType>()),
      _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
      _uCond(atomKK->k_uCond.view<DeviceType>()),
      _uMech(atomKK->k_uMech.view<DeviceType>()),
      _uChem(atomKK->k_uChem.view<DeviceType>()),
      _list(list.view<DeviceType>()),
      _pbc(pbc.view<DeviceType>()),
      _pbc_flag(pbc_flag.view<DeviceType>()),
      _firstrecv(firstrecv.view<DeviceType>()),
      _sendnum_scan(sendnum_scan.view<DeviceType>()),
      _g2l(g2l.view<DeviceType>()),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz),_datamask(datamask) {};

// NOLINTNEXTLINE
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
      _x(i+_nfirst,0) = _x(j,0);
      _x(i+_nfirst,1) = _x(j,1);
      _x(i+_nfirst,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _x(i+_nfirst,0) = _x(j,0) + _pbc(ii,0)*_xprd;
        _x(i+_nfirst,1) = _x(j,1) + _pbc(ii,1)*_yprd;
        _x(i+_nfirst,2) = _x(j,2) + _pbc(ii,2)*_zprd;
      } else {
        _x(i+_nfirst,0) = _x(j,0) + _pbc(ii,0)*_xprd + _pbc(ii,5)*_xy + _pbc(ii,4)*_xz;
        _x(i+_nfirst,1) = _x(j,1) + _pbc(ii,1)*_yprd + _pbc(ii,3)*_yz;
        _x(i+_nfirst,2) = _x(j,2) + _pbc(ii,2)*_zprd;
      }
    }

    if constexpr (!DEFAULT) {

      // DIPOLE package

      if (_datamask & MU_MASK) {
        _mu(i+_nfirst,0) = _mu(j,0);
        _mu(i+_nfirst,1) = _mu(j,1);
        _mu(i+_nfirst,2) = _mu(j,2);
      }

      // SPIN package

      if (_datamask & SP_MASK) {
        _sp(i+_nfirst,0) = _sp(j,0);
        _sp(i+_nfirst,1) = _sp(j,1);
        _sp(i+_nfirst,2) = _sp(j,2);
        _sp(i+_nfirst,3) = _sp(j,3);
      }

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _dpdTheta(i+_nfirst) = _dpdTheta(j);
        _uCond(i+_nfirst) = _uCond(j);
        _uMech(i+_nfirst) = _uMech(j);
        _uChem(i+_nfirst) = _uChem(j);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_comm_self_fused_kokkos(const int &n,
                                               const DAT::tdual_int_2d_lr &list,
                                               const DAT::tdual_int_1d &sendnum_scan,
                                               const DAT::tdual_int_1d &firstrecv,
                                               const DAT::tdual_int_1d &pbc_flag,
                                               const DAT::tdual_int_2d &pbc,
                                               const DAT::tdual_int_1d &g2l) {
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,datamask_comm);
    if (domain->triclinic) {
      if (comm_x_only) {
        struct AtomVecKokkos_PackCommSelfFused<LMPHostType,1,1> f(atomKK,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,datamask_comm);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackCommSelfFused<LMPHostType,1,0> f(atomKK,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,datamask_comm);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (comm_x_only) {
        struct AtomVecKokkos_PackCommSelfFused<LMPHostType,0,1> f(atomKK,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,datamask_comm);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackCommSelfFused<LMPHostType,0,0> f(atomKK,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,datamask_comm);
        Kokkos::parallel_for(n,f);
      }
    }
    atomKK->modified(HostKK,datamask_comm);
  } else {
    atomKK->sync(Device,datamask_comm);
    if (domain->triclinic) {
      if (comm_x_only) {
        struct AtomVecKokkos_PackCommSelfFused<LMPDeviceType,1,1> f(atomKK,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,datamask_comm);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackCommSelfFused<LMPDeviceType,1,0> f(atomKK,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,datamask_comm);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (comm_x_only) {
        struct AtomVecKokkos_PackCommSelfFused<LMPDeviceType,0,1> f(atomKK,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,datamask_comm);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackCommSelfFused<LMPDeviceType,0,0> f(atomKK,list,pbc,pbc_flag,firstrecv,sendnum_scan,g2l,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,datamask_comm);
        Kokkos::parallel_for(n,f);
      }
    }
    atomKK->modified(Device,datamask_comm);
  }

  if (bonus_flag) pack_comm_self_fused_bonus_kokkos(n,list,sendnum_scan,
                                                    firstrecv,g2l);

  return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC,int DEFORM_VREMAP>
struct AtomVecKokkos_PackCommVel {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr_randomread _x;
  typename AT::t_int_1d_randomread _mask;
  typename AT::t_kkfloat_1d_3_randomread _v;
  typename AT::t_kkfloat_1d_3_randomread _angmom;
  typename AT::t_kkfloat_1d_4_randomread _mu;
  typename AT::t_kkfloat_1d_4_randomread _sp;
  typename AT::t_kkfloat_1d_3_randomread _omega;
  typename AT::t_kkfloat_1d_randomread _dpdTheta,_uCond,_uMech,_uChem;
  typename AT::t_double_2d_lr_um _buf;
  typename AT::t_int_1d_const _list;
  double _xprd,_yprd,_zprd,_xy,_xz,_yz;
  double _pbc[6];
  double _h_rate[6];
  const int _deform_vremap;
  uint64_t _datamask;

  AtomVecKokkos_PackCommVel(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_double_2d_lr &buf,
    const typename DAT::tdual_int_1d &list,
    const double &xprd, const double &yprd, const double &zprd,
    const double &xy, const double &xz, const double &yz, const int* const pbc,
    const double * const h_rate,
    const int &deform_vremap,
    const uint64_t &datamask):
    _x(atomKK->k_x.view<DeviceType>()),
    _mask(atomKK->k_mask.view<DeviceType>()),
    _v(atomKK->k_v.view<DeviceType>()),
    _angmom(atomKK->k_angmom.view<DeviceType>()),
    _omega(atomKK->k_omega.view<DeviceType>()),
    _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
    _uCond(atomKK->k_uCond.view<DeviceType>()),
    _uMech(atomKK->k_uMech.view<DeviceType>()),
    _uChem(atomKK->k_uChem.view<DeviceType>()),
    _list(list.view<DeviceType>()),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz),
    _deform_vremap(deform_vremap),
    _datamask(datamask)
  {
    const size_t elements = atomKK->avecKK->size_forward + atomKK->avecKK->size_velocity;
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,buf,maxsend,elements);
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
    _h_rate[0] = h_rate[0]; _h_rate[1] = h_rate[1]; _h_rate[2] = h_rate[2];
    _h_rate[3] = h_rate[3]; _h_rate[4] = h_rate[4]; _h_rate[5] = h_rate[5];
  }

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    int m = 0;
    const int j = _list(i);
    if constexpr (PBC_FLAG == 0) {
      _buf(i,m++) = _x(j,0);
      _buf(i,m++) = _x(j,1);
      _buf(i,m++) = _x(j,2);
      _buf(i,m++) = _v(j,0);
      _buf(i,m++) = _v(j,1);
      _buf(i,m++) = _v(j,2);
    } else {
      if (TRICLINIC == 0) {
        _buf(i,m++) = _x(j,0) + _pbc[0]*_xprd;
        _buf(i,m++) = _x(j,1) + _pbc[1]*_yprd;
        _buf(i,m++) = _x(j,2) + _pbc[2]*_zprd;
             } else {
        _buf(i,m++) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _buf(i,m++) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _buf(i,m++) = _x(j,2) + _pbc[2]*_zprd;
      }

      if constexpr (DEFORM_VREMAP == 0) {
        _buf(i,m++) = _v(j,0);
        _buf(i,m++) = _v(j,1);
        _buf(i,m++) = _v(j,2);
      } else {
        if (_mask(i) & _deform_vremap) {
          _buf(i,m++) = _v(j,0) + _pbc[0]*_h_rate[0] + _pbc[5]*_h_rate[5] + _pbc[4]*_h_rate[4];
          _buf(i,m++) = _v(j,1) + _pbc[1]*_h_rate[1] + _pbc[3]*_h_rate[3];
          _buf(i,m++) = _v(j,2) + _pbc[2]*_h_rate[2];
        } else {
          _buf(i,m++) = _v(j,0);
          _buf(i,m++) = _v(j,1);
          _buf(i,m++) = _v(j,2);
        }
      }
    }

    // angmom: included for ellipsoid

    if (_datamask & ANGMOM_MASK) {
      _buf(i,m++) = _angmom(j,0);
      _buf(i,m++) = _angmom(j,1);
      _buf(i,m++) = _angmom(j,2);
    }

    // DIPOLE package

    if (_datamask & MU_MASK) {
      _buf(i,m++) = _mu(j,0);
      _buf(i,m++) = _mu(j,1);
      _buf(i,m++) = _mu(j,2);
    }

    // SPIN package

    if (_datamask & SP_MASK) {
      _buf(i,m++) = _sp(j,0);
      _buf(i,m++) = _sp(j,1);
      _buf(i,m++) = _sp(j,2);
      _buf(i,m++) = _sp(j,3);
    }

    // SPHERE package

    if (_datamask & OMEGA_MASK) {
      _buf(i,m++) = _omega(j,0);
      _buf(i,m++) = _omega(j,1);
      _buf(i,m++) = _omega(j,2);
    }

      // DPD-REACT package

    if (_datamask & DPDTHETA_MASK) {
      _buf(i,m++) = _dpdTheta(j);
      _buf(i,m++) = _uCond(j);
      _buf(i,m++) = _uMech(j);
      _buf(i,m++) = _uChem(j);
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_comm_vel_kokkos(
  const int &n,
  const DAT::tdual_int_1d &list,
  const DAT::tdual_double_2d_lr &buf,
  const int &pbc_flag,
  const int* const pbc)
{
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,datamask_comm_vel);
    if (pbc_flag) {
      if (deform_vremap) {
        if (domain->triclinic) {
          struct AtomVecKokkos_PackCommVel<LMPHostType,1,1,1> f(
            atomKK,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
            datamask_comm_vel);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommVel<LMPHostType,1,0,1> f(
            atomKK,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
            datamask_comm_vel);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (domain->triclinic) {
          struct AtomVecKokkos_PackCommVel<LMPHostType,1,1,0> f(
            atomKK,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
            datamask_comm_vel);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommVel<LMPHostType,1,0,0> f(
            atomKK,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
            datamask_comm_vel);
          Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecKokkos_PackCommVel<LMPHostType,0,1,0> f(
          atomKK,
          buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
          datamask_comm_vel);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackCommVel<LMPHostType,0,0,0> f(
          atomKK,
          buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
          datamask_comm_vel);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    atomKK->sync(Device,datamask_comm_vel);
    if (pbc_flag) {
      if (deform_vremap) {
        if (domain->triclinic) {
          struct AtomVecKokkos_PackCommVel<LMPDeviceType,1,1,1> f(
            atomKK,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
            datamask_comm_vel);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommVel<LMPDeviceType,1,0,1> f(
            atomKK,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
            datamask_comm_vel);
          Kokkos::parallel_for(n,f);
        }
      } else {
        if (domain->triclinic) {
          struct AtomVecKokkos_PackCommVel<LMPDeviceType,1,1,0> f(
            atomKK,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
            datamask_comm_vel);
          Kokkos::parallel_for(n,f);
        } else {
          struct AtomVecKokkos_PackCommVel<LMPDeviceType,1,0,0> f(
            atomKK,
            buf,list,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
            datamask_comm_vel);
          Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecKokkos_PackCommVel<LMPDeviceType,0,1,0> f(
          atomKK,
          buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
          datamask_comm_vel);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecKokkos_PackCommVel<LMPDeviceType,0,0,0> f(
          atomKK,
          buf,list,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap,
          datamask_comm_vel);
        Kokkos::parallel_for(n,f);
      }
    }
  }

  if (bonus_flag) pack_comm_bonus_kokkos(n, list, buf, 1);

  return n*(size_forward + size_velocity);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int DEFAULT>
struct AtomVecKokkos_UnpackCommVel {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_kkfloat_1d_3 _v;
  typename AT::t_kkfloat_1d_3 _angmom;
  typename AT::t_kkfloat_1d_4 _mu;
  typename AT::t_kkfloat_1d_4 _sp;
  typename AT::t_kkfloat_1d_3 _omega;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem;
  typename AT::t_double_2d_lr_const _buf;
  int _first;
  uint64_t _datamask;

  AtomVecKokkos_UnpackCommVel(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_double_2d_lr &buf,
    const int &first, const uint64_t &datamask):
    _x(atomKK->k_x.view<DeviceType>()),
    _v(atomKK->k_v.view<DeviceType>()),
    _angmom(atomKK->k_angmom.view<DeviceType>()),
    _mu(atomKK->k_mu.view<DeviceType>()),
    _sp(atomKK->k_sp.view<DeviceType>()),
    _omega(atomKK->k_omega.view<DeviceType>()),
    _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
    _uCond(atomKK->k_uCond.view<DeviceType>()),
    _uMech(atomKK->k_uMech.view<DeviceType>()),
    _uChem(atomKK->k_uChem.view<DeviceType>()),
    _first(first),_datamask(datamask)
  {
    const size_t elements = atomKK->avecKK->size_forward + atomKK->avecKK->size_velocity;
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,buf,maxsend,elements);
  };

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    int m = 0;
    _x(i+_first,0) = _buf(i,m++);
    _x(i+_first,1) = _buf(i,m++);
    _x(i+_first,2) = _buf(i,m++);
    _v(i+_first,0) = _buf(i,m++);
    _v(i+_first,1) = _buf(i,m++);
    _v(i+_first,2) = _buf(i,m++);

    if constexpr (!DEFAULT) {

      // angmom: included for ellipsoid

      if (_datamask & ANGMOM_MASK) {
        _angmom(i+_first,0) = _buf(i,m++);
        _angmom(i+_first,1) = _buf(i,m++);
        _angmom(i+_first,2) = _buf(i,m++);
      }

      // DIPOLE package

      if (_datamask & MU_MASK) {
        _mu(i+_first,0) = _buf(i,m++);
        _mu(i+_first,1) = _buf(i,m++);
        _mu(i+_first,2) = _buf(i,m++);
      }

      // SPIN package

      if (_datamask & SP_MASK) {
        _sp(i+_first,0) = _buf(i,m++);
        _sp(i+_first,1) = _buf(i,m++);
        _sp(i+_first,2) = _buf(i,m++);
        _sp(i+_first,3) = _buf(i,m++);
      }

      // SPHERE package

      if (_datamask & OMEGA_MASK) {
        _omega(i+_first,0) = _buf(i,m++);
        _omega(i+_first,1) = _buf(i,m++);
        _omega(i+_first,2) = _buf(i,m++);
      }

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _dpdTheta(i+_first) = _buf(i,m++);
        _uCond(i+_first) = _buf(i,m++);
        _uMech(i+_first) = _buf(i,m++);
        _uChem(i+_first) = _buf(i,m++);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_comm_vel_kokkos(const int &n, const int &first,
    const DAT::tdual_double_2d_lr &buf) {

  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(HostKK,datamask_comm_vel);
    if (!ncomm_vel) {
      struct AtomVecKokkos_UnpackCommVel<LMPHostType,1> f(atomKK,buf,first,datamask_comm_vel);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnpackCommVel<LMPHostType,0> f(atomKK,buf,first,datamask_comm_vel);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(HostKK,datamask_comm_vel);
  } else {
    atomKK->sync(Device,datamask_comm_vel);
    if (!ncomm_vel) {
      struct AtomVecKokkos_UnpackCommVel<LMPDeviceType,1> f(atomKK,buf,first,datamask_comm_vel);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnpackCommVel<LMPDeviceType,0> f(atomKK,buf,first,datamask_comm_vel);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(Device,datamask_comm_vel);
  }

  if (bonus_flag) unpack_comm_bonus_kokkos(n, first, buf, 1);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int DEFAULT>
struct AtomVecKokkos_PackReverse {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkacc_1d_3_randomread _f,_fm,_fm_long;
  typename AT::t_kkacc_1d_3_randomread _torque;
  typename AT::t_double_2d_lr _buf;
  int _first;
  uint64_t _datamask;

  AtomVecKokkos_PackReverse(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_double_2d_lr &buf,
    const int &first, const uint64_t &datamask):
      _f(atomKK->k_f.view<DeviceType>()),
      _torque(atomKK->k_torque.view<DeviceType>()),
      _fm(atomKK->k_fm.view<DeviceType>()),
      _fm_long(atomKK->k_fm_long.view<DeviceType>()),
      _first(first),_datamask(datamask) {
        const size_t elements = atomKK->avecKK->size_reverse;
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/elements;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
      };

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    int m = 0;
    _buf(i,m++) = _f(i+_first,0);
    _buf(i,m++) = _f(i+_first,1);
    _buf(i,m++) = _f(i+_first,2);

    if constexpr (!DEFAULT) {

      // DIPLE package

      if (_datamask & TORQUE_MASK) {
        _buf(i,m++) = _torque(i+_first,0);
        _buf(i,m++) = _torque(i+_first,1);
        _buf(i,m++) = _torque(i+_first,2);
      }

      // SPIN package

      if (_datamask & FM_MASK) {
        _buf(i,m++) = _fm(i+_first,0);
        _buf(i,m++) = _fm(i+_first,1);
        _buf(i,m++) = _fm(i+_first,2);

        _buf(i,m++) = _fm_long(i+_first,0);
        _buf(i,m++) = _fm_long(i+_first,1);
        _buf(i,m++) = _fm_long(i+_first,2);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_reverse_kokkos(const int &n, const int &first,
    const DAT::tdual_double_2d_lr &buf) {
  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(HostKK,datamask_reverse);
    if (comm_f_only) {
      struct AtomVecKokkos_PackReverse<LMPHostType,1> f(atomKK,buf,first,datamask_reverse);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_PackReverse<LMPHostType,0> f(atomKK,buf,first,datamask_reverse);
      Kokkos::parallel_for(n,f);
    }
  } else {
    atomKK->sync(Device,datamask_reverse);
    if (comm_f_only) {
      struct AtomVecKokkos_PackReverse<LMPDeviceType,1> f(atomKK,buf,first,datamask_reverse);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_PackReverse<LMPDeviceType,0> f(atomKK,buf,first,datamask_reverse);
      Kokkos::parallel_for(n,f);
    }
  }

  return n*size_reverse;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int DEFAULT>
struct AtomVecKokkos_UnPackReverse {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkacc_1d_3 _f,_fm,_fm_long;
  typename AT::t_kkacc_1d_3 _torque;
  typename AT::t_double_2d_lr_const _buf;
  typename AT::t_int_1d_const _list;
  uint64_t _datamask;

  AtomVecKokkos_UnPackReverse(
    const AtomKokkos* atomKK,
    const typename DAT::tdual_double_2d_lr &buf,
    const typename DAT::tdual_int_1d &list,
    const uint64_t datamask):
      _f(atomKK->k_f.view<DeviceType>()),
      _torque(atomKK->k_torque.view<DeviceType>()),
      _fm(atomKK->k_fm.view<DeviceType>()),
      _fm_long(atomKK->k_fm_long.view<DeviceType>()),
      _list(list.view<DeviceType>()),
      _datamask(datamask) {
        const size_t elements = atomKK->avecKK->size_reverse;
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/elements;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
      };

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    int m = 0;
    const int j = _list(i);
    _f(j,0) += _buf(i,m++);
    _f(j,1) += _buf(i,m++);
    _f(j,2) += _buf(i,m++);

    if constexpr (!DEFAULT) {

      // DIPOLE package

      if (_datamask & TORQUE_MASK) {
        _torque(j,0) += _buf(i,m++);
        _torque(j,1) += _buf(i,m++);
        _torque(j,2) += _buf(i,m++);
      }

      // SPIN package

      if (_datamask & FM_MASK) {
        _fm(j,0) += _buf(i,m++);
        _fm(j,1) += _buf(i,m++);
        _fm(j,2) += _buf(i,m++);

        _fm_long(j,0) += _buf(i,m++);
        _fm_long(j,1) += _buf(i,m++);
        _fm_long(j,2) += _buf(i,m++);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_reverse_kokkos(const int &n,
                                          const DAT::tdual_int_1d &list,
                                          const DAT::tdual_double_2d_lr &buf)
{
  // Check whether to always run reverse communication on the host
  // Choose correct reverse UnPackReverse kernel

  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(HostKK,datamask_reverse);
    if (comm_f_only) {
      struct AtomVecKokkos_UnPackReverse<LMPHostType,1> f(atomKK,buf,list,datamask_reverse);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnPackReverse<LMPHostType,0> f(atomKK,buf,list,datamask_reverse);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(HostKK,datamask_reverse);
  } else {
    atomKK->sync(Device,datamask_reverse);
    if (comm_f_only) {
      struct AtomVecKokkos_UnPackReverse<LMPDeviceType,1> f(atomKK,buf,list,datamask_reverse);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnPackReverse<LMPDeviceType,0> f(atomKK,buf,list,datamask_reverse);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(Device,datamask_reverse);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int DEFAULT>
struct AtomVecKokkos_UnPackReverseSelf {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkacc_1d_3 _f,_fm,_fm_long;
  typename AT::t_kkacc_1d_3 _torque;
  typename AT::t_int_1d_const _list;
  int _nfirst;
  uint64_t _datamask;

  AtomVecKokkos_UnPackReverseSelf(
    const AtomKokkos* atomKK,
    const int &nfirst,
    const typename DAT::tdual_int_1d &list,
    const uint64_t &datamask):
      _f(atomKK->k_f.view<DeviceType>()),
      _torque(atomKK->k_torque.view<DeviceType>()),
      _fm(atomKK->k_fm.view<DeviceType>()),
      _fm_long(atomKK->k_fm_long.view<DeviceType>()),
      _nfirst(nfirst),_list(list.view<DeviceType>()),
      _datamask(datamask) {};

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    _f(j,0) += _f(i+_nfirst,0);
    _f(j,1) += _f(i+_nfirst,1);
    _f(j,2) += _f(i+_nfirst,2);

    if constexpr (!DEFAULT) {

      // DIPOLE package

      if (_datamask & TORQUE_MASK) {
        _torque(j,0) += _torque(i+_nfirst,0);
        _torque(j,1) += _torque(i+_nfirst,1);
        _torque(j,2) += _torque(i+_nfirst,2);
      }

      // SPIN package

      if (_datamask & FM_MASK) {
        _fm(j,0) += _fm(i+_nfirst,0);
        _fm(j,1) += _fm(i+_nfirst,1);
        _fm(j,2) += _fm(i+_nfirst,2);

        _fm_long(j,0) += _fm_long(i+_nfirst,0);
        _fm_long(j,1) += _fm_long(i+_nfirst,1);
        _fm_long(j,2) += _fm_long(i+_nfirst,2);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_reverse_self_kokkos(const int &n, const DAT::tdual_int_1d &list,
                                            const int nfirst) {
  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(HostKK,datamask_reverse);
    if (comm_f_only) {
      struct AtomVecKokkos_UnPackReverseSelf<LMPHostType,1> f(atomKK,nfirst,list,datamask_reverse);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnPackReverseSelf<LMPHostType,0> f(atomKK,nfirst,list,datamask_reverse);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(HostKK,datamask_reverse);
  } else {
    atomKK->sync(Device,datamask_reverse);
    if (comm_f_only) {
      struct AtomVecKokkos_UnPackReverseSelf<LMPDeviceType,1> f(atomKK,nfirst,list,datamask_reverse);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnPackReverseSelf<LMPDeviceType,0> f(atomKK,nfirst,list,datamask_reverse);
      Kokkos::parallel_for(n,f);
    }
    atomKK->modified(Device,datamask_reverse);
  }

  return n*size_reverse;
}
/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int DEFAULT>
struct AtomVecKokkos_PackBorder {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_double_2d_lr _buf;
  const typename AT::t_int_1d_const _list;
  const typename AT::t_kkfloat_1d_3_lr_randomread _x;
  const typename AT::t_tagint_1d_randomread _tag;
  const typename AT::t_int_1d_randomread _type;
  const typename AT::t_int_1d_randomread _mask;
  const typename AT::t_tagint_1d_randomread _molecule;
  const typename AT::t_kkfloat_1d_randomread _q;
  const typename AT::t_kkfloat_1d_4_randomread _mu;
  const typename AT::t_kkfloat_1d_4_randomread _sp;
  typename AT::t_kkfloat_1d_randomread _radius,_rmass;
  typename AT::t_kkfloat_1d_randomread _dpdTheta,_uCond,_uMech,_uChem,_uCG,_uCGnew;
  double _dx,_dy,_dz;
  uint64_t _datamask;

  AtomVecKokkos_PackBorder(
    const AtomKokkos* atomKK,
    const typename AT::t_double_2d_lr &buf,
    const typename AT::t_int_1d_const &list,
    const double &dx, const double &dy, const double &dz,
    const uint64_t &datamask):
      _buf(buf),_list(list),
      _x(atomKK->k_x.view<DeviceType>()),
      _tag(atomKK->k_tag.view<DeviceType>()),
      _type(atomKK->k_type.view<DeviceType>()),
      _mask(atomKK->k_mask.view<DeviceType>()),
      _molecule(atomKK->k_molecule.view<DeviceType>()),
      _q(atomKK->k_q.view<DeviceType>()),
      _mu(atomKK->k_mu.view<DeviceType>()),
      _sp(atomKK->k_sp.view<DeviceType>()),
      _radius(atomKK->k_radius.view<DeviceType>()),
      _rmass(atomKK->k_rmass.view<DeviceType>()),
      _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
      _uCond(atomKK->k_uCond.view<DeviceType>()),
      _uMech(atomKK->k_uMech.view<DeviceType>()),
      _uChem(atomKK->k_uChem.view<DeviceType>()),
      _uCG(atomKK->k_uCG.view<DeviceType>()),
      _uCGnew(atomKK->k_uCGnew.view<DeviceType>()),
      _dx(dx),_dy(dy),_dz(dz),_datamask(datamask) {}

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    int m = 0;
    if constexpr (PBC_FLAG == 0) {
      _buf(i,m++) = _x(j,0);
      _buf(i,m++) = _x(j,1);
      _buf(i,m++) = _x(j,2);
    } else {
      _buf(i,m++) = _x(j,0) + _dx;
      _buf(i,m++) = _x(j,1) + _dy;
      _buf(i,m++) = _x(j,2) + _dz;
    }

    _buf(i,m++) = d_ubuf(_tag(j)).d;
    _buf(i,m++) = d_ubuf(_type(j)).d;
    _buf(i,m++) = d_ubuf(_mask(j)).d;

    if constexpr (!DEFAULT) {

      if (_datamask & MOLECULE_MASK)
        _buf(i,m++) = d_ubuf(_molecule(j)).d;

      if (_datamask & Q_MASK)
        _buf(i,m++) = _q(j);

      if (_datamask & MU_MASK) {
        _buf(i,m++) = _mu(j,0);
        _buf(i,m++) = _mu(j,1);
        _buf(i,m++) = _mu(j,2);
        _buf(i,m++) = _mu(j,3);
      }

      if (_datamask & SP_MASK) {
        _buf(i,m++) = _sp(j,0);
        _buf(i,m++) = _sp(j,1);
        _buf(i,m++) = _sp(j,2);
        _buf(i,m++) = _sp(j,3);
      }

      if (_datamask & RADIUS_MASK)
        _buf(i,m++) = _radius(j);

      if (_datamask & RMASS_MASK)
        _buf(i,m++) = _rmass(j);

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _buf(i,m++) = _dpdTheta(j);
        _buf(i,m++) = _uCond(j);
        _buf(i,m++) = _uMech(j);
        _buf(i,m++) = _uChem(j);
        _buf(i,m++) = _uCG(j);
        _buf(i,m++) = _uCGnew(j);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                               DAT::tdual_double_2d_lr buf,
                                               int pbc_flag, int *pbc, ExecutionSpace space)
{
  atomKK->sync(space,datamask_border);

  double dx,dy,dz;

  if (pbc_flag != 0) {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (space == HostKK) {
      if (!nborder) {
        AtomVecKokkos_PackBorder<LMPHostType,1,1> f(
          atomKK,buf.view_host(), k_sendlist.view_host(),
          dx,dy,dz,datamask_border);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecKokkos_PackBorder<LMPHostType,1,0> f(
          atomKK,buf.view_host(), k_sendlist.view_host(),
          dx,dy,dz,datamask_border);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (!nborder) {
        AtomVecKokkos_PackBorder<LMPDeviceType,1,1> f(
          atomKK,buf.view_device(), k_sendlist.view_device(),
          dx,dy,dz,datamask_border);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecKokkos_PackBorder<LMPDeviceType,1,0> f(
          atomKK,buf.view_device(), k_sendlist.view_device(),
          dx,dy,dz,datamask_border);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    dx = dy = dz = 0;
    if (space == HostKK) {
      if (!nborder) {
        AtomVecKokkos_PackBorder<LMPHostType,0,1> f(
          atomKK,buf.view_host(), k_sendlist.view_host(),
          dx,dy,dz,datamask_border);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecKokkos_PackBorder<LMPHostType,0,0> f(
          atomKK,buf.view_host(), k_sendlist.view_host(),
          dx,dy,dz,datamask_border);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (!nborder) {
        AtomVecKokkos_PackBorder<LMPDeviceType,0,1> f(
          atomKK,buf.view_device(), k_sendlist.view_device(),
          dx,dy,dz,datamask_border);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecKokkos_PackBorder<LMPDeviceType,0,0> f(
          atomKK,buf.view_device(), k_sendlist.view_device(),
          dx,dy,dz,datamask_border);
        Kokkos::parallel_for(n,f);
      }
    }
  }

  if (bonus_flag) pack_border_bonus_kokkos(n, k_sendlist, buf, space);

  return n*size_border;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int DEFAULT>
struct AtomVecKokkos_UnpackBorder {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  const typename AT::t_double_2d_lr_const _buf;
  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_tagint_1d _molecule;
  typename AT::t_kkfloat_1d _q;
  typename AT::t_kkfloat_1d_4 _mu;
  typename AT::t_kkfloat_1d_4 _sp;
  typename AT::t_kkfloat_1d _radius,_rmass;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem,_uCG,_uCGnew;
  int _first;
  uint64_t _datamask;

  AtomVecKokkos_UnpackBorder(
    const AtomKokkos* atomKK,
    const typename AT::t_double_2d_lr_const &buf,
    const int &first, const uint64_t &datamask):
    _buf(buf),
    _x(atomKK->k_x.view<DeviceType>()),
    _tag(atomKK->k_tag.view<DeviceType>()),
    _type(atomKK->k_type.view<DeviceType>()),
    _mask(atomKK->k_mask.view<DeviceType>()),
    _molecule(atomKK->k_molecule.view<DeviceType>()),
    _q(atomKK->k_q.view<DeviceType>()),
    _mu(atomKK->k_mu.view<DeviceType>()),
    _sp(atomKK->k_sp.view<DeviceType>()),
    _radius(atomKK->k_radius.view<DeviceType>()),
    _rmass(atomKK->k_rmass.view<DeviceType>()),
    _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
    _uCond(atomKK->k_uCond.view<DeviceType>()),
    _uMech(atomKK->k_uMech.view<DeviceType>()),
    _uChem(atomKK->k_uChem.view<DeviceType>()),
    _uCG(atomKK->k_uCG.view<DeviceType>()),
    _uCGnew(atomKK->k_uCGnew.view<DeviceType>()),
    _first(first),_datamask(datamask) {
  };

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    int m = 0;
    _x(i+_first,0) = _buf(i,m++);
    _x(i+_first,1) = _buf(i,m++);
    _x(i+_first,2) = _buf(i,m++);
    _tag(i+_first) = (tagint) d_ubuf(_buf(i,m++)).i;
    _type(i+_first) = (int) d_ubuf(_buf(i,m++)).i;
    _mask(i+_first) = (int) d_ubuf(_buf(i,m++)).i;

    if constexpr (!DEFAULT) {

      if (_datamask & MOLECULE_MASK)
        _molecule(i+_first) = (tagint) d_ubuf(_buf(i,m++)).i;

      if (_datamask & Q_MASK)
        _q(i+_first) = _buf(i,m++);

      if (_datamask & MU_MASK) {
        _mu(i+_first,0) = _buf(i,m++);
        _mu(i+_first,1) = _buf(i,m++);
        _mu(i+_first,2) = _buf(i,m++);
        _mu(i+_first,3) = _buf(i,m++);
      }

      if (_datamask & SP_MASK) {
        _sp(i+_first,0) = _buf(i,m++);
        _sp(i+_first,1) = _buf(i,m++);
        _sp(i+_first,2) = _buf(i,m++);
        _sp(i+_first,3) = _buf(i,m++);
      }

      if (_datamask & RADIUS_MASK)
        _radius(i+_first) = _buf(i,m++);

      if (_datamask & RMASS_MASK)
        _rmass(i+_first) = _buf(i,m++);

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _dpdTheta(i+_first) = _buf(i,m++);
        _uCond(i+_first) = _buf(i,m++);
        _uMech(i+_first) = _buf(i,m++);
        _uChem(i+_first) = _buf(i,m++);
        _uCG(i+_first) = _buf(i,m++);
        _uCGnew(i+_first) = _buf(i,m++);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_border_kokkos(const int &n, const int &first,
                                               const DAT::tdual_double_2d_lr &buf,
                                               ExecutionSpace space) {
  while (first+n >= nmax) grow(0);

  atomKK->sync(space,datamask_border);

  if (space == HostKK) {
    if (!nborder) {
      struct AtomVecKokkos_UnpackBorder<LMPHostType,1>
        f(atomKK,buf.view_host(),first,datamask_border);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnpackBorder<LMPHostType,0>
        f(atomKK,buf.view_host(),first,datamask_border);
      Kokkos::parallel_for(n,f);
    }
  } else {
    if (!nborder) {
      struct AtomVecKokkos_UnpackBorder<LMPDeviceType,1>
        f(atomKK,buf.view_device(),first,datamask_border);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnpackBorder<LMPDeviceType,0>
        f(atomKK,buf.view_device(),first,datamask_border);
      Kokkos::parallel_for(n,f);
    }
  }

  if (bonus_flag) unpack_border_bonus_kokkos(n, first, buf, space);

  atomKK->modified(space,datamask_border);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int DEFORM_VREMAP>
struct AtomVecKokkos_PackBorderVel {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_double_2d_lr_um _buf;
  const typename AT::t_int_1d_const _list;
  const typename AT::t_kkfloat_1d_3_lr_randomread _x;
  typename AT::t_kkfloat_1d_3_randomread _v;
  const typename AT::t_tagint_1d_randomread _tag;
  const typename AT::t_int_1d_randomread _type;
  const typename AT::t_int_1d_randomread _mask;
  const typename AT::t_kkfloat_1d_3_randomread _angmom;
  const typename AT::t_tagint_1d_randomread _molecule;
  const typename AT::t_kkfloat_1d_randomread _q;
  const typename AT::t_kkfloat_1d_4_randomread _mu;
  const typename AT::t_kkfloat_1d_4_randomread _sp;
  typename AT::t_kkfloat_1d_randomread _radius,_rmass;
  typename AT::t_kkfloat_1d_3_randomread _omega;
  typename AT::t_kkfloat_1d_randomread _dpdTheta,_uCond,_uMech,_uChem,_uCG,_uCGnew;
  double _dx,_dy,_dz, _dvx, _dvy, _dvz;
  const int _deform_groupbit;
  const uint64_t _datamask;

  AtomVecKokkos_PackBorderVel(
    const AtomKokkos* atomKK,
    const typename AT::t_double_2d_lr &buf,
    const typename AT::t_int_1d_const &list,
    const double &dx, const double &dy, const double &dz,
    const double &dvx, const double &dvy, const double &dvz,
    const int &deform_groupbit,
    const uint64_t &datamask):
      _buf(buf),_list(list),_datamask(datamask),
      _x(atomKK->k_x.view<DeviceType>()),
      _tag(atomKK->k_tag.view<DeviceType>()),
      _type(atomKK->k_type.view<DeviceType>()),
      _mask(atomKK->k_mask.view<DeviceType>()),
      _angmom(atomKK->k_angmom.view<DeviceType>()),
      _molecule(atomKK->k_molecule.view<DeviceType>()),
      _q(atomKK->k_q.view<DeviceType>()),
      _v(atomKK->k_v.view<DeviceType>()),
      _mu(atomKK->k_mu.view<DeviceType>()),
      _sp(atomKK->k_sp.view<DeviceType>()),
      _radius(atomKK->k_radius.view<DeviceType>()),
      _rmass(atomKK->k_rmass.view<DeviceType>()),
      _omega(atomKK->k_omega.view<DeviceType>()),
      _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
      _uCond(atomKK->k_uCond.view<DeviceType>()),
      _uMech(atomKK->k_uMech.view<DeviceType>()),
      _uChem(atomKK->k_uChem.view<DeviceType>()),
      _uCG(atomKK->k_uCG.view<DeviceType>()),
      _uCGnew(atomKK->k_uCGnew.view<DeviceType>()),
      _dx(dx),_dy(dy),_dz(dz),
      _dvx(dvx),_dvy(dvy),_dvz(dvz),
      _deform_groupbit(deform_groupbit) {
        const size_t elements = atomKK->avecKK->size_border + atomKK->avecKK->size_velocity;
        const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
        _buf = typename AT::t_double_2d_lr_um(buf.data(),maxsend,elements);
      }

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    int m = 0;
    const int j = _list(i);
    if constexpr (PBC_FLAG == 0) {
      _buf(i,m++) = _x(j,0);
      _buf(i,m++) = _x(j,1);
      _buf(i,m++) = _x(j,2);
    } else {
      _buf(i,m++) = _x(j,0) + _dx;
      _buf(i,m++) = _x(j,1) + _dy;
      _buf(i,m++) = _x(j,2) + _dz;
    }
    _buf(i,m++) = d_ubuf(_tag(j)).d;
    _buf(i,m++) = d_ubuf(_type(j)).d;
    _buf(i,m++) = d_ubuf(_mask(j)).d;

    if constexpr (DEFORM_VREMAP) {
      if (_mask(i) & _deform_groupbit) {
        _buf(i,m++) = _v(j,0) + _dvx;
        _buf(i,m++) = _v(j,1) + _dvy;
        _buf(i,m++) = _v(j,2) + _dvz;
      }
    } else {
      _buf(i,m++) = _v(j,0);
      _buf(i,m++) = _v(j,1);
      _buf(i,m++) = _v(j,2);
    }

    // angmom: included for ellipsoid

    if (_datamask & ANGMOM_MASK) {
      _buf(i,m++) = _angmom(j,0);
      _buf(i,m++) = _angmom(j,1);
      _buf(i,m++) = _angmom(j,2);
    }

    if (_datamask & MOLECULE_MASK)
      _buf(i,m++) = d_ubuf(_molecule(j)).d;

    if (_datamask & Q_MASK)
      _buf(i,m++) = _q(j);

    if (_datamask & MU_MASK) {
      _buf(i,m++) = _mu(j,0);
      _buf(i,m++) = _mu(j,1);
      _buf(i,m++) = _mu(j,2);
      _buf(i,m++) = _mu(j,3);
    }

    if (_datamask & SP_MASK) {
      _buf(i,m++) = _sp(j,0);
      _buf(i,m++) = _sp(j,1);
      _buf(i,m++) = _sp(j,2);
      _buf(i,m++) = _sp(j,3);
    }

    if (_datamask & RADIUS_MASK)
      _buf(i,m++) = _radius(j);

    if (_datamask & RMASS_MASK)
      _buf(i,m++) = _rmass(j);

    if (_datamask & OMEGA_MASK) {
      _buf(i,m++) = _omega(j,0);
      _buf(i,m++) = _omega(j,1);
      _buf(i,m++) = _omega(j,2);
    }

    // DPD-REACT package

    if (_datamask & DPDTHETA_MASK) {
      _buf(i,m++) = _dpdTheta(j);
      _buf(i,m++) = _uCond(j);
      _buf(i,m++) = _uMech(j);
      _buf(i,m++) = _uChem(j);
      _buf(i,m++) = _uCG(j);
      _buf(i,m++) = _uCGnew(j);
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_border_vel_kokkos(
  int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_2d_lr buf,
  int pbc_flag, int *pbc, ExecutionSpace space)
{
  double dx = 0, dy = 0, dz = 0;
  double dvx = 0, dvy = 0, dvz = 0;

  atomKK->sync(space,datamask_border_vel);

  if (pbc_flag != 0) {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      if (space == HostKK) {
        AtomVecKokkos_PackBorderVel<LMPHostType,1,0> f(
          atomKK,
          buf.view_host(), k_sendlist.view_host(),
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit,datamask_border_vel);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecKokkos_PackBorderVel<LMPDeviceType,1,0> f(
          atomKK,
          buf.view_device(), k_sendlist.view_device(),
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit,datamask_border_vel);
        Kokkos::parallel_for(n,f);
      }
    }
    else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      if (space == HostKK) {
        AtomVecKokkos_PackBorderVel<LMPHostType,1,1> f(
          atomKK,
          buf.view_host(), k_sendlist.view_host(),
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit,datamask_border_vel);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecKokkos_PackBorderVel<LMPDeviceType,1,1> f(
          atomKK,
          buf.view_device(), k_sendlist.view_device(),
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit,datamask_border_vel);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    if (space == HostKK) {
      AtomVecKokkos_PackBorderVel<LMPHostType,0,0> f(
        atomKK,
        buf.view_host(), k_sendlist.view_host(),
        dx,dy,dz,dvx,dvy,dvz,
        deform_groupbit,datamask_border_vel);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecKokkos_PackBorderVel<LMPDeviceType,0,0> f(
        atomKK,
        buf.view_device(), k_sendlist.view_device(),
        dx,dy,dz,dvx,dvy,dvz,
        deform_groupbit,datamask_border_vel);
      Kokkos::parallel_for(n,f);
    }
  }

  if (bonus_flag) pack_border_bonus_kokkos(n, k_sendlist, buf, space, 1);

  return n*(size_border + size_velocity);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int DEFAULT>
struct AtomVecKokkos_UnpackBorderVel {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_double_2d_lr_const_um _buf;
  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_kkfloat_1d_3 _angmom;
  typename AT::t_tagint_1d _molecule;
  typename AT::t_kkfloat_1d _q;
  typename AT::t_kkfloat_1d_3 _v;
  typename AT::t_kkfloat_1d_4 _mu;
  typename AT::t_kkfloat_1d_4 _sp;
  typename AT::t_kkfloat_1d _radius,_rmass;
  typename AT::t_kkfloat_1d_3 _omega;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem,_uCG,_uCGnew;
  int _first;
  uint64_t _datamask;

  AtomVecKokkos_UnpackBorderVel(
    const AtomKokkos* atomKK,
    const typename AT::t_double_2d_lr_const &buf,
    const int &first,
    const uint64_t &datamask):
    _buf(buf),
    _x(atomKK->k_x.view<DeviceType>()),
    _tag(atomKK->k_tag.view<DeviceType>()),
    _type(atomKK->k_type.view<DeviceType>()),
    _mask(atomKK->k_mask.view<DeviceType>()),
    _angmom(atomKK->k_angmom.view<DeviceType>()),
    _molecule(atomKK->k_molecule.view<DeviceType>()),
    _q(atomKK->k_q.view<DeviceType>()),
    _v(atomKK->k_v.view<DeviceType>()),
    _mu(atomKK->k_mu.view<DeviceType>()),
    _sp(atomKK->k_sp.view<DeviceType>()),
    _radius(atomKK->k_radius.view<DeviceType>()),
    _rmass(atomKK->k_rmass.view<DeviceType>()),
    _omega(atomKK->k_omega.view<DeviceType>()),
    _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
    _uCond(atomKK->k_uCond.view<DeviceType>()),
    _uMech(atomKK->k_uMech.view<DeviceType>()),
    _uChem(atomKK->k_uChem.view<DeviceType>()),
    _uCG(atomKK->k_uCG.view<DeviceType>()),
    _uCGnew(atomKK->k_uCGnew.view<DeviceType>()),
    _first(first),_datamask(datamask)
  {
    const size_t elements = atomKK->avecKK->size_border + atomKK->avecKK->size_velocity;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename AT::t_double_2d_lr_const_um(buf.data(),maxsend,elements);
  };

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    int m = 0;
    _x(i+_first,0) = _buf(i,m++);
    _x(i+_first,1) = _buf(i,m++);
    _x(i+_first,2) = _buf(i,m++);
    _tag(i+_first) = static_cast<tagint>(d_ubuf(_buf(i,m++)).i);
    _type(i+_first) = static_cast<int>(d_ubuf(_buf(i,m++)).i);
    _mask(i+_first) = static_cast<int>(d_ubuf(_buf(i,m++)).i);
    _v(i+_first,0) = _buf(i,m++);
    _v(i+_first,1) = _buf(i,m++);
    _v(i+_first,2) = _buf(i,m++);

    if constexpr (!DEFAULT) {

      // angmom: included for ellipsoid

      if (_datamask & ANGMOM_MASK) {
        _angmom(i+_first,0) = _buf(i,m++);
        _angmom(i+_first,1) = _buf(i,m++);
        _angmom(i+_first,2) = _buf(i,m++);
      }

      if (_datamask & MOLECULE_MASK)
        _molecule(i+_first) = (tagint) d_ubuf(_buf(i,m++)).i;

      if (_datamask & Q_MASK)
        _q(i+_first) = _buf(i,m++);

      if (_datamask & MU_MASK) {
        _mu(i+_first,0) = _buf(i,m++);
        _mu(i+_first,1) = _buf(i,m++);
        _mu(i+_first,2) = _buf(i,m++);
        _mu(i+_first,3) = _buf(i,m++);
      }

      if (_datamask & SP_MASK) {
        _sp(i+_first,0) = _buf(i,m++);
        _sp(i+_first,1) = _buf(i,m++);
        _sp(i+_first,2) = _buf(i,m++);
        _sp(i+_first,3) = _buf(i,m++);
      }

      if (_datamask & RADIUS_MASK)
        _radius(i+_first) = _buf(i,m++);

      if (_datamask & RMASS_MASK)
        _rmass(i+_first) = _buf(i,m++);

      if (_datamask & OMEGA_MASK) {
        _omega(i+_first,0) = _buf(i,m++);
        _omega(i+_first,1) = _buf(i,m++);
        _omega(i+_first,2) = _buf(i,m++);
      }

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _dpdTheta(i+_first) = _buf(i,m++);
        _uCond(i+_first) = _buf(i,m++);
        _uMech(i+_first) = _buf(i,m++);
        _uChem(i+_first) = _buf(i,m++);
        _uCG(i+_first) = _buf(i,m++);
        _uCGnew(i+_first) = _buf(i,m++);
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::unpack_border_vel_kokkos(
    const int &n, const int &first,
    const DAT::tdual_double_2d_lr &buf,ExecutionSpace space) {

  while (first+n >= nmax) grow(0);

  atomKK->sync(space,datamask_border_vel);

  if (space == HostKK) {
    if (!ncomm_vel) {
      struct AtomVecKokkos_UnpackBorderVel<LMPHostType,1> f(
        atomKK,
        buf.view_host(),
        first,datamask_border_vel);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnpackBorderVel<LMPHostType,0> f(
        atomKK,
        buf.view_host(),
        first,datamask_border_vel);
      Kokkos::parallel_for(n,f);
    }
  } else {
    if (!ncomm_vel) {
      struct AtomVecKokkos_UnpackBorderVel<LMPDeviceType,1> f(
        atomKK,
        buf.view_device(),
        first,datamask_border_vel);
      Kokkos::parallel_for(n,f);
    } else {
      struct AtomVecKokkos_UnpackBorderVel<LMPDeviceType,0> f(
        atomKK,
        buf.view_device(),
        first,datamask_border_vel);
      Kokkos::parallel_for(n,f);
    }
  }

  if (bonus_flag) unpack_border_bonus_kokkos(n, first, buf, space, 1);

  atomKK->modified(space,datamask_border_vel);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int DEFAULT>
struct AtomVecKokkos_PackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_kkfloat_1d_3 _v;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_imageint_1d _image;
  typename AT::t_kkfloat_1d _q;
  typename AT::t_tagint_1d _molecule;
  typename AT::t_int_2d _nspecial;
  typename AT::t_tagint_2d _special;
  typename AT::t_int_1d _num_bond;
  typename AT::t_int_2d _bond_type;
  typename AT::t_tagint_2d _bond_atom;
  typename AT::t_int_1d _num_angle;
  typename AT::t_int_2d _angle_type;
  typename AT::t_tagint_2d _angle_atom1,_angle_atom2,_angle_atom3;
  typename AT::t_int_1d _num_dihedral;
  typename AT::t_int_2d _dihedral_type;
  typename AT::t_tagint_2d _dihedral_atom1,_dihedral_atom2,
    _dihedral_atom3,_dihedral_atom4;
  typename AT::t_int_1d _num_improper;
  typename AT::t_int_2d _improper_type;
  typename AT::t_tagint_2d _improper_atom1,_improper_atom2,
    _improper_atom3,_improper_atom4;
  typename AT::t_kkfloat_1d_4 _mu;
  typename AT::t_kkfloat_1d_4 _sp;
  typename AT::t_kkfloat_1d _radius,_rmass;
  typename AT::t_kkfloat_1d_3 _omega;
  typename AT::t_kkfloat_1d_3 _angmom;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem,_uCG,_uCGnew;

  typename AT::t_double_2d_lr_um _buf;
  typename AT::t_int_1d_const _sendlist;
  typename AT::t_int_1d_const _copylist;
  int _size_exchange;
  uint64_t _datamask;

  AtomVecKokkos_PackExchangeFunctor(
    const AtomKokkos* atomKK,
    const DAT::tdual_double_2d_lr buf,
    DAT::tdual_int_1d sendlist,
    DAT::tdual_int_1d copylist,
    const uint64_t datamask):
      _x(atomKK->k_x.view<DeviceType>()),
      _v(atomKK->k_v.view<DeviceType>()),
      _tag(atomKK->k_tag.view<DeviceType>()),
      _type(atomKK->k_type.view<DeviceType>()),
      _mask(atomKK->k_mask.view<DeviceType>()),
      _image(atomKK->k_image.view<DeviceType>()),
      _q(atomKK->k_q.view<DeviceType>()),
      _molecule(atomKK->k_molecule.view<DeviceType>()),
      _nspecial(atomKK->k_nspecial.view<DeviceType>()),
      _special(atomKK->k_special.view<DeviceType>()),
      _num_bond(atomKK->k_num_bond.view<DeviceType>()),
      _bond_type(atomKK->k_bond_type.view<DeviceType>()),
      _bond_atom(atomKK->k_bond_atom.view<DeviceType>()),
      _num_angle(atomKK->k_num_angle.view<DeviceType>()),
      _angle_type(atomKK->k_angle_type.view<DeviceType>()),
      _angle_atom1(atomKK->k_angle_atom1.view<DeviceType>()),
      _angle_atom2(atomKK->k_angle_atom2.view<DeviceType>()),
      _angle_atom3(atomKK->k_angle_atom3.view<DeviceType>()),
      _num_dihedral(atomKK->k_num_dihedral.view<DeviceType>()),
      _dihedral_type(atomKK->k_dihedral_type.view<DeviceType>()),
      _dihedral_atom1(atomKK->k_dihedral_atom1.view<DeviceType>()),
      _dihedral_atom2(atomKK->k_dihedral_atom2.view<DeviceType>()),
      _dihedral_atom3(atomKK->k_dihedral_atom3.view<DeviceType>()),
      _dihedral_atom4(atomKK->k_dihedral_atom4.view<DeviceType>()),
      _num_improper(atomKK->k_num_improper.view<DeviceType>()),
      _improper_type(atomKK->k_improper_type.view<DeviceType>()),
      _improper_atom1(atomKK->k_improper_atom1.view<DeviceType>()),
      _improper_atom2(atomKK->k_improper_atom2.view<DeviceType>()),
      _improper_atom3(atomKK->k_improper_atom3.view<DeviceType>()),
      _improper_atom4(atomKK->k_improper_atom4.view<DeviceType>()),
      _mu(atomKK->k_mu.view<DeviceType>()),
      _sp(atomKK->k_sp.view<DeviceType>()),
      _radius(atomKK->k_radius.view<DeviceType>()),
      _rmass(atomKK->k_rmass.view<DeviceType>()),
      _omega(atomKK->k_omega.view<DeviceType>()),
      _angmom(atomKK->k_angmom.view<DeviceType>()),
      _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
      _uCond(atomKK->k_uCond.view<DeviceType>()),
      _uMech(atomKK->k_uMech.view<DeviceType>()),
      _uChem(atomKK->k_uChem.view<DeviceType>()),
      _uCG(atomKK->k_uCG.view<DeviceType>()),
      _uCGnew(atomKK->k_uCGnew.view<DeviceType>()),

      _sendlist(sendlist.template view<DeviceType>()),
      _copylist(copylist.template view<DeviceType>()),
      _size_exchange(atomKK->avecKK->size_exchange),
      _datamask(datamask) {
        const int maxsendlist = (buf.template view<DeviceType>().extent(0)*
                                 buf.template view<DeviceType>().extent(1))/_size_exchange;
        buffer_view<DeviceType>(_buf,buf,maxsendlist,_size_exchange);
      }

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int &mysend) const {
    const int i = _sendlist(mysend);
    int m = 0;
    _buf(mysend,m++) = _size_exchange;

    _buf(mysend,m++) = _x(i,0);
    _buf(mysend,m++) = _x(i,1);
    _buf(mysend,m++) = _x(i,2);
    _buf(mysend,m++) = _v(i,0);
    _buf(mysend,m++) = _v(i,1);
    _buf(mysend,m++) = _v(i,2);
    _buf(mysend,m++) = d_ubuf(_tag(i)).d;
    _buf(mysend,m++) = d_ubuf(_type(i)).d;
    _buf(mysend,m++) = d_ubuf(_mask(i)).d;
    _buf(mysend,m++) = d_ubuf(_image(i)).d;

    if constexpr (!DEFAULT) {

      if (_datamask & Q_MASK)
        _buf(mysend,m++) = _q(i);

      if (_datamask & MOLECULE_MASK)
        _buf(mysend,m++) = d_ubuf(_molecule(i)).d;

      if (_datamask & BOND_MASK) {
        _buf(mysend,m++) = d_ubuf(_num_bond(i)).d;
        for (int k = 0; k < _num_bond(i); k++) {
          _buf(mysend,m++) = d_ubuf(_bond_type(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_bond_atom(i,k)).d;
        }
      }

      if (_datamask & ANGLE_MASK) {
        _buf(mysend,m++) = d_ubuf(_num_angle(i)).d;
        for (int k = 0; k < _num_angle(i); k++) {
          _buf(mysend,m++) = d_ubuf(_angle_type(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_angle_atom1(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_angle_atom2(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_angle_atom3(i,k)).d;
        }
      }

      if (_datamask & DIHEDRAL_MASK) {
        _buf(mysend,m++) = d_ubuf(_num_dihedral(i)).d;
        for (int k = 0; k < _num_dihedral(i); k++) {
          _buf(mysend,m++) = d_ubuf(_dihedral_type(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_dihedral_atom1(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_dihedral_atom2(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_dihedral_atom3(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_dihedral_atom4(i,k)).d;
        }
      }

      if (_datamask & IMPROPER_MASK) {
        _buf(mysend,m++) = d_ubuf(_num_improper(i)).d;
        for (int k = 0; k < _num_improper(i); k++) {
          _buf(mysend,m++) = d_ubuf(_improper_type(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_improper_atom1(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_improper_atom2(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_improper_atom3(i,k)).d;
          _buf(mysend,m++) = d_ubuf(_improper_atom4(i,k)).d;
        }
      }

      if (_datamask & SPECIAL_MASK) {
        _buf(mysend,m++) = d_ubuf(_nspecial(i,0)).d;
        _buf(mysend,m++) = d_ubuf(_nspecial(i,1)).d;
        _buf(mysend,m++) = d_ubuf(_nspecial(i,2)).d;
        for (int k = 0; k < _nspecial(i,2); k++)
          _buf(mysend,m++) = d_ubuf(_special(i,k)).d;
      }

      if (_datamask & MU_MASK) {
        _buf(mysend,m++) = _mu(i,0);
        _buf(mysend,m++) = _mu(i,1);
        _buf(mysend,m++) = _mu(i,2);
        _buf(mysend,m++) = _mu(i,3);
      }

      if (_datamask & SP_MASK) {
        _buf(mysend,m++) = _sp(i,0);
        _buf(mysend,m++) = _sp(i,1);
        _buf(mysend,m++) = _sp(i,2);
        _buf(mysend,m++) = _sp(i,3);
      }

      if (_datamask & RADIUS_MASK)
        _buf(mysend,m++) = _radius(i);

      if (_datamask & RMASS_MASK)
        _buf(mysend,m++) = _rmass(i);

      if (_datamask & OMEGA_MASK) {
        _buf(mysend,m++) = _omega(i,0);
        _buf(mysend,m++) = _omega(i,1);
        _buf(mysend,m++) = _omega(i,2);
      }

      // angmom: included for ellipsoid

      if (_datamask & ANGMOM_MASK) {
        _buf(mysend,m++) = _angmom(i,0);
        _buf(mysend,m++) = _angmom(i,1);
        _buf(mysend,m++) = _angmom(i,2);
      }

      // DPD-REACT package

      if (_datamask & DPDTHETA_MASK) {
        _buf(mysend,m++) = _dpdTheta(i);
        _buf(mysend,m++) = _uCond(i);
        _buf(mysend,m++) = _uMech(i);
        _buf(mysend,m++) = _uChem(i);
        _buf(mysend,m++) = _uCG(i);
        _buf(mysend,m++) = _uCGnew(i);
      }
    }

    const int j = _copylist(mysend);

    if (j > -1) {
      _x(i,0) = _x(j,0);
      _x(i,1) = _x(j,1);
      _x(i,2) = _x(j,2);
      _v(i,0) = _v(j,0);
      _v(i,1) = _v(j,1);
      _v(i,2) = _v(j,2);
      _tag(i) = _tag(j);
      _type(i) = _type(j);
      _mask(i) = _mask(j);
      _image(i) = _image(j);

      if constexpr (!DEFAULT) {

        if (_datamask & Q_MASK)
          _q(i) = _q(j);

        if (_datamask & MOLECULE_MASK)
          _molecule(i) = _molecule(j);

        if (_datamask & BOND_MASK) {
          _num_bond(i) = _num_bond(j);
          for (int k = 0; k < _num_bond(j); k++) {
            _bond_type(i,k) = _bond_type(j,k);
            _bond_atom(i,k) = _bond_atom(j,k);
          }
        }

        if (_datamask & ANGLE_MASK) {
          _num_angle(i) = _num_angle(j);
          for (int k = 0; k < _num_angle(j); k++) {
            _angle_type(i,k) = _angle_type(j,k);
            _angle_atom1(i,k) = _angle_atom1(j,k);
            _angle_atom2(i,k) = _angle_atom2(j,k);
            _angle_atom3(i,k) = _angle_atom3(j,k);
          }
        }

        if (_datamask & DIHEDRAL_MASK) {
          _num_dihedral(i) = _num_dihedral(j);
          for (int k = 0; k < _num_dihedral(j); k++) {
            _dihedral_type(i,k) = _dihedral_type(j,k);
            _dihedral_atom1(i,k) = _dihedral_atom1(j,k);
            _dihedral_atom2(i,k) = _dihedral_atom2(j,k);
            _dihedral_atom3(i,k) = _dihedral_atom3(j,k);
            _dihedral_atom4(i,k) = _dihedral_atom4(j,k);
          }
        }

        if (_datamask & IMPROPER_MASK) {
          _num_improper(i) = _num_improper(j);
          for (int k = 0; k < _num_improper(j); k++) {
            _improper_type(i,k) = _improper_type(j,k);
            _improper_atom1(i,k) = _improper_atom1(j,k);
            _improper_atom2(i,k) = _improper_atom2(j,k);
            _improper_atom3(i,k) = _improper_atom3(j,k);
            _improper_atom4(i,k) = _improper_atom4(j,k);
          }
        }

        if (_datamask & SPECIAL_MASK) {
          _nspecial(i,0) = _nspecial(j,0);
          _nspecial(i,1) = _nspecial(j,1);
          _nspecial(i,2) = _nspecial(j,2);
          for (int k = 0; k < _nspecial(j,2); k++)
            _special(i,k) = _special(j,k);
        }

        if (_datamask & MU_MASK) {
          _mu(i,0) = _mu(j,0);
          _mu(i,1) = _mu(j,1);
          _mu(i,2) = _mu(j,2);
          _mu(i,3) = _mu(j,3);
        }

        if (_datamask & SP_MASK) {
          _sp(i,0) = _sp(j,0);
          _sp(i,1) = _sp(j,1);
          _sp(i,2) = _sp(j,2);
          _sp(i,3) = _sp(j,3);
        }

        if (_datamask & RADIUS_MASK)
          _radius(i) = _radius(j);

        if (_datamask & RMASS_MASK)
          _rmass(i) = _rmass(j);

        if (_datamask & OMEGA_MASK) {
          _omega(i,0) = _omega(j,0);
          _omega(i,1) = _omega(j,1);
          _omega(i,2) = _omega(j,2);
        }

        if (_datamask & ANGMOM_MASK) {
          _angmom(i,0) = _angmom(j,0);
          _angmom(i,1) = _angmom(j,1);
          _angmom(i,2) = _angmom(j,2);
        }

        // DPD-REACT package

        if (_datamask & DPDTHETA_MASK) {
           _dpdTheta(i) = _dpdTheta(j);
          _uCond(i) = _uCond(j);
          _uMech(i) = _uMech(j);
          _uChem(i) = _uChem(j);
          _uCG(i) = _uCG(j);
          _uCGnew(i) = _uCGnew(j);
        }
      }
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::pack_exchange_kokkos(const int &nsend,DAT::tdual_double_2d_lr &k_buf,
                                                 DAT::tdual_int_1d k_sendlist,
                                                 DAT::tdual_int_1d k_copylist,
                                                 DAT::tdual_int_1d k_copylist_bonus,
                                                 ExecutionSpace space)
{
  set_size_exchange();

  if (!nsend) return 0;

  atomKK->sync(space,datamask_exchange);

  if (nsend > (int) (k_buf.view_host().extent(0)*
              k_buf.view_host().extent(1))/size_exchange) {
    int newsize = nsend*size_exchange/k_buf.view_host().extent(1)+1;
    k_buf.resize(newsize,k_buf.view_host().extent(1));
  }

  if (space == HostKK) {
    if (size_exchange == size_exchange_default) {
      AtomVecKokkos_PackExchangeFunctor<LMPHostType,1>
        f(atomKK,k_buf,k_sendlist,k_copylist,datamask_exchange);
      Kokkos::parallel_for(nsend,f);
    } else {
      AtomVecKokkos_PackExchangeFunctor<LMPHostType,0>
        f(atomKK,k_buf,k_sendlist,k_copylist,datamask_exchange);
      Kokkos::parallel_for(nsend,f);
    }
  } else {
    if (size_exchange == size_exchange_default) {
      AtomVecKokkos_PackExchangeFunctor<LMPDeviceType,1>
        f(atomKK,k_buf,k_sendlist,k_copylist,datamask_exchange);
      Kokkos::parallel_for(nsend,f);
    } else {
      AtomVecKokkos_PackExchangeFunctor<LMPDeviceType,0>
        f(atomKK,k_buf,k_sendlist,k_copylist,datamask_exchange);
      Kokkos::parallel_for(nsend,f);
    }
  }

  if (bonus_flag) pack_exchange_bonus_kokkos(nsend,k_buf,
                                             k_sendlist, k_copylist,
                                             k_copylist_bonus,
                                             space);

  atomKK->modified(space,datamask_exchange);

  return nsend*size_exchange;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int OUTPUT_INDICES,int DEFAULT>
struct AtomVecKokkos_UnpackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_kkfloat_1d_3_lr _x;
  typename AT::t_kkfloat_1d_3 _v;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_imageint_1d _image;
  typename AT::t_kkfloat_1d _q;
  typename AT::t_tagint_1d _molecule;
  typename AT::t_int_2d _nspecial;
  typename AT::t_tagint_2d _special;
  typename AT::t_int_1d _num_bond;
  typename AT::t_int_2d _bond_type;
  typename AT::t_tagint_2d _bond_atom;
  typename AT::t_int_1d _num_angle;
  typename AT::t_int_2d _angle_type;
  typename AT::t_tagint_2d _angle_atom1,_angle_atom2,_angle_atom3;
  typename AT::t_int_1d _num_dihedral;
  typename AT::t_int_2d _dihedral_type;
  typename AT::t_tagint_2d _dihedral_atom1,_dihedral_atom2,
    _dihedral_atom3,_dihedral_atom4;
  typename AT::t_int_1d _num_improper;
  typename AT::t_int_2d _improper_type;
  typename AT::t_tagint_2d _improper_atom1,_improper_atom2,
    _improper_atom3,_improper_atom4;
  typename AT::t_kkfloat_1d_4 _mu;
  typename AT::t_kkfloat_1d_4 _sp;
  typename AT::t_kkfloat_1d _radius,_rmass;
  typename AT::t_kkfloat_1d_3 _omega;
  typename AT::t_kkfloat_1d_3 _angmom;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem,_uCG,_uCGnew;

  typename AT::t_double_2d_lr_um _buf;
  typename AT::t_int_1d _nlocal;
  typename AT::t_int_1d _indices;
  int _dim;
  double _lo,_hi;
  int _size_exchange;
  uint64_t _datamask;

  AtomVecKokkos_UnpackExchangeFunctor(
    const AtomKokkos* atomKK,
    const DAT::tdual_double_2d_lr buf,
    DAT::tdual_int_1d nlocal,
    DAT::tdual_int_1d indices,
    int dim, double lo, double hi,
    uint64_t datamask):
      _x(atomKK->k_x.view<DeviceType>()),
      _v(atomKK->k_v.view<DeviceType>()),
      _tag(atomKK->k_tag.view<DeviceType>()),
      _type(atomKK->k_type.view<DeviceType>()),
      _mask(atomKK->k_mask.view<DeviceType>()),
      _image(atomKK->k_image.view<DeviceType>()),
      _q(atomKK->k_q.view<DeviceType>()),
      _molecule(atomKK->k_molecule.view<DeviceType>()),
      _nspecial(atomKK->k_nspecial.view<DeviceType>()),
      _special(atomKK->k_special.view<DeviceType>()),
      _num_bond(atomKK->k_num_bond.view<DeviceType>()),
      _bond_type(atomKK->k_bond_type.view<DeviceType>()),
      _bond_atom(atomKK->k_bond_atom.view<DeviceType>()),
      _num_angle(atomKK->k_num_angle.view<DeviceType>()),
      _angle_type(atomKK->k_angle_type.view<DeviceType>()),
      _angle_atom1(atomKK->k_angle_atom1.view<DeviceType>()),
      _angle_atom2(atomKK->k_angle_atom2.view<DeviceType>()),
      _angle_atom3(atomKK->k_angle_atom3.view<DeviceType>()),
      _num_dihedral(atomKK->k_num_dihedral.view<DeviceType>()),
      _dihedral_type(atomKK->k_dihedral_type.view<DeviceType>()),
      _dihedral_atom1(atomKK->k_dihedral_atom1.view<DeviceType>()),
      _dihedral_atom2(atomKK->k_dihedral_atom2.view<DeviceType>()),
      _dihedral_atom3(atomKK->k_dihedral_atom3.view<DeviceType>()),
      _dihedral_atom4(atomKK->k_dihedral_atom4.view<DeviceType>()),
      _num_improper(atomKK->k_num_improper.view<DeviceType>()),
      _improper_type(atomKK->k_improper_type.view<DeviceType>()),
      _improper_atom1(atomKK->k_improper_atom1.view<DeviceType>()),
      _improper_atom2(atomKK->k_improper_atom2.view<DeviceType>()),
      _improper_atom3(atomKK->k_improper_atom3.view<DeviceType>()),
      _improper_atom4(atomKK->k_improper_atom4.view<DeviceType>()),
      _mu(atomKK->k_mu.view<DeviceType>()),
      _sp(atomKK->k_sp.view<DeviceType>()),
      _radius(atomKK->k_radius.view<DeviceType>()),
      _rmass(atomKK->k_rmass.view<DeviceType>()),
      _omega(atomKK->k_omega.view<DeviceType>()),
      _angmom(atomKK->k_angmom.view<DeviceType>()),
      _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()),
      _uCond(atomKK->k_uCond.view<DeviceType>()),
      _uMech(atomKK->k_uMech.view<DeviceType>()),
      _uChem(atomKK->k_uChem.view<DeviceType>()),
      _uCG(atomKK->k_uCG.view<DeviceType>()),
      _uCGnew(atomKK->k_uCGnew.view<DeviceType>()),

      _nlocal(nlocal.template view<DeviceType>()),
      _indices(indices.template view<DeviceType>()),
      _dim(dim),_lo(lo),_hi(hi),_size_exchange(atomKK->avecKK->size_exchange),
      _datamask(datamask) {
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*
                             buf.template view<DeviceType>().extent(1))/_size_exchange;
    buffer_view<DeviceType>(_buf,buf,maxsendlist,_size_exchange);
  }

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int &myrecv) const {
    double x = _buf(myrecv,_dim+1);
    int i = -1;
    if (x >= _lo && x < _hi) {
      i = Kokkos::atomic_fetch_add(&_nlocal(0),1);
      int m = 1;
      _x(i,0) = _buf(myrecv,m++);
      _x(i,1) = _buf(myrecv,m++);
      _x(i,2) = _buf(myrecv,m++);
      _v(i,0) = _buf(myrecv,m++);
      _v(i,1) = _buf(myrecv,m++);
      _v(i,2) = _buf(myrecv,m++);
      _tag(i) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
      _type(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      _mask(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
      _image(i) = (imageint) d_ubuf(_buf(myrecv,m++)).i;

      if constexpr (!DEFAULT) {

        if (_datamask & Q_MASK)
          _q(i) = _buf(myrecv,m++);

        if (_datamask & MOLECULE_MASK)
          _molecule(i) = (tagint) d_ubuf(_buf(myrecv,m++)).i;

        if (_datamask & BOND_MASK) {
          _num_bond(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
          for (int k = 0; k < _num_bond(i); k++) {
            _bond_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
            _bond_atom(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          }
        }

        if (_datamask & ANGLE_MASK) {
          _num_angle(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
          for (int k = 0; k < _num_angle(i); k++) {
            _angle_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
            _angle_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
            _angle_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
            _angle_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          }
        }

        if (_datamask & DIHEDRAL_MASK) {
          _num_dihedral(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
          for (int k = 0; k < _num_dihedral(i); k++) {
            _dihedral_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
            _dihedral_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
            _dihedral_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
            _dihedral_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
            _dihedral_atom4(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          }
        }

        if (_datamask & IMPROPER_MASK) {
          _num_improper(i) = (int) d_ubuf(_buf(myrecv,m++)).i;
          for (int k = 0; k < _num_improper(i); k++) {
            _improper_type(i,k) = (int) d_ubuf(_buf(myrecv,m++)).i;
            _improper_atom1(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
            _improper_atom2(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
            _improper_atom3(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
            _improper_atom4(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
          }
        }

        if (_datamask & SPECIAL_MASK) {
          _nspecial(i,0) = (int) d_ubuf(_buf(myrecv,m++)).i;
          _nspecial(i,1) = (int) d_ubuf(_buf(myrecv,m++)).i;
          _nspecial(i,2) = (int) d_ubuf(_buf(myrecv,m++)).i;
          for (int k = 0; k < _nspecial(i,2); k++)
            _special(i,k) = (tagint) d_ubuf(_buf(myrecv,m++)).i;
        }

        if (_datamask & MU_MASK) {
          _mu(i,0) = _buf(myrecv,m++);
          _mu(i,1) = _buf(myrecv,m++);
          _mu(i,2) = _buf(myrecv,m++);
          _mu(i,3) = _buf(myrecv,m++);
        }

        if (_datamask & SP_MASK) {
          _sp(i,0) = _buf(myrecv,m++);
          _sp(i,1) = _buf(myrecv,m++);
          _sp(i,2) = _buf(myrecv,m++);
          _sp(i,3) = _buf(myrecv,m++);
        }

        if (_datamask & RADIUS_MASK)
          _radius(i) = _buf(myrecv,m++);

        if (_datamask & RMASS_MASK)
          _rmass(i) = _buf(myrecv,m++);

        if (_datamask & OMEGA_MASK) {
          _omega(i,0) = _buf(myrecv,m++);
          _omega(i,1) = _buf(myrecv,m++);
          _omega(i,2) = _buf(myrecv,m++);
        }

        if (_datamask & ANGMOM_MASK) {
          _angmom(i,0) = _buf(myrecv,m++);
          _angmom(i,1) = _buf(myrecv,m++);
          _angmom(i,2) = _buf(myrecv,m++);
        }

        // DPD-REACT package

        if (_datamask & DPDTHETA_MASK) {
          _dpdTheta(i) = _buf(myrecv,m++);
          _uCond(i) = _buf(myrecv,m++);
          _uMech(i) = _buf(myrecv,m++);
          _uChem(i) = _buf(myrecv,m++);
          _uCG(i) = _buf(myrecv,m++);
          _uCGnew(i) = _buf(myrecv,m++);
        }
      }
    }

    if constexpr (OUTPUT_INDICES)
      _indices(myrecv) = i;
  }
};

/* ---------------------------------------------------------------------- */
int AtomVecKokkos::unpack_exchange_kokkos(DAT::tdual_double_2d_lr &k_buf, int nrecv, int nlocal,
                                              int dim, double lo, double hi, ExecutionSpace space,
                                              DAT::tdual_int_1d &k_indices)
{
  while (nlocal + nrecv/size_exchange >= nmax) grow(0);

  atomKK->sync(space,datamask_exchange);

  if (space == HostKK) {
    k_count.view_host()(0) = nlocal;

    if (k_indices.view_host().data()) {
      if (size_exchange == size_exchange_default) {
        AtomVecKokkos_UnpackExchangeFunctor<LMPHostType,1,1>
          f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
        Kokkos::parallel_for(nrecv/size_exchange,f);
      } else {
        AtomVecKokkos_UnpackExchangeFunctor<LMPHostType,1,0>
          f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
        Kokkos::parallel_for(nrecv/size_exchange,f);
      }
    } else {
      if (size_exchange == size_exchange_default) {
        AtomVecKokkos_UnpackExchangeFunctor<LMPHostType,0,1>
          f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
        Kokkos::parallel_for(nrecv/size_exchange,f);
      } else {
        AtomVecKokkos_UnpackExchangeFunctor<LMPHostType,0,0>
          f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
        Kokkos::parallel_for(nrecv/size_exchange,f);
      }
    }
  } else {
    k_count.view_host()(0) = nlocal;
    k_count.modify_host();
    k_count.sync_device();

    if (k_indices.view_host().data()) {
      if (size_exchange == size_exchange_default) {
        AtomVecKokkos_UnpackExchangeFunctor<LMPDeviceType,1,1>
          f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
        Kokkos::parallel_for(nrecv/size_exchange,f);
      } else {
        AtomVecKokkos_UnpackExchangeFunctor<LMPDeviceType,1,0>
          f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
        Kokkos::parallel_for(nrecv/size_exchange,f);
      }
    } else {
      if (size_exchange == size_exchange_default) {
        AtomVecKokkos_UnpackExchangeFunctor<LMPDeviceType,0,1>
          f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
        Kokkos::parallel_for(nrecv/size_exchange,f);
      } else {
        AtomVecKokkos_UnpackExchangeFunctor<LMPDeviceType,0,0>
          f(atomKK,k_buf,k_count,k_indices,dim,lo,hi,datamask_exchange);
        Kokkos::parallel_for(nrecv/size_exchange,f);
      }
    }

    k_count.modify_device();
    k_count.sync_host();
  }

  if (bonus_flag) unpack_exchange_bonus_kokkos(k_buf,nrecv,space,k_indices);

  atomKK->modified(space,datamask_exchange);

  return k_count.view_host()(0);
}

/* ----------------------------------------------------------------------
   sort atom arrays on device with a single coalesced gather kernel

   The set of per-atom arrays that must be permuted by a spatial sort is
   exactly the set of persistent per-atom arrays that travel when an atom
   migrates to another MPI rank, i.e. the "exchange" set, so datamask_exchange
   selects the arrays and the same mask bits as pack/unpack_exchange gate the
   optional ones.

   To keep global-memory accesses coalesced and at native width (a flat
   double "array-of-structures" buffer is uncoalesced and widens 4-byte
   fields to 8 bytes), each array is gathered into a native-typed scratch
   array of the same type in sorted order: out(i,..) = in(permute(i),..),
   which writes contiguously across i. The scratch arrays are then swapped
   into the AtomKokkos k_* arrays (rebinding the legacy raw pointers) so no
   copy-back kernel or extra buffer traffic is needed.
------------------------------------------------------------------------- */

template<class DV>
static void grow_sort_1d(DV &scratch, int nmax)
{
  if ((int)scratch.view_device().extent(0) < nmax) scratch.resize(nmax);
}

template<class DV, class REF>
static void grow_sort_2d(DV &scratch, REF &ref, int nmax)
{
  if ((int)scratch.view_device().extent(0) < nmax ||
      (int)scratch.view_device().extent(1) != (int)ref.view_device().extent(1))
    scratch.resize(nmax,ref.view_device().extent(1));
}

template<class DV, class PTR>
static void swap_sort(MemoryKokkos *memoryKK, DV &kview, DV &scratch,
                      PTR &raw, int nmax, const char *name)
{
  std::swap(kview,scratch);
  memoryKK->grow_kokkos(kview,raw,nmax,name);  // rebind legacy raw pointer
}

namespace LAMMPS_NS {

template<class DeviceType,int DEFAULT,class PermuteView>
struct AtomVecKokkos_GatherSortFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  // _foo = source (current order), _foo_out = sorted scratch destination
  typename AT::t_kkfloat_1d_3_lr _x,_x_out;
  typename AT::t_kkfloat_1d_3 _v,_v_out;
  typename AT::t_tagint_1d _tag,_tag_out;
  typename AT::t_int_1d _type,_type_out;
  typename AT::t_int_1d _mask,_mask_out;
  typename AT::t_imageint_1d _image,_image_out;
  typename AT::t_kkfloat_1d _q,_q_out;
  typename AT::t_tagint_1d _molecule,_molecule_out;
  typename AT::t_int_2d _nspecial,_nspecial_out;
  typename AT::t_tagint_2d _special,_special_out;
  typename AT::t_int_1d _num_bond,_num_bond_out;
  typename AT::t_int_2d _bond_type,_bond_type_out;
  typename AT::t_tagint_2d _bond_atom,_bond_atom_out;
  typename AT::t_int_1d _num_angle,_num_angle_out;
  typename AT::t_int_2d _angle_type,_angle_type_out;
  typename AT::t_tagint_2d _angle_atom1,_angle_atom2,_angle_atom3;
  typename AT::t_tagint_2d _angle_atom1_out,_angle_atom2_out,_angle_atom3_out;
  typename AT::t_int_1d _num_dihedral,_num_dihedral_out;
  typename AT::t_int_2d _dihedral_type,_dihedral_type_out;
  typename AT::t_tagint_2d _dihedral_atom1,_dihedral_atom2,_dihedral_atom3,_dihedral_atom4;
  typename AT::t_tagint_2d _dihedral_atom1_out,_dihedral_atom2_out,_dihedral_atom3_out,_dihedral_atom4_out;
  typename AT::t_int_1d _num_improper,_num_improper_out;
  typename AT::t_int_2d _improper_type,_improper_type_out;
  typename AT::t_tagint_2d _improper_atom1,_improper_atom2,_improper_atom3,_improper_atom4;
  typename AT::t_tagint_2d _improper_atom1_out,_improper_atom2_out,_improper_atom3_out,_improper_atom4_out;
  typename AT::t_kkfloat_1d_4 _mu,_mu_out;
  typename AT::t_kkfloat_1d_4 _sp,_sp_out;
  typename AT::t_kkfloat_1d _radius,_radius_out,_rmass,_rmass_out;
  typename AT::t_kkfloat_1d_3 _omega,_omega_out,_angmom,_angmom_out;
  typename AT::t_kkfloat_1d _dpdTheta,_uCond,_uMech,_uChem,_uCG,_uCGnew;
  typename AT::t_kkfloat_1d _dpdTheta_out,_uCond_out,_uMech_out,_uChem_out,_uCG_out,_uCGnew_out;

  PermuteView _permute;
  uint64_t _datamask;

  AtomVecKokkos_GatherSortFunctor(const AtomKokkos* atomKK, AtomVecKokkos* avec,
    PermuteView permute, const uint64_t datamask):
      _x(atomKK->k_x.view<DeviceType>()), _x_out(avec->k_x_sort.view<DeviceType>()),
      _v(atomKK->k_v.view<DeviceType>()), _v_out(avec->k_v_sort.view<DeviceType>()),
      _tag(atomKK->k_tag.view<DeviceType>()), _tag_out(avec->k_tag_sort.view<DeviceType>()),
      _type(atomKK->k_type.view<DeviceType>()), _type_out(avec->k_type_sort.view<DeviceType>()),
      _mask(atomKK->k_mask.view<DeviceType>()), _mask_out(avec->k_mask_sort.view<DeviceType>()),
      _image(atomKK->k_image.view<DeviceType>()), _image_out(avec->k_image_sort.view<DeviceType>()),
      _q(atomKK->k_q.view<DeviceType>()), _q_out(avec->k_q_sort.view<DeviceType>()),
      _molecule(atomKK->k_molecule.view<DeviceType>()), _molecule_out(avec->k_molecule_sort.view<DeviceType>()),
      _nspecial(atomKK->k_nspecial.view<DeviceType>()), _nspecial_out(avec->k_nspecial_sort.view<DeviceType>()),
      _special(atomKK->k_special.view<DeviceType>()), _special_out(avec->k_special_sort.view<DeviceType>()),
      _num_bond(atomKK->k_num_bond.view<DeviceType>()), _num_bond_out(avec->k_num_bond_sort.view<DeviceType>()),
      _bond_type(atomKK->k_bond_type.view<DeviceType>()), _bond_type_out(avec->k_bond_type_sort.view<DeviceType>()),
      _bond_atom(atomKK->k_bond_atom.view<DeviceType>()), _bond_atom_out(avec->k_bond_atom_sort.view<DeviceType>()),
      _num_angle(atomKK->k_num_angle.view<DeviceType>()), _num_angle_out(avec->k_num_angle_sort.view<DeviceType>()),
      _angle_type(atomKK->k_angle_type.view<DeviceType>()), _angle_type_out(avec->k_angle_type_sort.view<DeviceType>()),
      _angle_atom1(atomKK->k_angle_atom1.view<DeviceType>()), _angle_atom2(atomKK->k_angle_atom2.view<DeviceType>()),
      _angle_atom3(atomKK->k_angle_atom3.view<DeviceType>()),
      _angle_atom1_out(avec->k_angle_atom1_sort.view<DeviceType>()), _angle_atom2_out(avec->k_angle_atom2_sort.view<DeviceType>()),
      _angle_atom3_out(avec->k_angle_atom3_sort.view<DeviceType>()),
      _num_dihedral(atomKK->k_num_dihedral.view<DeviceType>()), _num_dihedral_out(avec->k_num_dihedral_sort.view<DeviceType>()),
      _dihedral_type(atomKK->k_dihedral_type.view<DeviceType>()), _dihedral_type_out(avec->k_dihedral_type_sort.view<DeviceType>()),
      _dihedral_atom1(atomKK->k_dihedral_atom1.view<DeviceType>()), _dihedral_atom2(atomKK->k_dihedral_atom2.view<DeviceType>()),
      _dihedral_atom3(atomKK->k_dihedral_atom3.view<DeviceType>()), _dihedral_atom4(atomKK->k_dihedral_atom4.view<DeviceType>()),
      _dihedral_atom1_out(avec->k_dihedral_atom1_sort.view<DeviceType>()), _dihedral_atom2_out(avec->k_dihedral_atom2_sort.view<DeviceType>()),
      _dihedral_atom3_out(avec->k_dihedral_atom3_sort.view<DeviceType>()), _dihedral_atom4_out(avec->k_dihedral_atom4_sort.view<DeviceType>()),
      _num_improper(atomKK->k_num_improper.view<DeviceType>()), _num_improper_out(avec->k_num_improper_sort.view<DeviceType>()),
      _improper_type(atomKK->k_improper_type.view<DeviceType>()), _improper_type_out(avec->k_improper_type_sort.view<DeviceType>()),
      _improper_atom1(atomKK->k_improper_atom1.view<DeviceType>()), _improper_atom2(atomKK->k_improper_atom2.view<DeviceType>()),
      _improper_atom3(atomKK->k_improper_atom3.view<DeviceType>()), _improper_atom4(atomKK->k_improper_atom4.view<DeviceType>()),
      _improper_atom1_out(avec->k_improper_atom1_sort.view<DeviceType>()), _improper_atom2_out(avec->k_improper_atom2_sort.view<DeviceType>()),
      _improper_atom3_out(avec->k_improper_atom3_sort.view<DeviceType>()), _improper_atom4_out(avec->k_improper_atom4_sort.view<DeviceType>()),
      _mu(atomKK->k_mu.view<DeviceType>()), _mu_out(avec->k_mu_sort.view<DeviceType>()),
      _sp(atomKK->k_sp.view<DeviceType>()), _sp_out(avec->k_sp_sort.view<DeviceType>()),
      _radius(atomKK->k_radius.view<DeviceType>()), _radius_out(avec->k_radius_sort.view<DeviceType>()),
      _rmass(atomKK->k_rmass.view<DeviceType>()), _rmass_out(avec->k_rmass_sort.view<DeviceType>()),
      _omega(atomKK->k_omega.view<DeviceType>()), _omega_out(avec->k_omega_sort.view<DeviceType>()),
      _angmom(atomKK->k_angmom.view<DeviceType>()), _angmom_out(avec->k_angmom_sort.view<DeviceType>()),
      _dpdTheta(atomKK->k_dpdTheta.view<DeviceType>()), _uCond(atomKK->k_uCond.view<DeviceType>()),
      _uMech(atomKK->k_uMech.view<DeviceType>()), _uChem(atomKK->k_uChem.view<DeviceType>()),
      _uCG(atomKK->k_uCG.view<DeviceType>()), _uCGnew(atomKK->k_uCGnew.view<DeviceType>()),
      _dpdTheta_out(avec->k_dpdTheta_sort.view<DeviceType>()), _uCond_out(avec->k_uCond_sort.view<DeviceType>()),
      _uMech_out(avec->k_uMech_sort.view<DeviceType>()), _uChem_out(avec->k_uChem_sort.view<DeviceType>()),
      _uCG_out(avec->k_uCG_sort.view<DeviceType>()), _uCGnew_out(avec->k_uCGnew_sort.view<DeviceType>()),
      _permute(permute), _datamask(datamask) {}

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int &i) const {
    const int j = _permute(i);

    _x_out(i,0) = _x(j,0); _x_out(i,1) = _x(j,1); _x_out(i,2) = _x(j,2);
    _v_out(i,0) = _v(j,0); _v_out(i,1) = _v(j,1); _v_out(i,2) = _v(j,2);
    _tag_out(i) = _tag(j);
    _type_out(i) = _type(j);
    _mask_out(i) = _mask(j);
    _image_out(i) = _image(j);

    if constexpr (!DEFAULT) {

      if (_datamask & Q_MASK)
        _q_out(i) = _q(j);

      if (_datamask & MOLECULE_MASK)
        _molecule_out(i) = _molecule(j);

      if (_datamask & BOND_MASK) {
        _num_bond_out(i) = _num_bond(j);
        const int n = _bond_type.extent(1);
        for (int k = 0; k < n; k++) {
          _bond_type_out(i,k) = _bond_type(j,k);
          _bond_atom_out(i,k) = _bond_atom(j,k);
        }
      }

      if (_datamask & ANGLE_MASK) {
        _num_angle_out(i) = _num_angle(j);
        const int n = _angle_type.extent(1);
        for (int k = 0; k < n; k++) {
          _angle_type_out(i,k) = _angle_type(j,k);
          _angle_atom1_out(i,k) = _angle_atom1(j,k);
          _angle_atom2_out(i,k) = _angle_atom2(j,k);
          _angle_atom3_out(i,k) = _angle_atom3(j,k);
        }
      }

      if (_datamask & DIHEDRAL_MASK) {
        _num_dihedral_out(i) = _num_dihedral(j);
        const int n = _dihedral_type.extent(1);
        for (int k = 0; k < n; k++) {
          _dihedral_type_out(i,k) = _dihedral_type(j,k);
          _dihedral_atom1_out(i,k) = _dihedral_atom1(j,k);
          _dihedral_atom2_out(i,k) = _dihedral_atom2(j,k);
          _dihedral_atom3_out(i,k) = _dihedral_atom3(j,k);
          _dihedral_atom4_out(i,k) = _dihedral_atom4(j,k);
        }
      }

      if (_datamask & IMPROPER_MASK) {
        _num_improper_out(i) = _num_improper(j);
        const int n = _improper_type.extent(1);
        for (int k = 0; k < n; k++) {
          _improper_type_out(i,k) = _improper_type(j,k);
          _improper_atom1_out(i,k) = _improper_atom1(j,k);
          _improper_atom2_out(i,k) = _improper_atom2(j,k);
          _improper_atom3_out(i,k) = _improper_atom3(j,k);
          _improper_atom4_out(i,k) = _improper_atom4(j,k);
        }
      }

      if (_datamask & SPECIAL_MASK) {
        _nspecial_out(i,0) = _nspecial(j,0);
        _nspecial_out(i,1) = _nspecial(j,1);
        _nspecial_out(i,2) = _nspecial(j,2);
        const int n = _special.extent(1);
        for (int k = 0; k < n; k++)
          _special_out(i,k) = _special(j,k);
      }

      if (_datamask & MU_MASK) {
        _mu_out(i,0) = _mu(j,0); _mu_out(i,1) = _mu(j,1);
        _mu_out(i,2) = _mu(j,2); _mu_out(i,3) = _mu(j,3);
      }

      if (_datamask & SP_MASK) {
        _sp_out(i,0) = _sp(j,0); _sp_out(i,1) = _sp(j,1);
        _sp_out(i,2) = _sp(j,2); _sp_out(i,3) = _sp(j,3);
      }

      if (_datamask & RADIUS_MASK)
        _radius_out(i) = _radius(j);

      if (_datamask & RMASS_MASK)
        _rmass_out(i) = _rmass(j);

      if (_datamask & OMEGA_MASK) {
        _omega_out(i,0) = _omega(j,0);
        _omega_out(i,1) = _omega(j,1);
        _omega_out(i,2) = _omega(j,2);
      }

      if (_datamask & ANGMOM_MASK) {
        _angmom_out(i,0) = _angmom(j,0);
        _angmom_out(i,1) = _angmom(j,1);
        _angmom_out(i,2) = _angmom(j,2);
      }

      if (_datamask & DPDTHETA_MASK) {
        _dpdTheta_out(i) = _dpdTheta(j);
        _uCond_out(i) = _uCond(j);
        _uMech_out(i) = _uMech(j);
        _uChem_out(i) = _uChem(j);
        _uCG_out(i) = _uCG(j);
        _uCGnew_out(i) = _uCGnew(j);
      }
    }
  }
};

}    // namespace LAMMPS_NS

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  set_size_exchange();

  const int nlocal = atomKK->nlocal;
  if (nlocal == 0) return;
  const int nmax = atom->nmax;
  const uint64_t mask = datamask_exchange;

  atomKK->sync(Device,mask);

  // permutation vector: sorted slot i takes its data from old index permute(i)

  auto d_permute = Sorter.get_permute_vector();

  // grow native-typed scratch (only the arrays present in this style) to nmax

  grow_sort_1d(k_tag_sort,nmax);
  grow_sort_1d(k_type_sort,nmax);
  grow_sort_1d(k_mask_sort,nmax);
  grow_sort_1d(k_image_sort,nmax);
  grow_sort_1d(k_x_sort,nmax);
  grow_sort_1d(k_v_sort,nmax);
  if (mask & Q_MASK) grow_sort_1d(k_q_sort,nmax);
  if (mask & MOLECULE_MASK) grow_sort_1d(k_molecule_sort,nmax);
  if (mask & BOND_MASK) {
    grow_sort_1d(k_num_bond_sort,nmax);
    grow_sort_2d(k_bond_type_sort,atomKK->k_bond_type,nmax);
    grow_sort_2d(k_bond_atom_sort,atomKK->k_bond_atom,nmax);
  }
  if (mask & ANGLE_MASK) {
    grow_sort_1d(k_num_angle_sort,nmax);
    grow_sort_2d(k_angle_type_sort,atomKK->k_angle_type,nmax);
    grow_sort_2d(k_angle_atom1_sort,atomKK->k_angle_atom1,nmax);
    grow_sort_2d(k_angle_atom2_sort,atomKK->k_angle_atom2,nmax);
    grow_sort_2d(k_angle_atom3_sort,atomKK->k_angle_atom3,nmax);
  }
  if (mask & DIHEDRAL_MASK) {
    grow_sort_1d(k_num_dihedral_sort,nmax);
    grow_sort_2d(k_dihedral_type_sort,atomKK->k_dihedral_type,nmax);
    grow_sort_2d(k_dihedral_atom1_sort,atomKK->k_dihedral_atom1,nmax);
    grow_sort_2d(k_dihedral_atom2_sort,atomKK->k_dihedral_atom2,nmax);
    grow_sort_2d(k_dihedral_atom3_sort,atomKK->k_dihedral_atom3,nmax);
    grow_sort_2d(k_dihedral_atom4_sort,atomKK->k_dihedral_atom4,nmax);
  }
  if (mask & IMPROPER_MASK) {
    grow_sort_1d(k_num_improper_sort,nmax);
    grow_sort_2d(k_improper_type_sort,atomKK->k_improper_type,nmax);
    grow_sort_2d(k_improper_atom1_sort,atomKK->k_improper_atom1,nmax);
    grow_sort_2d(k_improper_atom2_sort,atomKK->k_improper_atom2,nmax);
    grow_sort_2d(k_improper_atom3_sort,atomKK->k_improper_atom3,nmax);
    grow_sort_2d(k_improper_atom4_sort,atomKK->k_improper_atom4,nmax);
  }
  if (mask & SPECIAL_MASK) {
    grow_sort_2d(k_nspecial_sort,atomKK->k_nspecial,nmax);
    grow_sort_2d(k_special_sort,atomKK->k_special,nmax);
  }
  if (mask & MU_MASK) grow_sort_1d(k_mu_sort,nmax);
  if (mask & SP_MASK) grow_sort_1d(k_sp_sort,nmax);
  if (mask & RADIUS_MASK) grow_sort_1d(k_radius_sort,nmax);
  if (mask & RMASS_MASK) grow_sort_1d(k_rmass_sort,nmax);
  if (mask & OMEGA_MASK) grow_sort_1d(k_omega_sort,nmax);
  if (mask & ANGMOM_MASK) grow_sort_1d(k_angmom_sort,nmax);
  if (mask & DPDTHETA_MASK) {
    grow_sort_1d(k_dpdTheta_sort,nmax);
    grow_sort_1d(k_uCond_sort,nmax);
    grow_sort_1d(k_uMech_sort,nmax);
    grow_sort_1d(k_uChem_sort,nmax);
    grow_sort_1d(k_uCG_sort,nmax);
    grow_sort_1d(k_uCGnew_sort,nmax);
  }

  // single coalesced gather kernel: scratch_*(i) = k_*(permute(i))

  if (size_exchange == size_exchange_default) {
    AtomVecKokkos_GatherSortFunctor<LMPDeviceType,1,decltype(d_permute)>
      f(atomKK,this,d_permute,mask);
    Kokkos::parallel_for(nlocal,f);
  } else {
    AtomVecKokkos_GatherSortFunctor<LMPDeviceType,0,decltype(d_permute)>
      f(atomKK,this,d_permute,mask);
    Kokkos::parallel_for(nlocal,f);
  }

  // swap sorted scratch into the atom arrays (no copy-back) and rebind the
  // legacy raw pointers; the old allocations stay in the scratch for reuse

  swap_sort(memoryKK,atomKK->k_tag,k_tag_sort,atomKK->tag,nmax,"atom:tag");
  swap_sort(memoryKK,atomKK->k_type,k_type_sort,atomKK->type,nmax,"atom:type");
  swap_sort(memoryKK,atomKK->k_mask,k_mask_sort,atomKK->mask,nmax,"atom:mask");
  swap_sort(memoryKK,atomKK->k_image,k_image_sort,atomKK->image,nmax,"atom:image");
  swap_sort(memoryKK,atomKK->k_x,k_x_sort,atomKK->x,nmax,"atom:x");
  swap_sort(memoryKK,atomKK->k_v,k_v_sort,atomKK->v,nmax,"atom:v");
  if (mask & Q_MASK) swap_sort(memoryKK,atomKK->k_q,k_q_sort,atomKK->q,nmax,"atom:q");
  if (mask & MOLECULE_MASK) swap_sort(memoryKK,atomKK->k_molecule,k_molecule_sort,atomKK->molecule,nmax,"atom:molecule");
  if (mask & BOND_MASK) {
    swap_sort(memoryKK,atomKK->k_num_bond,k_num_bond_sort,atomKK->num_bond,nmax,"atom:num_bond");
    swap_sort(memoryKK,atomKK->k_bond_type,k_bond_type_sort,atomKK->bond_type,nmax,"atom:bond_type");
    swap_sort(memoryKK,atomKK->k_bond_atom,k_bond_atom_sort,atomKK->bond_atom,nmax,"atom:bond_atom");
  }
  if (mask & ANGLE_MASK) {
    swap_sort(memoryKK,atomKK->k_num_angle,k_num_angle_sort,atomKK->num_angle,nmax,"atom:num_angle");
    swap_sort(memoryKK,atomKK->k_angle_type,k_angle_type_sort,atomKK->angle_type,nmax,"atom:angle_type");
    swap_sort(memoryKK,atomKK->k_angle_atom1,k_angle_atom1_sort,atomKK->angle_atom1,nmax,"atom:angle_atom1");
    swap_sort(memoryKK,atomKK->k_angle_atom2,k_angle_atom2_sort,atomKK->angle_atom2,nmax,"atom:angle_atom2");
    swap_sort(memoryKK,atomKK->k_angle_atom3,k_angle_atom3_sort,atomKK->angle_atom3,nmax,"atom:angle_atom3");
  }
  if (mask & DIHEDRAL_MASK) {
    swap_sort(memoryKK,atomKK->k_num_dihedral,k_num_dihedral_sort,atomKK->num_dihedral,nmax,"atom:num_dihedral");
    swap_sort(memoryKK,atomKK->k_dihedral_type,k_dihedral_type_sort,atomKK->dihedral_type,nmax,"atom:dihedral_type");
    swap_sort(memoryKK,atomKK->k_dihedral_atom1,k_dihedral_atom1_sort,atomKK->dihedral_atom1,nmax,"atom:dihedral_atom1");
    swap_sort(memoryKK,atomKK->k_dihedral_atom2,k_dihedral_atom2_sort,atomKK->dihedral_atom2,nmax,"atom:dihedral_atom2");
    swap_sort(memoryKK,atomKK->k_dihedral_atom3,k_dihedral_atom3_sort,atomKK->dihedral_atom3,nmax,"atom:dihedral_atom3");
    swap_sort(memoryKK,atomKK->k_dihedral_atom4,k_dihedral_atom4_sort,atomKK->dihedral_atom4,nmax,"atom:dihedral_atom4");
  }
  if (mask & IMPROPER_MASK) {
    swap_sort(memoryKK,atomKK->k_num_improper,k_num_improper_sort,atomKK->num_improper,nmax,"atom:num_improper");
    swap_sort(memoryKK,atomKK->k_improper_type,k_improper_type_sort,atomKK->improper_type,nmax,"atom:improper_type");
    swap_sort(memoryKK,atomKK->k_improper_atom1,k_improper_atom1_sort,atomKK->improper_atom1,nmax,"atom:improper_atom1");
    swap_sort(memoryKK,atomKK->k_improper_atom2,k_improper_atom2_sort,atomKK->improper_atom2,nmax,"atom:improper_atom2");
    swap_sort(memoryKK,atomKK->k_improper_atom3,k_improper_atom3_sort,atomKK->improper_atom3,nmax,"atom:improper_atom3");
    swap_sort(memoryKK,atomKK->k_improper_atom4,k_improper_atom4_sort,atomKK->improper_atom4,nmax,"atom:improper_atom4");
  }
  if (mask & SPECIAL_MASK) {
    swap_sort(memoryKK,atomKK->k_nspecial,k_nspecial_sort,atomKK->nspecial,nmax,"atom:nspecial");
    swap_sort(memoryKK,atomKK->k_special,k_special_sort,atomKK->special,nmax,"atom:special");
  }
  if (mask & MU_MASK) swap_sort(memoryKK,atomKK->k_mu,k_mu_sort,atomKK->mu,nmax,"atom:mu");
  if (mask & SP_MASK) swap_sort(memoryKK,atomKK->k_sp,k_sp_sort,atomKK->sp,nmax,"atom:sp");
  if (mask & RADIUS_MASK) swap_sort(memoryKK,atomKK->k_radius,k_radius_sort,atomKK->radius,nmax,"atom:radius");
  if (mask & RMASS_MASK) swap_sort(memoryKK,atomKK->k_rmass,k_rmass_sort,atomKK->rmass,nmax,"atom:rmass");
  if (mask & OMEGA_MASK) swap_sort(memoryKK,atomKK->k_omega,k_omega_sort,atomKK->omega,nmax,"atom:omega");
  if (mask & ANGMOM_MASK) swap_sort(memoryKK,atomKK->k_angmom,k_angmom_sort,atomKK->angmom,nmax,"atom:angmom");
  if (mask & DPDTHETA_MASK) {
    swap_sort(memoryKK,atomKK->k_dpdTheta,k_dpdTheta_sort,atomKK->dpdTheta,nmax,"atom:dpdTheta");
    swap_sort(memoryKK,atomKK->k_uCond,k_uCond_sort,atomKK->uCond,nmax,"atom:uCond");
    swap_sort(memoryKK,atomKK->k_uMech,k_uMech_sort,atomKK->uMech,nmax,"atom:uMech");
    swap_sort(memoryKK,atomKK->k_uChem,k_uChem_sort,atomKK->uChem,nmax,"atom:uChem");
    swap_sort(memoryKK,atomKK->k_uCG,k_uCG_sort,atomKK->uCG,nmax,"atom:uCG");
    swap_sort(memoryKK,atomKK->k_uCGnew,k_uCGnew_sort,atomKK->uCGnew,nmax,"atom:uCGnew");
  }

  // refresh cached device/host pointers, then mark device as the sorted source

  grow_pointers();
  atomKK->modified(Device,mask);
}

/* ---------------------------------------------------------------------- */

uint64_t AtomVecKokkos::field2mask(std::string field)
{
  if (field == "id")
    return TAG_MASK;
  else if (field == "type")
    return TYPE_MASK;
  else if (field == "mask")
    return MASK_MASK;
  else if (field == "image")
    return IMAGE_MASK;
  else if (field == "x")
    return X_MASK;
  else if (field == "v")
    return V_MASK;
  else if (field == "f")
    return F_MASK;
  else if (field == "rmass")
    return RMASS_MASK;
  else if (field == "q")
    return Q_MASK;
  else if (field == "mu")
    return MU_MASK;
  else if (field == "mu3")
    return MU_MASK;
  else if (field == "radius")
    return RADIUS_MASK;
  else if (field == "angmom")
    return ANGMOM_MASK;
  else if (field == "omega")
    return OMEGA_MASK;
  else if (field == "torque")
    return TORQUE_MASK;
  else if (field == "ellipsoid")
    return ELLIPSOID_MASK;
  else if (field == "molecule")
    return MOLECULE_MASK;
  else if (field == "nspecial")
    return SPECIAL_MASK;
  else if (field == "num_bond")
    return BOND_MASK;
  else if (field == "num_angle")
    return ANGLE_MASK;
  else if (field == "num_dihedral")
    return DIHEDRAL_MASK;
  else if (field == "num_improper")
    return IMPROPER_MASK;
  else if (field == "sp")
    return SP_MASK;
  else if (field == "fm")
    return FM_MASK;
  else if (field == "fm_long")
    return FML_MASK;
  else if (field == "rho") // conflicts with SPH package "rho"
    return DPDRHO_MASK;
  else if (field == "dpdTheta")
    return DPDTHETA_MASK;
  else if (field == "uCond")
    return UCOND_MASK;
  else if (field == "uMech")
    return UMECH_MASK;
  else if (field == "uChem")
    return UCHEM_MASK;
  else if (field == "uCG")
    return UCG_MASK;
  else if (field == "uCGnew")
    return UCGNEW_MASK;
  else if (field == "duChem")
    return DUCHEM_MASK;
  else
    return EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

int AtomVecKokkos::field2size(std::string field)
{
  if (field == "id") return 1;
  else if (field == "type") return 1;
  else if (field == "mask") return 1;
  else if (field == "image") return 1;
  else if (field == "x") return 3;
  else if (field == "v") return 3;
  else if (field == "f") return 3;
  else if (field == "rmass") return 1;
  else if (field == "q") return 1;
  else if (field == "mu") return 4;
  else if (field == "mu3") return 3;
  else if (field == "radius") return 1;
  else if (field == "angmom") return 3;
  else if (field == "omega") return 3;
  else if (field == "torque") return 3;
  else if (field == "ellipsoid") return 1;
  else if (field == "molecule") return 1;
  else if (field == "special") return 3+atom->maxspecial;
  else if (field == "num_bond") return 1+2*atom->bond_per_atom;
  else if (field == "num_angle") return 1+4*atom->angle_per_atom;
  else if (field == "num_dihedral") return 1+5*atom->dihedral_per_atom;
  else if (field == "num_improper") return 1+5*atom->dihedral_per_atom;
  else if (field == "sp") return 4;
  else if (field == "fm") return 3;
  else if (field == "fm_long") return 3;
  else if (field == "rho") return 1;
  else if (field == "dpdTheta") return 1;
  else if (field == "uCond") return 1;
  else if (field == "uMech") return 1;
  else if (field == "uChem") return 1;
  else if (field == "uCG") return 1;
  else if (field == "uCGnew") return 1;
  else if (field == "duChem") return 1;
  else return 0;
}

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::set_atom_masks()
{
  datamask_grow = EMPTY_MASK;
  for (int i = 0; i < default_grow.size(); i++)
    datamask_grow |= field2mask(default_grow[i]);
  for (int i = 0; i < ngrow; i++)
    datamask_grow |= field2mask(fields_grow[i]);

  datamask_comm = datamask_bonus;
  for (int i = 0; i < default_comm.size(); i++)
    datamask_comm |= field2mask(default_comm[i]);
  for (int i = 0; i < ncomm; i++)
    datamask_comm |= field2mask(fields_comm[i]);

  datamask_comm_vel = datamask_bonus;
  for (int i = 0; i < default_comm_vel.size(); i++)
    datamask_comm_vel |= field2mask(default_comm_vel[i]);
  for (int i = 0; i < ncomm_vel; i++)
    datamask_comm_vel |= field2mask(fields_comm_vel[i]);

  datamask_reverse = EMPTY_MASK;
  for (int i = 0; i < default_reverse.size(); i++)
    datamask_reverse |= field2mask(default_reverse[i]);
  for (int i = 0; i < nreverse; i++)
    datamask_reverse |= field2mask(fields_reverse[i]);

  datamask_border = datamask_bonus;
  for (int i = 0; i < default_border.size(); i++)
    datamask_border |= field2mask(default_border[i]);
  for (int i = 0; i < nborder; i++)
    datamask_border |= field2mask(fields_border[i]);

  datamask_border_vel = datamask_bonus;
  for (int i = 0; i < default_border_vel.size(); i++)
    datamask_border_vel |= field2mask(default_border_vel[i]);
  for (int i = 0; i < nborder_vel; i++)
    datamask_border_vel |= field2mask(fields_border_vel[i]);

  datamask_exchange = datamask_bonus;
  for (int i = 0; i < default_exchange.size(); i++)
    datamask_exchange |= field2mask(default_exchange[i]);
  for (int i = 0; i < nexchange; i++)
    datamask_exchange |= field2mask(fields_exchange[i]);
}

/* ---------------------------------------------------------------------- */

void AtomVecKokkos::set_size_exchange()
{
  size_exchange_default = 1; // 1 to store buffer length
  for (int i = 0; i < default_exchange.size(); i++)
    size_exchange_default += field2size(default_exchange[i]);

  size_exchange = size_exchange_default;

  for (int i = 0; i < nexchange; i++)
    size_exchange += field2size(fields_exchange[i]);
}
