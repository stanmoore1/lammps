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

#include "atom_vec_ellipsoid_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm_kokkos.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "math_const.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "modify.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

AtomVecEllipsoidKokkos::AtomVecEllipsoidKokkos(LAMMPS *lmp) : AtomVec(lmp),
AtomVecKokkos(lmp), AtomVecEllipsoid(lmp)
{
  no_border_vel_flag = 0;
  unpack_exchange_indices_flag = 1;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::grow(int n)
{
  auto DELTA = LMP_KOKKOS_AV_DELTA;
  int step = MAX(DELTA,nmax*0.01);
  if (n == 0) nmax += step;
  else nmax = n;
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  atomKK->sync(Device,ALL_MASK);
  atomKK->modified(Device,ALL_MASK);

  memoryKK->grow_kokkos(atomKK->k_tag,atomKK->tag,nmax,"atom:tag");
  memoryKK->grow_kokkos(atomKK->k_type,atomKK->type,nmax,"atom:type");
  memoryKK->grow_kokkos(atomKK->k_mask,atomKK->mask,nmax,"atom:mask");
  memoryKK->grow_kokkos(atomKK->k_image,atomKK->image,nmax,"atom:image");

  memoryKK->grow_kokkos(atomKK->k_x,atomKK->x,nmax,"atom:x");
  memoryKK->grow_kokkos(atomKK->k_v,atomKK->v,nmax,"atom:v");
  memoryKK->grow_kokkos(atomKK->k_f,atomKK->f,nmax,"atom:f");
  memoryKK->grow_kokkos(atomKK->k_rmass,atomKK->rmass,nmax,"atom:rmass");
  memoryKK->grow_kokkos(atomKK->k_angmom,atomKK->angmom,nmax,"atom:angmom");
  memoryKK->grow_kokkos(atomKK->k_torque,atomKK->torque,nmax,"atom:torque");
  memoryKK->grow_kokkos(atomKK->k_ellipsoid,atomKK->ellipsoid,nmax,"atom:ellipsoid");

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);

  grow_pointers();
  atomKK->sync(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::grow_pointers()
{
  tag = atomKK->tag;
  d_tag = atomKK->k_tag.d_view;
  h_tag = atomKK->k_tag.h_view;

  type = atomKK->type;
  d_type = atomKK->k_type.d_view;
  h_type = atomKK->k_type.h_view;
  mask = atomKK->mask;
  d_mask = atomKK->k_mask.d_view;
  h_mask = atomKK->k_mask.h_view;
  image = atomKK->image;
  d_image = atomKK->k_image.d_view;
  h_image = atomKK->k_image.h_view;

  x = atomKK->x;
  d_x = atomKK->k_x.d_view;
  h_x = atomKK->k_x.h_view;
  v = atomKK->v;
  d_v = atomKK->k_v.d_view;
  h_v = atomKK->k_v.h_view;
  f = atomKK->f;
  d_f = atomKK->k_f.d_view;
  h_f = atomKK->k_f.h_view;
  rmass = atomKK->rmass;
  d_rmass = atomKK->k_rmass.d_view;
  h_rmass = atomKK->k_rmass.h_view;
  angmom = atomKK->angmom;
  d_angmom = atomKK->k_angmom.d_view;
  h_angmom = atomKK->k_angmom.h_view;
  torque = atomKK->torque;
  d_torque = atomKK->k_torque.d_view;
  h_torque = atomKK->k_torque.h_view;
  ellipsoid = atomKK->ellipsoid;
  d_ellipsoid= atomKK->k_ellipsoid.d_view;
  h_ellipsoid = atomKK->k_ellipsoid.h_view;
}

/* ----------------------------------------------------------------------
   sort atom arrays on device
------------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  atomKK->sync(Device, ALL_MASK & ~F_MASK & ~TORQUE_MASK & ~ELLIPSOID_MASK);

  Sorter.sort(LMPDeviceType(), d_tag);
  Sorter.sort(LMPDeviceType(), d_type);
  Sorter.sort(LMPDeviceType(), d_mask);
  Sorter.sort(LMPDeviceType(), d_image);
  Sorter.sort(LMPDeviceType(), d_x);
  Sorter.sort(LMPDeviceType(), d_v);
  Sorter.sort(LMPDeviceType(), d_rmass);
  Sorter.sort(LMPDeviceType(), d_angmom);
  //Sorter.sort(LMPDeviceType(), d_ellipsoid); // IS THIS NEEDED, or MASK?

  atomKK->modified(Device, ALL_MASK & ~F_MASK & ~TORQUE_MASK & ~ELLIPSOID_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecEllipsoidKokkos_PackComm {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecEllipsoidKokkos_PackComm(
    const typename DAT::tdual_x_array &x,
    const typename DAT::tdual_float_1d &rmass,
    const typename DAT::tdual_xfloat_2d &buf,
    const typename DAT::tdual_int_2d &list,
    const int & iswap,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
    _x(x.view<DeviceType>()),
    _rmass(rmass.view<DeviceType>()),
    _list(list.view<DeviceType>()),_iswap(iswap),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz) {
    const size_t elements = 4;
    const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_um(buf.view<DeviceType>().data(),maxsend,elements);
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
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
    _buf(i,3) = _rmass(j);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecEllipsoidKokkos::pack_comm_kokkos(
  const int &n,
  const DAT::tdual_int_2d &list,
  const int & iswap,
  const DAT::tdual_xfloat_2d &buf,
  const int &pbc_flag,
  const int* const pbc)
{
  // Check whether to always run forward communication on the host
  // Choose correct forward PackComm kernel
  int n_return = n*size_forward; 
  if (commKK->forward_comm_on_host) {
    atomKK->sync(Host,X_MASK|RMASS_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecEllipsoidKokkos_PackComm<LMPHostType,1,1> f(
          atomKK->k_x,
          atomKK->k_rmass,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecEllipsoidKokkos_PackComm<LMPHostType,1,0> f(
          atomKK->k_x,
          atomKK->k_rmass,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecEllipsoidKokkos_PackComm<LMPHostType,0,1> f(
          atomKK->k_x,
          atomKK->k_rmass,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecEllipsoidKokkos_PackComm<LMPHostType,0,0> f(
          atomKK->k_x,
          atomKK->k_rmass,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
    if (bonus_flag) n_return += pack_comm_bonus_kokkos(
      n, list, buf, iswap, Host); 
  } else {
    atomKK->sync(Device,X_MASK|RMASS_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecEllipsoidKokkos_PackComm<LMPDeviceType,1,1> f(
          atomKK->k_x,
          atomKK->k_rmass,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecEllipsoidKokkos_PackComm<LMPDeviceType,1,0> f(
          atomKK->k_x,
          atomKK->k_rmass,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecEllipsoidKokkos_PackComm<LMPDeviceType,0,1> f(
          atomKK->k_x,
          atomKK->k_rmass,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecEllipsoidKokkos_PackComm<LMPDeviceType,0,0> f(
          atomKK->k_x,
          atomKK->k_rmass,
          buf,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
    if (bonus_flag) n_return += pack_comm_bonus_kokkos(
      n, list, buf, iswap, Host); 
  }
    
  return n_return;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC,int DEFORM_VREMAP>
struct AtomVecEllipsoidKokkos_PackCommVel {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  typename ArrayTypes<DeviceType>::t_v_array _v, _angmom;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];
  X_FLOAT _h_rate[6];
  const int _deform_vremap;

  AtomVecEllipsoidKokkos_PackCommVel(
    const typename DAT::tdual_x_array &x,
    const typename DAT::tdual_int_1d &mask,
    const typename DAT::tdual_float_1d &rmass,
    const typename DAT::tdual_v_array &v,
    const typename DAT::tdual_v_array &angmom,
    const typename DAT::tdual_xfloat_2d &buf,
    const typename DAT::tdual_int_2d &list,
    const int &iswap,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc,
    const double * const h_rate,
    const int &deform_vremap):
    _x(x.view<DeviceType>()),
    _mask(mask.view<DeviceType>()),
    _rmass(rmass.view<DeviceType>()),
    _v(v.view<DeviceType>()),
    _angmom(angmom.view<DeviceType>()),
    _list(list.view<DeviceType>()),_iswap(iswap),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz),
    _deform_vremap(deform_vremap)
  {
    const size_t elements = 9; // !--- HAD RADVARY HERE ---!
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_um(buf.view<DeviceType>().data(),maxsend,elements);
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
    _h_rate[0] = h_rate[0]; _h_rate[1] = h_rate[1]; _h_rate[2] = h_rate[2];
    _h_rate[3] = h_rate[3]; _h_rate[4] = h_rate[4]; _h_rate[5] = h_rate[5];
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
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
    _buf(i,6) = _angmom(j,0);
    _buf(i,7) = _angmom(j,1);
    _buf(i,8) = _angmom(j,2);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecEllipsoidKokkos::pack_comm_vel_kokkos(
  const int &n,
  const DAT::tdual_int_2d &list,
  const int & iswap,
  const DAT::tdual_xfloat_2d &buf,
  const int &pbc_flag,
  const int* const pbc)
{
  if (commKK->forward_comm_on_host) {
    atomKK->sync(Host,X_MASK|RMASS_MASK|V_MASK|ANGMOM_MASK);
    if (pbc_flag) {
      if (deform_vremap) {
        if (domain->triclinic) {
            struct AtomVecEllipsoidKokkos_PackCommVel<LMPHostType,1,1,1> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_rmass,
              atomKK->k_v,atomKK->k_angmom,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        } else {
            struct AtomVecEllipsoidKokkos_PackCommVel<LMPHostType,1,0,1> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_rmass,
              atomKK->k_v,atomKK->k_angmom,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        }
      } else {
        if (domain->triclinic) {
            struct AtomVecEllipsoidKokkos_PackCommVel<LMPHostType,1,1,0> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_rmass,
              atomKK->k_v,atomKK->k_angmom,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        } else {
            struct AtomVecEllipsoidKokkos_PackCommVel<LMPHostType,1,0,0> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_rmass,
              atomKK->k_v,atomKK->k_angmom,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
          struct AtomVecEllipsoidKokkos_PackCommVel<LMPHostType,0,1,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_rmass,
            atomKK->k_v,atomKK->k_angmom,
            buf,list,iswap,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
      } else {
          struct AtomVecEllipsoidKokkos_PackCommVel<LMPHostType,0,0,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_rmass,
            atomKK->k_v,atomKK->k_angmom,
            buf,list,iswap,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
      }
    }
  } else {
    atomKK->sync(Device,X_MASK|RMASS_MASK|V_MASK|ANGMOM_MASK);
    if (pbc_flag) {
      if (deform_vremap) {
        if (domain->triclinic) {
            struct AtomVecEllipsoidKokkos_PackCommVel<LMPDeviceType,1,1,1> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_rmass,
              atomKK->k_v,atomKK->k_angmom,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        } else {
            struct AtomVecEllipsoidKokkos_PackCommVel<LMPDeviceType,1,0,1> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_rmass,
              atomKK->k_v,atomKK->k_angmom,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        }
      } else {
        if (domain->triclinic) {
            struct AtomVecEllipsoidKokkos_PackCommVel<LMPDeviceType,1,1,0> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_rmass,
              atomKK->k_v,atomKK->k_angmom,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        } else {
            struct AtomVecEllipsoidKokkos_PackCommVel<LMPDeviceType,1,0,0> f(
              atomKK->k_x,atomKK->k_mask,
              atomKK->k_rmass,
              atomKK->k_v,atomKK->k_angmom,
              buf,list,iswap,
              domain->xprd,domain->yprd,domain->zprd,
              domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
            Kokkos::parallel_for(n,f);
        }
      }
    } else {
      if (domain->triclinic) {
          struct AtomVecEllipsoidKokkos_PackCommVel<LMPDeviceType,0,1,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_rmass,
            atomKK->k_v,atomKK->k_angmom,
            buf,list,iswap,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
      } else {
          struct AtomVecEllipsoidKokkos_PackCommVel<LMPDeviceType,0,0,0> f(
            atomKK->k_x,atomKK->k_mask,
            atomKK->k_rmass,
            atomKK->k_v,atomKK->k_angmom,
            buf,list,iswap,
            domain->xprd,domain->yprd,domain->zprd,
            domain->xy,domain->xz,domain->yz,pbc,h_rate,deform_vremap);
          Kokkos::parallel_for(n,f);
      }
    }
  }
  return n*(size_forward+size_velocity);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecEllipsoidKokkos_PackCommSelf {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_x_array _xw;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  int _nfirst;
  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecEllipsoidKokkos_PackCommSelf(
    const typename DAT::tdual_x_array &x,
    const typename DAT::tdual_float_1d &rmass,
    const int &nfirst,
    const typename DAT::tdual_int_2d &list,
    const int & iswap,
    const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
    const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
    _x(x.view<DeviceType>()),_xw(x.view<DeviceType>()),
    _rmass(rmass.view<DeviceType>()),
    _nfirst(nfirst),_list(list.view<DeviceType>()),_iswap(iswap),
    _xprd(xprd),_yprd(yprd),_zprd(zprd),
    _xy(xy),_xz(xz),_yz(yz) {
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
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
    _rmass(i+_nfirst) = _rmass(j);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecEllipsoidKokkos::pack_comm_self(
  const int &n, const DAT::tdual_int_2d &list, const int &iswap,
  const int nfirst, const int &pbc_flag, const int* const pbc) {

  if (commKK->forward_comm_on_host) {
    atomKK->sync(Host,X_MASK|RMASS_MASK);
    atomKK->modified(Host,X_MASK|RMASS_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecEllipsoidKokkos_PackCommSelf<LMPHostType,1,1> f(
          atomKK->k_x,
          atomKK->k_rmass,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecEllipsoidKokkos_PackCommSelf<LMPHostType,1,0> f(
          atomKK->k_x,
          atomKK->k_rmass,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecEllipsoidKokkos_PackCommSelf<LMPHostType,0,1> f(
          atomKK->k_x,
          atomKK->k_rmass,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecEllipsoidKokkos_PackCommSelf<LMPHostType,0,0> f(
          atomKK->k_x,
          atomKK->k_rmass,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    atomKK->sync(Device,X_MASK|RMASS_MASK);
    atomKK->modified(Device,X_MASK|RMASS_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecEllipsoidKokkos_PackCommSelf<LMPDeviceType,1,1> f(
          atomKK->k_x,
          atomKK->k_rmass,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecEllipsoidKokkos_PackCommSelf<LMPDeviceType,1,0> f(
          atomKK->k_x,
          atomKK->k_rmass,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecEllipsoidKokkos_PackCommSelf<LMPDeviceType,0,1> f(
          atomKK->k_x,
          atomKK->k_rmass,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      } else {
        struct AtomVecEllipsoidKokkos_PackCommSelf<LMPDeviceType,0,0> f(
          atomKK->k_x,
          atomKK->k_rmass,
          nfirst,list,iswap,
          domain->xprd,domain->yprd,domain->zprd,
          domain->xy,domain->xz,domain->yz,pbc);
        Kokkos::parallel_for(n,f);
      }
    }
  }
  return n*size_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecEllipsoidKokkos_UnpackComm {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um _buf;
  int _first;

  AtomVecEllipsoidKokkos_UnpackComm(
    const typename DAT::tdual_x_array &x,
    const typename DAT::tdual_float_1d &rmass,
    const typename DAT::tdual_xfloat_2d &buf,
    const int& first):
    _x(x.view<DeviceType>()),
    _rmass(rmass.view<DeviceType>()),
    _first(first)
  {
    const size_t elements = 4;
    const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um(buf.view<DeviceType>().data(),maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _rmass(i+_first) = _buf(i,3);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::unpack_comm_kokkos(
  const int &n, const int &first,
  const DAT::tdual_xfloat_2d &buf) {

  if (commKK->forward_comm_on_host) {
    atomKK->modified(Host,X_MASK|RMASS_MASK);
    struct AtomVecEllipsoidKokkos_UnpackComm<LMPHostType> f(
      atomKK->k_x,
      atomKK->k_rmass,
      buf,first);
    Kokkos::parallel_for(n,f);
  } else {
    atomKK->modified(Device,X_MASK|RMASS_MASK);
    struct AtomVecEllipsoidKokkos_UnpackComm<LMPDeviceType> f(
      atomKK->k_x,
      atomKK->k_rmass,
      buf,first);
    Kokkos::parallel_for(n,f);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecEllipsoidKokkos_UnpackCommVel {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  typename ArrayTypes<DeviceType>::t_v_array _v, _angmom;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_const _buf;
  int _first;

  AtomVecEllipsoidKokkos_UnpackCommVel(
    const typename DAT::tdual_x_array &x,
    const typename DAT::tdual_float_1d &rmass,
    const typename DAT::tdual_v_array &v,
    const typename DAT::tdual_v_array &angmom,
    const typename DAT::tdual_xfloat_2d &buf,
    const int& first):
    _x(x.view<DeviceType>()),
    _rmass(rmass.view<DeviceType>()),
    _v(v.view<DeviceType>()),
    _angmom(angmom.view<DeviceType>()),
    _first(first)
  {
    const size_t elements = 9; // !--- HAD RADVARY HERE ---!
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/elements;
    buffer_view<DeviceType>(_buf,buf,maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _v(i+_first,0) = _buf(i,3);
    _v(i+_first,1) = _buf(i,4);
    _v(i+_first,2) = _buf(i,5);
    _angmom(i+_first,0) = _buf(i,6);
    _angmom(i+_first,1) = _buf(i,7);
    _angmom(i+_first,2) = _buf(i,8);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::unpack_comm_vel_kokkos(
  const int &n, const int &first,
  const DAT::tdual_xfloat_2d &buf) {
  if (commKK->forward_comm_on_host) {
    atomKK->modified(Host,X_MASK|RMASS_MASK|V_MASK|ANGMOM_MASK);
      struct AtomVecEllipsoidKokkos_UnpackCommVel<LMPHostType> f(
        atomKK->k_x,
        atomKK->k_rmass,
        atomKK->k_v,atomKK->k_angmom,
        buf,first);
      Kokkos::parallel_for(n,f);
  } else {
    atomKK->modified(Device,X_MASK|RMASS_MASK|V_MASK|ANGMOM_MASK);
      struct AtomVecEllipsoidKokkos_UnpackCommVel<LMPDeviceType> f(
        atomKK->k_x,
        atomKK->k_rmass,
        atomKK->k_v,atomKK->k_angmom,
        buf,first);
      Kokkos::parallel_for(n,f);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG>
struct AtomVecEllipsoidKokkos_PackBorder {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  const typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  const typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  const typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  const typename ArrayTypes<DeviceType>::t_int_1d _type;
  const typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  X_FLOAT _dx,_dy,_dz;

  AtomVecEllipsoidKokkos_PackBorder(
    const typename ArrayTypes<DeviceType>::t_xfloat_2d &buf,
    const typename ArrayTypes<DeviceType>::t_int_2d_const &list,
    const int &iswap,
    const typename ArrayTypes<DeviceType>::t_x_array &x,
    const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
    const typename ArrayTypes<DeviceType>::t_int_1d &type,
    const typename ArrayTypes<DeviceType>::t_int_1d &mask,
    const typename ArrayTypes<DeviceType>::t_float_1d &rmass,
    const X_FLOAT &dx, const X_FLOAT &dy, const X_FLOAT &dz):
    _list(list),_iswap(iswap),
    _x(x),_tag(tag),_type(type),_mask(mask),
    _rmass(rmass),
    _dx(dx),_dy(dy),_dz(dz)
  {
    const size_t elements = 7;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_um(buf.data(),maxsend,elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
    } else {
      _buf(i,0) = _x(j,0) + _dx;
      _buf(i,1) = _x(j,1) + _dy;
      _buf(i,2) = _x(j,2) + _dz;
    }
    _buf(i,3) = d_ubuf(_tag(j)).d;
    _buf(i,4) = d_ubuf(_type(j)).d;
    _buf(i,5) = d_ubuf(_mask(j)).d;
    _buf(i,6) = _rmass(j);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecEllipsoidKokkos::pack_border_kokkos(
  int n, DAT::tdual_int_2d k_sendlist, DAT::tdual_xfloat_2d buf,int iswap,
  int pbc_flag, int *pbc, ExecutionSpace space)
{
  X_FLOAT dx,dy,dz;

  // This was in atom_vec_dpd_kokkos but doesn't appear in any other atom_vec
  atomKK->sync(space,ALL_MASK);

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
    if (space==Host) {
      AtomVecEllipsoidKokkos_PackBorder<LMPHostType,1> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        iswap,h_x,h_tag,h_type,h_mask,h_rmass,
        dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecEllipsoidKokkos_PackBorder<LMPDeviceType,1> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        iswap,d_x,d_tag,d_type,d_mask,d_rmass,
        dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }
  } else {
    dx = dy = dz = 0;
    if (space==Host) {
      AtomVecEllipsoidKokkos_PackBorder<LMPHostType,0> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        iswap,h_x,h_tag,h_type,h_mask,h_rmass,
        dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecEllipsoidKokkos_PackBorder<LMPDeviceType,0> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        iswap,d_x,d_tag,d_type,d_mask,d_rmass,
        dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }
  }
  return n*size_border;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int DEFORM_VREMAP>
struct AtomVecEllipsoidKokkos_PackBorderVel {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  const typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  const int _iswap;
  const typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  const typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  const typename ArrayTypes<DeviceType>::t_int_1d _type;
  const typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  typename ArrayTypes<DeviceType>::t_v_array _v, _angmom;
  X_FLOAT _dx,_dy,_dz, _dvx, _dvy, _dvz;
  const int _deform_groupbit;

  AtomVecEllipsoidKokkos_PackBorderVel(
    const typename ArrayTypes<DeviceType>::t_xfloat_2d &buf,
    const typename ArrayTypes<DeviceType>::t_int_2d_const &list,
    const int &iswap,
    const typename ArrayTypes<DeviceType>::t_x_array &x,
    const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
    const typename ArrayTypes<DeviceType>::t_int_1d &type,
    const typename ArrayTypes<DeviceType>::t_int_1d &mask,
    const typename ArrayTypes<DeviceType>::t_float_1d &rmass,
    const typename ArrayTypes<DeviceType>::t_v_array &v,
    const typename ArrayTypes<DeviceType>::t_v_array &angmom,
    const X_FLOAT &dx, const X_FLOAT &dy, const X_FLOAT &dz,
    const X_FLOAT &dvx, const X_FLOAT &dvy, const X_FLOAT &dvz,
    const int &deform_groupbit):
    _buf(buf),_list(list),_iswap(iswap),
    _x(x),_tag(tag),_type(type),_mask(mask),
    _rmass(rmass),
    _v(v), _angmom(angmom),
    _dx(dx),_dy(dy),_dz(dz),
    _dvx(dvx),_dvy(dvy),_dvz(dvz),
    _deform_groupbit(deform_groupbit)
  {
    const size_t elements = 13;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_um(buf.data(),maxsend,elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(_iswap,i);
    if (PBC_FLAG == 0) {
      _buf(i,0) = _x(j,0);
      _buf(i,1) = _x(j,1);
      _buf(i,2) = _x(j,2);
    } else {
      _buf(i,0) = _x(j,0) + _dx;
      _buf(i,1) = _x(j,1) + _dy;
      _buf(i,2) = _x(j,2) + _dz;
    }
    _buf(i,3) = d_ubuf(_tag(j)).d;
    _buf(i,4) = d_ubuf(_type(j)).d;
    _buf(i,5) = d_ubuf(_mask(j)).d;
    _buf(i,6) = _rmass(j);
    if (DEFORM_VREMAP) {
      if (_mask(i) & _deform_groupbit) {
        _buf(i,7) = _v(j,0) + _dvx;
        _buf(i,8) = _v(j,1) + _dvy;
        _buf(i,9) = _v(j,2) + _dvz;
      }
    }
    else {
      _buf(i,7) = _v(j,0);
      _buf(i,8) = _v(j,1);
      _buf(i,9) = _v(j,2);
    }
    _buf(i,10) = _angmom(j,0);
    _buf(i,11) = _angmom(j,1);
    _buf(i,12) = _angmom(j,2);
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecEllipsoidKokkos::pack_border_vel_kokkos(
  int n, DAT::tdual_int_2d k_sendlist, DAT::tdual_xfloat_2d buf,int iswap,
  int pbc_flag, int *pbc, ExecutionSpace space)
{
  X_FLOAT dx=0,dy=0,dz=0;
  X_FLOAT dvx=0,dvy=0,dvz=0;

  // This was in atom_vec_dpd_kokkos but doesn't appear in any other atom_vec
  atomKK->sync(space,ALL_MASK);

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
      if (space==Host) {
        AtomVecEllipsoidKokkos_PackBorderVel<LMPHostType,1,0> f(
          buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
          iswap,h_x,h_tag,h_type,h_mask,h_rmass,
          h_v, h_angmom,
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecEllipsoidKokkos_PackBorderVel<LMPDeviceType,1,0> f(
          buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
          iswap,d_x,d_tag,d_type,d_mask,d_rmass,
          d_v, d_angmom,
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit);
        Kokkos::parallel_for(n,f);
      }
    }
    else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      if (space==Host) {
        AtomVecEllipsoidKokkos_PackBorderVel<LMPHostType,1,1> f(
          buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
          iswap,h_x,h_tag,h_type,h_mask,h_rmass,
          h_v, h_angmom,
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit);
        Kokkos::parallel_for(n,f);
      } else {
        AtomVecEllipsoidKokkos_PackBorderVel<LMPDeviceType,1,1> f(
          buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
          iswap,d_x,d_tag,d_type,d_mask,d_rmass,
          d_v, d_angmom,
          dx,dy,dz,dvx,dvy,dvz,
          deform_groupbit);
        Kokkos::parallel_for(n,f);
      }
    }
  } else {
    if (space==Host) {
      AtomVecEllipsoidKokkos_PackBorderVel<LMPHostType,0,0> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        iswap,h_x,h_tag,h_type,h_mask,h_rmass,
        h_v, h_angmom,
        dx,dy,dz,dvx,dvy,dvz,
        deform_groupbit);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecEllipsoidKokkos_PackBorderVel<LMPDeviceType,0,0> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        iswap,d_x,d_tag,d_type,d_mask,d_rmass,
        d_v, d_angmom,
        dx,dy,dz,dvx,dvy,dvz,
        deform_groupbit);
      Kokkos::parallel_for(n,f);
    }
  }

  return n*(size_border + size_velocity);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecEllipsoidKokkos_UnpackBorder {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um _buf;
  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  typename ArrayTypes<DeviceType>::t_int_1d _type;
  typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  int _first;

  AtomVecEllipsoidKokkos_UnpackBorder(
    const typename ArrayTypes<DeviceType>::t_xfloat_2d &buf,
    const typename ArrayTypes<DeviceType>::t_x_array &x,
    const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
    const typename ArrayTypes<DeviceType>::t_int_1d &type,
    const typename ArrayTypes<DeviceType>::t_int_1d &mask,
    const typename ArrayTypes<DeviceType>::t_float_1d &rmass,
    const int& first):
    _buf(buf),_x(x),_tag(tag),_type(type),_mask(mask),
    _rmass(rmass),
    _first(first)
  {
    const size_t elements = 7;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um(buf.data(),maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _tag(i+_first) = static_cast<tagint> (d_ubuf(_buf(i,3)).i);
    _type(i+_first) = static_cast<int>  (d_ubuf(_buf(i,4)).i);
    _mask(i+_first) = static_cast<int>  (d_ubuf(_buf(i,5)).i);
    _rmass(i+_first) = _buf(i,6);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::unpack_border_kokkos(const int &n, const int &first,
                                               const DAT::tdual_xfloat_2d &buf,ExecutionSpace space) {
  while (first+n >= nmax) grow(0);
  if (space==Host) {
    struct AtomVecEllipsoidKokkos_UnpackBorder<LMPHostType> f(buf.view<LMPHostType>(),
      h_x,h_tag,h_type,h_mask,
      h_rmass,
      first);
    Kokkos::parallel_for(n,f);
  } else {
    struct AtomVecEllipsoidKokkos_UnpackBorder<LMPDeviceType> f(buf.view<LMPDeviceType>(),
      d_x,d_tag,d_type,d_mask,
      d_rmass,
      first);
    Kokkos::parallel_for(n,f);
  }
  atomKK->modified(space,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|RMASS_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecEllipsoidKokkos_UnpackBorderVel {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um _buf;
  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  typename ArrayTypes<DeviceType>::t_int_1d _type;
  typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_float_1d _rmass;
  typename ArrayTypes<DeviceType>::t_v_array _v;
  typename ArrayTypes<DeviceType>::t_v_array _angmom;
  int _first;

  AtomVecEllipsoidKokkos_UnpackBorderVel(
    const typename ArrayTypes<DeviceType>::t_xfloat_2d_const &buf,
    const typename ArrayTypes<DeviceType>::t_x_array &x,
    const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
    const typename ArrayTypes<DeviceType>::t_int_1d &type,
    const typename ArrayTypes<DeviceType>::t_int_1d &mask,
    const typename ArrayTypes<DeviceType>::t_float_1d &rmass,
    const typename ArrayTypes<DeviceType>::t_v_array &v,
    const typename ArrayTypes<DeviceType>::t_v_array &angmom,
    const int& first):
    _buf(buf),_x(x),_tag(tag),_type(type),_mask(mask),
    _rmass(rmass),
    _v(v), _angmom(angmom),
    _first(first)
  {
    const size_t elements = 13;
    const int maxsend = (buf.extent(0)*buf.extent(1))/elements;
    _buf = typename ArrayTypes<DeviceType>::t_xfloat_2d_const_um(buf.data(),maxsend,elements);
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    _x(i+_first,0) = _buf(i,0);
    _x(i+_first,1) = _buf(i,1);
    _x(i+_first,2) = _buf(i,2);
    _tag(i+_first) = static_cast<tagint> (d_ubuf(_buf(i,3)).i);
    _type(i+_first) = static_cast<int>  (d_ubuf(_buf(i,4)).i);
    _mask(i+_first) = static_cast<int>  (d_ubuf(_buf(i,5)).i);
    _rmass(i+_first) = _buf(i,6);
    _v(i+_first,0) = _buf(i,7);
    _v(i+_first,1) = _buf(i,8);
    _v(i+_first,2) = _buf(i,9);
    _angmom(i+_first,0) = _buf(i,10);
    _angmom(i+_first,1) = _buf(i,11);
    _angmom(i+_first,2) = _buf(i,12);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::unpack_border_vel_kokkos(
  const int &n, const int &first,
  const DAT::tdual_xfloat_2d &buf,ExecutionSpace space) {
  while (first+n >= nmax) grow(0);
  if (space==Host) {
    struct AtomVecEllipsoidKokkos_UnpackBorderVel<LMPHostType> f(buf.view<LMPHostType>(),
      h_x,h_tag,h_type,h_mask,h_rmass,
      h_v, h_angmom,
      first);
    Kokkos::parallel_for(n,f);
  } else {
    struct AtomVecEllipsoidKokkos_UnpackBorderVel<LMPDeviceType> f(buf.view<LMPDeviceType>(),
      d_x,d_tag,d_type,d_mask,d_rmass,
      d_v, d_angmom,
      first);
    Kokkos::parallel_for(n,f);
  }

  atomKK->modified(space,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|
                 RMASS_MASK|V_MASK|ANGMOM_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecEllipsoidKokkos_PackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array_randomread _x;
  typename AT::t_v_array_randomread _v;
  typename AT::t_tagint_1d_randomread _tag;
  typename AT::t_int_1d_randomread _type;
  typename AT::t_int_1d_randomread _mask;
  typename AT::t_imageint_1d_randomread _image;
  typename AT::t_float_1d_randomread _rmass;
  typename AT::t_v_array_randomread _angmom;
  typename AT::t_x_array _xw;
  typename AT::t_v_array _vw;
  typename AT::t_tagint_1d _tagw;
  typename AT::t_int_1d _typew;
  typename AT::t_int_1d _maskw;
  typename AT::t_imageint_1d _imagew;
  typename AT::t_float_1d _rmassw;
  typename AT::t_v_array _angmomw;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d_const _sendlist;
  typename AT::t_int_1d_const _copylist;
  int _size_exchange;

  AtomVecEllipsoidKokkos_PackExchangeFunctor(
    const AtomKokkos* atom,
    const typename AT::tdual_xfloat_2d buf,
    typename AT::tdual_int_1d sendlist,
    typename AT::tdual_int_1d copylist):
    _size_exchange(atom->avecKK->size_exchange),
    _x(atom->k_x.view<DeviceType>()),
    _v(atom->k_v.view<DeviceType>()),
    _tag(atom->k_tag.view<DeviceType>()),
    _type(atom->k_type.view<DeviceType>()),
    _mask(atom->k_mask.view<DeviceType>()),
    _image(atom->k_image.view<DeviceType>()),
    _rmass(atom->k_rmass.view<DeviceType>()),
    _angmom(atom->k_angmom.view<DeviceType>()),
    _xw(atom->k_x.view<DeviceType>()),
    _vw(atom->k_v.view<DeviceType>()),
    _tagw(atom->k_tag.view<DeviceType>()),
    _typew(atom->k_type.view<DeviceType>()),
    _maskw(atom->k_mask.view<DeviceType>()),
    _imagew(atom->k_image.view<DeviceType>()),
    _rmassw(atom->k_rmass.view<DeviceType>()),
    _angmomw(atom->k_angmom.view<DeviceType>()),
    _sendlist(sendlist.template view<DeviceType>()),
    _copylist(copylist.template view<DeviceType>()) {
    const int maxsend = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/_size_exchange;

    _buf = typename AT::t_xfloat_2d_um(buf.template view<DeviceType>().data(),maxsend,_size_exchange);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &mysend) const {
    const int i = _sendlist(mysend);
    _buf(mysend,0) = _size_exchange;
    _buf(mysend,1) = _x(i,0);
    _buf(mysend,2) = _x(i,1);
    _buf(mysend,3) = _x(i,2);
    _buf(mysend,4) = _v(i,0);
    _buf(mysend,5) = _v(i,1);
    _buf(mysend,6) = _v(i,2);
    _buf(mysend,7) = d_ubuf(_tag[i]).d;
    _buf(mysend,8) = d_ubuf(_type[i]).d;
    _buf(mysend,9) = d_ubuf(_mask[i]).d;
    _buf(mysend,10) = d_ubuf(_image[i]).d;
    _buf(mysend,11) = _rmass[i];
    _buf(mysend,12) = _angmom(i,0);
    _buf(mysend,13) = _angmom(i,1);
    _buf(mysend,14) = _angmom(i,2);
    const int j = _copylist(mysend);

    if (j>-1) {
      _xw(i,0) = _x(j,0);
      _xw(i,1) = _x(j,1);
      _xw(i,2) = _x(j,2);
      _vw(i,0) = _v(j,0);
      _vw(i,1) = _v(j,1);
      _vw(i,2) = _v(j,2);
      _tagw[i] = _tag(j);
      _typew[i] = _type(j);
      _maskw[i] = _mask(j);
      _imagew[i] = _image(j);
      _rmassw[i] = _rmass(j);
      _angmomw(i,0) = _angmom(j,0);
      _angmomw(i,1) = _angmom(j,1);
      _angmomw(i,2) = _angmom(j,2);
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecEllipsoidKokkos::pack_exchange_kokkos(
  const int &nsend,
  DAT::tdual_xfloat_2d &k_buf,
  DAT::tdual_int_1d k_sendlist,
  DAT::tdual_int_1d k_copylist,
  ExecutionSpace space)
{
  size_exchange = 15;

  if (nsend > (int) (k_buf.view<LMPHostType>().extent(0)*k_buf.view<LMPHostType>().extent(1))/size_exchange) {
    int newsize = nsend*16/k_buf.view<LMPHostType>().extent(1)+1;
    k_buf.resize(newsize,k_buf.view<LMPHostType>().extent(1));
  }
  atomKK->sync(space,X_MASK | V_MASK | TAG_MASK | TYPE_MASK |
             MASK_MASK | IMAGE_MASK | RMASS_MASK | ANGMOM_MASK);

  if (space == Host) {
    AtomVecEllipsoidKokkos_PackExchangeFunctor<LMPHostType> f(atomKK,k_buf,k_sendlist,k_copylist);
    Kokkos::parallel_for(nsend,f);
  } else {
    AtomVecEllipsoidKokkos_PackExchangeFunctor<LMPDeviceType> f(atomKK,k_buf,k_sendlist,k_copylist);
    Kokkos::parallel_for(nsend,f);
  }
  return nsend*size_exchange;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int OUTPUT_INDICES>
struct AtomVecEllipsoidKokkos_UnpackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array _x;
  typename AT::t_v_array _v;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_imageint_1d _image;
  typename AT::t_float_1d _rmass;
  typename AT::t_v_array _angmom;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d _nlocal;
  typename AT::t_int_1d _indices;
  int _dim;
  X_FLOAT _lo,_hi;
  int _size_exchange;

  AtomVecEllipsoidKokkos_UnpackExchangeFunctor(
    const AtomKokkos* atom,
    const typename AT::tdual_xfloat_2d buf,
    typename AT::tdual_int_1d nlocal,
    typename AT::tdual_int_1d indices,
    int dim, X_FLOAT lo, X_FLOAT hi):
      _size_exchange(atom->avecKK->size_exchange),
      _x(atom->k_x.view<DeviceType>()),
      _v(atom->k_v.view<DeviceType>()),
      _tag(atom->k_tag.view<DeviceType>()),
      _type(atom->k_type.view<DeviceType>()),
      _mask(atom->k_mask.view<DeviceType>()),
      _image(atom->k_image.view<DeviceType>()),
      _rmass(atom->k_rmass.view<DeviceType>()),
      _angmom(atom->k_angmom.view<DeviceType>()),
      _nlocal(nlocal.template view<DeviceType>()),
      _indices(indices.template view<DeviceType>()),
      _dim(dim),
      _lo(lo),_hi(hi)
  {
    const size_t size_exchange = 15;
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/size_exchange;

    buffer_view<DeviceType>(_buf,buf,maxsendlist,size_exchange);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &myrecv) const {
    X_FLOAT x = _buf(myrecv,_dim+1);
    int i = -1;
    if (x >= _lo && x < _hi) {
      i = Kokkos::atomic_fetch_add(&_nlocal(0),1);
      _x(i,0) = _buf(myrecv,1);
      _x(i,1) = _buf(myrecv,2);
      _x(i,2) = _buf(myrecv,3);
      _v(i,0) = _buf(myrecv,4);
      _v(i,1) = _buf(myrecv,5);
      _v(i,2) = _buf(myrecv,6);
      _tag[i] = (tagint) d_ubuf(_buf(myrecv,7)).i;
      _type[i] = (int) d_ubuf(_buf(myrecv,8)).i;
      _mask[i] = (int) d_ubuf(_buf(myrecv,9)).i;
      _image[i] = (imageint) d_ubuf(_buf(myrecv,10)).i;
      _rmass[i] = _buf(myrecv,11);
      _angmom(i,0) = _buf(myrecv,12);
      _angmom(i,1) = _buf(myrecv,13);
      _angmom(i,2) = _buf(myrecv,14);
    }
    if (OUTPUT_INDICES)
      _indices(myrecv) = i;
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecEllipsoidKokkos::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv, int nlocal,
                                                int dim, X_FLOAT lo, X_FLOAT hi, ExecutionSpace space,
                                                DAT::tdual_int_1d &k_indices)
{
  while (nlocal + nrecv/size_exchange >= nmax) grow(0);

  if (space == Host) {
    k_count.h_view(0) = nlocal;
    if (k_indices.h_view.data()) {
      AtomVecEllipsoidKokkos_UnpackExchangeFunctor<LMPHostType,1> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    } else {
      AtomVecEllipsoidKokkos_UnpackExchangeFunctor<LMPHostType,0> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    }
  } else {
    k_count.h_view(0) = nlocal;
    k_count.modify<LMPHostType>();
    k_count.sync<LMPDeviceType>();
    if (k_indices.h_view.data()) {
      AtomVecEllipsoidKokkos_UnpackExchangeFunctor<LMPDeviceType,1> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    } else {
      AtomVecEllipsoidKokkos_UnpackExchangeFunctor<LMPDeviceType,0> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
      Kokkos::parallel_for(nrecv/size_exchange,f);
    }
    k_count.modify<LMPDeviceType>();
    k_count.sync<LMPHostType>();
  }

  atomKK->modified(space,X_MASK | V_MASK | TAG_MASK | TYPE_MASK |
                 MASK_MASK | IMAGE_MASK | RMASS_MASK | ANGMOM_MASK);

  return k_count.h_view(0);
}

/* ---------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::sync(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.sync<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPDeviceType>();
    if (mask & RMASS_MASK) atomKK->k_rmass.sync<LMPDeviceType>();
    if (mask & ANGMOM_MASK) atomKK->k_angmom.sync<LMPDeviceType>();
    if (mask & TORQUE_MASK) atomKK->k_torque.sync<LMPDeviceType>();
    if (mask & ELLIPSOID_MASK) atomKK->k_ellipsoid.sync<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK->k_x.sync<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPHostType>();
    if (mask & RMASS_MASK) atomKK->k_rmass.sync<LMPHostType>();
    if (mask & ANGMOM_MASK) atomKK->k_angmom.sync<LMPHostType>();
    if (mask & TORQUE_MASK) atomKK->k_torque.sync<LMPHostType>();
    if (mask & ELLIPSOID_MASK) atomKK->k_ellipsoid.sync<LMPHostType>();
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::sync_overlapping_device(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x,space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v,space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f,space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type,space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask,space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space);
    if ((mask & RMASS_MASK) && atomKK->k_rmass.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_rmass,space);
    if ((mask & ANGMOM_MASK) && atomKK->k_angmom.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_angmom,space);
    if ((mask & TORQUE_MASK) && atomKK->k_torque.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_torque,space);
    if ((mask & ELLIPSOID_MASK) && atomKK->k_ellipsoid.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_ellipsoid,space);
  } else {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x,space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v,space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f,space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type,space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask,space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space);
    if ((mask & RMASS_MASK) && atomKK->k_rmass.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_rmass,space);
    if ((mask & ANGMOM_MASK) && atomKK->k_angmom.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_angmom,space);
    if ((mask & TORQUE_MASK) && atomKK->k_torque.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_torque,space);
    if ((mask & ELLIPSOID_MASK) && atomKK->k_ellipsoid.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_float_1d>(atomKK->k_ellipsoid,space);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecEllipsoidKokkos::modified(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.modify<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPDeviceType>();
    if (mask & RMASS_MASK) atomKK->k_rmass.modify<LMPDeviceType>();
    if (mask & ANGMOM_MASK) atomKK->k_angmom.modify<LMPDeviceType>();
    if (mask & TORQUE_MASK) atomKK->k_torque.modify<LMPDeviceType>();
    if (mask & ELLIPSOID_MASK) atomKK->k_ellipsoid.modify<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK->k_x.modify<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPHostType>();
    if (mask & RMASS_MASK) atomKK->k_rmass.modify<LMPHostType>();
    if (mask & ANGMOM_MASK) atomKK->k_angmom.modify<LMPHostType>();
    if (mask & TORQUE_MASK) atomKK->k_torque.modify<LMPHostType>();
    if (mask & ELLIPSOID_MASK) atomKK->k_ellipsoid.modify<LMPHostType>();
  }
}
