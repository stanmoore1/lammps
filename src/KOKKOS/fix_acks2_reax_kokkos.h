/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(acks2/reax/kk,FixACKS2ReaxKokkos<LMPDeviceType>)
FixStyle(acks2/reax/kk/device,FixACKS2ReaxKokkos<LMPDeviceType>)
FixStyle(acks2/reax/kk/host,FixACKS2ReaxKokkos<LMPHostType>)

#else

#ifndef LMP_FIX_ACKS2_REAX_KOKKOS_H
#define LMP_FIX_ACKS2_REAX_KOKKOS_H

#include "fix_acks2_reax.h"
#include "kokkos_type.h"
#include "neigh_list.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

struct TagSparseMatvec1 {};
struct TagSparseMatvec2 {};
struct TagSparseMatvec3 {};
struct TagZeroQGhosts{};

template<class DeviceType>
class FixACKS2ReaxKokkos : public FixACKS2Reax {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  FixACKS2ReaxKokkos(class LAMMPS *, int, char **);
  ~FixACKS2ReaxKokkos();

  void cleanup_copy();
  void init();
  void setup_pre_force(int);
  void pre_force(int);

  KOKKOS_INLINE_FUNCTION
  void num_neigh_item(int, int&) const;

  KOKKOS_INLINE_FUNCTION
  void zero_item(int) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void compute_h_item(int, int &, const bool &) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void compute_h_team(const typename Kokkos::TeamPolicy <DeviceType> ::member_type &team, int, int) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void compute_x_item(int, int &, const bool &) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void compute_x_team(const typename Kokkos::TeamPolicy <DeviceType> ::member_type &team, int, int) const;

  KOKKOS_INLINE_FUNCTION
  void matvec_item(int) const;

  KOKKOS_INLINE_FUNCTION
  void sparse12_item(int) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void sparse13_item(int) const;

  KOKKOS_INLINE_FUNCTION
  void sparse22_item(int) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void sparse23_item(int) const;

  KOKKOS_INLINE_FUNCTION
  void sparse32_item(int) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void sparse33_item(int) const;

  typedef typename Kokkos::TeamPolicy <DeviceType, TagSparseMatvec1> ::member_type membertype1;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagSparseMatvec1, const membertype1 &team) const;

  typedef typename Kokkos::TeamPolicy <DeviceType, TagSparseMatvec2> ::member_type membertype2;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagSparseMatvec2, const membertype2 &team) const;

  typedef typename Kokkos::TeamPolicy <DeviceType, TagSparseMatvec3> ::member_type membertype3;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagSparseMatvec3, const membertype3 &team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagZeroQGhosts, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void vecsum2_item(int) const;

  KOKKOS_INLINE_FUNCTION
  double norm1_item(int) const;

  KOKKOS_INLINE_FUNCTION
  double norm2_item(int) const;

  KOKKOS_INLINE_FUNCTION
  double dot1_item(int) const;

  KOKKOS_INLINE_FUNCTION
  double dot2_item(int) const;

  KOKKOS_INLINE_FUNCTION
  void precon1_item(int) const;

  KOKKOS_INLINE_FUNCTION
  void precon2_item(int) const;

  KOKKOS_INLINE_FUNCTION
  double precon_item(int) const;

  KOKKOS_INLINE_FUNCTION
  double vecacc1_item(int) const;

  KOKKOS_INLINE_FUNCTION
  double vecacc2_item(int) const;

  KOKKOS_INLINE_FUNCTION
  void calculate_q_item(int) const;

  KOKKOS_INLINE_FUNCTION
  double calculate_H_k(const F_FLOAT &r, const F_FLOAT &shld) const;

  struct params_acks2{
    KOKKOS_INLINE_FUNCTION
    params_acks2(){chi=0;eta=0;gamma=0;b_s_acks2=0;,refcharge=0;};
    KOKKOS_INLINE_FUNCTION
    params_acks2(int i){chi=0;eta=0;gamma=0;b_s_acks2=0;,refcharge=0;};
    F_FLOAT chi, eta, gamma, b_s_acks2, refcharge;
  };

  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();

 private:
  int inum;
  int allocated_flag;
  int need_dup;

  typename AT::t_int_scalar d_mfill_offset;

  typedef Kokkos::DualView<int***,DeviceType> tdual_int_1d;
  Kokkos::DualView<params_acks2*,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_acks2*, Kokkos::LayoutRight,DeviceType>::t_dev_const params;

  typename ArrayTypes<DeviceType>::t_x_array x;
  typename ArrayTypes<DeviceType>::t_v_array v;
  typename ArrayTypes<DeviceType>::t_f_array_const f;
  //typename ArrayTypes<DeviceType>::t_float_1d_randomread mass, q;
  typename ArrayTypes<DeviceType>::t_float_1d_randomread mass;
  typename ArrayTypes<DeviceType>::t_float_1d q;
  typename ArrayTypes<DeviceType>::t_int_1d type, mask;
  typename ArrayTypes<DeviceType>::t_tagint_1d tag;

  DAT::tdual_float_1d k_q;
  typename AT::t_float_1d d_q;
  HAT::t_float_1d h_q;

  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors;
  typename ArrayTypes<DeviceType>::t_int_1d_randomread d_ilist, d_numneigh;

  DAT::tdual_ffloat_1d k_tap;
  typename AT::t_ffloat_1d d_tap;

  typename AT::t_int_1d d_firstnbr;
  typename AT::t_int_1d d_numnbrs;
  typename AT::t_int_1d d_jlist;
  typename AT::t_ffloat_1d d_val;

  typename AT::t_int_1d d_firstnbr_X;
  typename AT::t_int_1d d_numnbrs_X;
  typename AT::t_int_1d d_jlist_X;
  typename AT::t_ffloat_1d d_val_X;

  DAT::tdual_ffloat_1d k_t, k_s;
  typename AT::t_ffloat_1d d_Hdia_inv, d_b_s, d_b_t, d_t, d_s;
  HAT::t_ffloat_1d h_t, h_s;
  typename AT::t_ffloat_1d_randomread r_b_s, r_b_t, r_t, r_s;

  DAT::tdual_ffloat_1d k_o, k_d;
  typename AT::t_ffloat_1d d_p, d_o, d_r, d_d;
  HAT::t_ffloat_1d h_o, h_d;
  typename AT::t_ffloat_1d_randomread r_p, r_o, r_r, r_d;

  DAT::tdual_ffloat_2d k_shield, k_s_hist, k_t_hist;
  typename AT::t_ffloat_2d d_shield, d_s_hist, d_t_hist;
  HAT::t_ffloat_2d h_s_hist, h_t_hist;
  typename AT::t_ffloat_2d_randomread r_s_hist, r_t_hist;

  Kokkos::Experimental::ScatterView<F_FLOAT*, typename AT::t_ffloat_1d::array_layout, typename KKDevice<DeviceType>::value, Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated> dup_o;
  Kokkos::Experimental::ScatterView<F_FLOAT*, typename AT::t_ffloat_1d::array_layout, typename KKDevice<DeviceType>::value, Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated> ndup_o;

  void init_shielding_k();
  void init_hist();
  void allocate_matrix();
  void allocate_array();
  void bicgstab_solve();
  void calculate_q();

  int neighflag, pack_flag;
  int nlocal,nall,nmax,newton_pair;
  int count, isuccess;
  double alpha, beta, delta, cutsq;

  int iswap;
  int first;
  typename AT::t_int_2d d_sendlist;
  typename AT::t_xfloat_1d_um v_buf;

  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
};

template <class DeviceType>
struct FixACKS2ReaxKokkosNumNeighFunctor  {
  typedef DeviceType  device_type ;
  typedef int value_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosNumNeighFunctor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, int &maxneigh) const {
    c.num_neigh_item(ii, maxneigh);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosMatVecFunctor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosMatVecFunctor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.matvec_item(ii);
  }
};

template <class DeviceType, int NEIGHFLAG>
struct FixACKS2ReaxKokkosComputeHFunctor {
  int atoms_per_team, vector_length;
  typedef int value_type;
  typedef Kokkos::ScratchMemorySpace<DeviceType> scratch_space;
  FixACKS2ReaxKokkos<DeviceType> c;

  FixACKS2ReaxKokkosComputeHFunctor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };

  FixACKS2ReaxKokkosComputeHFunctor(FixACKS2ReaxKokkos<DeviceType> *c_ptr,
                                  int _atoms_per_team, int _vector_length)
      : c(*c_ptr), atoms_per_team(_atoms_per_team),
        vector_length(_vector_length) {
    c.cleanup_copy();
  };

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, int &m_fill, const bool &final) const {
    c.template compute_h_item<NEIGHFLAG>(ii,m_fill,final);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(
      const typename Kokkos::TeamPolicy<DeviceType>::member_type &team) const {
    c.template compute_h_team<NEIGHFLAG>(team, atoms_per_team, vector_length);
  }

  size_t team_shmem_size(int team_size) const {
    size_t shmem_size =
        Kokkos::View<int *, scratch_space, Kokkos::MemoryUnmanaged>::shmem_size(
            atoms_per_team) + // s_ilist
        Kokkos::View<int *, scratch_space, Kokkos::MemoryUnmanaged>::shmem_size(
            atoms_per_team) + // s_numnbrs
        Kokkos::View<int *, scratch_space, Kokkos::MemoryUnmanaged>::shmem_size(
            atoms_per_team) + // s_firstnbr
        Kokkos::View<int **, scratch_space, Kokkos::MemoryUnmanaged>::
            shmem_size(atoms_per_team, vector_length) + // s_jtype
        Kokkos::View<int **, scratch_space, Kokkos::MemoryUnmanaged>::
            shmem_size(atoms_per_team, vector_length) + // s_j
        Kokkos::View<F_FLOAT **, scratch_space,
                     Kokkos::MemoryUnmanaged>::shmem_size(atoms_per_team,
                                                          vector_length); // s_r
    return shmem_size;
  }
};

template <class DeviceType, int NEIGHFLAG>
struct FixACKS2ReaxKokkosComputeXFunctor {
  int atoms_per_team, vector_length;
  typedef int value_type;
  typedef Kokkos::ScratchMemorySpace<DeviceType> scratch_space;
  FixACKS2ReaxKokkos<DeviceType> c;

  FixACKS2ReaxKokkosComputeXFunctor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };

  FixACKS2ReaxKokkosComputeXFunctor(FixACKS2ReaxKokkos<DeviceType> *c_ptr,
                                  int _atoms_per_team, int _vector_length)
      : c(*c_ptr), atoms_per_team(_atoms_per_team),
        vector_length(_vector_length) {
    c.cleanup_copy();
  };

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, int &m_fill, const bool &final) const {
    c.template compute_x_item<NEIGHFLAG>(ii,m_fill,final);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(
      const typename Kokkos::TeamPolicy<DeviceType>::member_type &team) const {
    c.template compute_x_team<NEIGHFLAG>(team, atoms_per_team, vector_length);
  }

  size_t team_shmem_size(int team_size) const {
    size_t shmem_size =
        Kokkos::View<int *, scratch_space, Kokkos::MemoryUnmanaged>::shmem_size(
            atoms_per_team) + // s_ilist
        Kokkos::View<int *, scratch_space, Kokkos::MemoryUnmanaged>::shmem_size(
            atoms_per_team) + // s_numnbrs
        Kokkos::View<int *, scratch_space, Kokkos::MemoryUnmanaged>::shmem_size(
            atoms_per_team) + // s_firstnbr
        Kokkos::View<int **, scratch_space, Kokkos::MemoryUnmanaged>::
            shmem_size(atoms_per_team, vector_length) + // s_jtype
        Kokkos::View<int **, scratch_space, Kokkos::MemoryUnmanaged>::
            shmem_size(atoms_per_team, vector_length) + // s_j
        Kokkos::View<F_FLOAT **, scratch_space,
                     Kokkos::MemoryUnmanaged>::shmem_size(atoms_per_team,
                                                          vector_length); // s_r
    return shmem_size;
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosZeroFunctor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosZeroFunctor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.zero_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosSparse12Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosSparse12Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.sparse12_item(ii);
  }
};

template <class DeviceType,int NEIGHFLAG>
struct FixACKS2ReaxKokkosSparse13Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosSparse13Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.template sparse13_item<NEIGHFLAG>(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosSparse22Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosSparse22Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.sparse22_item(ii);
  }
};

template <class DeviceType,int NEIGHFLAG>
struct FixACKS2ReaxKokkosSparse23Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosSparse23Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.template sparse23_item<NEIGHFLAG>(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosSparse32Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosSparse32Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.sparse32_item(ii);
  }
};

template <class DeviceType,int NEIGHFLAG>
struct FixACKS2ReaxKokkosSparse33Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosSparse33Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.template sparse33_item<NEIGHFLAG>(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosVecSum2Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosVecSum2Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.vecsum2_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosNorm1Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  typedef double value_type;
  FixACKS2ReaxKokkosNorm1Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, value_type &tmp) const {
    tmp += c.norm1_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosNorm2Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  typedef double value_type;
  FixACKS2ReaxKokkosNorm2Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, value_type &tmp) const {
    tmp += c.norm2_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosDot1Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  typedef double value_type;
  FixACKS2ReaxKokkosDot1Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, value_type &tmp) const {
    tmp += c.dot1_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosDot2Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  typedef double value_type;
  FixACKS2ReaxKokkosDot2Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, value_type &tmp) const {
    tmp += c.dot2_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosPrecon1Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosPrecon1Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.precon1_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosPrecon2Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosPrecon2Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.precon2_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosPreconFunctor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  typedef double value_type;
  FixACKS2ReaxKokkosPreconFunctor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, value_type &tmp) const {
    tmp += c.precon_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosVecAcc1Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  typedef double value_type;
  FixACKS2ReaxKokkosVecAcc1Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, value_type &tmp) const {
    tmp += c.vecacc1_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosVecAcc2Functor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  typedef double value_type;
  FixACKS2ReaxKokkosVecAcc2Functor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii, value_type &tmp) const {
    tmp += c.vecacc2_item(ii);
  }
};

template <class DeviceType>
struct FixACKS2ReaxKokkosCalculateQFunctor  {
  typedef DeviceType  device_type ;
  FixACKS2ReaxKokkos<DeviceType> c;
  FixACKS2ReaxKokkosCalculateQFunctor(FixACKS2ReaxKokkos<DeviceType>* c_ptr):c(*c_ptr) {
    c.cleanup_copy();
  };
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    c.calculate_q_item(ii);
  }
};

}

#endif
#endif
