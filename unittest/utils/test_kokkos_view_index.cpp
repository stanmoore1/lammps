/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Tests for the 64-bit index view types of the KOKKOS package (KKBigView,
// KKBigDualView, KKBigTransformView and the *_big and neighbors_2d typedefs
// in kokkos_type.h).
//
// Since Kokkos 5.3 there is no legacy view implementation anymore, and a view
// with the classic Kokkos::LayoutLeft/LayoutRight layouts computes the offset
// of view(i,j) in the common type of the index arguments.  With int indices
// this wraps around once the view holds more than 2^31 elements.  The views
// that can get that large (neighbor lists, ReaxFF bond lists, neighbor history,
// ...) are therefore declared with an explicit 64-bit index type.  These tests
// check that the index type survives all the derived types that LAMMPS uses,
// and that the offsets are correct beyond 2^31 elements.
//
// The large views are host views allocated without initialization, so only
// the address space is reserved and no memory is touched apart from the two
// elements that are written.

#include "lmptype.h"

#ifdef LMP_KOKKOS

#include "KOKKOS/kokkos_type.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <type_traits>

using LAMMPS_NS::bigint;

namespace {

template <class V> constexpr bool index_is_64bit = std::is_same_v<typename V::index_type, int64_t>;

// ----------------------------------------------------------------------
// compile time checks: every derived type keeps the 64-bit index type

template <class AT> constexpr bool check_array_types()
{
  static_assert(index_is_64bit<typename AT::t_neighbors_2d>);
  static_assert(index_is_64bit<typename AT::t_neighbors_2d_const>);
  static_assert(index_is_64bit<typename AT::t_neighbors_2d_um>);
  static_assert(index_is_64bit<typename AT::t_neighbors_2d_const_um>);
  static_assert(index_is_64bit<typename AT::t_neighbors_2d_randomread>);
  static_assert(index_is_64bit<typename AT::t_neighbors_2d_lr>);
  static_assert(index_is_64bit<typename AT::t_neighbors_2d_lr_randomread>);

  static_assert(index_is_64bit<typename AT::t_int_2d_big>);
  static_assert(index_is_64bit<typename AT::t_tagint_2d_big>);
  static_assert(index_is_64bit<typename AT::t_kkfloat_2d_big>);
  static_assert(index_is_64bit<typename AT::t_kkfloat_2d_dl_big>);
  static_assert(index_is_64bit<typename AT::t_double_2d_lr_big>);
  static_assert(index_is_64bit<typename AT::t_kkfloat_2d_lr_big>);
  static_assert(index_is_64bit<typename AT::t_int_2d_lr_big>);
  static_assert(index_is_64bit<typename AT::t_int_3d_lr_big>);
  static_assert(index_is_64bit<typename AT::t_double_3d_lr_big>);

  // the dual views: their device and host views, also the const ones
  static_assert(index_is_64bit<typename AT::tdual_int_2d_big::t_dev>);
  static_assert(index_is_64bit<typename AT::tdual_int_2d_big::t_host>);
  static_assert(index_is_64bit<typename AT::tdual_int_2d_big::t_dev_const>);
  static_assert(index_is_64bit<typename AT::tdual_int_2d_big::t_host_const>);
  static_assert(index_is_64bit<typename AT::tdual_tagint_2d_big::t_dev>);
  static_assert(index_is_64bit<typename AT::tdual_kkfloat_2d_big::t_dev>);
  static_assert(index_is_64bit<typename AT::tdual_double_2d_lr_big::t_host>);
  static_assert(index_is_64bit<typename AT::tdual_int_2d_lr_big::t_dev>);
  static_assert(index_is_64bit<typename AT::tdual_int_3d_lr_big::t_dev>);
  static_assert(index_is_64bit<typename AT::tdual_double_3d_lr_big::t_host>);

  // the device view of a dual view is the matching KKBigView, so they can be
  // assigned to each other
  static_assert(std::is_same_v<typename AT::tdual_int_2d_big::t_dev, typename AT::t_int_2d_big>);
  static_assert(
      std::is_same_v<typename AT::tdual_kkfloat_2d_big::t_dev, typename AT::t_kkfloat_2d_big>);
  return true;
}

static_assert(check_array_types<DAT>());
static_assert(check_array_types<HAT>());

// the TransformView wrappers
static_assert(index_is_64bit<DAT::ttransform_int_2d_big::kk_view::t_dev>);
static_assert(index_is_64bit<DAT::ttransform_int_2d_big::kk_view::t_host>);
static_assert(index_is_64bit<DAT::ttransform_tagint_2d_big::kk_view::t_dev>);
static_assert(index_is_64bit<DAT::ttransform_kkfloat_2d_big::kk_view::t_dev>);
static_assert(index_is_64bit<DAT::ttransform_kkfloat_2d_lr_big::kk_view::t_dev>);
// the host view that the legacy pointer arrays alias, i.e. what view_host()
// returns, indexes with 64-bit arithmetic too
static_assert(index_is_64bit<DAT::ttransform_int_2d_big::legacy_view>);
static_assert(index_is_64bit<DAT::ttransform_tagint_2d_big::legacy_view>);
static_assert(index_is_64bit<DAT::ttransform_kkfloat_2d_big::legacy_view>);
static_assert(index_is_64bit<DAT::ttransform_kkfloat_2d_lr_big::legacy_view>);
static_assert(
    std::is_same_v<DAT::ttransform_kkfloat_2d_lr_big::array_layout, Kokkos::LayoutRight>);
// pinned_mirror_type, the staging view for copies through a pinned host
// buffer, keeps the regular index type: it is only ever the source or target
// of a whole-view Kokkos::deep_copy, which switches to 64-bit loop indices by
// itself once a view spans more than 2^31 elements

// the regular view types stay as they are
static_assert(!index_is_64bit<DAT::t_int_2d>);

// ----------------------------------------------------------------------
// run time checks

// 70000 x 40000 = 2.8e9 elements, more than 2^31
constexpr int N0 = 70000;
constexpr int N1 = 40000;

template <class V> bigint offset_of_last(const V &v)
{
  const int i = static_cast<int>(v.extent(0)) - 1;
  const int j = static_cast<int>(v.extent(1)) - 1;
  return static_cast<bigint>(&v(i, j) - v.data());
}

template <class V> bigint expected_offset_of_last(const V &v)
{
  const bigint i = v.extent(0) - 1;
  const bigint j = v.extent(1) - 1;
  if (v.stride(1) == 1) return i * v.stride(0) + j;
  return j * v.stride(1) + i;
}

class KokkosViewIndex : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
      initialized_here = true;
    }
  }
  static void TearDownTestSuite()
  {
    if (initialized_here) Kokkos::finalize();
  }
  static bool initialized_here;
};

bool KokkosViewIndex::initialized_here = false;

TEST_F(KokkosViewIndex, small_views)
{
  // small views give the same offsets with either kind of indexing
  HAT::t_int_2d_big big("big", 7, 5);
  HAT::t_int_2d small("small", 7, 5);
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(&big(i, j) - big.data(), &small(i, j) - small.data());
      big(i, j) = small(i, j) = 10 * i + j;
    }

  // a dual view with 64-bit indexing works like any other dual view
  DAT::tdual_int_2d_big k_view("k_view", 7, 5);
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 5; ++j) k_view.view_host()(i, j) = big(i, j);
  k_view.modify_host();
  k_view.sync_device();
  k_view.modify_device();
  k_view.sync_host();
  k_view.resize(9, 5);
  EXPECT_EQ(k_view.extent(0), 9U);
  EXPECT_EQ(k_view.extent(1), 5U);
  EXPECT_EQ(k_view.view_host()(6, 4), 64);
  DAT::t_int_2d_big d_view = k_view.view_device();
  DAT::t_int_2d_big_const d_view_const = d_view;
  EXPECT_EQ(d_view_const.extent(0), 9U);
}

TEST_F(KokkosViewIndex, transform_view)
{
  DAT::ttransform_kkfloat_2d_lr_big k_vectors("k_vectors", 4, 3);
  auto h_view = k_vectors.view_host();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j) h_view(i, j) = i + 0.5 * j;
  k_vectors.modify_host();
  k_vectors.sync_device();
  k_vectors.modify_device();
  k_vectors.sync_host();
  EXPECT_DOUBLE_EQ(k_vectors.view_host()(3, 2), 4.0);

  // resize keeps the data and the 64-bit host view
  k_vectors.resize(6, 3);
  EXPECT_DOUBLE_EQ(k_vectors.view_host()(3, 2), 4.0);
  EXPECT_EQ(k_vectors.view_host().extent(0), 6U);
}

TEST_F(KokkosViewIndex, big_host_views)
{
  using LayoutLR = Kokkos::LayoutRight;
  using HostDevice = typename KKDevice<LMPHostType>::value;
  KKBigView<int8_t, 2, LayoutLR, HostDevice> v_lr(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "v_lr"), N0, N1);
  KKBigView<int8_t, 2, Kokkos::LayoutLeft, HostDevice> v_ll(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "v_ll"), N0, N1);
  ASSERT_GT(v_lr.span(), static_cast<size_t>(MAXSMALLINT));
  EXPECT_EQ(offset_of_last(v_lr), expected_offset_of_last(v_lr));
  EXPECT_EQ(offset_of_last(v_ll), expected_offset_of_last(v_ll));
  EXPECT_EQ(offset_of_last(v_lr), static_cast<bigint>(N0) * N1 - 1);

  // the first and the last element are distinct
  v_lr(0, 0) = 1;
  v_lr(N0 - 1, N1 - 1) = 2;
  EXPECT_EQ(v_lr(0, 0), 1);
  EXPECT_EQ(v_lr(N0 - 1, N1 - 1), 2);

  // the same for the host views of a dual view
  KKBigDualView<int8_t, 2, LayoutLR, HostDevice> k_view(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "k_view"), N0, N1);
  EXPECT_EQ(offset_of_last(k_view.view_host()), expected_offset_of_last(k_view.view_host()));
  EXPECT_EQ(offset_of_last(k_view.view_device()), expected_offset_of_last(k_view.view_device()));

  // the views in memory_kokkos.h that alias legacy pointer arrays are regular
  // views; they are indexed with a 64-bit first index to get the same result
  Kokkos::View<int8_t **, LayoutLR, HostDevice> v_regular(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "v_regular"), N0, N1);
  EXPECT_EQ(&v_regular((bigint) N0 - 1, N1 - 1) - v_regular.data(),
            static_cast<bigint>(N0) * N1 - 1);
}

} // namespace

#endif
