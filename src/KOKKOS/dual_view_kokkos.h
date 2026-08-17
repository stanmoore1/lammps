/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_DUAL_VIEW_KOKKOS_H
#define LMP_DUAL_VIEW_KOKKOS_H

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace LAMMPS_NS {

#ifndef LMP_KOKKOS_DEBUG_SYNC

// Production builds use Kokkos::DualView unchanged.  This is a type alias rather
// than a class, so every dual view in the package keeps exactly the type, layout
// and generated code it would have if Kokkos::DualView were spelled directly.

template <class DataType, class... Properties>
using DualView = Kokkos::DualView<DataType, Properties...>;

template <class... Args>
auto subview(Args &&...args)
{
  return Kokkos::subview(std::forward<Args>(args)...);
}

#else

/* ----------------------------------------------------------------------
   Sync-debugging dual view.

   Kokkos turns off its own coherence state machine whenever the host and device
   device_types match: sync(), modify() and their named variants return
   immediately and the two views share a single allocation.  That is every
   CPU-only build, which is why a missing sync() or modify() -- silent data
   corruption on a GPU -- cannot be observed without one.

   When that happens this class allocates a second buffer for the host side and
   drives the coherence state machine itself, so the host/device edge behaves the
   way it does on a GPU and the same bugs become reproducible on the CPU.  On a
   real GPU backend the two sides are already distinct and everything is
   forwarded to the base class unchanged.

   Constraints this class has to respect:
   - LAMMPS styles are copied by value into device functors (see copymode), and
     they hold dual views as members.  So no member may have a non-trivial
     destructor, and the coherence flags live in a Kokkos::View so that copies
     share them by reference -- the same reason Kokkos keeps its own
     modified_flags in a View rather than in plain ints.
   - view<Device>() is callable from device code, so it may not do host-only work.
------------------------------------------------------------------------- */

template <class DataType, class... Properties>
class DualView : public Kokkos::DualView<DataType, Properties...> {
 public:
  using base_type = Kokkos::DualView<DataType, Properties...>;
  using t_dev = typename base_type::t_dev;
  using t_host = typename base_type::t_host;

  // true when Kokkos would alias the two sides, i.e. when LAMMPS has to provide
  // the second allocation and the state machine itself

  static constexpr bool SPLIT = base_type::impl_dualview_is_single_device;

  using t_lmp_flags = Kokkos::View<unsigned int[2], Kokkos::LayoutLeft, Kokkos::HostSpace>;

 private:
  // The extra allocation is given to the HOST side, not the device side, and the
  // base class views are left to serve as the device side.  That ordering
  // matters: Kokkos::subview() of a dual view slices the base class views, and
  // every such subview in the package is consumed by a device kernel, so slicing
  // the base has to yield device data.  Splitting the device side instead would
  // hand those subviews a buffer the device never wrote to.
  t_host h_split;

  // lmp_flags(0) counts modifications of the host side, lmp_flags(1) of the
  // device side, exactly like Kokkos::DualView::modified_flags.  Held in a View
  // so that copies of this object share one set of counters.
  t_lmp_flags lmp_flags;

  void allocate_split()
  {
    if constexpr (SPLIT) {
      if (!base_type::view_host().data()) return;
      // create_mirror always allocates, unlike create_mirror_view
      h_split = Kokkos::create_mirror(base_type::view_host());
      Kokkos::deep_copy(h_split, base_type::view_host());
    }
  }

 public:
  DualView() : base_type() {}

  template <class... Args>
  DualView(const std::string &label, Args... args) : base_type(label, args...)
  {
    lmp_flags = t_lmp_flags("LAMMPS::DualView::lmp_flags");
    allocate_split();
  }

  template <class... P, class... Args>
  DualView(const Kokkos::Impl::ViewCtorProp<P...> &prop, Args... args) : base_type(prop, args...)
  {
    lmp_flags = t_lmp_flags("LAMMPS::DualView::lmp_flags");
    allocate_split();
  }

  // Conversion from a plain Kokkos::DualView, needed because Kokkos::subview()
  // deduces and returns the base type.  This has to be a template rather than
  // take base_type directly: subview() spells the space as a device_type, so it
  // hands back Kokkos::DualView<int*,LayoutRight,Device<Serial,HostSpace>> where
  // base_type is Kokkos::DualView<int*,LayoutRight,Serial>.  Those are distinct
  // types for overload resolution even though either can be built from the
  // other, so accept anything the base class itself accepts.
  //
  // Such a subview shares the base class buffers, so it sees the same device
  // data as its parent, but it gets a host buffer and coherence counters of its
  // own.  That is wrong for a subview and callers must use LAMMPS_NS::subview()
  // below instead, which slices both sides; this constructor exists only so that
  // an unconverted call site keeps compiling.

  template <class DT, class... DP,
            class = std::enable_if_t<
                std::is_constructible_v<Kokkos::DualView<DataType, Properties...>,
                                        const Kokkos::DualView<DT, DP...> &>>>
  DualView(const Kokkos::DualView<DT, DP...> &src) : base_type(src)
  {
    lmp_flags = t_lmp_flags("LAMMPS::DualView::lmp_flags");
    allocate_split();
  }

  // Build from already sliced buffers and a borrowed set of counters.  Only
  // LAMMPS_NS::subview() uses this; the counters are shared on purpose, so that
  // a sync of the parent is seen through the child and the other way round.

  DualView(const base_type &base, const t_host &host, const t_lmp_flags &flags)
      : base_type(base), h_split(host), lmp_flags(flags)
  {
  }

  // Conversion between two spellings of the same dual view, which keeps the host
  // buffer and the counters.  Without this the result of subview() below, whose
  // space is spelled as a device_type, would go through the Kokkos::DualView
  // constructor above when it is assigned to a member declared with an execution
  // space, and quietly lose the sharing that makes the subview work at all.

  template <class DT, class... DP,
            class = std::enable_if_t<
                std::is_constructible_v<Kokkos::DualView<DataType, Properties...>,
                                        const Kokkos::DualView<DT, DP...> &>>>
  DualView(const DualView<DT, DP...> &src)
      : base_type(static_cast<const Kokkos::DualView<DT, DP...> &>(src)),
        h_split(src.impl_h_split()), lmp_flags(src.impl_lmp_flags())
  {
  }

  const t_lmp_flags &impl_lmp_flags() const { return lmp_flags; }
  const t_host &impl_h_split() const { return h_split; }

  /* ---- the two views ---- */

  KOKKOS_INLINE_FUNCTION
  const t_dev &view_device() const { return base_type::view_device(); }

  KOKKOS_INLINE_FUNCTION
  const t_host &view_host() const
  {
    if constexpr (SPLIT)
      return h_split;
    else
      return base_type::view_host();
  }

  // On a CPU build LMPDeviceType and LMPHostType are the same type, so the
  // template argument cannot express host-versus-device intent and this always
  // means the device side.  Code that wants the host side must say view_host().

  template <class Device>
  KOKKOS_INLINE_FUNCTION auto view() const
  {
    if constexpr (SPLIT)
      return base_type::view_device();
    else
      return base_type::template view<Device>();
  }

  /* ---- coherence state ---- */

  bool need_sync_device() const
  {
    if constexpr (SPLIT) {
      if (!lmp_flags.data()) return false;
      return lmp_flags(1) < lmp_flags(0);
    } else
      return base_type::need_sync_device();
  }

  bool need_sync_host() const
  {
    if constexpr (SPLIT) {
      if (!lmp_flags.data()) return false;
      return lmp_flags(0) < lmp_flags(1);
    } else
      return base_type::need_sync_host();
  }

  void modify_device()
  {
    if constexpr (SPLIT) {
      if (!lmp_flags.data()) return;
      if ((lmp_flags(0) > 0) && (lmp_flags(1) > 0))
        Kokkos::abort("LAMMPS::DualView::modify_device ERROR: concurrent modification "
                      "of host and device views");
      lmp_flags(1) = (lmp_flags(1) > lmp_flags(0) ? lmp_flags(1) : lmp_flags(0)) + 1;
    } else
      base_type::modify_device();
  }

  void modify_host()
  {
    if constexpr (SPLIT) {
      if (!lmp_flags.data()) return;
      if ((lmp_flags(0) > 0) && (lmp_flags(1) > 0))
        Kokkos::abort("LAMMPS::DualView::modify_host ERROR: concurrent modification "
                      "of host and device views");
      lmp_flags(0) = (lmp_flags(0) > lmp_flags(1) ? lmp_flags(0) : lmp_flags(1)) + 1;
    } else
      base_type::modify_host();
  }

  template <class Device>
  void modify()
  {
    if constexpr (SPLIT)
      modify_device();
    else
      base_type::template modify<Device>();
  }

  void sync_device()
  {
    if constexpr (SPLIT) {
      if (!lmp_flags.data() || !h_split.data()) return;
      if (lmp_flags(0) > lmp_flags(1)) {
        Kokkos::deep_copy(base_type::view_device(), h_split);
        lmp_flags(0) = lmp_flags(1) = 0;
      }
    } else
      base_type::sync_device();
  }

  void sync_host()
  {
    if constexpr (SPLIT) {
      if (!lmp_flags.data() || !h_split.data()) return;
      if (lmp_flags(1) > lmp_flags(0)) {
        Kokkos::deep_copy(h_split, base_type::view_device());
        lmp_flags(0) = lmp_flags(1) = 0;
      }
    } else
      base_type::sync_host();
  }

  template <class Device>
  void sync()
  {
    if constexpr (SPLIT)
      sync_device();
    else
      base_type::template sync<Device>();
  }

  void clear_sync_state()
  {
    if constexpr (SPLIT) {
      if (lmp_flags.data()) lmp_flags(0) = lmp_flags(1) = 0;
    }
    base_type::clear_sync_state();
  }

  /* ---- resizing has to carry the second allocation along ---- */

  template <class... Args>
  void resize(Args... args)
  {
    if constexpr (SPLIT) {
      // A default constructed dual view carries no counters, and resizing is the
      // one way it gains data without being replaced wholesale, so allocate them
      // here or every later modify_host() would quietly do nothing and the
      // checks would pass by never seeing the writes.  Kokkos::DualView does the
      // same for its own flags in impl_resize().
      if (!lmp_flags.data()) lmp_flags = t_lmp_flags("LAMMPS::DualView::lmp_flags");

      // Fold the host side back into the base allocation first, so that the base
      // class resize preserves it.  Without this any newer host data would be
      // dropped and the new buffer silently rebuilt from a stale device copy.
      // TransformView::resize() handles its legacy edge the same way.
      sync_device();
    }

    base_type::resize(args...);

    if constexpr (SPLIT) {
      h_split = t_host();
      allocate_split();
      if (lmp_flags.data()) lmp_flags(0) = lmp_flags(1) = 0;
    }
  }
};

/* ----------------------------------------------------------------------
   Slice a dual view.

   Kokkos::subview() only knows about the base class buffers, so it would slice
   the device side and leave the child with a host buffer of its own.  Writing
   to the child's host view and then syncing the parent, which is what the
   communication code does with its slices of k_swap, would then quietly lose
   the data.  Slice both sides here and let parent and child share one set of
   coherence counters.
------------------------------------------------------------------------- */

template <class DataType, class... Properties, class... Args>
auto subview(const DualView<DataType, Properties...> &src, Args... args)
{
  using src_type = DualView<DataType, Properties...>;
  using base_type = typename src_type::base_type;

  auto base = Kokkos::subview(static_cast<const base_type &>(src), args...);

  using result_type = DualView<typename decltype(base)::traits::data_type,
                               typename decltype(base)::traits::array_layout,
                               typename decltype(base)::traits::device_type>;

  if constexpr (src_type::SPLIT)
    return result_type(base, Kokkos::subview(src.view_host(), args...), src.impl_lmp_flags());
  else
    return result_type(base);
}

#endif    // LMP_KOKKOS_DEBUG_SYNC

}    // namespace LAMMPS_NS

#endif
