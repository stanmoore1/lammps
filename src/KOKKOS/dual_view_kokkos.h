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

#else

/* ----------------------------------------------------------------------
   Sync-debugging dual view.

   Kokkos turns off its own coherence state machine whenever the host and device
   device_types match: sync(), modify() and their named variants return
   immediately and the two views share a single allocation.  That is every
   CPU-only build, which is why a missing sync() or modify() -- silent data
   corruption on a GPU -- cannot be observed without one.

   When that happens this class allocates a second buffer for the device side and
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
  // second device-side allocation, empty unless SPLIT
  t_dev d_split;

  // lmp_flags(0) counts modifications of the host side, lmp_flags(1) of the
  // device side, exactly like Kokkos::DualView::modified_flags.  Held in a View
  // so that copies of this object share one set of counters.
  t_lmp_flags lmp_flags;

  void allocate_split()
  {
    if constexpr (SPLIT) {
      if (!base_type::view_device().data()) return;
      // create_mirror always allocates, unlike create_mirror_view
      d_split = Kokkos::create_mirror(base_type::view_device());
      Kokkos::deep_copy(d_split, base_type::view_device());
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
  // Note the result does not share this object's coherence counters: a subview
  // gets its own, which is a missed check rather than a false alarm.  All such
  // uses today are on communication bookkeeping arrays, not per-atom data.

  template <class DT, class... DP,
            class = std::enable_if_t<
                std::is_constructible_v<Kokkos::DualView<DataType, Properties...>,
                                        const Kokkos::DualView<DT, DP...> &>>>
  DualView(const Kokkos::DualView<DT, DP...> &src) : base_type(src)
  {
    lmp_flags = t_lmp_flags("LAMMPS::DualView::lmp_flags");
    allocate_split();
  }

  /* ---- the two views ---- */

  KOKKOS_INLINE_FUNCTION
  const t_dev &view_device() const
  {
    if constexpr (SPLIT)
      return d_split;
    else
      return base_type::view_device();
  }

  KOKKOS_INLINE_FUNCTION
  const t_host &view_host() const { return base_type::view_host(); }

  // On a CPU build LMPDeviceType and LMPHostType are the same type, so the
  // template argument cannot express host-versus-device intent and this always
  // means the device side.  Code that wants the host side must say view_host().

  template <class Device>
  KOKKOS_INLINE_FUNCTION auto view() const
  {
    if constexpr (SPLIT)
      return d_split;
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
      if (!lmp_flags.data() || !d_split.data()) return;
      if (lmp_flags(0) > lmp_flags(1)) {
        Kokkos::deep_copy(d_split, base_type::view_host());
        lmp_flags(0) = lmp_flags(1) = 0;
      }
    } else
      base_type::sync_device();
  }

  void sync_host()
  {
    if constexpr (SPLIT) {
      if (!lmp_flags.data() || !d_split.data()) return;
      if (lmp_flags(1) > lmp_flags(0)) {
        Kokkos::deep_copy(base_type::view_host(), d_split);
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
      // Fold the device side back into the host allocation first, so that the
      // base class resize preserves it.  Without this the device data would be
      // dropped and the new buffer silently rebuilt from a stale host copy.
      // TransformView::resize() handles its legacy edge the same way.
      sync_host();
    }

    base_type::resize(args...);

    if constexpr (SPLIT) {
      d_split = t_dev();
      allocate_split();
      if (lmp_flags.data()) lmp_flags(0) = lmp_flags(1) = 0;
    }
  }
};

#endif    // LMP_KOKKOS_DEBUG_SYNC

}    // namespace LAMMPS_NS

#endif
