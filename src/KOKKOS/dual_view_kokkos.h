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

#ifdef LMP_KOKKOS_DEBUG_SYNC
#include <execinfo.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#endif

namespace LAMMPS_NS {

// Defined in datamask_audit_kokkos.cpp.  A sync writes the very side the audit
// watches, so the audit has to be told once the copy has landed, or it reports
// the style's own sync as if the style had written the array itself.
void datamask_audit_note_copy(const void *device_data);

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

  // Watch mode state, allocated only for the views LMP_KOKKOS_WATCH selects: the
  // contents of each side as they were at the previous coherence call, and the
  // counters as they stood then.  See watch() below.
  t_host shadow_h, shadow_d;
  t_lmp_flags shadow_flags;
  // name of the call the shadows were taken at, so a report can bracket the
  // unclaimed write between two calls rather than only naming where it surfaced
  using t_watch_op = Kokkos::View<char[32], Kokkos::HostSpace>;
  t_watch_op shadow_op;

  // create_mirror always allocates, unlike create_mirror_view, and zero fills
  // unless told otherwise.  copy_across carries the base contents over, which is
  // right when the two sides are meant to agree, and wrong after a resize, where
  // Kokkos leaves the other side freshly zeroed and marks the resized one.
  // Control mode, LMP_KOKKOS_ALIAS=1: do not split at all, so the host side is
  // the base allocation and the build behaves exactly like an ordinary one.
  // If a case fails when split and passes here, the split is showing a real
  // coherence bug; if it fails here too, the emulation itself is at fault.
  bool alias_mode() const
  {
    static const char *f = std::getenv("LMP_KOKKOS_ALIAS");
    if (!f) return false;
    if (*f == '1' && f[1] == '\0') return true;    // every view
    return base_type::view_host().label().find(f) != std::string::npos;
  }

  void allocate_split(bool copy_across = true)
  {
    if constexpr (SPLIT) {
      if (alias_mode()) { h_split = base_type::view_host(); return; }
      if (!base_type::view_host().data()) return;
      h_split = Kokkos::create_mirror(base_type::view_host());
      if (copy_across) Kokkos::deep_copy(h_split, base_type::view_host());
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

  // Event trace for one view, selected by a substring in LMP_KOKKOS_TRACE.
  // Nothing is looked up and nothing printed unless that variable is set, so an
  // ordinary sync debugging run pays only a pointer test per operation.

  static const char *trace_filter()
  {
    static const char *f = std::getenv("LMP_KOKKOS_TRACE");
    return f;
  }

  void trace(const char *op) const
  {
    const char *f = trace_filter();
    if (!f) return;
    const std::string label = base_type::view_device().label();
    if (label.find(f) == std::string::npos) return;
    std::fprintf(stderr, "[dualview] %-24s %-14s flags=(%u,%u)\n", label.c_str(), op,
                 lmp_flags.data() ? lmp_flags(0) : 0u, lmp_flags.data() ? lmp_flags(1) : 0u);
  }

  // Coherence check, enabled by LMP_KOKKOS_VERIFY.  When the counters say the
  // two sides agree, they have to hold the same bytes; if they do not, some
  // copy was skipped or a claim was dropped while the data really did differ.
  // This is what catches a wrong claim, which no comparison of one side alone
  // can see.

  static const char *verify_filter()
  {
    static const char *f = std::getenv("LMP_KOKKOS_VERIFY");
    return f;
  }

  void verify(const char *when) const
  {
    const char *f = verify_filter();
    if (!f) return;
    if constexpr (SPLIT) {
      if (!lmp_flags.data() || !h_split.data()) return;
      if (lmp_flags(0) != 0 || lmp_flags(1) != 0) return;    // a claim is pending
      const std::string label = base_type::view_device().label();
      if (*f && label.find(f) == std::string::npos) return;
      const char *d = (const char *) base_type::view_device().data();
      const char *h = (const char *) h_split.data();
      if (!d || !h) return;
      const size_t n = h_split.span() * sizeof(typename t_host::value_type);
      for (size_t b = 0; b < n; b++) {
        if (d[b] == h[b]) continue;
        std::fprintf(stderr,
                     "[verify] %s: host and device differ at byte %zu of %zu while the "
                     "counters call them in sync (at %s)\n",
                     label.c_str(), b, n, when);
        break;
      }
    }
  }

  // Paranoid mode, selected by a substring in LMP_KOKKOS_PARANOID (empty string
  // for every view).  Each claim is followed straight away by the copy it
  // implies, so the two sides never actually diverge.  This does not report
  // anything: it is for bisecting.  If a run is correct with a view forced this
  // way and wrong without, then a sync of that view is missing somewhere.

  static const char *paranoid_filter()
  {
    static const char *f = std::getenv("LMP_KOKKOS_PARANOID");
    return f;
  }

  bool paranoid() const
  {
    const char *f = paranoid_filter();
    if (!f) return false;
    if (!*f) return true;
    return base_type::view_device().label().find(f) != std::string::npos;
  }

  void settle_from_host()
  {
    if constexpr (SPLIT) {
      if (!paranoid() || !lmp_flags.data() || !h_split.data()) return;
      Kokkos::deep_copy(base_type::view_device(), h_split);
      lmp_flags(0) = lmp_flags(1) = 0;
      watch_refresh();
      datamask_audit_note_copy(base_type::view_device().data());
    }
  }

  /* ---- watch mode ---------------------------------------------------------

     LMP_KOKKOS_VERIFY only sees a view whose counters call it in sync, and the
     ordinary way to write a dual view -- fill one side, then claim it -- leaves
     the counters saying exactly that for as long as it takes to reach the claim.
     So it cannot tell a forgotten claim from a claim that has not happened yet.

     Watch mode removes the ambiguity by remembering, for the views whose label
     contains LMP_KOKKOS_WATCH, what each side held at the previous coherence
     call.  At the next one it compares:

       host differs from its shadow    -> the host side was written since
       device differs from its shadow  -> the device side was written since

     which is a fact about the data and needs no interpretation.  A write is
     legitimate when the counter for that side went up in the meantime, or when
     the call we are entering is the claim for it.  Anything else is a write
     nobody claimed: on a GPU the next sync in that direction silently discards
     it.  The report names the view, the element, both values and the call that
     found it; set LMP_KOKKOS_WATCH_BT to add a backtrace, which points straight
     at the routine that needs the claim.

     The shadows are then brought up to date, so one bug is reported once rather
     than at every later call.
  --------------------------------------------------------------------------- */

  static const char *watch_filter()
  {
    static const char *f = std::getenv("LMP_KOKKOS_WATCH");
    return f;
  }

  bool watched() const
  {
    const char *f = watch_filter();
    if (!f) return false;
    if (!*f) return true;
    return base_type::view_device().label().find(f) != std::string::npos;
  }

  static void watch_backtrace()
  {
    if (!std::getenv("LMP_KOKKOS_WATCH_BT")) return;
    void *frames[32];
    const int n = backtrace(frames, 32);
    backtrace_symbols_fd(frames, n, fileno(stderr));
  }

  // Report the first element in which the two buffers differ.  The values are
  // printed as the value type reads them, so an index array shows the atom it
  // points at rather than a byte pattern.
  void watch_report(const char *side, const char *op, const t_host &now, const t_host &was) const
  {
    using value_type = typename std::remove_const<typename t_host::value_type>::type;
    const value_type *a = (const value_type *) now.data();
    const value_type *b = (const value_type *) was.data();
    const size_t n = now.span();
    const std::string label = base_type::view_device().label();
    for (size_t i = 0; i < n; i++) {
      if (!std::memcmp(&a[i], &b[i], sizeof(value_type))) continue;
      std::fprintf(stderr,
                   "[watch] %s: the %s side was written but never claimed\n"
                   "        the write is between %s and %s\n"
                   "        element %zu of %zu changed ",
                   label.c_str(), side, shadow_op.data() ? shadow_op.data() : "the start",
                   op, i, n);
      if constexpr (std::is_floating_point_v<value_type>)
        std::fprintf(stderr, "from %g to %g\n", (double) b[i], (double) a[i]);
      else if constexpr (std::is_integral_v<value_type>)
        std::fprintf(stderr, "from %lld to %lld\n", (long long) b[i], (long long) a[i]);
      else
        std::fprintf(stderr, "(value type is not printable)\n");
      std::fprintf(stderr, "        counters are (host %u, device %u)\n",
                   lmp_flags(0), lmp_flags(1));
      watch_backtrace();
      return;
    }
  }

  // claim_host/claim_device say which write the call we are entering is about to
  // account for, so that the ordinary fill-then-claim sequence is not reported.
  void watch(const char *op, bool claim_host = false, bool claim_device = false)
  {
    if constexpr (SPLIT) {
      if (!watched()) return;
      if (!lmp_flags.data() || !h_split.data()) return;
      if (h_split.data() == base_type::view_host().data()) return;    // alias mode

      const t_dev &dev = base_type::view_device();
      if (shadow_h.data() && shadow_h.span() == h_split.span()) {
        const bool host_wrote =
            std::memcmp(h_split.data(), shadow_h.data(),
                        h_split.span() * sizeof(typename t_host::value_type)) != 0;
        const bool dev_wrote =
            std::memcmp(dev.data(), shadow_d.data(),
                        dev.span() * sizeof(typename t_dev::value_type)) != 0;

        if (host_wrote && !claim_host && lmp_flags(0) <= shadow_flags(0))
          watch_report("host", op, h_split, shadow_h);
        if (dev_wrote && !claim_device && lmp_flags(1) <= shadow_flags(1)) {
          t_host dev_now = Kokkos::create_mirror(dev);
          Kokkos::deep_copy(dev_now, dev);
          watch_report("device", op, dev_now, shadow_d);
        }
      }

      watch_op_name() = op;
    }
    watch_refresh();
  }

  // Take the shadows from the current contents without checking anything.  Used
  // after this class has itself changed a side -- a sync copy or a resize -- so
  // that its own writes are not reported as somebody's missing claim.
  void watch_refresh()
  {
    if constexpr (SPLIT) {
      if (!watched()) return;
      if (!lmp_flags.data() || !h_split.data()) return;
      if (h_split.data() == base_type::view_host().data()) return;
      if (!shadow_h.data() || shadow_h.span() != h_split.span()) {
        shadow_h = Kokkos::create_mirror(h_split);
        shadow_d = Kokkos::create_mirror(h_split);
        if (!shadow_flags.data()) shadow_flags = t_lmp_flags("LAMMPS::DualView::shadow_flags");
        if (!shadow_op.data()) shadow_op = t_watch_op("LAMMPS::DualView::shadow_op");
      }
      Kokkos::deep_copy(shadow_h, h_split);
      Kokkos::deep_copy(shadow_d, base_type::view_device());
      shadow_flags(0) = lmp_flags(0);
      shadow_flags(1) = lmp_flags(1);
      if (shadow_op.data() && watch_op_name()) {
        std::strncpy(shadow_op.data(), watch_op_name(), 31);
        shadow_op(31) = 0;
      }
    }
  }

  // set by watch() so watch_refresh() can record which call the shadows belong
  // to; a refresh that follows a copy this class made keeps the caller's name
  static const char *&watch_op_name()
  {
    static const char *name = nullptr;
    return name;
  }

  void settle_from_device()
  {
    if constexpr (SPLIT) {
      if (!paranoid() || !lmp_flags.data() || !h_split.data()) return;
      Kokkos::deep_copy(h_split, base_type::view_device());
      lmp_flags(0) = lmp_flags(1) = 0;
      watch_refresh();
    }
  }

  /* ---- the two views ---- */

  KOKKOS_INLINE_FUNCTION
  const t_dev &view_device() const
  {
    stale_check(true);
    return base_type::view_device();
  }

  KOKKOS_INLINE_FUNCTION
  const t_host &view_host() const
  {
    stale_check(false);
    if constexpr (SPLIT)
      return h_split;
    else
      return base_type::view_host();
  }

  /* ---- stale read reporting, enabled by LMP_KOKKOS_STALE=<label> -----------

     Watch mode sees a write nobody claimed.  The other half of the bug class is
     a read of a side that somebody else has claimed and not copied over: the
     counters are perfectly consistent, the data is simply old.  Nothing in the
     coherence state can distinguish that from a legitimate handing out of the
     view just before a sync, so this is reported rather than fatal, and is
     filtered by label -- it is meant to be pointed at one suspect view.

     Combine with LMP_KOKKOS_WATCH_BT for the backtrace of the reader.
  --------------------------------------------------------------------------- */

  static const char *stale_filter()
  {
    static const char *f = std::getenv("LMP_KOKKOS_STALE");
    return f;
  }

  void stale_check(bool want_device) const
  {
    if constexpr (SPLIT) {
      const char *f = stale_filter();
      if (!f) return;
      if (!lmp_flags.data() || !h_split.data()) return;
      if (want_device ? !need_sync_device() : !need_sync_host()) return;
      const std::string label = base_type::view_device().label();
      if (*f && label.find(f) == std::string::npos) return;
      std::fprintf(stderr,
                   "[stale] %s: the %s side is handed out while the %s side is newer, "
                   "counters (host %u, device %u)\n",
                   label.c_str(), want_device ? "device" : "host",
                   want_device ? "host" : "device", lmp_flags(0), lmp_flags(1));
      watch_backtrace();
    }
  }

  // On a CPU build LMPDeviceType and LMPHostType are the same type, so the
  // template argument cannot express host-versus-device intent and this always
  // means the device side.  Code that wants the host side must say view_host().

  template <class Device>
  KOKKOS_INLINE_FUNCTION auto view() const
  {
    if constexpr (SPLIT) {
      stale_check(true);
      return base_type::view_device();
    } else
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
    trace("modify_device");
    verify("modify_device");
    watch("modify_device", false, true);
    if constexpr (SPLIT) {
      if (!lmp_flags.data()) return;

      // Claim first and test afterwards, the way Kokkos::DualView does: the case
      // worth catching is a claim on one side while the other side still holds
      // one, and testing first would let exactly that through.
      lmp_flags(1) = (lmp_flags(1) > lmp_flags(0) ? lmp_flags(1) : lmp_flags(0)) + 1;
      if (lmp_flags(0) && lmp_flags(1))
        Kokkos::abort(("LAMMPS::DualView::modify_device ERROR: concurrent modification of "
                       "host and device views in DualView \"" +
                       base_type::view_device().label() + "\"")
                          .c_str());
      settle_from_device();
    } else
      base_type::modify_device();
  }

  void modify_host()
  {
    trace("modify_host");
    verify("modify_host");
    watch("modify_host", true, false);
    if constexpr (SPLIT) {
      if (!lmp_flags.data()) return;

      // see modify_device(): claim first, then test
      lmp_flags(0) = (lmp_flags(0) > lmp_flags(1) ? lmp_flags(0) : lmp_flags(1)) + 1;
      if (lmp_flags(0) && lmp_flags(1))
        Kokkos::abort(("LAMMPS::DualView::modify_host ERROR: concurrent modification of "
                       "host and device views in DualView \"" +
                       base_type::view_device().label() + "\"")
                          .c_str());
      settle_from_host();
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
    trace("sync_device");
    verify("sync_device");
    watch("sync_device");
    if constexpr (SPLIT) {
      if (!lmp_flags.data() || !h_split.data()) return;
      if (lmp_flags(0) > lmp_flags(1)) {
        Kokkos::deep_copy(base_type::view_device(), h_split);
        lmp_flags(0) = lmp_flags(1) = 0;
        watch_refresh();
        datamask_audit_note_copy(base_type::view_device().data());
      }
    } else
      base_type::sync_device();
  }

  void sync_host()
  {
    trace("sync_host");
    verify("sync_host");
    watch("sync_host");
    if constexpr (SPLIT) {
      if (!lmp_flags.data() || !h_split.data()) return;
      if (lmp_flags(1) > lmp_flags(0)) {
        Kokkos::deep_copy(h_split, base_type::view_device());
        lmp_flags(0) = lmp_flags(1) = 0;
        watch_refresh();
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
    trace("clear_sync_state");
    watch("clear_sync_state");
    if constexpr (SPLIT) {
      if (lmp_flags.data()) lmp_flags(0) = lmp_flags(1) = 0;
    }
    base_type::clear_sync_state();
  }

  /* ---- resizing has to carry the second allocation along ---- */

  template <class... Args>
  void resize(Args... args)
  {
    trace("resize");
    watch("resize");
    if constexpr (SPLIT) {
      // A default constructed dual view carries no counters, and resizing is the
      // one way it gains data without being replaced wholesale, so allocate them
      // here or every later modify_host() would quietly do nothing and the
      // checks would pass by never seeing the writes.  Kokkos::DualView does the
      // same for its own flags in impl_resize().
      if (!lmp_flags.data()) lmp_flags = t_lmp_flags("LAMMPS::DualView::lmp_flags");

      // Kokkos resizes on whichever side the counters say is newer, keeps that
      // side's contents, and marks it modified.  A tie goes to the device, so it
      // resizes on the host only when the host counter is strictly higher, i.e.
      // when something marked the host and has not synced since.  All of that
      // happens only when the two sides really differ, so a build without a GPU
      // never sees the claim left behind -- and code that then marks the other
      // side without clearing this one is exactly the bug worth finding.
      const bool on_device = (lmp_flags(1) >= lmp_flags(0));

      // Resizing on the host: fold it into the base, which the base class resize
      // preserves, and copy it back afterwards.  Kokkos would leave the device
      // side zeroed here; keeping the values is the more forgiving of the two and
      // the counter still says the device owes a sync.
      if (!on_device) sync_device();

      base_type::resize(args...);

      h_split = t_host();

      // Paranoid mode keeps the two sides equal, so carry the contents across
      // and leave no claim; otherwise follow Kokkos and leave the other side
      // zeroed with the resized one claimed.
      if (paranoid()) {
        allocate_split(true);
        lmp_flags(0) = lmp_flags(1) = 0;
        watch_refresh();
        return;
      }

      allocate_split(!on_device);

      lmp_flags(0) = lmp_flags(1) = 0;
      if (on_device)
        lmp_flags(1) = 1;
      else
        lmp_flags(0) = 1;
      watch_refresh();
      return;
    }

    base_type::resize(args...);
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
