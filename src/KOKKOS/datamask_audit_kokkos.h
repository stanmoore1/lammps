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

#ifndef LMP_DATAMASK_AUDIT_KOKKOS_H
#define LMP_DATAMASK_AUDIT_KOKKOS_H

#include "pointers.h"

#include <cstdint>
#include <string>
#include <vector>

namespace LAMMPS_NS {

#ifndef LMP_KOKKOS_DEBUG_SYNC

// Without the sync debugging option this compiles away.  The class still exists
// so the call sites need no conditional compilation of their own.

class DatamaskAudit {
 public:
  DatamaskAudit(LAMMPS *, const char *, const char *, uint64_t) {}
  static void enable(int) {}
  static void note_modified(uint64_t) {}
  static void note_synced(uint64_t) {}
  static void report(LAMMPS *) {}
  static void trace_end(const char *, const char *) {}
};

#else

/* ----------------------------------------------------------------------
   Check a style against what it declares that it changes.

   Every KOKKOS style declares the per-atom arrays it writes in datamask_modify,
   and the package copies data between the host and the device based on that
   declaration.  A style that changes an array it did not declare leaves the
   other copy stale, and a later copy in the opposite direction then overwrites
   the new values with the old ones.  On a GPU that silently changes the
   results; on a CPU it cannot be seen at all, because both copies are the same
   memory there.

   This compares the contents of the arrays rather than the coherence flags,
   because a style that forgets to declare a write leaves those flags looking
   perfectly clean.  Only the data itself shows what happened.  Note also that
   ModifyKokkos issues modified() with datamask_modify after every style call,
   so the modified() calls inside a style are redundant with it: what has to be
   right is the declaration, not the call.

   Declaring more than is written is only wasteful and is not reported, except
   for a style that declares every array: that one is reported, because there is
   then nothing left to compare and silence would read as a clean result.
------------------------------------------------------------------------- */

class DatamaskAudit {
 public:
  DatamaskAudit(LAMMPS *lmp, const char *what, const char *style, uint64_t datamask_modify);
  ~DatamaskAudit();

  // off during setup and input processing, which rewrite whatever they like
  static void enable(int flag);

  // A style may declare EMPTY_MASK and mark what it changed itself, per routine,
  // which is just as correct as declaring it up front.  AtomKokkos::modified()
  // reports the masks it is given here so that those count as declared too.
  static void note_modified(uint64_t mask);

  // A sync writes the very side being watched, so it would look like the style
  // had written it.  Take the affected arrays' contents again instead, which
  // keeps a later write by the style itself visible.
  static void note_synced(uint64_t mask);

  // called from the dual view once a sync has written the device side
  void rebaseline_one(const void *device_data);
  static void report(LAMMPS *lmp);
  static void trace_end(const char *what, const char *style);

  struct Array {
    uint64_t bit;
    const char *name;
    const char *data;
    size_t bytes;
    int stride;    // bytes per atom, to name the atom that changed
    bool stale;    // the device side owed a sync when the style started
  };

 private:
  LAMMPS *lmp;
  const char *what;
  std::string style;
  uint64_t declared;
  int nall;
  std::vector<Array> arrays;
  std::vector<std::vector<char>> before;
  bool active;

  void rebaseline(uint64_t mask);
};

#endif    // LMP_KOKKOS_DEBUG_SYNC

}    // namespace LAMMPS_NS

#endif
