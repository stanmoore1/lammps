#ifndef LMP_OXDNA_SCREENED_TOGGLE_H
#define LMP_OXDNA_SCREENED_TOGGLE_H

#include <cstdlib>

// -----------------------------------------------------------------------------
// Developer / CI test aid (NOT a model option).
//
// The screened edge-parallel ("GPUPair") path for the oxDNA hbond/xstk/coaxstk
// pair styles is the device execution path; on the host the per-atom path runs.
// To exercise (and regression-test) the device path - including the fused
// hbond+xstk kernel - under the Kokkos Serial backend, OXDNA_KK_FORCE_SCREENED=1
// makes the screened path also run on the host. It defaults OFF and does not
// affect GPU runs or the default host path. It does NOT control kernel fusion:
// fusion is enabled automatically (see PairOxdnaHbondKokkos::compute).
// -----------------------------------------------------------------------------

namespace LAMMPS_NS {

static inline bool oxdna_force_screened_host()
{
  static int cached = -1;
  if (cached < 0) {
    const char *e = std::getenv("OXDNA_KK_FORCE_SCREENED");
    cached = (e && e[0] == '1') ? 1 : 0;
  }
  return cached == 1;
}

// Benchmark aid: OXDNA_KK_NO_FUSE=1 disables the fused hbond+xstk kernel, so the
// split hbond and xstk screened kernels run instead. Lets the fused vs split
// kernels be A/B-timed on the same build/GPU. Defaults OFF (fusion on).
static inline bool oxdna_disable_fusion()
{
  static int cached = -1;
  if (cached < 0) {
    const char *e = std::getenv("OXDNA_KK_NO_FUSE");
    cached = (e && e[0] == '1') ? 1 : 0;
  }
  return cached == 1;
}

}    // namespace LAMMPS_NS

#endif
