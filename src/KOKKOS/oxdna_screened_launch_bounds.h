#ifndef LMP_OXDNA_SCREENED_LAUNCH_BOUNDS_H
#define LMP_OXDNA_SCREENED_LAUNCH_BOUNDS_H

// ---------------------------------------------------------------------------
// Tunable launch bounds for the screened-pair / fused (device) oxdna kernels.
//
// The screened "GPUPair" operators - and especially the fused hbond+xstk kernel,
// which keeps both interactions' state live in one thread - are register-heavy,
// so without a register cap the compiler may use enough registers to limit
// occupancy. Kokkos::LaunchBounds emits CUDA __launch_bounds__(MaxThreadsPerBlock,
// MinBlocksPerSM), letting the compiler trade registers for occupancy. This is
// the lever to test whether the fused kernel's slowdown is occupancy (register)
// limited.
//
// These defaults are a starting point, NOT a tuned optimum: the sweet spot is
// GPU- and precision-dependent (too aggressive a MinBlocks forces register
// spills and regresses). Sweep on the target GPU, e.g.
//   -DOXDNA_SCREENED_MAXT=64  -DOXDNA_SCREENED_MINB=16
//   -DOXDNA_SCREENED_MAXT=128 -DOXDNA_SCREENED_MINB=8
// comparing achieved occupancy / registers-per-thread in Nsight Compute.
// LaunchBounds is ignored on CPU backends, so this is a no-op there.
// ---------------------------------------------------------------------------

#ifndef OXDNA_SCREENED_MAXT
#define OXDNA_SCREENED_MAXT 128
#endif
#ifndef OXDNA_SCREENED_MINB
#define OXDNA_SCREENED_MINB 6
#endif

#define OXDNA_SCREENED_LAUNCH_BOUNDS \
  Kokkos::LaunchBounds<OXDNA_SCREENED_MAXT, OXDNA_SCREENED_MINB>

#endif
