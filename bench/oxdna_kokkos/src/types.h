#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>

// ---------------------------------------------------------------------------
// Precision (mirrors the oxDNA CUDA backends, see README "Fidelity to oxDNA
// CUDA"):
//   default                   oxDNA CUDA_DOUBLE build (backend_precision = double):
//                             everything in double.
//   OXDNA_SINGLE_PRECISION    oxDNA MD_CUDABackend with c_number = float
//                             (backend_precision = float): everything in float.
//   OXDNA_MIXED_PRECISION     oxDNA CUDAMixedBackend (backend_precision = mixed,
//                             the default of the CUDA benchmarks): float4
//                             positions/orientations/forces/torques/velocities
//                             for the force kernels, lists and thermostat, plus
//                             double copies of positions, orientations,
//                             velocities and angular momenta used (only) by the
//                             integrator.
//
// c_number : the force-kernel / list number type (oxDNA c_number)
// m_number : the integrator number type (oxDNA LR_double4 in mixed precision)
// ---------------------------------------------------------------------------
#if defined(OXDNA_MIXED_PRECISION) && defined(OXDNA_SINGLE_PRECISION)
#error "OXDNA_MIXED_PRECISION and OXDNA_SINGLE_PRECISION are mutually exclusive"
#endif

#if defined(OXDNA_MIXED_PRECISION)
using c_number = float;
using m_number = double;
constexpr bool OXDNA_MIXED = true;
#elif defined(OXDNA_SINGLE_PRECISION)
using c_number = float;
using m_number = float;
constexpr bool OXDNA_MIXED = false;
#else
using c_number = double;
using m_number = double;
constexpr bool OXDNA_MIXED = false;
#endif

inline const char *oxdna_precision_name() {
    if (OXDNA_MIXED) return "mixed";
    return (sizeof(c_number) == 4) ? "float" : "double";
}

// ---------------------------------------------------------------------------
// Launch configuration (mirrors CUDABaseBackend::_init_CUDA_kernel_cfgs):
// every oxDNA CUDA kernel is launched with threads_per_block threads per block
// (default 2 * warpSize = 64 on NVIDIA, 128 on AMD) and ceil(n / tpb) blocks.
// Kokkos picks the block size of a RangePolicy itself; LaunchBounds<TPB>
// caps it at TPB (the occupancy search starts at TPB and only goes down if
// fewer threads give strictly more resident threads, which never happens for
// the oxDNA kernels), and emits __launch_bounds__(TPB), which does not limit
// registers below the unbounded default for TPB <= 64. The oxDNA input key
// `threads_per_block` is a compile-time value here (-DOXDNA_THREADS_PER_BLOCK).
// ---------------------------------------------------------------------------
#ifndef OXDNA_THREADS_PER_BLOCK
#if defined(KOKKOS_ENABLE_HIP)
#define OXDNA_THREADS_PER_BLOCK 128
#else
#define OXDNA_THREADS_PER_BLOCK 64
#endif
#endif
using OxLaunch = Kokkos::LaunchBounds<OXDNA_THREADS_PER_BLOCK, 0>;
using OxPolicy = Kokkos::RangePolicy<OxLaunch>;

// Tuning overrides for the force kernels (default: the upstream launch
// configuration above, no register cap): OXDNA_NB_MAXT / OXDNA_NB_MINB for the
// nonbonded kernel (per-particle dna_forces / DNA3_forces, or the edge kernel)
// and OXDNA_BOND_MAXT / OXDNA_BOND_MINB for the bonded edge kernel; a MINB > 0
// emits __launch_bounds__(MAXT, MINB), which caps registers (not upstream).
#ifndef OXDNA_NB_MAXT
#define OXDNA_NB_MAXT OXDNA_THREADS_PER_BLOCK
#endif
#ifndef OXDNA_NB_MINB
#define OXDNA_NB_MINB 0
#endif
#ifndef OXDNA_BOND_MAXT
#define OXDNA_BOND_MAXT OXDNA_THREADS_PER_BLOCK
#endif
#ifndef OXDNA_BOND_MINB
#define OXDNA_BOND_MINB 0
#endif
using OxForcePolicy = Kokkos::RangePolicy<Kokkos::LaunchBounds<OXDNA_NB_MAXT, OXDNA_NB_MINB>>;
using OxBondPolicy  = Kokkos::RangePolicy<Kokkos::LaunchBounds<OXDNA_BOND_MAXT, OXDNA_BOND_MINB>>;

// 4-component aligned vector -- mirrors cuda_defs.h c_number4 / float4.
struct alignas(sizeof(c_number) * 4) c_number4 {
    c_number x, y, z, w;
};

// Quaternion: same layout as GPU_quat in standalone oxDNA (w, x, y, z convention)
using GPU_quat = c_number4;

// Bonded strand neighbours: 3' neighbour (n3) and 5' neighbour (n5).
// Index -1 means no neighbour (strand terminus).
struct LR_bonds {
    int n3, n5;
};

// Per-particle 4-component arrays stored AoS (LayoutRight) so a particle's four
// components are contiguous in memory -- matching the standalone oxDNA
// c_number4/float4 layout, which gives one coalesced transaction per particle.
using Vec4  = Kokkos::View<c_number *[4], Kokkos::LayoutRight>;
using Vec4c = Kokkos::View<const c_number *[4], Kokkos::LayoutRight>;
// integrator arrays (double in mixed precision: oxDNA LR_double4)
using Vec4m = Kokkos::View<m_number *[4], Kokkos::LayoutRight>;

// Read-only (RandomAccess) views route scattered gather reads through the GPU's
// read-only/texture data cache (__ldg), matching the standalone oxDNA CUDA
// kernels (__restrict__ const pointers).
using Vec4cr = Kokkos::View<const c_number *[4], Kokkos::LayoutRight,
                            Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
template <class T>
using RandomRead = Kokkos::View<const T *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

// Host-pinned scalar flag written directly by device kernels and read by the
// host (oxDNA _d_are_lists_old / _d_cell_overflow: cudaMallocHost bool).
using PinnedFlag = Kokkos::View<int, Kokkos::SharedHostPinnedSpace>;

// Box: periodic boundary conditions (orthogonal), mirrors CUDABox.
// Positions are NOT folded into the box during MD (as upstream); pair
// separations use the minimum image with rint, bonded separations the plain
// difference.
struct SimBox {
    c_number Lx, Ly, Lz;

    // CUDABox::minimum_image(r_i, r_j) applied to d = r_j - r_i
    KOKKOS_INLINE_FUNCTION void wrap(c_number &dx, c_number &dy, c_number &dz) const {
        dx -= Kokkos::rint(dx / Lx) * Lx;
        dy -= Kokkos::rint(dy / Ly) * Ly;
        dz -= Kokkos::rint(dz / Lz) * Lz;
    }

    // CUDABox::compute_cell_spl_idx (float4 overload; FLT/DBL_EPSILON by type)
    KOKKOS_INLINE_FUNCTION int cell_coord(c_number v, c_number L, int n) const {
        const c_number eps = (sizeof(c_number) == 4) ? c_number(1.1920928955078125e-07)
                                                     : c_number(2.220446049250313e-16);
        return static_cast<int>((v / L - Kokkos::floor(v / L)) * (c_number(1) - eps) * n);
    }
};
