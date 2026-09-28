#pragma once

#include <Kokkos_Core.hpp>

// Precision switch:
//   default                  c_number = double, c_acc = double  (LAMMPS DOUBLE_DOUBLE)
//   OXDNA_SINGLE_PRECISION   c_number = float,  c_acc = float   (LAMMPS SINGLE_SINGLE)
//   OXDNA_MIXED_PRECISION    c_number = float,  c_acc = double  (LAMMPS SINGLE_DOUBLE)
// c_number is the compute type (KK_FLOAT); c_acc is the force/torque/energy
// accumulation type (KK_ACC_FLOAT): the force and torque arrays, the atomic
// scatters and the per-atom register accumulators all use c_acc, as in LAMMPS.
#if defined(OXDNA_MIXED_PRECISION)
using c_number = float;
using c_acc    = double;
#elif defined(OXDNA_SINGLE_PRECISION)
using c_number = float;
using c_acc    = float;
#else
using c_number = double;
using c_acc    = double;
#endif

// 4-component aligned vector — mirrors cuda_defs.h c_number4 / float4.
// The 128-bit alignment enables coalesced reads on GPU when stored in
// Kokkos::View<c_number*[4]>.
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
// components are contiguous in memory — matching the standalone oxDNA
// c_number4/float4 layout, which gives one coalesced transaction per particle
// for the scattered reads in the per-edge nonbonded kernel.
using Vec4  = Kokkos::View<c_number *[4], Kokkos::LayoutRight>;
using Vec4c = Kokkos::View<const c_number *[4], Kokkos::LayoutRight>;

// Read-only (RandomAccess) views route scattered gather reads through the GPU's
// read-only/texture data cache (__ldg), matching the standalone oxDNA CUDA
// kernels. Used for the per-atom data the per-edge force kernel gathers at
// random indices (positions, orientations, types, bonds).
using Vec4cr = Kokkos::View<const c_number *[4], Kokkos::LayoutRight,
                            Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
template <class T>
using RandomRead = Kokkos::View<const T *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

// Force / torque arrays in the accumulation precision (LAMMPS t_kkacc_*).
using VecA4  = Kokkos::View<c_acc *[4], Kokkos::LayoutRight>;
using VecA4c = Kokkos::View<const c_acc *[4], Kokkos::LayoutRight>;

// LAMMPS special-neighbor encoding. With the oxDNA excv style present LAMMPS
// sets special_flag = 2, so bonded (1-2) pairs stay in every neighbor list with
// their special-bond bits set in the top bits of the neighbor index; each
// kernel decodes them with sbmask()/NEIGHMASK and applies special_lj = 0
// (excv only knocks out backbone-backbone; all other terms skip the pair).
constexpr int OX_SBBITS    = 30;
constexpr int OX_NEIGHMASK = 0x1FFFFFFF;
KOKKOS_INLINE_FUNCTION int ox_sbmask(int j) { return (j >> OX_SBBITS) & 3; }

// Launch policy of the LAMMPS excv and dh kernels (OxdnaRangePolicy in
// mf_oxdna_kokkos.h): LaunchBounds<64,1> on CUDA, <128,1> on HIP, plain
// elsewhere. All other LAMMPS oxDNA kernels use a plain RangePolicy.
// Override with -DOXDNA_EXCVDH_MAXT=.. -DOXDNA_EXCVDH_MINB=.. to sweep.
#if !defined(OXDNA_EXCVDH_MAXT)
#if defined(KOKKOS_ENABLE_HIP)
#define OXDNA_EXCVDH_MAXT 128
#else
#define OXDNA_EXCVDH_MAXT 64
#endif
#endif
#if !defined(OXDNA_EXCVDH_MINB)
#define OXDNA_EXCVDH_MINB 1
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
using OxdnaRangePolicy =
    Kokkos::RangePolicy<Kokkos::LaunchBounds<OXDNA_EXCVDH_MAXT, OXDNA_EXCVDH_MINB>>;
#else
using OxdnaRangePolicy = Kokkos::RangePolicy<>;
#endif


// Box: periodic boundary conditions (orthogonal)
struct SimBox {
    c_number Lx, Ly, Lz;

    KOKKOS_INLINE_FUNCTION c_number Lx_half() const { return Lx * 0.5; }
    KOKKOS_INLINE_FUNCTION c_number Ly_half() const { return Ly * 0.5; }
    KOKKOS_INLINE_FUNCTION c_number Lz_half() const { return Lz * 0.5; }

    KOKKOS_INLINE_FUNCTION void wrap(c_number &dx, c_number &dy, c_number &dz) const {
        if (dx >  Lx_half()) dx -= Lx;
        if (dx < -Lx_half()) dx += Lx;
        if (dy >  Ly_half()) dy -= Ly;
        if (dy < -Ly_half()) dy += Ly;
        if (dz >  Lz_half()) dz -= Lz;
        if (dz < -Lz_half()) dz += Lz;
    }
};
