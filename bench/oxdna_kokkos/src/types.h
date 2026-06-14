#pragma once

#include <Kokkos_Core.hpp>

// Precision switch: define OXDNA_SINGLE_PRECISION for float, otherwise double
#ifdef OXDNA_SINGLE_PRECISION
using c_number = float;
#else
using c_number = double;
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
