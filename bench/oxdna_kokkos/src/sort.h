#pragma once

// Hilbert-curve particle sorting, a port of src/CUDA/CUDA_sort.cu and
// CUDABaseBackend::_sort_index / MD_CUDABackend::_sort_particles
// (CUDAMixedBackend::_sort_particles in mixed precision).
//
// Upstream default: CUDA_sort_every = 0 (disabled). With CUDA_sort_every = n > 0
// the particles are re-ordered along a 3D Hilbert curve (depth 8, cubic box
// side Lx) on every n-th Verlet-list update (sim_step: `if(are_lists_old &&
// sort_every > 0 && N_updates % sort_every == 0)`, i.e. right before the list
// rebuild):
//   reset_sorted_hindex -> hilbert_curve -> thrust::sort_by_key (here
//   Kokkos::Experimental::sort_by_key, which calls thrust on CUDA) ->
//   get_inverted_sorted_hindex -> permute_particles (positions, velocities,
//   angular momenta, orientations, bonds remapped through the inverse
//   permutation, particles_to_mols) -> device-to-device copies back.
// In mixed precision the double arrays are first converted to float, the
// float arrays are permuted and converted back (as upstream: sorting rounds
// the double state to float).
//
// Beyond upstream: the base/particle-type arrays (btype, ptype) are permuted
// too. Upstream keeps btype inside poss.w (permuted with the positions) but
// does not permute CUDADNA3Interaction::_d_particle_types, so DNA3 +
// CUDA_sort_every > 0 is broken upstream; here it is correct.

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

namespace hilbert {

KOKKOS_INLINE_FUNCTION void vertex_swap(int &v, int &a, int &b, const int mask) {
    const int va = ((v >> a) & mask);
    const int vb = ((v >> b) & mask);
    v = v ^ (va << a) ^ (vb << b) ^ (va << b) ^ (vb << a);
    const int c = b;
    b = a;
    a = c;
}

KOKKOS_INLINE_FUNCTION int hilbert_code(float rx, float ry, float rz, int depth) {
    int hcode = 0;
    int a = 21, b = 18, c = 12, d = 15, e = 3, f = 0, g = 6, h = 9;
    int vc = 1U << b ^ 2U << c ^ 3U << d ^ 4U << e ^ 5U << f ^ 6U << g ^ 7U << h;
    constexpr int MASK = (1 << 3) - 1;
    for (int i = 0; i < depth; ++i) {
        const int x = Kokkos::signbit(rx) ? 1 : 0;
        const int y = Kokkos::signbit(ry) ? 1 : 0;
        const int z = Kokkos::signbit(rz) ? 1 : 0;
        const int v = (vc >> (3 * (x + (y << 1) + (z << 2))) & MASK);
        rx = 2 * rx - (0.5f - x);
        ry = 2 * ry - (0.5f - y);
        rz = 2 * rz - (0.5f - z);
        if (v == 0) {
            vertex_swap(vc, b, h, MASK);
            vertex_swap(vc, c, e, MASK);
        } else if (v == 1 || v == 2) {
            vertex_swap(vc, c, g, MASK);
            vertex_swap(vc, d, h, MASK);
        } else if (v == 3 || v == 4) {
            vertex_swap(vc, a, c, MASK);
            vertex_swap(vc, f, h, MASK);
        } else if (v == 5 || v == 6) {
            vertex_swap(vc, a, e, MASK);
            vertex_swap(vc, b, f, MASK);
        } else if (v == 7) {
            vertex_swap(vc, a, g, MASK);
            vertex_swap(vc, d, f, MASK);
        }
        hcode = (hcode << 3) + v;
    }
    return hcode;
}

} // namespace hilbert

struct HilbertSorter {
    int N = 0;
    int depth = 8;          // "fixed value for depth (8)"
    float box_side = 0;     // hilb_box_side (float, Lx only)
    Kokkos::View<int *> hindex, sorted_hindex, inv_sorted_hindex;
    // buffers (_d_buff_*)
    Vec4 buff_poss, buff_vels, buff_Ls, buff_orientations;
    Kokkos::View<LR_bonds *> buff_bonds;
    Kokkos::View<int *> buff_strand, buff_btype;
    Kokkos::View<uint8_t *> buff_ptype;

    void init(int N_in, const SimBox &box) {
        N = N_in;
        box_side = static_cast<float>(box.Lx);
        hindex = Kokkos::View<int *>("hindex", N);
        sorted_hindex = Kokkos::View<int *>("sorted_hindex", N);
        inv_sorted_hindex = Kokkos::View<int *>("inv_sorted_hindex", N);
        buff_poss = Vec4("buff_poss", N);
        buff_vels = Vec4("buff_vels", N);
        buff_Ls = Vec4("buff_Ls", N);
        buff_orientations = Vec4("buff_orientations", N);
        buff_bonds = Kokkos::View<LR_bonds *>("buff_bonds", N);
        buff_strand = Kokkos::View<int *>("buff_particles_to_mols", N);
        buff_btype = Kokkos::View<int *>("buff_btype", N);
        buff_ptype = Kokkos::View<uint8_t *>("buff_ptype", N);
        reset_sorted_hindex();
    }

    void reset_sorted_hindex() {
        auto s = sorted_hindex;
        Kokkos::parallel_for("reset_sorted_hindex", OxPolicy(0, N), KOKKOS_LAMBDA(int i) { s(i) = i; });
    }

    // CUDABaseBackend::_sort_index
    void sort_index(const Vec4 &poss) {
        reset_sorted_hindex();
        auto hi = hindex;
        const int dep = depth;
        const float side = box_side;
        Kokkos::parallel_for("hilbert_curve", OxPolicy(0, N), KOKKOS_LAMBDA(int i) {
            float rx = poss(i, 0), ry = poss(i, 1), rz = poss(i, 2);
            const int n = 1 << dep;
            rx /= side; ry /= side; rz /= side;
            rx = (rx - Kokkos::floor(rx)) * n;
            ry = (ry - Kokkos::floor(ry)) * n;
            rz = (rz - Kokkos::floor(rz)) * n;
            rx = (Kokkos::floor(rx) + 0.5f) / n;
            ry = (Kokkos::floor(ry) + 0.5f) / n;
            rz = (Kokkos::floor(rz) + 0.5f) / n;
            rx -= 0.5f; ry -= 0.5f; rz -= 0.5f;
            hi(i) = hilbert::hilbert_code(rx, ry, rz, dep);   // N_unsortable = 0
        });
        Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), hindex, sorted_hindex);
        auto s = sorted_hindex;
        auto inv = inv_sorted_hindex;
        Kokkos::parallel_for("get_inverted_sorted_hindex", OxPolicy(0, N),
                             KOKKOS_LAMBDA(int i) { inv(s(i)) = i; });
    }

    // MD_CUDABackend::_sort_particles (+ CUDAMixedBackend conversions)
    void sort_particles(ParticleArrays &p) {
        mixed_double_to_float(p);   // (possd, velsd, Lsd, orientationsd) -> float
        sort_index(p.poss);
        auto s = sorted_hindex;
        auto inv = inv_sorted_hindex;
        auto poss = p.poss, vels = p.vels, Ls = p.Ls, ori = p.orientations;
        auto bonds = p.bonds;
        auto strand = p.strand;
        auto btype = p.btype;
        auto ptype = p.ptype;
        auto bp = buff_poss, bv = buff_vels, bL = buff_Ls, bo = buff_orientations;
        auto bb = buff_bonds;
        auto bs = buff_strand;
        auto bbt = buff_btype;
        auto bpt = buff_ptype;
        Kokkos::parallel_for("permute_particles", OxPolicy(0, N), KOKKOS_LAMBDA(int i) {
            const int j = s(i);
            const LR_bonds b = bonds(j);
            LR_bonds nb = {-1, -1};
            if (b.n3 >= 0) nb.n3 = inv(b.n3);
            if (b.n5 >= 0) nb.n5 = inv(b.n5);
            bb(i) = nb;
            for (int c = 0; c < 4; c++) {
                bo(i, c) = ori(j, c);
                bp(i, c) = poss(j, c);
                bv(i, c) = vels(j, c);
                bL(i, c) = Ls(j, c);
            }
            bs(i) = strand(j);
            bbt(i) = btype(j);
            bpt(i) = ptype(j);
        });
        Kokkos::deep_copy(p.orientations, buff_orientations);
        Kokkos::deep_copy(p.poss, buff_poss);
        Kokkos::deep_copy(p.bonds, buff_bonds);
        Kokkos::deep_copy(p.vels, buff_vels);
        Kokkos::deep_copy(p.Ls, buff_Ls);
        Kokkos::deep_copy(p.strand, buff_strand);
        Kokkos::deep_copy(p.btype, buff_btype);
        Kokkos::deep_copy(p.ptype, buff_ptype);
        p.init_strand_ends();   // beyond upstream (its _d_is_strand_end is not permuted)
        if constexpr (OXDNA_MIXED) {
            float4_to_double4(p.orientations, p.orientationsd);
            float4_to_double4(p.poss, p.possd);
            float4_to_double4(p.vels, p.velsd);
            float4_to_double4(p.Ls, p.Lsd);
        }
    }
};
