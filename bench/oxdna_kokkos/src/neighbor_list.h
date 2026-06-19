#pragma once

// Cell-list + flat edge list neighbor list for GPU-parallel oxDNA.
// Strategy (mirrors CUDASimpleVerletList.cu from standalone oxDNA):
//   1. bin_particles: assign particles to 3D cells
//   2. build_neighbor_matrix: for each particle, check 27-cell neighbourhood,
//      keep pairs (i<j) within cutoff that are NOT bonded (n3/n5 exclusion)
//   3. compress_to_edge_list: Kokkos::parallel_scan builds a flat pair list
//      (one entry per unique pair), eliminating duplicates.
//
// Result: edge_i[k] and edge_j[k] for k=0..N_edges-1, where each pair
// is stored once with i<j. Main force kernel iterates 0..N_edges with
// one Kokkos thread per pair — no inner neighbor loop, perfect load balance.

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>

struct NeighborList {
    // Flat edge list: one entry per unique pair (i,j) with i<j
    Kokkos::View<int *> edge_i;
    Kokkos::View<int *> edge_j;
    int N_edges = 0;
    int edge_capacity = 0;   // allocated length of edge_i/edge_j (grow-only)

    // Verlet skin: list rebuilt when any particle moves > skin/2 from list pos
    c_number skin;
    c_number cutoff;
    c_number cutsq;     // (cutoff)^2

    // Reference positions at last build
    Kokkos::View<c_number *[4]> list_poss;
    // Flag: needs rebuild?
    Kokkos::View<int *> d_needs_rebuild;

    // Internal: dense neighbor matrix and per-particle counts
    Kokkos::View<int *>    d_num_neigh;      // number of neighbours per particle
    Kokkos::View<int *>    d_neigh_offsets;  // prefix-sum offsets
    Kokkos::View<int **>   d_neigh_matrix;   // [N][max_neigh]
    int max_neigh = 0;

    // Cell list
    Kokkos::View<int *>    d_cell_count;     // particles per cell
    Kokkos::View<int *>    d_cell_offset;    // prefix-sum for cell start
    Kokkos::View<int *>    d_cell_members;   // flat list of particle ids per cell
    int N_cells = 0;
    int max_per_cell = 0;
    int Ncx = 0, Ncy = 0, Ncz = 0;         // cells per dimension

    void init(c_number cut, c_number skin_in, int N, const SimBox &box) {
        cutoff = cut;
        skin   = skin_in;
        cutsq  = (cut + skin) * (cut + skin);  // rebuild using extended cutoff+skin

        list_poss      = Kokkos::View<c_number *[4]>("list_poss", N);
        d_needs_rebuild= Kokkos::View<int *>("needs_rebuild", 1);
        d_num_neigh    = Kokkos::View<int *>("num_neigh", N);
        d_neigh_offsets= Kokkos::View<int *>("neigh_offsets", N + 1);

        // cell dimensions: at least 1 cell, cell edge >= cutoff+skin
        c_number cell_size = cutoff + skin;
        Ncx = std::max(1, static_cast<int>(box.Lx / cell_size));
        Ncy = std::max(1, static_cast<int>(box.Ly / cell_size));
        Ncz = std::max(1, static_cast<int>(box.Lz / cell_size));
        N_cells = Ncx * Ncy * Ncz;

        // Conservative upper bound: 20 particles per cell
        max_per_cell = std::max(20, 4 * N / N_cells + 8);
        d_cell_count  = Kokkos::View<int *>("cell_count",   N_cells);
        d_cell_offset = Kokkos::View<int *>("cell_offset",  N_cells + 1);
        d_cell_members= Kokkos::View<int *>("cell_members", N_cells * max_per_cell);

        // Conservative upper bound for max neighbours per particle:
        // ~26 cells * max_per_cell
        max_neigh = std::min(N - 1, 27 * max_per_cell);
        d_neigh_matrix = Kokkos::View<int **>("neigh_matrix", N, max_neigh);

        edge_i = Kokkos::View<int *>("edge_i", 0);
        edge_j = Kokkos::View<int *>("edge_j", 0);
    }

    // Full rebuild
    void build(const ParticleArrays &p, const SimBox &box);

    // Check if rebuild needed (max displacement > skin/2)
    bool needs_rebuild(const ParticleArrays &p, const SimBox &box);
};

// -----------------------------------------------------------------------
// Kernel tags
// -----------------------------------------------------------------------
struct TagClearCells {};
struct TagBinParticles {};
struct TagCellPrefixSum {};
struct TagBuildNeighMatrix {};
struct TagCountNeighNoDuplicates {};
struct TagCompressEdges {};
struct TagCheckDisplacement {};

// -----------------------------------------------------------------------
// Functor owning all views (passed by value to Kokkos::parallel_* )
// -----------------------------------------------------------------------
struct NeighListFunctor {
    Kokkos::View<c_number *[4]> poss;
    Kokkos::View<LR_bonds *>    bonds;
    Kokkos::View<int *>         d_cell_count;
    Kokkos::View<int *>         d_cell_offset;
    Kokkos::View<int *>         d_cell_members;
    Kokkos::View<int *>         d_num_neigh;
    Kokkos::View<int **>        d_neigh_matrix;
    Kokkos::View<int *>         d_neigh_offsets;
    Kokkos::View<int *>         edge_i;
    Kokkos::View<int *>         edge_j;
    Kokkos::View<c_number *[4]> list_poss;
    Kokkos::View<int *>         d_needs_rebuild;
    SimBox box;
    c_number cutsq;
    c_number skin_half_sq;
    int N, Ncx, Ncy, Ncz, max_per_cell, max_neigh, N_cells;

    KOKKOS_INLINE_FUNCTION
    int cell_index(c_number x, c_number y, c_number z) const {
        auto ci = [](c_number v, c_number L, int Nc) {
            int c = static_cast<int>((v / L + 0.5) * Nc);
            if (c < 0)  c += Nc;
            if (c >= Nc) c -= Nc;
            return c;
        };
        return ci(x, box.Lx, Ncx) * Ncy * Ncz
             + ci(y, box.Ly, Ncy) * Ncz
             + ci(z, box.Lz, Ncz);
    }

    // 1. Clear cell counts
    KOKKOS_INLINE_FUNCTION
    void operator()(TagClearCells, int i) const {
        d_cell_count(i) = 0;
    }

    // 2. Bin each particle into a cell (atomic increment)
    KOKKOS_INLINE_FUNCTION
    void operator()(TagBinParticles, int i) const {
        int cid = cell_index(poss(i,0), poss(i,1), poss(i,2));
        int slot = Kokkos::atomic_fetch_add(&d_cell_count(cid), 1);
        if (slot < max_per_cell)
            d_cell_members(cid * max_per_cell + slot) = i;
    }

    // 3. Build dense neighbour matrix
    KOKKOS_INLINE_FUNCTION
    void operator()(TagBuildNeighMatrix, int i) const {
        c_number xi = poss(i,0), yi = poss(i,1), zi = poss(i,2);
        int n3i = bonds(i).n3, n5i = bonds(i).n5;

        auto ci = [](c_number v, c_number L, int Nc) {
            int c = static_cast<int>((v / L + 0.5) * Nc);
            if (c < 0)  c += Nc;
            if (c >= Nc) c -= Nc;
            return c;
        };
        int cx0 = ci(xi, box.Lx, Ncx);
        int cy0 = ci(yi, box.Ly, Ncy);
        int cz0 = ci(zi, box.Lz, Ncz);

        int count = 0;
        for (int dcx = -1; dcx <= 1; dcx++) {
        for (int dcy = -1; dcy <= 1; dcy++) {
        for (int dcz = -1; dcz <= 1; dcz++) {
            int cx = (cx0 + dcx + Ncx) % Ncx;
            int cy = (cy0 + dcy + Ncy) % Ncy;
            int cz = (cz0 + dcz + Ncz) % Ncz;
            int cid = cx * Ncy * Ncz + cy * Ncz + cz;
            int ncell = d_cell_count(cid);
            if (ncell > max_per_cell) ncell = max_per_cell;
            for (int k = 0; k < ncell; k++) {
                int j = d_cell_members(cid * max_per_cell + k);
                if (j <= i) continue;       // store pair once (i < j)
                if (j == n3i || j == n5i) continue; // skip bonded
                c_number dx = poss(j,0) - xi;
                c_number dy = poss(j,1) - yi;
                c_number dz = poss(j,2) - zi;
                box.wrap(dx, dy, dz);
                if (dx*dx + dy*dy + dz*dz < cutsq && count < max_neigh) {
                    d_neigh_matrix(i, count++) = j;
                }
            }
        }}}
        d_num_neigh(i) = count;
    }

    // 4. Compress to flat edge list via parallel_scan
    KOKKOS_INLINE_FUNCTION
    void operator()(TagCompressEdges, int i, int &update, bool final) const {
        if (i < N) {
            if (final) d_neigh_offsets(i) = update;
            update += d_num_neigh(i);
        } else if (final) {
            d_neigh_offsets(N) = update;
        }
    }

    // 5. Fill edge arrays after scan
    KOKKOS_INLINE_FUNCTION
    void fill_edges(int i) const {
        int base = d_neigh_offsets(i);
        int cnt  = d_num_neigh(i);
        for (int k = 0; k < cnt; k++) {
            edge_i(base + k) = i;
            edge_j(base + k) = d_neigh_matrix(i, k);
        }
    }

    // 6. Check displacement
    KOKKOS_INLINE_FUNCTION
    void operator()(TagCheckDisplacement, int i, int &flag) const {
        c_number dx = poss(i,0) - list_poss(i,0);
        c_number dy = poss(i,1) - list_poss(i,1);
        c_number dz = poss(i,2) - list_poss(i,2);
        box.wrap(dx, dy, dz);
        if (dx*dx + dy*dy + dz*dz > skin_half_sq) flag = 1;
    }
};

// -----------------------------------------------------------------------
// NeighborList implementation
// -----------------------------------------------------------------------
inline void NeighborList::build(const ParticleArrays &p, const SimBox &box) {
    int N = p.N;
    NeighListFunctor f;
    f.poss           = p.poss;
    f.bonds          = p.bonds;
    f.d_cell_count   = d_cell_count;
    f.d_cell_offset  = d_cell_offset;
    f.d_cell_members = d_cell_members;
    f.d_num_neigh    = d_num_neigh;
    f.d_neigh_matrix = d_neigh_matrix;
    f.d_neigh_offsets= d_neigh_offsets;
    f.list_poss      = list_poss;
    f.d_needs_rebuild= d_needs_rebuild;
    f.box            = box;
    f.cutsq          = cutsq;
    f.skin_half_sq   = (skin / 2) * (skin / 2);
    f.N              = N;
    f.Ncx = Ncx; f.Ncy = Ncy; f.Ncz = Ncz;
    f.max_per_cell   = max_per_cell;
    f.max_neigh      = max_neigh;
    f.N_cells        = N_cells;

    Kokkos::parallel_for("clear_cells", N_cells, KOKKOS_LAMBDA(int i) {
        f.d_cell_count(i) = 0;
    });
    Kokkos::parallel_for("bin_particles", N, KOKKOS_LAMBDA(int i) {
        f(TagBinParticles{}, i);
    });
    Kokkos::parallel_for("build_neigh_matrix", N, KOKKOS_LAMBDA(int i) {
        f(TagBuildNeighMatrix{}, i);
    });

    // prefix sum to get total edge count
    int total_edges = 0;
    Kokkos::parallel_scan("compress_edges",
        Kokkos::RangePolicy<>(0, N + 1),
        KOKKOS_LAMBDA(int i, int &update, bool final) {
            if (i < N) {
                if (final) f.d_neigh_offsets(i) = update;
                update += f.d_num_neigh(i);
            } else if (final) {
                f.d_neigh_offsets(N) = update;
            }
        });

    // Read total from host
    {
        auto h_off = Kokkos::create_mirror_view(d_neigh_offsets);
        Kokkos::deep_copy(h_off, d_neigh_offsets);
        total_edges = h_off(N);
    }
    N_edges = total_edges;

    // Grow-only edge arrays: reallocate only when the count exceeds capacity,
    // avoiding a device malloc/free on every rebuild (the count is steady once
    // equilibrated). The force kernel iterates only [0, N_edges).
    if (total_edges > edge_capacity) {
        edge_capacity = total_edges + total_edges / 5 + 64;  // ~20% headroom
        edge_i = Kokkos::View<int *>("edge_i", edge_capacity);
        edge_j = Kokkos::View<int *>("edge_j", edge_capacity);
    }
    f.edge_i = edge_i;
    f.edge_j = edge_j;

    // Fill edge arrays
    Kokkos::parallel_for("fill_edges", N, KOKKOS_LAMBDA(int i) {
        f.fill_edges(i);
    });

    // Save reference positions for displacement check
    Kokkos::deep_copy(list_poss, p.poss);
    Kokkos::deep_copy(d_needs_rebuild, 0);
}

inline bool NeighborList::needs_rebuild(const ParticleArrays &p, const SimBox &box) {
    int N = p.N;
    c_number skin_half_sq = (skin / 2) * (skin / 2);
    auto poss_d    = p.poss;
    auto lp        = list_poss;
    auto box_d     = box;

    int flag = 0;
    Kokkos::parallel_reduce("check_disp", N,
        KOKKOS_LAMBDA(int i, int &flag_l) {
            c_number dx = poss_d(i,0) - lp(i,0);
            c_number dy = poss_d(i,1) - lp(i,1);
            c_number dz = poss_d(i,2) - lp(i,2);
            box_d.wrap(dx, dy, dz);
            if (dx*dx + dy*dy + dz*dz > skin_half_sq) flag_l = 1;
        },
        Kokkos::Max<int>(flag));

    return (flag != 0);
}
