#pragma once

// Verlet list, a port of the oxDNA default CUDA list (CUDA_list = verlet,
// src/CUDA/Lists/CUDASimpleVerletList.cu + CUDA_simple_verlet.cuh):
//
//   init (once):
//     cells per side = floor(L / rverlet) (at least 3), capped by
//       cells_auto_optimisation (default true) at ceil((2N/V)^(1/3) L);
//     max_N_per_cell = round(max_density_multiplier (3) * the largest cell
//       occupancy of the initial configuration) (count kernel + device max,
//       one host read), clamped to [5, N+1];
//     max_neigh = min(4/3 pi max_N_per_cell, N - 1); the neighbour matrix is
//       stored column-major, matrix[j * N + i] (Kokkos LayoutLeft).
//   update (whenever the first-step kernel flagged are_lists_old):
//     1. memset of the cell counters;
//     2. simple_fill_cells: one thread per particle, atomic slot in its cell,
//        sets a host-pinned overflow flag if a cell is full;
//     3. device synchronisation and host check of the overflow flag (upstream
//        throws; here the cells are grown and step 2 repeated, see below);
//     4. simple_update_neigh_list: one thread per particle, visits its cell and
//        the 26 neighbouring cells (upstream visiting order), keeps the
//        non-bonded particles with |r_ij|^2 < rverlet^2 (minimum image) --
//        a FULL list (both i->j and j->i), writes list_poss[i] = r_i and the
//        neighbour count.
//     With use_edge = true step 4 is edge_update_neigh_list (also counts the
//     neighbours m < i), followed by a host write of offsets[N] = 0, a device
//     exclusive scan of the counts, a host read of N_edges = offsets[N] and
//     compress_matrix_neighs, which writes the edges (from = i, to = m < i).
//   rverlet = rcut + 2 * verlet_skin (float constant verlet_sqr_rverlet), where
//   rcut is the interaction cutoff of the upstream interaction class.
//
// Beyond upstream: on a cell overflow (upstream throws) the cells are grown
// (max_N_per_cell doubled) and re-filled, and a neighbour-matrix overflow
// (more than max_neigh neighbours; upstream writes out of bounds) sets the
// flag too and grows the matrix; both print a warning. A particle is never
// dropped silently (the previous version of this code did that).

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

// CUDA edge_bond
struct EdgeBond {
    int from, to;
};

struct NeighborList {
    // settings
    double rcut = 0, verlet_skin = 0;
    float  sqr_rverlet = 0;          // verlet_sqr_rverlet (__constant__ float)
    bool   use_edge = false;
    bool   auto_optimisation = true; // cells_auto_optimisation
    double max_density_multiplier = 3;
    int    N = 0;

    // cells
    int N_cells_side[3] = {0, 0, 0};
    int N_cells = 0, max_N_per_cell = 0;
    Kokkos::View<int *> d_counters_cells;
    Kokkos::View<int *> d_cells;

    // neighbour matrix (column-major as upstream: element (i, j) at j * N + i)
    int max_neigh = 0;
    Kokkos::View<int **, Kokkos::LayoutLeft> d_matrix_neighs;
    Kokkos::View<int *> d_number_neighs;

    // edge list (use_edge)
    Kokkos::View<EdgeBond *> d_edge_list;
    Kokkos::View<int *> d_number_neighs_no_doubles;
    int N_edges = 0;

    // positions at the last update (_d_list_poss, written by the update kernel)
    Vec4 list_poss;

    // host-pinned overflow flag (_d_cell_overflow)
    PinnedFlag cell_overflow;

    int N_updates = 0;

    void compute_N_cells_side(const SimBox &box, double min_cell_size) {
        const double sides[3] = {box.Lx, box.Ly, box.Lz};
        const double V = sides[0] * sides[1] * sides[2];
        const double max_factor = std::pow(2. * N / V, 1. / 3.);
        for (int i = 0; i < 3; i++) {
            N_cells_side[i] = (int)(std::floor(sides[i] / min_cell_size) + 0.1);
            if (N_cells_side[i] < 3) N_cells_side[i] = 3;
            if (auto_optimisation && N_cells_side[i] > std::ceil(max_factor * sides[i]))
                N_cells_side[i] = (int)std::ceil(max_factor * sides[i]);
        }
    }

    // _largest_N_in_cells: count kernel + device max (thrust::max_element)
    int largest_N_in_cells(const Vec4 &poss, const SimBox &box) const {
        Kokkos::View<int *> counters("counters_cells_tmp", N_cells);
        const int nx = N_cells_side[0], ny = N_cells_side[1], nz = N_cells_side[2];
        Kokkos::parallel_for("count_N_in_cells", Kokkos::RangePolicy<Kokkos::LaunchBounds<64, 0>>(0, N),
            KOKKOS_LAMBDA(int i) {
                const int cx = box.cell_coord(poss(i, 0), box.Lx, nx);
                const int cy = box.cell_coord(poss(i, 1), box.Ly, ny);
                const int cz = box.cell_coord(poss(i, 2), box.Lz, nz);
                Kokkos::atomic_inc(&counters((cz * ny + cy) * nx + cx));
            });
        int max_N = 0;
        Kokkos::parallel_reduce("max_N_in_cells", N_cells,
            KOKKOS_LAMBDA(int c, int &m) { if (counters(c) > m) m = counters(c); },
            Kokkos::Max<int>(max_N));
        return max_N;
    }

    void init(int N_in, double rcut_in, double skin, const SimBox &box, const Vec4 &poss,
              bool use_edge_in) {
        N = N_in;
        rcut = rcut_in;
        verlet_skin = skin;
        use_edge = use_edge_in;
        const c_number rverlet = static_cast<c_number>(rcut + 2 * verlet_skin);
        const c_number sqr_rv = rverlet * rverlet;
        sqr_rverlet = static_cast<float>(sqr_rv);

        // _init_cells
        compute_N_cells_side(box, std::sqrt(static_cast<double>(sqr_rv)));
        N_cells = N_cells_side[0] * N_cells_side[1] * N_cells_side[2];
        max_N_per_cell = (int)std::round(max_density_multiplier * largest_N_in_cells(poss, box));
        if (max_N_per_cell > N) max_N_per_cell = N + 1;
        if (max_N_per_cell < 5) max_N_per_cell = 5;
        d_counters_cells = Kokkos::View<int *>("counters_cells", N_cells);
        d_cells = Kokkos::View<int *>("cells", static_cast<size_t>(N_cells) * max_N_per_cell);

        max_neigh = std::min((int)(4 * M_PI * max_N_per_cell / 3.), N - 1);
        if (max_neigh < 1) max_neigh = 1;
        d_number_neighs = Kokkos::View<int *>("number_neighs", N);
        d_matrix_neighs = Kokkos::View<int **, Kokkos::LayoutLeft>("matrix_neighs", N, max_neigh);
        cell_overflow = PinnedFlag("cell_overflow");
        cell_overflow() = 0;
        if (use_edge) {
            d_edge_list = Kokkos::View<EdgeBond *>("edge_list", static_cast<size_t>(N) * max_neigh);
            d_number_neighs_no_doubles = Kokkos::View<int *>("number_neighs_no_doubles", N + 1);
        }
        list_poss = Vec4("list_poss", N);
    }

    void update(const ParticleArrays &p, const SimBox &box);
};

// -----------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------
struct FillCellsFunctor {
    Vec4c poss;
    Kokkos::View<int *> cells, counters_cells;
    PinnedFlag cell_overflow;
    SimBox box;
    int nx, ny, nz, max_N_per_cell;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        const int cx = box.cell_coord(poss(i, 0), box.Lx, nx);
        const int cy = box.cell_coord(poss(i, 1), box.Ly, ny);
        const int cz = box.cell_coord(poss(i, 2), box.Lz, nz);
        const int index = (cz * ny + cy) * nx + cx;
        const int slot = Kokkos::atomic_fetch_add(&counters_cells(index), 1);
        if (slot < max_N_per_cell) cells(index * max_N_per_cell + slot) = i;
        if (slot + 1 >= max_N_per_cell) cell_overflow() = 1;
    }
};

template <bool EDGE>
struct UpdateNeighListFunctor {
    Vec4c poss;
    Vec4 list_poss;
    RandomRead<int> counters_cells;   // tex1Dfetch upstream
    Kokkos::View<const int *> cells;
    Kokkos::View<int **, Kokkos::LayoutLeft> matrix_neighs;
    Kokkos::View<int *> number_neighs, number_neighs_no_doubles;
    Kokkos::View<const LR_bonds *> bonds;
    PinnedFlag overflow;
    SimBox box;
    float sqr_rverlet;
    int nx, ny, nz, max_N_per_cell, max_neigh;

    KOKKOS_INLINE_FUNCTION int neigh_cell(int x, int y, int z, int ox, int oy, int oz) const {
        x = (x + nx + ox) % nx;
        y = (y + ny + oy) % ny;
        z = (z + nz + oz) % nz;
        return (z * ny + y) * nx + x;
    }

    KOKKOS_INLINE_FUNCTION void visit(int i, int cell_ind, c_number rx, c_number ry, c_number rz,
                                      const LR_bonds &b, int &N_n, int &N_nd) const {
        const int size = counters_cells(cell_ind);
        for (int k = 0; k < size; k++) {
            const int m = cells(cell_ind * max_N_per_cell + k);
            // no bonded neighbours in our list!
            if (m == i || b.n3 == m || b.n5 == m) continue;
            c_number dx = poss(m, 0) - rx, dy = poss(m, 1) - ry, dz = poss(m, 2) - rz;
            box.wrap(dx, dy, dz);
            if (dx * dx + dy * dy + dz * dz < c_number(sqr_rverlet)) {
                if (N_n < max_neigh) matrix_neighs(i, N_n) = m;
                else overflow() = 1;
                N_n++;
                if (EDGE && i > m) N_nd++;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        const c_number rx = poss(i, 0), ry = poss(i, 1), rz = poss(i, 2), rw = poss(i, 3);
        const LR_bonds b = bonds(i);
        int N_n = 0, N_nd = 0;
        const int x = box.cell_coord(rx, box.Lx, nx);
        const int y = box.cell_coord(ry, box.Ly, ny);
        const int z = box.cell_coord(rz, box.Lz, nz);
        // this cell, then the 26 neighbours grouped into 13 pairs of opposite cells
        constexpr int off[26][3] = {
            {-1, -1, -1}, {+1, +1, +1}, {-1, -1, +1}, {+1, +1, -1}, {-1, +1, +1}, {+1, -1, -1},
            {+1, -1, +1}, {-1, +1, -1}, {-1, -1, 0}, {+1, +1, 0}, {-1, +1, 0}, {+1, -1, 0},
            {-1, 0, -1}, {+1, 0, +1}, {-1, 0, +1}, {+1, 0, -1}, {0, -1, -1}, {0, +1, +1},
            {0, -1, +1}, {0, +1, -1}, {-1, 0, 0}, {+1, 0, 0}, {0, -1, 0}, {0, +1, 0},
            {0, 0, -1}, {0, 0, +1}};
        visit(i, (z * ny + y) * nx + x, rx, ry, rz, b, N_n, N_nd);
        for (int c = 0; c < 26; c++)
            visit(i, neigh_cell(x, y, z, off[c][0], off[c][1], off[c][2]), rx, ry, rz, b, N_n, N_nd);

        list_poss(i, 0) = rx; list_poss(i, 1) = ry; list_poss(i, 2) = rz; list_poss(i, 3) = rw;
        number_neighs(i) = (N_n < max_neigh) ? N_n : max_neigh;
        if (EDGE) number_neighs_no_doubles(i) = N_nd;
    }
};

// -----------------------------------------------------------------------
// CUDASimpleVerletList::update
// -----------------------------------------------------------------------
inline void NeighborList::update(const ParticleArrays &p, const SimBox &box) {
    // _init_cells(poss): host arithmetic only while the box does not change
    compute_N_cells_side(box, std::sqrt(static_cast<double>(static_cast<c_number>(sqr_rverlet))));
    if (N_cells_side[0] * N_cells_side[1] * N_cells_side[2] != N_cells)
        throw std::runtime_error("NeighborList: the number of cells changed (box changes are not supported)");

    for (;;) {
        Kokkos::deep_copy(d_counters_cells, 0);

        FillCellsFunctor fill{p.poss, d_cells, d_counters_cells, cell_overflow, box,
                              N_cells_side[0], N_cells_side[1], N_cells_side[2], max_N_per_cell};
        Kokkos::parallel_for("simple_fill_cells", OxPolicy(0, N), fill);

        Kokkos::fence();   // cudaDeviceSynchronize() before reading the pinned flag
        if (cell_overflow() == 0) break;
        // Upstream throws here ("A cell contains more than _max_n_per_cell
        // particles"); this code grows the cells and re-bins instead (never
        // silently drops a particle; costs nothing unless a cell overflows).
        cell_overflow() = 0;
        if (max_N_per_cell > N)
            throw std::runtime_error("NeighborList: cell overflow with max_N_per_cell > N");
        max_N_per_cell = std::min(2 * max_N_per_cell, N + 1);
        std::fprintf(stderr, "Warning: a Verlet-list cell overflowed (upstream aborts here); "
                     "growing max_N_per_cell to %d and re-binning\n", max_N_per_cell);
        d_cells = Kokkos::View<int *>("cells", static_cast<size_t>(N_cells) * max_N_per_cell);
    }

    auto setup = [&](auto &f) {
        f.poss = p.poss; f.list_poss = list_poss; f.counters_cells = d_counters_cells;
        f.cells = d_cells; f.matrix_neighs = d_matrix_neighs; f.number_neighs = d_number_neighs;
        f.number_neighs_no_doubles = d_number_neighs_no_doubles; f.bonds = p.bonds;
        f.overflow = cell_overflow; f.box = box; f.sqr_rverlet = sqr_rverlet;
        f.nx = N_cells_side[0]; f.ny = N_cells_side[1]; f.nz = N_cells_side[2];
        f.max_N_per_cell = max_N_per_cell; f.max_neigh = max_neigh;
    };

    for (;;) {
        if (use_edge) {
            UpdateNeighListFunctor<true> f;
            setup(f);
            Kokkos::parallel_for("edge_update_neigh_list", OxPolicy(0, N), f);
        } else {
            UpdateNeighListFunctor<false> f;
            setup(f);
            Kokkos::parallel_for("simple_update_neigh_list", OxPolicy(0, N), f);
        }
        Kokkos::fence();
        if (cell_overflow() == 0) break;
        // Neighbour-matrix overflow (upstream writes out of bounds): grow, redo.
        cell_overflow() = 0;
        if (max_neigh >= N - 1)
            throw std::runtime_error("NeighborList: neighbour-matrix overflow with max_neigh = N - 1");
        max_neigh = std::min(2 * max_neigh, N - 1);
        std::fprintf(stderr, "Warning: Verlet neighbour matrix overflowed; growing max_neigh to %d\n", max_neigh);
        d_matrix_neighs = Kokkos::View<int **, Kokkos::LayoutLeft>("matrix_neighs", N, max_neigh);
        if (use_edge) d_edge_list = Kokkos::View<EdgeBond *>("edge_list", static_cast<size_t>(N) * max_neigh);
    }

    if (use_edge) {
        // d_number_neighs_no_doubles_w[_N] = 0 (host write), exclusive scan,
        // N_edges = d_number_neighs_no_doubles_w[_N] (host read)
        auto last = Kokkos::subview(d_number_neighs_no_doubles, N);
        Kokkos::deep_copy(last, 0);
        auto nnd = d_number_neighs_no_doubles;
        Kokkos::parallel_scan("exclusive_scan_no_doubles", N + 1,
            KOKKOS_LAMBDA(int i, int &upd, bool final) {
                const int v = nnd(i);
                if (final) nnd(i) = upd;
                upd += v;
            });
        int h_last = 0;
        Kokkos::deep_copy(h_last, last);
        N_edges = h_last;

        auto matrix = d_matrix_neighs;
        auto nn = d_number_neighs;
        auto edges = d_edge_list;
        Kokkos::parallel_for("compress_matrix_neighs", OxPolicy(0, N), KOKKOS_LAMBDA(int i) {
            int ctr = 0;
            const int off = nnd(i);
            for (int k = 0; k < nn(i); k++) {
                const int m = matrix(i, k);
                if (i > m) {
                    edges(off + ctr) = EdgeBond{i, m};
                    ctr++;
                }
            }
        });
    }
}
