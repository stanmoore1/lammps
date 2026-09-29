#pragma once

// LAMMPS KOKKOS neighbor machinery for the ghost-atom mode (lammps_ghosts):
//
//   NBinKokkos (nbin_kokkos.cpp)           bins over the local + ghost atoms:
//     MemsetZeroFunctor (bincount), NPairKokkosBinAtomsFunctor (atomic slot per
//     bin, overflow -> regrow to the maximum occupancy and re-bin)
//   NStencilBin<HALF,3D,ortho>             half stencil ("upper" bins, own bin first)
//   NPairKokkos half/bin/newton            NPairKokkosBuildFunctor (flat build_Item,
//     one thread per local atom): own bin j > i, ghosts only "above" i
//     (z, then y, then x), then the stencil bins; rsq <= cutneighsq(itype,jtype);
//     bonded (1-2) partners keep the special bit (special_flag = 2); overflow ->
//     maxneighs = 1.2 * max and rebuild; scalars to device / host around the build
//   NeighborKokkos::check_distance          REDUCE over the local atoms, x - xhold
//     against (skin/2)^2 (no minimum image: positions are unwrapped between
//     rebuilds); TagNeighborXhold on every build
//   NeighBondKokkos::bond_all               REDUCE over the local atoms: the bond
//     partner through the atom map, then closest_image() over its images
//     (sametag chain); atomic slot in the bond list; scalars to device / host;
//     then k_bondlist.sync_host() (build_topology_kk)
//
// On the GPU LAMMPS builds the pair list with the shared-memory variant
// (build_ItemGPU, a TeamPolicy with one team per two bins) when the team size
// fits; the bench uses the flat variant, which LAMMPS uses on the host and as
// the GPU fallback (see README).

#include "types.h"
#include "particles.h"
#include "neighbor_list.h"
#include "lammps_comm.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

struct LammpsNeigh {
    double cutneighmax = 0, cutneighmaxsq = 0, skin = 0, triggersq = 0;
    double bboxlo[3] = {}, bboxhi[3] = {}, binsize[3] = {}, bininv[3] = {};
    int nbin[3] = {}, mbinlo[3] = {}, mbin[3] = {};
    int mbins = 0, atoms_per_bin = 16, nstencil = 0;
    Kokkos::View<int *> bincount, atom2bin, stencil;
    Kokkos::View<int **, Kokkos::LayoutRight> bins;
    Kokkos::View<int> d_resize;
    Kokkos::View<int>::host_mirror_type h_resize;
    // neighbor:scalars (resize, new_maxneighs)
    Kokkos::View<int[2]> d_scalars;
    Kokkos::View<int[2]>::host_mirror_type h_scalars;
    Kokkos::View<c_number *[3], Kokkos::LayoutRight> xhold;
    Kokkos::View<c_number **> cutneighsq;                  // (ntypes+1)^2
    // bond list host copy (the list itself is ParticleArrays::bondlist)
    Kokkos::View<int **, Kokkos::LayoutRight>::host_mirror_type h_bondlist;
    Kokkos::View<int[2]> d_bscalars;                       // (nlist, fail_flag)
    Kokkos::View<int[2]>::host_mirror_type h_bscalars;
    long long ncalls = 0;                                  // neighbor->ncalls

    // cutneigh: list radius (cutforce + skin); skin: LAMMPS skin (2 * verlet_skin)
    void setup(const LammpsComm &comm, double cutneigh, double skin_in, double binsize_user = 0) {
        cutneighmax = cutneigh; cutneighmaxsq = cutneigh * cutneigh;
        skin = skin_in; triggersq = 0.25 * skin * skin;
        const double binsize_optimal = (binsize_user > 0) ? binsize_user : 0.5 * cutneighmax;
        const double binsizeinv = 1.0 / binsize_optimal;
        constexpr double SMALL = 1.0e-6;
        for (int d = 0; d < 3; d++) {
            bboxlo[d] = comm.lo[d]; bboxhi[d] = comm.hi[d];
            const double bbox = bboxhi[d] - bboxlo[d];
            nbin[d] = static_cast<int>(bbox * binsizeinv);
            if (nbin[d] == 0) nbin[d] = 1;
            binsize[d] = bbox / nbin[d];
            bininv[d] = 1.0 / binsize[d];
            // ghost extent (bsubbox = subbox +- cutghost)
            double coord = comm.lo[d] - comm.cutghost - SMALL * bbox;
            int lo = static_cast<int>((coord - bboxlo[d]) * bininv[d]);
            if (coord < bboxlo[d]) lo = lo - 1;
            coord = comm.hi[d] + comm.cutghost + SMALL * bbox;
            int hi = static_cast<int>((coord - bboxlo[d]) * bininv[d]);
            mbinlo[d] = lo - 1;
            hi = hi + 1;
            mbin[d] = hi - mbinlo[d] + 1;
        }
        mbins = mbin[0] * mbin[1] * mbin[2];
        bincount = Kokkos::View<int *>("Neighbor::bincount", mbins);
        bins = Kokkos::View<int **, Kokkos::LayoutRight>("Neighbor::bins", mbins, atoms_per_bin);
        d_resize = Kokkos::View<int>("Neighbor::resize");
        h_resize = Kokkos::create_mirror_view(d_resize);
        d_scalars = Kokkos::View<int[2]>("neighbor:scalars");
        h_scalars = Kokkos::create_mirror_view(d_scalars);
        d_bscalars = Kokkos::View<int[2]>("NeighBond:scalars");
        h_bscalars = Kokkos::create_mirror_view(d_bscalars);

        // NStencil::create_setup + NStencilBin<HALF=1, DIM_3D=1, TRI=0>::create
        int s[3];
        for (int d = 0; d < 3; d++) {
            s[d] = static_cast<int>(cutneighmax * bininv[d]);
            if (s[d] * binsize[d] < cutneighmax) s[d]++;
        }
        auto bin_distance = [&](int i, int j, int k) {
            const double dx = (i > 0) ? (i - 1) * binsize[0] : (i == 0 ? 0.0 : (i + 1) * binsize[0]);
            const double dy = (j > 0) ? (j - 1) * binsize[1] : (j == 0 ? 0.0 : (j + 1) * binsize[1]);
            const double dz = (k > 0) ? (k - 1) * binsize[2] : (k == 0 ? 0.0 : (k + 1) * binsize[2]);
            return dx * dx + dy * dy + dz * dz;
        };
        std::vector<int> st;
        st.push_back(0);
        for (int k = 0; k <= s[2]; k++)
            for (int j = -s[1]; j <= s[1]; j++)
                for (int i = -s[0]; i <= s[0]; i++) {
                    if (k <= 0 && j <= 0 && (j != 0 || i <= 0)) continue;
                    if (bin_distance(i, j, k) < cutneighmaxsq)
                        st.push_back(k * mbin[1] * mbin[0] + j * mbin[0] + i);
                }
        nstencil = static_cast<int>(st.size());
        stencil = Kokkos::View<int *>("NStencil::stencil", nstencil);
        auto h_st = Kokkos::create_mirror_view(stencil);
        for (int n = 0; n < nstencil; n++) h_st(n) = st[n];
        Kokkos::deep_copy(stencil, h_st);

        cutneighsq = Kokkos::View<c_number **>("neigh:cutneighsq", 5, 5);
        Kokkos::deep_copy(cutneighsq, static_cast<c_number>(cutneighmaxsq));
    }

    // NBinKokkos::bin_atoms over the local + ghost atoms
    void bin_atoms(const ParticleArrays &p) {
        const int nall = p.N + p.nghost;
        if (atom2bin.extent_int(0) < p.nmax) atom2bin = Kokkos::View<int *>("Neighbor::atom2bin", p.nmax);
        h_resize() = 1;
        while (h_resize() > 0) {
            h_resize() = 0;
            Kokkos::deep_copy(d_resize, h_resize);
            auto bc = bincount;
            Kokkos::parallel_for("MemsetZeroFunctor", mbins, KOKKOS_LAMBDA(int i) { bc(i) = 0; });
            auto x = p.poss; auto a2b = atom2bin; auto bn = bins; auto rs = d_resize;
            const double lo0 = bboxlo[0], lo1 = bboxlo[1], lo2 = bboxlo[2];
            const double hi0 = bboxhi[0], hi1 = bboxhi[1], hi2 = bboxhi[2];
            const double bi0 = bininv[0], bi1 = bininv[1], bi2 = bininv[2];
            const int nb0 = nbin[0], nb1 = nbin[1], nb2 = nbin[2];
            const int ml0 = mbinlo[0], ml1 = mbinlo[1], ml2 = mbinlo[2];
            const int mx = mbin[0], my = mbin[1], nmb = mbins;
            Kokkos::parallel_for("NPairKokkosBinAtomsFunctor", nall, KOKKOS_LAMBDA(int i) {
                auto c2b = [](double v, double blo, double bhi, double inv, int nbd) {
                    int ix;
                    if (v >= bhi) ix = static_cast<int>((v - bhi) * inv) + nbd;
                    else if (v >= blo) { ix = static_cast<int>((v - blo) * inv); ix = (ix < nbd - 1) ? ix : nbd - 1; }
                    else ix = static_cast<int>((v - blo) * inv) - 1;
                    return ix;
                };
                const int ix = c2b(static_cast<double>(x(i, 0)), lo0, hi0, bi0, nb0) - ml0;
                const int iy = c2b(static_cast<double>(x(i, 1)), lo1, hi1, bi1, nb1) - ml1;
                const int iz = c2b(static_cast<double>(x(i, 2)), lo2, hi2, bi2, nb2) - ml2;
                const int ibin = iz * my * mx + iy * mx + ix;
                if (ibin < 0 || ibin >= nmb) Kokkos::abort("Atom outside of neighbor bin range");
                a2b(i) = ibin;
                const int ac = Kokkos::atomic_fetch_add(&bc(ibin), 1);
                if (ac < static_cast<int>(bn.extent(1))) bn(ibin, ac) = i;
                else rs() = 1;
            });
            Kokkos::deep_copy(h_resize, d_resize);
            if (h_resize()) {
                int maxc = 0;
                Kokkos::parallel_reduce("NBinKokkos::max_bincount", mbins,
                    KOKKOS_LAMBDA(int i, int &m) { if (bc(i) > m) m = bc(i); }, Kokkos::Max<int>(maxc));
                atoms_per_bin = maxc + ((maxc / 10 > 16) ? maxc / 10 : 16);
                bins = Kokkos::View<int **, Kokkos::LayoutRight>("Neighbor::bins", mbins, atoms_per_bin);
            }
        }
    }

    // NPairKokkos<half, newton>::build (flat build_Item over the local atoms)
    void build_pairs(const ParticleArrays &p, NeighborList &nl) {
        const int nlocal = p.N;
        if (nl.d_num_neigh.extent_int(0) < p.nmax) nl.d_num_neigh = Kokkos::View<int *>("num_neigh", p.nmax);
        if (nl.d_ilist.extent_int(0) < p.nmax) nl.d_ilist = Kokkos::View<int *>("neighlist:ilist", p.nmax);
        if (nl.d_neigh_matrix.extent_int(0) < p.nmax || nl.max_neigh <= 0) {
            if (nl.max_neigh <= 0) nl.max_neigh = 64;
            nl.d_neigh_matrix = Kokkos::View<int **>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighlist:neighbors"), p.nmax, nl.max_neigh);
        }
        h_scalars(0) = 1;
        while (h_scalars(0)) {
            h_scalars(0) = 0;
            h_scalars(1) = nl.max_neigh;
            Kokkos::deep_copy(d_scalars, h_scalars);
            auto x = p.poss; auto bt = p.btype; auto tg = p.tag; auto bd = p.bonds;
            auto a2b = atom2bin; auto bc = bincount; auto bn = bins; auto st = stencil;
            auto nnum = nl.d_num_neigh; auto nmat = nl.d_neigh_matrix; auto ilist = nl.d_ilist;
            auto sc = d_scalars; auto cn = cutneighsq;
            const int ns = nstencil, maxn = nl.max_neigh;
            const double xh = 0.5 * (bboxhi[0] - bboxlo[0]), yh = 0.5 * (bboxhi[1] - bboxlo[1]),
                         zh = 0.5 * (bboxhi[2] - bboxlo[2]);
            Kokkos::parallel_for("NPairKokkosBuildFunctor<half,newton>", nlocal, KOKKOS_LAMBDA(int i) {
                int n = 0;
                const double xtmp = static_cast<double>(x(i, 0));
                const double ytmp = static_cast<double>(x(i, 1));
                const double ztmp = static_cast<double>(x(i, 2));
                const int itype = bt(i) + 1;
                const int ibin = a2b(i);
                const LR_bonds bi = bd(i);
                auto consider = [&](int j) {
                    const int jtype = bt(j) + 1;
                    const double delx = xtmp - static_cast<double>(x(j, 0));
                    const double dely = ytmp - static_cast<double>(x(j, 1));
                    const double delz = ztmp - static_cast<double>(x(j, 2));
                    const double rsq = delx * delx + dely * dely + delz * delz;
                    if (rsq <= static_cast<double>(cn(itype, jtype))) {
                        // find_special: 1-2 partners (tags of the 3'/5' neighbours)
                        const int tj = tg(j);
                        const int which = (tj == bi.n3 || tj == bi.n5) ? 1 : 0;
                        int entry = j;
                        if (which != 0) {
                            // minimum_image_check: a special partner farther than
                            // half the box is stored as a plain neighbor
                            const bool far = Kokkos::fabs(delx) > xh || Kokkos::fabs(dely) > yh ||
                                             Kokkos::fabs(delz) > zh;
                            if (!far) entry = j ^ (which << OX_SBBITS);
                        }
                        if (n < maxn) nmat(i, n++) = entry;
                        else n++;
                    }
                };
                // rest of i's own bin: owned j beyond i, ghosts "above" i
                for (int m = 0; m < bc(ibin); m++) {
                    const int j = bn(ibin, m);
                    if (j <= i) continue;
                    if (j >= nlocal) {
                        if (static_cast<double>(x(j, 2)) < ztmp) continue;
                        if (static_cast<double>(x(j, 2)) == ztmp) {
                            if (static_cast<double>(x(j, 1)) < ytmp) continue;
                            if (static_cast<double>(x(j, 1)) == ytmp && static_cast<double>(x(j, 0)) < xtmp) continue;
                        }
                    }
                    consider(j);
                }
                for (int k = 0; k < ns; k++) {
                    const int jbin = ibin + st(k);
                    if (jbin == ibin) continue;
                    for (int m = 0; m < bc(jbin); m++) consider(bn(jbin, m));
                }
                nnum(i) = n;
                if (n > maxn) {
                    sc(0) = 1;
                    if (n > sc(1)) sc(1) = n;   // as LAMMPS: no atomics (safe in the resize loop)
                }
                ilist(i) = i;
            });
            Kokkos::deep_copy(h_scalars, d_scalars);
            if (h_scalars(0)) {
                nl.max_neigh = static_cast<int>(h_scalars(1) * 1.2);
                nl.d_neigh_matrix = Kokkos::View<int **>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "neighlist:neighbors"), p.nmax, nl.max_neigh);
            }
        }
    }

    // number of pairs in the list (diagnostics only; not part of a step)
    static int count_pairs(const ParticleArrays &p, const NeighborList &nl) {
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.d_num_neigh);
        int tot = 0;
        for (int i = 0; i < p.N; i++) tot += h(i);
        return tot;
    }

    // NeighborKokkos::build_kokkos: xhold of the local atoms (dist_check)
    void store_xhold(const ParticleArrays &p) {
        if (xhold.extent_int(0) < p.nmax) xhold = Kokkos::View<c_number *[3], Kokkos::LayoutRight>("neigh:xhold", p.nmax);
        auto x = p.poss; auto xh = xhold;
        Kokkos::parallel_for("TagNeighborXhold", p.N, KOKKOS_LAMBDA(int i) {
            xh(i, 0) = x(i, 0); xh(i, 1) = x(i, 1); xh(i, 2) = x(i, 2);
        });
    }

    // NeighborKokkos::check_distance_kokkos
    bool check_distance(const ParticleArrays &p) const {
        auto x = p.poss; auto xh = xhold;
        const double tsq = triggersq;
        int flag = 0;
        Kokkos::parallel_reduce("TagNeighborCheckDistance", p.N, KOKKOS_LAMBDA(int i, int &f) {
            const double dx = static_cast<double>(x(i, 0) - xh(i, 0));
            const double dy = static_cast<double>(x(i, 1) - xh(i, 1));
            const double dz = static_cast<double>(x(i, 2) - xh(i, 2));
            if (dx * dx + dy * dy + dz * dz > tsq) f = 1;
        }, Kokkos::Max<int>(flag));
        return flag > 0;   // (the Max reducer starts from INT_MIN)
    }

    // NeighBondKokkos::bond_all (newton_bond on) into p.bondlist / p.nbonds,
    // then k_bondlist.sync_host() (build_topology_kk)
    void bond_all(ParticleArrays &p) {
        if (p.bondlist.extent_int(0) == 0) {
            p.bondlist = Kokkos::View<int **, Kokkos::LayoutRight>("neighbor:neighbor->bondlist", p.N + 16, 3);
        }
        int nmissing = 0;
        int fail = 0;
        do {
            nmissing = 0;
            Kokkos::deep_copy(d_bscalars, 0);
            auto nb = p.num_bond; auto ba = p.bond_atom; auto map = p.map_array; auto st = p.sametag;
            auto x = p.poss; auto bl = p.bondlist; auto sc = d_bscalars;
            const int mb = p.bondlist.extent_int(0);
            Kokkos::parallel_reduce("TagNeighBondBondAll", p.N, KOKKOS_LAMBDA(int i, int &miss) {
                for (int m = 0; m < nb(i); m++) {
                    int atom1 = map(ba(i, m));
                    if (atom1 == -1) { miss++; continue; }
                    // closest_image(i, atom1)
                    const double xi0 = static_cast<double>(x(i, 0)), xi1 = static_cast<double>(x(i, 1)),
                                 xi2 = static_cast<double>(x(i, 2));
                    int j = atom1, closest = atom1;
                    double dx = xi0 - static_cast<double>(x(j, 0)), dy = xi1 - static_cast<double>(x(j, 1)),
                           dz = xi2 - static_cast<double>(x(j, 2));
                    double rsqmin = dx * dx + dy * dy + dz * dz;
                    while (st(j) >= 0) {
                        j = st(j);
                        dx = xi0 - static_cast<double>(x(j, 0));
                        dy = xi1 - static_cast<double>(x(j, 1));
                        dz = xi2 - static_cast<double>(x(j, 2));
                        const double rsq = dx * dx + dy * dy + dz * dz;
                        if (rsq < rsqmin) { rsqmin = rsq; closest = j; }
                    }
                    atom1 = closest;
                    const int n = Kokkos::atomic_fetch_add(&sc(0), 1);
                    if (n >= mb && !sc(1)) sc(1) = 1;
                    if (sc(1)) continue;
                    bl(n, 0) = i;
                    bl(n, 1) = atom1;
                    bl(n, 2) = 1;
                }
            }, nmissing);
            Kokkos::deep_copy(h_bscalars, d_bscalars);
            p.nbonds = h_bscalars(0);
            fail = h_bscalars(1);
            if (fail)
                p.bondlist = Kokkos::View<int **, Kokkos::LayoutRight>("neighbor:neighbor->bondlist", p.nbonds + 10000, 3);
        } while (fail);
        if (nmissing) throw std::runtime_error("lammps_ghosts: bond atoms missing");
        if (p.prime_bond.extent_int(0) < p.nbonds)
            p.prime_bond = Kokkos::View<int *[4], Kokkos::LayoutLeft>("prime_bond", p.nbonds);
        if (h_bondlist.extent(0) != p.bondlist.extent(0)) h_bondlist = Kokkos::create_mirror_view(p.bondlist);
        dual_sync(h_bondlist, p.bondlist);
    }
};
