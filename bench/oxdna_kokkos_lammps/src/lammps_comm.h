#pragma once

// Ghost atoms and per-step communication of a single-rank LAMMPS KOKKOS run
// (lammps_ghosts = 1): the periodic images of the atoms within the ghost
// cutoff are stored after the local atoms, and every exchange with the
// neighbouring "processor" is a swap with self. Mirrors src/KOKKOS/comm_kokkos.cpp
// (CommKokkos, "comm device", i.e. the GPU default) and domain_kokkos.cpp /
// atom_kokkos.cpp of LAMMPS oxdna3KK-kk-fixes:
//
//   rebuild steps (Verlet: domain->pbc(), comm->exchange(), comm->borders()):
//     domain_pbc() DomainPBCFunctor: fold the local atoms into the box and
//                  update their image flags
//     map_clear()  atom map reset (ViewFill of map_array)
//     exchange()   3 x (count reset, BuildExchangeListFunctor over the local
//                  atoms, count to host, copylist_bonus fill): on one rank no
//                  atom leaves after pbc(), so nothing is packed
//     borders()    6 swaps (x lo/hi, y lo/hi, z lo/hi) x (count to device,
//                  BuildBorderListFunctor (TeamPolicy + team_scan), count to
//                  host, PackBorder, PackBorderBonus, UnpackBorder,
//                  UnpackBorderBonus); later dimensions also scan the ghosts
//                  of earlier ones (edge / corner images); then
//                  copy_swap_info() on the host (sendlist to host, per-ghost
//                  pbc / g2l tables to device) and map_set (2 kernels)
//   every other step (comm->forward_comm()):
//     forward_comm()  AtomVecKokkos_PackCommSelfFused (x) +
//                     AtomVecEllipsoidKokkos_PackCommSelfFusedBonus (quat)
//   every step (Verlet, newton on: comm->reverse_comm()):
//     reverse_comm()  AtomVecKokkos_PackReverseSelfFused (f + torque, atomics:
//                     kk-fixes e40c43a443)
//
// Border data per atom (atom_style hybrid bond ellipsoid oxdna): x, tag,
// type, mask, molecule, id3p, id5p, qeff, rmass, ellipsoid flag (13 values)
// + bonus shape and quat (7 values). The bench keeps the oxDNA type twice
// (btype for oxDNA1/2, ptype for oxDNA3) and id3p/id5p as bonds(i) (owner
// indices, the bench's tags).

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>
#include <stdexcept>

// DualView-style synchronisation of a device View and its host mirror: a real
// copy on a GPU, nothing when the mirror aliases the device data (host
// backends), exactly as LAMMPS' DualView::sync_host() / sync_device().
template <class Dst, class Src>
inline void dual_sync(const Dst &dst, const Src &src) {
    if (static_cast<const void *>(dst.data()) != static_cast<const void *>(src.data()))
        Kokkos::deep_copy(dst, src);
}

struct LammpsComm {
    static constexpr int NSWAP = 6;
    static constexpr int SIZE_BORDER = 13;
    static constexpr int SIZE_BORDER_BONUS = 7;

    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0}, prd[3] = {0, 0, 0};
    double cutghost = 0;

    // per swap (host copies of the CommBrick arrays)
    int sendnum[NSWAP] = {}, firstrecv[NSWAP] = {}, pbc_flag[NSWAP] = {};
    int pbc[NSWAP][3] = {};
    double slablo[NSWAP] = {}, slabhi[NSWAP] = {};
    int maxsendlist = 0;
    Kokkos::View<int **, Kokkos::LayoutRight> sendlist;    // (NSWAP, maxsendlist)
    Kokkos::View<int> d_total_send;
    Kokkos::View<int>::host_mirror_type h_total_send;
    Kokkos::View<double **, Kokkos::LayoutRight> buf, buf_bonus;

    // fused self-communication (copy_swap_info)
    int totalsend = 0;
    Kokkos::View<int *> d_sendnum_scan, d_firstrecv;       // NSWAP
    Kokkos::View<int *> d_pbc_flag, d_g2l;                 // totalsend
    Kokkos::View<int *[3], Kokkos::LayoutRight> d_pbc;     // totalsend

    // exchange
    Kokkos::View<int> d_count;
    Kokkos::View<int>::host_mirror_type h_count;
    Kokkos::View<int *> exchange_sendlist, exchange_copylist_bonus;

    // atom map error flag (the map itself is ParticleArrays::map_array / sametag)
    Kokkos::View<int> d_map_error;
    Kokkos::View<int>::host_mirror_type h_map_error;

    // Set up the box and the swap slabs (CommBrick::setup for one rank, all
    // dimensions periodic, maxneed = 1: the ghost cutoff is below the box).
    void setup(const SimBox &box, const double box_lo[3], double cutghost_in, int N) {
        cutghost = cutghost_in;
        const double L[3] = {(double)box.Lx, (double)box.Ly, (double)box.Lz};
        for (int d = 0; d < 3; d++) {
            lo[d] = box_lo[d]; prd[d] = L[d]; hi[d] = lo[d] + L[d];
            if (cutghost > L[d])
                throw std::runtime_error("lammps_ghosts: ghost cutoff larger than the box is not supported");
        }
        constexpr double BIG = 1.0e20;
        for (int d = 0; d < 3; d++) {
            const int s0 = 2 * d, s1 = 2 * d + 1;
            // send to the lower neighbour (self): atoms near lo, image +L
            slablo[s0] = -BIG;            slabhi[s0] = lo[d] + cutghost;
            pbc_flag[s0] = 1;             pbc[s0][0] = pbc[s0][1] = pbc[s0][2] = 0; pbc[s0][d] = 1;
            // send to the upper neighbour (self): atoms near hi, image -L
            slablo[s1] = hi[d] - cutghost; slabhi[s1] = BIG;
            pbc_flag[s1] = 1;             pbc[s1][0] = pbc[s1][1] = pbc[s1][2] = 0; pbc[s1][d] = -1;
        }
        maxsendlist = 64;
        sendlist = Kokkos::View<int **, Kokkos::LayoutRight>("comm:sendlist", NSWAP, maxsendlist);
        d_total_send = Kokkos::View<int>("comm:total_send");
        h_total_send = Kokkos::create_mirror_view(d_total_send);
        buf = Kokkos::View<double **, Kokkos::LayoutRight>("comm:buf_send", maxsendlist, SIZE_BORDER);
        buf_bonus = Kokkos::View<double **, Kokkos::LayoutRight>("comm:buf_send_bonus", maxsendlist, SIZE_BORDER_BONUS);
        d_sendnum_scan = Kokkos::View<int *>("comm:sendnum_scan", NSWAP);
        d_firstrecv    = Kokkos::View<int *>("comm:firstrecv", NSWAP);
        d_count = Kokkos::View<int>("comm:k_count");
        h_count = Kokkos::create_mirror_view(d_count);
        exchange_sendlist       = Kokkos::View<int *>("comm:k_exchange_sendlist", 100);
        exchange_copylist_bonus = Kokkos::View<int *>("comm:k_exchange_copylist_bonus", 100);
        (void)N;
        d_map_error = Kokkos::View<int>("atom:error_flag");
        h_map_error = Kokkos::create_mirror_view(d_map_error);
    }

    // ------------------------------------------------------------------
    // domain->pbc(): DomainPBCFunctor over the local atoms
    // ------------------------------------------------------------------
    void domain_pbc(ParticleArrays &p) const {
        auto x = p.poss; auto img = p.image;
        const double l0 = lo[0], l1 = lo[1], l2 = lo[2];
        const double h0 = hi[0], h1 = hi[1], h2 = hi[2];
        const double p0 = prd[0], p1 = prd[1], p2 = prd[2];
        Kokkos::parallel_for("DomainPBCFunctor", p.N, KOKKOS_LAMBDA(int i) {
            const double lov[3] = {l0, l1, l2}, hiv[3] = {h0, h1, h2}, pv[3] = {p0, p1, p2};
            for (int d = 0; d < 3; d++) {
                if (static_cast<double>(x(i, d)) < lov[d]) {
                    x(i, d) += static_cast<c_number>(pv[d]);
                    img(i, d)--;
                }
                if (static_cast<double>(x(i, d)) >= hiv[d]) {
                    x(i, d) -= static_cast<c_number>(pv[d]);
                    img(i, d)++;
                    if (static_cast<double>(x(i, d)) < lov[d]) x(i, d) = static_cast<c_number>(lov[d]);
                }
            }
        });
    }

    // atom->map_clear() (device map: fill map_array with -1)
    void map_clear(ParticleArrays &p) const {
        Kokkos::deep_copy(p.map_array, -1);
    }

    // ------------------------------------------------------------------
    // comm->exchange(): per dimension, find the local atoms outside
    // [lo, hi) (none after pbc() on one rank; nothing is packed or sent)
    // ------------------------------------------------------------------
    void exchange(ParticleArrays &p) {
        if (exchange_sendlist.extent_int(0) < p.N) {
            exchange_sendlist       = Kokkos::View<int *>("comm:k_exchange_sendlist", p.N);
            exchange_copylist_bonus = Kokkos::View<int *>("comm:k_exchange_copylist_bonus", p.N);
        }
        auto x = p.poss; auto sl = exchange_sendlist; auto cnt = d_count;
        for (int d = 0; d < 3; d++) {
            Kokkos::deep_copy(d_count, 0);
            const double l = lo[d], h = hi[d];
            Kokkos::parallel_for("BuildExchangeListFunctor", p.N, KOKKOS_LAMBDA(int i) {
                const double xi = static_cast<double>(x(i, d));
                if (xi < l || xi >= h) {
                    const int m = Kokkos::atomic_fetch_add(&cnt(), 1);
                    if (m < static_cast<int>(sl.extent(0))) sl(m) = i;
                }
            });
            dual_sync(h_count, d_count);
            if (h_count() != 0)
                throw std::runtime_error("lammps_ghosts: an atom left the box after pbc()");
            // no bonus data leaves (kk-fixes: count_bonus == 0 path)
            Kokkos::deep_copy(exchange_copylist_bonus, -1);
        }
    }

    // ------------------------------------------------------------------
    // comm->borders(): create the ghost atoms
    // ------------------------------------------------------------------
    void borders(ParticleArrays &p) {
        p.nghost = 0;
        int iswap = 0;
        for (int d = 0; d < 3; d++) {
            int nfirst = 0, nlast = 0;
            for (int ineed = 0; ineed < 2; ineed++, iswap++) {
                if (ineed % 2 == 0) { nfirst = nlast; nlast = p.N + p.nghost; }
                nfirst = 0;   // one rank, maxneed = 1: every swap of a dimension scans [0, nlast)
                int nsend = build_border_list(p, iswap, nfirst, nlast, d);
                if (nsend > maxsendlist) {
                    grow_list(nsend);
                    nsend = build_border_list(p, iswap, nfirst, nlast, d);
                }
                if (buf.extent_int(0) < nsend) {
                    buf = Kokkos::View<double **, Kokkos::LayoutRight>("comm:buf_send", nsend + nsend / 5 + 16, SIZE_BORDER);
                    buf_bonus = Kokkos::View<double **, Kokkos::LayoutRight>("comm:buf_send_bonus", nsend + nsend / 5 + 16,
                                                                           SIZE_BORDER_BONUS);
                }
                const int first = p.N + p.nghost;
                if (first + nsend > p.nmax) p.grow(first + nsend + (first + nsend) / 5 + 16);
                pack_border(p, iswap, nsend);
                unpack_border(p, first, nsend);
                sendnum[iswap] = nsend;
                firstrecv[iswap] = first;
                p.nghost += nsend;
            }
        }
        copy_swap_info(p);
        map_set(p);
    }

    int build_border_list(ParticleArrays &p, int iswap, int nfirst, int nlast, int dim) {
        h_total_send() = 0;
        dual_sync(d_total_send, h_total_send);
        auto x = p.poss; auto sl = sendlist; auto ns = d_total_send;
        const double l = slablo[iswap], h = slabhi[iswap];
        const int maxs = maxsendlist;
        const int n = nlast - nfirst;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        const int team_size = 128;
#else
        const int team_size = 1;
#endif
        using TP = Kokkos::TeamPolicy<>;
        const int league = (n + team_size - 1) / team_size;
        if (league > 0) {
            Kokkos::parallel_for("BuildBorderListFunctor", TP(league, team_size),
                KOKKOS_LAMBDA(const TP::member_type &dev) {
                    const int chunk = (n + dev.league_size() - 1) / dev.league_size();
                    const int teamstart = chunk * dev.league_rank() + nfirst;
                    const int teamend = (teamstart + chunk) < nlast ? (teamstart + chunk) : nlast;
                    int mysend = 0;
                    for (int i = teamstart + dev.team_rank(); i < teamend; i += dev.team_size()) {
                        const double xi = static_cast<double>(x(i, dim));
                        if (xi >= l && xi <= h) mysend++;
                    }
                    const int my_store_pos = dev.team_scan(mysend, &ns());
                    if (my_store_pos + mysend < maxs) {
                        mysend = my_store_pos;
                        for (int i = teamstart + dev.team_rank(); i < teamend; i += dev.team_size()) {
                            const double xi = static_cast<double>(x(i, dim));
                            if (xi >= l && xi <= h) sl(iswap, mysend++) = i;
                        }
                    }
                });
        }
        dual_sync(h_total_send, d_total_send);
        return h_total_send();
    }

    void grow_list(int n) {
        maxsendlist = n + n / 2 + 16;
        Kokkos::resize(sendlist, NSWAP, maxsendlist);
    }

    // AtomVecKokkos_PackBorder + AtomVecEllipsoidKokkos_PackBorderBonus
    void pack_border(ParticleArrays &p, int iswap, int nsend) {
        auto x = p.poss; auto tg = p.tag; auto bt = p.btype; auto pt = p.ptype; auto mk = p.mask;
        auto mol = p.molecule; auto bd = p.bonds; auto q = p.qeff; auto rm = p.rmass;
        auto ell = p.ellipsoid; auto sh = p.shape; auto quat = p.orientations;
        auto sl = sendlist; auto b = buf; auto bb = buf_bonus;
        const double dx = pbc[iswap][0] * prd[0], dy = pbc[iswap][1] * prd[1], dz = pbc[iswap][2] * prd[2];
        Kokkos::parallel_for("AtomVecKokkos_PackBorder", nsend, KOKKOS_LAMBDA(int i) {
            const int j = sl(iswap, i);
            b(i, 0) = static_cast<double>(x(j, 0)) + dx;
            b(i, 1) = static_cast<double>(x(j, 1)) + dy;
            b(i, 2) = static_cast<double>(x(j, 2)) + dz;
            b(i, 3) = tg(j);
            b(i, 4) = bt(j) + 16 * pt(j);          // type (btype and oxDNA3 ptype)
            b(i, 5) = mk(j);
            b(i, 6) = mol(j);
            b(i, 7) = bd(j).n3;                    // id3p
            b(i, 8) = bd(j).n5;                    // id5p
            b(i, 9) = q(j);                        // qeff
            b(i, 10) = rm(j);                      // rmass
            b(i, 11) = ell(j) >= 0 ? 1 : 0;        // ellipsoid flag
            b(i, 12) = 0;
        });
        Kokkos::parallel_for("AtomVecEllipsoidKokkos_PackBorderBonus", nsend, KOKKOS_LAMBDA(int i) {
            const int j = sl(iswap, i);
            const int e = ell(j);
            if (e < 0) return;
            bb(i, 0) = sh(e, 0); bb(i, 1) = sh(e, 1); bb(i, 2) = sh(e, 2);
            bb(i, 3) = quat(e, 0); bb(i, 4) = quat(e, 1); bb(i, 5) = quat(e, 2); bb(i, 6) = quat(e, 3);
        });
    }

    // AtomVecKokkos_UnpackBorder + AtomVecEllipsoidKokkos_UnpackBorderBonus
    void unpack_border(ParticleArrays &p, int first, int nrecv) {
        auto x = p.poss; auto tg = p.tag; auto bt = p.btype; auto pt = p.ptype; auto mk = p.mask;
        auto mol = p.molecule; auto bd = p.bonds; auto q = p.qeff; auto rm = p.rmass;
        auto ell = p.ellipsoid; auto sh = p.shape; auto quat = p.orientations;
        auto b = buf; auto bb = buf_bonus;
        Kokkos::parallel_for("AtomVecKokkos_UnpackBorder", nrecv, KOKKOS_LAMBDA(int i) {
            const int k = first + i;
            x(k, 0) = static_cast<c_number>(b(i, 0));
            x(k, 1) = static_cast<c_number>(b(i, 1));
            x(k, 2) = static_cast<c_number>(b(i, 2));
            tg(k) = static_cast<int>(b(i, 3));
            const int t = static_cast<int>(b(i, 4));
            bt(k) = t % 16; pt(k) = static_cast<uint8_t>(t / 16);
            mk(k) = static_cast<int>(b(i, 5));
            mol(k) = static_cast<int>(b(i, 6));
            bd(k).n3 = static_cast<int>(b(i, 7));
            bd(k).n5 = static_cast<int>(b(i, 8));
            q(k) = static_cast<c_number>(b(i, 9));
            rm(k) = static_cast<c_number>(b(i, 10));
            ell(k) = (b(i, 11) != 0) ? k : -1;     // ghost bonus appended after the local ones
        });
        Kokkos::parallel_for("AtomVecEllipsoidKokkos_UnpackBorderBonus", nrecv, KOKKOS_LAMBDA(int i) {
            const int k = first + i;
            const int e = ell(k);
            if (e < 0) return;
            sh(e, 0) = static_cast<c_number>(bb(i, 0)); sh(e, 1) = static_cast<c_number>(bb(i, 1));
            sh(e, 2) = static_cast<c_number>(bb(i, 2));
            quat(e, 0) = static_cast<c_number>(bb(i, 3)); quat(e, 1) = static_cast<c_number>(bb(i, 4));
            quat(e, 2) = static_cast<c_number>(bb(i, 5)); quat(e, 3) = static_cast<c_number>(bb(i, 6));
        });
    }

    // CommKokkos::copy_swap_info(): host-side per-ghost tables for the fused
    // self communication (sendlist to host, pbc / g2l tables to device)
    void copy_swap_info(ParticleArrays &p) {
        auto h_sendlist = Kokkos::create_mirror_view(sendlist);
        dual_sync(h_sendlist, sendlist);
        auto h_scan = Kokkos::create_mirror_view(d_sendnum_scan);
        auto h_first = Kokkos::create_mirror_view(d_firstrecv);
        int scan = 0;
        for (int s = 0; s < NSWAP; s++) {
            scan += sendnum[s];
            h_scan(s) = scan;
            h_first(s) = firstrecv[s];
        }
        totalsend = scan;
        if (d_pbc_flag.extent_int(0) < totalsend) {
            const int n = totalsend + totalsend / 5 + 16;
            d_pbc_flag = Kokkos::View<int *>("comm:pbc_flag", n);
            d_g2l = Kokkos::View<int *>("comm:g2l", n);
            d_pbc = Kokkos::View<int *[3], Kokkos::LayoutRight>("comm:pbc", n);
        }
        auto h_pf = Kokkos::create_mirror_view(d_pbc_flag);
        auto h_g2l = Kokkos::create_mirror_view(d_g2l);
        auto h_pbc = Kokkos::create_mirror_view(d_pbc);
        const int nlocal = p.N;
        for (int s = 0; s < NSWAP; s++)
            for (int i = 0; i < sendnum[s]; i++) {
                const int source = h_sendlist(s, i) - nlocal;
                const int dest = firstrecv[s] + i - nlocal;
                h_pf(dest) = pbc_flag[s];
                for (int d = 0; d < 3; d++) h_pbc(dest, d) = pbc[s][d];
                h_g2l(dest) = nlocal + source;
                if (source >= 0) {
                    h_pf(dest) = h_pf(dest) || h_pf(source);
                    for (int d = 0; d < 3; d++) h_pbc(dest, d) += h_pbc(source, d);
                    h_g2l(dest) = h_g2l(source);
                }
            }
        dual_sync(d_sendnum_scan, h_scan);
        dual_sync(d_firstrecv, h_first);
        dual_sync(d_pbc_flag, h_pf);
        dual_sync(d_g2l, h_g2l);
        dual_sync(d_pbc, h_pbc);
    }

    // atom->map_set() (map_set_device): ghosts first, then the local atoms,
    // so map(tag) ends on the owned atom; sametag chains all images of a tag
    void map_set(ParticleArrays &p) {
        auto map = p.map_array; auto st = p.sametag; auto tg = p.tag; auto err = d_map_error;
        const int nlocal = p.N, nall = p.N + p.nghost;
        Kokkos::parallel_for("AtomKokkos::map_set_device (ghosts)", nall - nlocal, KOKKOS_LAMBDA(int ii) {
            const int i = nall - 1 - ii;
            const int t = tg(i);
            if (t < 0 || t >= static_cast<int>(map.extent(0))) { err() = 1; return; }
            st(i) = Kokkos::atomic_exchange(&map(t), i);
        });
        Kokkos::deep_copy(d_map_error, 0);
        Kokkos::parallel_for("AtomKokkos::map_set_device (local)", nlocal, KOKKOS_LAMBDA(int ii) {
            const int i = nlocal - 1 - ii;
            const int t = tg(i);
            if (t < 0 || t >= static_cast<int>(map.extent(0))) { err() = 1; return; }
            st(i) = Kokkos::atomic_exchange(&map(t), i);
        });
        Kokkos::deep_copy(h_map_error, d_map_error);
        if (h_map_error()) throw std::runtime_error("lammps_ghosts: invalid tag in map_set");
    }

    // ------------------------------------------------------------------
    // comm->forward_comm(): fused self communication of x and the bonus quat
    // ------------------------------------------------------------------
    void forward_comm(ParticleArrays &p) const {
        if (totalsend <= 0) return;
        auto x = p.poss; auto sl = sendlist; auto scan = d_sendnum_scan; auto frst = d_firstrecv;
        auto pf = d_pbc_flag; auto pb = d_pbc; auto g2l = d_g2l;
        const double p0 = prd[0], p1 = prd[1], p2 = prd[2];
        Kokkos::parallel_for("AtomVecKokkos_PackCommSelfFused", totalsend, KOKKOS_LAMBDA(int ii) {
            int iswap = 0;
            while (ii >= scan(iswap)) iswap++;
            int i = ii;
            if (iswap > 0) i = ii - scan(iswap - 1);
            const int nfirst = frst(iswap);
            const int nlocal = frst(0);
            int j = sl(iswap, i);
            if (j >= nlocal) j = g2l(j - nlocal);
            if (pf(ii) == 0) {
                x(i + nfirst, 0) = x(j, 0);
                x(i + nfirst, 1) = x(j, 1);
                x(i + nfirst, 2) = x(j, 2);
            } else {
                x(i + nfirst, 0) = x(j, 0) + static_cast<c_number>(pb(ii, 0) * p0);
                x(i + nfirst, 1) = x(j, 1) + static_cast<c_number>(pb(ii, 1) * p1);
                x(i + nfirst, 2) = x(j, 2) + static_cast<c_number>(pb(ii, 2) * p2);
            }
        });
        auto ell = p.ellipsoid; auto quat = p.orientations;
        Kokkos::parallel_for("AtomVecEllipsoidKokkos_PackCommSelfFusedBonus", totalsend, KOKKOS_LAMBDA(int ii) {
            int iswap = 0;
            while (ii >= scan(iswap)) iswap++;
            int i = ii;
            if (iswap > 0) i = ii - scan(iswap - 1);
            const int nfirst = frst(iswap);
            const int nlocal = frst(0);
            int j = sl(iswap, i);
            if (j >= nlocal) j = g2l(j - nlocal);
            if (ell(i + nfirst) >= 0 && ell(j) >= 0) {
                const int eg = ell(i + nfirst), el = ell(j);
                quat(eg, 0) = quat(el, 0); quat(eg, 1) = quat(el, 1);
                quat(eg, 2) = quat(el, 2); quat(eg, 3) = quat(el, 3);
            }
        });
    }

    // ------------------------------------------------------------------
    // comm->reverse_comm(): fused self communication of f and torque; all
    // ghost images of an atom are reduced in one kernel, so atomics
    // ------------------------------------------------------------------
    void reverse_comm(ParticleArrays &p) const {
        if (totalsend <= 0) return;
        auto f = p.forces; auto t = p.torques; auto sl = sendlist; auto scan = d_sendnum_scan;
        auto frst = d_firstrecv; auto g2l = d_g2l;
        Kokkos::parallel_for("AtomVecKokkos_PackReverseSelfFused", totalsend, KOKKOS_LAMBDA(int ii) {
            int iswap = 0;
            while (ii >= scan(iswap)) iswap++;
            int i = ii;
            if (iswap > 0) i = ii - scan(iswap - 1);
            const int nfirst = frst(iswap);
            const int nlocal = frst(0);
            int j = sl(iswap, i);
            if (j >= nlocal) j = g2l(j - nlocal);
            Kokkos::atomic_add(&f(j, 0), f(i + nfirst, 0));
            Kokkos::atomic_add(&f(j, 1), f(i + nfirst, 1));
            Kokkos::atomic_add(&f(j, 2), f(i + nfirst, 2));
            Kokkos::atomic_add(&t(j, 0), t(i + nfirst, 0));
            Kokkos::atomic_add(&t(j, 1), t(i + nfirst, 1));
            Kokkos::atomic_add(&t(j, 2), t(i + nfirst, 2));
        });
    }
};
