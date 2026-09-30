#pragma once

// LAMMPS-faithful oxDNA3 kernels (tracks the LAMMPS KOKKOS oxDNA3 styles on
// LAMMPS branch origin/oxdna3KK-kk-fixes, 2afff57fb5), with the physics of the
// CUDA-faithful sibling bench/oxdna_kokkos (upstream oxDNA DNA3_nomesh):
// every kernel calls the dna3:: physics functions of dna3_forces.h, so the
// energies, forces and torques equal the sibling's to floating-point round-off.
// Only the kernel structure (which list each term iterates, what it reads, how
// it scatters, launch policy) follows LAMMPS, exactly as for oxDNA1/2 in
// dna_forces.h / bonded.h.
//
// Per MD step (GPU, HALFTHREAD, newton on), LAMMPS order:
//   pre_force : LRF (compute_lrf)
//   Pair      : [prime_neighs_pair] excv -> [stk's prime_neighs_bond] stk ->
//               hbond -> [prime_neighs_oxdna3_xstk] xstk -> coaxstk -> dh
//   Bond      : [fene's prime_neighs_bond] fene (+ overstretch flag host copy
//               on energy steps)
// Bracketed precomputes only run in lammps_overhead mode on neighbor-rebuild
// steps. Kernels:
//   * excv (oxdna3/excv): one thread per atom a over the half list incl. the
//     special (bonded 1-2) pairs, OxdnaRangePolicy, atom a in registers,
//     atomics to b per active site pair; order bkbk (x special_lj), bk(a)-bs(b),
//     bs(a)-bk(b), bs-bs. Bonded pairs (topology test) use the tetramer tables
//     (bonded excluded volume), nonbonded pairs the NO_TYPE-flank tables.
//   * stk (oxdna3/stk), fene (oxdna3/fene): lean = per-particle gather
//     (bonded_part<>, like the sibling), lammps_overhead = per-bond atomic
//     scatter reading (a, b, a3p, b5p) from the prime_bond table (bonded_pair<>).
//   * hbond, xstk, coaxstk: one thread per screened pair (fix OXDNA/NPAIR),
//     plain RangePolicy, special pairs exit first. xstk reads its four flank
//     indices from the per-screened-pair prime table (overhead) or bonds()
//     (lean) and, in lammps_overhead mode, scatters in two rounds of atomics
//     (force + site torques, then the pure torques: 18 atomics) like LAMMPS.
//   * dh (oxdna3/dh == oxdna2/dh): the oxDNA2 DHFunctor (identical backbone
//     site and Debye-Huckel form), per atom over the half list, per-atom qeff.
// Pair orientation: for a pair a < b (half list / screened list) the physics
// is evaluated with p = b (the larger index) and q = a, r = pos(a) - pos(b),
// which is the orientation of the sibling's CUDA edge list, so pair-asymmetric
// tables are used identically.

#include "../types.h"
#include "../particles.h"
#include "../neighbor_list.h"
#include "params.h"
#include "params_dna3.h"
#include "dna3_forces.h"
#include "dna_forces.h"   // ScatterF4, compute_lrf, unpack_pair, launch_term, DHFunctor, build_prime_pair
#include "bonded.h"       // bond_precompute, ensure_bondlist
#include <Kokkos_Core.hpp>
#include <stdexcept>

// oxDNA3 parameter bundle used by the LAMMPS-faithful driver: the DNA3 tables
// plus a DNAParams carrying only what the (shared) oxDNA2 dh kernel reads.
struct DNA3Model {
    DNA3Params p;
    DNAParams  dh;
};

inline DNA3Model make_dna3_model(const DNA3Options &o, bool lammps_coaxstk_terminal = false) {
    DNA3Model m;
    m.p = make_oxdna3_params(o);
    m.p.cxst_terminal_only = lammps_coaxstk_terminal;
    m.p.cxst_t4_blunt      = lammps_coaxstk_terminal;
    for (int t = 1; t < 4; t++)
        if (m.p.pos_back1[t] != m.p.pos_back1[0] || m.p.pos_back2[t] != m.p.pos_back2[0])
            throw std::runtime_error("oxDNA3: type-dependent backbone sites are not supported by the dh kernel");
    DNAParams &d = m.dh;
    d = DNAParams{};
    d.model          = 3;
    d.pb1            = m.p.pos_back1[0];
    d.pb2            = m.p.pos_back2[0];
    d.dh_enabled     = true;
    d.dh_half_ends   = m.p.dh_half_ends;
    d.dh_prefactor   = m.p.dh_prefactor;
    d.dh_minus_kappa = m.p.dh_minus_kappa;
    d.dh_RHIGH       = m.p.dh_RHIGH;
    d.dh_RC          = m.p.dh_RC;
    d.dh_B           = m.p.dh_B;
    d.cutsq_nb       = m.p.cutsq_nb;
    d.screen_cutsq   = m.p.screen_cutsq;
    return m;
}

// Kernel selection (validation tools only; the MD loop runs all of them).
namespace dna3k {
enum : int {
    EXCV = 1, STK = 2, HBOND = 4, XSTK = 8, COAXSTK = 16, DH = 32, FENE = 64,
    PAIR = EXCV | STK | HBOND | XSTK | COAXSTK | DH,
    ALL  = PAIR | FENE
};
// excv sub-masks (validation): nonbonded / bonded excluded volume
enum : int { EXCV_NB = 1, EXCV_BONDED = 2, EXCV_BOTH = 3 };

using dna3::v3;
using dna3sd::NO_TYPE;

KOKKOS_INLINE_FUNCTION v3 ld3(const Vec4cr &v, int i) { return {v(i, 0), v(i, 1), v(i, 2)}; }

KOKKOS_INLINE_FUNCTION int flank_type(const RandomRead<uint8_t> &pt, int idx) {
    return (idx < 0) ? NO_TYPE : int(pt(idx));
}

// Scatter the result of particle_particle_interaction (F, T on p) to both
// particles: p gets F, T; q gets -F and -T + r x F (r = q - p).
template <class AF, class AT>
KOKKOS_INLINE_FUNCTION void scatter_pq(const AF &af, const AT &at, int p, int q,
                                       const v3 &r, const v3 &F, const v3 &T) {
    const v3 Tq = dna3::cross(r, F) - T;
    af(p, 0) += F.x;  af(p, 1) += F.y;  af(p, 2) += F.z;
    at(p, 0) += T.x;  at(p, 1) += T.y;  at(p, 2) += T.z;
    af(q, 0) -= F.x;  af(q, 1) -= F.y;  af(q, 2) -= F.z;
    at(q, 0) += Tq.x; at(q, 1) += Tq.y; at(q, 2) += Tq.z;
}
} // namespace dna3k

using dna3::v3;

// -----------------------------------------------------------------------
// EXCV (pair oxdna3/excv): per atom over the half list incl. bonded pairs.
// USE_PRIME: flank indices of bonded pairs from the prime_neighs_pair table
// (lammps_overhead), else from bonds(). MASK selects nonbonded/bonded excv
// (validation; production = EXCV_BOTH).
// -----------------------------------------------------------------------
template <bool USE_PRIME, int MASK = dna3k::EXCV_BOTH>
struct Excv3Functor {
    Vec4cr poss, nx, ny;
    RandomRead<LR_bonds> bonds;
    RandomRead<uint8_t> ptype;
    RandomRead<int> tag;
    Kokkos::View<const int *>  ilist;
    Kokkos::View<const int *>  num_neigh;
    Kokkos::View<const int **> neigh_matrix;
    Kokkos::View<const int ***> prime_pair;
    DNA3Params par;
    ScatterF4 sf, st;
    SimBox box;

    // One excluded-volume site pair: rc = (site of q) - (site of p), where p
    // is b (P_IS_B) or a. Parameters from table family member m at the
    // tetramer index (t0, t1, t2, t3). The force on p is Fe; atom a is kept
    // in registers, atom b gets atomics.
    template <class AF, class AT>
    KOKKOS_INLINE_FUNCTION
    void term(const v3 &rc, bool p_is_b, const v3 &sa, const v3 &sb, int m,
              int t0, int t1, int t2, int t3, c_number factor,
              c_acc (&ftmp)[3], c_acc (&ttmp)[3], const AF &af, const AT &at, int ib,
              c_acc &ev) const {
        using namespace dna3sd;
        const c_number rcut = par(EXCL_RC + m, t0, t1, t2, t3);
        if (dna3::dot(rc, rc) >= rcut * rcut) return;
        v3 Fe;
        c_number e = dna3::excluded_volume(rc, Fe, par(EXCL_S + m, t0, t1, t2, t3),
                                           par(EXCL_R + m, t0, t1, t2, t3),
                                           par(EXCL_B + m, t0, t1, t2, t3), rcut, par.excl_eps);
        Fe = Fe * factor;
        ev += e * factor;
        const v3 ta = dna3::cross(sa, Fe), tb = dna3::cross(sb, Fe);
        if (p_is_b) {
            ftmp[0] -= Fe.x; ftmp[1] -= Fe.y; ftmp[2] -= Fe.z;
            ttmp[0] -= ta.x; ttmp[1] -= ta.y; ttmp[2] -= ta.z;
            af(ib, 0) += Fe.x; af(ib, 1) += Fe.y; af(ib, 2) += Fe.z;
            at(ib, 0) += tb.x; at(ib, 1) += tb.y; at(ib, 2) += tb.z;
        } else {
            ftmp[0] += Fe.x; ftmp[1] += Fe.y; ftmp[2] += Fe.z;
            ttmp[0] += ta.x; ttmp[1] += ta.y; ttmp[2] += ta.z;
            af(ib, 0) -= Fe.x; af(ib, 1) -= Fe.y; af(ib, 2) -= Fe.z;
            at(ib, 0) -= tb.x; at(ib, 1) -= tb.y; at(ib, 2) -= tb.z;
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(int ii) const { c_acc ev = 0; (*this)(ii, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int ii, c_acc &ev) const {
        using namespace dna3k;
        const int ia = USE_PRIME ? ilist(ii) : ii;
        const int m = num_neigh(ia);
        const v3 pa = ld3(poss, ia);
        const int ta = ptype(ia);
        const LR_bonds ba = bonds(ia);
        v3 back_a, stack_a, base_a;
        dna3::set_interaction_sites(par, ta, ld3(nx, ia), ld3(ny, ia), back_a, stack_a, base_a);

        c_acc ftmp[3] = {0, 0, 0}, ttmp[3] = {0, 0, 0};
        auto af = sf.access();
        auto at = st.access();

        for (int k = 0; k < m; k++) {
            const int braw = neigh_matrix(ia, k);
            const c_number factor_lj = ox_sbmask(braw) ? c_number(0) : c_number(1);
            const int ib = braw & OX_NEIGHMASK;
            const int tb = ptype(ib);
            v3 back_b, stack_b, base_b;
            dna3::set_interaction_sites(par, tb, ld3(nx, ib), ld3(ny, ib), back_b, stack_b, base_b);
            v3 r = pa - ld3(poss, ib);                 // r = q - p with p = b, q = a
            box.wrap(r.x, r.y, r.z);

            // topology (tetramer) test: bond 1 = b is the 5' partner of the
            // bond (b.n3 == a), bond 2 = a is (a.n3 == b)
            const LR_bonds bb = bonds(ib);
            int bond = 0, f0 = -1, f3 = -1;
            // lammps_overhead: LAMMPS' tag test tag(a) == id3p(b) && tag(b) == id5p(a)
            const int tag_a = USE_PRIME ? tag(ia) : ia;
            const int tag_b = USE_PRIME ? tag(ib) : ib;
            if (bb.n3 == tag_a && ba.n5 == tag_b) {
                bond = 1;
                if (USE_PRIME) { f0 = prime_pair(ia, k, 0); f3 = prime_pair(ia, k, 1); }
                else           { f0 = ba.n3;                f3 = bb.n5; }
            } else if (bb.n5 == tag_a && ba.n3 == tag_b) {
                bond = 2;
                if (USE_PRIME) { f0 = prime_pair(ia, k, 2); f3 = prime_pair(ia, k, 3); }
                else           { f0 = bb.n3;                f3 = ba.n5; }
            }

            // backbone-backbone (x special_lj: zero for bonded pairs)
            if ((MASK & EXCV_NB) != 0)
                term(r + back_a - back_b, true, back_a, back_b, 0, NO_TYPE, ta, tb, NO_TYPE,
                     factor_lj, ftmp, ttmp, af, at, ib, ev);
            if (bond == 0) {
                if ((MASK & EXCV_NB) == 0) continue;
                // nonbonded: (NO_TYPE, type(q), type(p), NO_TYPE), p = b
                term(r + back_a - base_b, true, back_a, base_b, 3, NO_TYPE, ta, tb, NO_TYPE,
                     c_number(1), ftmp, ttmp, af, at, ib, ev);    // bk(a)-bs(b): p-base/q-back
                term(r + base_a - back_b, true, base_a, back_b, 2, NO_TYPE, ta, tb, NO_TYPE,
                     c_number(1), ftmp, ttmp, af, at, ib, ev);    // bs(a)-bk(b): p-back/q-base
                term(r + base_a - base_b, true, base_a, base_b, 1, NO_TYPE, ta, tb, NO_TYPE,
                     c_number(1), ftmp, ttmp, af, at, ib, ev);    // bs-bs
            } else {
                if ((MASK & EXCV_BONDED) == 0) continue;
                const int t0 = flank_type(ptype, f0), t3 = flank_type(ptype, f3);
                if (bond == 1) {
                    // n5 = b, n3 = a: (type(a.n3), ta, tb, type(b.n5)), r = pos(a) - pos(b)
                    term(r + back_a - base_b, true, back_a, base_b, 5, t0, ta, tb, t3,
                         c_number(1), ftmp, ttmp, af, at, ib, ev);    // n5-base vs n3-back
                    term(r + base_a - back_b, true, base_a, back_b, 6, t0, ta, tb, t3,
                         c_number(1), ftmp, ttmp, af, at, ib, ev);    // n5-back vs n3-base
                    term(r + base_a - base_b, true, base_a, base_b, 4, t0, ta, tb, t3,
                         c_number(1), ftmp, ttmp, af, at, ib, ev);    // base-base
                } else {
                    // n5 = a, n3 = b: (type(b.n3), tb, ta, type(a.n5)), r' = pos(b) - pos(a)
                    const v3 rr = -r;
                    term(rr + base_b - back_a, false, back_a, base_b, 6, t0, tb, ta, t3,
                         c_number(1), ftmp, ttmp, af, at, ib, ev);    // n5-back vs n3-base
                    term(rr + back_b - base_a, false, base_a, back_b, 5, t0, tb, ta, t3,
                         c_number(1), ftmp, ttmp, af, at, ib, ev);    // n5-base vs n3-back
                    term(rr + base_b - base_a, false, base_a, base_b, 4, t0, tb, ta, t3,
                         c_number(1), ftmp, ttmp, af, at, ib, ev);    // base-base
                }
            }
        }

        af(ia, 0) += ftmp[0]; af(ia, 1) += ftmp[1]; af(ia, 2) += ftmp[2];
        at(ia, 0) += ttmp[0]; at(ia, 1) += ttmp[1]; at(ia, 2) += ttmp[2];
    }
};

// -----------------------------------------------------------------------
// HBOND (pair oxdna3/hbond): one thread per screened pair. Special pairs exit
// first, then the base-pair type test (only complementary pairs bond).
// -----------------------------------------------------------------------
struct Hbond3Functor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<uint8_t> ptype;
    Kokkos::View<const uint64_t *> sp;
    DNA3Params par;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int e) const { c_acc ev = 0; (*this)(e, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int e, c_acc &ev) const {
        using namespace dna3k;
        int ia, braw;
        unpack_pair(sp(e), ia, braw);
        if (ox_sbmask(braw)) return;                 // special_lj = 0
        const int ib = braw & OX_NEIGHMASK;
        const int ta = ptype(ia), tb = ptype(ib);
        if (ta + tb != 3) return;                    // A-T / G-C only (alpha_hb == 0)

        v3 r = ld3(poss, ia) - ld3(poss, ib);
        box.wrap(r.x, r.y, r.z);
        const dna3::NeighTypes none{NO_TYPE, NO_TYPE};   // not used by H-bonding
        v3 F = {0, 0, 0}, T = {0, 0, 0};
        const c_number e_p = dna3::particle_particle_interaction<dna3::HYDROGEN_BONDING>(
            par, r, tb, ld3(nx, ib), ld3(ny, ib), ld3(nz, ib),
            ta, ld3(nx, ia), ld3(ny, ia), ld3(nz, ia), F, T, none, none);
        if (e_p == 0) return;
        ev += e_p;
        scatter_pq(sf.template access<OxScatterAtomic>(), st.template access<OxScatterAtomic>(), ib, ia, r, F, T);
    }
};

// -----------------------------------------------------------------------
// XSTK (pair oxdna3/xstk): one thread per screened pair. Special exit first,
// then the 4 flank indices (3'/5' neighbours of a and b) and their 4 types:
// from the per-screened-pair prime table (USE_PRIME, lammps_overhead) or from
// bonds() (lean). SPLIT: LAMMPS' two scatter rounds (force + site torques,
// then the pure torques; 18 atomics); otherwise one round of 12.
// -----------------------------------------------------------------------
template <bool USE_PRIME, bool SPLIT>
struct Xstk3Functor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<uint8_t> ptype;
    RandomRead<LR_bonds> bonds;
    Kokkos::View<const int *[4], Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::RandomAccess>> prime;
    Kokkos::View<const uint64_t *> sp;
    DNA3Params par;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int e) const { c_acc ev = 0; (*this)(e, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int e, c_acc &ev) const {
        using namespace dna3k;
        int ia, braw;
        unpack_pair(sp(e), ia, braw);
        if (ox_sbmask(braw)) return;                 // special_lj = 0
        const int ib = braw & OX_NEIGHMASK;
        const int ta = ptype(ia), tb = ptype(ib);

        int a3, a5, b3, b5;
        if (USE_PRIME) {
            a3 = prime(e, 0); a5 = prime(e, 1); b3 = prime(e, 2); b5 = prime(e, 3);
        } else {
            const LR_bonds ba = bonds(ia), bb = bonds(ib);
            a3 = ba.n3; a5 = ba.n5; b3 = bb.n3; b5 = bb.n5;
        }
        const dna3::NeighTypes q_neighs{flank_type(ptype, a3), flank_type(ptype, a5)};
        const dna3::NeighTypes p_neighs{flank_type(ptype, b3), flank_type(ptype, b5)};

        v3 r = ld3(poss, ia) - ld3(poss, ib);
        box.wrap(r.x, r.y, r.z);
        const v3 b1 = ld3(nx, ib);
        v3 F = {0, 0, 0}, T = {0, 0, 0};
        const c_number e_p = dna3::particle_particle_interaction<dna3::CROSS_STACKING>(
            par, r, tb, b1, ld3(ny, ib), ld3(nz, ib),
            ta, ld3(nx, ia), ld3(ny, ia), ld3(nz, ia), F, T, p_neighs, q_neighs);
        if (e_p == 0) return;
        ev += e_p;
        auto af = sf.template access<OxScatterAtomic>();
        auto at = st.template access<OxScatterAtomic>();
        if (!SPLIT) { scatter_pq(af, at, ib, ia, r, F, T); return; }

        // round 1: forces and the site (r x F) torques
        const v3 Ts = dna3::cross(b1 * par.pos_base[tb], F);
        const v3 Tqs = dna3::cross(r, F) - Ts;
        af(ib, 0) += F.x;   af(ib, 1) += F.y;   af(ib, 2) += F.z;
        at(ib, 0) += Ts.x;  at(ib, 1) += Ts.y;  at(ib, 2) += Ts.z;
        af(ia, 0) -= F.x;   af(ia, 1) -= F.y;   af(ia, 2) -= F.z;
        at(ia, 0) += Tqs.x; at(ia, 1) += Tqs.y; at(ia, 2) += Tqs.z;
        // round 2: the pure (angular) torques
        const v3 Tp = T - Ts;
        at(ib, 0) += Tp.x;  at(ib, 1) += Tp.y;  at(ib, 2) += Tp.z;
        at(ia, 0) -= Tp.x;  at(ia, 1) -= Tp.y;  at(ia, 2) -= Tp.z;
    }
};

// -----------------------------------------------------------------------
// COAXSTK (pair oxdna3/coaxstk): one thread per screened pair, as oxDNA2.
// With the LAMMPS-only terminal criterion (par.cxst_terminal_only) both
// nucleotides must be strand ends. The K selection needs the flank types of
// both nucleotides (bonds() + 4 type reads).
// -----------------------------------------------------------------------
struct Coaxstk3Functor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<uint8_t> ptype;
    RandomRead<LR_bonds> bonds;
    Kokkos::View<const uint64_t *> sp;
    DNA3Params par;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int e) const { c_acc ev = 0; (*this)(e, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int e, c_acc &ev) const {
        using namespace dna3k;
        int ia, braw;
        unpack_pair(sp(e), ia, braw);
        if (ox_sbmask(braw)) return;                 // special_lj = 0
        const int ib = braw & OX_NEIGHMASK;
        const LR_bonds ba = bonds(ia), bb = bonds(ib);
        if (par.cxst_terminal_only) {
            if (ba.n3 >= 0 && ba.n5 >= 0) return;
            if (bb.n3 >= 0 && bb.n5 >= 0) return;
        }
        const int ta = ptype(ia), tb = ptype(ib);
        const dna3::NeighTypes q_neighs{flank_type(ptype, ba.n3), flank_type(ptype, ba.n5)};
        const dna3::NeighTypes p_neighs{flank_type(ptype, bb.n3), flank_type(ptype, bb.n5)};

        v3 r = ld3(poss, ia) - ld3(poss, ib);
        box.wrap(r.x, r.y, r.z);
        v3 F = {0, 0, 0}, T = {0, 0, 0};
        const c_number e_p = dna3::particle_particle_interaction<dna3::COAXIAL_STACKING>(
            par, r, tb, ld3(nx, ib), ld3(ny, ib), ld3(nz, ib),
            ta, ld3(nx, ia), ld3(ny, ia), ld3(nz, ia), F, T, p_neighs, q_neighs);
        if (e_p == 0) return;
        ev += e_p;
        scatter_pq(sf.template access<OxScatterAtomic>(), st.template access<OxScatterAtomic>(), ib, ia, r, F, T);
    }
};

// -----------------------------------------------------------------------
// stk / fene, lean: one thread per particle gathers its two bonds (each bond
// evaluated twice, no atomics) with bonded_part<>, exactly as the sibling's
// DNA3BondedFunctor; the energy is counted on the n5 side.
// -----------------------------------------------------------------------
template <int TERMS>
struct Bonded3GatherFunctor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<LR_bonds> bonds;
    RandomRead<uint8_t> ptype;
    VecA4 forces, torques;
    DNA3Params par;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const { c_acc ev = 0; (*this)(i, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int i, c_acc &ev) const {
        using namespace dna3k;
        const LR_bonds pb = bonds(i);
        if (pb.n3 < 0 && pb.n5 < 0) return;
        const v3 ppos = ld3(poss, i);
        const v3 a1 = ld3(nx, i), a2 = ld3(ny, i), a3 = ld3(nz, i);
        const int pt = ptype(i);
        const int p_n3_type = flank_type(ptype, pb.n3);
        const int p_n5_type = flank_type(ptype, pb.n5);

        v3 F = {0, 0, 0}, T = {0, 0, 0};
        if (pb.n3 >= 0) {
            const int q = pb.n3;
            v3 r = ld3(poss, q) - ppos;
            box.wrap(r.x, r.y, r.z);
            ev += dna3::bonded_part<true, TERMS>(par, r, pt, a1, a2, a3, p_n5_type,
                                                 int(ptype(q)), ld3(nx, q), ld3(ny, q), ld3(nz, q),
                                                 flank_type(ptype, bonds(q).n3), F, T);
        }
        if (pb.n5 >= 0) {
            const int q = pb.n5;
            v3 r = ppos - ld3(poss, q);
            box.wrap(r.x, r.y, r.z);
            dna3::bonded_part<false, TERMS>(par, r, int(ptype(q)), ld3(nx, q), ld3(ny, q), ld3(nz, q),
                                            flank_type(ptype, bonds(q).n5),
                                            pt, a1, a2, a3, p_n3_type, F, T);
        }
        forces(i, 0) += F.x;  forces(i, 1) += F.y;  forces(i, 2) += F.z;
        torques(i, 0) += T.x; torques(i, 1) += T.y; torques(i, 2) += T.z;
    }
};

// -----------------------------------------------------------------------
// stk / fene, lammps_overhead: one thread per bond over the prime_bond table
// (a = 3' end, b = 5' end, a3p, b5p), 4 type reads, one evaluation
// (bonded_pair<>), atomic scatter to both ends. stk returns before any
// atomics when the bond does not stack; fene raises the overstretch flag.
// -----------------------------------------------------------------------
template <int TERMS>
struct Bonded3ScatterFunctor {
    Vec4cr poss, nx, ny, nz;
    Kokkos::View<const int *[4], Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::RandomAccess>> prime;
    RandomRead<uint8_t> ptype;
    Kokkos::View<int> flag;
    ScatterF4 sf, st;
    DNA3Params par;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int in) const { c_acc ev = 0; (*this)(in, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int in, c_acc &ev) const {
        using namespace dna3k;
        constexpr bool IS_FENE = (TERMS & dna3::BACKBONE) != 0;
        const int a = prime(in, 0), b = prime(in, 1);        // a = 3' end (n3), b = 5' end (n5)
        const int a3p = prime(in, 2), b5p = prime(in, 3);
        const int ta = ptype(a), tb = ptype(b);
        const int t3 = flank_type(ptype, a3p), t5 = flank_type(ptype, b5p);

        v3 r = ld3(poss, a) - ld3(poss, b);                  // r = n3pos - n5pos
        box.wrap(r.x, r.y, r.z);
        v3 F5 = {0, 0, 0}, T5 = {0, 0, 0}, T3 = {0, 0, 0};
        c_number denom = 1;
        const c_number e = dna3::bonded_pair<TERMS>(par, r, tb, ld3(nx, b), ld3(ny, b), ld3(nz, b), t5,
                                                    ta, ld3(nx, a), ld3(ny, a), ld3(nz, a), t3,
                                                    F5, T5, T3, denom);
        if (!IS_FENE && e == 0) return;
        if (IS_FENE && !(denom >= c_number(0.2))) flag() = 1;
        ev += e;

        auto af = sf.access();
        auto at = st.access();
        af(b, 0) += F5.x; af(b, 1) += F5.y; af(b, 2) += F5.z;
        at(b, 0) += T5.x; at(b, 1) += T5.y; at(b, 2) += T5.z;
        af(a, 0) -= F5.x; af(a, 1) -= F5.y; af(a, 2) -= F5.z;
        at(a, 0) += T3.x; at(a, 1) += T3.y; at(a, 2) += T3.z;
    }
};

// -----------------------------------------------------------------------
// fix OXDNA/PRIME_NEIGHS::compute_prime_neighs_oxdna3_xstk (lammps_overhead,
// neighbor-rebuild steps, called from the xstk style): one thread per
// screened pair (plain RangePolicy) writes the flanks (3'a, 5'a, 3'b, 5'b).
// (LAMMPS resolves them through tag->local map lookups; here bonds() reads.)
// -----------------------------------------------------------------------
inline void build_prime_xstk(const ParticleArrays &p, const NeighborList &nl) {
    const int np = nl.N_screened;
    if (nl.prime_xstk.extent_int(0) < np)
        nl.prime_xstk = Kokkos::View<int *[4], Kokkos::LayoutLeft>("prime_neighs_oxdna3_xstk", np);
    if (np <= 0) return;
    auto tab = nl.prime_xstk; auto sp = nl.screened_pair; auto bonds = p.bonds;
    Kokkos::View<const int *, Kokkos::MemoryTraits<Kokkos::RandomAccess>> map = p.map_array;
    Kokkos::parallel_for("oxdna3_prime_neighs_xstk", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(int e) {
        int a, braw;
        unpack_pair(sp(e), a, braw);
        const int b = braw & OX_NEIGHMASK;
        const LR_bonds ba = bonds(a), bb = bonds(b);
        tab(e, 0) = map_tag(map, ba.n3);
        tab(e, 1) = map_tag(map, ba.n5);
        tab(e, 2) = map_tag(map, bb.n3);
        tab(e, 3) = map_tag(map, bb.n5);
    });
}

// -----------------------------------------------------------------------
// Per-style launchers (each makes its own ScatterView, energy reduction only
// on energy steps).
// -----------------------------------------------------------------------
template <int MASK = dna3k::EXCV_BOTH>
inline c_acc run_excv3(ParticleArrays &p, const NeighborList &nl_all, const DNA3Params &par,
                       const SimBox &box, bool want_energy, bool lammps_overhead,
                       bool neigh_rebuilt) {
    const NeighborList &nl = nl_all.list_for(NeighborList::LIST_EXCV);   // trimmed excv list
    if (lammps_overhead && (neigh_rebuilt || nl.prime_pair.extent(0) == 0)) build_prime_pair(p, nl);
    if (lammps_overhead && !nl_has_ilist(p, nl)) {
        nl.d_ilist = Kokkos::View<int *>("neighlist:ilist", nl.d_num_neigh.extent(0));
        auto il = nl.d_ilist;
        Kokkos::parallel_for("neighlist_ilist_init", il.extent_int(0), KOKKOS_LAMBDA(int i) { il(i) = i; });
    }
    auto setup = [&](auto &f) {
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny;
        f.bonds = p.bonds; f.ptype = p.ptype; f.tag = p.tag; f.ilist = nl.d_ilist;
        f.num_neigh = nl.d_num_neigh; f.neigh_matrix = nl.d_neigh_matrix;
        if (lammps_overhead) f.prime_pair = nl.prime_pair;
        f.par = par; f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques); f.box = box;
    };
    if (lammps_overhead) {
        Excv3Functor<true, MASK> f; setup(f);
        return launch_term<OxdnaRangePolicy>("oxdna3_excv", p.N, f, want_energy);
    }
    Excv3Functor<false, MASK> f; setup(f);
    return launch_term<OxdnaRangePolicy>("oxdna3_excv", p.N, f, want_energy);
}

inline c_acc run_hbond3(ParticleArrays &p, const NeighborList &nl, const DNA3Params &par,
                        const SimBox &box, bool want_energy) {
    Hbond3Functor f;
    f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz; f.ptype = p.ptype;
    f.sp = nl.screened_pair; f.par = par; f.box = box;
    f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
    return launch_term<Kokkos::RangePolicy<>>("oxdna3_hbond", nl.N_screened, f, want_energy);
}

inline c_acc run_xstk3(ParticleArrays &p, const NeighborList &nl, const DNA3Params &par,
                       const SimBox &box, bool want_energy, bool lammps_overhead,
                       bool neigh_rebuilt) {
    using Plain = Kokkos::RangePolicy<>;
    auto setup = [&](auto &f) {
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz;
        f.ptype = p.ptype; f.bonds = p.bonds;
        f.sp = nl.screened_pair; f.par = par; f.box = box;
        f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
    };
    if (lammps_overhead) {
        if (neigh_rebuilt || nl.prime_xstk.extent_int(0) < nl.N_screened) build_prime_xstk(p, nl);
        Xstk3Functor<true, true> f; setup(f); f.prime = nl.prime_xstk;
        return launch_term<Plain>("oxdna3_xstk", nl.N_screened, f, want_energy);
    }
    Xstk3Functor<false, false> f; setup(f);
    return launch_term<Plain>("oxdna3_xstk", nl.N_screened, f, want_energy);
}

inline c_acc run_coaxstk3(ParticleArrays &p, const NeighborList &nl, const DNA3Params &par,
                          const SimBox &box, bool want_energy) {
    Coaxstk3Functor f;
    f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz;
    f.ptype = p.ptype; f.bonds = p.bonds;
    f.sp = nl.screened_pair; f.par = par; f.box = box;
    f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
    return launch_term<Kokkos::RangePolicy<>>("oxdna3_coaxstk", nl.N_screened, f, want_energy);
}

// stk (TERMS = STACKING) or fene (TERMS = BACKBONE)
template <int TERMS>
inline c_acc run_bonded3(ParticleArrays &p, const DNA3Params &par, const SimBox &box,
                         bool want_energy, bool lammps_overhead, const char *label) {
    using Plain = Kokkos::RangePolicy<>;
    c_acc etot = 0;
    if (lammps_overhead) {
        if (p.nbonds <= 0) return etot;
        Bonded3ScatterFunctor<TERMS> f;
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz;
        f.prime = p.prime_bond; f.ptype = p.ptype; f.flag = p.overstretch_flag;
        f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
        f.par = par; f.box = box;
        if (want_energy) Kokkos::parallel_reduce(label, Plain(0, p.nbonds), f, etot);
        else             Kokkos::parallel_for(label, Plain(0, p.nbonds), f);
        return etot;
    }
    Bonded3GatherFunctor<TERMS> f;
    f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz;
    f.bonds = p.bonds; f.ptype = p.ptype;
    f.forces = p.forces; f.torques = p.torques; f.par = par; f.box = box;
    if (want_energy) Kokkos::parallel_reduce(label, Plain(0, p.N), f, etot);
    else             Kokkos::parallel_for(label, Plain(0, p.N), f);
    return etot;
}

// bond oxdna3/fene + the overstretch-flag host round trip (energy steps only)
inline c_acc run_fene3(ParticleArrays &p, const DNA3Params &par, const SimBox &box,
                       bool want_energy, bool lammps_overhead) {
    c_acc e = run_bonded3<dna3::BACKBONE>(p, par, box, want_energy, lammps_overhead, "oxdna3_fene");
    if (lammps_overhead && want_energy) {
        Kokkos::deep_copy(p.overstretch_flag_host, p.overstretch_flag);
        if (p.overstretch_flag_host() == 1) Kokkos::deep_copy(p.overstretch_flag, 0);
    }
    return e;
}

// -----------------------------------------------------------------------
// Per-step drivers in LAMMPS order (Pair section, then Bond section).
// eterm (optional, validation): per-kernel energies indexed by
//   0 excv, 1 stk, 2 hbond, 3 xstk, 4 coaxstk, 5 dh, 6 fene.
// kmask (validation): subset of kernels to run (dna3k:: bits); the
// precomputes always run as in the MD loop.
// -----------------------------------------------------------------------
inline c_acc compute_pair_forces_step_dna3(ParticleArrays &p, const NeighborList &nl,
                                           const DNA3Model &m, const SimBox &box,
                                           bool want_energy, bool lammps_overhead,
                                           bool neigh_rebuilt, c_acc *eterm = nullptr,
                                           int kmask = dna3k::PAIR, bool run_lrf = true) {
    using namespace dna3k;
    if (run_lrf) compute_lrf(p, lammps_overhead);                         // fix OXDNA/LRF
    if (lammps_overhead) ensure_bondlist(p);
    c_acc et[6] = {0, 0, 0, 0, 0, 0};
    if (kmask & EXCV)
        et[0] = run_excv3(p, nl, m.p, box, want_energy, lammps_overhead, neigh_rebuilt);
    if (lammps_overhead && neigh_rebuilt)
        bond_precompute(p, "oxdna_stk_prime_neighs_bond");                // stk's own copy
    if (kmask & STK)
        et[1] = run_bonded3<dna3::STACKING>(p, m.p, box, want_energy, lammps_overhead, "oxdna3_stk");
    if (kmask & HBOND)
        et[2] = run_hbond3(p, nl, m.p, box, want_energy);
    if (kmask & XSTK)
        et[3] = run_xstk3(p, nl, m.p, box, want_energy, lammps_overhead, neigh_rebuilt);
    if (kmask & COAXSTK)
        et[4] = run_coaxstk3(p, nl, m.p, box, want_energy);
    if (kmask & DH)
        et[5] = run_dh(p, nl, m.dh, box, want_energy, lammps_overhead);
    c_acc e = 0;
    for (int t = 0; t < 6; t++) {
        e += et[t];
        if (eterm) eterm[t] = et[t];
    }
    return e;
}

inline c_acc compute_bond_forces_step_dna3(ParticleArrays &p, const DNA3Model &m,
                                           const SimBox &box, bool want_energy,
                                           bool lammps_overhead, c_acc *eterm = nullptr,
                                           bool neigh_rebuilt = false) {
    if (lammps_overhead) {
        ensure_bondlist(p);
        if (neigh_rebuilt) bond_precompute(p, "oxdna_fene_prime_neighs_bond");   // fene's own copy
    }
    const c_acc e = run_fene3(p, m.p, box, want_energy, lammps_overhead);
    if (eterm) eterm[6] = e;
    return e;
}

// Standalone force evaluation for the validation tools (fd_test / xcheck):
// zero forces, then the selected kernels as on a neighbor-rebuild step.
inline c_number compute_forces_dna3(ParticleArrays &p, const NeighborList &nl,
                                    const DNA3Model &m, const SimBox &box,
                                    bool lammps_overhead, int kmask = dna3k::ALL,
                                    c_acc *eterm = nullptr) {
    p.zero_forces();
    c_acc e = compute_pair_forces_step_dna3(p, nl, m, box, true, lammps_overhead, true, eterm, kmask);
    if (kmask & dna3k::FENE) e += compute_bond_forces_step_dna3(p, m, box, true, lammps_overhead, eterm, true);
    Kokkos::fence();
    return static_cast<c_number>(e);
}

// Excluded volume split into its nonbonded / bonded parts (validation only).
inline c_number compute_excv_part_dna3(ParticleArrays &p, const NeighborList &nl,
                                       const DNA3Model &m, const SimBox &box,
                                       bool lammps_overhead, bool bonded) {
    p.zero_forces();
    compute_lrf(p, lammps_overhead);
    c_acc e = bonded ? run_excv3<dna3k::EXCV_BONDED>(p, nl, m.p, box, true, lammps_overhead, true)
                     : run_excv3<dna3k::EXCV_NB>(p, nl, m.p, box, true, lammps_overhead, true);
    Kokkos::fence();
    return static_cast<c_number>(e);
}
