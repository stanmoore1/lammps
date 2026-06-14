#pragma once

// DNA nonbonded force computation kernel.
// One Kokkos thread per edge (pair i,j) — no inner loop, perfect load balance.
// Force accumulation uses Kokkos::ScatterView (atomic on GPU, duplicated on OpenMP).
//
// Physics: excluded volume + hydrogen bonding + stacking + cross-stacking.
// Only oxDNA1 is implemented; oxDNA2/oxDNA3 can be added via template flags.

#include "../types.h"
#include "../particles.h"
#include "../neighbor_list.h"
#include "orient.h"
#include "mf_oxdna.h"
#include "params.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <cmath>

using namespace MFOxdna;

// -----------------------------------------------------------------------
// Cross product: c = a × b
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void cross3(const c_number a[3], const c_number b[3], c_number c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

KOKKOS_INLINE_FUNCTION
c_number dot3(const c_number a[3], const c_number b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// -----------------------------------------------------------------------
// Force/torque accumulation for one site-site term (F3 excluded volume).
// ra_site = vector from COM of particle a to interaction site of a.
// delf    = accumulated force on a (pointing from b→a).
// delta   = accumulated torque on a.
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void add_excv_contrib(const c_number ra_site[3], const c_number rb_site[3],
                      const c_number delr_site[3], const ExcvParams &ep,
                      c_number rsq, bool bonded_exclusion,
                      c_number (&delf)[3], c_number (&delta)[3],
                      c_number (&delf_b)[3], c_number (&delta_b)[3],
                      c_number &evdwl) {
    if (bonded_exclusion) return;
    if (rsq >= ep.cutsq_c) return;

    c_number fpair = 0;
    c_number U = F3(rsq, ep.cutsq_ast, ep.cut_c, ep.lj1, ep.lj2, ep.eps, ep.b, fpair);
    evdwl += U;

    c_number df[3] = {delr_site[0]*fpair, delr_site[1]*fpair, delr_site[2]*fpair};
    delf[0] += df[0]; delf[1] += df[1]; delf[2] += df[2];
    delf_b[0] -= df[0]; delf_b[1] -= df[1]; delf_b[2] -= df[2];

    c_number d[3], db[3];
    cross3(ra_site, df, d);
    delta[0] += d[0]; delta[1] += d[1]; delta[2] += d[2];
    cross3(rb_site, df, db);
    delta_b[0] -= db[0]; delta_b[1] -= db[1]; delta_b[2] -= db[2];
}

// -----------------------------------------------------------------------
// Safe acos: clamp to [-1,1] and handle near-zero sin(theta) gracefully.
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
c_number safe_acos(c_number x) {
    if (x >  1) x =  1;
    if (x < -1) x = -1;
    return Kokkos::acos(x);
}

// -----------------------------------------------------------------------
// H-bonding interaction for one pair (i,j).
// Returns energy and accumulates delf_a, delta_a (force on a), and
// delf_b, delta_b (force on b).
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
c_number hbond_pair(const c_number ra_bs[3], const c_number rb_bs[3],
                    const c_number delr_bs[3],  // b_site → a_site
                    c_number rinv_bs,
                    const DNAParams &par, int atype, int btype,
                    const c_number ax[3], const c_number az[3],
                    const c_number bx[3], const c_number bz[3],
                    c_number (&delf_a)[3], c_number (&delta_a)[3],
                    c_number (&delf_b)[3], c_number (&delta_b)[3]) {
    c_number r_bs = 1 / rinv_bs;
    const F1Params &fp = par.hb_f1;

    c_number f1 = F1(r_bs, fp.eps, fp.a, fp.cut_0,
                     fp.cut_lc, fp.cut_hc, fp.cut_lo, fp.cut_hi,
                     fp.b_lo, fp.b_hi, fp.shift);
    if (f1 == 0) return 0;

    // theta1: angle between -ax and bx (DNA base-base angle)
    c_number cost1 = -dot3(ax, bx);
    c_number theta1 = safe_acos(cost1);
    c_number f4t1 = F4(theta1, par.hb_t1.a, par.hb_t1.theta_0,
                       par.hb_t1.dtheta_ast, par.hb_t1.b, par.hb_t1.dtheta_c);
    if (f4t1 == 0) return 0;

    c_number norm[3] = {delr_bs[0]*rinv_bs, delr_bs[1]*rinv_bs, delr_bs[2]*rinv_bs};

    // theta2: angle between -ax and delr_bs
    c_number cost2 = -dot3(ax, norm);
    c_number theta2 = safe_acos(cost2);
    c_number f4t2 = F4(theta2, par.hb_t2.a, par.hb_t2.theta_0,
                       par.hb_t2.dtheta_ast, par.hb_t2.b, par.hb_t2.dtheta_c);
    if (f4t2 == 0) return 0;

    // theta3: angle between bx and delr_bs
    c_number cost3 = dot3(bx, norm);
    c_number theta3 = safe_acos(cost3);
    c_number f4t3 = F4(theta3, par.hb_t3.a, par.hb_t3.theta_0,
                       par.hb_t3.dtheta_ast, par.hb_t3.b, par.hb_t3.dtheta_c);
    if (f4t3 == 0) return 0;

    // theta4: angle between az and bz
    c_number cost4 = dot3(az, bz);
    c_number theta4 = safe_acos(cost4);
    c_number f4t4 = F4(theta4, par.hb_t4.a, par.hb_t4.theta_0,
                       par.hb_t4.dtheta_ast, par.hb_t4.b, par.hb_t4.dtheta_c);
    if (f4t4 == 0) return 0;

    // theta7: angle between -az and delr_bs
    c_number cost7 = -dot3(az, norm);
    c_number theta7 = safe_acos(cost7);
    c_number f4t7 = F4(theta7, par.hb_t7.a, par.hb_t7.theta_0,
                       par.hb_t7.dtheta_ast, par.hb_t7.b, par.hb_t7.dtheta_c);
    if (f4t7 == 0) return 0;

    // theta8: angle between bz and delr_bs
    c_number cost8 = dot3(bz, norm);
    c_number theta8 = safe_acos(cost8);
    c_number f4t8 = F4(theta8, par.hb_t8.a, par.hb_t8.theta_0,
                       par.hb_t8.dtheta_ast, par.hb_t8.b, par.hb_t8.dtheta_c);

    c_number alpha = par.alpha_hb[atype][btype];
    c_number evdwl = f1 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * alpha;
    if (evdwl == 0) return 0;

    c_number df1 = DF1(r_bs, fp.eps, fp.a, fp.cut_0,
                       fp.cut_lc, fp.cut_hc, fp.cut_lo, fp.cut_hi,
                       fp.b_lo, fp.b_hi);

    c_number sint1 = Kokkos::sqrt(Kokkos::fmax(c_number(0), 1 - cost1*cost1));
    c_number sint2 = Kokkos::sqrt(Kokkos::fmax(c_number(0), 1 - cost2*cost2));
    c_number sint3 = Kokkos::sqrt(Kokkos::fmax(c_number(0), 1 - cost3*cost3));
    c_number sint4 = Kokkos::sqrt(Kokkos::fmax(c_number(0), 1 - cost4*cost4));
    c_number sint7 = Kokkos::sqrt(Kokkos::fmax(c_number(0), 1 - cost7*cost7));
    c_number sint8 = Kokkos::sqrt(Kokkos::fmax(c_number(0), 1 - cost8*cost8));

    c_number df4t1 = (sint1 > 1e-8) ? DF4(theta1, par.hb_t1.a, par.hb_t1.theta_0,
        par.hb_t1.dtheta_ast, par.hb_t1.b, par.hb_t1.dtheta_c) / sint1 : 0;
    c_number df4t2 = (sint2 > 1e-8) ? DF4(theta2, par.hb_t2.a, par.hb_t2.theta_0,
        par.hb_t2.dtheta_ast, par.hb_t2.b, par.hb_t2.dtheta_c) / sint2 : 0;
    c_number df4t3 = (sint3 > 1e-8) ? DF4(theta3, par.hb_t3.a, par.hb_t3.theta_0,
        par.hb_t3.dtheta_ast, par.hb_t3.b, par.hb_t3.dtheta_c) / sint3 : 0;
    c_number df4t4 = (sint4 > 1e-8) ? DF4(theta4, par.hb_t4.a, par.hb_t4.theta_0,
        par.hb_t4.dtheta_ast, par.hb_t4.b, par.hb_t4.dtheta_c) / sint4 : 0;
    c_number df4t7 = (sint7 > 1e-8) ? DF4(theta7, par.hb_t7.a, par.hb_t7.theta_0,
        par.hb_t7.dtheta_ast, par.hb_t7.b, par.hb_t7.dtheta_c) / sint7 : 0;
    c_number df4t8 = (sint8 > 1e-8) ? DF4(theta8, par.hb_t8.a, par.hb_t8.theta_0,
        par.hb_t8.dtheta_ast, par.hb_t8.b, par.hb_t8.dtheta_c) / sint8 : 0;

    c_number prod_rest = f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * alpha;
    c_number delf[3] = {0, 0, 0};

    // radial force
    c_number finc = -df1 * prod_rest;
    delf[0] += delr_bs[0] * finc;
    delf[1] += delr_bs[1] * finc;
    delf[2] += delr_bs[2] * finc;

    // theta2 force (depends on ax and norm)
    if (theta2 != 0) {
        finc = -f1 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * alpha * rinv_bs;
        delf[0] += (norm[0]*cost2 + ax[0]) * finc;
        delf[1] += (norm[1]*cost2 + ax[1]) * finc;
        delf[2] += (norm[2]*cost2 + ax[2]) * finc;
    }
    // theta3 force (depends on bx and norm)
    if (theta3 != 0) {
        finc = -f1 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * alpha * rinv_bs;
        delf[0] += (norm[0]*cost3 - bx[0]) * finc;
        delf[1] += (norm[1]*cost3 - bx[1]) * finc;
        delf[2] += (norm[2]*cost3 - bx[2]) * finc;
    }
    // theta7 force
    if (theta7 != 0) {
        finc = -f1 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * alpha * rinv_bs;
        delf[0] += (norm[0]*cost7 + az[0]) * finc;
        delf[1] += (norm[1]*cost7 + az[1]) * finc;
        delf[2] += (norm[2]*cost7 + az[2]) * finc;
    }
    // theta8 force
    if (theta8 != 0) {
        finc = -f1 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * alpha * rinv_bs;
        delf[0] += (norm[0]*cost8 - bz[0]) * finc;
        delf[1] += (norm[1]*cost8 - bz[1]) * finc;
        delf[2] += (norm[2]*cost8 - bz[2]) * finc;
    }

    delf_a[0] += delf[0]; delf_a[1] += delf[1]; delf_a[2] += delf[2];
    delf_b[0] -= delf[0]; delf_b[1] -= delf[1]; delf_b[2] -= delf[2];

    c_number d_a[3], d_b[3];
    cross3(ra_bs, delf, d_a);
    delta_a[0] += d_a[0]; delta_a[1] += d_a[1]; delta_a[2] += d_a[2];
    cross3(rb_bs, delf, d_b);
    delta_b[0] -= d_b[0]; delta_b[1] -= d_b[1]; delta_b[2] -= d_b[2];

    // pure torques (theta1, theta4 — no positional component)
    if (theta1 != 0) {
        c_number ax_x_bx[3];
        cross3(ax, bx, ax_x_bx);
        finc = -f1 * df4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * alpha;
        delta_a[0] -= ax_x_bx[0] * finc;
        delta_a[1] -= ax_x_bx[1] * finc;
        delta_a[2] -= ax_x_bx[2] * finc;
        delta_b[0] += ax_x_bx[0] * finc;
        delta_b[1] += ax_x_bx[1] * finc;
        delta_b[2] += ax_x_bx[2] * finc;
    }
    if (theta4 != 0) {
        c_number az_x_bz[3];
        cross3(az, bz, az_x_bz);
        finc = -f1 * f4t1 * f4t2 * f4t3 * df4t4 * f4t7 * f4t8 * alpha;
        delta_a[0] += az_x_bz[0] * finc;
        delta_a[1] += az_x_bz[1] * finc;
        delta_a[2] += az_x_bz[2] * finc;
        delta_b[0] -= az_x_bz[0] * finc;
        delta_b[1] -= az_x_bz[1] * finc;
        delta_b[2] -= az_x_bz[2] * finc;
    }

    return evdwl;
}

// -----------------------------------------------------------------------
// Main nonbonded force dispatch — runs one kernel per pair (flat edge list)
// -----------------------------------------------------------------------
struct DNAForcesFunctor {
    // Input views (read-only)
    Kokkos::View<const c_number *[4]> poss;
    Kokkos::View<const c_number *[4]> orientations;
    Kokkos::View<const int *>         btype;
    Kokkos::View<const int *>         edge_i;
    Kokkos::View<const int *>         edge_j;

    // Params (small struct; lives in constant cache on GPU when accessed uniformly)
    DNAParams par;

    // Force/torque accumulators (ScatterView handles atomics on GPU)
    using ScatterF = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::LayoutRight,
        Kokkos::DefaultExecutionSpace,
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>;

    ScatterF sf;  // forces
    ScatterF st;  // torques

    SimBox box;

    KOKKOS_INLINE_FUNCTION
    void operator()(int edge, c_number &ev) const {
        const int ia = edge_i(edge);
        const int ib = edge_j(edge);

        // Load positions with PBC
        c_number xai = poss(ia,0), yai = poss(ia,1), zai = poss(ia,2);
        c_number xbi = poss(ib,0), ybi = poss(ib,1), zbi = poss(ib,2);

        c_number dx = xbi - xai;
        c_number dy = ybi - yai;
        c_number dz = zbi - zai;
        box.wrap(dx, dy, dz);

        // Load quaternions and get orientation vectors
        c_number axa[3], aya[3], aza[3];
        c_number axb[3], ayb[3], azb[3];
        get_vectors_from_quat_view(orientations, ia, axa, aya, aza);
        get_vectors_from_quat_view(orientations, ib, axb, ayb, azb);

        // Site offsets
        c_number d_cbk = par.d_cbk;
        c_number d_cbs = par.d_cbs;

        // COM → backbone site for a and b
        c_number ra_cbk[3] = {d_cbk*axa[0], d_cbk*axa[1], d_cbk*axa[2]};
        c_number rb_cbk[3] = {d_cbk*axb[0], d_cbk*axb[1], d_cbk*axb[2]};

        // COM → base site for a and b
        c_number ra_cbs[3] = {d_cbs*axa[0], d_cbs*axa[1], d_cbs*axa[2]};
        c_number rb_cbs[3] = {d_cbs*axb[0], d_cbs*axb[1], d_cbs*axb[2]};

        // Force/torque accumulators (local, then one atomic update at the end)
        c_number delf_a[3] = {0,0,0}, delf_b[3] = {0,0,0};
        c_number delta_a[3]= {0,0,0}, delta_b[3]= {0,0,0};
        c_number evdwl = 0;

        // ---- Excluded volume ----
        // backbone-backbone
        {
            c_number drx = dx + rb_cbk[0] - ra_cbk[0];
            c_number dry = dy + rb_cbk[1] - ra_cbk[1];
            c_number drz = dz + rb_cbk[2] - ra_cbk[2];
            c_number rsq = drx*drx + dry*dry + drz*drz;
            c_number delr[3] = {drx, dry, drz};
            add_excv_contrib(ra_cbk, rb_cbk, delr, par.excv_bkbk,
                             rsq, false, delf_a, delta_a, delf_b, delta_b, evdwl);
        }
        // backbone-base (a backbone → b base)
        {
            c_number drx = dx + rb_cbs[0] - ra_cbk[0];
            c_number dry = dy + rb_cbs[1] - ra_cbk[1];
            c_number drz = dz + rb_cbs[2] - ra_cbk[2];
            c_number rsq = drx*drx + dry*dry + drz*drz;
            c_number delr[3] = {drx, dry, drz};
            add_excv_contrib(ra_cbk, rb_cbs, delr, par.excv_bkbs,
                             rsq, false, delf_a, delta_a, delf_b, delta_b, evdwl);
        }
        // base-backbone (a base → b backbone)
        {
            c_number drx = dx + rb_cbk[0] - ra_cbs[0];
            c_number dry = dy + rb_cbk[1] - ra_cbs[1];
            c_number drz = dz + rb_cbk[2] - ra_cbs[2];
            c_number rsq = drx*drx + dry*dry + drz*drz;
            c_number delr[3] = {drx, dry, drz};
            add_excv_contrib(ra_cbs, rb_cbk, delr, par.excv_bkbs,
                             rsq, false, delf_a, delta_a, delf_b, delta_b, evdwl);
        }
        // base-base
        {
            c_number drx = dx + rb_cbs[0] - ra_cbs[0];
            c_number dry = dy + rb_cbs[1] - ra_cbs[1];
            c_number drz = dz + rb_cbs[2] - ra_cbs[2];
            c_number rsq = drx*drx + dry*dry + drz*drz;
            c_number delr[3] = {drx, dry, drz};
            add_excv_contrib(ra_cbs, rb_cbs, delr, par.excv_bsbs,
                             rsq, false, delf_a, delta_a, delf_b, delta_b, evdwl);
        }

        // ---- Hydrogen bonding ----
        {
            // Displacement vector: b base site → a base site
            c_number drx = dx + rb_cbs[0] - ra_cbs[0];
            c_number dry = dy + rb_cbs[1] - ra_cbs[1];
            c_number drz = dz + rb_cbs[2] - ra_cbs[2];
            c_number rsq = drx*drx + dry*dry + drz*drz;
            c_number r   = Kokkos::sqrt(rsq);
            if (r > 0 && r < par.hb_f1.cut_hc + 1) {
                c_number delr[3] = {drx, dry, drz};
                int at = btype(ia), bt = btype(ib);
                evdwl += hbond_pair(ra_cbs, rb_cbs, delr, 1/r, par,
                                    at, bt, axa, aza, axb, azb,
                                    delf_a, delta_a, delf_b, delta_b);
            }
        }

        ev += evdwl;

        // Accumulate forces and torques atomically
        auto af = sf.access();
        auto at_v = st.access();
        af(ia, 0) += delf_a[0]; af(ia, 1) += delf_a[1]; af(ia, 2) += delf_a[2];
        af(ib, 0) += delf_b[0]; af(ib, 1) += delf_b[1]; af(ib, 2) += delf_b[2];
        at_v(ia, 0) += delta_a[0]; at_v(ia, 1) += delta_a[1]; at_v(ia, 2) += delta_a[2];
        at_v(ib, 0) += delta_b[0]; at_v(ib, 1) += delta_b[1]; at_v(ib, 2) += delta_b[2];
    }
};

// -----------------------------------------------------------------------
// Compute all nonbonded forces for all N_edges pairs
// -----------------------------------------------------------------------
inline c_number compute_nonbonded_forces(
    ParticleArrays &p,
    const NeighborList &nl,
    const DNAParams &par,
    const SimBox &box)
{
    if (nl.N_edges == 0) return 0;

    using SV = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::LayoutRight,
        Kokkos::DefaultExecutionSpace,
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>;

    SV sf(p.forces);
    SV st(p.torques);

    DNAForcesFunctor fun;
    fun.poss         = p.poss;
    fun.orientations = p.orientations;
    fun.btype        = p.btype;
    fun.edge_i       = nl.edge_i;
    fun.edge_j       = nl.edge_j;
    fun.par          = par;
    fun.sf           = sf;
    fun.st           = st;
    fun.box          = box;

    c_number etot = 0;
    Kokkos::parallel_reduce("dna_forces_nonbonded",
        Kokkos::RangePolicy<>(0, nl.N_edges), fun, etot);

    Kokkos::Experimental::contribute(p.forces, sf);
    Kokkos::Experimental::contribute(p.torques, st);

    return etot;
}
