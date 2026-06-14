#pragma once

// Bonded stacking interaction (oxDNA1).
// One Kokkos thread per particle. Each particle i (5' nucleotide) checks its
// 3' neighbour j = bonds(i).n3.  The interaction involves:
//   - F1 radial term on the stacking sites
//   - F4 angular terms (theta4, theta5, theta6)
//   - F5 dihedral terms (cosphi1, cosphi2) on the backbone sites
//
// Ported from LAMMPS src/CG-DNA/pair_oxdna_stk.cpp.
// Particle labelling: a = j (3' end), b = i (5' end).

#include "../types.h"
#include "../particles.h"
#include "params.h"
#include "orient.h"
#include "mf_oxdna.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

struct StackingFunctor {
    Kokkos::View<const c_number *[4]> poss;
    Kokkos::View<const c_number *[4]> orientations;
    Kokkos::View<const LR_bonds *>    bonds;
    SimBox box;
    F1Params  stk_f1;
    F4Params  stk_t4, stk_t5, stk_t6;
    F5Params  stk_cp1, stk_cp2;
    c_number  d_cstk;
    c_number  d_cbk;

    using SV = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::LayoutRight,
        Kokkos::DefaultExecutionSpace,
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>;

    SV sf, st;

    KOKKOS_INLINE_FUNCTION
    void operator()(int i, c_number &ev) const {
        int j = bonds(i).n3;   // j = 3' neighbour (a in LAMMPS)
        if (j < 0) return;
        // b = i (5'), a = j (3')

        c_number ax_a[3], ay_a[3], az_a[3];  // orientation of a=j
        c_number ax_b[3], ay_b[3], az_b[3];  // orientation of b=i
        get_vectors_from_quat_view(orientations, j, ax_a, ay_a, az_a);
        get_vectors_from_quat_view(orientations, i, ax_b, ay_b, az_b);

        // Stacking site vectors from COM (= d_cstk * ax)
        c_number ra_cstk[3] = {d_cstk*ax_a[0], d_cstk*ax_a[1], d_cstk*ax_a[2]};
        c_number rb_cstk[3] = {d_cstk*ax_b[0], d_cstk*ax_b[1], d_cstk*ax_b[2]};

        // Vector from a's stacking site to b's stacking site
        c_number delr_stk[3] = {
            poss(i,0) + rb_cstk[0] - poss(j,0) - ra_cstk[0],
            poss(i,1) + rb_cstk[1] - poss(j,1) - ra_cstk[1],
            poss(i,2) + rb_cstk[2] - poss(j,2) - ra_cstk[2]
        };
        box.wrap(delr_stk[0], delr_stk[1], delr_stk[2]);

        c_number rsq_stk = delr_stk[0]*delr_stk[0] + delr_stk[1]*delr_stk[1]
                         + delr_stk[2]*delr_stk[2];
        c_number r_stk   = Kokkos::sqrt(rsq_stk);
        c_number rinv    = 1 / r_stk;

        c_number n_stk[3] = {delr_stk[0]*rinv, delr_stk[1]*rinv, delr_stk[2]*rinv};

        c_number f1 = MFOxdna::F1(r_stk,
            stk_f1.eps, stk_f1.a, stk_f1.cut_0,
            stk_f1.cut_lc, stk_f1.cut_hc,
            stk_f1.cut_lo, stk_f1.cut_hi,
            stk_f1.b_lo, stk_f1.b_hi, stk_f1.shift);
        if (f1 == 0) return;

        // theta4 = acos(az_b · az_a)
        c_number cost4 = az_a[0]*az_b[0] + az_a[1]*az_b[1] + az_a[2]*az_b[2];
        if (cost4 >  1) cost4 =  1;
        if (cost4 < -1) cost4 = -1;
        c_number theta4 = Kokkos::acos(cost4);

        c_number f4t4 = MFOxdna::F4(theta4,
            stk_t4.a, stk_t4.theta_0, stk_t4.dtheta_ast, stk_t4.b, stk_t4.dtheta_c);
        if (f4t4 == 0) return;

        // theta5 = acos(n_stk · az_b)
        c_number cost5 = n_stk[0]*az_b[0] + n_stk[1]*az_b[1] + n_stk[2]*az_b[2];
        if (cost5 >  1) cost5 =  1;
        if (cost5 < -1) cost5 = -1;
        c_number theta5 = Kokkos::acos(cost5);

        c_number f4t5 = MFOxdna::F4(theta5,
            stk_t5.a, stk_t5.theta_0, stk_t5.dtheta_ast, stk_t5.b, stk_t5.dtheta_c);
        if (f4t5 == 0) return;

        // theta6 = acos(n_stk · az_a)
        c_number cost6 = n_stk[0]*az_a[0] + n_stk[1]*az_a[1] + n_stk[2]*az_a[2];
        if (cost6 >  1) cost6 =  1;
        if (cost6 < -1) cost6 = -1;
        c_number theta6 = Kokkos::acos(cost6);

        // Backbone site vectors from COM (= d_cbk * ax)
        c_number ra_cbk[3] = {d_cbk*ax_a[0], d_cbk*ax_a[1], d_cbk*ax_a[2]};
        c_number rb_cbk[3] = {d_cbk*ax_b[0], d_cbk*ax_b[1], d_cbk*ax_b[2]};

        // Vector from a's backbone site to b's backbone site
        c_number delr_bk[3] = {
            poss(i,0) + rb_cbk[0] - poss(j,0) - ra_cbk[0],
            poss(i,1) + rb_cbk[1] - poss(j,1) - ra_cbk[1],
            poss(i,2) + rb_cbk[2] - poss(j,2) - ra_cbk[2]
        };
        box.wrap(delr_bk[0], delr_bk[1], delr_bk[2]);

        c_number rsq_bk   = delr_bk[0]*delr_bk[0] + delr_bk[1]*delr_bk[1]
                          + delr_bk[2]*delr_bk[2];
        c_number r_bk     = Kokkos::sqrt(rsq_bk);
        c_number rinv_bk  = 1 / r_bk;
        c_number n_bk[3]  = {delr_bk[0]*rinv_bk, delr_bk[1]*rinv_bk, delr_bk[2]*rinv_bk};

        // cosphi1 = n_bk · ay_b,  cosphi2 = n_bk · ay_a
        c_number cosphi1 = n_bk[0]*ay_b[0] + n_bk[1]*ay_b[1] + n_bk[2]*ay_b[2];
        c_number cosphi2 = n_bk[0]*ay_a[0] + n_bk[1]*ay_a[1] + n_bk[2]*ay_a[2];
        if (cosphi1 >  1) cosphi1 =  1;
        if (cosphi1 < -1) cosphi1 = -1;
        if (cosphi2 >  1) cosphi2 =  1;
        if (cosphi2 < -1) cosphi2 = -1;

        c_number f4t6 = MFOxdna::F4(theta6,
            stk_t6.a, stk_t6.theta_0, stk_t6.dtheta_ast, stk_t6.b, stk_t6.dtheta_c);

        c_number f5c1 = MFOxdna::F5(-cosphi1,
            stk_cp1.a, stk_cp1.x_ast, stk_cp1.b, stk_cp1.x_c);
        c_number f5c2 = MFOxdna::F5(-cosphi2,
            stk_cp2.a, stk_cp2.x_ast, stk_cp2.b, stk_cp2.x_c);

        c_number energy = f1 * f4t4 * f4t5 * f4t6 * f5c1 * f5c2;
        if (energy == 0) return;

        ev += energy;

        // Derivatives
        c_number df1  = MFOxdna::DF1(r_stk,
            stk_f1.eps, stk_f1.a, stk_f1.cut_0,
            stk_f1.cut_lc, stk_f1.cut_hc,
            stk_f1.cut_lo, stk_f1.cut_hi,
            stk_f1.b_lo, stk_f1.b_hi);

        c_number sinT4 = Kokkos::sqrt(1 - cost4*cost4);
        c_number df4t4 = (sinT4 > 1e-12)
            ? MFOxdna::DF4(theta4,
                stk_t4.a, stk_t4.theta_0, stk_t4.dtheta_ast, stk_t4.b, stk_t4.dtheta_c)
              / sinT4
            : c_number(0);

        c_number sinT5 = Kokkos::sqrt(1 - cost5*cost5);
        c_number df4t5 = (sinT5 > 1e-12)
            ? MFOxdna::DF4(theta5,
                stk_t5.a, stk_t5.theta_0, stk_t5.dtheta_ast, stk_t5.b, stk_t5.dtheta_c)
              / sinT5
            : c_number(0);

        c_number sinT6 = Kokkos::sqrt(1 - cost6*cost6);
        c_number df4t6 = (sinT6 > 1e-12)
            ? MFOxdna::DF4(theta6,
                stk_t6.a, stk_t6.theta_0, stk_t6.dtheta_ast, stk_t6.b, stk_t6.dtheta_c)
              / sinT6
            : c_number(0);

        c_number df5c1 = MFOxdna::DF5(-cosphi1,
            stk_cp1.a, stk_cp1.x_ast, stk_cp1.b, stk_cp1.x_c);
        c_number df5c2 = MFOxdna::DF5(-cosphi2,
            stk_cp2.a, stk_cp2.x_ast, stk_cp2.b, stk_cp2.x_c);

        // ---- Stacking site forces ----
        c_number delf_s[3] = {0, 0, 0};

        // Radial force
        c_number finc = -df1 * f4t4 * f4t5 * f4t6 * f5c1 * f5c2;
        delf_s[0] += delr_stk[0] * finc;
        delf_s[1] += delr_stk[1] * finc;
        delf_s[2] += delr_stk[2] * finc;

        // theta5 force (from d(n_stk)/d(site_position))
        if (theta5 != 0) {
            finc = -f1 * f4t4 * df4t5 * f4t6 * f5c1 * f5c2 * rinv;
            delf_s[0] += (n_stk[0]*cost5 - az_b[0]) * finc;
            delf_s[1] += (n_stk[1]*cost5 - az_b[1]) * finc;
            delf_s[2] += (n_stk[2]*cost5 - az_b[2]) * finc;
        }

        // theta6 force
        if (theta6 != 0) {
            finc = -f1 * f4t4 * f4t5 * df4t6 * f5c1 * f5c2 * rinv;
            delf_s[0] += (n_stk[0]*cost6 - az_a[0]) * finc;
            delf_s[1] += (n_stk[1]*cost6 - az_a[1]) * finc;
            delf_s[2] += (n_stk[2]*cost6 - az_a[2]) * finc;
        }

        // ---- Backbone site forces ----
        c_number delf_b[3] = {0, 0, 0};

        // cosphi1 force
        if (cosphi1 != 0) {
            finc = -f1 * f4t4 * f4t5 * f4t6 * df5c1 * f5c2 * rinv_bk;
            delf_b[0] += (n_bk[0]*cosphi1 - ay_b[0]) * finc;
            delf_b[1] += (n_bk[1]*cosphi1 - ay_b[1]) * finc;
            delf_b[2] += (n_bk[2]*cosphi1 - ay_b[2]) * finc;
        }

        // cosphi2 force
        if (cosphi2 != 0) {
            finc = -f1 * f4t4 * f4t5 * f4t6 * f5c1 * df5c2 * rinv_bk;
            delf_b[0] += (n_bk[0]*cosphi2 - ay_a[0]) * finc;
            delf_b[1] += (n_bk[1]*cosphi2 - ay_a[1]) * finc;
            delf_b[2] += (n_bk[2]*cosphi2 - ay_a[2]) * finc;
        }

        // ---- Accumulate forces and site torques ----
        auto af  = sf.access();
        auto at_v = st.access();

        // Stacking site: f[j=a] -= delf_s, f[i=b] += delf_s
        af(j, 0) -= delf_s[0]; af(j, 1) -= delf_s[1]; af(j, 2) -= delf_s[2];
        af(i, 0) += delf_s[0]; af(i, 1) += delf_s[1]; af(i, 2) += delf_s[2];

        // Torques from stacking site forces: tau = r_site × F
        c_number delta_s[3], deltb_s[3];
        cross3b(ra_cstk, delf_s, delta_s);
        cross3b(rb_cstk, delf_s, deltb_s);
        at_v(j, 0) -= delta_s[0]; at_v(j, 1) -= delta_s[1]; at_v(j, 2) -= delta_s[2];
        at_v(i, 0) += deltb_s[0]; at_v(i, 1) += deltb_s[1]; at_v(i, 2) += deltb_s[2];

        // Backbone site: f[j=a] -= delf_b, f[i=b] += delf_b
        af(j, 0) -= delf_b[0]; af(j, 1) -= delf_b[1]; af(j, 2) -= delf_b[2];
        af(i, 0) += delf_b[0]; af(i, 1) += delf_b[1]; af(i, 2) += delf_b[2];

        // Torques from backbone site forces
        c_number delta_b[3], deltb_b[3];
        cross3b(ra_cbk, delf_b, delta_b);
        cross3b(rb_cbk, delf_b, deltb_b);
        at_v(j, 0) -= delta_b[0]; at_v(j, 1) -= delta_b[1]; at_v(j, 2) -= delta_b[2];
        at_v(i, 0) += deltb_b[0]; at_v(i, 1) += deltb_b[1]; at_v(i, 2) += deltb_b[2];

        // ---- Pure torques (not expressible as r × F) ----
        // delta accumulates onto torque[j=a] as: torque[j] -= delta
        // deltb accumulates onto torque[i=b] as: torque[i] += deltb
        c_number delta[3] = {0,0,0};
        c_number deltb[3] = {0,0,0};

        // theta4: t4dir = cross(az_a, az_b)
        if (theta4 != 0) {
            c_number tpair = -f1 * df4t4 * f4t5 * f4t6 * f5c1 * f5c2;
            c_number t4dir[3];
            cross3b(az_a, az_b, t4dir);
            delta[0] += t4dir[0]*tpair;  delta[1] += t4dir[1]*tpair;  delta[2] += t4dir[2]*tpair;
            deltb[0] += t4dir[0]*tpair;  deltb[1] += t4dir[1]*tpair;  deltb[2] += t4dir[2]*tpair;
        }

        // theta5: t5dir = cross(n_stk, az_b); deltb only
        if (theta5 != 0) {
            c_number tpair = -f1 * f4t4 * df4t5 * f4t6 * f5c1 * f5c2;
            c_number t5dir[3];
            cross3b(n_stk, az_b, t5dir);
            deltb[0] += t5dir[0]*tpair;  deltb[1] += t5dir[1]*tpair;  deltb[2] += t5dir[2]*tpair;
        }

        // theta6: t6dir = cross(n_stk, az_a); delta -= ... (so torque[j] += t6dir*tpair)
        if (theta6 != 0) {
            c_number tpair = -f1 * f4t4 * f4t5 * df4t6 * f5c1 * f5c2;
            c_number t6dir[3];
            cross3b(n_stk, az_a, t6dir);
            delta[0] -= t6dir[0]*tpair;  delta[1] -= t6dir[1]*tpair;  delta[2] -= t6dir[2]*tpair;
        }

        // cosphi1: cosphi1dir = cross(n_bk, ay_b); deltb only
        if (cosphi1 != 0) {
            c_number tpair = -f1 * f4t4 * f4t5 * f4t6 * df5c1 * f5c2;
            c_number cpdir[3];
            cross3b(n_bk, ay_b, cpdir);
            deltb[0] += cpdir[0]*tpair;  deltb[1] += cpdir[1]*tpair;  deltb[2] += cpdir[2]*tpair;
        }

        // cosphi2: cosphi2dir = cross(n_bk, ay_a); delta -= ... (so torque[j] += cosphi2dir*tpair)
        if (cosphi2 != 0) {
            c_number tpair = -f1 * f4t4 * f4t5 * f4t6 * f5c1 * df5c2;
            c_number cpdir[3];
            cross3b(n_bk, ay_a, cpdir);
            delta[0] -= cpdir[0]*tpair;  delta[1] -= cpdir[1]*tpair;  delta[2] -= cpdir[2]*tpair;
        }

        at_v(j, 0) -= delta[0]; at_v(j, 1) -= delta[1]; at_v(j, 2) -= delta[2];
        at_v(i, 0) += deltb[0]; at_v(i, 1) += deltb[1]; at_v(i, 2) += deltb[2];
    }
};

inline c_number compute_stacking_forces(ParticleArrays &p,
                                        const DNAParams &par,
                                        const SimBox &box) {
    using SV = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::LayoutRight,
        Kokkos::DefaultExecutionSpace,
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>;

    SV sf(p.forces);
    SV st(p.torques);

    StackingFunctor fun;
    fun.poss         = p.poss;
    fun.orientations = p.orientations;
    fun.bonds        = p.bonds;
    fun.box          = box;
    fun.stk_f1       = par.stk_f1;
    fun.stk_t4       = par.stk_t4;
    fun.stk_t5       = par.stk_t5;
    fun.stk_t6       = par.stk_t6;
    fun.stk_cp1      = par.stk_cp1;
    fun.stk_cp2      = par.stk_cp2;
    fun.d_cstk       = par.d_cstk;
    fun.d_cbk        = par.d_cbk;
    fun.sf           = sf;
    fun.st           = st;

    c_number etot = 0;
    Kokkos::parallel_reduce("stacking_forces",
        Kokkos::RangePolicy<>(0, p.N), fun, etot);

    Kokkos::Experimental::contribute(p.forces, sf);
    Kokkos::Experimental::contribute(p.torques, st);
    return etot;
}
