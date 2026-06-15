#pragma once

// Bonded backbone FENE spring.
// One Kokkos thread per particle. Each particle checks its 3' neighbour (n3).
// Newton's 3rd law: both particle i and n3 get equal and opposite forces.
// Uses ScatterView (atomics on GPU) for force/torque accumulation.

#include "../types.h"
#include "../particles.h"
#include "params.h"
#include "orient.h"
#include "mf_oxdna.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

// Helper
KOKKOS_INLINE_FUNCTION
void cross3b(const c_number a[3], const c_number b[3], c_number c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

// One site-site bonded excluded-volume term (F3). a = particle i, b = n3.
// d = delr_com + rb_site - ra_site (a->b). Force on a is -df, on b is +df.
KOKKOS_INLINE_FUNCTION
void bonded_excv_term(const c_number ra[3], const c_number rb[3], const c_number d[3],
                      const ExcvParams &ep,
                      c_number (&fi)[3], c_number (&ti)[3],
                      c_number (&fn)[3], c_number (&tn)[3], c_number &e) {
    c_number rsq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    if (rsq >= ep.cutsq_c) return;
    c_number fpair = 0;
    e += MFOxdna::F3(rsq, ep.cutsq_ast, ep.cut_c, ep.lj1, ep.lj2, ep.eps, ep.b, fpair);
    c_number df[3] = {d[0]*fpair, d[1]*fpair, d[2]*fpair};
    fi[0]-=df[0]; fi[1]-=df[1]; fi[2]-=df[2];
    fn[0]+=df[0]; fn[1]+=df[1]; fn[2]+=df[2];
    c_number c[3];
    cross3b(ra, df, c); ti[0]-=c[0]; ti[1]-=c[1]; ti[2]-=c[2];
    cross3b(rb, df, c); tn[0]+=c[0]; tn[1]+=c[1]; tn[2]+=c[2];
}

// FENE potential: U = -k*Delta^2/2 * ln(1 - ((r-r0)/Delta)^2)
// Valid only when |r - r0| < Delta.
struct BackboneFunctor {
    Kokkos::View<const c_number *[4]> poss;
    Kokkos::View<const c_number *[4]> orientations;
    Kokkos::View<const LR_bonds *>    bonds;
    SimBox box;
    FeneParams fene;
    c_number pb1, pb2;  // backbone site = pb1*a1 + pb2*a2 (grooved for oxDNA2)
    c_number d_cbs;     // base site offset along nx
    ExcvParams excv_bsbs;  // bonded base-base excluded volume (EXCL_S2)
    ExcvParams excv_bkbs;  // bonded back-base excluded volume (EXCL_S3/S4)

    using SV = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::DefaultExecutionSpace::array_layout,
        Kokkos::DefaultExecutionSpace,
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>;

    SV sf, st;

    KOKKOS_INLINE_FUNCTION
    void operator()(int i, c_number &ev) const {
        int n3 = bonds(i).n3;
        if (n3 < 0) return;

        // Orientation of i and n3
        c_number ax_i[3], ay_i[3], az_i[3];
        c_number ax_n[3], ay_n[3], az_n[3];
        get_vectors_from_quat_view(orientations, i,  ax_i, ay_i, az_i);
        get_vectors_from_quat_view(orientations, n3, ax_n, ay_n, az_n);

        // Backbone site vectors (grooved for oxDNA2: pb1*a1 + pb2*a2)
        c_number ra[3] = {pb1*ax_i[0]+pb2*ay_i[0], pb1*ax_i[1]+pb2*ay_i[1], pb1*ax_i[2]+pb2*ay_i[2]};
        c_number rb[3] = {pb1*ax_n[0]+pb2*ay_n[0], pb1*ax_n[1]+pb2*ay_n[1], pb1*ax_n[2]+pb2*ay_n[2]};

        // Vector between backbone sites: n3 → i
        c_number dx = poss(i,0) - poss(n3,0) + ra[0] - rb[0];
        c_number dy = poss(i,1) - poss(n3,1) + ra[1] - rb[1];
        c_number dz = poss(i,2) - poss(n3,2) + ra[2] - rb[2];
        box.wrap(dx, dy, dz);

        c_number rsq = dx*dx + dy*dy + dz*dz;
        c_number r   = Kokkos::sqrt(rsq);
        c_number r0  = fene.r0;
        c_number Delta = fene.Delta;
        c_number k   = fene.k;

        c_number dr = r - r0;
        c_number t  = dr / Delta;
        if (Kokkos::fabs(t) >= c_number(1.0)) return;  // bond completely broken

        // Cap force at max_backbone_force=20000 (same as standalone oxDNA).
        // Beyond t_mbf, force magnitude would exceed ~20000 reduced units.
        // mbf_xmax = 0.24995 → t_mbf = 0.24995/0.25 = 0.9998
        static const c_number t_mbf = c_number(0.9998);
        if (t >  t_mbf) t =  t_mbf;
        if (t < -t_mbf) t = -t_mbf;

        c_number denom = 1 - t * t;
        c_number U     = c_number(-0.5) * k * Kokkos::log(denom);
        c_number fpair = -k * t / (r * Delta * denom);

        ev += U;

        c_number delf[3] = {dx*fpair, dy*fpair, dz*fpair};

        auto af = sf.access();
        auto at_v = st.access();
        af(i,  0) += delf[0]; af(i,  1) += delf[1]; af(i,  2) += delf[2];
        af(n3, 0) -= delf[0]; af(n3, 1) -= delf[1]; af(n3, 2) -= delf[2];

        c_number da[3], db[3];
        cross3b(ra, delf, da);
        cross3b(rb, delf, db);
        at_v(i,  0) += da[0]; at_v(i,  1) += da[1]; at_v(i,  2) += da[2];
        at_v(n3, 0) -= db[0]; at_v(n3, 1) -= db[1]; at_v(n3, 2) -= db[2];

        // ---- Bonded excluded volume (base-base, base-back, back-base) ----
        // a = i, b = n3. Backbone sites already as ra/rb; base sites along nx.
        c_number ra_bs[3] = {d_cbs*ax_i[0], d_cbs*ax_i[1], d_cbs*ax_i[2]};
        c_number rb_bs[3] = {d_cbs*ax_n[0], d_cbs*ax_n[1], d_cbs*ax_n[2]};

        c_number rcom[3] = {poss(n3,0)-poss(i,0), poss(n3,1)-poss(i,1), poss(n3,2)-poss(i,2)};
        box.wrap(rcom[0], rcom[1], rcom[2]);

        c_number fi[3] = {0,0,0}, ti[3] = {0,0,0}, fn[3] = {0,0,0}, tn[3] = {0,0,0};
        c_number eb = 0;

        // base(i)-base(n3)
        { c_number d[3] = {rcom[0]+rb_bs[0]-ra_bs[0], rcom[1]+rb_bs[1]-ra_bs[1], rcom[2]+rb_bs[2]-ra_bs[2]};
          bonded_excv_term(ra_bs, rb_bs, d, excv_bsbs, fi, ti, fn, tn, eb); }
        // base(i)-back(n3)
        { c_number d[3] = {rcom[0]+rb[0]-ra_bs[0], rcom[1]+rb[1]-ra_bs[1], rcom[2]+rb[2]-ra_bs[2]};
          bonded_excv_term(ra_bs, rb, d, excv_bkbs, fi, ti, fn, tn, eb); }
        // back(i)-base(n3)
        { c_number d[3] = {rcom[0]+rb_bs[0]-ra[0], rcom[1]+rb_bs[1]-ra[1], rcom[2]+rb_bs[2]-ra[2]};
          bonded_excv_term(ra, rb_bs, d, excv_bkbs, fi, ti, fn, tn, eb); }

        ev += eb;
        af(i,  0) += fi[0]; af(i,  1) += fi[1]; af(i,  2) += fi[2];
        af(n3, 0) += fn[0]; af(n3, 1) += fn[1]; af(n3, 2) += fn[2];
        at_v(i,  0) += ti[0]; at_v(i,  1) += ti[1]; at_v(i,  2) += ti[2];
        at_v(n3, 0) += tn[0]; at_v(n3, 1) += tn[1]; at_v(n3, 2) += tn[2];
    }
};

inline c_number compute_backbone_forces(ParticleArrays &p,
                                        const DNAParams &par,
                                        const SimBox &box) {
    using SV = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::DefaultExecutionSpace::array_layout,
        Kokkos::DefaultExecutionSpace,
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>;

    SV sf(p.forces);
    SV st(p.torques);

    BackboneFunctor fun;
    fun.poss         = p.poss;
    fun.orientations = p.orientations;
    fun.bonds        = p.bonds;
    fun.box          = box;
    fun.fene         = par.fene;
    fun.pb1          = par.pb1;
    fun.pb2          = par.pb2;
    fun.d_cbs        = par.d_cbs;
    fun.excv_bsbs    = par.excv_bsbs;
    fun.excv_bkbs    = par.excv_bkbs;
    fun.sf           = sf;
    fun.st           = st;

    c_number etot = 0;
    Kokkos::parallel_reduce("backbone_forces",
        Kokkos::RangePolicy<>(0, p.N), fun, etot);

    Kokkos::Experimental::contribute(p.forces, sf);
    Kokkos::Experimental::contribute(p.torques, st);
    return etot;
}
