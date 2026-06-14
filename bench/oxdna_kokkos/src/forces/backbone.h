#pragma once

// Bonded backbone FENE spring.
// One Kokkos thread per particle. Each particle checks its 3' neighbour (n3).
// Newton's 3rd law: both particle i and n3 get equal and opposite forces.
// Uses ScatterView (atomics on GPU) for force/torque accumulation.

#include "../types.h"
#include "../particles.h"
#include "params.h"
#include "orient.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

// Helper
KOKKOS_INLINE_FUNCTION
void cross3b(const c_number a[3], const c_number b[3], c_number c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

// FENE potential: U = -k*Delta^2/2 * ln(1 - ((r-r0)/Delta)^2)
// Valid only when |r - r0| < Delta.
struct BackboneFunctor {
    Kokkos::View<const c_number *[4]> poss;
    Kokkos::View<const c_number *[4]> orientations;
    Kokkos::View<const LR_bonds *>    bonds;
    SimBox box;
    FeneParams fene;
    c_number d_cbk;  // backbone site offset along nx

    using SV = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::LayoutRight,
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

        // Backbone site vectors
        c_number ra[3] = {d_cbk*ax_i[0], d_cbk*ax_i[1], d_cbk*ax_i[2]};
        c_number rb[3] = {d_cbk*ax_n[0], d_cbk*ax_n[1], d_cbk*ax_n[2]};

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
        if (Kokkos::fabs(t) >= 1) return;  // outside valid FENE region

        c_number denom = 1 - t * t;
        c_number U     = -k * Delta * Delta * 0.5 * Kokkos::log(denom);
        c_number fpair = -k * dr / (r * denom);   // dU/dr * (1/r)

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
    }
};

inline c_number compute_backbone_forces(ParticleArrays &p,
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

    BackboneFunctor fun;
    fun.poss         = p.poss;
    fun.orientations = p.orientations;
    fun.bonds        = p.bonds;
    fun.box          = box;
    fun.fene         = par.fene;
    fun.d_cbk        = par.d_cbk;
    fun.sf           = sf;
    fun.st           = st;

    c_number etot = 0;
    Kokkos::parallel_reduce("backbone_forces",
        Kokkos::RangePolicy<>(0, p.N), fun, etot);

    Kokkos::Experimental::contribute(p.forces, sf);
    Kokkos::Experimental::contribute(p.torques, st);
    return etot;
}
