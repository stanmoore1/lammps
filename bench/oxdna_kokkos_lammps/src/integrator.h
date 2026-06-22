#pragma once

// Velocity Verlet integrator with quaternion orientation update.
//
// Each MD step:
//   first_step:  v += F*dt/2,  r += v*dt,  L += T*dt/2,  q = integrate_quat(L, q, dt)
//   [rebuild neighbor list, zero forces, compute forces]
//   second_step: v += F*dt/2,  L += T*dt/2
//
// The quaternion update follows the same algorithm as CUDA_MD.cuh in standalone
// oxDNA: convert L to rotation axis+angle, apply as quaternion multiplication.

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

// Quaternion product: q_out = q_a * q_b
// Convention: q = (w, x, y, z) stored as (.x=w, .y=x, .z=y, .w=z)
// We store as poss(i,0)=w, (1)=x, (2)=y, (3)=z
KOKKOS_INLINE_FUNCTION
void quat_multiply(c_number aw, c_number ax, c_number ay, c_number az,
                   c_number bw, c_number bx, c_number by, c_number bz,
                   c_number &rw, c_number &rx, c_number &ry, c_number &rz) {
    rw = aw*bw - ax*bx - ay*by - az*bz;
    rx = aw*bx + ax*bw + ay*bz - az*by;
    ry = aw*by - ax*bz + ay*bw + az*bx;
    rz = aw*bz + ax*by - ay*bx + az*bw;
}

// Update quaternion from angular momentum vector L (ox, oy, oz) over time dt.
// The unit inertia tensor I=1 means omega = L exactly.
// Rotation angle = |L|*dt, axis = L/|L|.
KOKKOS_INLINE_FUNCTION
void update_orientation(c_number &qw, c_number &qx, c_number &qy, c_number &qz,
                        c_number Lx, c_number Ly, c_number Lz,
                        c_number dt) {
    c_number norm = Kokkos::sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
    if (norm < c_number(1e-14)) return;

    c_number angle = norm * dt;
    c_number s = Kokkos::sin(angle * c_number(0.5));
    c_number c = Kokkos::cos(angle * c_number(0.5));

    c_number inv_norm = 1 / norm;
    c_number rw = c;
    c_number rx = Lx * inv_norm * s;
    c_number ry = Ly * inv_norm * s;
    c_number rz = Lz * inv_norm * s;

    // Torque and angular momentum are accumulated in the LAB frame (the force
    // kernels build lab-frame torques and the inertia is unit-isotropic), so the
    // finite rotation is applied by LEFT-multiplication: q(t+dt) = r ⊗ q(t).
    c_number nw, nx, ny, nz;
    quat_multiply(rw, rx, ry, rz, qw, qx, qy, qz, nw, nx, ny, nz);

    // Re-normalise for numerical stability
    c_number mag = Kokkos::sqrt(nw*nw + nx*nx + ny*ny + nz*nz);
    c_number imag = 1 / mag;
    qw = nw * imag;
    qx = nx * imag;
    qy = ny * imag;
    qz = nz * imag;
}

// -----------------------------------------------------------------------
// First half-step: v += F*dt/2, r += v*dt, L += T*dt/2, update orientation
// -----------------------------------------------------------------------
struct FirstStepFunctor {
    Vec4 poss;
    Vec4 vels;
    Vec4 Ls;
    Vec4c forces;
    Vec4c torques;
    Vec4 orientations;
    c_number dt;
    SimBox box;

    // Optional fused neighbor-list displacement check (set check_rebuild = true).
    // Avoids a separate full-N reduction every step: while updating positions we
    // flag a rebuild if a particle has moved more than skin/2 from its reference
    // position at the last list build. Mirrors oxDNA's _d_are_lists_old flag set
    // inside the first-step kernel.
    Vec4c list_poss;
    Kokkos::View<int *>               rebuild_flag;
    c_number skin_half_sq = 0;
    bool     check_rebuild = false;

    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const {
        c_number dt_half = dt * c_number(0.5);

        // Velocity half-step
        vels(i,0) += forces(i,0) * dt_half;
        vels(i,1) += forces(i,1) * dt_half;
        vels(i,2) += forces(i,2) * dt_half;

        // Position full step
        poss(i,0) += vels(i,0) * dt;
        poss(i,1) += vels(i,1) * dt;
        poss(i,2) += vels(i,2) * dt;

        // Apply periodic boundary (modular fold: handles any displacement size)
        poss(i,0) -= box.Lx * Kokkos::floor(poss(i,0) / box.Lx + c_number(0.5));
        poss(i,1) -= box.Ly * Kokkos::floor(poss(i,1) / box.Ly + c_number(0.5));
        poss(i,2) -= box.Lz * Kokkos::floor(poss(i,2) / box.Lz + c_number(0.5));

        // Fused Verlet-list rebuild check (displacement since last build)
        if (check_rebuild) {
            c_number ddx = poss(i,0) - list_poss(i,0);
            c_number ddy = poss(i,1) - list_poss(i,1);
            c_number ddz = poss(i,2) - list_poss(i,2);
            box.wrap(ddx, ddy, ddz);
            if (ddx*ddx + ddy*ddy + ddz*ddz > skin_half_sq)
                Kokkos::atomic_max(&rebuild_flag(0), 1);
        }

        // Angular momentum half-step
        Ls(i,0) += torques(i,0) * dt_half;
        Ls(i,1) += torques(i,1) * dt_half;
        Ls(i,2) += torques(i,2) * dt_half;

        // Orientation update
        c_number qw = orientations(i,0), qx = orientations(i,1),
                 qy = orientations(i,2), qz = orientations(i,3);
        update_orientation(qw, qx, qy, qz,
                           Ls(i,0), Ls(i,1), Ls(i,2), dt);
        orientations(i,0) = qw; orientations(i,1) = qx;
        orientations(i,2) = qy; orientations(i,3) = qz;
    }
};

// -----------------------------------------------------------------------
// Second half-step: v += F*dt/2, L += T*dt/2
// -----------------------------------------------------------------------
struct SecondStepFunctor {
    Vec4 vels;
    Vec4 Ls;
    Vec4c forces;
    Vec4c torques;
    c_number dt;

    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const {
        c_number dt_half = dt * c_number(0.5);
        vels(i,0) += forces(i,0) * dt_half;
        vels(i,1) += forces(i,1) * dt_half;
        vels(i,2) += forces(i,2) * dt_half;
        Ls(i,0) += torques(i,0) * dt_half;
        Ls(i,1) += torques(i,1) * dt_half;
        Ls(i,2) += torques(i,2) * dt_half;
    }
};

inline void first_step(ParticleArrays &p, c_number dt, const SimBox &box,
                       Vec4c list_poss = {},
                       Kokkos::View<int *> rebuild_flag = {},
                       c_number skin_half_sq = 0) {
    FirstStepFunctor f;
    f.poss         = p.poss;
    f.vels         = p.vels;
    f.Ls           = p.Ls;
    f.forces       = p.forces;
    f.torques      = p.torques;
    f.orientations = p.orientations;
    f.dt           = dt;
    f.box          = box;
    // Enable the fused rebuild check only when a reference position array and a
    // flag are supplied (default-constructed Views are empty).
    f.list_poss     = list_poss;
    f.rebuild_flag  = rebuild_flag;
    f.skin_half_sq  = skin_half_sq;
    f.check_rebuild = (list_poss.data() != nullptr && rebuild_flag.data() != nullptr);
    Kokkos::parallel_for("first_step", Kokkos::RangePolicy<>(0, p.N), f);
}

inline void second_step(ParticleArrays &p, c_number dt) {
    SecondStepFunctor f;
    f.vels    = p.vels;
    f.Ls      = p.Ls;
    f.forces  = p.forces;
    f.torques = p.torques;
    f.dt      = dt;
    Kokkos::parallel_for("second_step", Kokkos::RangePolicy<>(0, p.N), f);
}

// Compute kinetic energy (translational + rotational)
// Assumes unit mass and unit inertia tensor (as in oxDNA)
inline c_number kinetic_energy(const ParticleArrays &p) {
    auto vels = p.vels;
    auto Ls   = p.Ls;
    c_number ekin = 0;
    Kokkos::parallel_reduce("kinetic_energy", p.N,
        KOKKOS_LAMBDA(int i, c_number &e) {
            c_number vx = vels(i,0), vy = vels(i,1), vz = vels(i,2);
            c_number lx = Ls(i,0),   ly = Ls(i,1),   lz = Ls(i,2);
            e += 0.5 * (vx*vx + vy*vy + vz*vz)
               + 0.5 * (lx*lx + ly*ly + lz*lz);
        }, ekin);
    return ekin;
}
