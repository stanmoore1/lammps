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
    VecA4c forces;
    VecA4c torques;
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
    // Fold positions into the box every step (lean / minimum-image path).
    // The LAMMPS ghost-atom path keeps them unwrapped between neighbor-list
    // rebuilds and folds them in domain->pbc() on rebuild steps only.
    bool     fold = true;

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
        if (fold) {
            poss(i,0) -= box.Lx * Kokkos::floor(poss(i,0) / box.Lx + c_number(0.5));
            poss(i,1) -= box.Ly * Kokkos::floor(poss(i,1) / box.Ly + c_number(0.5));
            poss(i,2) -= box.Lz * Kokkos::floor(poss(i,2) / box.Lz + c_number(0.5));
        }

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
    VecA4c forces;
    VecA4c torques;
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

// -----------------------------------------------------------------------
// LAMMPS VerletKokkos fuse_integrate: final_integrate() of the previous step
// and initial_integrate() of this one in ONE kernel over the local atoms
// (FixNVEAsphereKokkosFusedIntegrateFunctor), used when the previous step
// had no output and no thermostat refresh (fuse_check). Same per-atom
// operations in the same order as second_step() + first_step(), so the
// result is bitwise identical; no per-step position fold (ghost mode).
// -----------------------------------------------------------------------
struct FusedStepFunctor {
    SecondStepFunctor second;
    FirstStepFunctor  first;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const { second(i); first(i); }
};

inline void first_step(ParticleArrays &p, c_number dt, const SimBox &box,
                       Vec4c list_poss = {},
                       Kokkos::View<int *> rebuild_flag = {},
                       c_number skin_half_sq = 0, bool fold = true,
                       const char *label = "first_step") {
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
    f.fold = fold;
    Kokkos::parallel_for(label, Kokkos::RangePolicy<>(0, p.N), f);
}

inline void fused_step(ParticleArrays &p, c_number dt, const SimBox &box) {
    FusedStepFunctor f;
    f.second.vels = p.vels; f.second.Ls = p.Ls; f.second.forces = p.forces;
    f.second.torques = p.torques; f.second.dt = dt;
    f.first.poss = p.poss; f.first.vels = p.vels; f.first.Ls = p.Ls;
    f.first.forces = p.forces; f.first.torques = p.torques;
    f.first.orientations = p.orientations; f.first.dt = dt; f.first.box = box;
    f.first.check_rebuild = false; f.first.fold = false;
    Kokkos::parallel_for("fused_integrate", Kokkos::RangePolicy<>(0, p.N), f);
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

// =======================================================================
// Optional LAMMPS fix nve/asphere/kk integrator (lammps_integrator = 1,
// ghost-atom path only). Mirrors FixNVEAsphereKokkos (kk-fixes):
//   initial:  v += dtf/m f; x += dtv v; angm = angmom + dtf torque;
//             principal moments from the bonus shape and rmass
//             (0.2 m (s1^2 + s2^2), ...); omega from angm and q
//             (mq_to_omega); Richardson iteration of the quaternion
//             (MathExtraKokkos::richardson) through bonus(ellipsoid(i));
//             angmom = angm
//   final:    v += dtf/m f; angmom += dtf torque
//   fused:    final of the previous step + initial of this one (one kernel,
//             v += 2 dtf/m f as LAMMPS)
// This changes the dynamics slightly (Richardson instead of the exact
// rotation of the bench integrator, mass / inertia from lammps_mass /
// lammps_shape), so it is off by default. With the defaults (mass 1, shape
// radii sqrt(2.5): inertia 1) the bench's unit mass / unit isotropic inertia
// convention (kinetic energy, thermostat) still holds.
// =======================================================================
namespace mek {
KOKKOS_INLINE_FUNCTION void quat_to_mat(const c_number *q, c_number m[3][3]) {
    const c_number w2 = q[0]*q[0], i2 = q[1]*q[1], j2 = q[2]*q[2], k2 = q[3]*q[3];
    const c_number twoij = c_number(2)*q[1]*q[2], twoik = c_number(2)*q[1]*q[3];
    const c_number twojk = c_number(2)*q[2]*q[3], twoiw = c_number(2)*q[1]*q[0];
    const c_number twojw = c_number(2)*q[2]*q[0], twokw = c_number(2)*q[3]*q[0];
    m[0][0] = w2+i2-j2-k2; m[0][1] = twoij-twokw; m[0][2] = twojw+twoik;
    m[1][0] = twoij+twokw; m[1][1] = w2-i2+j2-k2; m[1][2] = twojk-twoiw;
    m[2][0] = twoik-twojw; m[2][1] = twojk+twoiw; m[2][2] = w2-i2-j2+k2;
}
KOKKOS_INLINE_FUNCTION void mq_to_omega(const c_number *m, const c_number *q, const c_number *mom, c_number *w) {
    c_number rot[3][3], wb[3];
    quat_to_mat(q, rot);
    wb[0] = Kokkos::fma(rot[1][0], m[1], Kokkos::fma(rot[0][0], m[0], rot[2][0]*m[2]));
    wb[1] = Kokkos::fma(rot[1][1], m[1], Kokkos::fma(rot[0][1], m[0], rot[2][1]*m[2]));
    wb[2] = Kokkos::fma(rot[1][2], m[1], Kokkos::fma(rot[0][2], m[0], rot[2][2]*m[2]));
    for (int d = 0; d < 3; d++) wb[d] = (mom[d] == c_number(0)) ? c_number(0) : wb[d] / mom[d];
    w[0] = Kokkos::fma(rot[0][1], wb[1], Kokkos::fma(rot[0][0], wb[0], rot[0][2]*wb[2]));
    w[1] = Kokkos::fma(rot[1][1], wb[1], Kokkos::fma(rot[1][0], wb[0], rot[1][2]*wb[2]));
    w[2] = Kokkos::fma(rot[2][1], wb[1], Kokkos::fma(rot[2][0], wb[0], rot[2][2]*wb[2]));
}
KOKKOS_INLINE_FUNCTION void vecquat(const c_number *a, const c_number *b, c_number *c) {
    c[0] = -Kokkos::fma(a[0], b[1], Kokkos::fma(a[1], b[2], a[2] * b[3]));
    c[1] = Kokkos::fma(b[0], a[0], Kokkos::fma(a[1], b[3], -a[2] * b[2]));
    c[2] = Kokkos::fma(b[0], a[1], Kokkos::fma(a[2], b[1], -a[0] * b[3]));
    c[3] = Kokkos::fma(b[0], a[2], Kokkos::fma(a[0], b[2], -a[1] * b[1]));
}
KOKKOS_INLINE_FUNCTION void qnormalize(c_number *q) {
    c_number sum = q[3] * q[3];
    sum = Kokkos::fma(q[2], q[2], sum);
    sum = Kokkos::fma(q[1], q[1], sum);
    sum = Kokkos::fma(q[0], q[0], sum);
    const c_number norm = Kokkos::rsqrt(sum);
    q[0] *= norm; q[1] *= norm; q[2] *= norm; q[3] *= norm;
}
KOKKOS_INLINE_FUNCTION void richardson(c_number *q, const c_number *m, c_number *w, const c_number *mom, c_number dtq) {
    c_number wq[4], qfull[4], qhalf[4];
    vecquat(w, q, wq);
    for (int k = 0; k < 4; k++) qfull[k] = Kokkos::fma(dtq, wq[k], q[k]);
    qnormalize(qfull);
    for (int k = 0; k < 4; k++) qhalf[k] = Kokkos::fma(c_number(0.5)*dtq, wq[k], q[k]);
    qnormalize(qhalf);
    mq_to_omega(m, qhalf, mom, w);
    vecquat(w, qhalf, wq);
    for (int k = 0; k < 4; k++) qhalf[k] = Kokkos::fma(c_number(0.5)*dtq, wq[k], qhalf[k]);
    qnormalize(qhalf);
    for (int k = 0; k < 4; k++) q[k] = Kokkos::fma(c_number(2), qhalf[k], -qfull[k]);
    qnormalize(q);
}
} // namespace mek

// MODE: 0 = initial_integrate, 1 = final_integrate, 2 = fused_integrate
template <int MODE>
struct NVEAsphereFunctor {
    Vec4 poss, vels, Ls, quat;
    VecA4c forces, torques;
    Kokkos::View<const int *> ellipsoid;
    Kokkos::View<const c_number *> rmass;
    Vec4c shape;
    c_number dtf, dtv;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        const c_number rm = rmass(i);
        if (MODE == 1) {
            const c_number dtfm = dtf / rm;
            for (int d = 0; d < 3; d++) vels(i,d) += dtfm * static_cast<c_number>(forces(i,d));
            for (int d = 0; d < 3; d++) Ls(i,d) += dtf * static_cast<c_number>(torques(i,d));
            return;
        }
        const c_number dtq = c_number(0.5) * dtv;
        const c_number dtfm = (MODE == 2 ? c_number(2) : c_number(1)) * dtf / rm;
        for (int d = 0; d < 3; d++) vels(i,d) += dtfm * static_cast<c_number>(forces(i,d));
        if (MODE == 2)
            for (int d = 0; d < 3; d++) Ls(i,d) += dtf * static_cast<c_number>(torques(i,d));
        for (int d = 0; d < 3; d++) poss(i,d) += dtv * vels(i,d);
        c_number angm[3];
        for (int d = 0; d < 3; d++) angm[d] = Kokkos::fma(dtf, static_cast<c_number>(torques(i,d)), Ls(i,d));
        const int e = ellipsoid(i);
        const c_number s0 = shape(e,0), s1 = shape(e,1), s2 = shape(e,2);
        const c_number inertia[3] = {c_number(0.2)*rm*(s1*s1 + s2*s2), c_number(0.2)*rm*(s0*s0 + s2*s2),
                                     c_number(0.2)*rm*(s0*s0 + s1*s1)};
        c_number q[4] = {quat(e,0), quat(e,1), quat(e,2), quat(e,3)}, omega[3];
        mek::mq_to_omega(angm, q, inertia, omega);
        mek::richardson(q, angm, omega, inertia, dtq);
        quat(e,0) = q[0]; quat(e,1) = q[1]; quat(e,2) = q[2]; quat(e,3) = q[3];
        for (int d = 0; d < 3; d++) Ls(i,d) = angm[d];
    }
};

template <int MODE>
inline void nve_asphere(ParticleArrays &p, c_number dt) {
    NVEAsphereFunctor<MODE> f;
    f.poss = p.poss; f.vels = p.vels; f.Ls = p.Ls; f.quat = p.orientations;
    f.forces = p.forces; f.torques = p.torques; f.ellipsoid = p.ellipsoid;
    f.rmass = p.rmass; f.shape = p.shape;
    f.dtv = dt; f.dtf = c_number(0.5) * dt;        // LAMMPS lj units: ftm2v = 1
    const char *name = MODE == 0 ? "FixNVEAsphereKokkosInitialIntegrateFunctor"
                     : MODE == 1 ? "FixNVEAsphereKokkosFinalIntegrateFunctor"
                                 : "FixNVEAsphereKokkosFusedIntegrateFunctor";
    Kokkos::parallel_for(name, Kokkos::RangePolicy<>(0, p.N), f);
}
