#pragma once

// Velocity-Verlet integrator for rigid nucleotides, a port of the oxDNA CUDA
// kernels CUDA_MD.cuh (first_step / second_step, precision float or double)
// and CUDA_mixed.cuh (first_step_mixed / second_step_mixed, mixed precision).
//
// As upstream:
//   * angular momenta L and torques T are in the BODY frame (the force kernels
//     rotate the lab-frame torque into the body frame at their end, see
//     _vectors_transpose_c_number4_product), and the finite rotation over dt is
//     applied by RIGHT-multiplication, q(t+dt) = q(t) * R(L_body dt), without
//     re-normalising the quaternion;
//   * the time step and the squared Verlet skin are single-precision constants
//     (__constant__ float MD_dt, MD_sqr_verlet_skin) in every precision mode;
//   * positions are not folded back into the box;
//   * the first-step kernel also flags a Verlet-list rebuild (particle moved
//     by more than the skin since the last build, plain difference) by writing
//     1 into a host-pinned flag (_d_are_lists_old);
//   * second_step stores K = v^2/2 and L^2/2 in the .w components.
//
// Deliberate deviations (numerically, not structurally, relevant):
//   * in the float/double kernel the orientation update uses sqrt/fmax of the
//     working precision; upstream calls sqrtf/fmaxf there, i.e. rounds to float
//     even in the CUDA_DOUBLE build;
//   * |L| == 0 skips the rotation (upstream divides by zero -> NaN; oxDNA
//     refuses such initial configurations unless refresh_vel = true).

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

// quat_multiply (CUDA_lr_common.cuh); storage (w, x, y, z) = (0, 1, 2, 3)
template <class T>
KOKKOS_INLINE_FUNCTION void quat_multiply(T aw, T ax, T ay, T az, T bw, T bx, T by, T bz,
                                          T &rw, T &rx, T &ry, T &rz) {
    rw = aw * bw - ax * bx - ay * by - az * bz;
    rx = aw * bx + ax * bw + ay * bz - az * by;
    ry = aw * by - ax * bz + ay * bw + az * bx;
    rz = aw * bz + ax * by - ay * bx + az * bw;
}

// _get_updated_orientation (CUDA_MD.cuh / CUDA_mixed.cuh): L is the body-frame
// angular momentum (unit inertia), dt the float MD_dt promoted to T.
template <class T>
KOKKOS_INLINE_FUNCTION void get_updated_orientation(T Lx, T Ly, T Lz, T dt,
                                                    T &qw, T &qx, T &qy, T &qz) {
    const T norm = Kokkos::sqrt(Lx * Lx + Ly * Ly + Lz * Lz);
    if (!(norm > T(0))) return;
    Lx /= norm; Ly /= norm; Lz /= norm;
    const T sintheta = Kokkos::sin(dt * norm);
    const T costheta = Kokkos::cos(dt * norm);
    const T rw = T(0.5) * Kokkos::sqrt(Kokkos::fmax(T(0), T(2) + T(2) * costheta));
    const T winv = T(1) / rw;
    const T rx = T(0.5) * Lx * sintheta * winv;
    const T ry = T(0.5) * Ly * sintheta * winv;
    const T rz = T(0.5) * Lz * sintheta * winv;
    T nw, nx, ny, nz;
    quat_multiply(qw, qx, qy, qz, rw, rx, ry, rz, nw, nx, ny, nz);
    qw = nw; qx = nx; qy = ny; qz = nz;
}

// MD constants: __constant__ float MD_dt[1], MD_sqr_verlet_skin[1]
struct MDConstants {
    float dt = 0;               // MD_dt
    float sqr_verlet_skin = 0;  // MD_sqr_verlet_skin
};

// -----------------------------------------------------------------------
// first_step (CUDA_MD.cuh), float/double builds
// -----------------------------------------------------------------------
struct FirstStepFunctor {
    Vec4 poss, vels, Ls, orientations;
    Vec4c forces, torques, list_poss;
    PinnedFlag are_lists_old;
    MDConstants md;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        const c_number dt = md.dt;
        const c_number F0 = forces(i, 0), F1 = forces(i, 1), F2 = forces(i, 2);
        c_number r0 = poss(i, 0), r1 = poss(i, 1), r2 = poss(i, 2);
        c_number v0 = vels(i, 0), v1 = vels(i, 1), v2 = vels(i, 2);

        v0 += F0 * (dt * c_number(0.5f));
        v1 += F1 * (dt * c_number(0.5f));
        v2 += F2 * (dt * c_number(0.5f));

        r0 += v0 * dt;
        r1 += v1 * dt;
        r2 += v2 * dt;

        vels(i, 0) = v0; vels(i, 1) = v1; vels(i, 2) = v2;
        poss(i, 0) = r0; poss(i, 1) = r1; poss(i, 2) = r2;

        c_number L0 = Ls(i, 0), L1 = Ls(i, 1), L2 = Ls(i, 2);
        L0 += torques(i, 0) * (dt * c_number(0.5f));
        L1 += torques(i, 1) * (dt * c_number(0.5f));
        L2 += torques(i, 2) * (dt * c_number(0.5f));
        Ls(i, 0) = L0; Ls(i, 1) = L1; Ls(i, 2) = L2;

        c_number qw = orientations(i, 0), qx = orientations(i, 1),
                 qy = orientations(i, 2), qz = orientations(i, 3);
        get_updated_orientation<c_number>(L0, L1, L2, dt, qw, qx, qy, qz);
        orientations(i, 0) = qw; orientations(i, 1) = qx;
        orientations(i, 2) = qy; orientations(i, 3) = qz;

        // do verlet lists need to be updated? (quad_distance, no minimum image)
        const c_number d0 = list_poss(i, 0) - r0, d1 = list_poss(i, 1) - r1,
                       d2 = list_poss(i, 2) - r2;
        if (d0 * d0 + d1 * d1 + d2 * d2 > c_number(md.sqr_verlet_skin)) are_lists_old() = 1;
    }
};

// -----------------------------------------------------------------------
// first_step_mixed (CUDA_mixed.cuh): float forces/torques, double v, r, L, q;
// writes the float copies of r and q used by the force kernels.
// -----------------------------------------------------------------------
struct FirstStepMixedFunctor {
    Vec4 poss, orientations;              // float4 copies (written)
    Vec4m possd, orientationsd, velsd, Lsd;
    Vec4c forces, torques, list_poss;
    PinnedFlag are_lists_old;
    MDConstants md;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        const float dt = md.dt;
        const float F0 = forces(i, 0), F1 = forces(i, 1), F2 = forces(i, 2);
        m_number v0 = velsd(i, 0), v1 = velsd(i, 1), v2 = velsd(i, 2);
        // F.x * MD_dt[0] * 0.5f is evaluated in float, then added to the double
        v0 += F0 * dt * 0.5f;
        v1 += F1 * dt * 0.5f;
        v2 += F2 * dt * 0.5f;
        velsd(i, 0) = v0; velsd(i, 1) = v1; velsd(i, 2) = v2;

        m_number r0 = possd(i, 0), r1 = possd(i, 1), r2 = possd(i, 2);
        r0 += v0 * dt;
        r1 += v1 * dt;
        r2 += v2 * dt;
        possd(i, 0) = r0; possd(i, 1) = r1; possd(i, 2) = r2;

        const float rf0 = static_cast<float>(r0), rf1 = static_cast<float>(r1),
                    rf2 = static_cast<float>(r2);
        poss(i, 0) = rf0; poss(i, 1) = rf1; poss(i, 2) = rf2;
        poss(i, 3) = static_cast<float>(possd(i, 3));

        // any_rigid_body is true for nucleotides
        m_number L0 = Lsd(i, 0), L1 = Lsd(i, 1), L2 = Lsd(i, 2);
        L0 += torques(i, 0) * dt * 0.5f;
        L1 += torques(i, 1) * dt * 0.5f;
        L2 += torques(i, 2) * dt * 0.5f;
        Lsd(i, 0) = L0; Lsd(i, 1) = L1; Lsd(i, 2) = L2;

        m_number qw = orientationsd(i, 0), qx = orientationsd(i, 1),
                 qy = orientationsd(i, 2), qz = orientationsd(i, 3);
        get_updated_orientation<m_number>(L0, L1, L2, static_cast<m_number>(dt), qw, qx, qy, qz);
        orientationsd(i, 0) = qw; orientationsd(i, 1) = qx;
        orientationsd(i, 2) = qy; orientationsd(i, 3) = qz;
        orientations(i, 0) = static_cast<float>(qw); orientations(i, 1) = static_cast<float>(qx);
        orientations(i, 2) = static_cast<float>(qy); orientations(i, 3) = static_cast<float>(qz);

        const float d0 = list_poss(i, 0) - rf0, d1 = list_poss(i, 1) - rf1,
                    d2 = list_poss(i, 2) - rf2;
        if (d0 * d0 + d1 * d1 + d2 * d2 > md.sqr_verlet_skin) are_lists_old() = 1;
    }
};

// -----------------------------------------------------------------------
// second_step (CUDA_MD.cuh)
// -----------------------------------------------------------------------
struct SecondStepFunctor {
    Vec4 vels, Ls;
    Vec4c forces, torques;
    MDConstants md;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        const c_number dt = md.dt;
        c_number v0 = vels(i, 0), v1 = vels(i, 1), v2 = vels(i, 2);
        v0 += forces(i, 0) * dt * c_number(0.5f);
        v1 += forces(i, 1) * dt * c_number(0.5f);
        v2 += forces(i, 2) * dt * c_number(0.5f);
        vels(i, 0) = v0; vels(i, 1) = v1; vels(i, 2) = v2;
        vels(i, 3) = (v0 * v0 + v1 * v1 + v2 * v2) * c_number(0.5f);

        c_number L0 = Ls(i, 0), L1 = Ls(i, 1), L2 = Ls(i, 2);
        L0 += torques(i, 0) * dt * c_number(0.5f);
        L1 += torques(i, 1) * dt * c_number(0.5f);
        L2 += torques(i, 2) * dt * c_number(0.5f);
        Ls(i, 0) = L0; Ls(i, 1) = L1; Ls(i, 2) = L2;
        Ls(i, 3) = (L0 * L0 + L1 * L1 + L2 * L2) * c_number(0.5f);
    }
};

// -----------------------------------------------------------------------
// second_step_mixed (CUDA_mixed.cuh)
// -----------------------------------------------------------------------
struct SecondStepMixedFunctor {
    Vec4m velsd, Lsd;
    Vec4c forces, torques;
    MDConstants md;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        const float dt = md.dt;
        velsd(i, 0) += forces(i, 0) * dt * 0.5f;
        velsd(i, 1) += forces(i, 1) * dt * 0.5f;
        velsd(i, 2) += forces(i, 2) * dt * 0.5f;
        Lsd(i, 0) += torques(i, 0) * dt * 0.5f;
        Lsd(i, 1) += torques(i, 1) * dt * 0.5f;
        Lsd(i, 2) += torques(i, 2) * dt * 0.5f;
    }
};

// Launchers (particles kernel cfg: threads_per_block, ceil(N / tpb) blocks)
inline void first_step(ParticleArrays &p, const MDConstants &md, const Vec4 &list_poss,
                       const PinnedFlag &are_lists_old) {
    if constexpr (OXDNA_MIXED) {
        FirstStepMixedFunctor f;
        f.poss = p.poss; f.orientations = p.orientations;
        f.possd = p.possd; f.orientationsd = p.orientationsd; f.velsd = p.velsd; f.Lsd = p.Lsd;
        f.forces = p.forces; f.torques = p.torques; f.list_poss = list_poss;
        f.are_lists_old = are_lists_old; f.md = md;
        Kokkos::parallel_for("first_step_mixed", OxPolicy(0, p.N), f);
    } else {
        FirstStepFunctor f;
        f.poss = p.poss; f.vels = p.vels; f.Ls = p.Ls; f.orientations = p.orientations;
        f.forces = p.forces; f.torques = p.torques; f.list_poss = list_poss;
        f.are_lists_old = are_lists_old; f.md = md;
        Kokkos::parallel_for("first_step", OxPolicy(0, p.N), f);
    }
}

inline void second_step(ParticleArrays &p, const MDConstants &md) {
    if constexpr (OXDNA_MIXED) {
        SecondStepMixedFunctor f;
        f.velsd = p.velsd; f.Lsd = p.Lsd; f.forces = p.forces; f.torques = p.torques; f.md = md;
        Kokkos::parallel_for("second_step_mixed", OxPolicy(0, p.N), f);
    } else {
        SecondStepFunctor f;
        f.vels = p.vels; f.Ls = p.Ls; f.forces = p.forces; f.torques = p.torques; f.md = md;
        Kokkos::parallel_for("second_step", OxPolicy(0, p.N), f);
    }
}

// Kinetic energy (translational + rotational; unit mass and inertia) of the
// host copies, in double, like the upstream KineticEnergy observable that the
// energy output evaluates on the CPU after apply_simulation_data_changes().
inline double kinetic_energy_host(const ParticleArraysHost &h) {
    double K = 0;
    for (int i = 0; i < h.N; i++) {
        const double vx = h.vels(i, 0), vy = h.vels(i, 1), vz = h.vels(i, 2);
        const double lx = h.Ls(i, 0), ly = h.Ls(i, 1), lz = h.Ls(i, 2);
        K += 0.5 * (vx * vx + vy * vy + vz * vz) + 0.5 * (lx * lx + ly * ly + lz * lz);
    }
    return K;
}
