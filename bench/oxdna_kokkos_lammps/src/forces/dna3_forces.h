#pragma once

// oxDNA3 physics (sequence-dependent, tetramer tables).
//
// The functions in namespace dna3 are copied from the CUDA-faithful sibling
// bench/oxdna_kokkos/src/forces/dna3_forces.h (a port of upstream oxDNA
// c2c74cc0, src/CUDA/Interactions/CUDA_DNA3.cuh, verified against the
// standalone DNA3_nomesh interaction). The two benches are deliberately
// self-contained, so the physics is duplicated here, as it is for oxDNA1/2.
// Changes relative to the sibling copy (none alters the default physics):
//   * the coaxial-stacking theta4 term can get the LAMMPS-only mirrored lobe
//     (DNA3Params::cxst_t4_blunt, off by default; lammps_coaxstk_terminal);
//   * bonded_pair<TERMS>() evaluates one bond ONCE and returns the force on
//     the n5 end plus the torques on BOTH ends, with exactly the arithmetic of
//     bonded_part<true> (n5 side) and bonded_part<false> (n3 side). The
//     LAMMPS-faithful per-bond scatter kernels (lammps_overhead mode) need it;
//     the lean gather kernels use bonded_part<> itself, as the sibling does;
//   * the CUDA-structured functors/launchers are not copied: the
//     LAMMPS-structured kernels that call these functions are in
//     dna3_kernels.h.
//
// Sign conventions (as upstream):
//   bonded_part / bonded_pair: r = pos(n3) - pos(n5), where n5 is the particle
//     whose n3 neighbour is the other one ("n3" = the 3' end of the bond);
//   particle_particle_interaction: r = pos(q) - pos(p); F, T act on p; the
//     torque on q is -T + r x F.

#include "../types.h"
#include "orient.h"
#include "params_dna3.h"
#include <Kokkos_Core.hpp>

namespace dna3 {

// Term selection mask (upstream DNAInteraction/DNA2Interaction enum order).
enum : int {
    BACKBONE = 1, BONDED_EXCLUDED_VOLUME = 2, STACKING = 4,
    NONBONDED_EXCLUDED_VOLUME = 8, HYDROGEN_BONDING = 16, CROSS_STACKING = 32,
    COAXIAL_STACKING = 64, DEBYE_HUCKEL = 128,
    BONDED = BACKBONE | BONDED_EXCLUDED_VOLUME | STACKING,
    NONBONDED = NONBONDED_EXCLUDED_VOLUME | HYDROGEN_BONDING | CROSS_STACKING |
                COAXIAL_STACKING | DEBYE_HUCKEL,
    ALL = BONDED | NONBONDED
};

// ---------------------------------------------------------------------------
// Minimal 3-vector (the xyz part of CUDA's c_number4)
// ---------------------------------------------------------------------------
struct v3 {
    c_number x, y, z;
};
KOKKOS_INLINE_FUNCTION v3 operator+(const v3 &a, const v3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
KOKKOS_INLINE_FUNCTION v3 operator-(const v3 &a, const v3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
KOKKOS_INLINE_FUNCTION v3 operator-(const v3 &a) { return {-a.x, -a.y, -a.z}; }
KOKKOS_INLINE_FUNCTION v3 operator*(const v3 &a, c_number s) { return {a.x * s, a.y * s, a.z * s}; }
KOKKOS_INLINE_FUNCTION v3 operator*(c_number s, const v3 &a) { return {a.x * s, a.y * s, a.z * s}; }
KOKKOS_INLINE_FUNCTION v3 operator/(const v3 &a, c_number s) { return {a.x / s, a.y / s, a.z / s}; }
KOKKOS_INLINE_FUNCTION v3 &operator+=(v3 &a, const v3 &b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
KOKKOS_INLINE_FUNCTION v3 &operator-=(v3 &a, const v3 &b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
KOKKOS_INLINE_FUNCTION c_number dot(const v3 &a, const v3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
KOKKOS_INLINE_FUNCTION v3 cross(const v3 &a, const v3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
KOKKOS_INLINE_FUNCTION c_number vmod(const v3 &a) { return Kokkos::sqrt(dot(a, a)); }
KOKKOS_INLINE_FUNCTION c_number SQR(c_number x) { return x * x; }

// CUDA_LRACOS
KOKKOS_INLINE_FUNCTION c_number lracos(c_number x) {
    constexpr c_number PI = c_number(3.141592653589793);
    return (x >= c_number(1)) ? c_number(0) : (x <= c_number(-1)) ? PI : Kokkos::acos(x);
}

// CUDA stably_normalised (returns 0 for an exactly zero vector instead of NaN)
KOKKOS_INLINE_FUNCTION v3 stably_normalised(const v3 &v) {
    c_number mx = Kokkos::fmax(Kokkos::fmax(Kokkos::fabs(v.x), Kokkos::fabs(v.y)), Kokkos::fabs(v.z));
    if (!(mx > c_number(0))) return {0, 0, 0};
    v3 res = v / mx;
    c_number m = vmod(res);
    return (m > c_number(0)) ? res / m : res;
}

// Quaternion row -> body axes (a1, a2, a3)
template <class View>
KOKKOS_INLINE_FUNCTION void axes(const View &ori, int i, v3 &a1, v3 &a2, v3 &a3) {
    c_number x[3], y[3], z[3];
    get_vectors_from_quat_view(ori, i, x, y, z);
    a1 = {x[0], x[1], x[2]};
    a2 = {y[0], y[1], y[2]};
    a3 = {z[0], z[1], z[2]};
}

// ---------------------------------------------------------------------------
// Modulation functions with tetramer-table lookups (CUDA_DNA3.cuh)
//   tetramer (n3_2, n3_1, n5_1, n5_2)
// ---------------------------------------------------------------------------
using namespace dna3sd;

// _excluded_volume: returns energy, F = force on the "p" end of r (r = q - p)
KOKKOS_INLINE_FUNCTION
c_number excluded_volume(const v3 &r, v3 &F, c_number sigma, c_number rstar, c_number b,
                         c_number rc, c_number eps) {
    c_number rsqr = dot(r, r);
    F = {0, 0, 0};
    c_number e = 0;
    if (rsqr < SQR(rc)) {
        if (rsqr > SQR(rstar)) {
            c_number rmod = Kokkos::sqrt(rsqr);
            c_number rrc = rmod - rc;
            c_number fmod = 2 * eps * b * rrc / rmod;
            F = r * fmod;
            e = eps * b * SQR(rrc);
        } else {
            c_number lj_part = SQR(sigma) / rsqr;
            lj_part = lj_part * lj_part * lj_part;
            c_number fmod = 24 * eps * (lj_part - 2 * SQR(lj_part)) / rsqr;
            F = r * fmod;
            e = 4 * eps * (SQR(lj_part) - lj_part);
        }
    }
    return e;
}

KOKKOS_INLINE_FUNCTION
c_number f1_sd(const DNA3Params &P, c_number r, int type, int a, int b, int c, int d) {
    c_number val = 0;
    if (r < P(F1_RCHIGH + type, a, b, c, d)) {
        c_number eps = P(F1_EPS + type, a, b, c, d);
        if (r > P(F1_RHIGH + type, a, b, c, d)) {
            val = eps * P(F1_BHIGH + type, a, b, c, d) * SQR(r - P(F1_RCHIGH + type, a, b, c, d));
        } else if (r > P(F1_RLOW + type, a, b, c, d)) {
            c_number tmp = 1 - Kokkos::exp(-(r - P(F1_R0 + type, a, b, c, d)) * P(F1_A + type, a, b, c, d));
            val = eps * SQR(tmp) - P(F1_SHIFT + type, a, b, c, d);
        } else if (r > P(F1_RCLOW + type, a, b, c, d)) {
            val = eps * P(F1_BLOW + type, a, b, c, d) * SQR(r - P(F1_RCLOW + type, a, b, c, d));
        }
    }
    return val;
}

KOKKOS_INLINE_FUNCTION
c_number f1D_sd(const DNA3Params &P, c_number r, int type, int a, int b, int c, int d) {
    c_number val = 0;
    if (r < P(F1_RCHIGH + type, a, b, c, d)) {
        c_number eps = P(F1_EPS + type, a, b, c, d);
        if (r > P(F1_RHIGH + type, a, b, c, d)) {
            val = 2 * eps * P(F1_BHIGH + type, a, b, c, d) * (r - P(F1_RCHIGH + type, a, b, c, d));
        } else if (r > P(F1_RLOW + type, a, b, c, d)) {
            c_number A = P(F1_A + type, a, b, c, d);
            c_number tmp = Kokkos::exp(-(r - P(F1_R0 + type, a, b, c, d)) * A);
            val = 2 * eps * (1 - tmp) * tmp * A;
        } else if (r > P(F1_RCLOW + type, a, b, c, d)) {
            val = 2 * eps * P(F1_BLOW + type, a, b, c, d) * (r - P(F1_RCLOW + type, a, b, c, d));
        }
    }
    return val;
}

KOKKOS_INLINE_FUNCTION
c_number f2_sd(const DNA3Params &P, c_number r, c_number K, int type, int a, int b, int c, int d) {
    c_number val = 0;
    if (r < P(F2_RCHIGH + type, a, b, c, d)) {
        if (r > P(F2_RHIGH + type, a, b, c, d)) {
            val = K * P(F2_BHIGH + type, a, b, c, d) * SQR(r - P(F2_RCHIGH + type, a, b, c, d));
        } else if (r > P(F2_RLOW + type, a, b, c, d)) {
            c_number r0 = P(F2_R0 + type, a, b, c, d);
            val = (K / 2) * (SQR(r - r0) - SQR(P(F2_RC + type, a, b, c, d) - r0));
        } else if (r > P(F2_RCLOW + type, a, b, c, d)) {
            val = K * P(F2_BLOW + type, a, b, c, d) * SQR(r - P(F2_RCLOW + type, a, b, c, d));
        }
    }
    return val;
}

KOKKOS_INLINE_FUNCTION
c_number f2D_sd(const DNA3Params &P, c_number r, c_number K, int type, int a, int b, int c, int d) {
    c_number val = 0;
    if (r < P(F2_RCHIGH + type, a, b, c, d)) {
        if (r > P(F2_RHIGH + type, a, b, c, d)) {
            val = 2 * K * P(F2_BHIGH + type, a, b, c, d) * (r - P(F2_RCHIGH + type, a, b, c, d));
        } else if (r > P(F2_RLOW + type, a, b, c, d)) {
            val = K * (r - P(F2_R0 + type, a, b, c, d));
        } else if (r > P(F2_RCLOW + type, a, b, c, d)) {
            val = 2 * K * P(F2_BLOW + type, a, b, c, d) * (r - P(F2_RCLOW + type, a, b, c, d));
        }
    }
    return val;
}

// non-SD f4 (coaxial stacking): t is the angle
KOKKOS_INLINE_FUNCTION
c_number f4(c_number t, c_number t0, c_number ts, c_number tc, c_number a, c_number b) {
    c_number val = 0;
    t = Kokkos::fabs(t - t0);
    if (t < tc) val = (t > ts) ? b * SQR(tc - t) : 1 - a * SQR(t);
    return val;
}

KOKKOS_INLINE_FUNCTION
c_number f4D(c_number t, c_number t0, c_number ts, c_number tc, c_number a, c_number b) {
    c_number val = 0;
    t -= t0;
    c_number m = Kokkos::copysign(c_number(1), t);
    t = Kokkos::fabs(t);
    if (t < tc) val = (t > ts) ? 2 * m * b * (t - tc) : -2 * m * a * t;
    return val;
}

KOKKOS_INLINE_FUNCTION
c_number f4_pure_harmonic(c_number t, c_number a, c_number b) {
    t -= b;
    return (t < 0) ? c_number(0) : a * SQR(t);
}

KOKKOS_INLINE_FUNCTION
c_number f4D_pure_harmonic(c_number t, c_number a, c_number b) {
    t -= b;
    return (t < 0) ? c_number(0) : 2 * a * t;
}

KOKKOS_INLINE_FUNCTION
c_number f4_sd(const DNA3Params &P, c_number t, int type, int a, int b, int c, int d) {
    c_number val = 0;
    t -= P(F4_T0 + type, a, b, c, d);
    if (t < 0) t *= -1;
    c_number tc = P(F4_TC + type, a, b, c, d);
    if (t < tc) {
        if (t > P(F4_TS + type, a, b, c, d)) val = P(F4_B + type, a, b, c, d) * SQR(tc - t);
        else                                 val = 1 - P(F4_A + type, a, b, c, d) * SQR(t);
    }
    return val;
}

KOKKOS_INLINE_FUNCTION
c_number f4D_sd(const DNA3Params &P, c_number t, int type, int a, int b, int c, int d) {
    c_number val = 0, m = 1;
    t -= P(F4_T0 + type, a, b, c, d);
    if (t < 0) { t *= -1; m = -1; }
    c_number tc = P(F4_TC + type, a, b, c, d);
    if (t < tc) {
        if (t > P(F4_TS + type, a, b, c, d)) val = m * 2 * P(F4_B + type, a, b, c, d) * (t - tc);
        else                                 val = -m * 2 * P(F4_A + type, a, b, c, d) * t;
    }
    return val;
}

KOKKOS_INLINE_FUNCTION
c_number f5_sd(const DNA3Params &P, c_number f, int type, int a, int b, int c, int d) {
    c_number val = 0;
    c_number xc = P(F5_XC + type, a, b, c, d);
    if (f > xc) {
        if (f < P(F5_XS + type, a, b, c, d)) val = P(F5_B + type, a, b, c, d) * SQR(xc - f);
        else if (f < 0)                      val = 1 - P(F5_A + type, a, b, c, d) * SQR(f);
        else                                 val = 1;
    }
    return val;
}

KOKKOS_INLINE_FUNCTION
c_number f5D_sd(const DNA3Params &P, c_number f, int type, int a, int b, int c, int d) {
    c_number val = 0;
    c_number xc = P(F5_XC + type, a, b, c, d);
    if (f > xc) {
        if (f < P(F5_XS + type, a, b, c, d)) val = 2 * P(F5_B + type, a, b, c, d) * (f - xc);
        else if (f < 0)                      val = -2 * P(F5_A + type, a, b, c, d) * f;
    }
    return val;
}

// DNA3_set_interaction_sites (dummy bases, btype == 4, are not supported)
KOKKOS_INLINE_FUNCTION
void set_interaction_sites(const DNA3Params &P, int type, const v3 &a1, const v3 &a2,
                           v3 &r_back, v3 &r_stack, v3 &r_base) {
    r_back  = a1 * P.pos_back1[type] + a2 * P.pos_back2[type];
    r_stack = a1 * P.pos_stack[type];
    r_base  = a1 * P.pos_base[type];
}

// neigh_types: types of the n3/n5 neighbours (NO_TYPE at strand ends)
struct NeighTypes {
    int n3, n5;
    KOKKOS_INLINE_FUNCTION bool is_end() const { return n3 == NO_TYPE || n5 == NO_TYPE; }
};

// ---------------------------------------------------------------------------
// _DNA3_bonded_part<qIsN3>: one bond between n5 (the particle whose n3 is the
// other) and n3; r = n3pos - n5pos. qIsN3: the calling thread owns n5 and
// receives +F/+T; otherwise it owns n3 and receives -F / its own torque.
// Returns the bond energy (FENE + bonded excluded volume + stacking).
// ---------------------------------------------------------------------------
template <bool qIsN3, int TERMS>
KOKKOS_INLINE_FUNCTION
c_number bonded_part(const DNA3Params &P, const v3 &r,
                     int n5type, const v3 &n5x, const v3 &n5y, const v3 &n5z, int neigh_n5_type,
                     int n3type, const v3 &n3x, const v3 &n3y, const v3 &n3z, int neigh_n3_type,
                     v3 &F, v3 &T) {
    const int t0 = neigh_n3_type, t1 = n3type, t2 = n5type, t3 = neigh_n5_type;
    c_number energy_tot = 0;

    v3 n5pos_back, n5pos_base, n5pos_stack;
    set_interaction_sites(P, n5type, n5x, n5y, n5pos_back, n5pos_stack, n5pos_base);
    v3 n3pos_back, n3pos_base, n3pos_stack;
    set_interaction_sites(P, n3type, n3x, n3y, n3pos_back, n3pos_stack, n3pos_base);

    v3 Ftmp = {0, 0, 0}, Ttmp = {0, 0, 0};

    // FENE
    if constexpr ((TERMS & BACKBONE) != 0) {
        v3 rback = r + n3pos_back - n5pos_back;
        c_number rbackmod = vmod(rback);
        c_number rbackr0 = rbackmod - P(FENE_R0, t0, t1, t2, t3);
        c_number fene_delta2 = P(FENE_DELTA2, t0, t1, t2, t3);
        Ftmp = rback * ((P.fene_eps * rbackr0 / (fene_delta2 - SQR(rbackr0))) / rbackmod);
        energy_tot += -P.fene_eps * c_number(0.5) * Kokkos::log(1 - SQR(rbackr0) / fene_delta2);
        Ttmp = qIsN3 ? cross(n5pos_back, Ftmp) : cross(n3pos_back, Ftmp);
    }

    // EXCLUDED VOLUME (_DNA3_bonded_excluded_volume)
    if constexpr ((TERMS & BONDED_EXCLUDED_VOLUME) != 0) {
        v3 Fe;
        // BASE-BASE
        v3 rcenter = r + n3pos_base - n5pos_base;
        energy_tot += excluded_volume(rcenter, Fe, P(EXCL_S + 4, t0, t1, t2, t3), P(EXCL_R + 4, t0, t1, t2, t3),
                                      P(EXCL_B + 4, t0, t1, t2, t3), P(EXCL_RC + 4, t0, t1, t2, t3), P.excl_eps);
        Ttmp += qIsN3 ? cross(n5pos_base, Fe) : cross(n3pos_base, Fe);
        Ftmp += Fe;
        // n5-BASE vs. n3-BACK
        rcenter = r + n3pos_back - n5pos_base;
        energy_tot += excluded_volume(rcenter, Fe, P(EXCL_S + 5, t0, t1, t2, t3), P(EXCL_R + 5, t0, t1, t2, t3),
                                      P(EXCL_B + 5, t0, t1, t2, t3), P(EXCL_RC + 5, t0, t1, t2, t3), P.excl_eps);
        Ttmp += qIsN3 ? cross(n5pos_base, Fe) : cross(n3pos_back, Fe);
        Ftmp += Fe;
        // n5-BACK vs. n3-BASE
        rcenter = r + n3pos_base - n5pos_back;
        energy_tot += excluded_volume(rcenter, Fe, P(EXCL_S + 6, t0, t1, t2, t3), P(EXCL_R + 6, t0, t1, t2, t3),
                                      P(EXCL_B + 6, t0, t1, t2, t3), P(EXCL_RC + 6, t0, t1, t2, t3), P.excl_eps);
        Ttmp += qIsN3 ? cross(n5pos_back, Fe) : cross(n3pos_base, Fe);
        Ftmp += Fe;
    }

    if (qIsN3) { F += Ftmp; T += Ttmp; }
    else       { F -= Ftmp; T -= Ttmp; }

    // STACKING
    if constexpr ((TERMS & STACKING) != 0) {
        constexpr c_number PI = c_number(3.141592653589793);
        const c_number GAMMA = P.gamma;
        v3 rstack = r + n3pos_stack - n5pos_stack;
        c_number rstackmod = vmod(rstack);
        v3 rstackdir = rstack / rstackmod;
        // reference backbone position with equal groove widths
        v3 rbackref = r + n3x * P.pos_back - n5x * P.pos_back;
        c_number rbackrefmod = vmod(rbackref);

        c_number t4 = lracos(dot(n3z, n5z));
        c_number cost5 = dot(n5z, rstackdir);
        c_number t5 = lracos(cost5);
        c_number cost6 = -dot(n3z, rstackdir);
        c_number t6 = lracos(cost6);
        c_number cosphi1 = dot(n5y, rbackref) / rbackrefmod;
        c_number cosphi2 = dot(n3y, rbackref) / rbackrefmod;

        c_number f1 = f1_sd(P, rstackmod, STCK_F1, t0, t1, t2, t3);
        c_number f4t4 = f4_sd(P, t4, STCK_F4_THETA4, t0, t1, t2, t3);
        c_number f4t5 = f4_sd(P, PI - t5, STCK_F4_THETA5, t0, t1, t2, t3);
        c_number f4t6 = f4_sd(P, t6, STCK_F4_THETA6, t0, t1, t2, t3);
        c_number f5phi1 = f5_sd(P, cosphi1, STCK_F5_PHI1, t0, t1, t2, t3);
        c_number f5phi2 = f5_sd(P, cosphi2, STCK_F5_PHI2, t0, t1, t2, t3);

        c_number energy = f1 * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2;

        if (energy != c_number(0)) {
            c_number f1D = f1D_sd(P, rstackmod, STCK_F1, t0, t1, t2, t3);
            c_number f4t4D = f4D_sd(P, t4, STCK_F4_THETA4, t0, t1, t2, t3);
            c_number f4t5D = f4D_sd(P, PI - t5, STCK_F4_THETA5, t0, t1, t2, t3);
            c_number f4t6D = f4D_sd(P, t6, STCK_F4_THETA6, t0, t1, t2, t3);
            c_number f5phi1D = f5D_sd(P, cosphi1, STCK_F5_PHI1, t0, t1, t2, t3);
            c_number f5phi2D = f5D_sd(P, cosphi2, STCK_F5_PHI2, t0, t1, t2, t3);

            // RADIAL
            Ftmp = rstackdir * (energy * f1D / f1);
            // THETA 5
            Ftmp += stably_normalised(n5z - cost5 * rstackdir) * (energy * f4t5D / (f4t5 * rstackmod));
            // THETA 6
            Ftmp += stably_normalised(n3z + cost6 * rstackdir) * (energy * f4t6D / (f4t6 * rstackmod));

            const c_number rbrc = SQR(rbackrefmod) * rbackrefmod;
            // COS PHI 1 (p -> a = n5, q -> b = n3)
            c_number ra2 = dot(rstackdir, n5y);
            c_number ra1 = dot(rstackdir, n5x);
            c_number rb1 = dot(rstackdir, n3x);
            c_number a2b1 = dot(n5y, n3x);
            c_number dcosphi1dr = (SQR(rstackmod) * ra2 - ra2 * SQR(rbackrefmod) - rstackmod * (a2b1 + ra2 * (-ra1 + rb1)) * GAMMA + a2b1 * (-ra1 + rb1) * SQR(GAMMA)) / rbrc;
            c_number dcosphi1dra1 = rstackmod * GAMMA * (rstackmod * ra2 - a2b1 * GAMMA) / rbrc;
            c_number dcosphi1dra2 = -rstackmod / rbackrefmod;
            c_number dcosphi1drb1 = -(rstackmod * GAMMA * (rstackmod * ra2 - a2b1 * GAMMA)) / rbrc;
            c_number dcosphi1da1b1 = SQR(GAMMA) * (-rstackmod * ra2 + a2b1 * GAMMA) / rbrc;
            c_number dcosphi1da2b1 = GAMMA / rbackrefmod;

            c_number force_part_phi1 = energy * f5phi1D / f5phi1;

            Ftmp -= (rstackdir * dcosphi1dr + ((n5y - ra2 * rstackdir) * dcosphi1dra2 + (n5x - ra1 * rstackdir) * dcosphi1dra1 + (n3x - rb1 * rstackdir) * dcosphi1drb1) / rstackmod) * force_part_phi1;

            // COS PHI 2 (p -> b = n3, q -> a = n5)
            ra2 = dot(rstackdir, n3y);
            ra1 = rb1;
            rb1 = dot(rstackdir, n5x);
            a2b1 = dot(n3y, n5x);
            c_number dcosphi2dr = ((rstackmod * ra2 + a2b1 * GAMMA) * (rstackmod + (rb1 - ra1) * GAMMA) - ra2 * SQR(rbackrefmod)) / rbrc;
            c_number dcosphi2dra1 = -rstackmod * GAMMA * (rstackmod * ra2 + a2b1 * GAMMA) / rbrc;
            c_number dcosphi2dra2 = -rstackmod / rbackrefmod;
            c_number dcosphi2drb1 = (rstackmod * GAMMA * (rstackmod * ra2 + a2b1 * GAMMA)) / rbrc;
            c_number dcosphi2da1b1 = -SQR(GAMMA) * (rstackmod * ra2 + a2b1 * GAMMA) / rbrc;
            c_number dcosphi2da2b1 = -GAMMA / rbackrefmod;

            c_number force_part_phi2 = energy * f5phi2D / f5phi2;

            Ftmp -= (rstackdir * dcosphi2dr + ((n3y - rstackdir * ra2) * dcosphi2dra2 + (n3x - rstackdir * ra1) * dcosphi2dra1 + (n5x - rstackdir * rb1) * dcosphi2drb1) / rstackmod) * force_part_phi2;

            Ttmp = qIsN3 ? cross(n5pos_stack, Ftmp) : cross(n3pos_stack, Ftmp);

            // THETA 4
            Ttmp += stably_normalised(cross(n3z, n5z)) * (-energy * f4t4D / f4t4);

            // PHI 1 & PHI 2
            if (qIsN3) {
                Ttmp += (-force_part_phi1 * dcosphi1dra2) * cross(rstackdir, n5y) - cross(rstackdir, n5x) * force_part_phi1 * dcosphi1dra1;
                Ttmp += (-force_part_phi2 * dcosphi2drb1) * cross(rstackdir, n5x);
            } else {
                Ttmp += force_part_phi1 * dcosphi1drb1 * cross(rstackdir, n3x);
                Ttmp += force_part_phi2 * dcosphi2dra2 * cross(rstackdir, n3y) + force_part_phi2 * dcosphi2dra1 * cross(rstackdir, n3x);
            }

            Ttmp += force_part_phi1 * dcosphi1da2b1 * cross(n5y, n3x) + cross(n5x, n3x) * force_part_phi1 * dcosphi1da1b1;
            Ttmp += force_part_phi2 * dcosphi2da2b1 * cross(n5x, n3y) + cross(n5x, n3x) * force_part_phi2 * dcosphi2da1b1;

            energy_tot += energy;
            if (qIsN3) {
                // THETA 5
                Ttmp += stably_normalised(cross(rstackdir, n5z)) * energy * f4t5D / f4t5;
                T += Ttmp;
                F += Ftmp;
            } else {
                // THETA 6
                Ttmp += stably_normalised(cross(rstackdir, n3z)) * (-energy * f4t6D / f4t6);
                T -= Ttmp;
                F -= Ftmp;
            }
        }
    }
    return energy_tot;
}

// ---------------------------------------------------------------------------
// bonded_pair<TERMS>: one bond evaluated ONCE (per-bond scatter kernels).
// Same arguments and conventions as bonded_part (r = n3pos - n5pos). Adds the
// force on the n5 end to F5 (the n3 end gets -F5) and the torques on the n5
// and n3 ends to T5 / T3, reproducing exactly the per-side arithmetic of
// bonded_part<true, TERMS> (n5) and bonded_part<false, TERMS> (n3). Returns
// the bond energy. fene_denom (BACKBONE only) receives 1 - (r - r0)^2 / Delta^2,
// the FENE log argument (<= 0 or NaN: outside the FENE range).
// The bonded excluded volume is not supported here (the LAMMPS-faithful excv
// kernel computes it from the special 1-2 neighbours).
// ---------------------------------------------------------------------------
template <int TERMS>
KOKKOS_INLINE_FUNCTION
c_number bonded_pair(const DNA3Params &P, const v3 &r,
                     int n5type, const v3 &n5x, const v3 &n5y, const v3 &n5z, int neigh_n5_type,
                     int n3type, const v3 &n3x, const v3 &n3y, const v3 &n3z, int neigh_n3_type,
                     v3 &F5, v3 &T5, v3 &T3, c_number &fene_denom) {
    static_assert((TERMS & BONDED_EXCLUDED_VOLUME) == 0,
                  "bonded_pair: the bonded excluded volume is computed by the excv kernel");
    const int t0 = neigh_n3_type, t1 = n3type, t2 = n5type, t3 = neigh_n5_type;
    c_number energy_tot = 0;

    v3 n5pos_back, n5pos_base, n5pos_stack;
    set_interaction_sites(P, n5type, n5x, n5y, n5pos_back, n5pos_stack, n5pos_base);
    v3 n3pos_back, n3pos_base, n3pos_stack;
    set_interaction_sites(P, n3type, n3x, n3y, n3pos_back, n3pos_stack, n3pos_base);

    // FENE
    if constexpr ((TERMS & BACKBONE) != 0) {
        v3 rback = r + n3pos_back - n5pos_back;
        c_number rbackmod = vmod(rback);
        c_number rbackr0 = rbackmod - P(FENE_R0, t0, t1, t2, t3);
        c_number fene_delta2 = P(FENE_DELTA2, t0, t1, t2, t3);
        v3 Ftmp = rback * ((P.fene_eps * rbackr0 / (fene_delta2 - SQR(rbackr0))) / rbackmod);
        fene_denom = 1 - SQR(rbackr0) / fene_delta2;
        energy_tot += -P.fene_eps * c_number(0.5) * Kokkos::log(fene_denom);
        F5 += Ftmp;
        T5 += cross(n5pos_back, Ftmp);
        T3 -= cross(n3pos_back, Ftmp);
    }

    // STACKING
    if constexpr ((TERMS & STACKING) != 0) {
        constexpr c_number PI = c_number(3.141592653589793);
        const c_number GAMMA = P.gamma;
        v3 rstack = r + n3pos_stack - n5pos_stack;
        c_number rstackmod = vmod(rstack);
        v3 rstackdir = rstack / rstackmod;
        v3 rbackref = r + n3x * P.pos_back - n5x * P.pos_back;
        c_number rbackrefmod = vmod(rbackref);

        c_number t4 = lracos(dot(n3z, n5z));
        c_number cost5 = dot(n5z, rstackdir);
        c_number t5 = lracos(cost5);
        c_number cost6 = -dot(n3z, rstackdir);
        c_number t6 = lracos(cost6);
        c_number cosphi1 = dot(n5y, rbackref) / rbackrefmod;
        c_number cosphi2 = dot(n3y, rbackref) / rbackrefmod;

        c_number f1 = f1_sd(P, rstackmod, STCK_F1, t0, t1, t2, t3);
        c_number f4t4 = f4_sd(P, t4, STCK_F4_THETA4, t0, t1, t2, t3);
        c_number f4t5 = f4_sd(P, PI - t5, STCK_F4_THETA5, t0, t1, t2, t3);
        c_number f4t6 = f4_sd(P, t6, STCK_F4_THETA6, t0, t1, t2, t3);
        c_number f5phi1 = f5_sd(P, cosphi1, STCK_F5_PHI1, t0, t1, t2, t3);
        c_number f5phi2 = f5_sd(P, cosphi2, STCK_F5_PHI2, t0, t1, t2, t3);

        c_number energy = f1 * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2;

        if (energy != c_number(0)) {
            c_number f1D = f1D_sd(P, rstackmod, STCK_F1, t0, t1, t2, t3);
            c_number f4t4D = f4D_sd(P, t4, STCK_F4_THETA4, t0, t1, t2, t3);
            c_number f4t5D = f4D_sd(P, PI - t5, STCK_F4_THETA5, t0, t1, t2, t3);
            c_number f4t6D = f4D_sd(P, t6, STCK_F4_THETA6, t0, t1, t2, t3);
            c_number f5phi1D = f5D_sd(P, cosphi1, STCK_F5_PHI1, t0, t1, t2, t3);
            c_number f5phi2D = f5D_sd(P, cosphi2, STCK_F5_PHI2, t0, t1, t2, t3);

            // RADIAL
            v3 Ftmp = rstackdir * (energy * f1D / f1);
            // THETA 5
            Ftmp += stably_normalised(n5z - cost5 * rstackdir) * (energy * f4t5D / (f4t5 * rstackmod));
            // THETA 6
            Ftmp += stably_normalised(n3z + cost6 * rstackdir) * (energy * f4t6D / (f4t6 * rstackmod));

            const c_number rbrc = SQR(rbackrefmod) * rbackrefmod;
            // COS PHI 1
            c_number ra2 = dot(rstackdir, n5y);
            c_number ra1 = dot(rstackdir, n5x);
            c_number rb1 = dot(rstackdir, n3x);
            c_number a2b1 = dot(n5y, n3x);
            c_number dcosphi1dr = (SQR(rstackmod) * ra2 - ra2 * SQR(rbackrefmod) - rstackmod * (a2b1 + ra2 * (-ra1 + rb1)) * GAMMA + a2b1 * (-ra1 + rb1) * SQR(GAMMA)) / rbrc;
            c_number dcosphi1dra1 = rstackmod * GAMMA * (rstackmod * ra2 - a2b1 * GAMMA) / rbrc;
            c_number dcosphi1dra2 = -rstackmod / rbackrefmod;
            c_number dcosphi1drb1 = -(rstackmod * GAMMA * (rstackmod * ra2 - a2b1 * GAMMA)) / rbrc;
            c_number dcosphi1da1b1 = SQR(GAMMA) * (-rstackmod * ra2 + a2b1 * GAMMA) / rbrc;
            c_number dcosphi1da2b1 = GAMMA / rbackrefmod;

            c_number force_part_phi1 = energy * f5phi1D / f5phi1;

            Ftmp -= (rstackdir * dcosphi1dr + ((n5y - ra2 * rstackdir) * dcosphi1dra2 + (n5x - ra1 * rstackdir) * dcosphi1dra1 + (n3x - rb1 * rstackdir) * dcosphi1drb1) / rstackmod) * force_part_phi1;

            // COS PHI 2
            ra2 = dot(rstackdir, n3y);
            ra1 = rb1;
            rb1 = dot(rstackdir, n5x);
            a2b1 = dot(n3y, n5x);
            c_number dcosphi2dr = ((rstackmod * ra2 + a2b1 * GAMMA) * (rstackmod + (rb1 - ra1) * GAMMA) - ra2 * SQR(rbackrefmod)) / rbrc;
            c_number dcosphi2dra1 = -rstackmod * GAMMA * (rstackmod * ra2 + a2b1 * GAMMA) / rbrc;
            c_number dcosphi2dra2 = -rstackmod / rbackrefmod;
            c_number dcosphi2drb1 = (rstackmod * GAMMA * (rstackmod * ra2 + a2b1 * GAMMA)) / rbrc;
            c_number dcosphi2da1b1 = -SQR(GAMMA) * (rstackmod * ra2 + a2b1 * GAMMA) / rbrc;
            c_number dcosphi2da2b1 = -GAMMA / rbackrefmod;

            c_number force_part_phi2 = energy * f5phi2D / f5phi2;

            Ftmp -= (rstackdir * dcosphi2dr + ((n3y - rstackdir * ra2) * dcosphi2dra2 + (n3x - rstackdir * ra1) * dcosphi2dra1 + (n5x - rstackdir * rb1) * dcosphi2drb1) / rstackmod) * force_part_phi2;

            // torque terms shared by both ends
            const v3 Tt4 = stably_normalised(cross(n3z, n5z)) * (-energy * f4t4D / f4t4);
            const v3 Tab1 = force_part_phi1 * dcosphi1da2b1 * cross(n5y, n3x) + cross(n5x, n3x) * force_part_phi1 * dcosphi1da1b1;
            const v3 Tab2 = force_part_phi2 * dcosphi2da2b1 * cross(n5x, n3y) + cross(n5x, n3x) * force_part_phi2 * dcosphi2da1b1;

            // n5 end (bonded_part<true>)
            {
                v3 Ttmp = cross(n5pos_stack, Ftmp);
                Ttmp += Tt4;
                Ttmp += (-force_part_phi1 * dcosphi1dra2) * cross(rstackdir, n5y) - cross(rstackdir, n5x) * force_part_phi1 * dcosphi1dra1;
                Ttmp += (-force_part_phi2 * dcosphi2drb1) * cross(rstackdir, n5x);
                Ttmp += Tab1;
                Ttmp += Tab2;
                Ttmp += stably_normalised(cross(rstackdir, n5z)) * energy * f4t5D / f4t5;   // THETA 5
                T5 += Ttmp;
            }
            // n3 end (bonded_part<false>)
            {
                v3 Ttmp = cross(n3pos_stack, Ftmp);
                Ttmp += Tt4;
                Ttmp += force_part_phi1 * dcosphi1drb1 * cross(rstackdir, n3x);
                Ttmp += force_part_phi2 * dcosphi2dra2 * cross(rstackdir, n3y) + force_part_phi2 * dcosphi2dra1 * cross(rstackdir, n3x);
                Ttmp += Tab1;
                Ttmp += Tab2;
                Ttmp += stably_normalised(cross(rstackdir, n3z)) * (-energy * f4t6D / f4t6);   // THETA 6
                T3 -= Ttmp;
            }
            F5 += Ftmp;
            energy_tot += energy;
        }
    }
    return energy_tot;
}

// ---------------------------------------------------------------------------
// _DNA3_particle_particle_DNA_interaction: nonbonded pair (p, q), r = q - p.
// Accumulates the force F and the (lab-frame) torque T acting on p.
// ---------------------------------------------------------------------------
template <int TERMS>
KOKKOS_INLINE_FUNCTION
c_number particle_particle_interaction(const DNA3Params &P, const v3 &r,
                                       int ptype, const v3 &a1, const v3 &a2, const v3 &a3,
                                       int qtype, const v3 &b1, const v3 &b2, const v3 &b3,
                                       v3 &F, v3 &T, const NeighTypes &p_neighs, const NeighTypes &q_neighs) {
    constexpr c_number PI = c_number(3.141592653589793);
    // btype == type for the four natural bases (no custom btypes >= 300 here)
    const int int_type = ptype + qtype;

    v3 ppos_back, ppos_base, ppos_stack;
    set_interaction_sites(P, ptype, a1, a2, ppos_back, ppos_stack, ppos_base);
    v3 qpos_back, qpos_base, qpos_stack;
    set_interaction_sites(P, qtype, b1, b2, qpos_back, qpos_stack, qpos_base);

    c_number etot = 0;
    v3 Ftmp = {0, 0, 0};
    v3 Ttmp = {0, 0, 0};

    // excluded volume (_DNA3_nonbonded_excluded_volume, flanks = NO_TYPE)
    if constexpr ((TERMS & NONBONDED_EXCLUDED_VOLUME) != 0) {
        const int t0 = NO_TYPE, t1 = qtype, t2 = ptype, t3 = NO_TYPE;
        v3 Fe, Fs = {0, 0, 0};
        // BASE-BASE
        v3 rcenter = r + qpos_base - ppos_base;
        etot += excluded_volume(rcenter, Fe, P(EXCL_S + 1, t0, t1, t2, t3), P(EXCL_R + 1, t0, t1, t2, t3),
                                P(EXCL_B + 1, t0, t1, t2, t3), P(EXCL_RC + 1, t0, t1, t2, t3), P.excl_eps);
        v3 torquep = cross(ppos_base, Fe);
        Fs += Fe;
        // p-BASE vs. q-BACK
        rcenter = r + qpos_back - ppos_base;
        etot += excluded_volume(rcenter, Fe, P(EXCL_S + 3, t0, t1, t2, t3), P(EXCL_R + 3, t0, t1, t2, t3),
                                P(EXCL_B + 3, t0, t1, t2, t3), P(EXCL_RC + 3, t0, t1, t2, t3), P.excl_eps);
        torquep += cross(ppos_base, Fe);
        Fs += Fe;
        // p-BACK vs. q-BASE
        rcenter = r + qpos_base - ppos_back;
        etot += excluded_volume(rcenter, Fe, P(EXCL_S + 2, t0, t1, t2, t3), P(EXCL_R + 2, t0, t1, t2, t3),
                                P(EXCL_B + 2, t0, t1, t2, t3), P(EXCL_RC + 2, t0, t1, t2, t3), P.excl_eps);
        torquep += cross(ppos_back, Fe);
        Fs += Fe;
        // BACK-BACK
        rcenter = r + qpos_back - ppos_back;
        etot += excluded_volume(rcenter, Fe, P(EXCL_S + 0, t0, t1, t2, t3), P(EXCL_R + 0, t0, t1, t2, t3),
                                P(EXCL_B + 0, t0, t1, t2, t3), P(EXCL_RC + 0, t0, t1, t2, t3), P.excl_eps);
        torquep += cross(ppos_back, Fe);
        Fs += Fe;
        Ttmp += torquep;
        F += Fs;
    }

    // DEBYE HUCKEL
    if constexpr ((TERMS & DEBYE_HUCKEL) != 0) {
        v3 rbackbone = r + qpos_back - ppos_back;
        c_number rbackmod = vmod(rbackbone);
        if (rbackmod < P.dh_RC) {
            v3 rbackdir = rbackbone / rbackmod;
            c_number e;
            if (rbackmod < P.dh_RHIGH) {
                c_number ex = Kokkos::exp(P.dh_minus_kappa * rbackmod);
                Ftmp = rbackdir * (-P.dh_prefactor * ex * (P.dh_minus_kappa / rbackmod - 1 / SQR(rbackmod)));
                e = ex * (P.dh_prefactor / rbackmod);
            } else {
                Ftmp = rbackdir * (-2 * P.dh_B * (rbackmod - P.dh_RC));
                e = P.dh_B * SQR(rbackmod - P.dh_RC);
            }
            // half-charged strand ends
            if (P.dh_half_ends && p_neighs.is_end()) { Ftmp = Ftmp * c_number(0.5); e *= c_number(0.5); }
            if (P.dh_half_ends && q_neighs.is_end()) { Ftmp = Ftmp * c_number(0.5); e *= c_number(0.5); }
            Ttmp -= cross(ppos_back, Ftmp);
            F -= Ftmp;
            etot += e;
        }
    }

    // HYDROGEN BONDING
    v3 rhydro = r + qpos_base - ppos_base;
    c_number rhydromodsqr = dot(rhydro, rhydro);
    if constexpr ((TERMS & HYDROGEN_BONDING) != 0) {
        if (int_type == 3 && SQR(P(F1_RCLOW + HYDR_F1, 0, qtype, ptype, 0)) < rhydromodsqr &&
            rhydromodsqr < SQR(P(F1_RCHIGH + HYDR_F1, 0, qtype, ptype, 0))) {
            const int t0 = 0, t1 = qtype, t2 = ptype, t3 = 0;
            c_number rhydromod = Kokkos::sqrt(rhydromodsqr);
            v3 rhydrodir = rhydro / rhydromod;

            c_number t1a = lracos(-dot(a1, b1));
            c_number cost2 = -dot(b1, rhydrodir);
            c_number t2a = lracos(cost2);
            c_number cost3 = dot(a1, rhydrodir);
            c_number t3a = lracos(cost3);
            c_number t4a = lracos(dot(a3, b3));
            c_number cost7 = -dot(rhydrodir, b3);
            c_number t7a = lracos(cost7);
            c_number cost8 = dot(rhydrodir, a3);
            c_number t8a = lracos(cost8);

            c_number f1 = f1_sd(P, rhydromod, HYDR_F1, t0, t1, t2, t3);
            c_number f4t1 = f4_sd(P, t1a, HYDR_F4_THETA1, t0, t1, t2, t3);
            c_number f4t2 = f4_sd(P, t2a, HYDR_F4_THETA2, t0, t1, t2, t3);
            c_number f4t3 = f4_sd(P, t3a, HYDR_F4_THETA3, t0, t1, t2, t3);
            c_number f4t4 = f4_sd(P, t4a, HYDR_F4_THETA4, t0, t1, t2, t3);
            c_number f4t7 = f4_sd(P, t7a, HYDR_F4_THETA7, t0, t1, t2, t3);
            c_number f4t8 = f4_sd(P, t8a, HYDR_F4_THETA8, t0, t1, t2, t3);

            c_number hb_energy = f1 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8;
            etot += hb_energy;

            if (hb_energy < c_number(0)) {
                c_number f1D = f1D_sd(P, rhydromod, HYDR_F1, t0, t1, t2, t3);
                c_number f4t1D = -f4D_sd(P, t1a, HYDR_F4_THETA1, t0, t1, t2, t3);
                c_number f4t2D = -f4D_sd(P, t2a, HYDR_F4_THETA2, t0, t1, t2, t3);
                c_number f4t3D = f4D_sd(P, t3a, HYDR_F4_THETA3, t0, t1, t2, t3);
                c_number f4t4D = f4D_sd(P, t4a, HYDR_F4_THETA4, t0, t1, t2, t3);
                c_number f4t7D = -f4D_sd(P, t7a, HYDR_F4_THETA7, t0, t1, t2, t3);
                c_number f4t8D = f4D_sd(P, t8a, HYDR_F4_THETA8, t0, t1, t2, t3);

                // RADIAL PART
                Ftmp = rhydrodir * hb_energy * f1D / f1;
                // TETA4
                Ttmp -= stably_normalised(cross(a3, b3)) * (-hb_energy * f4t4D / f4t4);
                // TETA1
                Ttmp -= stably_normalised(cross(a1, b1)) * (-hb_energy * f4t1D / f4t1);
                // TETA2
                Ftmp -= stably_normalised(b1 + rhydrodir * cost2) * (hb_energy * f4t2D / (f4t2 * rhydromod));
                // TETA3
                c_number part = -hb_energy * f4t3D / f4t3;
                Ftmp -= stably_normalised(a1 - rhydrodir * cost3) * (-part / rhydromod);
                Ttmp += stably_normalised(cross(rhydrodir, a1)) * part;
                // THETA7
                Ftmp -= stably_normalised(b3 + rhydrodir * cost7) * (hb_energy * f4t7D / (f4t7 * rhydromod));
                // THETA8
                part = -hb_energy * f4t8D / f4t8;
                Ftmp -= stably_normalised(a3 - rhydrodir * cost8) * (-part / rhydromod);
                Ttmp += stably_normalised(cross(rhydrodir, a3)) * part;

                Ttmp += cross(ppos_base, Ftmp);
                F += Ftmp;
            }
        }
    }

    // CROSS STACKING
    const int type_p_n3 = p_neighs.n3, type_p_n5 = p_neighs.n5;
    const int type_q_n3 = q_neighs.n3, type_q_n5 = q_neighs.n5;
    if constexpr ((TERMS & CROSS_STACKING) != 0) {
        v3 rcstack = rhydro;
        c_number rcstackmod = Kokkos::sqrt(rhydromodsqr);
        v3 rcstackdir = rcstack / rcstackmod;
        c_number cost7 = -dot(rcstackdir, b3);
        c_number cost8 = dot(rcstackdir, a3);
        if ((cost7 > 0 && cost8 > 0 &&
             P(F2_RCLOW + CRST_F2_33, type_q_n3, qtype, ptype, type_p_n3) < rcstackmod &&
             rcstackmod < P(F2_RCHIGH + CRST_F2_33, type_q_n3, qtype, ptype, type_p_n3)) ||
            (cost7 < 0 && cost8 < 0 &&
             P(F2_RCLOW + CRST_F2_55, type_q_n5, qtype, ptype, type_p_n5) < rcstackmod &&
             rcstackmod < P(F2_RCHIGH + CRST_F2_55, type_q_n5, qtype, ptype, type_p_n5))) {
            const int a33 = type_q_n3, b33 = qtype, c33 = ptype, d33 = type_p_n3;
            const int a55 = type_q_n5, b55 = qtype, c55 = ptype, d55 = type_p_n5;

            c_number t1a = lracos(-dot(a1, b1));
            c_number cost2 = -dot(b1, rcstackdir);
            c_number t2a = lracos(cost2);
            c_number cost3 = dot(a1, rcstackdir);
            c_number t3a = lracos(cost3);
            c_number t4a = lracos(dot(a3, b3));
            c_number t7a = lracos(cost7);
            c_number t8a = lracos(cost8);

            // 3'3' diagonal
            c_number K33 = P(F2_K + CRST_F2_33, a33, b33, c33, d33);
            c_number f2_33 = f2_sd(P, rcstackmod, K33, CRST_F2_33, a33, b33, c33, d33);
            c_number f4t1_33 = f4_sd(P, t1a, CRST_F4_THETA1_33, a33, b33, c33, d33);
            c_number f4t2_33 = f4_sd(P, t2a, CRST_F4_THETA2_33, a33, b33, c33, d33);
            c_number f4t3_33 = f4_sd(P, t3a, CRST_F4_THETA3_33, a33, b33, c33, d33);
            c_number f4t4_33 = f4_sd(P, t4a, CRST_F4_THETA4_33, a33, b33, c33, d33);
            c_number f4t7_33 = f4_sd(P, t7a, CRST_F4_THETA7_33, a33, b33, c33, d33);
            c_number f4t8_33 = f4_sd(P, t8a, CRST_F4_THETA8_33, a33, b33, c33, d33);
            // 5'5' diagonal
            c_number K55 = P(F2_K + CRST_F2_55, a55, b55, c55, d55);
            c_number f2_55 = f2_sd(P, rcstackmod, K55, CRST_F2_55, a55, b55, c55, d55);
            c_number f4t1_55 = f4_sd(P, t1a, CRST_F4_THETA1_55, a55, b55, c55, d55);
            c_number f4t2_55 = f4_sd(P, t2a, CRST_F4_THETA2_55, a55, b55, c55, d55);
            c_number f4t3_55 = f4_sd(P, t3a, CRST_F4_THETA3_55, a55, b55, c55, d55);
            c_number f4t4_55 = f4_sd(P, t4a, CRST_F4_THETA4_55, a55, b55, c55, d55);
            c_number f4t7_55 = f4_sd(P, t7a, CRST_F4_THETA7_55, a55, b55, c55, d55);
            c_number f4t8_55 = f4_sd(P, t8a, CRST_F4_THETA8_55, a55, b55, c55, d55);

            c_number cstk_energy = f2_33 * f4t1_33 * f4t2_33 * f4t3_33 * f4t4_33 * f4t7_33 * f4t8_33 +
                                   f2_55 * f4t1_55 * f4t2_55 * f4t3_55 * f4t4_55 * f4t7_55 * f4t8_55;

            if (cstk_energy < c_number(0)) {
                c_number f2D_33 = f2D_sd(P, rcstackmod, K33, CRST_F2_33, a33, b33, c33, d33);
                c_number f4t1Dsin_33 = -f4D_sd(P, t1a, CRST_F4_THETA1_33, a33, b33, c33, d33);
                c_number f4t2Dsin_33 = -f4D_sd(P, t2a, CRST_F4_THETA2_33, a33, b33, c33, d33);
                c_number f4t3Dsin_33 = f4D_sd(P, t3a, CRST_F4_THETA3_33, a33, b33, c33, d33);
                c_number f4t4Dsin_33 = f4D_sd(P, t4a, CRST_F4_THETA4_33, a33, b33, c33, d33);
                c_number f4t7Dsin_33 = -f4D_sd(P, t7a, CRST_F4_THETA7_33, a33, b33, c33, d33);
                c_number f4t8Dsin_33 = f4D_sd(P, t8a, CRST_F4_THETA8_33, a33, b33, c33, d33);

                c_number f2D_55 = f2D_sd(P, rcstackmod, K55, CRST_F2_55, a55, b55, c55, d55);
                c_number f4t1Dsin_55 = -f4D_sd(P, t1a, CRST_F4_THETA1_55, a55, b55, c55, d55);
                c_number f4t2Dsin_55 = -f4D_sd(P, t2a, CRST_F4_THETA2_55, a55, b55, c55, d55);
                c_number f4t3Dsin_55 = f4D_sd(P, t3a, CRST_F4_THETA3_55, a55, b55, c55, d55);
                c_number f4t4Dsin_55 = f4D_sd(P, t4a, CRST_F4_THETA4_55, a55, b55, c55, d55);
                c_number f4t7Dsin_55 = -f4D_sd(P, t7a, CRST_F4_THETA7_55, a55, b55, c55, d55);
                c_number f4t8Dsin_55 = f4D_sd(P, t8a, CRST_F4_THETA8_55, a55, b55, c55, d55);

                // RADIAL PART
                Ftmp = rcstackdir * ((f2D_33 * f4t1_33 * f4t2_33 * f4t3_33 * f4t4_33 * f4t7_33 * f4t8_33) +
                                     (f2D_55 * f4t1_55 * f4t2_55 * f4t3_55 * f4t4_55 * f4t7_55 * f4t8_55));
                // THETA1
                Ttmp -= stably_normalised(cross(a1, b1)) *
                        (-f2_33 * f4t1Dsin_33 * f4t2_33 * f4t3_33 * f4t4_33 * f4t7_33 * f4t8_33 -
                         f2_55 * f4t1Dsin_55 * f4t2_55 * f4t3_55 * f4t4_55 * f4t7_55 * f4t8_55);
                // TETA2
                Ftmp -= stably_normalised(b1 + rcstackdir * cost2) *
                        ((f2_33 * f4t1_33 * f4t2Dsin_33 * f4t3_33 * f4t4_33 * f4t7_33 * f4t8_33 +
                          f2_55 * f4t1_55 * f4t2Dsin_55 * f4t3_55 * f4t4_55 * f4t7_55 * f4t8_55) / rcstackmod);
                // TETA3
                Ftmp -= stably_normalised(a1 - rcstackdir * cost3) *
                        ((f2_33 * f4t1_33 * f4t2_33 * f4t3Dsin_33 * f4t4_33 * f4t7_33 * f4t8_33 +
                          f2_55 * f4t1_55 * f4t2_55 * f4t3Dsin_55 * f4t4_55 * f4t7_55 * f4t8_55) / rcstackmod);
                Ttmp += stably_normalised(cross(rcstackdir, a1)) *
                        (-f2_33 * f4t1_33 * f4t2_33 * f4t3Dsin_33 * f4t4_33 * f4t7_33 * f4t8_33 -
                         f2_55 * f4t1_55 * f4t2_55 * f4t3Dsin_55 * f4t4_55 * f4t7_55 * f4t8_55);
                // TETA4
                Ttmp -= stably_normalised(cross(a3, b3)) *
                        (-f2_33 * f4t1_33 * f4t2_33 * f4t3_33 * f4t4Dsin_33 * f4t7_33 * f4t8_33 -
                         f2_55 * f4t1_55 * f4t2_55 * f4t3_55 * f4t4Dsin_55 * f4t7_55 * f4t8_55);
                // THETA7
                Ftmp -= stably_normalised(b3 + rcstackdir * cost7) *
                        ((f2_33 * f4t1_33 * f4t2_33 * f4t3_33 * f4t4_33 * f4t7Dsin_33 * f4t8_33 +
                          f2_55 * f4t1_55 * f4t2_55 * f4t3_55 * f4t4_55 * f4t7Dsin_55 * f4t8_55) / rcstackmod);
                // THETA8
                Ftmp -= stably_normalised(a3 - rcstackdir * cost8) *
                        ((f2_33 * f4t1_33 * f4t2_33 * f4t3_33 * f4t4_33 * f4t7_33 * f4t8Dsin_33 +
                          f2_55 * f4t1_55 * f4t2_55 * f4t3_55 * f4t4_55 * f4t7_55 * f4t8Dsin_55) / rcstackmod);
                Ttmp += stably_normalised(cross(rcstackdir, a3)) *
                        (-f2_33 * f4t1_33 * f4t2_33 * f4t3_33 * f4t4_33 * f4t7_33 * f4t8Dsin_33 -
                         f2_55 * f4t1_55 * f4t2_55 * f4t3_55 * f4t4_55 * f4t7_55 * f4t8Dsin_55);

                Ttmp += cross(ppos_base, Ftmp);
                F += Ftmp;
                etot += cstk_energy;
            }
        }
    }

    // COAXIAL STACKING
    if constexpr ((TERMS & COAXIAL_STACKING) != 0) {
        v3 rstack = r + qpos_stack - ppos_stack;
        c_number rstackmodsqr = dot(rstack, rstack);
        if (SQR(P(F2_RCLOW + CXST_F2, 0, qtype, ptype, 0)) < rstackmodsqr &&
            rstackmodsqr < SQR(P(F2_RCHIGH + CXST_F2, 0, qtype, ptype, 0))) {
            c_number rstackmod = Kokkos::sqrt(rstackmodsqr);
            v3 rstackdir = rstack / rstackmod;

            c_number t1a = lracos(-dot(a1, b1));
            c_number t4a = lracos(dot(a3, b3));
            c_number cost5 = dot(a3, rstackdir);
            c_number t5a = lracos(cost5);
            c_number cost6 = -dot(b3, rstackdir);
            c_number t6a = lracos(cost6);

            // K selection: 3'-end p facing 5'-end q, the reverse, or symmetric
            int ka, kb, kc, kd, ktab;
            if (type_p_n3 == NO_TYPE && type_q_n5 == NO_TYPE) {
                ka = type_q_n3; kb = qtype; kc = ptype; kd = type_p_n5; ktab = F2_K;
            } else if (type_p_n5 == NO_TYPE && type_q_n3 == NO_TYPE) {
                ka = type_p_n5; kb = ptype; kc = qtype; kd = type_q_n3; ktab = F2_K;
            } else {
                ka = type_q_n3; kb = qtype; kc = ptype; kd = type_p_n5; ktab = F2_K_SYMM;
            }
            c_number K = P(ktab + CXST_F2, ka, kb, kc, kd);
            c_number f2 = f2_sd(P, rstackmod, K, CXST_F2, ka, kb, kc, kd);

            c_number f4t1 = f4(t1a, P.cx_t1_t0, P.cx_t1_ts, P.cx_t1_tc, P.cx_t1_a, P.cx_t1_b) +
                            f4_pure_harmonic(t1a, P.cx_t1_sa, P.cx_t1_sb);
            c_number f4t4 = f4(t4a, P.cx_t4_t0, P.cx_t4_ts, P.cx_t4_tc, P.cx_t4_a, P.cx_t4_b);
            // LAMMPS-only mirrored theta4 lobe (off by default)
            if (P.cxst_t4_blunt)
                f4t4 += f4(t4a, PI - P.cx_t4_t0, P.cx_t4_ts, P.cx_t4_tc, P.cx_t4_a, P.cx_t4_b);
            c_number f4t5 = f4(t5a, P.cx_t5_t0, P.cx_t5_ts, P.cx_t5_tc, P.cx_t5_a, P.cx_t5_b) +
                            f4(PI - t5a, P.cx_t5_t0, P.cx_t5_ts, P.cx_t5_tc, P.cx_t5_a, P.cx_t5_b);
            c_number f4t6 = f4(t6a, P.cx_t5_t0, P.cx_t5_ts, P.cx_t5_tc, P.cx_t5_a, P.cx_t5_b) +
                            f4(PI - t6a, P.cx_t5_t0, P.cx_t5_ts, P.cx_t5_tc, P.cx_t5_a, P.cx_t5_b);

            c_number cxst_energy = f2 * f4t1 * f4t4 * f4t5 * f4t6;

            if (cxst_energy < c_number(0)) {
                c_number f2D = f2D_sd(P, rstackmod, K, CXST_F2, ka, kb, kc, kd);
                c_number f4t1D = -f4D(t1a, P.cx_t1_t0, P.cx_t1_ts, P.cx_t1_tc, P.cx_t1_a, P.cx_t1_b) -
                                 f4D_pure_harmonic(t1a, P.cx_t1_sa, P.cx_t1_sb);
                c_number f4t4D = f4D(t4a, P.cx_t4_t0, P.cx_t4_ts, P.cx_t4_tc, P.cx_t4_a, P.cx_t4_b);
                if (P.cxst_t4_blunt)
                    f4t4D += f4D(t4a, PI - P.cx_t4_t0, P.cx_t4_ts, P.cx_t4_tc, P.cx_t4_a, P.cx_t4_b);
                c_number f4t5D = f4D(t5a, P.cx_t5_t0, P.cx_t5_ts, P.cx_t5_tc, P.cx_t5_a, P.cx_t5_b) -
                                 f4D(PI - t5a, P.cx_t5_t0, P.cx_t5_ts, P.cx_t5_tc, P.cx_t5_a, P.cx_t5_b);
                c_number f4t6D = -f4D(t6a, P.cx_t5_t0, P.cx_t5_ts, P.cx_t5_tc, P.cx_t5_a, P.cx_t5_b) +
                                 f4D(PI - t6a, P.cx_t5_t0, P.cx_t5_ts, P.cx_t5_tc, P.cx_t5_a, P.cx_t5_b);

                // RADIAL PART
                Ftmp = rstackdir * (cxst_energy * f2D / f2);
                // THETA1
                Ttmp -= stably_normalised(cross(a1, b1)) * (-cxst_energy * f4t1D / f4t1);
                // TETA4
                Ttmp -= stably_normalised(cross(a3, b3)) * (-cxst_energy * f4t4D / f4t4);
                // THETA5
                c_number part = cxst_energy * f4t5D / f4t5;
                Ftmp -= stably_normalised(a3 - rstackdir * cost5) / rstackmod * part;
                Ttmp -= stably_normalised(cross(rstackdir, a3)) * part;
                // THETA6
                Ftmp -= stably_normalised(b3 + rstackdir * cost6) * (cxst_energy * f4t6D / (f4t6 * rstackmod));

                Ttmp += cross(ppos_stack, Ftmp);
                F += Ftmp;
                etot += cxst_energy;
            }
        }
    }
    T += Ttmp;
    return etot;
}

} // namespace dna3
