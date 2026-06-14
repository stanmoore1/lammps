#pragma once

// Modulation functions F1–F6 for the oxDNA force field.
// Ported directly from LAMMPS src/KOKKOS/mf_oxdna_kokkos.h with
// KK_FLOAT replaced by c_number and Kokkos:: math calls retained.

#include "../types.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace MFOxdna {

// F1: piecewise exponential radial modulation (base-pairing, stacking)
KOKKOS_INLINE_FUNCTION
static c_number F1(c_number r, c_number eps, c_number a, c_number cut_0,
                   c_number cut_lc, c_number cut_hc, c_number cut_lo,
                   c_number cut_hi, c_number b_lo, c_number b_hi, c_number shift) {
    if (r > cut_hc) {
        return 0;
    } else if (r > cut_hi) {
        return eps * b_hi * (r - cut_hc) * (r - cut_hc);
    } else if (r > cut_lo) {
        c_number tmp = 1 - Kokkos::exp(-(r - cut_0) * a);
        return eps * tmp * tmp - shift;
    } else if (r > cut_lc) {
        return eps * b_lo * (r - cut_lc) * (r - cut_lc);
    } else {
        return 0;
    }
}

KOKKOS_INLINE_FUNCTION
static c_number DF1(c_number r, c_number eps, c_number a, c_number cut_0,
                    c_number cut_lc, c_number cut_hc, c_number cut_lo,
                    c_number cut_hi, c_number b_lo, c_number b_hi) {
    if (r > cut_hc) {
        return 0;
    } else if (r > cut_hi) {
        return 2 * eps * b_hi * (1 - cut_hc / r);
    } else if (r > cut_lo) {
        c_number tmp = Kokkos::exp(-(r - cut_0) * a);
        return 2 * eps * (1 - tmp) * tmp * a / r;
    } else if (r > cut_lc) {
        return 2 * eps * b_lo * (1 - cut_lc / r);
    } else {
        return 0;
    }
}

// F2: harmonic radial modulation (cross-stacking)
KOKKOS_INLINE_FUNCTION
static c_number F2(c_number r, c_number k, c_number cut_0, c_number cut_lc,
                   c_number cut_hc, c_number cut_lo, c_number cut_hi,
                   c_number b_lo, c_number b_hi, c_number cut_c) {
    if (r < cut_lc || r > cut_hc) return 0;
    if (r < cut_lo)  return k * b_lo * (cut_lc - r) * (cut_lc - r);
    if (r < cut_hi)  return k * 0.5 * ((r - cut_0) * (r - cut_0) - (cut_0 - cut_c) * (cut_0 - cut_c));
    return k * b_hi * (cut_hc - r) * (cut_hc - r);
}

KOKKOS_INLINE_FUNCTION
static c_number DF2(c_number r, c_number k, c_number cut_0, c_number cut_lc,
                    c_number cut_hc, c_number cut_lo, c_number cut_hi,
                    c_number b_lo, c_number b_hi) {
    if (r < cut_lc || r > cut_hc) return 0;
    if (r < cut_lo)  return 2 * k * b_lo * (r - cut_lc);
    if (r < cut_hi)  return k * (r - cut_0);
    return 2 * k * b_hi * (r - cut_hc);
}

// F3: repulsive LJ + smooth tail. Returns potential energy, writes radial force
// to fpair (force / r, to be multiplied by displacement to get force vector).
KOKKOS_INLINE_FUNCTION
static c_number F3(c_number rsq, c_number cutsq_ast, c_number cut_c,
                   c_number lj1, c_number lj2, c_number eps, c_number b,
                   c_number &fpair) {
    c_number evdwl = 0;
    if (rsq < cutsq_ast) {
        c_number r2inv = 1 / rsq;
        c_number r6inv = r2inv * r2inv * r2inv;
        fpair = r2inv * r6inv * (12 * lj1 * r6inv - 6 * lj2);
        evdwl = r6inv * (lj1 * r6inv - lj2);
    } else {
        c_number r    = Kokkos::sqrt(rsq);
        c_number rinv = 1 / r;
        fpair = 2 * eps * b * (cut_c * rinv - 1);
        evdwl = eps * b * (cut_c - r) * (cut_c - r);
    }
    return evdwl;
}

// F4: angular modulation (cos-based)
KOKKOS_INLINE_FUNCTION
static c_number F4(c_number theta, c_number a, c_number theta_0,
                   c_number dtheta_ast, c_number b, c_number dtheta_c) {
    c_number dtheta = theta - theta_0;
    if (Kokkos::fabs(dtheta) > dtheta_c) return 0;
    if (dtheta > dtheta_ast)  return b * (dtheta - dtheta_c) * (dtheta - dtheta_c);
    if (dtheta > -dtheta_ast) return 1 - a * dtheta * dtheta;
    return b * (dtheta + dtheta_c) * (dtheta + dtheta_c);
}

// DF4: derivative of F4. The sin(theta) factor from d(cos theta)/d theta is
// handled externally — caller must multiply by sin(theta) / r as needed.
KOKKOS_INLINE_FUNCTION
static c_number DF4(c_number theta, c_number a, c_number theta_0,
                    c_number dtheta_ast, c_number b, c_number dtheta_c) {
    c_number dtheta = theta - theta_0;
    if (Kokkos::fabs(dtheta) > dtheta_c) return 0;
    if (dtheta > dtheta_ast)  return 2 * b * (dtheta - dtheta_c);
    if (dtheta > -dtheta_ast) return -2 * a * dtheta;
    return 2 * b * (dtheta + dtheta_c);
}

// F5: dihedral-type modulation
KOKKOS_INLINE_FUNCTION
static c_number F5(c_number x, c_number a, c_number x_ast,
                   c_number b, c_number x_c) {
    if (x >= 0)    return 1;
    if (x > x_ast) return 1 - a * x * x;
    if (x > x_c)   return b * (x - x_c) * (x - x_c);
    return 0;
}

KOKKOS_INLINE_FUNCTION
static c_number DF5(c_number x, c_number a, c_number x_ast,
                    c_number b, c_number x_c) {
    if (x >= 0)    return 0;
    if (x > x_ast) return -2 * a * x;
    if (x > x_c)   return 2 * b * (x - x_c);
    return 0;
}

// F6: harmonic penalty (used for backbone restoring)
KOKKOS_INLINE_FUNCTION
static c_number F6(c_number theta, c_number a, c_number b) {
    return (theta < b) ? 0 : 0.5 * a * (theta - b) * (theta - b);
}

KOKKOS_INLINE_FUNCTION
static c_number DF6(c_number theta, c_number a, c_number b) {
    return (theta < b) ? 0 : a * (theta - b);
}

} // namespace MFOxdna
