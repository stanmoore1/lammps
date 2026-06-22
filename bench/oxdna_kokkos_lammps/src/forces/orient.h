#pragma once

// Quaternion → orientation basis vectors.
// Computes the 3×3 rotation matrix rows (nx, ny, nz) from a unit quaternion
// (q0=w, q1=x, q2=y, q3=z).  Uses fused-multiply-add for numerical stability
// on FP32 — same formula as fix_oxdna_lrf_kokkos.cpp:171-188.

#include "../types.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

KOKKOS_INLINE_FUNCTION
void get_vectors_from_quat(c_number q0, c_number q1, c_number q2, c_number q3,
                           c_number (&nx)[3], c_number (&ny)[3], c_number (&nz)[3]) {
    const c_number two = 2;

    // nx = first row of rotation matrix
    nx[0] = Kokkos::fma(q0, q0, Kokkos::fma(q1, q1, -Kokkos::fma(q2, q2, q3 * q3)));
    nx[1] = two * Kokkos::fma(q1, q2,  q0 * q3);
    nx[2] = two * Kokkos::fma(q1, q3, -q0 * q2);

    // ny = second row
    ny[0] = two * Kokkos::fma(q1, q2, -q0 * q3);
    ny[1] = Kokkos::fma(q0, q0, Kokkos::fma(q2, q2, -Kokkos::fma(q1, q1, q3 * q3)));
    ny[2] = two * Kokkos::fma(q2, q3,  q0 * q1);

    // nz = third row
    nz[0] = two * Kokkos::fma(q1, q3,  q0 * q2);
    nz[1] = two * Kokkos::fma(q2, q3, -q0 * q1);
    nz[2] = Kokkos::fma(q0, q0, q3 * q3 - Kokkos::fma(q1, q1, q2 * q2));
}

// Convenience overload loading from a Kokkos::View<c_number*[4]> row
template <typename ViewType>
KOKKOS_INLINE_FUNCTION
void get_vectors_from_quat_view(const ViewType &ori, int i,
                                c_number (&nx)[3], c_number (&ny)[3], c_number (&nz)[3]) {
    get_vectors_from_quat(ori(i, 0), ori(i, 1), ori(i, 2), ori(i, 3), nx, ny, nz);
}
