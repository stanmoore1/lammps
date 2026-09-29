#pragma once

// Brownian ("John") thermostat, a port of the oxDNA CUDABrownianThermostat
// (src/CUDA/Thermostats/CUDABrownianThermostat.cu, CUDABaseThermostat.cu):
//   * one persistent RNG state per particle (curandState = XORWOW, 48 bytes),
//     initialised once by a setup kernel, loaded and stored by the thermostat
//     kernel of every thermostat step;
//   * the kernel (launch cfg: threads_per_block, ceil(N / tpb) blocks) runs on
//     steps with curr_step % newtonian_steps == 0; per particle it draws a
//     uniform number and, if < pt, refreshes v with two Box-Muller pairs
//     (x, y) and (z, trash); then the same for L with probability pr;
//     refreshed vectors store |v|^2/2 in .w;
//   * pt / diff_coeff / dt are read as float (getInputFloat), then
//       pt = 2 T n dt / (T n dt + 2 D)   (if pt is not given)
//       D  = T n dt (1/pt - 1/2),  pr = 2 T n dt / (T n dt + 2 (3 D)),
//     rescale factor sqrt(T) (unit mass and inertia);
//   * in mixed precision the kernel works on the float velocity / angular
//     momentum arrays: double -> float conversions before and float -> double
//     after it (CUDAMixedBackend::_thermalize), i.e. every thermostat step
//     rounds all velocities to float, as upstream.
//
// The XORWOW generator, curand_uniform() and the Box-Muller gaussian() are
// bit-level ports of curand's. Not mirrored: curand_init(seed, IND, 0) puts
// particle IND on subsequence IND (skip-ahead by IND * 2^67 with curand's
// precomputed jump matrices); here each particle's state is initialised with
// curand_init's scrambling of a per-particle seed (splitmix64 of seed and IND)
// instead -- the same generator and cost, statistically independent streams,
// not bit-identical to curand. The upstream seed itself comes from lrand48().
//
// refresh_vel (MDBackend::_generate_vel) runs on the host as upstream:
// srand48(seed), then per particle vx, vy, vz, Lx, Ly, Lz from the Marsaglia
// polar method of Utils::gaussian() (drand48), times sqrt(T); with the same
// seed this reproduces the initial velocities of the standalone oxDNA.

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// ---------------------------------------------------------------------------
// curandStateXORWOW layout (48 bytes)
// ---------------------------------------------------------------------------
struct XorwowState {
    uint32_t d, v[5];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};
static_assert(sizeof(XorwowState) == 48, "XorwowState must match curandStateXORWOW (48 bytes)");

KOKKOS_INLINE_FUNCTION uint32_t xorwow_next(XorwowState &s) {
    uint32_t t = s.v[0] ^ (s.v[0] >> 2);
    s.v[0] = s.v[1];
    s.v[1] = s.v[2];
    s.v[2] = s.v[3];
    s.v[3] = s.v[4];
    s.v[4] = (s.v[4] ^ (s.v[4] << 4)) ^ (t ^ (t << 1));
    s.d += 362437;
    return s.v[4] + s.d;
}

// curand_uniform: (0, 1]
KOKKOS_INLINE_FUNCTION float curand_uniform(XorwowState &s) {
    constexpr float CURAND_2POW32_INV = 2.3283064e-10f;
    return xorwow_next(s) * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
}

// _curand_init_scratch (seed scrambling), without the subsequence skip-ahead
KOKKOS_INLINE_FUNCTION void xorwow_init(uint64_t seed, XorwowState &s) {
    const uint32_t s0 = static_cast<uint32_t>(seed) ^ 0xaad26b49UL;
    const uint32_t s1 = static_cast<uint32_t>(seed >> 32) ^ 0xf7dcefddUL;
    const uint32_t t0 = 1099087573UL * s0;
    const uint32_t t1 = 2591861531UL * s1;
    s.d = 6615241 + t1 + t0;
    s.v[0] = 123456789UL + t0;
    s.v[1] = 362436069UL ^ t0;
    s.v[2] = 521288629UL + t1;
    s.v[3] = 88675123UL ^ t1;
    s.v[4] = 5783321UL + t0;
    s.boxmuller_flag = 0;
    s.boxmuller_flag_double = 0;
    s.boxmuller_extra = 0.f;
    s.boxmuller_extra_double = 0.;
}

KOKKOS_INLINE_FUNCTION uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// gaussian() of CUDA_lr_common.cuh (float and double overloads)
KOKKOS_INLINE_FUNCTION void gaussian(XorwowState &s, float &outx, float &outy) {
    const float r = Kokkos::sqrt(-2.0f * Kokkos::log(curand_uniform(s)));
    const float phi = 2.f * 3.141592653589793f * curand_uniform(s);
#if defined(__CUDA_ARCH__)
    outx = r * __cosf(phi);   // fast intrinsics, as upstream
    outy = r * __sinf(phi);
#else
    outx = r * Kokkos::cos(phi);
    outy = r * Kokkos::sin(phi);
#endif
}
KOKKOS_INLINE_FUNCTION void gaussian(XorwowState &s, double &outx, double &outy) {
    const double r = Kokkos::sqrt(-2. * Kokkos::log(static_cast<double>(curand_uniform(s))));
    const double phi = 2 * 3.14159265358979323846 * curand_uniform(s);
    outx = r * Kokkos::cos(phi);
    outy = r * Kokkos::sin(phi);
}

// ---------------------------------------------------------------------------
// refresh_vel on the host (MDBackend::_generate_vel with Utils::gaussian)
// ---------------------------------------------------------------------------
struct HostGaussian {
    bool has_next = false;
    double next = 0;
    double operator()() {
        if (has_next) { has_next = false; return next; }
        double u, v, w = 2.;
        while (w >= 1.0) {
            u = 2. * drand48() - 1.0;
            v = 2. * drand48() - 1.0;
            w = u * u + v * v;
        }
        w = std::sqrt((-2. * std::log(w)) / w);
        next = v * w;
        has_next = true;
        return u * w;
    }
};

inline void generate_velocities_host(ParticleArraysHost &h, double T, long seed) {
    srand48(seed);
    HostGaussian g;
    const double rf = std::sqrt(T);
    for (int i = 0; i < h.N; i++) {
        h.vels(i, 0) = static_cast<c_number>(g() * rf);
        h.vels(i, 1) = static_cast<c_number>(g() * rf);
        h.vels(i, 2) = static_cast<c_number>(g() * rf);
        h.Ls(i, 0) = static_cast<c_number>(g() * rf);
        h.Ls(i, 1) = static_cast<c_number>(g() * rf);
        h.Ls(i, 2) = static_cast<c_number>(g() * rf);
    }
}

// ---------------------------------------------------------------------------
// CUDABrownianThermostat
// ---------------------------------------------------------------------------
struct BrownianThermostatKernel {
    Kokkos::View<XorwowState *> rand_state;
    Vec4 vels, Ls;
    c_number rescale_factor, pt, pr;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        XorwowState state = rand_state(i);
        if (curand_uniform(state) < pt) {
            c_number vx, vy, vz, trash;
            gaussian(state, vx, vy);
            gaussian(state, vz, trash);
            vx *= rescale_factor; vy *= rescale_factor; vz *= rescale_factor;
            vels(i, 0) = vx; vels(i, 1) = vy; vels(i, 2) = vz;
            vels(i, 3) = (vx * vx + vy * vy + vz * vz) * c_number(0.5f);
        }
        if (curand_uniform(state) < pr) {
            c_number lx, ly, lz, trash;
            gaussian(state, lx, ly);
            gaussian(state, lz, trash);
            lx *= rescale_factor; ly *= rescale_factor; lz *= rescale_factor;
            Ls(i, 0) = lx; Ls(i, 1) = ly; Ls(i, 2) = lz;
            Ls(i, 3) = (lx * lx + ly * ly + lz * lz) * c_number(0.5f);
        }
        rand_state(i) = state;
    }
};

struct Thermostat {
    Kokkos::View<XorwowState *> rand_state;
    c_number rescale_factor = 0;
    c_number pt = 0, pr = 0;
    int      newtonian_steps = 0;
    bool     enabled = false;

    // T          : target temperature
    // newt       : newtonian_steps (>0 to enable)
    // dt         : integration timestep
    // diff_coeff : translational diffusion coefficient (used if pt_in <= 0)
    // pt_in      : translational refresh probability (if > 0, used directly)
    void init(double T, int newt, double dt, double diff_coeff, double pt_in,
              uint64_t seed, int N) {
        enabled = (newt > 0);
        if (!enabled) return;
        newtonian_steps = newt;
        // BrownianThermostat::get_settings reads pt, diff_coeff, dt as float
        double pt_d = static_cast<float>(pt_in > 0 ? pt_in : 0.0);
        const double D_in = static_cast<float>(diff_coeff);
        const double dt_d = static_cast<float>(dt);
        const double Tndt = T * newt * dt_d;
        if (pt_d == 0.) pt_d = (2 * Tndt) / (Tndt + 2 * D_in);
        if (pt_d > 1.) throw std::runtime_error("Brownian thermostat: pt > 1 (reduce diff_coeff or dt)");
        const double D = Tndt * (1. / pt_d - 1. / 2.);
        const double pr_d = (2 * Tndt) / (Tndt + 2 * 3 * D);
        pt = static_cast<c_number>(pt_d);
        pr = static_cast<c_number>(pr_d);
        rescale_factor = static_cast<c_number>(std::sqrt(T));

        // _setup_rand: setup_curand kernel, one state per particle
        rand_state = Kokkos::View<XorwowState *>("curand_states", N);
        auto rs = rand_state;
        Kokkos::parallel_for("setup_curand", OxPolicy(0, N), KOKKOS_LAMBDA(int i) {
            XorwowState s;
            xorwow_init(splitmix64(seed ^ splitmix64(static_cast<uint64_t>(i))), s);
            rs(i) = s;
        });
    }

    bool would_activate(long long curr_step) const {
        return enabled && (curr_step % newtonian_steps == 0);
    }

    // CUDABrownianThermostat::apply_cuda on the (float) vels / Ls arrays
    void apply_kernel(ParticleArrays &p) const {
        BrownianThermostatKernel k{rand_state, p.vels, p.Ls, rescale_factor, pt, pr};
        Kokkos::parallel_for("brownian_thermostat", OxPolicy(0, p.N), k);
    }

    // MD_CUDABackend::_thermalize / CUDAMixedBackend::_thermalize
    void thermalize(ParticleArrays &p, long long curr_step) const {
        if (!would_activate(curr_step)) return;
        if constexpr (OXDNA_MIXED) {
            double4_to_float4(p.velsd, p.vels);
            double4_to_float4(p.Lsd, p.Ls);
            apply_kernel(p);
            float4_to_double4(p.vels, p.velsd);
            float4_to_double4(p.Ls, p.Lsd);
        } else {
            apply_kernel(p);
        }
    }
};
