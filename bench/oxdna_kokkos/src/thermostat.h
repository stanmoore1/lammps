#pragma once

// Andersen-style ("John") thermostat, as used by the standalone oxDNA MD
// backend. Every `newtonian_steps` MD steps the thermostat is applied: each
// particle's linear velocity is refreshed from the Maxwell-Boltzmann
// distribution with probability pt, and its angular momentum with probability
// pr. Masses and the (isotropic) inertia are unity, so each component is drawn
// from N(0, sqrt(T)).
//
// Disabled by default (newtonian_steps <= 0) so the engine runs NVE, which is
// the configuration used for force-throughput benchmarking.

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstdint>

struct Thermostat {
    using Pool = Kokkos::Random_XorShift64_Pool<>;
    Pool     pool;
    c_number sqrtT = 0;
    c_number pt = 0, pr = 0;
    bool     enabled = false;

    void init(c_number T, c_number pt_in, c_number pr_in, uint64_t seed) {
        pool    = Pool(seed);
        sqrtT   = Kokkos::sqrt(T);   // unit mass and unit inertia
        pt      = pt_in;
        pr      = pr_in;
        enabled = (pt_in > 0 || pr_in > 0);
    }

    void apply(ParticleArrays &p) const {
        if (!enabled) return;
        auto vels = p.vels;
        auto Ls   = p.Ls;
        Pool pool_ = pool;
        c_number sT = sqrtT, pt_ = pt, pr_ = pr;
        Kokkos::parallel_for("thermostat", p.N, KOKKOS_LAMBDA(int i) {
            auto gen = pool_.get_state();
            if (gen.drand() < pt_) {
                vels(i,0) = sT * gen.normal();
                vels(i,1) = sT * gen.normal();
                vels(i,2) = sT * gen.normal();
            }
            if (gen.drand() < pr_) {
                Ls(i,0) = sT * gen.normal();
                Ls(i,1) = sT * gen.normal();
                Ls(i,2) = sT * gen.normal();
            }
            pool_.free_state(gen);
        });
    }
};
