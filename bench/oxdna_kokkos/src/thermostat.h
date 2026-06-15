#pragma once

// Brownian ("John") thermostat, matching the standalone oxDNA
// BrownianThermostat. Every `newtonian_steps` MD steps each particle's linear
// velocity is refreshed from the Maxwell-Boltzmann distribution with
// probability pt, and its angular momentum with probability pr; masses and the
// (isotropic) inertia are unity, so each component is drawn from N(0, sqrt(T)).
//
// The refresh probabilities are derived from the translational diffusion
// coefficient exactly as in the standalone (rotational Dr = 3 Dt):
//     pt = 2 T n dt / (T n dt + 2 D)        [or supplied directly]
//     D  = T n dt (1/pt - 1/2)
//     pr = 2 T n dt / (T n dt + 2 (3 D))
// where n = newtonian_steps. Disabled by default (newtonian_steps <= 0 -> NVE).

#include "types.h"
#include "particles.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstdint>
#include <stdexcept>

struct Thermostat {
    using Pool = Kokkos::Random_XorShift64_Pool<>;
    Pool     pool;
    c_number sqrtT = 0;
    c_number pt = 0, pr = 0;
    bool     enabled = false;

    // T          : target temperature
    // newt       : newtonian_steps (>0 to enable)
    // dt         : integration timestep
    // diff_coeff : translational diffusion coefficient (used if pt_in <= 0)
    // pt_in      : translational refresh probability (if > 0, used directly)
    void init(c_number T, int newt, c_number dt, c_number diff_coeff,
              c_number pt_in, uint64_t seed) {
        enabled = (newt > 0);
        if (!enabled) return;
        pool  = Pool(seed);
        sqrtT = Kokkos::sqrt(T);                 // unit mass and inertia

        c_number Tndt = T * newt * dt;
        pt = pt_in;
        if (pt <= 0) pt = (2 * Tndt) / (Tndt + 2 * diff_coeff);
        if (pt > 1) throw std::runtime_error("Brownian thermostat: pt > 1 (reduce diff_coeff or dt)");
        // back out the diffusion coefficient consistent with pt, then pr (Dr = 3 Dt)
        c_number D = Tndt * (1 / pt - c_number(0.5));
        pr = (2 * Tndt) / (Tndt + 2 * 3 * D);
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
