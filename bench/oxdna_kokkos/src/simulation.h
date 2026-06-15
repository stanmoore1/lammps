#pragma once

#include "types.h"
#include "particles.h"
#include "neighbor_list.h"
#include "integrator.h"
#include "thermostat.h"
#include "forces/dna_forces.h"
#include "forces/backbone.h"
#include "forces/stacking.h"
#include "forces/params.h"
#include "io/topology_reader.h"
#include "io/config_reader.h"
#include <chrono>
#include <iostream>
#include <string>

struct SimConfig {
    std::string topology_file;
    std::string config_file;
    long long   nsteps      = 10000;
    c_number    dt          = 1e-3;
    c_number    T           = 0.1;
    c_number    cutoff      = 2.5;
    c_number    skin        = 0.3;
    int         output_freq = 1000;
    // Thermostat (Andersen / "John"). newtonian_steps <= 0 disables it (NVE).
    int         newtonian_steps = 0;
    c_number    pt          = 0.1;   // translational refresh probability
    c_number    pr          = 0.1;   // rotational refresh probability
    uint64_t    seed        = 12345;
};

class Simulation {
public:
    explicit Simulation(const SimConfig &cfg) : cfg_(cfg) {}

    void init() {
        // I/O
        read_topology(cfg_.topology_file, host_, N_);
        read_config(cfg_.config_file, host_, box_, step_);

        // Device arrays
        dev_.allocate(N_);
        copy_to_device(host_, dev_);

        // Force-field
        par_ = make_oxdna1_params(cfg_.T);

        // Thermostat (optional)
        if (cfg_.newtonian_steps > 0)
            thermo_.init(cfg_.T, cfg_.pt, cfg_.pr, cfg_.seed);

        // Neighbor list
        nl_.init(cfg_.cutoff, cfg_.skin, N_, box_);
        nl_.build(dev_, box_);

        // Initial forces
        dev_.zero_forces();
        epot_ = compute_backbone_forces(dev_, par_, box_);
        epot_ += compute_stacking_forces(dev_, par_, box_);
        epot_ += compute_nonbonded_forces(dev_, nl_, par_, box_);

        std::cout << "Initialized " << N_ << " particles, "
                  << nl_.N_edges << " nonbonded pairs.\n";
    }

    void run() {
        auto t_start = std::chrono::high_resolution_clock::now();
        long long step = step_;

        // Print initial (step 0) energy from forces computed in init()
        {
            Kokkos::fence();
            c_number ekin = kinetic_energy(dev_);
            std::printf("step %8lld  Epot=%12.6f  Ekin=%12.6f  Etot=%12.6f  pairs=%d\n",
                        step, (double)epot_, (double)ekin, (double)(ekin + epot_),
                        nl_.N_edges);
        }

        for (long long s = 0; s < cfg_.nsteps; s++, step++) {
            first_step(dev_, cfg_.dt, box_);

            if (nl_.needs_rebuild(dev_, box_)) {
                nl_.build(dev_, box_);
            }

            dev_.zero_forces();
            epot_  = compute_backbone_forces(dev_, par_, box_);
            epot_ += compute_stacking_forces(dev_, par_, box_);
            epot_ += compute_nonbonded_forces(dev_, nl_, par_, box_);

            second_step(dev_, cfg_.dt);

            if (cfg_.newtonian_steps > 0 && (step % cfg_.newtonian_steps == 0))
                thermo_.apply(dev_);

            if (s % cfg_.output_freq == 0) {
                Kokkos::fence();
                c_number ekin = kinetic_energy(dev_);
                c_number etot = ekin + epot_;
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                double ns_per_day = (double)s * cfg_.dt * 86400.0 / elapsed;
                std::printf("step %8lld  Epot=%12.6f  Ekin=%12.6f  Etot=%12.6f  "
                            "pairs=%d  ns/day=%.2f\n",
                            step, (double)epot_, (double)ekin, (double)etot,
                            nl_.N_edges, ns_per_day);
            }
        }

        Kokkos::fence();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        double ns_per_day = (double)cfg_.nsteps * cfg_.dt * 86400.0 / elapsed;
        std::printf("\nFinished %lld steps in %.3f s (%.2f ns/day)\n",
                    cfg_.nsteps, elapsed, ns_per_day);
    }

private:
    SimConfig        cfg_;
    int              N_   = 0;
    long long        step_= 0;
    SimBox           box_;
    ParticleArraysHost host_;
    ParticleArrays   dev_;
    DNAParams        par_;
    NeighborList     nl_;
    Thermostat       thermo_;
    c_number         epot_= 0;
};
