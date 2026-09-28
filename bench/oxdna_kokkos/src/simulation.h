#pragma once

#include "types.h"
#include "particles.h"
#include "neighbor_list.h"
#include "integrator.h"
#include "thermostat.h"
#include "forces/dna_forces.h"
#include "forces/bonded.h"
#include "forces/params.h"
#include "forces/dna3_forces.h"
#include "forces/params_dna3.h"
#include "io/topology_reader.h"
#include "io/config_reader.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

struct SimConfig {
    std::string topology_file;
    std::string config_file;
    std::string energy_file;          // if set, write oxDNA-style "time U K total" (per nucleotide)
    long long   nsteps      = 10000;
    c_number    dt          = 1e-3;
    c_number    T           = 0.1;
    c_number    cutoff      = 2.5;
    c_number    skin        = 0.3;
    int         output_freq = 1000;
    bool        timing      = false; // per-kernel breakdown (adds fences); off = production
    int         model       = 1;     // 1 = oxDNA1, 2 = oxDNA2, 3 = oxDNA3
    c_number    salt        = 0.5;   // salt concentration [mol/L] (oxDNA2/oxDNA3)
    // oxDNA3 only: sequence-dependence file (upstream key seq_dep_file) and
    // Debye-Huckel options (upstream DNA2Interaction keys)
    std::string seq_dep_file;        // empty -> default_dna3_seq_file()
    bool        use_average_seq = false;
    bool        dh_half_charged_ends = true;
    double      dh_lambda   = 0.3616455;
    double      dh_strength = 0.0543;
    double      dh_rhigh    = -1;    // < 0 -> 3 * Debye length
    bool        dna3_consistent_gamma = false;   // see DNA3Options::consistent_gamma
    bool        refresh_vel = false; // regenerate velocities from Maxwell-Boltzmann at startup
    // Brownian ("John") thermostat. newtonian_steps <= 0 disables it (NVE).
    int         newtonian_steps = 0;
    c_number    diff_coeff  = 2.5;   // translational diffusion coefficient
    c_number    pt          = 0.0;   // refresh probability (if >0, overrides diff_coeff)
    uint64_t    seed        = 12345;
};

// Default oxDNA3 parameter file: the copy shipped in params/ (absolute path
// baked in at configure time), falling back to paths relative to the working
// directory. An explicit `seq_dep_file` is used as given (relative to the
// working directory, as in the standalone oxDNA).
inline std::string default_dna3_seq_file() {
    const char *cands[] = {
#ifdef OXDNA_DEFAULT_SEQ_DEP_FILE
        OXDNA_DEFAULT_SEQ_DEP_FILE,
#endif
        "params/oxDNA3_sequence_dependent_parameters.txt",
        "oxDNA3_sequence_dependent_parameters.txt",
    };
    for (const char *c : cands) {
        std::ifstream t(c);
        if (t.good()) return c;
    }
    return "oxDNA3_sequence_dependent_parameters.txt";
}

inline DNA3Options dna3_options(const SimConfig &cfg) {
    DNA3Options o;
    o.T = cfg.T;
    o.salt = cfg.salt;
    o.average = cfg.use_average_seq;
    o.seq_file = cfg.seq_dep_file.empty() ? default_dna3_seq_file() : cfg.seq_dep_file;
    o.dh_half_charged_ends = cfg.dh_half_charged_ends;
    o.dh_lambda = cfg.dh_lambda;
    o.dh_strength = cfg.dh_strength;
    o.dh_rhigh = cfg.dh_rhigh;
    o.consistent_gamma = cfg.dna3_consistent_gamma;
    return o;
}

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

        // Regenerate velocities from Maxwell-Boltzmann at T (oxDNA refresh_vel).
        // Required for velocity-less configs and to match the reference, which
        // refreshes velocities at startup when refresh_vel = true.
        if (cfg_.refresh_vel)
            randomize_velocities(dev_, cfg_.T, cfg_.seed);

        // Force-field
        c_number cutsq_nb;
        if (cfg_.model == 3) {
            DNA3Options o = dna3_options(cfg_);
            if (!o.average) std::cout << "oxDNA3: sequence-dependent parameters from " << o.seq_file << "\n";
            par3_ = make_oxdna3_params(o);
            cutsq_nb = par3_.cutsq_nb;
        } else {
            par_ = (cfg_.model == 2) ? make_oxdna2_params(cfg_.T, cfg_.salt)
                                     : make_oxdna1_params(cfg_.T);
            cutsq_nb = par_.cutsq_nb;
        }

        // Thermostat (optional)
        thermo_.init(cfg_.T, cfg_.newtonian_steps, cfg_.dt, cfg_.diff_coeff,
                     cfg_.pt, cfg_.seed);

        // Neighbor list: cover the longest-range interaction (e.g. Debye-Huckel)
        c_number nl_cut = std::max(static_cast<double>(cfg_.cutoff),
                                   std::sqrt(static_cast<double>(cutsq_nb)));
        nl_.init(nl_cut, cfg_.skin, N_, box_);
        nl_.build(dev_, box_);

        // Initial forces: nonbonded (atomic scatter) first, then the bonded
        // gather kernel adds on top (each thread owns its particle, no atomics).
        dev_.zero_forces();
        epot_ = compute_forces(true);

        std::cout << "Precision: "
                  << (sizeof(c_number) == 4 ? "single (float)" : "double")
                  << " (" << (sizeof(c_number) * 8) << "-bit c_number)\n";
        std::cout << "Initialized " << N_ << " particles, "
                  << nl_.N_edges << " nonbonded pairs.\n";
    }

    void run() {
        // Per-section timers (LAMMPS-style breakdown). Each section boundary
        // fences only when -timing is on, so a production run (timing off) keeps
        // the kernels pipelined and reports the true loop time; the breakdown is
        // exact on CPU and adds one sync per section on GPU (like LAMMPS
        // `timer full`).
        double t_neigh = 0, t_bond = 0, t_nb = 0, t_mod = 0, t_out = 0;
        auto clk  = []{ return std::chrono::high_resolution_clock::now(); };
        auto sec  = [](auto a, auto b){ return std::chrono::duration<double>(b - a).count(); };
        auto mark = [&]{ if (cfg_.timing) Kokkos::fence(); return clk(); };

        long long step = step_;

        // Energy output matches the standalone oxDNA: energies are per nucleotide
        // and time = step * dt. stdout columns: "step time U K total"; the
        // optional energy_file gets oxDNA's "time U K total".
        std::ofstream efile;
        if (!cfg_.energy_file.empty()) efile.open(cfg_.energy_file);
        const double invN = (N_ > 0) ? 1.0 / N_ : 0.0;
        std::printf("# %10s %14s %14s %14s %14s\n", "step", "time", "U", "K", "total");
        auto emit = [&](long long st) {
            c_number ekin = kinetic_energy(dev_);   // reduction syncs
            double U = (double)epot_ * invN, K = (double)ekin * invN, tot = U + K;
            double time = (double)st * cfg_.dt;
            std::printf("%12lld %14.6f %14.6f %14.6f %14.6f\n", st, time, U, K, tot);
            if (efile) efile << std::fixed << std::setprecision(6)
                             << time << ' ' << U << ' ' << K << ' ' << tot << '\n';
        };

        // Initial (step 0) energy from forces computed in init()
        Kokkos::fence();
        emit(step);

        Kokkos::fence();
        auto loop0 = clk();
        for (long long s = 0; s < cfg_.nsteps; s++, step++) {
            auto a = clk();
            // Fused first step: integrate AND flag a rebuild if any particle has
            // moved > skin since the last build (sets nl_.d_needs_rebuild on
            // device), avoiding a separate full-N reduction every step.
            first_step(dev_, cfg_.dt, box_,
                       nl_.list_poss, nl_.d_needs_rebuild, nl_.rebuild_disp_sq());
            auto b = mark(); t_mod += sec(a, b);

            if (nl_.flag_is_set()) nl_.build(dev_, box_);
            auto c = mark(); t_neigh += sec(b, c);

            // Potential energy is only needed on output steps. Off those steps,
            // run the force kernels as plain parallel_for (no per-step reduction
            // kernel / device->host scalar copy), which keeps higher occupancy.
            const bool want_e = ((s + 1) % cfg_.output_freq == 0);

            dev_.zero_forces();
            epot_  = (cfg_.model == 3)
                   ? compute_nonbonded_forces_dna3(dev_, nl_, par3_, box_, want_e)
                   : compute_nonbonded_forces(dev_, nl_, par_, box_, want_e);
            auto e = mark(); t_nb += sec(c, e);

            epot_ += (cfg_.model == 3)
                   ? compute_bonded_forces_dna3(dev_, par3_, box_, want_e)
                   : compute_bonded_forces(dev_, par_, box_, want_e);
            auto f = mark(); t_bond += sec(e, f);

            second_step(dev_, cfg_.dt);
            if (cfg_.newtonian_steps > 0 && (step % cfg_.newtonian_steps == 0))
                thermo_.apply(dev_);
            auto g = mark(); t_mod += sec(f, g);

            if ((s + 1) % cfg_.output_freq == 0) emit(step + 1);
            auto h = mark(); t_out += sec(g, h);
        }
        Kokkos::fence();
        auto loop1 = clk();
        double loop_time = sec(loop0, loop1);

        print_performance(loop_time, t_neigh, t_bond, t_nb, t_mod, t_out);
    }

private:
    // Nonbonded (atomic scatter) first, then the bonded gather kernel on top.
    c_number compute_forces(bool want_e) {
        if (cfg_.model == 3) {
            c_number e = compute_nonbonded_forces_dna3(dev_, nl_, par3_, box_, want_e);
            return e + compute_bonded_forces_dna3(dev_, par3_, box_, want_e);
        }
        c_number e = compute_nonbonded_forces(dev_, nl_, par_, box_, want_e);
        return e + compute_bonded_forces(dev_, par_, box_, want_e);
    }

    void print_performance(double loop, double t_neigh, double t_bond,
                           double t_nb, double t_mod, double t_out) const {
        const long long nsteps = cfg_.nsteps;
        const int    nthreads = Kokkos::DefaultExecutionSpace().concurrency();
        const char  *backend  = Kokkos::DefaultExecutionSpace::name();

        const double tau_per_day = (loop > 0) ? (double)nsteps * cfg_.dt * 86400.0 / loop : 0.0;
        const double steps_per_s = (loop > 0) ? (double)nsteps / loop : 0.0;
        const double matomstep_s = (loop > 0) ? (double)nsteps * N_ / loop / 1e6 : 0.0;

        std::printf("\nLoop time of %g on 1 procs (%s x %d) for %lld steps with %d atoms\n",
                    loop, backend, nthreads, nsteps, N_);
        std::printf("\nPerformance: %.3f tau/day, %.3f timesteps/s, %.3f Matom-step/s\n",
                    tau_per_day, steps_per_s, matomstep_s);

        if (!cfg_.timing) {
            std::printf("(set 'timing = 1' in the input file for the per-kernel breakdown)\n");
            return;
        }

        const double sum   = t_neigh + t_bond + t_nb + t_mod + t_out;
        const double other = (loop > sum) ? (loop - sum) : 0.0;
        auto row = [&](const char *name, double t) {
            std::printf("%-22s | %10.4f | %6.2f | %10.3f\n",
                        name, t, loop > 0 ? 100.0 * t / loop : 0.0,
                        nsteps > 0 ? 1e6 * t / nsteps : 0.0);
        };
        std::printf("\nKernel timing breakdown:\n");
        std::printf("%-22s | %10s | %6s | %10s\n", "Section", "time (s)", "%loop", "us/step");
        std::printf("------------------------------------------------------------\n");
        row("Neigh",                   t_neigh);   // neighbor list build + rebuild check
        row("Pair: nonbonded",         t_nb);      // LAMMPS: Pair (excv/hbond/xstk/coax/dh)
        row("Bonded (FENE+excv+stk)",  t_bond);    // LAMMPS: Bond + bonded part of Pair
        row("Modify (integ+thermo)",   t_mod);     // LAMMPS: Modify (nve + thermostat)
        row("Output",                  t_out);
        row("Other",                   other);
        std::printf("------------------------------------------------------------\n");
        row("Total (loop)",            loop);
    }

public:

private:
    SimConfig        cfg_;
    int              N_   = 0;
    long long        step_= 0;
    SimBox           box_;
    ParticleArraysHost host_;
    ParticleArrays   dev_;
    DNAParams        par_;
    DNA3Params       par3_;
    NeighborList     nl_;
    Thermostat       thermo_;
    c_number         epot_= 0;
};
