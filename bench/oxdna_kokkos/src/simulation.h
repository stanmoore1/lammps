#pragma once

// MD driver, structured as the standalone oxDNA CUDA MD run it mirrors:
// SimManager::run (main loop, observables, fix_diffusion) around
// MD_CUDABackend::sim_step / CUDAMixedBackend (see README, "Fidelity to oxDNA
// CUDA").
//
// Per step (sim_step):
//   first step kernel (integrate + flag "lists old" into a host-pinned int)
//     [timer "First Step" -> device sync]
//   if lists old and CUDA_sort_every > 0 and N_updates % sort_every == 0:
//     Hilbert sort                                  [timer "Hilbert sorting" -> sync]
//   if lists old: Verlet-list update, flag = 0, N_updates++   [timer "Lists" -> sync]
//   set_external_forces (zero F, T), force kernel(s), second step kernel
//     (+ CUDA_print_energy: device sum of F.w, host read)     [timer "Forces" -> sync]
//   thermostat (if curr_step % newtonian_steps == 0)          [timer "Thermostat" -> sync]
// Outside sim_step, before it (SimManager::run): every fix_diffusion_every
// steps fix_diffusion (full device->host copy, strand COMs back into the box,
// host->device copy), and on steps with curr_step % print_energy_every == 0
// the energy output, preceded by the full device->host copy of
// apply_simulation_data_changes().
//
// Upstream every timer pause calls cudaDeviceSynchronize() (TimingManager::
// enable_sync() in the MD_CUDABackend constructor); here each section ends
// with Kokkos::fence() unless `timer_sync = 0`.

#include "types.h"
#include "particles.h"
#include "neighbor_list.h"
#include "integrator.h"
#include "thermostat.h"
#include "sort.h"
#include "forces/dna_forces.h"
#include "forces/bonded.h"
#include "forces/params.h"
#include "forces/dna3_forces.h"
#include "forces/params_dna3.h"
#include "io/topology_reader.h"
#include "io/config_reader.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

struct SimConfig {
    std::string topology_file;
    std::string config_file;
    std::string energy_file;          // if set, write oxDNA-style "time U K total" (per nucleotide)
    long long   nsteps      = 10000;
    double      dt          = 1e-3;
    double      T           = 0.1;
    double      skin        = 0.3;
    long long   output_freq = 1000;   // print_energy_every
    bool        timing      = false;  // print the per-section breakdown
    bool        timer_sync  = true;   // fence at every section end (oxDNA synchronised timers)
    int         model       = 1;      // 1 = oxDNA1, 2 = oxDNA2, 3 = oxDNA3
    double      salt        = 0.5;    // salt concentration [mol/L] (oxDNA2/oxDNA3)
    // oxDNA3 only: sequence-dependence file (upstream key seq_dep_file) and
    // Debye-Huckel options (upstream DNA2Interaction keys)
    std::string seq_dep_file;         // empty -> default_dna3_seq_file()
    bool        use_average_seq = false;
    bool        dh_half_charged_ends = true;
    double      dh_lambda   = 0.3616455;
    double      dh_strength = 0.0543;
    double      dh_rhigh    = -1;     // < 0 -> 3 * Debye length
    bool        dna3_consistent_gamma = false;   // see DNA3Options::consistent_gamma
    bool        refresh_vel = false;  // regenerate velocities from Maxwell-Boltzmann at startup
    bool        reset_initial_com_momentum = false;
    bool        restart_step_counter = false;
    // Brownian ("John") thermostat. newtonian_steps <= 0 disables it (NVE).
    int         newtonian_steps = 0;
    double      diff_coeff  = 2.5;    // translational diffusion coefficient
    double      pt          = 0.0;    // refresh probability (if >0, overrides diff_coeff)
    uint64_t    seed        = 12345;
    // oxDNA CUDA backend options (upstream defaults)
    bool        use_edge    = false;  // use_edge
    int         sort_every  = 0;      // CUDA_sort_every
    bool        print_energy_gpu = false;   // CUDA_print_energy
    bool        fix_diffusion = true; // fix_diffusion
    long long   fix_diffusion_every = 100000;
    double      max_density_multiplier = 3;
    bool        cells_auto_optimisation = true;
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

// Device sum of F.w in double (GpuUtils::sum_c_number4_to_double_on_GPU)
inline double sum_forces_w(const Vec4 &F, int N) {
    double e = 0;
    Kokkos::parallel_reduce("sum_c_number4_to_double", N,
        KOKKOS_LAMBDA(int i, double &s) { s += static_cast<double>(F(i, 3)); }, e);
    return e;
}

class Simulation {
public:
    explicit Simulation(const SimConfig &cfg) : cfg_(cfg) {}

    void init() {
        // I/O (SimBackend::init: topology, configuration with the strand shift
        // of fix_diffusion, orientation orthonormalisation)
        read_topology(cfg_.topology_file, host_, N_);
        long long conf_step = 0;
        read_config(cfg_.config_file, host_, box_, conf_step, cfg_.fix_diffusion);
        curr_step_ = cfg_.restart_step_counter ? 0 : conf_step;

        // SimManager: srand48(seed); MDBackend::init: refresh_vel on the host
        srand48(static_cast<long>(cfg_.seed));
        if (cfg_.refresh_vel) {
            generate_velocities_host(host_, cfg_.T, static_cast<long>(cfg_.seed));
        } else {
            for (int i = 0; i < N_; i++) {
                const double l2 = (double)host_.Ls(i, 0) * host_.Ls(i, 0) + (double)host_.Ls(i, 1) * host_.Ls(i, 1)
                                + (double)host_.Ls(i, 2) * host_.Ls(i, 2);
                if (std::sqrt(l2) < 1.e-10)
                    throw std::runtime_error("Particle " + std::to_string(i) + " has 0 angular momentum in the "
                                             "initial configuration: set refresh_vel = true (as upstream)");
            }
        }
        if (cfg_.reset_initial_com_momentum) {
            double c[3] = {0, 0, 0};
            for (int i = 0; i < N_; i++) for (int d = 0; d < 3; d++) c[d] += host_.vels(i, d);
            for (int i = 0; i < N_; i++) for (int d = 0; d < 3; d++)
                host_.vels(i, d) = static_cast<c_number>(host_.vels(i, d) - c[d] / N_);
        }

        // Device arrays (apply_changes_to_simulation_data: host -> device,
        // then in mixed precision float -> double copies)
        dev_.allocate(N_);
        copy_to_device(host_, dev_);

        // Force-field
        double rcut;
        if (cfg_.model == 3) {
            DNA3Options o = dna3_options(cfg_);
            if (!o.average) std::cout << "oxDNA3: sequence-dependent parameters from " << o.seq_file << "\n";
            par3_ = make_oxdna3_params(o);
            rcut = par3_.rcut;
        } else {
            par_ = (cfg_.model == 2) ? make_oxdna2_params(cfg_.T, cfg_.salt)
                                     : make_oxdna1_params(cfg_.T);
            rcut = par_.rcut;
        }

        md_.dt = static_cast<float>(cfg_.dt);
        md_.sqr_verlet_skin = static_cast<float>(static_cast<c_number>(cfg_.skin * cfg_.skin));

        // thermostat (seed: lrand48(), as MD_CUDABackend::init)
        thermo_.init(cfg_.T, cfg_.newtonian_steps, cfg_.dt, cfg_.diff_coeff, cfg_.pt,
                     static_cast<uint64_t>(lrand48()), N_);

        are_lists_old_ = PinnedFlag("are_lists_old");
        are_lists_old_() = 0;

        // CUDA lists: init + first update, set_external_forces, forces
        nl_.max_density_multiplier = cfg_.max_density_multiplier;
        nl_.auto_optimisation = cfg_.cells_auto_optimisation;
        nl_.init(N_, rcut, cfg_.skin, box_, dev_.poss, cfg_.use_edge);
        if (cfg_.sort_every > 0) sorter_.init(N_, box_);
        nl_.update(dev_, box_);
        set_external_forces(dev_);
        compute_forces(want_energy_at(curr_step_));
        if (cfg_.print_energy_gpu && !cfg_.use_edge)   // (upstream: first value after the first step)
            cuda_energy_ = sum_forces_w(dev_.forces, N_) / (2. * N_);
        Kokkos::fence();

        std::cout << "Precision: " << oxdna_precision_name() << " (backend_precision = "
                  << oxdna_precision_name() << "; force kernels " << (sizeof(c_number) * 8)
                  << "-bit, integrator " << (sizeof(m_number) * 8) << "-bit)\n";
        std::cout << "Forces: " << (cfg_.use_edge ? "edge-based (use_edge = true)" : "per-particle (use_edge = false)")
                  << ", threads_per_block = " << OXDNA_THREADS_PER_BLOCK
                  << ", CUDA_sort_every = " << cfg_.sort_every << "\n";
        std::cout << "Verlet list: rcut = " << rcut << ", rverlet = " << rcut + 2 * cfg_.skin
                  << ", cells " << nl_.N_cells_side[0] << "x" << nl_.N_cells_side[1] << "x" << nl_.N_cells_side[2]
                  << ", max_N_per_cell = " << nl_.max_N_per_cell << ", max_neigh = " << nl_.max_neigh << "\n";
        std::cout << "Initialized " << N_ << " particles";
        if (cfg_.use_edge) std::cout << ", " << nl_.N_edges << " edges";
        std::cout << ".\n";
    }

    void run() {
        double t_first = 0, t_sort = 0, t_lists = 0, t_forces = 0, t_thermo = 0, t_out = 0;
        auto clk  = []{ return std::chrono::high_resolution_clock::now(); };
        auto sec  = [](auto a, auto b){ return std::chrono::duration<double>(b - a).count(); };
        auto mark = [&]{ if (cfg_.timer_sync) Kokkos::fence(); return clk(); };

        std::ofstream efile;
        if (!cfg_.energy_file.empty()) efile.open(cfg_.energy_file);
        const double invN = (N_ > 0) ? 1.0 / N_ : 0.0;
        std::printf("# %10s %14s %14s %14s %14s%s\n", "step", "time", "U", "K", "total",
                    cfg_.print_energy_gpu ? "    CUDA_energy" : "");

        // print_observables(): apply_simulation_data_changes (full copy to the
        // host), then the energies are evaluated on the host copies
        auto print_observables = [&](long long st) {
            if (st % cfg_.output_freq != 0) return;
            copy_to_host(dev_, host_);
            const double K = kinetic_energy_host(host_);
            double U;
            if (cfg_.use_edge) {
                U = epot_edge_;
            } else {
                U = 0;
                for (int i = 0; i < N_; i++) U += (double)host_.forces(i, 3);
                U *= 0.5;
            }
            U *= invN;
            const double Kn = K * invN, tot = U + Kn, time = (double)st * cfg_.dt;
            if (cfg_.print_energy_gpu)
                std::printf("%12lld %14.6f %14.6f %14.6f %14.6f %14.6f\n", st, time, U, Kn, tot, cuda_energy_);
            else
                std::printf("%12lld %14.6f %14.6f %14.6f %14.6f\n", st, time, U, Kn, tot);
            if (efile) efile << std::fixed << std::setprecision(6)
                             << time << ' ' << U << ' ' << Kn << ' ' << tot << '\n';
        };

        Kokkos::fence();
        auto loop0 = clk();
        long long steps_run = 0;
        for (; steps_run < cfg_.nsteps; steps_run++) {
            auto a = clk();
            if (steps_run > 0 && steps_run % cfg_.fix_diffusion_every == 0) fix_diffusion();
            print_observables(curr_step_);
            auto b = mark(); t_out += sec(a, b);

            // ---- sim_step ----
            first_step(dev_, md_, nl_.list_poss, are_lists_old_);
            Kokkos::fence();   // timer sync; also needed to read the pinned flag
            const bool lists_old = (are_lists_old_() != 0);
            auto c2 = clk(); t_first += sec(b, c2);

            if (lists_old && cfg_.sort_every > 0 && (nl_.N_updates % cfg_.sort_every == 0))
                sorter_.sort_particles(dev_);
            auto d = mark(); t_sort += sec(c2, d);

            if (lists_old) {
                nl_.update(dev_, box_);
                are_lists_old_() = 0;
                nl_.N_updates++;
            }
            auto e = mark(); t_lists += sec(d, e);

            set_external_forces(dev_);
            compute_forces(want_energy_at(curr_step_ + 1));
            second_step(dev_, md_);
            if (cfg_.print_energy_gpu && !cfg_.use_edge)
                cuda_energy_ = sum_forces_w(dev_.forces, N_) / (2. * N_);
            auto f = mark(); t_forces += sec(e, f);

            thermo_.thermalize(dev_, curr_step_);
            auto g = mark(); t_thermo += sec(f, g);

            curr_step_++;
        }
        {
            auto a = clk();
            if (steps_run > 1 && steps_run % cfg_.fix_diffusion_every == 0) fix_diffusion();
            print_observables(curr_step_);
            Kokkos::fence();
            t_out += sec(a, clk());
        }
        Kokkos::fence();
        auto loop1 = clk();
        double loop_time = sec(loop0, loop1);

        print_performance(loop_time, t_first, t_sort, t_lists, t_forces, t_thermo, t_out);
    }

private:
    bool want_energy_at(long long st) const {
        return cfg_.use_edge && (st % cfg_.output_freq == 0);
    }

    // CUDA interaction compute_forces (after set_external_forces)
    void compute_forces(bool want_energy) {
        if (!cfg_.use_edge) {
            if (cfg_.model == 3) compute_forces_per_particle_dna3(dev_, nl_, par3_, box_, true);
            else                 compute_forces_per_particle(dev_, nl_, par_, box_, true);
            return;
        }
        // edge-based: nonbonded edge kernel (atomics), then the bonded gather
        // kernel on top, which rotates the torques into the body frame
        c_number e;
        if (cfg_.model == 3) {
            e  = compute_nonbonded_forces_dna3(dev_, nl_, par3_, box_, want_energy);
            e += compute_bonded_forces_dna3(dev_, par3_, want_energy, true);
        } else {
            e  = compute_nonbonded_forces(dev_, nl_, par_, box_, want_energy);
            e += compute_bonded_forces(dev_, par_, want_energy, true);
        }
        if (want_energy) epot_edge_ = e;
    }

    // SimBackend::fix_diffusion (CPU energy checks not mirrored)
    void fix_diffusion() {
        if (!cfg_.fix_diffusion) return;
        copy_to_host(dev_, host_);
        shift_strands_into_box(host_, box_);
        for (int i = 0; i < N_; i++) {   // orientation.orthonormalize()
            double n = 0;
            for (int c = 0; c < 4; c++) n += (double)host_.orientations(i, c) * host_.orientations(i, c);
            n = std::sqrt(n);
            for (int c = 0; c < 4; c++) host_.orientations(i, c) = static_cast<c_number>(host_.orientations(i, c) / n);
        }
        Kokkos::deep_copy(dev_.poss, host_.poss);
        Kokkos::deep_copy(dev_.bonds, host_.bonds);
        Kokkos::deep_copy(dev_.orientations, host_.orientations);
        Kokkos::deep_copy(dev_.strand, host_.strand);
        Kokkos::deep_copy(dev_.vels, host_.vels);
        Kokkos::deep_copy(dev_.Ls, host_.Ls);
        mixed_float_to_double(dev_);
    }

    void print_performance(double loop, double t_first, double t_sort, double t_lists,
                           double t_forces, double t_thermo, double t_out) const {
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
        std::printf("Verlet-list updates: %d%s\n", nl_.N_updates,
                    cfg_.timer_sync ? "" : " (timer_sync = 0: sections not synchronised)");

        if (!cfg_.timing) {
            std::printf("(set 'timing = 1' in the input file for the per-section breakdown)\n");
            return;
        }

        const double sum   = t_first + t_sort + t_lists + t_forces + t_thermo + t_out;
        const double other = (loop > sum) ? (loop - sum) : 0.0;
        auto row = [&](const char *name, double t) {
            std::printf("%-22s | %10.4f | %6.2f | %10.3f\n",
                        name, t, loop > 0 ? 100.0 * t / loop : 0.0,
                        nsteps > 0 ? 1e6 * t / nsteps : 0.0);
        };
        std::printf("\nSection timing breakdown (oxDNA timers):\n");
        std::printf("%-22s | %10s | %6s | %10s\n", "Section", "time (s)", "%loop", "us/step");
        std::printf("------------------------------------------------------------\n");
        row("First Step",        t_first);   // LAMMPS: Modify (initial_integrate)
        row("Hilbert sorting",   t_sort);
        row("Lists",             t_lists);   // LAMMPS: Neigh
        row("Forces",            t_forces);  // LAMMPS: Pair + Bond (+ final_integrate)
        row("Thermostat",        t_thermo);  // LAMMPS: Modify (thermostat fix)
        row("Output",            t_out);     // LAMMPS: Output
        row("Other",             other);
        std::printf("------------------------------------------------------------\n");
        row("Total (loop)",      loop);
    }

    SimConfig        cfg_;
    int              N_   = 0;
    long long        curr_step_ = 0;
    SimBox           box_;
    ParticleArraysHost host_;
    ParticleArrays   dev_;
    DNAParams        par_;
    DNA3Params       par3_;
    NeighborList     nl_;
    HilbertSorter    sorter_;
    Thermostat       thermo_;
    MDConstants      md_;
    PinnedFlag       are_lists_old_;
    double           epot_edge_ = 0;
    double           cuda_energy_ = 0;
};
