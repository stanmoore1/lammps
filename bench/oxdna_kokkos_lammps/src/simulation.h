#pragma once

#include "types.h"
#include "particles.h"
#include "neighbor_list.h"
#include "integrator.h"
#include "thermostat.h"
#include "forces/dna_forces.h"
#include "forces/bonded.h"
#include "forces/params.h"
#include "forces/params_dna3.h"
#include "forces/dna3_forces.h"
#include "forces/dna3_kernels.h"
#include "lammps_framework.h"
#include "io/topology_reader.h"
#include "io/config_reader.h"
#include <chrono>
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
    c_number    dt          = 1e-3;
    c_number    T           = 0.1;
    c_number    cutoff      = 2.5;
    c_number    skin        = 0.3;
    int         output_freq = 1000;
    bool        timing      = false; // per-kernel breakdown (adds fences); off = production
    bool        lammps_overhead = false; // add LAMMPS per-step framework overheads (bond precompute, per-kernel scatter, host flag copy)
    bool        fuse_hbxstk = false; // fuse hbond+xstk into one screened-pair kernel (shared base-site geometry)
    bool        coaxstk_terminal = false; // oxDNA1/2/3: LAMMPS-only terminal-nucleotide coaxstk + blunt theta4 lobe
    // Finer LAMMPS-fidelity switches (see README, "Fidelity to LAMMPS KOKKOS").
    // -1 = follow lammps_overhead (these do not change the physics).
    int         lammps_tables = -1;  // per-type coefficient tables (oxDNA1/2 kernels)
    int         lammps_ghosts = -1;  // ghost atoms + comm + LAMMPS neighbor build + Verlet sequence
    // physics-changing options, off by default:
    bool        lammps_cutoff = false;  // list radius = LAMMPS cutforce (site cutoffs) + skin
    bool        lammps_integrator = false;  // fix nve/asphere/kk (Richardson; ghost mode only)
    double      lammps_mass  = 1.0;          // rmass for lammps_integrator
    double      lammps_shape = 1.5811388300841898;  // ellipsoid radii (sqrt(2.5): inertia 1 for mass 1)
    // comm_modify cutoff (ghost cutoff; 0 = the list radius, LAMMPS default)
    double      comm_cutoff = 0;
    // neigh_modify every / check (lammps_ghosts): rebuild schedule
    int         neigh_every = 1;
    bool        neigh_check = true;
    int         model       = 1;     // 1 = oxDNA1, 2 = oxDNA2, 3 = oxDNA3
    c_number    salt        = 0.5;   // salt concentration [mol/L] (oxDNA2/oxDNA3)
    // oxDNA3 only: sequence-dependence file (upstream key seq_dep_file) and
    // Debye-Huckel options (upstream DNA2Interaction keys), as in bench/oxdna_kokkos
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

// Default oxDNA3 parameter file. This bench does not ship its own copy: it
// uses the one of the CUDA-faithful sibling (bench/oxdna_kokkos/params/),
// whose absolute path is baked in at configure time
// (OXDNA_DEFAULT_SEQ_DEP_FILE, see CMakeLists.txt), falling back to paths
// relative to the working directory (the bench root or a tests/<case>/
// directory). An explicit `seq_dep_file` is used as given (relative to the
// working directory, as in the standalone oxDNA).
inline std::string default_dna3_seq_file() {
    const char *cands[] = {
#ifdef OXDNA_DEFAULT_SEQ_DEP_FILE
        OXDNA_DEFAULT_SEQ_DEP_FILE,
#endif
        "../oxdna_kokkos/params/oxDNA3_sequence_dependent_parameters.txt",
        "../../../oxdna_kokkos/params/oxDNA3_sequence_dependent_parameters.txt",
        "oxDNA3_sequence_dependent_parameters.txt",
    };
    for (const char *c : cands) {
        std::ifstream t(c);
        if (t.good()) return c;
    }
    throw std::runtime_error("oxDNA3: sequence-dependence file not found; set seq_dep_file "
                             "(bench/oxdna_kokkos/params/oxDNA3_sequence_dependent_parameters.txt) "
                             "or use_average_seq = 1");
}

inline DNA3Options dna3_options(const SimConfig &cfg) {
    DNA3Options o;
    o.T = cfg.T;
    o.salt = cfg.salt;
    o.average = cfg.use_average_seq;
    if (!o.average)
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
        c_number cutsq_nb, screen_cutsq;
        double screen_site_cut = 0, site_cut_max = 0;
        if (cfg_.model == 3) {
            DNA3Options o = dna3_options(cfg_);
            if (!o.average) std::cout << "oxDNA3: sequence-dependent parameters from " << o.seq_file << "\n";
            if (cfg_.fuse_hbxstk)
                std::cerr << "Warning: fuse_hbond_xstk is not available for oxDNA3 and is ignored\n";
            dna3_ = make_dna3_model(o, cfg_.coaxstk_terminal);
            cutsq_nb     = dna3_.p.cutsq_nb;
            screen_cutsq = dna3_.p.screen_cutsq;
            screen_site_cut = dna3_.p.screen_site_cut;
            site_cut_max    = dna3_.p.site_cut_max;
        } else {
            par_ = (cfg_.model == 2) ? make_oxdna2_params(cfg_.T, cfg_.salt)
                                     : make_oxdna1_params(cfg_.T);
            if (cfg_.coaxstk_terminal) {     // oxDNA2 and (kk-fixes) oxDNA1
                par_.cxst_terminal_only = true;
                par_.cxst_t4_blunt      = true;
            }
            cutsq_nb     = par_.cutsq_nb;
            screen_cutsq = par_.screen_cutsq;
            screen_site_cut = par_.screen_site_cut;
            site_cut_max    = par_.site_cut_max;
        }

        // Thermostat (optional)
        thermo_.init(cfg_.T, cfg_.newtonian_steps, cfg_.dt, cfg_.diff_coeff,
                     cfg_.pt, cfg_.seed);

        const bool ghosts = use_ghosts();
        dev_.use_tables = use_tables();
        if (cfg_.lammps_ghosts == 1 && !cfg_.lammps_overhead)
            std::cerr << "Warning: lammps_ghosts needs lammps_overhead = 1 (per-bond kernels); ignored\n";
        if (cfg_.lammps_ghosts > 0 && cfg_.fuse_hbxstk)
            std::cerr << "Warning: fuse_hbond_xstk has no LAMMPS equivalent\n";

        // Neighbor list radius (LAMMPS: cutneighmax = cutforce + skin, with
        // skin = 2 * verlet_skin). Default: the COM range of every term plus
        // 2 * verlet_skin (exact: no interaction can be missed between
        // rebuilds). lammps_cutoff = 1: LAMMPS' cutforce, the largest SITE
        // cutoff, which relies on the skin to cover the site offsets.
        const double vs = static_cast<double>(cfg_.skin);
        double nl_cut = std::max(static_cast<double>(cfg_.cutoff), std::sqrt(static_cast<double>(cutsq_nb)));
        if (cfg_.lammps_cutoff) nl_cut = site_cut_max;
        nl_.init(nl_cut, cfg_.skin, N_, box_);
        // COM screen cutoff for the hbond/xstk/coaxstk pair kernels, as LAMMPS
        // fix OXDNA/NPAIR/kk (kk-fixes): the largest registered site cutoff
        // (hbond, xstk incl. oxdna3/xstk, coaxstk) + 2 * max_site_offset()
        // (0.43) + the full LAMMPS skin (= 2 * verlet_skin, the most two atoms
        // can approach each other between rebuilds). This is never shorter
        // than the exact range of the bench's physics (screen_cutsq) plus
        // 2 * verlet_skin, so no interacting pair is dropped.
        {
            const double cut = screen_site_cut + 2.0 * LAMMPS_MAX_SITE_OFFSET + 2.0 * vs;
            const double exact = std::sqrt(static_cast<double>(screen_cutsq)) + 2.0 * vs;
            nl_.screen_cutsq = static_cast<c_number>(std::max(cut, exact) * std::max(cut, exact));
        }
        // Debye-Huckel per-atom charges are border data: set them before the
        // first ghost creation
        if (cfg_.model == 3) dev_.ensure_qeff(dna3_.dh.dh_half_ends);
        else if (par_.dh_enabled) dev_.ensure_qeff(par_.dh_half_ends);

        if (cfg_.lammps_integrator && !ghosts)
            std::cerr << "Warning: lammps_integrator needs lammps_ghosts (and lammps_overhead); ignored\n";
        if (cfg_.lammps_integrator && ghosts) {
            // per-atom mass and bonus shape (border data: set before the ghosts)
            auto rm = dev_.rmass; auto sh = dev_.shape;
            const c_number m = static_cast<c_number>(cfg_.lammps_mass), r = static_cast<c_number>(cfg_.lammps_shape);
            Kokkos::parallel_for("set_rmass_shape", N_, KOKKOS_LAMBDA(int i) {
                rm(i) = m; sh(i,0) = r; sh(i,1) = r; sh(i,2) = r;
            });
            if (cfg_.lammps_mass != 1.0 || std::fabs(0.4 * cfg_.lammps_mass * cfg_.lammps_shape * cfg_.lammps_shape - 1.0) > 1e-12)
                std::cerr << "Warning: lammps_integrator with mass / inertia != 1: the kinetic energy output and "
                             "the Brownian thermostat still assume unit mass and inertia\n";
        }
        if (ghosts) {
            lmp_.setup(dev_, nl_, box_, nl_cut + 2.0 * vs, 2.0 * vs, cfg_.comm_cutoff,
                       cfg_.neigh_every, cfg_.neigh_check);
            lmp_.rebuild(dev_, nl_);
            LammpsFramework::force_clear(dev_);
            compute_lrf(dev_, true);
            nl_.build_screen(dev_, box_);
            epot_  = static_cast<c_number>(pair_step(true, /*neigh_rebuilt=*/true, /*run_lrf=*/false));
            epot_ += static_cast<c_number>(bond_step(true, true));
            LammpsFramework::virial(dev_);
            lmp_.reverse(dev_);
            nl_.N_edges = LammpsNeigh::count_pairs(dev_, nl_);
        } else {
            nl_.build(dev_, box_);
            // Initial forces, in the LAMMPS kernel order (runs the rebuild-step
            // precomputes too, so the per-bond tables exist before the first step).
            dev_.zero_forces();
            epot_  = static_cast<c_number>(pair_step(true, /*neigh_rebuilt=*/true));
            epot_ += static_cast<c_number>(bond_step(true, true));
        }

        std::cout << "Precision: "
                  << (sizeof(c_number) == 4 ? "single (float)" : "double")
                  << " (" << (sizeof(c_number) * 8) << "-bit c_number)\n";
        std::cout << "Initialized " << N_ << " particles, "
                  << nl_.N_edges << " neighbor pairs (" << nl_.N_screened
                  << " screened).\n";
        if (cfg_.lammps_overhead)
            std::printf("LAMMPS mode: tables %s, ghosts %s", dev_.use_tables ? "on" : "off", ghosts ? "on" : "off");
        if (ghosts)
            std::printf(" (%d ghost atoms, ghost cutoff %.4f, list radius %.4f, %d bins, %d stencil bins)",
                        dev_.nghost, lmp_.comm.cutghost, lmp_.neigh.cutneighmax, lmp_.neigh.mbins,
                        lmp_.neigh.nstencil);
        if (cfg_.lammps_overhead) std::printf("\n");
    }

    void run() {
        // Per-section timers (LAMMPS-style breakdown). Each section boundary
        // fences only when -timing is on, so a production run (timing off) keeps
        // the kernels pipelined and reports the true loop time; the breakdown is
        // exact on CPU and adds one sync per section on GPU (like LAMMPS
        // `timer full`).
        Timers tm;
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
        if (use_ghosts()) run_lammps(tm, step, emit, clk, sec, mark);
        else              run_lean(tm, step, emit, clk, sec, mark);
        Kokkos::fence();
        auto loop1 = clk();
        double loop_time = sec(loop0, loop1);

        print_performance(loop_time, tm);
        if (use_ghosts())
            std::printf("Neighbor list builds = %lld\nDangerous builds = %lld\n",
                        lmp_.nbuilds, lmp_.ndanger);
    }

private:
    struct Timers { double neigh = 0, bond = 0, pair = 0, mod = 0, out = 0, comm = 0; };

    // Lean / minimum-image MD loop (also lammps_overhead without ghosts).
    template <class Emit, class Clk, class Sec, class Mark>
    void run_lean(Timers &tm, long long &step, Emit &emit, Clk &clk, Sec &sec, Mark &mark) {
        for (long long s = 0; s < cfg_.nsteps; s++, step++) {
            auto a = clk();
            // Fused first step: integrate AND flag a rebuild if any particle has
            // moved > skin since the last build (sets nl_.d_needs_rebuild on
            // device), avoiding a separate full-N reduction every step.
            first_step(dev_, cfg_.dt, box_,
                       nl_.list_poss, nl_.d_needs_rebuild, nl_.rebuild_disp_sq());
            auto b = mark(); tm.mod += sec(a, b);

            const bool neigh_rebuilt = nl_.flag_is_set();
            if (neigh_rebuilt) nl_.build(dev_, box_);
            auto c = mark(); tm.neigh += sec(b, c);

            // Potential energy is only needed on output steps. Off those steps,
            // run the force kernels as plain parallel_for (no per-step reduction
            // kernel / device->host scalar copy), which keeps higher occupancy.
            const bool want_e = ((s + 1) % cfg_.output_freq == 0);

            // Pair section (LAMMPS order): LRF, excv, stk, hbond, xstk,
            // coaxstk, dh -- then the Bond section (fene).
            dev_.zero_forces();
            c_acc ep = pair_step(want_e, neigh_rebuilt);
            auto e = mark(); tm.pair += sec(c, e);

            ep += bond_step(want_e, neigh_rebuilt);
            if (want_e) epot_ = static_cast<c_number>(ep);
            auto f = mark(); tm.bond += sec(e, f);

            second_step(dev_, cfg_.dt);
            if (cfg_.newtonian_steps > 0 && (step % cfg_.newtonian_steps == 0))
                thermo_.apply(dev_);
            auto g = mark(); tm.mod += sec(f, g);

            if ((s + 1) % cfg_.output_freq == 0) emit(step + 1);
            auto h = mark(); tm.out += sec(g, h);
        }
    }

    // LAMMPS VerletKokkos::run() sequence with ghost atoms (lammps_ghosts):
    //   [initial_integrate, unless fused into the previous step]
    //   neighbor->decide() (every / check_distance reduce) ->
    //     rebuild: pbc, map_clear, exchange, borders (+ map_set), neighbor build
    //              (xhold, bins, half/bin/newton list, bond topology)
    //     else:    forward_comm (x + quat)
    //   force_clear (f, torque over nall)
    //   pre_force: fix OXDNA/LRF (nall) [+ fix OXDNA/NPAIR screen, rebuild steps]
    //   pair (excv, stk, hbond, xstk, coaxstk, dh) [+ virial fdotr, energy steps]
    //   bond (fene)
    //   reverse_comm (f + torque)
    //   final_integrate, fused with the next initial_integrate unless this is
    //   an output / thermostat / last step (fuse_check)
    // Timer sections as LAMMPS: pbc/exchange/borders/forward/reverse -> Comm,
    // decide + the neighbor build -> Neigh, integration and pre_force (LRF,
    // screen) -> Modify; the force clear is not attributed (LAMMPS stamps it
    // nowhere, so it ends up in "Other").
    template <class Emit, class Clk, class Sec, class Mark>
    void run_lammps(Timers &tm, long long &step, Emit &emit, Clk &clk, Sec &sec, Mark &mark) {
        bool fused_pending = false;   // this step's initial_integrate already ran
        for (long long s = 0; s < cfg_.nsteps; s++, step++) {
            auto a = clk();
            if (!fused_pending) {
                if (cfg_.lammps_integrator) nve_asphere<0>(dev_, cfg_.dt);
                else first_step(dev_, cfg_.dt, box_, {}, {}, 0, /*fold=*/false, "initial_integrate");
            }
            fused_pending = false;
            auto b = mark(); tm.mod += sec(a, b);

            const bool neigh_rebuilt = lmp_.decide(dev_);
            auto b2 = mark(); tm.neigh += sec(b, b2);
            if (neigh_rebuilt) {
                lmp_.comm.domain_pbc(dev_);
                lmp_.comm.map_clear(dev_);
                lmp_.comm.exchange(dev_);
                lmp_.comm.borders(dev_);
                auto c0 = mark(); tm.comm += sec(b2, c0);
                lmp_.neigh.ncalls++;
                if (lmp_.check) lmp_.neigh.store_xhold(dev_);   // dist_check only
                lmp_.neigh.bin_atoms(dev_);
                lmp_.neigh.build_pairs(dev_, nl_);
                lmp_.neigh.bond_all(dev_);
                lmp_.ago = 0; lmp_.nbuilds++;
                auto c1 = mark(); tm.neigh += sec(c0, c1);
                b2 = c1;
            } else {
                lmp_.forward(dev_);
                auto c1 = mark(); tm.comm += sec(b2, c1);
                b2 = c1;
            }

            const bool want_e = ((s + 1) % cfg_.output_freq == 0);
            const bool thermo_now = (cfg_.newtonian_steps > 0 && (step % cfg_.newtonian_steps == 0));

            LammpsFramework::force_clear(dev_);
            auto c = mark();
            compute_lrf(dev_, true);                                    // fix OXDNA/LRF pre_force
            if (neigh_rebuilt) nl_.build_screen(dev_, box_);            // fix OXDNA/NPAIR pre_force
            auto d = mark(); tm.mod += sec(c, d);

            c_acc ep = pair_step(want_e, neigh_rebuilt, /*run_lrf=*/false);
            if (want_e) LammpsFramework::virial(dev_);
            auto e = mark(); tm.pair += sec(d, e);

            ep += bond_step(want_e, neigh_rebuilt);
            if (want_e) epot_ = static_cast<c_number>(ep);
            auto f = mark(); tm.bond += sec(e, f);

            lmp_.reverse(dev_);
            auto f2 = mark(); tm.comm += sec(f, f2);

            const bool last = (s + 1 == cfg_.nsteps);
            if (!want_e && !thermo_now && !last) {
                if (cfg_.lammps_integrator) nve_asphere<2>(dev_, cfg_.dt);
                else fused_step(dev_, cfg_.dt, box_);     // final(s) + initial(s+1)
                fused_pending = true;
            } else {
                if (cfg_.lammps_integrator) nve_asphere<1>(dev_, cfg_.dt);
                else second_step(dev_, cfg_.dt);
                if (thermo_now) thermo_.apply(dev_);
            }
            auto g = mark(); tm.mod += sec(f2, g);

            if (want_e) emit(step + 1);
            auto h = mark(); tm.out += sec(g, h);
        }
    }

    bool use_tables() const {
        return cfg_.lammps_overhead && cfg_.model != 3 && (cfg_.lammps_tables < 0 ? true : cfg_.lammps_tables != 0);
    }
    bool use_ghosts() const {
        return cfg_.lammps_overhead && (cfg_.lammps_ghosts < 0 ? true : cfg_.lammps_ghosts != 0);
    }

    // Force evaluation, dispatched on the model on the host (the oxDNA1/2 and
    // oxDNA3 kernels are separate; no per-pair model branch).
    c_acc pair_step(bool want_e, bool neigh_rebuilt, bool run_lrf = true) {
        if (cfg_.lammps_overhead && neigh_rebuilt && !use_ghosts()) {
            // minimum-image overhead mode: model the device->host copy of the
            // (static) bond list that neigh_bond build_topology_kk does on
            // every rebuild (the ghost mode rebuilds the list itself)
            ensure_bondlist(dev_);
            auto h_bl = Kokkos::create_mirror_view(dev_.bondlist);
            Kokkos::deep_copy(h_bl, dev_.bondlist);
        }
        if (cfg_.model == 3)
            return compute_pair_forces_step_dna3(dev_, nl_, dna3_, box_, want_e,
                                                 cfg_.lammps_overhead, neigh_rebuilt, nullptr,
                                                 dna3k::PAIR, run_lrf);
        return compute_pair_forces_step(dev_, nl_, par_, box_, want_e,
                                        cfg_.lammps_overhead, cfg_.fuse_hbxstk, neigh_rebuilt, run_lrf);
    }
    c_acc bond_step(bool want_e, bool neigh_rebuilt) {
        if (cfg_.model == 3)
            return compute_bond_forces_step_dna3(dev_, dna3_, box_, want_e, cfg_.lammps_overhead,
                                                 nullptr, neigh_rebuilt);
        return compute_bond_forces_step(dev_, par_, box_, want_e, cfg_.lammps_overhead, neigh_rebuilt);
    }

    void print_performance(double loop, const Timers &tm) const {
        const double t_neigh = tm.neigh, t_bond = tm.bond, t_pair = tm.pair, t_mod = tm.mod,
                     t_out = tm.out, t_comm = tm.comm;
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

        const double sum   = t_neigh + t_bond + t_pair + t_mod + t_out + t_comm;
        const double other = (loop > sum) ? (loop - sum) : 0.0;
        auto row = [&](const char *name, double t) {
            std::printf("%-22s | %10.4f | %6.2f | %10.3f\n",
                        name, t, loop > 0 ? 100.0 * t / loop : 0.0,
                        nsteps > 0 ? 1e6 * t / nsteps : 0.0);
        };
        std::printf("\nKernel timing breakdown:\n");
        std::printf("%-22s | %10s | %6s | %10s\n", "Section", "time (s)", "%loop", "us/step");
        std::printf("------------------------------------------------------------\n");
        row("Pair",                    t_pair);    // LAMMPS Pair: [lrf+]excv+stk+hbond+xstk+coaxstk+dh
        row("Bond",                    t_bond);    // LAMMPS Bond: fene
        row("Neigh",                   t_neigh);   // neighbor list build + rebuild check
        if (use_ghosts())
        row("Comm",                    t_comm);    // pbc, exchange, borders, forward + reverse comm
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
    DNA3Model        dna3_;
    NeighborList     nl_;
    LammpsFramework  lmp_;
    Thermostat       thermo_;
    c_number         epot_= 0;
};
