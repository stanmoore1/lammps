// Prints Kokkos potential energy (total and per group) for one model, for
// cross-checking against the standalone oxDNA split potential energy and
// against the CUDA-faithful sibling bench/oxdna_kokkos (its xcheck).
// Groups follow the LAMMPS style split: "nonbonded" = excv (incl. the bonded
// excluded volume, computed from the special 1-2 neighbours as in LAMMPS) +
// HB + cross + coax + DH; "bonded" = stacking + FENE.
// For model 3 (oxDNA3) every term is also printed separately (per particle),
// in the column order of the standalone `potential_energy split = true`
// observable (and of the sibling's xcheck): FENE, bonded excv, stacking,
// nonbonded excv, HB, cross stacking, coaxial stacking, Debye-Huckel. The
// bonded / nonbonded excluded volume are the two halves of the excv kernel.
//
// usage: xcheck <model> <T> <salt> <top> <conf> [ft_out|-] [overhead]
//               [--overhead] [--only=<term>] [--consistent-gamma]
//               [--average-seq] [--seq=<file>] [--terminal]
//   overhead / --overhead  run the lammps_overhead kernel variants
//   --only=<term>        (model 3) dump the forces/torques of a single term:
//                        fene, bexc, stck, nexc, hb, crst, cxst, dh, excv
//   --consistent-gamma   (model 3) dna3_consistent_gamma = 1
//   --average-seq        (model 3) use_average_seq = 1
//   --seq=<file>         (model 3) seq_dep_file (default: see README)
//   --terminal           lammps_coaxstk_terminal = 1 (all models)
//   --tables=0|1         lammps_tables (default: = overhead; oxDNA1/2)
//   --ghosts             LAMMPS ghost-atom path (needs --overhead): periodic
//                        images as ghost atoms, borders + map, binned
//                        half/bin/newton list, bond topology with closest
//                        images, no minimum image, reverse communication of
//                        the ghost forces / torques before the dump
//   --comm-cutoff=<x>    ghost cutoff (comm_modify cutoff; default: list radius)
//   --shift=<dx,dy,dz>   translate the configuration (then fold into the box)
//                        so that strands cross the periodic boundaries
// The oxDNA3 screened-pair list uses the bare screen range (no skin margin),
// so any interaction the screen would miss shows up as a difference to the
// sibling (which uses the full neighbour list).
#include <Kokkos_Core.hpp>
#include "../src/simulation.h"
#include "../src/forces/dna_forces.h"
#include "../src/forces/bonded.h"
#include "../src/forces/dna3_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include "../src/lammps_framework.h"

int main(int argc, char**argv){
    std::string only, seqf;
    bool consistent_gamma = false, average_seq = false, overhead = false, terminal = false;
    bool ghosts = false;
    int tables = -1;
    double comm_cutoff = 0, shift[3] = {0, 0, 0};
    {
        int j = 1;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a.rfind("--only=", 0) == 0) only = a.substr(7);
            else if (a.rfind("--tables=", 0) == 0) tables = std::atoi(a.substr(9).c_str());
            else if (a == "--ghosts") ghosts = true;
            else if (a.rfind("--comm-cutoff=", 0) == 0) comm_cutoff = std::atof(a.substr(14).c_str());
            else if (a.rfind("--shift=", 0) == 0)
                std::sscanf(a.substr(8).c_str(), "%lf,%lf,%lf", &shift[0], &shift[1], &shift[2]);
            else if (a.rfind("--seq=", 0) == 0) seqf = a.substr(6);
            else if (a == "--consistent-gamma") consistent_gamma = true;
            else if (a == "--average-seq") average_seq = true;
            else if (a == "--overhead") overhead = true;
            else if (a == "--terminal") terminal = true;
            else argv[j++] = argv[i];
        }
        argc = j;
    }
    int model = (argc>1)? std::atoi(argv[1]) : 1;
    double T  = (argc>2)? std::atof(argv[2]) : 0.1;
    double salt=(argc>3)? std::atof(argv[3]) : 0.5;
    const char* top  = (argc>4)? argv[4] : "tests/8bp_duplex/test.top";
    const char* conf = (argc>5)? argv[5] : "tests/8bp_duplex/test.conf";
    const char* ftout= (argc>6 && std::string(argv[6]) != "-")? argv[6] : nullptr;  // if set, dump per-particle force/torque
    if ((argc>7) && std::string(argv[7]) == "overhead") overhead = true;
    if (ghosts && !overhead) { std::fprintf(stderr, "--ghosts needs --overhead\n"); return 1; }
    Kokkos::initialize(argc,argv);
    {
        ParticleArraysHost host; int N;
        read_topology(top, host, N);
        SimBox box; long long step;
        read_config(conf, host, box, step);
        if (shift[0] != 0 || shift[1] != 0 || shift[2] != 0) {
            const double L[3] = {(double)box.Lx, (double)box.Ly, (double)box.Lz};
            for (int i = 0; i < N; i++)
                for (int d = 0; d < 3; d++) {
                    double x = (double)host.poss(i, d) + shift[d];
                    x -= L[d] * std::floor(x / L[d]);            // fold into [0, L)
                    host.poss(i, d) = static_cast<c_number>(x);
                }
        }
        ParticleArrays dev; dev.allocate(N); copy_to_device(host, dev);
        dev.use_tables = overhead && (tables < 0 ? true : tables != 0);
        NeighborList nl;
        LammpsFramework lmp;
        const double skin = 1.0;   // verlet_skin of the list (LAMMPS skin 2.0)
        // LAMMPS ghost path: ghosts, binned list with the bench's list radius,
        // bond list; the screen is rebuilt from the new list
        auto ghost_setup = [&](double nl_cut) {
            lmp.setup(dev, nl, box, nl_cut + 2 * skin, 2 * skin, comm_cutoff);
            lmp.rebuild(dev, nl);
            nl.build_screen(dev, box);
            nl.N_edges = LammpsNeigh::count_pairs(dev, nl);
        };
        auto ghost_finish = [&]() { if (ghosts) lmp.reverse(dev); };
        c_number e_nb = 0, e_bond = 0;
        if (model == 3) {
            SimConfig cfg; cfg.T = T; cfg.salt = salt;
            cfg.dna3_consistent_gamma = consistent_gamma;
            cfg.use_average_seq = average_seq;
            if (!seqf.empty()) cfg.seq_dep_file = seqf;
            DNA3Options o = dna3_options(cfg);
            DNA3Model m = make_dna3_model(o, terminal);
            double nl_cut = std::max(2.5, std::sqrt((double)m.p.cutsq_nb));
            nl.init(nl_cut, skin, N, box);
            nl.screen_cutsq = m.p.screen_cutsq;
            if (ghosts) { dev.ensure_qeff(m.dh.dh_half_ends); ghost_setup(nl_cut); }
            else nl.build(dev, box);
            using namespace dna3k;
            // per-term energies (each evaluated separately)
            double et[8];
            et[0] = compute_forces_dna3(dev, nl, m, box, overhead, FENE);
            et[1] = compute_excv_part_dna3(dev, nl, m, box, overhead, true);
            et[2] = compute_forces_dna3(dev, nl, m, box, overhead, STK);
            et[3] = compute_excv_part_dna3(dev, nl, m, box, overhead, false);
            et[4] = compute_forces_dna3(dev, nl, m, box, overhead, HBOND);
            et[5] = compute_forces_dna3(dev, nl, m, box, overhead, XSTK);
            et[6] = compute_forces_dna3(dev, nl, m, box, overhead, COAXSTK);
            et[7] = compute_forces_dna3(dev, nl, m, box, overhead, DH);
            std::printf("oxDNA3 parameters: %s%s%s\n", o.average ? "average sequence" : o.seq_file.c_str(),
                        overhead ? "  [lammps_overhead kernels]" : "  [lean kernels]",
                        ghosts ? " [ghosts]" : "");
            if (ghosts) std::printf("  ghost atoms: %d\n", dev.nghost);
            std::printf("  screened pairs: %d of %d (screen %.6f)\n", nl.N_screened, nl.N_edges,
                        std::sqrt((double)m.p.screen_cutsq));
            std::printf("  per-term energy / particle (standalone split order):\n  ");
            const char *nm[8] = {"FENE", "BEXC", "STCK", "NEXC", "HB", "CRST", "CXST", "DH"};
            for (int t = 0; t < 8; t++) std::printf(" %s=%.8f", nm[t], et[t] / N);
            std::printf("\n  split:");
            for (int t = 0; t < 8; t++) std::printf(" %10.6f", et[t] / N);
            std::printf("\n  split(17g):");
            for (int t = 0; t < 8; t++) std::printf(" %.17g", et[t]);
            std::printf("\n");

            c_acc ek[7];
            compute_forces_dna3(dev, nl, m, box, overhead, ALL, ek);
            e_nb   = static_cast<c_number>(ek[0] + ek[2] + ek[3] + ek[4] + ek[5]);
            e_bond = static_cast<c_number>(ek[1] + ek[6]);
            std::printf("  TOTAL(17g) = %.17g\n", (double)(ek[0] + ek[1] + ek[2] + ek[3] + ek[4] + ek[5] + ek[6]));

            if (!only.empty()) {   // leave only this term's forces/torques in dev
                if      (only == "fene") compute_forces_dna3(dev, nl, m, box, overhead, FENE);
                else if (only == "bexc") compute_excv_part_dna3(dev, nl, m, box, overhead, true);
                else if (only == "stck") compute_forces_dna3(dev, nl, m, box, overhead, STK);
                else if (only == "nexc") compute_excv_part_dna3(dev, nl, m, box, overhead, false);
                else if (only == "excv") compute_forces_dna3(dev, nl, m, box, overhead, EXCV);
                else if (only == "hb")   compute_forces_dna3(dev, nl, m, box, overhead, HBOND);
                else if (only == "crst") compute_forces_dna3(dev, nl, m, box, overhead, XSTK);
                else if (only == "cxst") compute_forces_dna3(dev, nl, m, box, overhead, COAXSTK);
                else if (only == "dh")   compute_forces_dna3(dev, nl, m, box, overhead, DH);
                else { std::fprintf(stderr, "unknown term '%s'\n", only.c_str()); std::exit(1); }
                std::printf("  force/torque dump restricted to term '%s'\n", only.c_str());
            }
            ghost_finish();
        } else {
            DNAParams par = (model==2)? make_oxdna2_params(T,salt) : make_oxdna1_params(T);
            par.cxst_terminal_only = terminal;
            par.cxst_t4_blunt      = terminal;
            double nl_cut = std::max(2.5, std::sqrt((double)par.cutsq_nb));
            nl.init(nl_cut, skin, N, box);
            if (ghosts) {
                if (par.dh_enabled) dev.ensure_qeff(par.dh_half_ends);
                ghost_setup(nl_cut);
                std::printf("  [ghosts] ghost atoms: %d, pairs: %d, screened: %d\n", dev.nghost, nl.N_edges,
                            nl.N_screened);
            } else nl.build(dev, box);

            dev.zero_forces();
            e_nb   = compute_nonbonded_forces(dev, nl, par, box, true, overhead);
            e_bond = compute_bonded_forces(dev, par, box, true, overhead);
            ghost_finish();
            Kokkos::fence();
        }
        c_number tot = e_nb + e_bond;
        std::printf("Kokkos oxDNA%d  N=%d  T=%.4f salt=%.3f\n", model, N, T, salt);
        std::printf("  nonbonded(all)       = %12.6f  (%.6f /particle)\n", (double)e_nb,   (double)e_nb/N);
        std::printf("  bonded(stk+FENE)     = %12.6f  (%.6f /particle)\n", (double)e_bond, (double)e_bond/N);
        std::printf("  TOTAL                = %12.6f  (%.6f /particle)\n", (double)tot,    (double)tot/N);

        if (ftout) {
            auto F = Kokkos::create_mirror_view(dev.forces);  Kokkos::deep_copy(F, dev.forces);
            auto Tq= Kokkos::create_mirror_view(dev.torques); Kokkos::deep_copy(Tq, dev.torques);
            FILE* f = std::fopen(ftout, "w");
            for (int i=0;i<N;i++)
                std::fprintf(f, "%d %.17g %.17g %.17g %.17g %.17g %.17g\n", i,
                             (double)F(i,0),(double)F(i,1),(double)F(i,2),
                             (double)Tq(i,0),(double)Tq(i,1),(double)Tq(i,2));
            std::fclose(f);
            std::printf("  wrote per-particle force/torque (lab frame) to %s\n", ftout);
        }
    }
    Kokkos::finalize();
    return 0;
}
