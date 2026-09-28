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
//   --terminal           (model 3) lammps_coaxstk_terminal = 1
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

int main(int argc, char**argv){
    std::string only, seqf;
    bool consistent_gamma = false, average_seq = false, overhead = false, terminal = false;
    {
        int j = 1;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a.rfind("--only=", 0) == 0) only = a.substr(7);
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
    Kokkos::initialize(argc,argv);
    {
        ParticleArraysHost host; int N;
        read_topology(top, host, N);
        SimBox box; long long step;
        read_config(conf, host, box, step);
        ParticleArrays dev; dev.allocate(N); copy_to_device(host, dev);
        NeighborList nl;
        c_number e_nb = 0, e_bond = 0;
        if (model == 3) {
            SimConfig cfg; cfg.T = T; cfg.salt = salt;
            cfg.dna3_consistent_gamma = consistent_gamma;
            cfg.use_average_seq = average_seq;
            if (!seqf.empty()) cfg.seq_dep_file = seqf;
            DNA3Options o = dna3_options(cfg);
            DNA3Model m = make_dna3_model(o, terminal);
            double nl_cut = std::max(2.5, std::sqrt((double)m.p.cutsq_nb));
            nl.init(nl_cut, 1.0, N, box);
            nl.screen_cutsq = m.p.screen_cutsq;
            nl.build(dev, box);
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
            std::printf("oxDNA3 parameters: %s%s\n", o.average ? "average sequence" : o.seq_file.c_str(),
                        overhead ? "  [lammps_overhead kernels]" : "  [lean kernels]");
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
        } else {
            DNAParams par = (model==2)? make_oxdna2_params(T,salt) : make_oxdna1_params(T);
            double nl_cut = std::max(2.5, std::sqrt((double)par.cutsq_nb));
            nl.init(nl_cut, 1.0, N, box);
            nl.build(dev, box);

            dev.zero_forces();
            e_nb   = compute_nonbonded_forces(dev, nl, par, box, true, overhead);
            e_bond = compute_bonded_forces(dev, par, box, true, overhead);
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
