// Prints Kokkos potential energy (total and per group) for one model, for
// cross-checking against the standalone oxDNA split potential energy.
// Groups: backbone(FENE+bonded excv), stacking, nonbonded(excv+HB+cross+coax+DH).
// For model 3 (oxDNA3) every term is also printed separately (per particle),
// in the column order of the standalone `potential_energy split = true`
// observable: FENE, bonded excv, stacking, nonbonded excv, HB, cross stacking,
// coaxial stacking, Debye-Huckel.
//
// usage: xcheck <model> <T> <salt> <top> <conf> [ft_out] [seq_dep_file]
//               [--only=<term>] [--consistent-gamma]
//   --only=<term>        (model 3) dump the forces/torques of a single term:
//                        fene, bexc, stck, nexc, hb, crst, cxst, dh (matches the
//                        standalone DNA_enable_<term> switches)
//   --consistent-gamma   (model 3) exact stacking-dihedral gradient
//                        (dna3_consistent_gamma = 1, see README)
//   --average-seq        (model 3) use_average_seq = 1
//   --edge               dump the forces/torques of the edge kernels (oxDNA
//                        use_edge = true) instead of the default per-particle
//                        kernels (use_edge = false)
// The group / per-term energies are evaluated with the edge kernels; the
// TOTAL is printed for both paths.
#include <Kokkos_Core.hpp>
#include "../src/simulation.h"
#include "../src/forces/dna_forces.h"
#include "../src/forces/bonded.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <type_traits>

int main(int argc, char**argv){
    // strip the optional --flags from the positional arguments
    std::string only;
    bool consistent_gamma = false, average_seq = false, edge = false;
    {
        int j = 1;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a.rfind("--only=", 0) == 0) only = a.substr(7);
            else if (a == "--consistent-gamma") consistent_gamma = true;
            else if (a == "--average-seq") average_seq = true;
            else if (a == "--edge") edge = true;
            else argv[j++] = argv[i];
        }
        argc = j;
    }
    int model = (argc>1)? std::atoi(argv[1]) : 1;
    double T  = (argc>2)? std::atof(argv[2]) : 0.1;
    double salt=(argc>3)? std::atof(argv[3]) : 0.5;
    const char* top  = (argc>4)? argv[4] : "tests/8bp_duplex/test.top";
    const char* conf = (argc>5)? argv[5] : "tests/8bp_duplex/test.conf";
    const char* ftout= (argc>6 && argv[6][0] != '-')? argv[6] : nullptr;  // per-particle force/torque
    const char* seqf = (argc>7)? argv[7] : nullptr;  // oxDNA3 seq_dep_file (default: params/)
    Kokkos::initialize(argc,argv);
    {
        ParticleArraysHost host; int N;
        read_topology(top, host, N);
        SimBox box; long long step;
        read_config(conf, host, box, step);
        ParticleArrays dev; dev.allocate(N); copy_to_device(host, dev);
        c_number e_nb = 0, e_bond = 0;
        double e_pp = 0;
        NeighborList nl;
        if (model == 3) {
            SimConfig cfg; cfg.T = T; cfg.salt = salt;
            cfg.dna3_consistent_gamma = consistent_gamma;
            cfg.use_average_seq = average_seq;
            if (seqf) cfg.seq_dep_file = seqf;
            DNA3Options o = dna3_options(cfg);
            DNA3Params par = make_oxdna3_params(o);
            double nl_cut = std::max(2.5, std::sqrt((double)par.cutsq_nb));
            nl.init(N, nl_cut, 1.0, box, dev.poss, true);
            nl.update(dev, box);
            // per-term energies (each evaluated separately, TERMS mask)
            double et[8];
            auto ppterm = [&](auto tag) {
                set_external_forces(dev);
                compute_forces_per_particle_dna3<decltype(tag)::value>(dev, nl, par, box, false);
                return sum_forces_w(dev.forces, N) * 0.5; };
            auto nbterm = [&](auto tag) {
                dev.zero_forces();
                double e = compute_nonbonded_forces_dna3<decltype(tag)::value>(dev, nl, par, box);
                Kokkos::fence(); return e; };
            auto bterm = [&](auto tag) {
                dev.zero_forces();
                double e = compute_bonded_forces_dna3<decltype(tag)::value>(dev, par);
                Kokkos::fence(); return e; };
            et[0] = bterm(std::integral_constant<int, dna3::BACKBONE>{});
            et[1] = bterm(std::integral_constant<int, dna3::BONDED_EXCLUDED_VOLUME>{});
            et[2] = bterm(std::integral_constant<int, dna3::STACKING>{});
            et[3] = nbterm(std::integral_constant<int, dna3::NONBONDED_EXCLUDED_VOLUME>{});
            et[4] = nbterm(std::integral_constant<int, dna3::HYDROGEN_BONDING>{});
            et[5] = nbterm(std::integral_constant<int, dna3::CROSS_STACKING>{});
            et[6] = nbterm(std::integral_constant<int, dna3::COAXIAL_STACKING>{});
            et[7] = nbterm(std::integral_constant<int, dna3::DEBYE_HUCKEL>{});
            std::printf("oxDNA3 parameters: %s\n", o.seq_file.c_str());
            std::printf("  per-term energy / particle (standalone split order):\n  ");
            const char *nm[8] = {"FENE", "BEXC", "STCK", "NEXC", "HB", "CRST", "CXST", "DH"};
            for (int t = 0; t < 8; t++) std::printf(" %s=%.8f", nm[t], et[t] / N);
            std::printf("\n  split:");
            for (int t = 0; t < 8; t++) std::printf(" %10.6f", et[t] / N);
            std::printf("\n");

            dev.zero_forces();
            e_nb   = compute_nonbonded_forces_dna3(dev, nl, par, box);
            e_bond = compute_bonded_forces_dna3(dev, par);
            Kokkos::fence();
            if (!edge) {   // default per-particle kernel (lab-frame torques)
                set_external_forces(dev);
                compute_forces_per_particle_dna3(dev, nl, par, box, false);
                e_pp = sum_forces_w(dev.forces, N) * 0.5;
            }

            if (!only.empty() && !edge) {   // per-particle kernel restricted to one term
                using std::integral_constant;
                if      (only == "fene") ppterm(integral_constant<int, dna3::BACKBONE>{});
                else if (only == "bexc") ppterm(integral_constant<int, dna3::BONDED_EXCLUDED_VOLUME>{});
                else if (only == "stck") ppterm(integral_constant<int, dna3::STACKING>{});
                else if (only == "nexc") ppterm(integral_constant<int, dna3::NONBONDED_EXCLUDED_VOLUME>{});
                else if (only == "hb")   ppterm(integral_constant<int, dna3::HYDROGEN_BONDING>{});
                else if (only == "crst") ppterm(integral_constant<int, dna3::CROSS_STACKING>{});
                else if (only == "cxst") ppterm(integral_constant<int, dna3::COAXIAL_STACKING>{});
                else if (only == "dh")   ppterm(integral_constant<int, dna3::DEBYE_HUCKEL>{});
                else { std::fprintf(stderr, "unknown term '%s'\n", only.c_str()); std::exit(1); }
                std::printf("  force/torque dump restricted to term '%s'\n", only.c_str());
            } else if (!only.empty()) {   // leave only this term's forces/torques in dev
                if      (only == "fene") bterm(std::integral_constant<int, dna3::BACKBONE>{});
                else if (only == "bexc") bterm(std::integral_constant<int, dna3::BONDED_EXCLUDED_VOLUME>{});
                else if (only == "stck") bterm(std::integral_constant<int, dna3::STACKING>{});
                else if (only == "nexc") nbterm(std::integral_constant<int, dna3::NONBONDED_EXCLUDED_VOLUME>{});
                else if (only == "hb")   nbterm(std::integral_constant<int, dna3::HYDROGEN_BONDING>{});
                else if (only == "crst") nbterm(std::integral_constant<int, dna3::CROSS_STACKING>{});
                else if (only == "cxst") nbterm(std::integral_constant<int, dna3::COAXIAL_STACKING>{});
                else if (only == "dh")   nbterm(std::integral_constant<int, dna3::DEBYE_HUCKEL>{});
                else { std::fprintf(stderr, "unknown term '%s'\n", only.c_str()); std::exit(1); }
                std::printf("  force/torque dump restricted to term '%s'\n", only.c_str());
            }
        } else {
            DNAParams par = (model==2)? make_oxdna2_params(T,salt) : make_oxdna1_params(T);
            double nl_cut = std::max(2.5, std::sqrt((double)par.cutsq_nb));
            nl.init(N, nl_cut, 1.0, box, dev.poss, true);
            nl.update(dev, box);

            dev.zero_forces();
            e_nb   = compute_nonbonded_forces(dev, nl, par, box);
            e_bond = compute_bonded_forces(dev, par);
            Kokkos::fence();
            if (!edge) {   // default per-particle kernel (lab-frame torques)
                set_external_forces(dev);
                compute_forces_per_particle(dev, nl, par, box, false);
                e_pp = sum_forces_w(dev.forces, N) * 0.5;
            }
        }
        c_number tot = e_nb + e_bond;
        std::printf("Kokkos oxDNA%d  N=%d  T=%.4f salt=%.3f\n", model, N, T, salt);
        std::printf("  nonbonded(all)       = %12.6f  (%.6f /particle)\n", (double)e_nb,   (double)e_nb/N);
        std::printf("  bonded(FENE+excv+stk)= %12.6f  (%.6f /particle)\n", (double)e_bond, (double)e_bond/N);
        std::printf("  TOTAL                = %12.6f  (%.6f /particle)\n", (double)tot,    (double)tot/N);
        if (!edge)
            std::printf("  TOTAL per-particle   = %12.6f  (%.6f /particle)\n", e_pp, e_pp/N);

        if (ftout) {
            auto F = Kokkos::create_mirror_view(dev.forces);  Kokkos::deep_copy(F, dev.forces);
            auto Tq= Kokkos::create_mirror_view(dev.torques); Kokkos::deep_copy(Tq, dev.torques);
            FILE* f = std::fopen(ftout, "w");
            for (int i=0;i<N;i++)
                std::fprintf(f, "%d %.10g %.10g %.10g %.10g %.10g %.10g\n", i,
                             (double)F(i,0),(double)F(i,1),(double)F(i,2),
                             (double)Tq(i,0),(double)Tq(i,1),(double)Tq(i,2));
            std::fclose(f);
            std::printf("  wrote per-particle force/torque (lab frame, %s kernels) to %s\n",
                        edge ? "edge" : "per-particle", ftout);
        }
    }
    Kokkos::finalize();
    return 0;
}
