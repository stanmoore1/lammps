#include <Kokkos_Core.hpp>
#include "simulation.h"
#include "io/input_reader.h"
#include <iostream>
#include <string>

static void print_usage(const char *prog) {
    std::cout
        << "Usage: " << prog << " <input_file>\n\n"
        << "Reads a standalone-oxDNA-style input file (key = value). Recognized keys:\n"
        << "  topology, conf_file                 (mandatory: .top / .conf files)\n"
        << "  interaction_type                    DNA | DNA1 -> oxDNA1, DNA2 -> oxDNA2\n"
        << "  salt_concentration                  mol/L (oxDNA2)\n"
        << "  T                                   e.g. 20C, 300K, or a number (oxDNA units)\n"
        << "  dt, steps, verlet_skin, print_energy_every, seed\n"
        << "  thermostat                          brownian | john (NVT); else NVE\n"
        << "  newtonian_steps, diff_coeff, pt     Brownian thermostat parameters\n"
        << "  timing                              0 | 1 (per-kernel timing breakdown)\n\n"
        << "Other oxDNA keys (backend, CUDA_list, trajectory_file, ...) are ignored.\n"
        << "Kokkos runtime flags (--kokkos-*) are also accepted.\n";
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // First positional (non-flag) argument is the input file; Kokkos flags
        // (--kokkos-*) and -h/--help are handled separately.
        std::string input;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a == "-h" || a == "--help") { print_usage(argv[0]); Kokkos::finalize(); return 0; }
            if (!a.empty() && a[0] != '-') { input = a; break; }
        }

        if (input.empty()) {
            print_usage(argv[0]);
            Kokkos::finalize();
            return 1;
        }

        try {
            SimConfig cfg;
            read_input(input, cfg);
            Simulation sim(cfg);
            sim.init();
            sim.run();
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << "\n";
            Kokkos::finalize();
            return 1;
        }
    }
    Kokkos::finalize();
    return 0;
}
