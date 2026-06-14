#include <Kokkos_Core.hpp>
#include "simulation.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

static void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  -top  <file>   Topology file (.top)\n"
              << "  -conf <file>   Configuration file (.conf)\n"
              << "  -steps <N>     Number of steps (default: 10000)\n"
              << "  -dt   <dt>     Timestep (default: 0.001)\n"
              << "  -T    <T>      Temperature (default: 0.1)\n"
              << "  -cut  <r>      Nonbonded cutoff (default: 2.5)\n"
              << "  -skin <r>      Verlet skin (default: 0.3)\n"
              << "  -freq <N>      Output frequency (default: 1000)\n";
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        SimConfig cfg;
        bool got_top = false, got_conf = false;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-top")   == 0 && i+1 < argc) { cfg.topology_file = argv[++i]; got_top  = true; }
            else if (strcmp(argv[i], "-conf")  == 0 && i+1 < argc) { cfg.config_file   = argv[++i]; got_conf = true; }
            else if (strcmp(argv[i], "-steps") == 0 && i+1 < argc) { cfg.nsteps   = std::atoll(argv[++i]); }
            else if (strcmp(argv[i], "-dt")    == 0 && i+1 < argc) { cfg.dt       = static_cast<c_number>(std::atof(argv[++i])); }
            else if (strcmp(argv[i], "-T")     == 0 && i+1 < argc) { cfg.T        = static_cast<c_number>(std::atof(argv[++i])); }
            else if (strcmp(argv[i], "-cut")   == 0 && i+1 < argc) { cfg.cutoff   = static_cast<c_number>(std::atof(argv[++i])); }
            else if (strcmp(argv[i], "-skin")  == 0 && i+1 < argc) { cfg.skin     = static_cast<c_number>(std::atof(argv[++i])); }
            else if (strcmp(argv[i], "-freq")  == 0 && i+1 < argc) { cfg.output_freq = std::atoi(argv[++i]); }
            else if (strcmp(argv[i], "-h")     == 0 || strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]); Kokkos::finalize(); return 0;
            }
            // Skip Kokkos arguments (--kokkos-*)
        }

        if (!got_top || !got_conf) {
            print_usage(argv[0]);
            Kokkos::finalize();
            return 1;
        }

        try {
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
