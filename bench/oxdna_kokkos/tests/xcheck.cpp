// Prints Kokkos potential energy (total and per group) for one model, for
// cross-checking against the standalone oxDNA split potential energy.
// Groups: backbone(FENE+bonded excv), stacking, nonbonded(excv+HB+cross+coax+DH).
#include <Kokkos_Core.hpp>
#include "../src/simulation.h"
#include "../src/forces/dna_forces.h"
#include "../src/forces/backbone.h"
#include "../src/forces/stacking.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char**argv){
    int model = (argc>1)? std::atoi(argv[1]) : 1;
    double T  = (argc>2)? std::atof(argv[2]) : 0.1;
    double salt=(argc>3)? std::atof(argv[3]) : 0.5;
    const char* top  = (argc>4)? argv[4] : "tests/8bp_duplex/test.top";
    const char* conf = (argc>5)? argv[5] : "tests/8bp_duplex/test.conf";
    Kokkos::initialize(argc,argv);
    {
        ParticleArraysHost host; int N;
        read_topology(top, host, N);
        SimBox box; long long step;
        read_config(conf, host, box, step);
        ParticleArrays dev; dev.allocate(N); copy_to_device(host, dev);
        DNAParams par = (model==2)? make_oxdna2_params(T,salt) : make_oxdna1_params(T);
        NeighborList nl;
        double nl_cut = std::max(2.5, std::sqrt((double)par.cutsq_nb));
        nl.init(nl_cut, 1.0, N, box);
        nl.build(dev, box);

        dev.zero_forces();
        c_number e_back = compute_backbone_forces(dev, par, box);
        c_number e_stk  = compute_stacking_forces(dev, par, box);
        c_number e_nb   = compute_nonbonded_forces(dev, nl, par, box);
        Kokkos::fence();
        c_number tot = e_back + e_stk + e_nb;
        std::printf("Kokkos oxDNA%d  N=%d  T=%.4f salt=%.3f\n", model, N, T, salt);
        std::printf("  backbone(FENE+bexcv) = %12.6f  (%.6f /particle)\n", (double)e_back, (double)e_back/N);
        std::printf("  stacking             = %12.6f  (%.6f /particle)\n", (double)e_stk,  (double)e_stk/N);
        std::printf("  nonbonded(all)       = %12.6f  (%.6f /particle)\n", (double)e_nb,   (double)e_nb/N);
        std::printf("  TOTAL                = %12.6f  (%.6f /particle)\n", (double)tot,    (double)tot/N);
    }
    Kokkos::finalize();
    return 0;
}
