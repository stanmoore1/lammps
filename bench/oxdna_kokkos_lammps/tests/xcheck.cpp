// Prints Kokkos potential energy (total and per group) for one model, for
// cross-checking against the standalone oxDNA split potential energy.
// Groups: backbone(FENE+bonded excv), stacking, nonbonded(excv+HB+cross+coax+DH).
#include <Kokkos_Core.hpp>
#include "../src/simulation.h"
#include "../src/forces/dna_forces.h"
#include "../src/forces/bonded.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char**argv){
    int model = (argc>1)? std::atoi(argv[1]) : 1;
    double T  = (argc>2)? std::atof(argv[2]) : 0.1;
    double salt=(argc>3)? std::atof(argv[3]) : 0.5;
    const char* top  = (argc>4)? argv[4] : "tests/8bp_duplex/test.top";
    const char* conf = (argc>5)? argv[5] : "tests/8bp_duplex/test.conf";
    const char* ftout= (argc>6)? argv[6] : nullptr;  // if set, dump per-particle force/torque
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
        c_number e_nb   = compute_nonbonded_forces(dev, nl, par, box);
        c_number e_bond = compute_bonded_forces(dev, par, box);
        Kokkos::fence();
        c_number tot = e_nb + e_bond;
        std::printf("Kokkos oxDNA%d  N=%d  T=%.4f salt=%.3f\n", model, N, T, salt);
        std::printf("  nonbonded(all)       = %12.6f  (%.6f /particle)\n", (double)e_nb,   (double)e_nb/N);
        std::printf("  bonded(FENE+excv+stk)= %12.6f  (%.6f /particle)\n", (double)e_bond, (double)e_bond/N);
        std::printf("  TOTAL                = %12.6f  (%.6f /particle)\n", (double)tot,    (double)tot/N);

        if (ftout) {
            auto F = Kokkos::create_mirror_view(dev.forces);  Kokkos::deep_copy(F, dev.forces);
            auto Tq= Kokkos::create_mirror_view(dev.torques); Kokkos::deep_copy(Tq, dev.torques);
            FILE* f = std::fopen(ftout, "w");
            for (int i=0;i<N;i++)
                std::fprintf(f, "%d %.10g %.10g %.10g %.10g %.10g %.10g\n", i,
                             (double)F(i,0),(double)F(i,1),(double)F(i,2),
                             (double)Tq(i,0),(double)Tq(i,1),(double)Tq(i,2));
            std::fclose(f);
            std::printf("  wrote per-particle force/torque (lab frame) to %s\n", ftout);
        }
    }
    Kokkos::finalize();
    return 0;
}
