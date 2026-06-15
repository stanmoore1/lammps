// Finite-difference validation of forces & torques against -dE/dx.
// Builds the configuration, then for selected DOFs compares the analytic
// force/torque to a central finite difference of the total potential energy.
#include <Kokkos_Core.hpp>
#include "../src/simulation.h"
#include "../src/forces/dna_forces.h"
#include "../src/forces/backbone.h"
#include "../src/forces/stacking.h"
#include <cstdio>
#include <cmath>

enum Term { BACKBONE, STACKING, NONBONDED, ALL };

struct Sys {
    ParticleArraysHost host;
    ParticleArrays dev;
    DNAParams par;
    NeighborList nl;
    SimBox box;
    int N;
};

static c_number energy(Sys &s, Term t) {
    s.dev.zero_forces();
    c_number e = 0;
    if (t==BACKBONE || t==ALL) e += compute_backbone_forces(s.dev, s.par, s.box);
    if (t==STACKING || t==ALL) e += compute_stacking_forces(s.dev, s.par, s.box);
    if (t==NONBONDED|| t==ALL) e += compute_nonbonded_forces(s.dev, s.nl, s.par, s.box);
    Kokkos::fence();
    return e;
}

// rotate particle k's quaternion about lab axis e (unit) by angle ang (LEFT mult = lab frame)
static void rotate(Sys &s, int k, const double e[3], double ang) {
    auto q = Kokkos::create_mirror_view(s.dev.orientations);
    Kokkos::deep_copy(q, s.dev.orientations);
    double c = std::cos(ang/2), sn = std::sin(ang/2);
    double rw=c, rx=sn*e[0], ry=sn*e[1], rz=sn*e[2];
    double qw=q(k,0), qx=q(k,1), qy=q(k,2), qz=q(k,3);
    // r * q  (left multiply)
    q(k,0)=rw*qw - rx*qx - ry*qy - rz*qz;
    q(k,1)=rw*qx + rx*qw + ry*qz - rz*qy;
    q(k,2)=rw*qy - rx*qz + ry*qw + rz*qx;
    q(k,3)=rw*qz + rx*qy - ry*qx + rz*qw;
    Kokkos::deep_copy(s.dev.orientations, q);
}

int main(int argc, char**argv){
    Kokkos::initialize(argc,argv);
    {
        Sys s;
        read_topology("tests/8bp_duplex/test.top", s.host, s.N);
        long long step;
        read_config("tests/8bp_duplex/test.conf", s.host, s.box, step);
        s.dev.allocate(s.N);
        copy_to_device(s.host, s.dev);
        s.par = make_oxdna1_params(0.1);
        s.nl.init(2.5, 0.3, s.N, s.box);
        s.nl.build(s.dev, s.box);

        const double h = 1e-5;
        const char* names[4]={"BACKBONE","STACKING","NONBONDED","ALL"};
        for (int term=0; term<4; term++) {
            // analytic forces/torques
            energy(s, (Term)term);
            auto F = Kokkos::create_mirror_view(s.dev.forces);  Kokkos::deep_copy(F, s.dev.forces);
            auto Tq= Kokkos::create_mirror_view(s.dev.torques); Kokkos::deep_copy(Tq, s.dev.torques);
            auto P = Kokkos::create_mirror_view(s.dev.poss);

            double maxferr=0, maxterr=0;
            for (int k=0; k<s.N; k++) {
                // translational
                for (int d=0; d<3; d++) {
                    Kokkos::deep_copy(P, s.dev.poss); double x0=P(k,d);
                    P(k,d)=x0+h; Kokkos::deep_copy(s.dev.poss,P); double ep=energy(s,(Term)term);
                    P(k,d)=x0-h; Kokkos::deep_copy(s.dev.poss,P); double em=energy(s,(Term)term);
                    P(k,d)=x0;   Kokkos::deep_copy(s.dev.poss,P);
                    double fnum = -(ep-em)/(2*h);
                    double err = std::fabs(fnum - F(k,d));
                    if (err>maxferr) maxferr=err;
                }
                // rotational (lab-frame torque) about x,y,z
                for (int d=0; d<3; d++) {
                    double e[3]={0,0,0}; e[d]=1;
                    rotate(s,k,e, h); double ep=energy(s,(Term)term);
                    rotate(s,k,e,-2*h); double em=energy(s,(Term)term);
                    rotate(s,k,e, h); // restore
                    double tnum = -(ep-em)/(2*h);
                    double err = std::fabs(tnum - Tq(k,d));
                    if (err>maxterr) maxterr=err;
                }
            }
            std::printf("%-10s  max|F_analytic - F_fd| = %.3e   max|T_analytic - T_fd| = %.3e\n",
                        names[term], maxferr, maxterr);
        }
    }
    Kokkos::finalize();
    return 0;
}
