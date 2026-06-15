// Comprehensive self-checking test suite for the oxDNA-Kokkos force field.
//
// For both oxDNA1 and oxDNA2 it checks:
//   1. analytic forces & torques vs central finite difference of the energy
//      (every term: backbone, stacking, nonbonded, and all combined),
//   2. NVE total-energy conservation over a short trajectory,
//   3. Andersen-thermostat temperature control (equipartition).
//
// Exits non-zero if any check exceeds its tolerance.
#include <Kokkos_Core.hpp>
#include "../src/simulation.h"
#include "../src/forces/dna_forces.h"
#include "../src/forces/backbone.h"
#include "../src/forces/stacking.h"
#include "../src/integrator.h"
#include "../src/thermostat.h"
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

static int    g_fail = 0;
static void check(const char* what, double err, double tol) {
    bool ok = (err <= tol);
    if (!ok) g_fail++;
    std::printf("  [%s] %-34s err=%.3e (tol=%.1e)\n", ok ? "PASS" : "FAIL", what, err, tol);
}

static void load(Sys &s, int model) {
    read_topology("tests/8bp_duplex/test.top", s.host, s.N);
    long long step;
    read_config("tests/8bp_duplex/test.conf", s.host, s.box, step);
    s.dev.allocate(s.N);
    copy_to_device(s.host, s.dev);
    s.par = (model == 2) ? make_oxdna2_params(0.1, 0.5) : make_oxdna1_params(0.1);
    double nl_cut = std::max(2.5, std::sqrt((double)s.par.cutsq_nb));
    s.nl.init(nl_cut, 1.0, s.N, s.box);
    s.nl.build(s.dev, s.box);
}

static c_number energy(Sys &s, Term t) {
    s.dev.zero_forces();
    c_number e = 0;
    if (t==BACKBONE || t==ALL) e += compute_backbone_forces(s.dev, s.par, s.box);
    if (t==STACKING || t==ALL) e += compute_stacking_forces(s.dev, s.par, s.box);
    if (t==NONBONDED|| t==ALL) e += compute_nonbonded_forces(s.dev, s.nl, s.par, s.box);
    Kokkos::fence();
    return e;
}

static void rotate(Sys &s, int k, const double e[3], double ang) {
    auto q = Kokkos::create_mirror_view(s.dev.orientations);
    Kokkos::deep_copy(q, s.dev.orientations);
    double c = std::cos(ang/2), sn = std::sin(ang/2);
    double rw=c, rx=sn*e[0], ry=sn*e[1], rz=sn*e[2];
    double qw=q(k,0), qx=q(k,1), qy=q(k,2), qz=q(k,3);
    q(k,0)=rw*qw - rx*qx - ry*qy - rz*qz;
    q(k,1)=rw*qx + rx*qw + ry*qz - rz*qy;
    q(k,2)=rw*qy - rx*qz + ry*qw + rz*qx;
    q(k,3)=rw*qz + rx*qy - ry*qx + rz*qw;
    Kokkos::deep_copy(s.dev.orientations, q);
}

// Relative FD error of forces & torques for one energy term.
static void fd_term(Sys &s, Term t, const char* name) {
    energy(s, t);
    auto F = Kokkos::create_mirror_view(s.dev.forces);  Kokkos::deep_copy(F, s.dev.forces);
    auto Tq= Kokkos::create_mirror_view(s.dev.torques); Kokkos::deep_copy(Tq, s.dev.torques);
    auto P = Kokkos::create_mirror_view(s.dev.poss);
    const double h = 1e-5;
    double maxferr=0, maxterr=0, maxf=1e-12, maxtq=1e-12;
    for (int k=0; k<s.N; k++) {
        for (int d=0; d<3; d++) { maxf=std::max(maxf,std::fabs(F(k,d))); maxtq=std::max(maxtq,std::fabs(Tq(k,d))); }
        for (int d=0; d<3; d++) {
            Kokkos::deep_copy(P, s.dev.poss); double x0=P(k,d);
            P(k,d)=x0+h; Kokkos::deep_copy(s.dev.poss,P); double ep=energy(s,t);
            P(k,d)=x0-h; Kokkos::deep_copy(s.dev.poss,P); double em=energy(s,t);
            P(k,d)=x0;   Kokkos::deep_copy(s.dev.poss,P);
            maxferr=std::max(maxferr, std::fabs(-(ep-em)/(2*h) - F(k,d)));
        }
        for (int d=0; d<3; d++) {
            double e[3]={0,0,0}; e[d]=1;
            rotate(s,k,e, h); double ep=energy(s,t);
            rotate(s,k,e,-2*h); double em=energy(s,t);
            rotate(s,k,e, h);
            maxterr=std::max(maxterr, std::fabs(-(ep-em)/(2*h) - Tq(k,d)));
        }
    }
    char buf[64];
    std::snprintf(buf,sizeof buf,"FD force %s", name);   check(buf, maxferr/maxf, 5e-3);
    std::snprintf(buf,sizeof buf,"FD torque %s", name);  check(buf, maxterr/maxtq, 5e-3);
}

// One NVE / NVT step (mirrors simulation.h).
static void md_step(Sys &s, c_number dt, Thermostat *th, int step, int newt) {
    first_step(s.dev, dt, s.box);
    if (s.nl.needs_rebuild(s.dev, s.box)) s.nl.build(s.dev, s.box);
    s.dev.zero_forces();
    compute_backbone_forces(s.dev, s.par, s.box);
    compute_stacking_forces(s.dev, s.par, s.box);
    compute_nonbonded_forces(s.dev, s.nl, s.par, s.box);
    second_step(s.dev, dt);
    if (th && newt>0 && step%newt==0) th->apply(s.dev);
}

static c_number potential(Sys &s) {
    s.dev.zero_forces();
    c_number e = compute_backbone_forces(s.dev, s.par, s.box);
    e += compute_stacking_forces(s.dev, s.par, s.box);
    e += compute_nonbonded_forces(s.dev, s.nl, s.par, s.box);
    Kokkos::fence();
    return e;
}

static void test_conservation(Sys &s, c_number dt, int nsteps) {
    c_number e0 = potential(s) + kinetic_energy(s.dev);
    double maxdrift = 0;
    for (int step=1; step<=nsteps; step++) {
        md_step(s, dt, nullptr, step, 0);
        if (step % 50 == 0) {
            c_number et = potential(s) + kinetic_energy(s.dev);
            maxdrift = std::max(maxdrift, std::fabs((double)(et - e0)));
        }
    }
    check("NVE energy drift", maxdrift / std::fabs((double)e0), 2e-3);
}

static void test_thermostat(Sys &s, c_number T, int nsteps) {
    Thermostat th; th.init(T, 0.1, 0.1, 777);
    c_number dt = 2e-3;
    // equilibrate, then average kinetic energy
    for (int step=1; step<=nsteps/2; step++) md_step(s, dt, &th, step, 50);
    double sum=0; int cnt=0;
    for (int step=nsteps/2+1; step<=nsteps; step++) {
        md_step(s, dt, &th, step, 50);
        if (step % 20 == 0) { sum += (double)kinetic_energy(s.dev); cnt++; }
    }
    double meanK = sum/cnt;
    double expect = 3.0 * s.N * (double)T;     // (6N/2) kT, 6 DOF/particle
    check("thermostat <Ekin> vs 3NkT", std::fabs(meanK-expect)/expect, 0.25);
}

int main(int argc, char**argv){
    Kokkos::initialize(argc,argv);
    {
        const char* names[4]={"backbone","stacking","nonbonded","all"};
        const c_number dts[3] = {0, 5e-4, 1e-4};  // [model index]; oxDNA2 needs smaller dt
        for (int model=1; model<=2; model++) {
            std::printf("================ oxDNA%d ================\n", model);
            { Sys s; load(s, model);
              for (int t=0;t<4;t++) fd_term(s,(Term)t,names[t]); }
            { Sys s; load(s, model);
              test_conservation(s, dts[model], 3000); }
            { Sys s; load(s, model);
              test_thermostat(s, 0.1, 4000); }
        }
        std::printf("\n%s (%d failure%s)\n", g_fail==0?"ALL TESTS PASSED":"TESTS FAILED",
                    g_fail, g_fail==1?"":"s");
    }
    Kokkos::finalize();
    return g_fail==0 ? 0 : 1;
}
