// Comprehensive self-checking test suite for the oxDNA-Kokkos force field.
//
// For oxDNA1, oxDNA2 and oxDNA3 it checks:
//   1. analytic forces & torques vs central finite difference of the energy
//      (every term: backbone, stacking, nonbonded, and all combined; for
//      oxDNA3 each of the 8 upstream terms separately, on the relaxed duplex
//      and on a nicked + perturbed duplex that switches on the terms that are
//      zero in an ideal duplex: bonded/nonbonded excluded volume, coaxial
//      stacking),
//   2. NVE total-energy conservation over a short trajectory,
//   3. Andersen-thermostat temperature control (equipartition).
//
// Exits non-zero if any check exceeds its tolerance.
#include <Kokkos_Core.hpp>
#include "../src/simulation.h"
#include "../src/forces/dna_forces.h"
#include "../src/forces/bonded.h"
#include "../src/forces/dna3_forces.h"
#include "../src/integrator.h"
#include "../src/thermostat.h"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <type_traits>

enum Term { NONBONDED, BONDED, ALL };

struct Sys {
    ParticleArraysHost host;
    ParticleArrays dev;
    DNAParams par;
    DNA3Params par3;
    int model = 1;
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

// Deterministic perturbation (positions by up to +-dx, orientations by
// rotations of up to +-dang about random axes) to populate the terms that are
// zero in a relaxed duplex.
static void perturb(ParticleArraysHost &h, int N, double dx, double dang, uint64_t seed) {
    auto rnd = [&seed]() {   // uniform in [-1, 1)
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return ((seed >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
    };
    for (int k = 0; k < N; k++) {
        for (int d = 0; d < 3; d++) h.poss(k, d) += dx * rnd();
        double e[3] = {rnd(), rnd(), rnd()};
        double n = std::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        double ang = dang * rnd();
        double c = std::cos(ang/2), sn = std::sin(ang/2);
        double rw=c, rx=sn*e[0]/n, ry=sn*e[1]/n, rz=sn*e[2]/n;
        double qw=h.orientations(k,0), qx=h.orientations(k,1), qy=h.orientations(k,2), qz=h.orientations(k,3);
        h.orientations(k,0)=rw*qw - rx*qx - ry*qy - rz*qz;
        h.orientations(k,1)=rw*qx + rx*qw + ry*qz - rz*qy;
        h.orientations(k,2)=rw*qy - rx*qz + ry*qw + rz*qx;
        h.orientations(k,3)=rw*qz + rx*qy - ry*qx + rz*qw;
    }
}

static void load(Sys &s, int model, bool nicked = false, bool consistent_gamma = false) {
    s.model = model;
    const char *top  = (model == 3 && nicked) ? "tests/8bp_duplex/test_dna3_nicked.top"
                                              : "tests/8bp_duplex/test.top";
    // oxDNA3 uses a duplex relaxed with the oxDNA3 model: in test.conf (an
    // oxDNA1 configuration) the first bond is outside the oxDNA3 FENE range.
    const char *conf = (model == 3) ? "tests/8bp_duplex/test_dna3.conf"
                                    : "tests/8bp_duplex/test.conf";
    read_topology(top, s.host, s.N);
    long long step;
    read_config(conf, s.host, s.box, step);
    if (model == 3 && nicked) perturb(s.host, s.N, 0.03, 0.15, 2024);
    s.dev.allocate(s.N);
    copy_to_device(s.host, s.dev);
    double cutsq;
    if (model == 3) {
        SimConfig cfg; cfg.T = 0.1; cfg.salt = 0.5;
        cfg.dna3_consistent_gamma = consistent_gamma;
        s.par3 = make_oxdna3_params(dna3_options(cfg));
        cutsq = s.par3.cutsq_nb;
    } else {
        s.par = (model == 2) ? make_oxdna2_params(0.1, 0.5) : make_oxdna1_params(0.1);
        cutsq = s.par.cutsq_nb;
    }
    double nl_cut = std::max(2.5, std::sqrt(cutsq));
    s.nl.init(nl_cut, 1.0, s.N, s.box);
    s.nl.build(s.dev, s.box);
}

static c_number energy(Sys &s, Term t) {
    s.dev.zero_forces();
    c_number e = 0;
    if (s.model == 3) {
        if (t==NONBONDED|| t==ALL) e += compute_nonbonded_forces_dna3(s.dev, s.nl, s.par3, s.box);
        if (t==BONDED   || t==ALL) e += compute_bonded_forces_dna3(s.dev, s.par3, s.box);
    } else {
        if (t==NONBONDED|| t==ALL) e += compute_nonbonded_forces(s.dev, s.nl, s.par, s.box);
        if (t==BONDED   || t==ALL) e += compute_bonded_forces(s.dev, s.par, s.box);
    }
    Kokkos::fence();
    return e;
}

// oxDNA3 single-term energy (TERMS mask), forces/torques left in s.dev
template <int TERMS>
static c_number energy3(Sys &s) {
    s.dev.zero_forces();
    c_number e = 0;
    if constexpr ((TERMS & dna3::NONBONDED) != 0)
        e += compute_nonbonded_forces_dna3<TERMS & dna3::NONBONDED>(s.dev, s.nl, s.par3, s.box);
    if constexpr ((TERMS & dna3::BONDED) != 0)
        e += compute_bonded_forces_dna3<TERMS & dna3::BONDED>(s.dev, s.par3, s.box);
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

// Relative FD error of forces & torques for one energy function E(s).
template <class EFun>
static void fd_generic(Sys &s, EFun E, const char* name, double tol = 5e-3) {
    double e0 = E(s);
    // create_mirror (not _view): on host backends a mirror view would alias
    // s.dev.forces and be overwritten by the displaced evaluations below
    auto F = Kokkos::create_mirror(s.dev.forces);  Kokkos::deep_copy(F, s.dev.forces);
    auto Tq= Kokkos::create_mirror(s.dev.torques); Kokkos::deep_copy(Tq, s.dev.torques);
    auto P = Kokkos::create_mirror_view(s.dev.poss);
    // FD step: 1e-5 in double; in single precision the energy rounding error
    // (~1e-7 relative) needs a larger step to give a meaningful derivative
    const double h = (sizeof(c_number) == 8) ? 1e-5 : 2e-3;
    double maxferr=0, maxterr=0, maxf=1e-12, maxtq=1e-12;
    for (int k=0; k<s.N; k++) {
        for (int d=0; d<3; d++) { maxf=std::max(maxf,std::fabs((double)F(k,d))); maxtq=std::max(maxtq,std::fabs((double)Tq(k,d))); }
        for (int d=0; d<3; d++) {
            Kokkos::deep_copy(P, s.dev.poss); double x0=P(k,d);
            P(k,d)=x0+h; Kokkos::deep_copy(s.dev.poss,P); double ep=E(s);
            P(k,d)=x0-h; Kokkos::deep_copy(s.dev.poss,P); double em=E(s);
            P(k,d)=x0;   Kokkos::deep_copy(s.dev.poss,P);
            maxferr=std::max(maxferr, std::fabs(-(ep-em)/(2*h) - F(k,d)));
        }
        for (int d=0; d<3; d++) {
            double e[3]={0,0,0}; e[d]=1;
            rotate(s,k,e, h); double ep=E(s);
            rotate(s,k,e,-2*h); double em=E(s);
            rotate(s,k,e, h);
            maxterr=std::max(maxterr, std::fabs(-(ep-em)/(2*h) - Tq(k,d)));
        }
    }
    char buf[64];
    if (s.model == 3) std::printf("  %-22s E = %12.6f  max|F| = %.3e  max|T| = %.3e\n", name, e0, maxf, maxtq);
    std::snprintf(buf,sizeof buf,"FD force %s", name);   check(buf, maxferr/maxf, tol);
    std::snprintf(buf,sizeof buf,"FD torque %s", name);  check(buf, maxterr/maxtq, tol);
}

static void fd_term(Sys &s, Term t, const char* name) {
    fd_generic(s, [t](Sys &ss) { return (double)energy(ss, t); }, name);
}

// Every oxDNA3 term separately, plus the bonded/nonbonded/all groups.
// require_nonzero: the term must actually contribute in this configuration.
// oxDNA3 forces are the exact gradient (up to FD truncation/rounding) in double
static const double TOL3 = (sizeof(c_number) == 8) ? 1e-6 : 5e-3;

static void fd_dna3_all_terms(Sys &s, bool require_nonzero) {
    auto run = [&](auto tag, const char *name) {
        constexpr int M = decltype(tag)::value;
        if (require_nonzero) {
            double e = energy3<M>(s);
            check((std::string("term active: ") + name).c_str(), (e != 0) ? 0.0 : 1.0, 0.0);
        }
        fd_generic(s, [](Sys &ss) { return (double)energy3<M>(ss); }, name, TOL3);
    };
    using std::integral_constant;
    run(integral_constant<int, dna3::BACKBONE>{},                  "FENE");
    run(integral_constant<int, dna3::BONDED_EXCLUDED_VOLUME>{},    "bonded excv");
    run(integral_constant<int, dna3::STACKING>{},                  "stacking");
    run(integral_constant<int, dna3::NONBONDED_EXCLUDED_VOLUME>{}, "nonbonded excv");
    run(integral_constant<int, dna3::HYDROGEN_BONDING>{},          "H-bonding");
    run(integral_constant<int, dna3::CROSS_STACKING>{},            "cross stacking");
    run(integral_constant<int, dna3::COAXIAL_STACKING>{},          "coaxial stacking");
    run(integral_constant<int, dna3::DEBYE_HUCKEL>{},              "Debye-Huckel");
    run(integral_constant<int, dna3::NONBONDED>{},                 "nonbonded");
    run(integral_constant<int, dna3::BONDED>{},                    "bonded");
    run(integral_constant<int, dna3::ALL>{},                       "all");
}

// Stacking with the cos(phi1)/cos(phi2) modulation active: in the relaxed
// duplex all cos(phi) >= 0 (f5 == 1), so rotate a few nucleotides about their
// own a1 axis to push cos(phi) into the (XC, 0) range. Upstream's force uses
// GAMMA = 0.74 while the oxDNA3 stacking site implies 0.77, so the upstream-
// faithful forces deviate slightly from -grad E here (reported, looser tol);
// with the consistent gamma the analytic gradient must be exact.
static void fd_dna3_stacking_phi() {
    Sys s; load(s, 3);
    for (int k : {1, 12}) {
        auto q = Kokkos::create_mirror_view(s.dev.orientations);
        Kokkos::deep_copy(q, s.dev.orientations);
        c_number x[3], y[3], z[3];
        get_vectors_from_quat(q(k,0), q(k,1), q(k,2), q(k,3), x, y, z);
        double e[3] = {x[0], x[1], x[2]};
        rotate(s, k, e, 0.8);
    }
    auto E = [](Sys &ss) { return (double)energy3<dna3::STACKING>(ss); };
    fd_generic(s, E, "stacking phi (upstream)", 5e-3);
    auto F0 = Kokkos::create_mirror(s.dev.forces);
    E(s); Kokkos::deep_copy(F0, s.dev.forces);
    SimConfig cfg; cfg.T = 0.1; cfg.salt = 0.5; cfg.dna3_consistent_gamma = true;
    s.par3 = make_oxdna3_params(dna3_options(cfg));
    fd_generic(s, E, "stacking phi (consist.)", TOL3);
    auto F1 = Kokkos::create_mirror(s.dev.forces);
    E(s); Kokkos::deep_copy(F1, s.dev.forces);
    double dmax = 0;
    for (int k = 0; k < s.N; k++)
        for (int d = 0; d < 3; d++) dmax = std::max(dmax, std::fabs((double)(F0(k,d) - F1(k,d))));
    // the dihedral derivative path must actually be exercised
    check("phi derivatives active (dF != 0)", dmax > 0 ? 0.0 : 1.0, 0.0);
    std::printf("  max |F(upstream gamma) - F(consistent gamma)| = %.3e\n", dmax);
}

static void compute_all(Sys &s) {
    s.dev.zero_forces();
    if (s.model == 3) {
        compute_nonbonded_forces_dna3(s.dev, s.nl, s.par3, s.box);
        compute_bonded_forces_dna3(s.dev, s.par3, s.box);
    } else {
        compute_nonbonded_forces(s.dev, s.nl, s.par, s.box);
        compute_bonded_forces(s.dev, s.par, s.box);
    }
}

// One NVE / NVT step (mirrors simulation.h).
static void md_step(Sys &s, c_number dt, Thermostat *th, int step, int newt) {
    first_step(s.dev, dt, s.box);
    if (s.nl.needs_rebuild(s.dev, s.box)) s.nl.build(s.dev, s.box);
    compute_all(s);
    second_step(s.dev, dt);
    if (th && newt>0 && step%newt==0) th->apply(s.dev);
}

static c_number potential(Sys &s) {
    return energy(s, ALL);
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
    c_number dt = 2e-3;
    // strong direct coupling (pt=0.3) so the small system equilibrates quickly;
    // this checks equipartition, not the diff_coeff -> pt mapping.
    Thermostat th; th.init(T, 50, dt, 0.0, 0.3, 777);
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
        const char* names[3]={"nonbonded","bonded","all"};
        const c_number dts[4] = {0, 5e-4, 1e-4, 5e-4};  // [model]; oxDNA2 needs smaller dt
        for (int model=1; model<=3; model++) {
            std::printf("================ oxDNA%d ================\n", model);
            if (model == 3) {
                std::printf("-- relaxed duplex (tests/8bp_duplex/test_dna3.conf)\n");
                { Sys s; load(s, model); fd_dna3_all_terms(s, false); }
                std::printf("-- nicked + perturbed duplex (all 8 terms active)\n");
                { Sys s; load(s, model, true); fd_dna3_all_terms(s, true); }
                std::printf("-- stacking with active cos(phi1/phi2) modulation\n");
                fd_dna3_stacking_phi();
            } else {
                Sys s; load(s, model);
                for (int t=0;t<3;t++) fd_term(s,(Term)t,names[t]);
            }
            { Sys s; load(s, model);
              test_conservation(s, dts[model], 3000); }
            if (model == 3) {
                std::printf("-- NVE with the consistent stacking gamma (dna3_consistent_gamma = 1)\n");
                Sys s; load(s, model, false, true);
                test_conservation(s, dts[model], 3000);
            }
            { Sys s; load(s, model);
              test_thermostat(s, 0.1, 12000); }
        }
        std::printf("\n%s (%d failure%s)\n", g_fail==0?"ALL TESTS PASSED":"TESTS FAILED",
                    g_fail, g_fail==1?"":"s");
    }
    Kokkos::finalize();
    return g_fail==0 ? 0 : 1;
}
