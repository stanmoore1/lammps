// Comprehensive self-checking test suite for the oxDNA-Kokkos force field
// (LAMMPS-faithful kernels).
//
// For oxDNA1, oxDNA2 and oxDNA3 it checks (on the 8bp duplex):
//   1. analytic forces & torques vs central finite difference of the energy
//      (every term group; for oxDNA3 every LAMMPS-style kernel separately --
//      fene, stk, excv (split into its bonded / nonbonded halves), hbond,
//      xstk, coaxstk, dh -- in lean AND lammps_overhead mode, on the relaxed
//      duplex, on a nicked + perturbed duplex in which every term is active,
//      with the LAMMPS-only terminal coaxial stacking, and on a configuration
//      with active stacking cos(phi1/phi2) modulation),
//   2. NVE total-energy conservation over a short trajectory,
//   3. Andersen-thermostat temperature control (equipartition).
// It also runs the FD check on a nicked 8bp duplex (tests/8bp_nicked, coaxial
// stacking across the nick), for oxDNA1, oxDNA2 and oxDNA2 with the LAMMPS-only
// terminal-nucleotide coaxial stacking (+ blunt-end theta4 lobe), and in
// lammps_overhead mode.
// The finite-difference checks need a double-precision build (single
// precision uses a larger FD step and looser tolerances).
//
// Exits non-zero if any check exceeds its tolerance.
#include <Kokkos_Core.hpp>
#include "../src/simulation.h"
#include "../src/forces/dna_forces.h"
#include "../src/forces/bonded.h"
#include "../src/forces/dna3_kernels.h"
#include "../src/integrator.h"
#include "../src/thermostat.h"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <string>

enum Term { NONBONDED, BONDED, ALL };

struct Sys {
    ParticleArraysHost host;
    ParticleArrays dev;
    DNAParams par;
    DNA3Model m3;
    int model = 1;
    NeighborList nl;
    SimBox box;
    int N;
};

static int    g_fail = 0;
static void check(const char* what, double err, double tol) {
    bool ok = (err <= tol);
    if (!ok) g_fail++;
    std::printf("  [%s] %-38s err=%.3e (tol=%.1e)\n", ok ? "PASS" : "FAIL", what, err, tol);
}

static bool g_overhead = false;
static const bool DOUBLE = (sizeof(c_number) == 8);

static void load(Sys &s, int model, const char *top = "tests/8bp_duplex/test.top",
                 const char *conf = "tests/8bp_duplex/test.conf", bool terminal = false) {
    s.model = model;
    read_topology(top, s.host, s.N);
    long long step;
    read_config(conf, s.host, s.box, step);
    s.dev.allocate(s.N);
    copy_to_device(s.host, s.dev);
    s.par = (model == 2) ? make_oxdna2_params(0.1, 0.5) : make_oxdna1_params(0.1);
    s.par.cxst_terminal_only = terminal;
    s.par.cxst_t4_blunt      = terminal;
    double nl_cut = std::max(2.5, std::sqrt((double)s.par.cutsq_nb));
    s.nl.init(nl_cut, 1.0, s.N, s.box);
    s.nl.build(s.dev, s.box);
}

// Deterministic perturbation (positions by up to +-dx, orientations by
// rotations of up to +-dang about random axes) to populate the terms that are
// zero in a relaxed duplex (same generator as bench/oxdna_kokkos).
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

// oxDNA3 system: the relaxed duplex (test_dna3.conf), optionally with the
// nicked topology and the perturbation of bench/oxdna_kokkos's fd_test.
static void load3(Sys &s, bool nicked = false, bool consistent_gamma = false, bool terminal = false) {
    s.model = 3;
    const char *top  = nicked ? "tests/8bp_duplex/test_dna3_nicked.top" : "tests/8bp_duplex/test.top";
    read_topology(top, s.host, s.N);
    long long step;
    read_config("tests/8bp_duplex/test_dna3.conf", s.host, s.box, step);
    if (nicked) perturb(s.host, s.N, 0.03, 0.15, 2024);
    s.dev.allocate(s.N);
    copy_to_device(s.host, s.dev);
    SimConfig cfg; cfg.T = 0.1; cfg.salt = 0.5;
    cfg.dna3_consistent_gamma = consistent_gamma;
    s.m3 = make_dna3_model(dna3_options(cfg), terminal);
    const double skin = 1.0;
    double nl_cut = std::max(2.5, std::sqrt((double)s.m3.p.cutsq_nb));
    s.nl.init(nl_cut, skin, s.N, s.box);
    const double scr = std::sqrt((double)s.m3.p.screen_cutsq) + skin;   // as Simulation::init
    s.nl.screen_cutsq = static_cast<c_number>(scr * scr);
    s.nl.build(s.dev, s.box);
}

static c_number energy(Sys &s, Term t) {
    if (s.model == 3) {
        const int mask = (t == NONBONDED) ? (dna3k::EXCV | dna3k::HBOND | dna3k::XSTK | dna3k::COAXSTK | dna3k::DH)
                       : (t == BONDED)    ? (dna3k::STK | dna3k::FENE) : dna3k::ALL;
        return compute_forces_dna3(s.dev, s.nl, s.m3, s.box, g_overhead, mask);
    }
    s.dev.zero_forces();
    c_number e = 0;
    if (t==NONBONDED|| t==ALL) e += compute_nonbonded_forces(s.dev, s.nl, s.par, s.box, true, g_overhead);
    if (t==BONDED   || t==ALL) e += compute_bonded_forces(s.dev, s.par, s.box, true, g_overhead);
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
    const double h = DOUBLE ? 1e-5 : 2e-3;
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
    char buf[80];
    if (s.model == 3) std::printf("  %-24s E = %12.6f  max|F| = %.3e  max|T| = %.3e\n", name, e0, maxf, maxtq);
    std::snprintf(buf,sizeof buf,"FD force %s", name);   check(buf, maxferr/maxf, tol);
    std::snprintf(buf,sizeof buf,"FD torque %s", name);  check(buf, maxterr/maxtq, tol);
}

static void fd_term(Sys &s, Term t, const char* name) {
    fd_generic(s, [t](Sys &ss) { return (double)energy(ss, t); }, name);
}

// oxDNA3 forces are the exact gradient (up to FD truncation/rounding) in double
static const double TOL3 = DOUBLE ? 1e-6 : 5e-3;

// Every oxDNA3 kernel separately (the excv kernel also split into its
// bonded / nonbonded halves), plus the pair/bond groups and everything.
// require_nonzero: the term must actually contribute in this configuration.
static void fd_dna3_all_terms(Sys &s, bool require_nonzero) {
    using namespace dna3k;
    auto run = [&](auto E, const char *name) {
        if (require_nonzero) {
            double e = E(s);
            check((std::string("term active: ") + name).c_str(), (e != 0) ? 0.0 : 1.0, 0.0);
        }
        fd_generic(s, E, name, TOL3);
    };
    auto kern = [](int mask) {
        return [mask](Sys &ss) { return (double)compute_forces_dna3(ss.dev, ss.nl, ss.m3, ss.box, g_overhead, mask); };
    };
    auto excv_part = [](bool bonded) {
        return [bonded](Sys &ss) { return (double)compute_excv_part_dna3(ss.dev, ss.nl, ss.m3, ss.box, g_overhead, bonded); };
    };
    run(kern(FENE),            "fene");
    run(excv_part(true),       "excv (bonded)");
    run(kern(STK),             "stk");
    run(excv_part(false),      "excv (nonbonded)");
    run(kern(HBOND),           "hbond");
    run(kern(XSTK),            "xstk");
    run(kern(COAXSTK),         "coaxstk");
    run(kern(DH),              "dh");
    run(kern(EXCV),            "excv");
    run(kern(EXCV | HBOND | XSTK | COAXSTK | DH), "nonbonded");
    run(kern(STK | FENE),      "bonded");
    run(kern(dna3k::ALL),      "all");
}

// Stacking with the cos(phi1)/cos(phi2) modulation active: in the relaxed
// duplex all cos(phi) >= 0 (f5 == 1), so rotate a few nucleotides about their
// own a1 axis to push cos(phi) into the (XC, 0) range. Upstream's force uses
// GAMMA = 0.74 while the oxDNA3 stacking site implies 0.77, so the upstream-
// faithful forces deviate slightly from -grad E here (reported, looser tol);
// with the consistent gamma the analytic gradient must be exact.
static void fd_dna3_stacking_phi() {
    Sys s; load3(s);
    for (int k : {1, 12}) {
        auto q = Kokkos::create_mirror_view(s.dev.orientations);
        Kokkos::deep_copy(q, s.dev.orientations);
        c_number x[3], y[3], z[3];
        get_vectors_from_quat(q(k,0), q(k,1), q(k,2), q(k,3), x, y, z);
        double e[3] = {x[0], x[1], x[2]};
        rotate(s, k, e, 0.8);
    }
    auto E = [](Sys &ss) { return (double)compute_forces_dna3(ss.dev, ss.nl, ss.m3, ss.box, g_overhead, dna3k::STK); };
    fd_generic(s, E, "stk phi (upstream)", 5e-3);
    auto F0 = Kokkos::create_mirror(s.dev.forces);
    E(s); Kokkos::deep_copy(F0, s.dev.forces);
    SimConfig cfg; cfg.T = 0.1; cfg.salt = 0.5; cfg.dna3_consistent_gamma = true;
    s.m3 = make_dna3_model(dna3_options(cfg));
    fd_generic(s, E, "stk phi (consist.)", TOL3);
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
    energy(s, ALL);
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
        for (int model=1; model<=2; model++) {
            std::printf("================ oxDNA%d ================\n", model);
            { Sys s; load(s, model);
              for (int t=0;t<3;t++) fd_term(s,(Term)t,names[t]); }
            { Sys s; load(s, model);
              test_conservation(s, dts[model], 3000); }
            { Sys s; load(s, model);
              test_thermostat(s, 0.1, 12000); }
        }
        struct NickCase { int model; bool terminal; bool overhead; const char *label; };
        const NickCase nick[4] = {
            {1, false, false, "oxDNA1, nicked duplex"},
            {2, false, false, "oxDNA2, nicked duplex"},
            {2, true,  false, "oxDNA2, nicked, LAMMPS terminal coaxstk"},
            {2, true,  true,  "oxDNA2, nicked, terminal coaxstk, lammps_overhead"}};
        for (const NickCase &c : nick) {
            std::printf("================ %s ================\n", c.label);
            g_overhead = c.overhead;
            Sys s; load(s, c.model, "tests/8bp_nicked/test.top", "tests/8bp_nicked/test.conf", c.terminal);
            std::printf("  E_nonbonded = %.10f\n", (double)energy(s, NONBONDED));
            for (int t=0;t<3;t++) fd_term(s,(Term)t,names[t]);
        }
        g_overhead = false;

        // ---------------- oxDNA3 ----------------
        for (int ov = 0; ov <= 1; ov++) {
            g_overhead = (ov == 1);
            const char *mode = g_overhead ? "lammps_overhead" : "lean";
            std::printf("================ oxDNA3 (%s kernels) ================\n", mode);
            std::printf("-- relaxed duplex (tests/8bp_duplex/test_dna3.conf)\n");
            { Sys s; load3(s); fd_dna3_all_terms(s, false); }
            std::printf("-- nicked + perturbed duplex (all terms active)\n");
            { Sys s; load3(s, true); fd_dna3_all_terms(s, true); }
            std::printf("-- nicked + perturbed duplex, LAMMPS terminal coaxstk + blunt theta4 lobe\n");
            { Sys s; load3(s, true, false, true);
              auto E = [](Sys &ss) { return (double)compute_forces_dna3(ss.dev, ss.nl, ss.m3, ss.box, g_overhead, dna3k::COAXSTK); };
              check("term active: coaxstk (terminal)", E(s) != 0 ? 0.0 : 1.0, 0.0);
              fd_generic(s, E, "coaxstk (terminal)", TOL3);
              fd_generic(s, [](Sys &ss) { return (double)compute_forces_dna3(ss.dev, ss.nl, ss.m3, ss.box, g_overhead, dna3k::ALL); },
                         "all (terminal)", TOL3);
              // Blunt-end lobe: turning the nick's 5' end (particle 4) by pi
              // about its a1 axis maps theta4 -> pi - theta4 and leaves theta1,
              // r and the symmetric theta5/theta6 terms unchanged, so the
              // mirrored lobe must reproduce the original coaxial energy.
              const double e_orig = E(s);
              auto q = Kokkos::create_mirror_view(s.dev.orientations);
              Kokkos::deep_copy(q, s.dev.orientations);
              c_number x[3], y[3], z[3];
              get_vectors_from_quat(q(4,0), q(4,1), q(4,2), q(4,3), x, y, z);
              const double ax[3] = {x[0], x[1], x[2]};
              rotate(s, 4, ax, 3.141592653589793);
              const double e_flip = E(s);
              std::printf("  coaxstk: E(orig) = %.12f  E(flipped, blunt lobe) = %.12f\n", e_orig, e_flip);
              check("blunt lobe: E(flipped) == E(orig)", std::fabs(e_flip - e_orig) / std::fabs(e_orig),
                    DOUBLE ? 1e-9 : 1e-4);
              fd_generic(s, E, "coaxstk (blunt lobe)", TOL3); }
            std::printf("-- stacking with active cos(phi1/phi2) modulation\n");
            fd_dna3_stacking_phi();
        }
        g_overhead = false;
        {
            std::printf("================ oxDNA3 MD ================\n");
            for (int ov = 0; ov <= 1; ov++) {
                g_overhead = (ov == 1);
                std::printf("-- NVE (%s kernels)\n", g_overhead ? "lammps_overhead" : "lean");
                Sys s; load3(s);
                test_conservation(s, dts[3], 3000);
            }
            g_overhead = false;
            std::printf("-- NVE with the consistent stacking gamma (dna3_consistent_gamma = 1)\n");
            { Sys s; load3(s, false, true); test_conservation(s, dts[3], 3000); }
            { Sys s; load3(s); test_thermostat(s, 0.1, 12000); }
        }
        std::printf("\n%s (%d failure%s)\n", g_fail==0?"ALL TESTS PASSED":"TESTS FAILED",
                    g_fail, g_fail==1?"":"s");
    }
    Kokkos::finalize();
    return g_fail==0 ? 0 : 1;
}
