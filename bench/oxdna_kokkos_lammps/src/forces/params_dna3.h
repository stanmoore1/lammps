#pragma once

// oxDNA3 force-field parameters (sequence-dependent, tetramer tables).
//
// Faithful host-side port of the standalone oxDNA DNA3Interaction constructor
// and DNA3Interaction::init() (upstream oxDNA c2c74cc0, 2026-09-21): the
// defaults are the model.h constants (float literals, exactly as upstream),
// the independent parameters are read from the sequence-dependence file
// (oxDNA3_sequence_dependent_parameters.txt, parsed like getInputFloat: atof
// then rounded to float), the strand-end (NO_TYPE) entries are averaged, the
// symmetric coaxial K is built, and the "enslaved" smoothing parameters are
// recomputed from continuity/differentiability. All of this runs in double on
// the host; the finished tables are then uploaded ONCE to a single device View
// (float in single precision, c_number otherwise), the Kokkos analogue of the
// CUDA __constant__/__device__ MD_*_SD arrays filled by
// CUDADNA3Interaction::cuda_init().
//
// Table layout (identical to upstream MultiDimArray / CUDA OxDNA3Params):
//   T(i, j, k, l) with dims [6][5][5][6], row-major, i.e.
//   index = ((i*5 + j)*5 + k)*6 + l, where for a bonded pair (p, q = p.n3)
//   (i, j, k, l) = (type(q.n3), type(q), type(p), type(p.n5)) and NO_TYPE = 5
//   marks a missing neighbour (strand end). Types use the oxDNA convention
//   A=0, G=1, C=2, T=3 (NOT the A,C,G,T btype order used elsewhere here).
// Families of tables (e.g. the 7 excluded-volume sigmas) are stored back to
// back; see the dna3sd:: offsets below.

#include "../types.h"
#include "lmp_cuts.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dna3sd {

constexpr int DIM_A   = 6;              // TETRAMER_DIM_A (types incl. dummy + NO_TYPE)
constexpr int DIM_B   = 5;              // TETRAMER_DIM_B
constexpr int NO_TYPE = 5;              // missing neighbour (strand end)
constexpr int TSIZE   = DIM_A * DIM_B * DIM_B * DIM_A;   // 900 entries per table

KOKKOS_INLINE_FUNCTION
constexpr int tidx(int i, int j, int k, int l) { return ((i * DIM_B + j) * DIM_B + k) * DIM_A + l; }

// Table offsets (in units of TSIZE). Sub-indices follow upstream model.h, e.g.
// F1_EPS + HYDR_F1, F4_A + STCK_F4_THETA5, EXCL_S + 4 (bonded base-base).
constexpr int FENE_R0 = 0, FENE_DELTA = 1, FENE_DELTA2 = 2;
constexpr int EXCL_S = 3, EXCL_R = 10, EXCL_B = 17, EXCL_RC = 24;  // 7 each
constexpr int F1_EPS = 31, F1_A = 33, F1_RC = 35, F1_R0 = 37, F1_BLOW = 39, F1_BHIGH = 41;
constexpr int F1_RLOW = 43, F1_RHIGH = 45, F1_RCLOW = 47, F1_RCHIGH = 49, F1_SHIFT = 51;  // 2 each
constexpr int F2_K = 53, F2_K_SYMM = 57, F2_RC = 61, F2_R0 = 65, F2_BLOW = 69, F2_RLOW = 73;
constexpr int F2_RCLOW = 77, F2_BHIGH = 81, F2_RCHIGH = 85, F2_RHIGH = 89;  // 4 each
constexpr int F4_A = 93, F4_B = 114, F4_T0 = 135, F4_TS = 156, F4_TC = 177;  // 21 each
constexpr int F5_A = 198, F5_B = 202, F5_XC = 206, F5_XS = 210;  // 4 each
constexpr int N_TABLES = 214;

// Sub-indices (upstream model.h)
constexpr int HYDR_F1 = 0, STCK_F1 = 1;
constexpr int CRST_F2 = 0, CXST_F2 = 1, CRST_F2_33 = 2, CRST_F2_55 = 3;
constexpr int STCK_F4_THETA4 = 0, STCK_F4_THETA5 = 1, STCK_F4_THETA6 = 1;
constexpr int HYDR_F4_THETA1 = 2, HYDR_F4_THETA2 = 3, HYDR_F4_THETA3 = 3, HYDR_F4_THETA4 = 4;
constexpr int HYDR_F4_THETA7 = 5, HYDR_F4_THETA8 = 5;
constexpr int CRST_F4_THETA1 = 6, CRST_F4_THETA2 = 7, CRST_F4_THETA3 = 7, CRST_F4_THETA4 = 8;
constexpr int CRST_F4_THETA7 = 9, CRST_F4_THETA8 = 9;
constexpr int CXST_F4_THETA1 = 10, CXST_F4_THETA4 = 11, CXST_F4_THETA5 = 12, CXST_F4_THETA6 = 12;
constexpr int CRST_F4_THETA1_33 = 13, CRST_F4_THETA2_33 = 14, CRST_F4_THETA3_33 = 14;
constexpr int CRST_F4_THETA4_33 = 15, CRST_F4_THETA7_33 = 16, CRST_F4_THETA8_33 = 16;
constexpr int CRST_F4_THETA1_55 = 17, CRST_F4_THETA2_55 = 18, CRST_F4_THETA3_55 = 18;
constexpr int CRST_F4_THETA4_55 = 19, CRST_F4_THETA7_55 = 20, CRST_F4_THETA8_55 = 20;
constexpr int STCK_F5_PHI1 = 0, STCK_F5_PHI2 = 1, CXST_F5_PHI3 = 2, CXST_F5_PHI4 = 3;

// oxDNA base-type encoding (Utils::encode_base): A=0, G=1, C=2, T=3, dummy=4
inline char encode_base(int b) {
    static const char c[5] = {'A', 'G', 'C', 'T', 'D'};
    return (b >= 0 && b < 5) ? c[b] : '?';
}
} // namespace dna3sd

// ---------------------------------------------------------------------------
// Device-side parameter bundle, passed by value into the DNA3 functors.
// ---------------------------------------------------------------------------
using SDTableView = Kokkos::View<const c_number *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

struct DNA3Params {
    // All sequence-dependent tables, back to back (dna3sd:: offsets).
    SDTableView sd;

    // Interaction-site offsets per oxDNA type (A, G, C, T): CUDA MD_POS_*[1 + type].
    c_number pos_back1[4], pos_back2[4], pos_stack[4], pos_base[4];

    // Non-SD constants used by the kernels (model.h)
    c_number pos_back;       // POS_BACK (reference backbone for the stacking dihedrals)
    c_number gamma;          // GAMMA = POS_STACK - POS_BACK (upstream uses 0.74 for oxDNA3 too)
    c_number excl_eps;       // EXCL_EPS
    c_number fene_eps;       // FENE_EPS (or FENE_EPS from the SD file)

    // Coaxial-stacking angular terms (non-SD, oxDNA2 values; CUDA uses the
    // model.h constants directly): theta1 (+ pure harmonic), theta4, theta5(=6)
    c_number cx_t1_t0, cx_t1_ts, cx_t1_tc, cx_t1_a, cx_t1_b, cx_t1_sa, cx_t1_sb;
    c_number cx_t4_t0, cx_t4_ts, cx_t4_tc, cx_t4_a, cx_t4_b;
    c_number cx_t5_t0, cx_t5_ts, cx_t5_tc, cx_t5_a, cx_t5_b;

    // Debye-Huckel (as DNA2)
    c_number dh_prefactor = 0, dh_minus_kappa = 0, dh_RHIGH = 0, dh_RC = 0, dh_B = 0;
    bool     dh_half_ends = true;

    // Nonbonded COM-COM cutoff^2 used for the neighbour list (max over all terms)
    c_number cutsq_nb;

    // COM-COM range of the screened-pair kernels (hbond, xstk, coaxstk; the
    // LAMMPS fix OXDNA/NPAIR screen), squared, WITHOUT the skin margin: the
    // largest outer radial cutoff of each term plus the COM->site offsets of
    // the two nucleotide types it can act between (see make_oxdna3_params).
    c_number screen_cutsq = 0;
    // LAMMPS per-style COM cutoffs (init_one, lmp_cuts.h), from the tables
    // and interaction sites of this model (see make_oxdna3_params)
    LmpStyleCuts lmp;

    // LAMMPS-only coaxial-stacking variant (lammps_coaxstk_terminal input
    // toggle, off by default = upstream physics), as for oxDNA2: the coaxstk
    // kernel only acts between two strand-terminal nucleotides
    // (cxst_terminal_only, checked in the kernel) and theta4 gets a second,
    // mirrored lobe f4(theta4; pi - theta4_0) (cxst_t4_blunt, in
    // dna3::particle_particle_interaction).
    bool cxst_terminal_only = false;
    bool cxst_t4_blunt      = false;

    KOKKOS_INLINE_FUNCTION
    c_number operator()(int table, int i, int j, int k, int l) const {
        return sd(table * dna3sd::TSIZE + dna3sd::tidx(i, j, k, l));
    }
};

// ---------------------------------------------------------------------------
// Host-side construction (double precision), mirroring DNA3Interaction.
// ---------------------------------------------------------------------------
struct DNA3Options {
    double T    = 0.1;
    double salt = 0.5;
    bool   average = false;            // use_average_seq (see README: bench default false)
    std::string seq_file;              // seq_dep_file
    bool   dh_half_charged_ends = true;
    double dh_lambda = 0.3616455;      // dh_lambda
    double dh_strength = 0.0543;       // dh_strength
    double dh_rhigh = -1;              // debye_huckel_rhigh (< 0: 3 * lambda)
    // Stacking cos(phi1/2) derivatives: upstream (CPU and CUDA) use
    // GAMMA = POS_STACK - POS_BACK = 0.74 (the oxDNA1/2 stacking-site offset),
    // but the oxDNA3 stacking site sits at 0.37, so the exact gradient of the
    // oxDNA3 energy needs gamma = 0.37 + 0.4 = 0.77. false = upstream-faithful.
    bool   consistent_gamma = false;   // dna3_consistent_gamma (bench-only key)
};

namespace dna3sd {

struct HostTables {
    std::vector<double> t;
    HostTables() : t(static_cast<size_t>(N_TABLES) * TSIZE, 0.0) {}
    double &operator()(int table, int i, int j, int k, int l) {
        return t[static_cast<size_t>(table) * TSIZE + tidx(i, j, k, l)];
    }
    void fill(int table, double v) {
        for (int n = 0; n < TSIZE; n++) t[static_cast<size_t>(table) * TSIZE + n] = v;
    }
    // MultiDimArray::get_average_par
    double avg(int table, int i, int j, int k, int l) {
        double a = 0, den = 4;
        if (i == 5 && l == 5) den = 16;
        if (i == 5 && l != 5) {
            for (int m = 0; m < 4; m++) a += (*this)(table, m, j, k, l);
        } else if (l == 5 && i != 5) {
            for (int m = 0; m < 4; m++) a += (*this)(table, i, j, k, m);
        } else if (l == 5 && i == 5) {
            for (int m = 0; m < 4; m++)
                for (int n = 0; n < 4; n++) a += (*this)(table, m, j, k, n);
        } else {
            throw std::runtime_error("oxDNA3: get_average_par called with wrong indices");
        }
        return a / den;
    }
};

// Minimal "key = value" reader for the sequence-dependence file (oxDNA
// input_file semantics: '#' comments, whitespace-trimmed key and value).
inline std::map<std::string, std::string> read_seq_file(const std::string &fname) {
    std::ifstream f(fname);
    if (!f) throw std::runtime_error("oxDNA3: cannot open sequence dependence file '" + fname + "'");
    std::map<std::string, std::string> kv;
    auto trim = [](const std::string &s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return std::string();
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    };
    std::string line;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
        if (!k.empty()) kv[k] = v;
    }
    return kv;
}

} // namespace dna3sd

// Build the oxDNA3 parameter set and upload the tables to the device.
inline DNA3Params make_oxdna3_params(const DNA3Options &opt) {
    using namespace dna3sd;
    // float(PI) arithmetic exactly as model.h macros in double code: PI is a
    // double macro, the other literals are floats.
    constexpr double PI = 3.141592653589793238462643383279502884197;
    const double T = opt.T;
    HostTables H;

    // ---------------- constructor: fill with average (model.h) values -------
    H.fill(FENE_R0, 0.7564f);        // FENE_R0_OXDNA2
    H.fill(FENE_DELTA, 0.25f);       // FENE_DELTA
    H.fill(FENE_DELTA2, 0.0625f);    // FENE_DELTA2
    const double excl_s[7]  = {0.70f, 0.33f, 0.515f, 0.515f, 0.33f, 0.515f, 0.515f};
    const double excl_r[7]  = {0.675f, 0.32f, 0.50f, 0.50f, 0.32f, 0.50f, 0.50f};
    const double excl_b[7]  = {892.016223343f, 4119.70450017f, 1707.30627298f, 1707.30627298f,
                               4119.70450017f, 1707.30627298f, 1707.30627298f};
    const double excl_rc[7] = {0.711879214356f, 0.335388426126f, 0.52329943261f, 0.52329943261f,
                               0.335388426126f, 0.52329943261f, 0.52329943261f};
    for (int m = 0; m < 7; m++) {
        H.fill(EXCL_S + m, excl_s[m]);  H.fill(EXCL_R + m, excl_r[m]);
        H.fill(EXCL_B + m, excl_b[m]);  H.fill(EXCL_RC + m, excl_rc[m]);
    }
    // F1: HYDR (oxDNA1 well depth!) and STCK. Upstream fills the stacking eps
    // in the constructor with the (not yet initialised) member _T; we use T.
    H.fill(F1_EPS + HYDR_F1, 1.077f);
    H.fill(F1_EPS + STCK_F1, 1.3448f + 2.6568f * T);
    H.fill(F1_A + HYDR_F1, 8.f);            H.fill(F1_A + STCK_F1, 6.f);
    H.fill(F1_RC + HYDR_F1, 0.75f);         H.fill(F1_RC + STCK_F1, 0.9f);
    H.fill(F1_R0 + HYDR_F1, 0.4f);          H.fill(F1_R0 + STCK_F1, 0.4f);
    H.fill(F1_BLOW + HYDR_F1, -126.243f);   H.fill(F1_BLOW + STCK_F1, -68.1857f);
    H.fill(F1_BHIGH + HYDR_F1, -7.87708f);  H.fill(F1_BHIGH + STCK_F1, -3.12992f);
    H.fill(F1_RLOW + HYDR_F1, 0.34f);       H.fill(F1_RLOW + STCK_F1, 0.32f);
    H.fill(F1_RHIGH + HYDR_F1, 0.7f);       H.fill(F1_RHIGH + STCK_F1, 0.75f);
    H.fill(F1_RCLOW + HYDR_F1, 0.276908f);  H.fill(F1_RCLOW + STCK_F1, 0.23239f);
    H.fill(F1_RCHIGH + HYDR_F1, 0.783775f); H.fill(F1_RCHIGH + STCK_F1, 0.956f);

    // F4 (21 angular terms): A, B, T0, TS, TC in upstream index order
    struct F4Def { double a, b, t0, ts, tc; };
    const F4Def f4d[21] = {
        {1.3f, 6.4381f, 0.f, 0.8f, 0.961538f},                  //  0 STCK_THETA4
        {0.9f, 3.89361f, 0.f, 0.95f, 1.16959f},                 //  1 STCK_THETA5
        {1.5f, 4.16038f, 0.f, 0.7f, 0.952381f},                 //  2 HYDR_THETA1
        {1.5f, 4.16038f, 0.f, 0.7f, 0.952381f},                 //  3 HYDR_THETA2
        {0.46f, 0.133855f, PI, 0.7f, 3.10559f},                 //  4 HYDR_THETA4
        {4.f, 17.0526f, PI * 0.5f, 0.45f, 0.555556f},           //  5 HYDR_THETA7
        {2.25f, 7.00545f, PI - 2.35f, 0.58f, 0.766284f},        //  6 CRST_THETA1
        {1.70f, 6.2469f, 1.f, 0.68f, 0.865052f},                //  7 CRST_THETA2
        {1.50f, 2.59556f, 0.f, 0.65f, 1.02564f},                //  8 CRST_THETA4
        {1.70f, 6.2469f, 0.875f, 0.68f, 0.865052f},             //  9 CRST_THETA7
        {2.f, 10.9032f, PI - 0.25f, 0.65f, 0.769231f},          // 10 CXST_THETA1 (T0_OXDNA2)
        {1.3f, 6.4381f, 0.f, 0.8f, 0.961538f},                  // 11 CXST_THETA4
        {0.9f, 3.89361f, 0.f, 0.95f, 1.16959f},                 // 12 CXST_THETA5
        {2.25f, 7.00545f, PI - 2.35f, 0.58f, 0.766284f},        // 13 CRST_THETA1_33
        {1.70f, 6.2469f, 1.f, 0.68f, 0.865052f},                // 14 CRST_THETA2_33
        {1.50f, 2.59556f, PI, 0.65f, 1.02564f},                 // 15 CRST_THETA4_33
        {1.70f, 6.2469f, 0.875f, 0.68f, 0.865052f},             // 16 CRST_THETA7_33
        {2.25f, 7.00545f, PI - 2.35f, 0.58f, 0.766284f},        // 17 CRST_THETA1_55
        {1.70f, 6.2469f, 1.f, 0.68f, 0.865052f},                // 18 CRST_THETA2_55
        {1.50f, 2.59556f, PI, 0.65f, 1.02564f},                 // 19 CRST_THETA4_55
        {1.70f, 6.2469f, PI - 0.875f, 0.68f, 0.865052f},        // 20 CRST_THETA7_55
    };
    for (int m = 0; m < 21; m++) {
        H.fill(F4_A + m, f4d[m].a);   H.fill(F4_B + m, f4d[m].b);
        H.fill(F4_T0 + m, f4d[m].t0); H.fill(F4_TS + m, f4d[m].ts);
        H.fill(F4_TC + m, f4d[m].tc);
    }

    // F2: CRST, CXST (oxDNA2 K, oxDNA3 R0), CRST_33, CRST_55
    const double f2k[4] = {47.5f, 58.5f, 47.5f, 47.5f};
    const double f2rc[4] = {0.675f, 0.6f, 0.675f, 0.675f};
    const double f2r0[4] = {0.575f, 0.370011f, 0.575f, 0.575f};
    const double f2blow[4] = {-0.888889f, -2.13158f, -0.888889f, -0.888889f};
    const double f2bhigh[4] = {-0.888889f, -2.13158f, -0.888889f, -0.888889f};
    const double f2rlow[4] = {0.495f, 0.22f, 0.495f, 0.495f};
    const double f2rhigh[4] = {0.655f, 0.58f, 0.655f, 0.655f};
    const double f2rclow[4] = {0.45f, 0.177778f, 0.45f, 0.45f};
    const double f2rchigh[4] = {0.7f, 0.6222222f, 0.7f, 0.7f};
    for (int m = 0; m < 4; m++) {
        H.fill(F2_K + m, f2k[m]);        H.fill(F2_K_SYMM + m, f2k[m]);
        H.fill(F2_RC + m, f2rc[m]);      H.fill(F2_R0 + m, f2r0[m]);
        H.fill(F2_BLOW + m, f2blow[m]);  H.fill(F2_BHIGH + m, f2bhigh[m]);
        H.fill(F2_RLOW + m, f2rlow[m]);  H.fill(F2_RHIGH + m, f2rhigh[m]);
        H.fill(F2_RCLOW + m, f2rclow[m]); H.fill(F2_RCHIGH + m, f2rchigh[m]);
    }

    // F5: STCK_PHI1, STCK_PHI2, CXST_PHI3, CXST_PHI4 (B[3] = CXST_PHI3_B upstream)
    for (int m = 0; m < 4; m++) {
        H.fill(F5_A + m, 2.0f);  H.fill(F5_B + m, 10.9032f);
        H.fill(F5_XC + m, -0.769231f);  H.fill(F5_XS + m, -0.65f);
    }

    double fene_eps = 2.0f;   // FENE_EPS

    // ---------------- init(): sequence-dependent values ---------------------
    if (!opt.average) {
        auto kv = read_seq_file(opt.seq_file);
        auto getf = [&](const std::string &key, double &dst) {
            auto it = kv.find(key);
            if (it == kv.end()) return false;
            dst = static_cast<double>(static_cast<float>(std::atof(it->second.c_str())));
            return true;
        };
        double stck_fact_eps;
        if (!getf("STCK_FACT_EPS", stck_fact_eps))
            throw std::runtime_error("oxDNA3: mandatory key STCK_FACT_EPS not found in '" + opt.seq_file + "'");
        {
            double v;
            if (getf("FENE_EPS", v)) fene_eps = v;
        }
        auto key2 = [](const char *p, int a, int b) {
            return std::string(p) + "_" + encode_base(a) + "_" + encode_base(b);
        };
        auto key4 = [](const char *p, int a, int b, int c, int d) {
            return std::string(p) + "_" + encode_base(a) + "_" + encode_base(b) + "_" +
                   encode_base(c) + "_" + encode_base(d);
        };
        double v;
        for (int i = 0; i < DIM_A - 2; i++)
        for (int j = 0; j < DIM_B - 1; j++)
        for (int k = 0; k < DIM_B - 1; k++)
        for (int l = 0; l < DIM_A - 2; l++) {
            // excluded volume (sigma and r* only; B/RC are enslaved)
            const char *es[7] = {"EXCL_S1", "EXCL_S2", "EXCL_S3", "EXCL_S4", "EXCL_S5", "EXCL_S6", "EXCL_S7"};
            const char *er[7] = {"EXCL_R1", "EXCL_R2", "EXCL_R3", "EXCL_R4", "EXCL_R5", "EXCL_R6", "EXCL_R7"};
            for (int m = 0; m < 7; m++) {
                if (getf(key4(es[m], i, j, k, l), v)) H(EXCL_S + m, i, j, k, l) = v;
            }
            for (int m = 0; m < 7; m++) {
                if (getf(key4(er[m], i, j, k, l), v)) H(EXCL_R + m, i, j, k, l) = v;
            }
            // FENE
            if (getf(key4("FENE_R0", i, j, k, l), v))    H(FENE_R0, i, j, k, l) = v;
            if (getf(key4("FENE_DELTA", i, j, k, l), v)) H(FENE_DELTA, i, j, k, l) = v;

            // F1 HYDR: pair-type dependent, stored at (k, i, j, l)
            if (getf(key2("HYDR", i, j), v))       H(F1_EPS + HYDR_F1, k, i, j, l) = v;
            if (getf(key2("HYDR_A", i, j), v))     H(F1_A + HYDR_F1, k, i, j, l) = v;
            if (getf(key2("HYDR_RC", i, j), v))    H(F1_RC + HYDR_F1, k, i, j, l) = v;
            if (getf(key2("HYDR_R0", i, j), v))    H(F1_R0 + HYDR_F1, k, i, j, l) = v;
            if (getf(key2("HYDR_RLOW", i, j), v))  H(F1_RLOW + HYDR_F1, k, i, j, l) = v;
            if (getf(key2("HYDR_RHIGH", i, j), v)) H(F1_RHIGH + HYDR_F1, k, i, j, l) = v;

            // F1 STCK: eps depends on the central dinucleotide (j, k) only
            if (getf(key2("STCK", j, k), v))
                H(F1_EPS + STCK_F1, i, j, k, l) = v * (1.0 - stck_fact_eps + (T * 9.0 * stck_fact_eps));
            if (getf(key4("STCK_A", i, j, k, l), v))     H(F1_A + STCK_F1, i, j, k, l) = v;
            if (getf(key4("STCK_RC", i, j, k, l), v))    H(F1_RC + STCK_F1, i, j, k, l) = v;
            if (getf(key4("STCK_R0", i, j, k, l), v))    H(F1_R0 + STCK_F1, i, j, k, l) = v;
            if (getf(key4("STCK_RLOW", i, j, k, l), v))  H(F1_RLOW + STCK_F1, i, j, k, l) = v;
            if (getf(key4("STCK_RHIGH", i, j, k, l), v)) H(F1_RHIGH + STCK_F1, i, j, k, l) = v;

            // F4 HYDR (pair dependent, stored at (k, i, j, l)); THETA3 aliases
            // THETA2 and THETA8 aliases THETA7 (same table index upstream).
            struct { const char *n; int idx; } hyd[6] = {
                {"HYDR_THETA1", HYDR_F4_THETA1}, {"HYDR_THETA2", HYDR_F4_THETA2},
                {"HYDR_THETA3", HYDR_F4_THETA3}, {"HYDR_THETA4", HYDR_F4_THETA4},
                {"HYDR_THETA7", HYDR_F4_THETA7}, {"HYDR_THETA8", HYDR_F4_THETA8}};
            for (auto &h : hyd) {
                std::string b = h.n;
                if (getf(key2((b + "_A").c_str(), i, j), v))  H(F4_A + h.idx, k, i, j, l) = v;
                if (getf(key2((b + "_T0").c_str(), i, j), v)) H(F4_T0 + h.idx, k, i, j, l) = v;
                if (getf(key2((b + "_TS").c_str(), i, j), v)) H(F4_TS + h.idx, k, i, j, l) = v;
            }
            // F4 STCK (tetramer dependent)
            struct { const char *n; int idx; } stk[2] = {
                {"STCK_THETA4", STCK_F4_THETA4}, {"STCK_THETA5", STCK_F4_THETA5}};
            for (auto &h : stk) {
                std::string b = h.n;
                if (getf(key4((b + "_A").c_str(), i, j, k, l), v))  H(F4_A + h.idx, i, j, k, l) = v;
                if (getf(key4((b + "_T0").c_str(), i, j, k, l), v)) H(F4_T0 + h.idx, i, j, k, l) = v;
                if (getf(key4((b + "_TS").c_str(), i, j, k, l), v)) H(F4_TS + h.idx, i, j, k, l) = v;
            }
            // F4 CRST (tetramer dependent), in upstream read order
            struct { const char *n; int idx; } crs[8] = {
                {"CRST_THETA4_%s_33", CRST_F4_THETA4_33}, {"CRST_THETA4_%s_55", CRST_F4_THETA4_55},
                {"CRST_THETA1_%s_33", CRST_F4_THETA1_33}, {"CRST_THETA2_%s_33", CRST_F4_THETA2_33},
                {"CRST_THETA7_%s_33", CRST_F4_THETA7_33}, {"CRST_THETA1_%s_55", CRST_F4_THETA1_55},
                {"CRST_THETA2_%s_55", CRST_F4_THETA2_55}, {"CRST_THETA7_%s_55", CRST_F4_THETA7_55}};
            for (auto &h : crs) {
                std::string tmpl = h.n;
                auto p = tmpl.find("%s");
                auto mk = [&](const char *what) {
                    std::string b = tmpl.substr(0, p) + what + tmpl.substr(p + 2);
                    return key4(b.c_str(), i, j, k, l);
                };
                if (getf(mk("A"), v))  H(F4_A + h.idx, i, j, k, l) = v;
                if (getf(mk("T0"), v)) H(F4_T0 + h.idx, i, j, k, l) = v;
                if (getf(mk("TS"), v)) H(F4_TS + h.idx, i, j, k, l) = v;
            }
            // F2 CRST: K pair dependent at (k, i, j, l); R0/RC/RLOW/RHIGH tetramer
            if (getf(key2("CRST_K_33", i, j), v)) H(F2_K + CRST_F2_33, k, i, j, l) = v;
            if (getf(key2("CRST_K_55", i, j), v)) H(F2_K + CRST_F2_55, k, i, j, l) = v;
            if (getf(key4("CRST_R0_33", i, j, k, l), v))    H(F2_R0 + CRST_F2_33, i, j, k, l) = v;
            if (getf(key4("CRST_RC_33", i, j, k, l), v))    H(F2_RC + CRST_F2_33, i, j, k, l) = v;
            if (getf(key4("CRST_RLOW_33", i, j, k, l), v))  H(F2_RLOW + CRST_F2_33, i, j, k, l) = v;
            if (getf(key4("CRST_RHIGH_33", i, j, k, l), v)) H(F2_RHIGH + CRST_F2_33, i, j, k, l) = v;
            if (getf(key4("CRST_R0_55", i, j, k, l), v))    H(F2_R0 + CRST_F2_55, i, j, k, l) = v;
            if (getf(key4("CRST_RC_55", i, j, k, l), v))    H(F2_RC + CRST_F2_55, i, j, k, l) = v;
            if (getf(key4("CRST_RLOW_55", i, j, k, l), v))  H(F2_RLOW + CRST_F2_55, i, j, k, l) = v;
            if (getf(key4("CRST_RHIGH_55", i, j, k, l), v)) H(F2_RHIGH + CRST_F2_55, i, j, k, l) = v;
            // F2 CXST: K depends on the central dinucleotide (j, k)
            if (getf(key2("CXST_K", j, k), v)) H(F2_K + CXST_F2, i, j, k, l) = v;
        }
    }

    // ---------------- strand-end (NO_TYPE) junction parameters --------------
    const int E = DIM_A - 1;   // NO_TYPE index
    for (int i = 0; i < DIM_A; i++)
    for (int j = 0; j < DIM_B - 1; j++)
    for (int k = 0; k < DIM_B - 1; k++) {
        // 2d (pair-type) parameters: copy the 0jk0 value
        for (int m : {CRST_F2_33, CRST_F2_55, CXST_F2}) {
            H(F2_K + m, i, j, k, E) = H(F2_K + m, 0, j, k, 0);
            H(F2_K + m, E, j, k, i) = H(F2_K + m, 0, j, k, 0);
        }
        auto avg_ends = [&](int table) {
            H(table, i, j, k, E) = H.avg(table, i, j, k, E);
            H(table, E, j, k, i) = H.avg(table, E, j, k, i);
        };
        avg_ends(FENE_DELTA);
        avg_ends(FENE_R0);
        for (int m = 0; m < 2; m++) {
            // upstream order: all (i,j,k,E) first, then all (E,j,k,i)
            for (int tb : {F1_EPS, F1_A, F1_R0, F1_RC, F1_RLOW, F1_RHIGH})
                H(tb + m, i, j, k, E) = H.avg(tb + m, i, j, k, E);
            for (int tb : {F1_EPS, F1_A, F1_R0, F1_RC, F1_RLOW, F1_RHIGH})
                H(tb + m, E, j, k, i) = H.avg(tb + m, E, j, k, i);
        }
        for (int m = 0; m < 4; m++) {
            for (int tb : {F2_R0, F2_RC, F2_RLOW, F2_RHIGH})
                H(tb + m, i, j, k, E) = H.avg(tb + m, i, j, k, E);
            for (int tb : {F2_R0, F2_RC, F2_RLOW, F2_RHIGH})
                H(tb + m, E, j, k, i) = H.avg(tb + m, E, j, k, i);
        }
        for (int m = 0; m < 7; m++) {
            for (int tb : {EXCL_S, EXCL_R})
                H(tb + m, i, j, k, E) = H.avg(tb + m, i, j, k, E);
            for (int tb : {EXCL_S, EXCL_R})
                H(tb + m, E, j, k, i) = H.avg(tb + m, E, j, k, i);
        }
        for (int m = 0; m < 21; m++) {
            for (int tb : {F4_A, F4_T0, F4_TS})
                H(tb + m, i, j, k, E) = H.avg(tb + m, i, j, k, E);
            for (int tb : {F4_A, F4_T0, F4_TS})
                H(tb + m, E, j, k, i) = H.avg(tb + m, E, j, k, i);
        }
        for (int m = 0; m < 4; m++) {
            for (int tb : {F5_A, F5_XS})
                H(tb + m, i, j, k, E) = H.avg(tb + m, i, j, k, E);
            for (int tb : {F5_A, F5_XS})
                H(tb + m, E, j, k, i) = H.avg(tb + m, E, j, k, i);
        }
    }

    // ---------------- symmetric coaxial/cross K ------------------------------
    for (int i = 0; i < DIM_A; i++)
    for (int j = 0; j < DIM_B - 1; j++)
    for (int k = 0; k < DIM_B - 1; k++)
    for (int l = 0; l < DIM_A; l++)
        for (int m = 0; m < 4; m++)
            H(F2_K_SYMM + m, i, j, k, l) = 0.5 * (H(F2_K + m, i, j, k, l) + H(F2_K + m, l, k, j, i));

    // ---------------- enslaved parameters (continuity / differentiability) --
    auto SQR = [](double x) { return x * x; };
    for (int i = 0; i < DIM_A; i++)
    for (int j = 0; j < DIM_B - 1; j++)
    for (int k = 0; k < DIM_B - 1; k++)
    for (int l = 0; l < DIM_A; l++) {
        H(FENE_DELTA2, i, j, k, l) = SQR(H(FENE_DELTA, i, j, k, l));
        // f1
        H(F1_RLOW + 0, i, j, k, l)  = H(F1_R0 + 0, i, j, k, l) - 0.06;
        H(F1_RHIGH + 0, i, j, k, l) = H(F1_R0 + 0, i, j, k, l) + 0.3;
        H(F1_RC + 0, i, j, k, l)    = H(F1_R0 + 0, i, j, k, l) + 0.35;
        H(F1_RLOW + 1, i, j, k, l)  = H(F1_R0 + 1, i, j, k, l) - 0.08;
        H(F1_RHIGH + 1, i, j, k, l) = H(F1_R0 + 1, i, j, k, l) + 0.35;
        H(F1_RC + 1, i, j, k, l)    = H(F1_R0 + 1, i, j, k, l) + 0.5;
        for (int m = 0; m < 2; m++) {
            const double A = H(F1_A + m, i, j, k, l), R0 = H(F1_R0 + m, i, j, k, l);
            const double RLOW = H(F1_RLOW + m, i, j, k, l), RHIGH = H(F1_RHIGH + m, i, j, k, l);
            const double RC = H(F1_RC + m, i, j, k, l);
            const double term1 = std::exp(-A * (RLOW - R0));
            const double term2 = std::exp(-A * (RC - R0));
            const double term3 = std::exp(-A * (RHIGH - R0));
            const double BLOW  = SQR(A * term1 * (1 - term1)) / (SQR(1 - term1) - SQR(1 - term2));
            const double BHIGH = SQR(A * term3 * (1 - term3)) / (SQR(1 - term3) - SQR(1 - term2));
            H(F1_BLOW + m, i, j, k, l)   = BLOW;
            H(F1_BHIGH + m, i, j, k, l)  = BHIGH;
            H(F1_RCLOW + m, i, j, k, l)  = RLOW - A / BLOW * (term1 * (1 - term1));
            H(F1_RCHIGH + m, i, j, k, l) = RHIGH - A / BHIGH * (term3 * (1 - term3));
            H(F1_SHIFT + m, i, j, k, l)  = H(F1_EPS + m, i, j, k, l) * SQR(1 - std::exp(-(RC - R0) * A));
        }
        // f2
        for (int m = 0; m < 4; m++) {
            const double R0 = H(F2_R0 + m, i, j, k, l);
            if (m == 1) {   // coaxial
                H(F2_RLOW + m, i, j, k, l)  = R0 - 0.18;
                H(F2_RHIGH + m, i, j, k, l) = R0 + 0.18;
                H(F2_RC + m, i, j, k, l)    = R0 + 0.2;
            } else {        // cross stacking
                H(F2_RLOW + m, i, j, k, l)  = R0 - 0.08;
                H(F2_RHIGH + m, i, j, k, l) = R0 + 0.08;
                H(F2_RC + m, i, j, k, l)    = R0 + 0.1;
            }
            const double term1 = H(F2_RLOW + m, i, j, k, l) - R0;
            const double term2 = H(F2_RHIGH + m, i, j, k, l) - R0;
            const double term3 = H(F2_RC + m, i, j, k, l) - R0;
            H(F2_RCLOW + m, i, j, k, l)  = H(F2_RLOW + m, i, j, k, l) - term1 + SQR(term3) / term1;
            H(F2_RCHIGH + m, i, j, k, l) = H(F2_RHIGH + m, i, j, k, l) - term2 + SQR(term3) / term2;
            H(F2_BLOW + m, i, j, k, l)  = -0.5 * term1 / (H(F2_RCLOW + m, i, j, k, l) - H(F2_RLOW + m, i, j, k, l));
            H(F2_BHIGH + m, i, j, k, l) = -0.5 * term2 / (H(F2_RCHIGH + m, i, j, k, l) - H(F2_RHIGH + m, i, j, k, l));
        }
        // f3
        for (int m = 0; m < 7; m++) {
            const double r = H(EXCL_R + m, i, j, k, l);
            const double tmp = SQR(H(EXCL_S + m, i, j, k, l) / r);
            const double term1 = tmp * tmp * tmp;
            const double term2 = 4. * (SQR(term1) - term1);
            const double term3 = 12. / r * (2. * SQR(term1) - term1);
            H(EXCL_RC + m, i, j, k, l) = term2 / term3 + r;
            H(EXCL_B + m, i, j, k, l)  = SQR(term3) / term2;
        }
        // f4 (note: TS read from the file is overwritten here, as upstream)
        for (int m = 0; m < 21; m++) {
            const double A = H(F4_A + m, i, j, k, l);
            const double TS = std::sqrt(0.81225 / A);
            const double TC = 1. / A / TS;
            H(F4_TS + m, i, j, k, l) = TS;
            H(F4_TC + m, i, j, k, l) = TC;
            H(F4_B + m, i, j, k, l)  = A * TS / (TC - TS);
        }
        // f5
        for (int m = 0; m < 4; m++) {
            const double A = H(F5_A + m, i, j, k, l), XS = H(F5_XS + m, i, j, k, l);
            const double term1 = 1. - A * SQR(XS);
            const double term2 = A * XS;
            H(F5_XC + m, i, j, k, l) = term1 / term2 + XS;
            H(F5_B + m, i, j, k, l)  = SQR(term2) / term1;
        }
    }

    // ---------------- assemble device parameter bundle -----------------------
    DNA3Params p;
    Kokkos::View<c_number *> dsd("dna3_sd_tables", static_cast<size_t>(N_TABLES) * TSIZE);
    auto hsd = Kokkos::create_mirror_view(dsd);
    for (size_t n = 0; n < H.t.size(); n++) hsd(n) = static_cast<c_number>(H.t[n]);
    Kokkos::deep_copy(dsd, hsd);
    p.sd = dsd;

    // Interaction sites, oxDNA type order A, G, C, T (model.h POS_*_X)
    const double back1[4] = {-0.3400f, -0.3400f, -0.3400f, -0.3400f};
    const double back2[4] = {0.3408f, 0.3408f, 0.3408f, 0.3408f};
    const double stack[4] = {0.37f, 0.37f, 0.37f, 0.37f};
    const double base[4]  = {0.43f, 0.43f, 0.37f, 0.37f};
    for (int t = 0; t < 4; t++) {
        p.pos_back1[t] = static_cast<c_number>(back1[t]);
        p.pos_back2[t] = static_cast<c_number>(back2[t]);
        p.pos_stack[t] = static_cast<c_number>(stack[t]);
        p.pos_base[t]  = static_cast<c_number>(base[t]);
    }
    p.pos_back = static_cast<c_number>(-0.4f);                 // POS_BACK
    // GAMMA: upstream value, or the one consistent with the oxDNA3 stacking site
    p.gamma    = opt.consistent_gamma ? static_cast<c_number>(0.37f - (-0.4f))
                                      : static_cast<c_number>(0.74f);
    p.excl_eps = static_cast<c_number>(2.0f);                  // EXCL_EPS
    p.fene_eps = static_cast<c_number>(fene_eps);

    // coaxial angular terms (model.h, oxDNA2 theta1)
    p.cx_t1_t0 = static_cast<c_number>(PI - 0.25f);
    p.cx_t1_ts = static_cast<c_number>(0.65f);
    p.cx_t1_tc = static_cast<c_number>(0.769231f);
    p.cx_t1_a  = static_cast<c_number>(2.f);
    p.cx_t1_b  = static_cast<c_number>(10.9032f);
    p.cx_t1_sa = static_cast<c_number>(20.f);
    p.cx_t1_sb = static_cast<c_number>(PI - 0.1f * (PI - (PI - 0.25f)));
    p.cx_t4_t0 = 0; p.cx_t4_ts = static_cast<c_number>(0.8f);  p.cx_t4_tc = static_cast<c_number>(0.961538f);
    p.cx_t4_a  = static_cast<c_number>(1.3f); p.cx_t4_b = static_cast<c_number>(6.4381f);
    p.cx_t5_t0 = 0; p.cx_t5_ts = static_cast<c_number>(0.95f); p.cx_t5_tc = static_cast<c_number>(1.16959f);
    p.cx_t5_a  = static_cast<c_number>(0.9f); p.cx_t5_b = static_cast<c_number>(3.89361f);

    // Debye-Huckel (DNA2Interaction::get_settings + DNA3Interaction::init)
    const double lambda = opt.dh_lambda * std::sqrt(T / 0.1) / std::sqrt(opt.salt);
    const double rhigh  = (opt.dh_rhigh >= 0) ? opt.dh_rhigh
                        : 3.0 * opt.dh_lambda * std::sqrt(T / 0.1f) / std::sqrt(opt.salt);
    const double q = opt.dh_strength, x = rhigh, la = lambda;
    const double B  = -(std::exp(-x / la) * q * q * (x + la) * (x + la)) / (-4. * x * x * x * la * la * q);
    const double RC = x * (q * x + 3. * q * la) / (q * (x + la));
    p.dh_prefactor   = static_cast<c_number>(q);
    p.dh_minus_kappa = static_cast<c_number>(-1.0 / lambda);
    p.dh_RHIGH       = static_cast<c_number>(rhigh);
    p.dh_RC          = static_cast<c_number>(RC);
    p.dh_B           = static_cast<c_number>(B);
    p.dh_half_ends   = opt.dh_half_charged_ends;

    // Neighbour-list cutoff: largest site-site range over the tables actually
    // used by the nonbonded kernel (true bases 0..3 and NO_TYPE flanks), plus
    // the two site offsets from the COM.
    const double back_off = std::sqrt(SQR(0.3400f) + SQR(0.3408f));
    double rmax_back = 0, rmax_base = 0, rmax_stack = 0;
    for (int i = 0; i < DIM_A; i++)
    for (int j = 0; j < 4; j++)
    for (int k = 0; k < 4; k++)
    for (int l = 0; l < DIM_A; l++) {
        if (i == 4 || l == 4) continue;   // dummy bases are not supported
        for (int m = 0; m < 4; m++) rmax_back = std::max(rmax_back, H(EXCL_RC + m, i, j, k, l));
        rmax_base = std::max(rmax_base, H(F1_RCHIGH + HYDR_F1, i, j, k, l));
        rmax_base = std::max(rmax_base, H(F2_RCHIGH + CRST_F2_33, i, j, k, l));
        rmax_base = std::max(rmax_base, H(F2_RCHIGH + CRST_F2_55, i, j, k, l));
        rmax_stack = std::max(rmax_stack, H(F2_RCHIGH + CXST_F2, i, j, k, l));
    }
    double max_cut = 2 * back_off + rmax_back;
    max_cut = std::max(max_cut, 2 * 0.43 + rmax_base);
    max_cut = std::max(max_cut, 2 * 0.43 + rmax_back);
    max_cut = std::max(max_cut, 2 * 0.37 + rmax_stack);
    max_cut = std::max(max_cut, 2 * back_off + RC);
    p.cutsq_nb = static_cast<c_number>(max_cut * max_cut);

    // Screened-pair range (fix OXDNA/NPAIR), per term and per pair of oxDNA
    // types (p, q): the outer radial cutoff over all flanks that can occur
    // (true bases 0..3 and NO_TYPE) plus the two base (hbond, xstk) or
    // stacking (coaxstk) site offsets of p and q. hbond only acts between
    // complementary bases (type sum 3), whose tables use flanks (0, 0).
    // Purine-purine cross stacking reaches rc + 2 * 0.43.
    double screen = 0;
    for (int tp = 0; tp < 4; tp++)
    for (int tq = 0; tq < 4; tq++) {
        const double obase  = static_cast<double>(p.pos_base[tp])  + static_cast<double>(p.pos_base[tq]);
        const double ostack = static_cast<double>(p.pos_stack[tp]) + static_cast<double>(p.pos_stack[tq]);
        if (tp + tq == 3)
            screen = std::max(screen, H(F1_RCHIGH + HYDR_F1, 0, tq, tp, 0) + obase);
        for (int i = 0; i < DIM_A; i++)
        for (int l = 0; l < DIM_A; l++) {
            if (i == 4 || l == 4) continue;   // dummy bases are not supported
            screen = std::max(screen, H(F2_RCHIGH + CRST_F2_33, i, tq, tp, l) + obase);
            screen = std::max(screen, H(F2_RCHIGH + CRST_F2_55, i, tq, tp, l) + obase);
            screen = std::max(screen, H(F2_RCHIGH + CXST_F2, i, tq, tp, l) + ostack);
        }
    }
    p.screen_cutsq = static_cast<c_number>(screen * screen);
    // LAMMPS init_one() cutoffs (lmp_cuts.h), max over the type pairs and the
    // flanks that occur (true bases 0..3 and NO_TYPE), with the interaction
    // sites of this model: grooved backbone, base 0.43 (A, G) / 0.37 (C, T),
    // stacking 0.37
    {
        LmpStyleCuts &c = p.lmp;
        const double bk = back_off;
        for (int tp = 0; tp < 4; tp++)
        for (int tq = 0; tq < 4; tq++) {
            const double bp = base[tp], bq = base[tq];
            c.hbond = std::max(c.hbond, H(F1_RCHIGH + HYDR_F1, 0, tq, tp, 0) + bp + bq);
            for (int i = 0; i < DIM_A; i++)
            for (int l = 0; l < DIM_A; l++) {
                if (i == 4 || l == 4) continue;   // dummy bases are not supported
                c.excv = std::max({c.excv, H(EXCL_RC + 0, i, tq, tp, l) + 2.0 * bk,
                                   H(EXCL_RC + 2, i, tq, tp, l) + bk + std::max(bp, bq),
                                   H(EXCL_RC + 3, i, tq, tp, l) + bk + std::max(bp, bq),
                                   H(EXCL_RC + 1, i, tq, tp, l) + bp + bq,
                                   H(EXCL_RC + 4, i, tq, tp, l) + bp + bq});
                c.xstk = std::max({c.xstk, H(F2_RCHIGH + CRST_F2_33, i, tq, tp, l) + bp + bq,
                                   H(F2_RCHIGH + CRST_F2_55, i, tq, tp, l) + bp + bq});
                c.coaxstk = std::max(c.coaxstk, H(F2_RCHIGH + CXST_F2, i, tq, tp, l) + stack[tp] + stack[tq]);
                c.stk = std::max(c.stk, H(F1_RCHIGH + STCK_F1, i, tq, tp, l));
            }
        }
        c.dh = RC + 2.0 * bk;
    }
    return p;
}
