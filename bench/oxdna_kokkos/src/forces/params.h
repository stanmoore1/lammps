#pragma once

// oxDNA1 force-field parameters.
// Values are taken directly from the standalone oxDNA `src/model.h` (lj/reduced
// units), so the Kokkos port reproduces the standalone oxDNA1 interaction.
// Derived quantities (lj1, lj2, smoothing b, outer cutoff) are computed at
// initialization from the fundamental input parameters.

#include "../types.h"
#include <cmath>
#include <Kokkos_Core.hpp>

// Compute the smoothing coefficient b and cutoff c for the F3 potential
// given sigma and cut_ast (transition point from LJ to smooth quadratic).
inline void f3_smoothing(double sigma, double cut_ast, double eps,
                         double &b_out, double &cut_c_out) {
    double sigma_over_cut  = sigma / cut_ast;
    double s6  = std::pow(sigma_over_cut, 6.0);
    double s12 = s6 * s6;
    double s7  = std::pow(sigma_over_cut, 7.0);
    double s13 = s6 * s7;

    double dU_at_ast = 4.0 / sigma * (6.0 * s7 - 12.0 * s13);
    double U_at_ast  = 4.0 * (s12 - s6);

    b_out     = (dU_at_ast * dU_at_ast) / (4.0 * U_at_ast);
    cut_c_out = cut_ast - 2.0 * U_at_ast / dU_at_ast;
}

// Parameters for a single F3 (excluded-volume) site-site interaction
struct ExcvParams {
    c_number eps;
    c_number lj1;       // 4*eps*sigma^12
    c_number lj2;       // 4*eps*sigma^6
    c_number b;         // smoothing coefficient
    c_number cutsq_ast; // (cut_ast)^2
    c_number cutsq_c;   // (cut_c)^2
    c_number cut_c;     // outer cutoff

    void init(double eps_in, double sigma, double cut_ast) {
        eps = static_cast<c_number>(eps_in);
        double b_d, cut_c_d;
        f3_smoothing(sigma, cut_ast, eps_in, b_d, cut_c_d);
        lj1     = static_cast<c_number>(4.0 * eps_in * std::pow(sigma, 12.0));
        lj2     = static_cast<c_number>(4.0 * eps_in * std::pow(sigma,  6.0));
        b       = static_cast<c_number>(b_d);
        cutsq_ast = static_cast<c_number>(cut_ast * cut_ast);
        cut_c   = static_cast<c_number>(cut_c_d);
        cutsq_c = static_cast<c_number>(cut_c_d * cut_c_d);
    }
};

// Parameters for the FENE backbone bond
struct FeneParams {
    c_number k;      // spring constant (=2.0 in default)
    c_number Delta;  // half-max extension (=0.25)
    c_number r0;     // equilibrium length (=0.7525)
};

// Parameters for the F1 radial modulation (H-bonding, stacking)
struct F1Params {
    c_number eps;
    c_number a;
    c_number cut_0;   // Morse minimum
    c_number cut_lc;  // lower smooth cutoff
    c_number cut_hc;  // upper hard cutoff
    c_number cut_lo;  // lower Morse boundary
    c_number cut_hi;  // upper Morse boundary
    c_number b_lo;
    c_number b_hi;
    c_number shift;
};

// Parameters for one angular term (F4)
struct F4Params {
    c_number a;
    c_number theta_0;
    c_number dtheta_ast;
    c_number b;
    c_number dtheta_c;
};

// Parameters for one F5 (dihedral) term
struct F5Params {
    c_number a;
    c_number x_ast;
    c_number b;
    c_number x_c;
};

// Parameters for one F2 term (cross-stacking / coaxial-stacking radial)
struct F2Params {
    c_number k;
    c_number cut_0;
    c_number cut_lc;
    c_number cut_hc;
    c_number cut_lo;
    c_number cut_hi;
    c_number b_lo;
    c_number b_hi;
    c_number cut_c;
};

// =====================================================================
// All force-field parameters in one struct, stored by value in the
// force functors so they are accessible on device.
// =====================================================================
struct DNAParams {
    // --- Excluded volume (backbone-backbone, backbone-base, base-base) ---
    ExcvParams excv_bkbk;   // backbone-backbone (EXCL_S1/R1)
    ExcvParams excv_bkbs;   // backbone-base     (EXCL_S3/R3)
    ExcvParams excv_bsbs;   // base-base         (EXCL_S2/R2)

    // Site offsets from COM along nx (oxDNA1 convention)
    c_number d_cbk;  // backbone site: POS_BACK  = -0.4
    c_number d_cbs;  // base site:     POS_BASE  = +0.4
    c_number d_cstk; // stacking site: POS_STACK = +0.34

    // --- FENE backbone bond ---
    FeneParams fene;

    // --- Hydrogen bonding ---
    F1Params hb_f1;
    F4Params hb_t1, hb_t2, hb_t3, hb_t4, hb_t7, hb_t8;
    // Watson-Crick complementarity gate / strength (A=0,C=1,G=2,T=3).
    // Nonzero only for A:T, T:A, C:G, G:C (HYDR_EPS_OXDNA = 1.077).
    c_number alpha_hb[4][4];

    // --- Stacking (bonded) ---
    F1Params stk_f1;
    F4Params stk_t4, stk_t5, stk_t6;
    F5Params stk_cp1, stk_cp2;

    // --- Cross-stacking (nonbonded) ---
    F2Params xstk_f2;
    F4Params xstk_t1, xstk_t2, xstk_t3, xstk_t4, xstk_t7, xstk_t8;

    // --- Coaxial stacking (nonbonded) ---
    F2Params cxst_f2;
    F4Params cxst_t1, cxst_t4, cxst_t5, cxst_t6;
    F5Params cxst_cp;   // phi3 (=phi4)

    // Global nonbonded COM-COM cutoff squared (max over all terms)
    c_number cutsq_nb;
};

// make_f4: build an F4 term from (a, theta0, dtheta_ast), deriving the
// smoothing coefficient b and the outer cutoff dtheta_c from C1 continuity.
// Matches the standalone F4_THETA_B / F4_THETA_TC values in model.h.
inline F4Params make_f4(double a, double th0, double dth_ast) {
    F4Params f;
    f.a          = static_cast<c_number>(a);
    f.theta_0    = static_cast<c_number>(th0);
    f.dtheta_ast = static_cast<c_number>(dth_ast);
    double dU    = 2 * a * dth_ast;
    double U     = 1 - a * dth_ast * dth_ast;
    f.b          = static_cast<c_number>(dU * dU / (4 * U));
    f.dtheta_c   = static_cast<c_number>(dth_ast + 2 * U / dU);
    return f;
}

// make_f5: build an F5 term from (a, x_ast). x_ast is stored negated because
// the standalone F5 acts on cos(phi) with negative XS/XC boundaries.
inline F5Params make_f5(double a, double x_ast) {
    F5Params f;
    f.a     = static_cast<c_number>(a);
    f.x_ast = static_cast<c_number>(-x_ast);
    double dU  = 2 * a * x_ast;
    double U   = 1 - a * x_ast * x_ast;
    f.b     = static_cast<c_number>(dU * dU / (4 * U));
    f.x_c   = static_cast<c_number>(-x_ast - 2 * U / dU);
    return f;
}

// Initialize all parameters for oxDNA1 (standalone model.h, average-sequence).
//   T        : reduced temperature (sets stacking strength)
//   hb_multi : extra additive H-bond strength (default 0 → HYDR_EPS_OXDNA)
inline DNAParams make_oxdna1_params(double T = 0.1, double hb_multi = 0.0) {
    DNAParams p;
    constexpr double PI  = 3.141592653589793;
    constexpr double PI2 = PI / 2;

    // ---- Site offsets (model.h POS_*) ----
    p.d_cbk  = static_cast<c_number>(-0.4);   // POS_BACK
    p.d_cbs  = static_cast<c_number>( 0.4);   // POS_BASE
    p.d_cstk = static_cast<c_number>( 0.34);  // POS_STACK

    // ---- Excluded volume (model.h EXCL_*) ----
    // EXCL_EPS=2; backbone-backbone S1/R1, backbone-base S3/R3, base-base S2/R2.
    p.excv_bkbk.init(2.0, 0.70,  0.675);
    p.excv_bkbs.init(2.0, 0.515, 0.50);
    p.excv_bsbs.init(2.0, 0.33,  0.32);

    // ---- FENE backbone bond ----
    // FENE_EPS=2, FENE_DELTA=0.25, FENE_R0_OXDNA=0.7525
    p.fene.k     = static_cast<c_number>(2.0);
    p.fene.Delta = static_cast<c_number>(0.25);
    p.fene.r0    = static_cast<c_number>(0.7525);

    // ---- Hydrogen bonding (model.h HYDR_*) ----
    // Radial F1: eps carried by alpha_hb gate; a=8, R0=0.4, RLOW/RHIGH=0.34/0.7,
    // RCLOW/RCHIGH=0.276908/0.783775, BLOW=-126.243, BHIGH=-7.87708.
    {
        const double eps   = 1.0;        // strength supplied by alpha_hb below
        const double a_hb  = 8.0;        // HYDR_A
        const double r0    = 0.4;        // HYDR_R0  (Morse centre, cut_0)
        const double rc    = 0.75;       // HYDR_RC  (only used for the shift)
        const double rlow  = 0.34;       // HYDR_RLOW   -> cut_lo
        const double rhigh = 0.7;        // HYDR_RHIGH  -> cut_hi
        const double rclow = 0.276908;   // HYDR_RCLOW  -> cut_lc
        const double rchigh= 0.783775;   // HYDR_RCHIGH -> cut_hc
        const double blow  = -126.243;   // HYDR_BLOW
        const double bhigh = -7.87708;   // HYDR_BHIGH
        const double m     = 1.0 - std::exp(-(rc - r0) * a_hb);
        p.hb_f1.eps    = static_cast<c_number>(eps);
        p.hb_f1.a      = static_cast<c_number>(a_hb);
        p.hb_f1.cut_0  = static_cast<c_number>(r0);
        p.hb_f1.cut_lo = static_cast<c_number>(rlow);
        p.hb_f1.cut_hi = static_cast<c_number>(rhigh);
        p.hb_f1.cut_lc = static_cast<c_number>(rclow);
        p.hb_f1.cut_hc = static_cast<c_number>(rchigh);
        p.hb_f1.b_lo   = static_cast<c_number>(blow);
        p.hb_f1.b_hi   = static_cast<c_number>(bhigh);
        p.hb_f1.shift  = static_cast<c_number>(eps * m * m);
    }

    // Angular F4 (model.h HYDR_THETA*): t1=t2=t3, t4, t7=t8.
    p.hb_t1 = make_f4(1.5,  0.0, 0.7);
    p.hb_t2 = make_f4(1.5,  0.0, 0.7);
    p.hb_t3 = make_f4(1.5,  0.0, 0.7);
    p.hb_t4 = make_f4(0.46, PI,  0.7);
    p.hb_t7 = make_f4(4.0,  PI2, 0.45);
    p.hb_t8 = make_f4(4.0,  PI2, 0.45);

    // Watson-Crick gate / strength (HYDR_EPS_OXDNA = 1.077). Sum of types == 3
    // identifies complementary pairs: A(0)+T(3), C(1)+G(2).
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            p.alpha_hb[i][j] = static_cast<c_number>(0.0);
    const c_number hb_eps = static_cast<c_number>(1.077 + hb_multi);
    p.alpha_hb[0][3] = hb_eps;  // A:T
    p.alpha_hb[3][0] = hb_eps;  // T:A
    p.alpha_hb[1][2] = hb_eps;  // C:G
    p.alpha_hb[2][1] = hb_eps;  // G:C

    // ---- Stacking (model.h STCK_*) ----
    {
        const double stk_a   = 6.0;                     // STCK_A
        const double stk_r0  = 0.4;                     // STCK_R0
        const double stk_rc  = 0.9;                     // STCK_RC (for shift)
        const double stk_eps = 1.3448 + 2.6568 * T;     // BASE_EPS + FACT_EPS*T
        p.stk_f1.eps    = static_cast<c_number>(stk_eps);
        p.stk_f1.a      = static_cast<c_number>(stk_a);
        p.stk_f1.cut_0  = static_cast<c_number>(stk_r0);
        p.stk_f1.cut_lo = static_cast<c_number>(0.32);     // STCK_RLOW
        p.stk_f1.cut_hi = static_cast<c_number>(0.75);     // STCK_RHIGH
        p.stk_f1.cut_lc = static_cast<c_number>(0.23239);  // STCK_RCLOW
        p.stk_f1.cut_hc = static_cast<c_number>(0.956);    // STCK_RCHIGH
        p.stk_f1.b_lo   = static_cast<c_number>(-68.1857); // STCK_BLOW
        p.stk_f1.b_hi   = static_cast<c_number>(-3.12992); // STCK_BHIGH
        const double m  = 1.0 - std::exp(-stk_a * (stk_rc - stk_r0));
        p.stk_f1.shift  = static_cast<c_number>(stk_eps * m * m);
    }
    p.stk_t4 = make_f4(1.3, 0.0, 0.8);   // STCK_THETA4
    p.stk_t5 = make_f4(0.9, 0.0, 0.95);  // STCK_THETA5
    p.stk_t6 = make_f4(0.9, 0.0, 0.95);  // STCK_THETA6
    p.stk_cp1 = make_f5(2.0, 0.65);      // STCK_PHI1
    p.stk_cp2 = make_f5(2.0, 0.65);      // STCK_PHI2

    // ---- Cross-stacking (model.h CRST_*) ----
    // F2: K=47.5, R0=0.575, RC=0.675, RLOW/RCLOW=0.495/0.45,
    //     RHIGH/RCHIGH=0.655/0.7, BLOW=BHIGH=-0.888889.
    p.xstk_f2.k      = static_cast<c_number>(47.5);
    p.xstk_f2.cut_0  = static_cast<c_number>(0.575);
    p.xstk_f2.cut_c  = static_cast<c_number>(0.675);
    p.xstk_f2.cut_lo = static_cast<c_number>(0.495);
    p.xstk_f2.cut_lc = static_cast<c_number>(0.45);
    p.xstk_f2.cut_hi = static_cast<c_number>(0.655);
    p.xstk_f2.cut_hc = static_cast<c_number>(0.7);
    p.xstk_f2.b_lo   = static_cast<c_number>(-0.888889);
    p.xstk_f2.b_hi   = static_cast<c_number>(-0.888889);
    p.xstk_t1 = make_f4(2.25, PI - 2.35, 0.58);  // CRST_THETA1
    p.xstk_t2 = make_f4(1.70, 1.0,       0.68);  // CRST_THETA2
    p.xstk_t3 = make_f4(1.70, 1.0,       0.68);  // CRST_THETA3
    p.xstk_t4 = make_f4(1.50, 0.0,       0.65);  // CRST_THETA4
    p.xstk_t7 = make_f4(1.70, 0.875,     0.68);  // CRST_THETA7
    p.xstk_t8 = make_f4(1.70, 0.875,     0.68);  // CRST_THETA8

    // ---- Coaxial stacking (model.h CXST_*, oxDNA1) ----
    // F2: K=46, R0=0.4, RC=0.6, RLOW/RCLOW=0.22/0.177778,
    //     RHIGH/RCHIGH=0.58/0.6222222, BLOW=BHIGH=-2.13158.
    p.cxst_f2.k      = static_cast<c_number>(46.0);
    p.cxst_f2.cut_0  = static_cast<c_number>(0.400);
    p.cxst_f2.cut_c  = static_cast<c_number>(0.6);
    p.cxst_f2.cut_lo = static_cast<c_number>(0.22);
    p.cxst_f2.cut_lc = static_cast<c_number>(0.177778);
    p.cxst_f2.cut_hi = static_cast<c_number>(0.58);
    p.cxst_f2.cut_hc = static_cast<c_number>(0.6222222);
    p.cxst_f2.b_lo   = static_cast<c_number>(-2.13158);
    p.cxst_f2.b_hi   = static_cast<c_number>(-2.13158);
    p.cxst_t1 = make_f4(2.0, PI - 0.60, 0.65);  // CXST_THETA1 (oxDNA1 T0)
    p.cxst_t4 = make_f4(1.3, 0.0,       0.8);   // CXST_THETA4
    p.cxst_t5 = make_f4(0.9, 0.0,       0.95);  // CXST_THETA5
    p.cxst_t6 = make_f4(0.9, 0.0,       0.95);  // CXST_THETA6
    p.cxst_cp = make_f5(2.0, 0.65);             // CXST_PHI3 (=PHI4)

    // ---- Global nonbonded COM-COM cutoff ----
    // Largest site-site range plus the two site offsets from the COM (~0.4 each).
    double max_cut = static_cast<double>(p.excv_bkbk.cut_c) + 0.8;
    max_cut = std::max(max_cut, static_cast<double>(p.hb_f1.cut_hc)   + 0.8);
    max_cut = std::max(max_cut, static_cast<double>(p.xstk_f2.cut_hc) + 0.8);
    max_cut = std::max(max_cut, static_cast<double>(p.cxst_f2.cut_hc) + 0.8);
    p.cutsq_nb = static_cast<c_number>(max_cut * max_cut);

    return p;
}
