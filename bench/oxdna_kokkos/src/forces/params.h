#pragma once

// oxDNA1 force-field parameters.
// All values are for lj-units as used in the LAMMPS CG-DNA package.
// Derived quantities (lj1, lj2, b, cut_c) are computed at initialization
// from the fundamental input parameters (epsilon, sigma, cut_ast).
//
// Reference input:
//   pair_coeff * * oxdna/excv 2.0 0.7 0.675  2.0 0.515 0.5  2.0 0.33 0.32
//   pair_coeff * * oxdna/hbond seqav 0.0 8.0 0.4 0.75 ...
//   pair_coeff * * oxdna/stk  seqav T 1.3448 2.6568 ...
//   bond_coeff  * 2.0 0.25 0.7525  (FENE: k, Delta, r0)

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

// Parameters for one F2 term (cross-stacking radial)
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
// All force-field parameters in one struct, stored in a Kokkos::View
// so they are accessible on device. Using a single-element View<DNAParams>
// keeps the data together for potential constant-memory mapping.
// =====================================================================
struct DNAParams {
    // --- Excluded volume (backbone-backbone, backbone-base, base-base) ---
    ExcvParams excv_bkbk;   // backbone-backbone (type-independent for DNA1)
    ExcvParams excv_bkbs;   // backbone-base
    ExcvParams excv_bsbs;   // base-base (non-neighbour)

    // Site offsets from COM along nx (oxDNA1 convention)
    c_number d_cbk;  // backbone site: -0.4
    c_number d_cbs;  // base site:     +0.4
    c_number d_cstk; // stacking site: +0.34

    // --- FENE backbone bond ---
    FeneParams fene;

    // --- Hydrogen bonding (seqav, default hbond coeff from in.duplex1) ---
    // F1 radial: args 2-11 of hbond pair_coeff
    F1Params hb_f1;
    // F4 angular: theta1..theta8 (six angles)
    F4Params hb_t1, hb_t2, hb_t3, hb_t4, hb_t7, hb_t8;
    // Sequence-specific h-bond strength multiplier
    // AT=0, TA=0, GC=0, CG=0 by default; AT=TA=1.077 for cognate pairs
    // Stored as 4x4 matrix (indexed by [atype][btype], A=0,C=1,G=2,T=3)
    c_number alpha_hb[4][4];

    // --- Stacking (seqav single-temperature) ---
    // F1 radial
    F1Params stk_f1;
    // F4 angular: theta4, theta5, theta6
    F4Params stk_t4, stk_t5, stk_t6;
    // F5 dihedral: cosphi1, cosphi2
    F5Params stk_cp1, stk_cp2;

    // --- Cross-stacking ---
    // F2 radial
    F2Params xstk_f2;
    F4Params xstk_t1, xstk_t2, xstk_t3, xstk_t4, xstk_t7, xstk_t8;
    F5Params xstk_cp1, xstk_cp2, xstk_cp3, xstk_cp4;

    // Global nonbonded cutoff squared (max over all terms)
    c_number cutsq_nb;
};

// Initialize all parameters for oxDNA1 with default values from
// the LAMMPS cgdna duplex1 example (lj units).
// hb_multi: sequence-averaged h-bond strength multiplier (1.077 for AT/GC pairs)
inline DNAParams make_oxdna1_params(double T = 0.1, double hb_multi = 0.0) {
    DNAParams p;

    // ---- Site offsets ----
    p.d_cbk  = static_cast<c_number>(-0.4);
    p.d_cbs  = static_cast<c_number>( 0.4);
    p.d_cstk = static_cast<c_number>( 0.34);

    // ---- Excluded volume ----
    // pair_coeff * * oxdna/excv 2.0 0.7 0.675  2.0 0.515 0.5  2.0 0.33 0.32
    p.excv_bkbk.init(2.0, 0.7,   0.675);
    p.excv_bkbs.init(2.0, 0.515, 0.5);
    p.excv_bsbs.init(2.0, 0.33,  0.32);

    // ---- FENE backbone bond ----
    // bond_coeff * 2.0 0.25 0.7525
    p.fene.k     = static_cast<c_number>(2.0);
    p.fene.Delta = static_cast<c_number>(0.25);
    p.fene.r0    = static_cast<c_number>(0.7525);

    // ---- Hydrogen bonding ----
    // pair_coeff * * oxdna/hbond seqav 0.0 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7
    //              0.46 3.14159 0.7  4.0 1.5707963 0.45  4.0 1.5707963 0.45
    {
        // radial F1
        p.hb_f1.eps     = static_cast<c_number>(8.0);
        p.hb_f1.a       = static_cast<c_number>(0.4);
        p.hb_f1.cut_0   = static_cast<c_number>(0.75);
        p.hb_f1.cut_lc  = static_cast<c_number>(0.34);
        p.hb_f1.cut_hc  = static_cast<c_number>(0.7);
        // The 1.5 0 pairs encode: cut_lo, cut_hi and smoothing coefficients b_lo, b_hi, shift
        // These are the standard values from the LAMMPS coeff block:
        p.hb_f1.cut_lo  = static_cast<c_number>(0.7);
        p.hb_f1.cut_hi  = static_cast<c_number>(0.7);
        p.hb_f1.b_lo    = static_cast<c_number>(0.0);
        p.hb_f1.b_hi    = static_cast<c_number>(0.0);
        p.hb_f1.shift   = static_cast<c_number>(0.0);
        // Actually the full hbond pair_coeff parsing is more complex.
        // The format is: seqav eps a cut_0 cut_lc cut_hc cut_lo cut_hi b_lo b_hi shift
        // From in.duplex1: seqav 0.0  8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7
        // First 0.0 is the sequence-specific multiplier override (0.0=use seqav)
        // Then: eps a cut_0 cut_lc cut_hc [b_hb_lo=auto] [b_hb_hi=auto]
        // Actual parameter block from pair_oxdna_hbond.cpp coeff():
        // eps=8.0, a=0.4, cut_hb_0=0.75, cut_hb_lc=0.34, cut_hb_hc=0.7,
        // cut_hb_lo=0.7 (from 1.5), cut_hb_hi=0.7 (smoothed)
        // b_lo and b_hi auto-derived for continuity; shift auto-derived
        // We use exact computed values below.
        double eps = 8.0, a_hb = 0.4, cut0 = 0.75, cutlc = 0.34, cuthc = 0.7;
        double cutlo = 0.7, cuthi = 0.7;
        // Smoothing coefficients b_lo, b_hi from continuity conditions:
        double b_lo, b_hi, shift;
        {
            double explo = std::exp(-(cutlo - cut0) * a_hb);
            double Ulo   = eps * (1 - explo) * (1 - explo);
            double dUlo  = 2 * eps * (1 - explo) * explo * a_hb;
            b_lo = dUlo * dUlo / (4 * Ulo);
            double cutlc_d = cutlo - 2 * Ulo / dUlo;
            (void)cutlc_d;
        }
        {
            double exphi = std::exp(-(cuthi - cut0) * a_hb);
            double Uhi   = eps * (1 - exphi) * (1 - exphi);
            double dUhi  = 2 * eps * (1 - exphi) * exphi * a_hb;
            b_hi = dUhi * dUhi / (4 * Uhi);
            double cuthc_d = cuthi - 2 * Uhi / dUhi;
            (void)cuthc_d;
        }
        {
            // shift = F1(cutlc) via smooth branch
            // Actually shift is chosen so F1(cutlc)=0 via Morse branch
            double exps = std::exp(-(cutlc - cut0) * a_hb);
            shift = eps * (1 - exps) * (1 - exps);
        }
        p.hb_f1.eps    = static_cast<c_number>(eps);
        p.hb_f1.a      = static_cast<c_number>(a_hb);
        p.hb_f1.cut_0  = static_cast<c_number>(cut0);
        p.hb_f1.cut_lc = static_cast<c_number>(cutlc);
        p.hb_f1.cut_hc = static_cast<c_number>(cuthc);
        p.hb_f1.cut_lo = static_cast<c_number>(cutlo);
        p.hb_f1.cut_hi = static_cast<c_number>(cuthi);
        p.hb_f1.b_lo   = static_cast<c_number>(b_lo);
        p.hb_f1.b_hi   = static_cast<c_number>(b_hi);
        p.hb_f1.shift  = static_cast<c_number>(shift);
    }

    // Angular F4 parameters from in.duplex1 hbond pair_coeff:
    // theta1: 0.46 pi 0.7   -> a=0.46, theta0=pi, dtheta_ast=0.7
    //   b and dtheta_c from continuity
    auto make_f4 = [](double a, double th0, double dth_ast) -> F4Params {
        F4Params f;
        f.a         = static_cast<c_number>(a);
        f.theta_0   = static_cast<c_number>(th0);
        f.dtheta_ast= static_cast<c_number>(dth_ast);
        double dU   = 2 * a * dth_ast;
        double U    = 1 - a * dth_ast * dth_ast;
        f.b         = static_cast<c_number>(dU * dU / (4 * U));
        f.dtheta_c  = static_cast<c_number>(dth_ast + 2 * U / dU);
        return f;
    };
    constexpr double PI = 3.141592653589793;
    constexpr double PI2 = PI / 2;

    p.hb_t1 = make_f4(0.46, PI,  0.7);
    p.hb_t2 = make_f4(4.0,  PI2, 0.45);
    p.hb_t3 = make_f4(4.0,  PI2, 0.45);
    p.hb_t4 = make_f4(1.5,  PI,  0.7);   // theta4 (not explicitly set in seqav, using same as t1 per convention)
    p.hb_t7 = make_f4(4.0,  PI2, 0.45);
    p.hb_t8 = make_f4(4.0,  PI2, 0.45);

    // Sequence-specific H-bond matrix (A=0, C=1, G=2, T=3)
    // Default: all 0 except AT (0,3) and TA (3,0) and GC (2,1) and CG (1,2)
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            p.alpha_hb[i][j] = static_cast<c_number>(0.0);
    p.alpha_hb[0][3] = static_cast<c_number>(1.077 + hb_multi);  // A:T
    p.alpha_hb[3][0] = static_cast<c_number>(1.077 + hb_multi);  // T:A
    p.alpha_hb[1][2] = static_cast<c_number>(1.077 + hb_multi);  // C:G
    p.alpha_hb[2][1] = static_cast<c_number>(1.077 + hb_multi);  // G:C

    // ---- Stacking ----
    // F1 radial parameters from standalone oxDNA model.h:
    // BASE_EPS=1.3448, FACT_EPS=2.6568, A=6.0, R0=0.4, RC=0.9
    // RLOW=0.32, RHIGH=0.75, RCLOW=0.23239, RCHIGH=0.956
    // BLOW=-68.1857, BHIGH=-3.12992
    // shift = eps*(1-exp(-A*(RC-R0)))^2 = eps*(1-exp(-3))^2 ~ 0.9029*eps
    {
        double stk_a   = 6.0;
        double stk_r0  = 0.4;
        double stk_rc  = 0.9;
        double stk_eps = 1.3448 + 2.6568 * T;
        p.stk_f1.eps    = static_cast<c_number>(stk_eps);
        p.stk_f1.a      = static_cast<c_number>(stk_a);
        p.stk_f1.cut_0  = static_cast<c_number>(stk_r0);
        p.stk_f1.cut_lo = static_cast<c_number>(0.32);
        p.stk_f1.cut_hi = static_cast<c_number>(0.75);
        p.stk_f1.cut_lc = static_cast<c_number>(0.23239);
        p.stk_f1.cut_hc = static_cast<c_number>(0.956);
        p.stk_f1.b_lo   = static_cast<c_number>(-68.1857);
        p.stk_f1.b_hi   = static_cast<c_number>(-3.12992);
        double ex_rc    = std::exp(-stk_a * (stk_rc - stk_r0));
        p.stk_f1.shift  = static_cast<c_number>(stk_eps * (1 - ex_rc) * (1 - ex_rc));
    }

    // F4 stacking angles from STCK_THETA{4,5,6} in standalone model.h:
    //   t4: a=1.3, theta0=0, dtheta_ast=0.8   (STCK_THETA4_A=1.3)
    //   t5: a=0.9, theta0=0, dtheta_ast=0.95
    //   t6: a=0.9, theta0=0, dtheta_ast=0.95
    p.stk_t4 = make_f4(1.3, 0.0, 0.8);
    p.stk_t5 = make_f4(0.9, 0.0, 0.95);
    p.stk_t6 = make_f4(0.9, 0.0, 0.95);
    // F5 cosphi: a=2.0, x_ast=0.65, b=auto, x_c=auto
    auto make_f5 = [](double a, double x_ast) -> F5Params {
        F5Params f;
        f.a     = static_cast<c_number>(a);
        f.x_ast = static_cast<c_number>(-x_ast);  // store negated: LAMMPS calls F5 with -x_ast
        double dU  = 2 * a * x_ast;
        double U   = 1 - a * x_ast * x_ast;
        f.b     = static_cast<c_number>(dU * dU / (4 * U));
        f.x_c   = static_cast<c_number>(-x_ast - 2 * U / dU);
        return f;
    };
    p.stk_cp1 = make_f5(2.0, 0.65);
    p.stk_cp2 = make_f5(2.0, 0.65);

    // ---- Cross-stacking ----
    // pair_coeff * * oxdna/xstk 47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793
    //   0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0 0.65 1.7 0.875 0.68 1.7 0.875 0.68
    {
        double xstk_k=47.5, xstk_cut0=0.575, xstk_cutlc=0.495, xstk_cuthc=0.655,
               xstk_cutlo=2.25, xstk_cuthi=0.791592653589793, xstk_cutc=0.58;
        // F2: k cut0 cutlc cuthc cutlo cuthi b_lo b_hi cut_c
        p.xstk_f2.k      = static_cast<c_number>(xstk_k);
        p.xstk_f2.cut_0  = static_cast<c_number>(xstk_cut0);
        p.xstk_f2.cut_lc = static_cast<c_number>(xstk_cutlc);
        p.xstk_f2.cut_hc = static_cast<c_number>(xstk_cuthc);
        p.xstk_f2.cut_lo = static_cast<c_number>(xstk_cutlo);
        p.xstk_f2.cut_hi = static_cast<c_number>(xstk_cuthi);
        p.xstk_f2.cut_c  = static_cast<c_number>(xstk_cutc);
        // b_lo, b_hi from continuity of F2
        {
            double U_lo = xstk_k * 0.5 * ((xstk_cutlo - xstk_cut0)*(xstk_cutlo - xstk_cut0)
                                         - (xstk_cut0 - xstk_cutc)*(xstk_cut0 - xstk_cutc));
            double dU_lo= xstk_k * (xstk_cutlo - xstk_cut0);
            p.xstk_f2.b_lo = static_cast<c_number>(dU_lo * dU_lo / (4 * U_lo));

            double U_hi = xstk_k * 0.5 * ((xstk_cuthi - xstk_cut0)*(xstk_cuthi - xstk_cut0)
                                          - (xstk_cut0 - xstk_cutc)*(xstk_cut0 - xstk_cutc));
            double dU_hi= xstk_k * (xstk_cuthi - xstk_cut0);
            p.xstk_f2.b_hi = static_cast<c_number>(dU_hi * dU_hi / (4 * U_hi));
        }
    }
    // cross-stacking angles: 1.7 1.0 0.68  1.7 1.0 0.68  1.5 0 0.65  1.7 0.875 0.68  1.7 0.875 0.68
    p.xstk_t1 = make_f4(1.7, 1.0, 0.68);
    p.xstk_t2 = make_f4(1.7, 1.0, 0.68);
    p.xstk_t3 = make_f4(1.5, 0.0, 0.65);
    p.xstk_t4 = make_f4(1.7, 0.875, 0.68);
    p.xstk_t7 = make_f4(1.7, 0.875, 0.68);
    p.xstk_t8 = make_f4(1.7, 0.875, 0.68);
    p.xstk_cp1 = make_f5(2.0, 0.65);
    p.xstk_cp2 = make_f5(2.0, 0.65);
    p.xstk_cp3 = make_f5(2.0, 0.65);
    p.xstk_cp4 = make_f5(2.0, 0.65);

    // ---- Global nonbonded cutoff ----
    // max cutoff for neighbor list: use backbone-backbone hard cutoff + skin
    double max_cut = static_cast<double>(p.excv_bkbk.cut_c);
    max_cut = std::max(max_cut, static_cast<double>(p.hb_f1.cut_hc) + 0.8); // hbond COM-COM range
    max_cut = std::max(max_cut, static_cast<double>(p.stk_f1.cut_hc) + 0.8);
    p.cutsq_nb = static_cast<c_number>(max_cut * max_cut);

    return p;
}
