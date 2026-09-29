#pragma once

// LAMMPS per-type coefficient tables for the oxDNA1/2 kernels
// (lammps_tables, on by default with lammps_overhead).
//
// Every LAMMPS KOKKOS oxDNA kernel reads its coefficients from device Views
// indexed by atom type: 2D (ntypes+1)^2 tables for the pair coefficients and
// 4D (ntypes+1)^4 tetramer tables where the coefficient depends on the 3'/5'
// flanking nucleotides (stacking radial cutoffs / shift / theta4, FENE Delta /
// r0, the bonded base-base excluded volume). For the sequence-averaged
// oxDNA1/2 parameter sets every entry holds the same values, so the tables
// reproduce the memory traffic without changing the physics: each kernel
// loads the entry of its (type a, type b[, flanks]) and passes it to the same
// physics helper as the lean path (the helpers are templated on the
// coefficient struct; member names match DNAParams).
//
// oxDNA3 needs no separate tables: its physics already reads the upstream
// sequence-dependent tetramer tables (DNA3Params::sd) per pair / bond.

#include "../types.h"
#include "params.h"
#include <Kokkos_Core.hpp>
#include <type_traits>

struct OxdnaTables {
    using RA = Kokkos::MemoryTraits<Kokkos::RandomAccess>;
    Kokkos::View<ExcvParams **>  excv_bkbk, excv_bkbs, excv_bsbs;   // 5 x 5
    Kokkos::View<ExcvParams *>   excv_bsbs4;                        // 625 (tetramer)
    Kokkos::View<HbondCoeffs **> hb;                                // 5 x 5
    Kokkos::View<XstkCoeffs **>  xstk;                              // 5 x 5
    Kokkos::View<CxstCoeffs **>  cxst;                              // 5 x 5
    Kokkos::View<StkCoeffs2 **>  stk2;                              // 5 x 5
    Kokkos::View<StkCoeffs4 *>   stk4;                              // 625
    Kokkos::View<FeneCoeffs4 *>  fene4;                             // 625 (1 bond type)
    Kokkos::View<c_number *>     fene_k;                            // nbondtypes + 1
    Kokkos::View<DhCoeffs **>    dh;                                // 5 x 5
    bool built = false;
};

// Fill (or refill) the tables from the DNAParams values.
inline void build_oxdna_tables(OxdnaTables &t, const DNAParams &par) {
    auto mk2 = [](auto &v, const char *name) {
        using V = std::remove_reference_t<decltype(v)>;
        v = V(name, 5, 5);
    };
    mk2(t.excv_bkbk, "tab_excv_bkbk"); mk2(t.excv_bkbs, "tab_excv_bkbs"); mk2(t.excv_bsbs, "tab_excv_bsbs");
    mk2(t.hb, "tab_hbond"); mk2(t.xstk, "tab_xstk"); mk2(t.cxst, "tab_coaxstk");
    mk2(t.stk2, "tab_stk2"); mk2(t.dh, "tab_dh");
    t.excv_bsbs4 = Kokkos::View<ExcvParams *>("tab_excv_bsbs4", 625);
    t.stk4       = Kokkos::View<StkCoeffs4 *>("tab_stk4", 625);
    t.fene4      = Kokkos::View<FeneCoeffs4 *>("tab_fene4", 625);
    t.fene_k     = Kokkos::View<c_number *>("tab_fene_k", 2);

    auto h_bkbk = Kokkos::create_mirror_view(t.excv_bkbk);
    auto h_bkbs = Kokkos::create_mirror_view(t.excv_bkbs);
    auto h_bsbs = Kokkos::create_mirror_view(t.excv_bsbs);
    auto h_hb   = Kokkos::create_mirror_view(t.hb);
    auto h_xs   = Kokkos::create_mirror_view(t.xstk);
    auto h_cx   = Kokkos::create_mirror_view(t.cxst);
    auto h_s2   = Kokkos::create_mirror_view(t.stk2);
    auto h_dh   = Kokkos::create_mirror_view(t.dh);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) {
            h_bkbk(i, j) = par.excv_bkbk;
            h_bkbs(i, j) = par.excv_bkbs;
            h_bsbs(i, j) = par.excv_bsbs;
            HbondCoeffs hb{par.hb_f1, par.hb_t1, par.hb_t2, par.hb_t3, par.hb_t4, par.hb_t7, par.hb_t8,
                           c_number(0)};
            if (i > 0 && j > 0) hb.eps = par.alpha_hb[i - 1][j - 1];
            h_hb(i, j) = hb;
            h_xs(i, j) = XstkCoeffs{par.xstk_f2, par.xstk_t1, par.xstk_t2, par.xstk_t3,
                                    par.xstk_t4, par.xstk_t7, par.xstk_t8};
            CxstCoeffs cx{};
            cx.cxst_f2 = par.cxst_f2; cx.cxst_t1 = par.cxst_t1; cx.cxst_t4 = par.cxst_t4;
            cx.cxst_t5 = par.cxst_t5; cx.cxst_t6 = par.cxst_t6; cx.cxst_cp = par.cxst_cp;
            cx.cxst_t1_SA = par.cxst_t1_SA; cx.cxst_t1_SB = par.cxst_t1_SB;
            cx.cxst_t1_mode = par.cxst_t1_mode; cx.cxst_has_cosphi = par.cxst_has_cosphi;
            cx.cxst_t4_blunt = par.cxst_t4_blunt; cx.d_cbk = par.d_cbk; cx.d_cstk = par.d_cstk;
            h_cx(i, j) = cx;
            h_s2(i, j) = StkCoeffs2{par.stk_f1.eps, par.stk_f1.a, par.stk_f1.b_lo, par.stk_f1.b_hi,
                                    par.stk_t4.theta_0, par.stk_t5, par.stk_t6, par.stk_cp1, par.stk_cp2};
            h_dh(i, j) = DhCoeffs{par.dh_prefactor, par.dh_minus_kappa, par.dh_B, par.dh_RHIGH, par.dh_RC,
                                  par.dh_RC * par.dh_RC};
        }
    auto h_e4 = Kokkos::create_mirror_view(t.excv_bsbs4);
    auto h_s4 = Kokkos::create_mirror_view(t.stk4);
    auto h_f4 = Kokkos::create_mirror_view(t.fene4);
    for (int n = 0; n < 625; n++) {
        h_e4(n) = par.excv_bsbs;
        const F1Params &f = par.stk_f1;
        h_s4(n) = StkCoeffs4{f.cut_0, f.cut_lc, f.cut_hc, f.cut_lo, f.cut_hi, f.shift,
                             par.stk_t4.a, par.stk_t4.dtheta_ast, par.stk_t4.b, par.stk_t4.dtheta_c};
        h_f4(n) = FeneCoeffs4{par.fene.Delta, par.fene.r0};
    }
    auto h_k = Kokkos::create_mirror_view(t.fene_k);
    h_k(0) = 0; h_k(1) = par.fene.k;
    Kokkos::deep_copy(t.excv_bkbk, h_bkbk); Kokkos::deep_copy(t.excv_bkbs, h_bkbs);
    Kokkos::deep_copy(t.excv_bsbs, h_bsbs); Kokkos::deep_copy(t.hb, h_hb);
    Kokkos::deep_copy(t.xstk, h_xs);        Kokkos::deep_copy(t.cxst, h_cx);
    Kokkos::deep_copy(t.stk2, h_s2);        Kokkos::deep_copy(t.dh, h_dh);
    Kokkos::deep_copy(t.excv_bsbs4, h_e4);
    Kokkos::deep_copy(t.stk4, h_s4);        Kokkos::deep_copy(t.fene4, h_f4);
    Kokkos::deep_copy(t.fene_k, h_k);
    t.built = true;
}

// Assemble the stacking coefficients of one bond from the 2D (a, b) and the
// 4D (a3p, a, b, b5p) tables, as pair oxdna*/stk/kk reads them.
KOKKOS_INLINE_FUNCTION
StkCoeffs assemble_stk(const StkCoeffs2 &s2, const StkCoeffs4 &s4, c_number d_cstk, c_number d_cbk) {
    StkCoeffs c;
    c.stk_f1 = F1Params{s2.eps, s2.a, s4.cut_0, s4.cut_lc, s4.cut_hc, s4.cut_lo, s4.cut_hi,
                        s2.b_lo, s2.b_hi, s4.shift};
    c.stk_t4 = F4Params{s4.t4_a, s2.t4_theta_0, s4.t4_dtheta_ast, s4.t4_b, s4.t4_dtheta_c};
    c.stk_t5 = s2.t5; c.stk_t6 = s2.t6;
    c.stk_cp1 = s2.cp1; c.stk_cp2 = s2.cp2;
    c.d_cstk = d_cstk; c.d_cbk = d_cbk;
    return c;
}
