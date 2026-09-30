#pragma once

// LAMMPS pair cutoffs of the oxDNA styles, as init_one() returns them on
// oxdna3KK-kk-fixes (e1a8c85f05, 1a43ca2e27): the neighbor lists hold pairs by
// the distance of the nucleotide centers of mass, so each style's cutoff is its
// interaction-site cutoff plus the distances of the two sites from the COM
// (max over the type pairs):
//
//   excv     max(bkbk + 2 bk, bkbs + bk + max(bs_i, bs_j), bsbs + bs_i + bs_j,
//                bonded bsbs (oxDNA3 cut4_bsbs) + bs_i + bs_j)
//   stk      cut_st_hc (site offsets not added: stk acts on bonded pairs only)
//   hbond    cut_hb_hc + bs_i + bs_j
//   xstk     cut_xst_hc (oxDNA3: over the 33 / 55 tetramer tables) + bs_i + bs_j
//   coaxstk  cut_cxst_hc + 2 stack
//   dh       cut_dh_c + 2 bk
//
// pair_style hybrid/overlay trims each sub-style's list to its own cutoff +
// skin (e2f233566c, neigh/trim on by default); fix OXDNA/NPAIR/kk screens its
// list at max(hbond, xstk, coaxstk) + skin (request_screen_cutoff()).

#include <algorithm>

struct LmpStyleCuts {
    double excv = 0, stk = 0, hbond = 0, xstk = 0, coaxstk = 0, dh = 0;   // dh = 0: no dh style

    // cutforce: the largest cutoff of all styles (master list radius - skin)
    double cutforce() const { return std::max({excv, stk, hbond, xstk, coaxstk, dh}); }
    // fix OXDNA/NPAIR/kk screen cutoff (without the skin)
    double screen() const { return std::max({hbond, xstk, coaxstk}); }
};
