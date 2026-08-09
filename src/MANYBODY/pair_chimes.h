/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Rebecca K. Lindsey (LLNL)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(chimesFF, PairCHIMES);    // PairStyle(key, class)

#else

#ifndef LMP_PAIR_CHIMES_H
#define LMP_PAIR_CHIMES_H

#include "pair.h"

#include <array>

#include "chimesFF.h"
#include <cstdint>
#include <vector>

/*    Functions required by LAMMPS:


settings     (done)        reads the input script line with arguments defined here
coeff        (done)        set coefficients for one i,j pair type
compute        (done)        workhorse routine that computes pairwise interactions
init_one    (done)        perform initalization for one i,j type pair
init_style     (done)        initialization specific to this pair style

write_restart            write i,j pair coeffs to restart file
read_restart            read i,j pair coeffs from restart file
write_restart_settings            write global settings to restart file
read_restart_settings            read global settings from restart file
single                force and energy fo a single pairwise interaction between two atoms
*/

namespace LAMMPS_NS {
class PairCHIMES : public Pair {
 public:
  // Variable definitions

  chimesFF *chimes_calculator;    // chimesFF instance

  char *chimesFF_paramfile;    // ChIMES parameter file

  std::vector<int>
      chimes_type;    // For i = LMP atom type indx, chimes_type[i-1] gives the ChIMES parameter file type idx

  double maxcut_2b;
  double maxcut_3b;
  double maxcut_4b;

  int n_3mers;    // number of neighborlist_Xmers entries
  int n_4mers;

  // Custom many-body neighbor lists, stored flat: cluster c occupies
  // neighborlist_3mers[3*c .. 3*c+2] and neighborlist_4mers[4*c .. 4*c+3].
  // Flat storage keeps the capacity across rebuilds, so the steady state does
  // no allocation at all, and the compute loops read it as one stream.

  std::vector<int> neighborlist_3mers;
  std::vector<int> neighborlist_4mers;

  // The packed cluster type index of each entry above, in the same order.  The
  // grouping pass already derives it from the atom types, and the atom types do
  // not change between rebuilds, so the force loop reads it instead of chasing
  // type[] and chimes_type[] once per atom of every cluster on every step.

  std::vector<int> mer_type_3b;
  std::vector<int> mer_type_4b;

  // Per-slot-key staging for the batched 2-body path.

  std::vector<int> b2_cnt;
  std::vector<std::array<int, CHIMES_VLEN>> b2_i, b2_j;
  std::vector<std::array<double, CHIMES_VLEN>> b2_dist;
  std::vector<std::array<std::array<double, CHDIM>, CHIMES_VLEN>> b2_dr;

  // Scratch for build_mb_neighlists(): the neighbors of the current atom that
  // can take part in one of its clusters, and the squared distances among the
  // 4-body candidates.  Kept as members so the capacity persists.

  struct MBCand {
    int idx;       // local atom index
    tagint tag;    // global atom ID, cached to keep it off the inner loop
    double rsq;    // squared distance to the owning atom
  };

  // Working storage for one atom's cluster enumeration.  It is a struct rather
  // than a set of members so that an accelerated variant can give each thread
  // its own copy and enumerate several atoms at once.

  struct MBScratch {
    std::vector<MBCand> cand_3b;
    std::vector<MBCand> cand_4b;

    // Which 4-body candidates can share a cluster.  Row a of cand_adj marks the
    // candidates within the 4-body cutoff of a, and row a of cand_ge marks those
    // whose global ID does not order before a's.  Both are bitmaps, cand_words
    // 64-bit words per row.

    std::vector<uint64_t> cand_adj;
    std::vector<uint64_t> cand_ge;
    int cand_words;
  };

  // The cutoffs and system arrays the enumeration reads, gathered once so the
  // per-atom function does not re-derive them.

  struct MBContext {
    double **x;
    tagint *tag;
    double cutsq_3b, cutsq_4b, cutsq_max;
    bool do_3b, do_4b;
  };

  MBScratch mb_scratch;

  MBContext mb_context() const;

  // Enumerate the clusters owned by atom i, appending them to out3 and out4.
  // Reads only its arguments and the neighbor list, so several atoms can be
  // enumerated at once given a scratch each.

  void mb_clusters_for_atom(int i, const MBContext &ctx, MBScratch &s, std::vector<int> &out3,
                            std::vector<int> &out4) const;

  // Scratch for grouping the triplets by cluster type (counting sort).

  std::vector<int> mer_key;
  std::vector<int> mer_scratch;
  std::vector<int> type_count;

  // 2-body vars for chimesFF access

  std::vector<double> dr;
  std::vector<double> dr_3b;
  std::vector<double> dr_4b;

  double dist;
  std::vector<double> dist_3b;
  std::vector<double> dist_4b;

  std::vector<double> force_2b;
  std::vector<double> force_3b;
  std::vector<double> force_4b;

  std::vector<int> typ_idxs_2b;

  // Constructor/Deconstructor

  PairCHIMES(class LAMMPS *);

  ~PairCHIMES() override;

  // Functions that have been written

  void settings(int narg, char **arg) override;
  void init_style() override;
  void coeff(int narg, char **arg) override;
  virtual void allocate();
  double init_one(int i, int j) override;
  void compute(int eflag, int vflag) override;
  virtual void build_mb_neighlists();
  // Counting sort of a cluster list by packed atom-type index, so the compute
  // loops see runs of one type and can batch them.  The width is a template
  // parameter so the per-cluster copy unrolls: leaving it a runtime argument
  // cost more than the sort saved.
  template <int WIDTH> void sort_mers_by_type(std::vector<int> &mers, int nmers,
                                              std::vector<int> &mer_type);
  inline double get_dist(int i, int j, double *dr);
  inline double get_dist(int i, int j);
  // Displacement and distance for a pair, but only if it is inside cutsq.
  // Returns false without taking the sqrt otherwise; dr is filled either way
  // since it is needed to decide.
  static inline bool within(double **x, int i, int j, double cutsq_pair, double *drp, double &dist)
  {
    drp[0] = x[j][0] - x[i][0];
    drp[1] = x[j][1] - x[i][1];
    drp[2] = x[j][2] - x[i][2];

    const double rsq = drp[0] * drp[0] + drp[1] * drp[1] + drp[2] * drp[2];

    if (rsq >= cutsq_pair) return false;

    dist = sqrt(rsq);

    return true;
  }

  // Position of the lowest set bit, for walking a candidate bitmap in
  // increasing candidate order.  The de Bruijn fallback keeps the cluster
  // enumeration available on compilers without the GNU builtin.

  static inline int lowest_bit(uint64_t v)
  {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(v);
#else
    static const int debruijn[64] = {0,  1,  2,  53, 3,  7,  54, 27, 4,  38, 41, 8,  34,
                                     55, 48, 28, 62, 5,  39, 46, 44, 42, 22, 9,  24, 35,
                                     59, 56, 49, 18, 29, 11, 63, 52, 6,  26, 37, 40, 33,
                                     47, 61, 45, 43, 21, 23, 58, 17, 10, 51, 25, 36, 32,
                                     60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12};

    return debruijn[((v & (~v + 1)) * 0x022fdd63cc95386dULL) >> 58];
#endif
  }

  static inline double dist_sq(double **x, int i, int j)
  {
    const double dx = x[j][0] - x[i][0];
    const double dy = x[j][1] - x[i][1];
    const double dz = x[j][2] - x[i][2];

    return dx * dx + dy * dy + dz * dz;
  }
  void set_chimes_type();
  void ev_tally_mb(int ninteractionatoms, const int *atmlist, double evdwl, const double *stress);

  // Neighbor list access used by build_mb_neighlists() and the 2-body loop.
  // Reading it through these members rather than through `list` directly lets
  // an accelerated variant supply the same lists from its own storage, so the
  // cluster enumeration and the pair loop are written once.

  int nl_inum;
  int *nl_ilist, *nl_numneigh, **nl_firstneigh;

  virtual void setup_neighlist_ptrs();

  // Functions I haven't worked on

  void write_restart();
  void read_restart();
  void write_restart_settings();
  void read_restart_settings();
  void single();
};
}    // namespace LAMMPS_NS

#endif
#endif
