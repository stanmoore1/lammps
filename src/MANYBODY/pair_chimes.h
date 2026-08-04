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

#include "chimesFF.h"
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

  // Scratch for build_mb_neighlists(): the neighbors of the current atom that
  // can take part in one of its clusters, and the squared distances among the
  // 4-body candidates.  Kept as members so the capacity persists.

  struct MBCand {
    int idx;       // local atom index
    tagint tag;    // global atom ID, cached to keep it off the inner loop
    double rsq;    // squared distance to the owning atom
  };

  std::vector<MBCand> cand_3b;
  std::vector<MBCand> cand_4b;
  std::vector<double> cand_rsq;

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
  std::vector<int> typ_idxs_3b;
  std::vector<int> typ_idxs_4b;

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
  void sort_3mers_by_type();
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

  static inline double dist_sq(double **x, int i, int j)
  {
    const double dx = x[j][0] - x[i][0];
    const double dy = x[j][1] - x[i][1];
    const double dz = x[j][2] - x[i][2];

    return dx * dx + dy * dy + dz * dz;
  }
  void set_chimes_type();
  void ev_tally_mb(int ninteractionatoms, const int *atmlist, double evdwl, const double *stress);

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
