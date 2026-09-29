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

#ifdef FIX_CLASS
// clang-format off
FixStyle(gemc,FixGEMC);
// clang-format on
#else

#ifndef LMP_FIX_GEMC_H
#define LMP_FIX_GEMC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixGEMC : public Fix {
 public:
  FixGEMC(class LAMMPS *, int, char **);
  ~FixGEMC() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void pre_exchange() override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  double compute_vector(int) override;

 private:
  // user provided inputs

  int ntranslate;             // number of translations attempted every nevery steps
  int nexchange;              // number of particle exchanges attempted every nevery steps
  int nvolume;                // number of volume exchanges attempted every nevery steps
  double box_temp;            // temperature of both boxes
  double displace;            // maximum displacement for translations
  double max_dlogvolratio;    // maximum change in log(V1/V2)
  int seed;                   // RNG seed
  int tune_every;             // adjust step sizes every this many invocations (0 = never)
  double tune_trans;          // target acceptance ratio of translations
  double tune_vol;            // target acceptance ratio of volume changes
  double tune_last[4];        // counters at the last adjustment
  int ntune_calls;            // number of invocations since the start of the run
  int full_flag;              // 1 if user requested full energy for all moves
  int local_flag;      // 1 if single-atom energies may be used for translations and exchanges
  int ghosts_stale;    // 1 if ghost atoms may be out of date

  // for evaluating probability

  double beta;             // 1 / kT
  double energy_stored;    // current potential energy
  class Compute *c_pe;     // compute to get full potential energy

  // for determining which move to make

  int nmoves;            // total MC moves (translate + exchange + volume)
  double pc_exchange;    // cumulative probability MC move is an exchange
  double pc_volume;      // cumulative probability MC move is exchange or volume change

  // for tracking how many attempts/successes

  double ntranslation_attempts;
  double ntranslation_successes;
  double nexchange_attempts;
  double nexchange_successes;
  double nvolume_attempts;
  double nvolume_successes;
  double nlast[6];       // counters at the time of the last progress message
  double logvolratio;    // log(V1/V2), identical in both boxes

  // particle - related props

  int natom_lower;                     // number of group atoms on lower ranks of my box
  int natom_local;                     // number of group atoms on this rank
  int natom_total;                     // number of group atoms in my box
  int gemc_nmax;                       // allocated length of local_gas_list
  int *local_gas_list;                 // local indices of group atoms
  std::vector<double> exchange_buf;    // storage for an atom removed during a trial exchange

  // domain - related props

  double xlo, ylo, zlo;       // lower domain bounds
  double xhi, yhi, zhi;       // upper domain bounds
  double *sublo, *subhi;      // sub domain bounds
  std::vector<Fix *> rfix;    // rigid fixes
  double voltot;              // V1+V2, conserved

  // for communication

  int me;         // rank in my box
  int myworld;    // index of my box (0 or 1)

  MPI_Comm comm_replica;    // for communication between rank 0 of both boxes

  class RanPark *random_universe;    // RNG synchronized across all ranks of both boxes
  class RanPark *random_world;       // RNG synchronized across all ranks of one box
  class RanPark *random_proc;        // RNG unique to each rank

  int progress;    // last percentage of the run reported

  void attempt_atomic_translation_full();
  void attempt_volume_change_full();
  void attempt_atomic_exchange_full();

  int accept_both(double, int);    // joint acceptance decision of both boxes
  void reset_comm();               // re-distribute atoms and rebuild ghosts and neighbor lists
  double energy_full();            // computes full potential energy
  double energy_local(int, int, tagint, double *, double * = nullptr,
                      double * = nullptr);    // pair energy of one atom
  int use_local();                            // 1 if the next move in my box may use energy_local()
  void refresh_ghosts();
  tagint insert_atom(
      int, int, double *,
      int);    // insert atom into my box                // re-distribute atoms and rebuild ghost atoms
  void update_gas_atoms_list();    // updates list of local group atoms
  int pick_random_gas_atom();      // picks random group atom
  void print_progress();
  void tune_steps();    // adjust maximum displacement and volume change
};

}    // namespace LAMMPS_NS

#endif
#endif
