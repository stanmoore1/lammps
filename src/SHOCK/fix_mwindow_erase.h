/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

# ifdef FIX_CLASS

FixStyle(mwindow/erase,FixMWindowErase)

#else

#ifndef FIX_MWINDOW_ERASE
#define FIX_MWINDOW_ERASE

#include "stdio.h"
#include "fix.h"

namespace LAMMPS_NS {

class FixMWindowErase : public Fix {
 public:
  int mw_erase_dim;                     // Moving Window technique erasing dim
  int mw_erase_side;                    // Moving Window technique erasing side
  double mw_erase_position_d;           // Moving Window technique erasing plane position.
  double mw_erase_d_min;                // Moving Window technique erasing plane minimum position.
  double mw_erase_d_max;                // Moving Window technique erasing plane maximum position.
  double mw_erase_position_Aerase;      // Moving Window technique erasing plane position.
  double mw_erase_rate_b;               // Moving Window technique erasing plane damping rate
  double mw_erase_rate_dwmax;           // Moving Window technique erasing plane target damping rate.
  int use_Number_of_Atoms;
  double E0, Ewish, Ewf;                // Moving Window technique unperturbed/target energy per atom.
  double Slope;
  double Etot, Elast;                   // current and last step potential energy

  FixMWindowErase(class LAMMPS *, int, char **);
  ~FixMWindowErase() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;
  double compute_scalar() override;
  double compute_vector(int) override;
  double memory_usage() override;
  void end_of_step() override;
  void write_restart(FILE *) override;
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);

 protected:
  double Rx, qq, bb, w;
  int me;
  int nmax;
  int *mark;
  int *list;

  char *id_compute_pe;
  // int iregion;
  int ndeleted;
  int nfreq;
  int nfreq_u_d;

  class Region *region; // replacement for iregion

  class Compute *compute_pe; // compute for potential energy
};

}
#endif
#endif

