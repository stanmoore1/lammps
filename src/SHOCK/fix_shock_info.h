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

#ifdef FIX_CLASS

FixStyle(shock/info,FixShockInfo)

#else

#ifndef FIX_SHOCK_INFO_H
#define FIX_SHOCK_INFO_H

#include "stdio.h"
#include "fix.h"

namespace LAMMPS_NS {

class FixShockInfo : public Fix {
 public:
  FixShockInfo(class LAMMPS *, int, char **);
  ~FixShockInfo();
  int setmask();
  void init();
  void end_of_step();

 private:
  int me;
  int nfreq, nrepeat,nmin;
  int dim,originflag,scaleflag;
  double origin,delta;
  char *einfo_fileprefix, *stress_fileprefix;
  char *id_compute_pe, *id_compute_stress;
  FILE *fp;

  int nlayers,nvalues,nsum,maxlayer;
  int stress_size_peratom;
  int cpnts_noT, cpnts_all;
  double xscale,yscale,zscale;
  double layer_volume;
  double *coord;
  double *count_one,*count_many,*count_total;
  double **values_one,**values_many,**values_total;
  double offset,invdelta; 
  double *variable_bin;

  class Compute *compute_pe, *compute_stress;
  class Compute *precompute_pe, *precompute_stress;
};

}

#endif
#endif
