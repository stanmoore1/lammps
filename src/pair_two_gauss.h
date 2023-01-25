/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(twogauss,PairTwoGauss)

#else

#ifndef LMP_PAIR_TWO_GAUSS_H
#define LMP_PAIR_TWO_GAUSS_H

#include "pair.h"

namespace LAMMPS_NS {

class PairTwoGauss : public Pair {
 public:
  PairTwoGauss(class LAMMPS *);
  ~PairTwoGauss();

  virtual void compute(int, int);

  virtual double single(int, int, int, int, double, double, double, double &);

  virtual void settings(int, char **);
  virtual void coeff(int, char **);

  virtual double init_one(int, int);

  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  virtual void write_data(FILE *fp);
  virtual void write_data_all(FILE *fp);

  virtual double memory_usage();

 protected:
  double cut_global;
  double **cut;
  double **hgauss,**sigmah,**rmh;
  double **hgauss2,**sigmah2,**rmh2; //second set of attractive gauss (use hybrid/overlay for repulsion)
  double **pgauss,**offset;
  double **pgauss2;

  virtual void allocate();
};

}

#endif
#endif
