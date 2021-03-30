/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   
    molmass/chunk computes the mass of each chunk

------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(mass/chunk,ComputeMASSChunk)

#else

#ifndef LMP_COMPUTE_MASS_CHUNK_H
#define LMP_COMPUTE_MASS_CHUNK_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeMASSChunk : public Compute {
 public:
  char *idchunk;              // fields accessed by other classes

  ComputeMASSChunk(class LAMMPS *, int, char **);
  ~ComputeMASSChunk();
  void init();
  void setup();
  void compute_vector();

  void lock_enable();
  void lock_disable();
  int lock_length();
  void lock(class Fix *, bigint, bigint);
  void unlock(class Fix *);

  double memory_usage();

 private:
  int nchunk,maxchunk;
  class ComputeChunkAtom *cchunk;

  double *massproc;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Chunk/atom compute does not exist for compute mass/chunk

Self-explanatory.

E: Compute mass/chunk does not use chunk/atom compute

The style of the specified compute is not chunk/atom.

*/
