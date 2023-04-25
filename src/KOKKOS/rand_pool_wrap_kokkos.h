// clang-format off
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

#ifndef RAND_POOL_WRAP_H
#define RAND_POOL_WRAP_H

#include "pointers.h"
#include "kokkos_type.h"
#include "random_mars.h"
#include "random_park.h"
#include "error.h"

namespace LAMMPS_NS {

struct RandWrap {
  class RanMars* rng_mars;
  class RanPark* rng_park;

  KOKKOS_INLINE_FUNCTION
  RandWrap() {
    rng_mars = nullptr;
    rng_park = nullptr;
  }

  KOKKOS_INLINE_FUNCTION
  double drand() {
   if (rng_mars) return rng_mars->uniform();
   else return rng_park->uniform();
  }

  KOKKOS_INLINE_FUNCTION
  double normal() {
   if (rng_mars) return rng_mars->gaussian();
   else return rng_park->gaussian();
  }
};

class RandPoolWrap : protected Pointers {
 public:
  RandPoolWrap(int, class LAMMPS *);
  ~RandPoolWrap() override;
  void destroy();
  void init(RanMars*);
  void init(RanPark*);

  RandWrap get_state() const
  {
#ifdef LMP_KOKKOS_GPU
    error->all(FLERR,"Cannot use Marsaglia or Park RNG with GPUs");
#endif

    RandWrap rand_wrap;
    if (random_mars) rand_wrap.rng_mars = random_mars;
    else rand_wrap.rng_park = random_park;

    return rand_wrap;
  }

  void free_state(RandWrap) const {}

 private:
  class RanMars* random_mars;
  class RanPark* random_park;  
};

}

#endif

