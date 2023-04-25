// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "rand_pool_wrap_kokkos.h"
#include "comm.h"
#include "lammps.h"
#include "kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

RandPoolWrap::RandPoolWrap(int, LAMMPS *lmp) : Pointers(lmp)
{
  random_mars = nullptr;
  random_park = nullptr;
}

/* ---------------------------------------------------------------------- */

RandPoolWrap::~RandPoolWrap()
{
}

void RandPoolWrap::destroy()
{
  if (random_mars) {
    delete random_mars;
    random_mars = nullptr;
  }

  if (random_park) {
    delete random_park;
    random_park = nullptr;
  }
}

void RandPoolWrap::init(RanMars* random)
{
  destroy();
  random_mars = random;
}

void RandPoolWrap::init(RanPark* random)
{
  destroy();                      
  random_park = random;
}
