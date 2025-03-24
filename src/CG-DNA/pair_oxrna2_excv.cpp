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
/* ----------------------------------------------------------------------
   Contributing author: Oliver Henrich (University of Strathclyde, Glasgow)
------------------------------------------------------------------------- */

#include "pair_oxrna2_excv.h"
#include "constants_oxdna.h"

using namespace LAMMPS_NS;

/* -----------------------------------------------------------------------
    compute vector COM-sugar-phosphate backbone interaction site in oxRNA2
-------------------------------------------------------------------------- */
void PairOxrna2Excv::compute_backbone_site(double e1[3], double /*e2*/[3],
  double e3[3], double rs[3]) const
{
  double d_cs_x = ConstantsOxdna::get_d_cs();
  double d_cs_z = ConstantsOxdna::get_d_cs_z();

  rs[0] = d_cs_x * e1[0] + d_cs_z * e3[0];
  rs[1] = d_cs_x * e1[1] + d_cs_z * e3[1];
  rs[2] = d_cs_x * e1[2] + d_cs_z * e3[2];

}
