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

#include "pair_oxdna2_excv.h"
#include "constants_oxdna.h"
#include "nucleotide_oxdna.h"

using namespace LAMMPS_NS;

/* -----------------------------------------------------------------------
    compute vector COM-sugar-phosphate backbone interaction site in oxDNA2
-------------------------------------------------------------------------- */
inline void PairOxdna2Excv::compute_backbone_site(double e1[3],
  double e2[3], double /*e3*/[3], double rbk[3]) const
{
  NucleotideOxdna2 oxdna2;
  oxdna2.backbone_site(e1, e2, NULL, rbk);
}
