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

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(ewald/dipole/spin,EwaldDipoleSpin);
// clang-format on
#else

#ifndef LMP_EWALD_DIPOLE_SPIN_H
#define LMP_EWALD_DIPOLE_SPIN_H

#include "ewald_dipole.h"

namespace LAMMPS_NS {

class EwaldDipoleSpin : public EwaldDipole {
 public:
  EwaldDipoleSpin(class LAMMPS *);
  ~EwaldDipoleSpin() override = default;
  void init() override;
  void compute(int, int) override;

 protected:
  double mub;           // Bohr magneton
  double mu_0;          // vacuum permeability
  double mub2mu0;       // mub^2 * mu_0/(4pi), prefactor for mech force
  double mub2mu0hbinv;  // mub2mu0 / hbar, prefactor for mag force
  double hbar;          // reduced Planck's constant

  void musum_musq() override;
  void eik_dot_r() override;
  void slabcorr() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
