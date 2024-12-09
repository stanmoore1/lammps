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

#ifndef MF_OXDNA_KOKKOS_H
#define MF_OXDNA_KOKKOS_H

#include "kokkos_base.h"

namespace LAMMPS_NS {

template<class DeviceType>
class mfOxdnaKokkos : public KokkosBase {
 public:
  mfOxdnaKokkos(class LAMMPS *) {};
  ~mfOxdnaKokkos() {};
/* ----------------------------------------------------------------------
   f1 modulation factor
   ---------------------------------------------------------------------- */
  KOKKOS_INLINE_FUNCTION
  void oxDNA_F1_KK(F_FLOAT r, F_FLOAT eps, F_FLOAT a, F_FLOAT cut_0, F_FLOAT cut_lc,
                   F_FLOAT cut_hc, F_FLOAT cut_lo, F_FLOAT cut_hi, F_FLOAT b_lo, 
                   F_FLOAT b_hi, F_FLOAT shift, F_FLOAT& f1) const 
  {
    if (r > cut_hc) {
     f1 = 0.0;
    } else if (r > cut_hi) {
     f1 = eps * b_hi * (r - cut_hc) * (r - cut_hc);
    } else if (r > cut_lo) {
     F_FLOAT tmp = 1 - exp(-(r - cut_0) * a);
     f1 = eps * tmp * tmp - shift;
    } else if (r > cut_lc) {
     f1 = eps * b_lo * (r - cut_lc) * (r - cut_lc);
    } else {
     f1 = 0.0;
    }
  };

};

}    // namespace LAMMPS_NS

#endif

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class mfOxdnaKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class mfOxdnaKokkos<LMPHostType>;
#endif
}