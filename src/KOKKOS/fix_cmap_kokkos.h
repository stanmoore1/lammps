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

#ifdef FIX_CLASS
// clang-format off
FixStyle(cmap/kk,FixCMAPKokkos<LMPDeviceType>);
FixStyle(cmap/kk/device,FixCMAPKokkos<LMPDeviceType>);
FixStyle(cmap/kk/host,FixCMAPKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_CMAP_KOKKOS_H
#define LMP_FIX_CMAP_KOKKOS_H

#include "fix_cmap.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class FixCMAPKokkos : public FixCMAP {
  typedef ArrayTypes<DeviceType> AT;

  public:
    FixCMAPKokkos(class LAMMPS *, int, char **);
    ~FixCMAPKokkos() override;

    void init() override;
    void pre_neighbor() override;
    void post_force(int) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int) const;

    void grow_arrays(int) override;
    void copy_arrays(int, int, int) override;
    void set_arrays(int) override;
    int pack_exchange(int, double *) override;
    int unpack_exchange(int, double *) override;

  protected:
    typename AT::t_x_array d_x;
    typename AT::t_f_array d_f;
    //typename AT::t_int_1d d_type, d_mask;

    DAT::tdual_int_1d k_sametag;
    typename AT::t_int_1d d_sametag;
    int map_style;
    DAT::tdual_int_1d k_map_array;
    dual_hash_type k_map_hash;

    DAT::tdual_int_1d k_num_crossterm;
    typename AT::t_int_1d d_num_crossterm;

    DAT::tdual_int_2d k_crosstermlist, k_crossterm_type;
    typename AT::t_int_2d d_crosstermlist, d_crossterm_type;

    DAT::tdual_tagint_2d k_crossterm_atom1, k_crossterm_atom2, k_crossterm_atom3;
    DAT::tdual_tagint_2d k_crossterm_atom4, k_crossterm_atom5;
    typename AT::t_tagint_2d d_crossterm_atom1, d_crossterm_atom2, d_crossterm_atom3;
    typename AT::t_tagint_2d d_crossterm_atom4, d_crossterm_atom5;

    DAT::tdual_float_1d k_g_axis;
    typename AT::t_float_1d d_g_axis;

    DAT::tdual_float_3d k_cmapgrid, k_d1cmapgrid, k_d2cmapgrid, k_d12cmapgrid;
    typename AT::t_float_3d d_cmapgrid, d_d1cmapgrid, d_d2cmapgrid, d_d12cmapgrid;

    // calculate dihedral angles
    KOKKOS_INLINE_FUNCTION
    double dihedral_angle_atan2(double, double, double, double, double, double, double, double,
      double, double) const;

    // perform bicubic interpolation at point of interest
    KOKKOS_INLINE_FUNCTION
    void bc_interpol(double, double, int, int, double *, double *, double *, double *,
      double &, double &, double &) const;

    // copied from Domain
    KOKKOS_INLINE_FUNCTION
    int closest_image(const int, int) const;

};

} // namespace LAMMPS_NS

#endif // LMP_FIX_CMAP_KOKKOS_H
#endif // FIX_CLASS