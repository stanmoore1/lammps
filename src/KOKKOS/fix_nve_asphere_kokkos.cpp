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

#include "fix_nve_asphere_kokkos.h"
#include "atom_masks.h"
#include "atom_kokkos.h"
#include "math_extra_kokkos.h"

using namespace LAMMPS_NS;

static constexpr double INERTIA = 0.2;       // moment of inertia prefactor for ellipsoid

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixNVEAsphereKokkos<DeviceType>::FixNVEAsphereKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNVEAsphere(lmp, narg, arg)
{
  kokkosable = 1;
  fuse_integrate_flag = 1;
  atomKK = (AtomKokkos *)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  avecEllipKK = dynamic_cast<AtomVecEllipsoidKokkos *>(atom->style_match("ellipsoid"));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVEAsphereKokkos<DeviceType>::cleanup_copy()
{
  id = style = nullptr;
  vatom = nullptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVEAsphereKokkos<DeviceType>::init()
{
  FixNVEAsphere::init();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVEAsphereKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  atomKK->sync(execution_space, X_MASK | V_MASK | F_MASK | ANGMOM_MASK | TORQUE_MASK |
                                RMASS_MASK | ELLIPSOID_MASK | BONUS_MASK | MASK_MASK);

  bonus = avecEllipKK->k_bonus.view<DeviceType>();
  ellipsoid = atomKK->k_ellipsoid.view<DeviceType>();

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  angmom = atomKK->k_angmom.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  FixNVEAsphereKokkosInitialIntegrateFunctor<DeviceType> f(this);
  Kokkos::parallel_for(nlocal,f);

  atomKK->modified(execution_space, X_MASK | V_MASK | ANGMOM_MASK |
                                    ELLIPSOID_MASK | BONUS_MASK);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVEAsphereKokkos<DeviceType>::initial_integrate_item(const int i) const
{
  // set timestep here since dt may have changed or come via rRESPA
  KK_FLOAT angm[3], inertia[3], omega[3];

  if (mask(i) & groupbit) {
    const KK_FLOAT rm = rmass(i);
    const KK_FLOAT dtfm = dtf / rm;
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    x(i,0) += dtv * v(i,0);
    x(i,1) += dtv * v(i,1);
    x(i,2) += dtv * v(i,2);

    // update angular momentum by 1/2 step into a local array
    angm[0] = Kokkos::fma(dtf, torque(i,0), angmom(i,0));
    angm[1] = Kokkos::fma(dtf, torque(i,1), angmom(i,1));
    angm[2] = Kokkos::fma(dtf, torque(i,2), angmom(i,2));

    // principal moments of inertia
    double *shape = bonus(ellipsoid(i)).shape;
    KK_FLOAT s0 = (KK_FLOAT) shape[0];
    KK_FLOAT s1 = (KK_FLOAT) shape[1];
    KK_FLOAT s2 = (KK_FLOAT) shape[2];
    inertia[0] = INERTIA*rm * (s1*s1 + s2*s2);
    inertia[1] = INERTIA*rm * (s0*s0 + s2*s2);
    inertia[2] = INERTIA*rm * (s0*s0 + s1*s1);

    // compute omega at 1/2 step from angmom at 1/2 step and current q
    // update quaternion a full step via Richardson iteration
    // returns new normalized quaternion
    double *quat = bonus(ellipsoid(i)).quat;
    KK_FLOAT qlocal[4];
    qlocal[0] = (KK_FLOAT) quat[0];
    qlocal[1] = (KK_FLOAT) quat[1];
    qlocal[2] = (KK_FLOAT) quat[2];
    qlocal[3] = (KK_FLOAT) quat[3];
    MathExtraKokkos::mq_to_omega(angm, qlocal, inertia, omega);
    const KK_FLOAT dtq = 0.5 * dtv;
    MathExtraKokkos::richardson(qlocal, angm, omega, inertia, dtq);
    // write back updated quaternion into the double bonus storage
    quat[0] = (double) qlocal[0];
    quat[1] = (double) qlocal[1];
    quat[2] = (double) qlocal[2];
    quat[3] = (double) qlocal[3];

    // write back updated angular momentum
    angmom(i,0) = angm[0];
    angmom(i,1) = angm[1];
    angmom(i,2) = angm[2];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVEAsphereKokkos<DeviceType>::final_integrate()
{
  atomKK->sync(execution_space, V_MASK | F_MASK | ANGMOM_MASK | TORQUE_MASK |
                                RMASS_MASK | MASK_MASK);

  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  angmom = atomKK->k_angmom.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  FixNVEAsphereKokkosFinalIntegrateFunctor<DeviceType> f(this);
  Kokkos::parallel_for(nlocal,f);

  atomKK->modified(execution_space, V_MASK | ANGMOM_MASK);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVEAsphereKokkos<DeviceType>::final_integrate_item(const int i) const
{
  if (mask(i) & groupbit) {
    const KK_FLOAT dtfm = dtf / rmass(i);
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);

    angmom(i,0) += dtf * torque(i,0);
    angmom(i,1) += dtf * torque(i,1);
    angmom(i,2) += dtf * torque(i,2);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVEAsphereKokkos<DeviceType>::fused_integrate(int /*vflag*/)
{
  atomKK->sync(execution_space, X_MASK | V_MASK | F_MASK | ANGMOM_MASK | TORQUE_MASK |
                                RMASS_MASK | ELLIPSOID_MASK | BONUS_MASK | MASK_MASK);

  bonus = avecEllipKK->k_bonus.view<DeviceType>();
  ellipsoid = atomKK->k_ellipsoid.view<DeviceType>();

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  angmom = atomKK->k_angmom.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  FixNVEAsphereKokkosFusedIntegrateFunctor<DeviceType> f(this);
  Kokkos::parallel_for(nlocal,f);

  atomKK->modified(execution_space, X_MASK | V_MASK | ANGMOM_MASK |
                                    ELLIPSOID_MASK | BONUS_MASK);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVEAsphereKokkos<DeviceType>::fused_integrate_item(const int i) const
{
  const KK_FLOAT dtq = 0.5 * dtv;
  KK_FLOAT angm[3];

  if (mask(i) & groupbit) {
    const KK_FLOAT rm = rmass(i);
    const KK_FLOAT dtfm = 2.0 * dtf / rm;
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    angmom(i,0) += dtf * torque(i,0);
    angmom(i,1) += dtf * torque(i,1);
    angmom(i,2) += dtf * torque(i,2);
    x(i,0) += dtv * v(i,0);
    x(i,1) += dtv * v(i,1);
    x(i,2) += dtv * v(i,2);

    // update angular momentum by 1/2 step into a local array
    angm[0] = Kokkos::fma(dtf, torque(i,0), angmom(i,0));
    angm[1] = Kokkos::fma(dtf, torque(i,1), angmom(i,1));
    angm[2] = Kokkos::fma(dtf, torque(i,2), angmom(i,2));

    // principal moments of inertia
    double *shape = bonus(ellipsoid(i)).shape;
    double *quat = bonus(ellipsoid(i)).quat;
    KK_FLOAT s0 = (KK_FLOAT) shape[0];
    KK_FLOAT s1 = (KK_FLOAT) shape[1];
    KK_FLOAT s2 = (KK_FLOAT) shape[2];
    KK_FLOAT inertia[3], omega[3];
    inertia[0] = INERTIA*rm * (s1*s1 + s2*s2);
    inertia[1] = INERTIA*rm * (s0*s0 + s2*s2);
    inertia[2] = INERTIA*rm * (s0*s0 + s1*s1);

    // compute omega at 1/2 step from angmom at 1/2 step and current q
    // update quaternion a full step via Richardson iteration
    // returns new normalized quaternion
    KK_FLOAT qlocal[4];
    qlocal[0] = (KK_FLOAT) quat[0];
    qlocal[1] = (KK_FLOAT) quat[1];
    qlocal[2] = (KK_FLOAT) quat[2];
    qlocal[3] = (KK_FLOAT) quat[3];
    MathExtraKokkos::mq_to_omega(angm, qlocal, inertia, omega);
    MathExtraKokkos::richardson(qlocal, angm, omega, inertia, dtq);
    // write back updated quaternion into the double bonus storage
    quat[0] = (double) qlocal[0];
    quat[1] = (double) qlocal[1];
    quat[2] = (double) qlocal[2];
    quat[3] = (double) qlocal[3];

    // write back updated angular momentum
    angmom(i,0) = angm[0];
    angmom(i,1) = angm[1];
    angmom(i,2) = angm[2];
  }
}

namespace LAMMPS_NS {
template class FixNVEAsphereKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNVEAsphereKokkos<LMPHostType>;
#endif
}
