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

#include <cmath>

using namespace LAMMPS_NS;

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
  atomKK->sync(execution_space, X_MASK | V_MASK | F_MASK | ANGMOM_MASK | TORQUE_MASK | RMASS_MASK | ELLIPSOID_MASK | MASK_MASK);
  
  auto avec = dynamic_cast<AtomVecEllipsoidKokkos *>(atom->style_match("ellipsoid")); // TODO: check if this is correct, may ask Stan at some point
  bonus = avec->k_bonus.view<DeviceType>();
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

  atomKK->modified(execution_space,  X_MASK | V_MASK | ANGMOM_MASK);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVEAsphereKokkos<DeviceType>::initial_integrate_item(const int i) const
{
  // set timestep here since dt may have changed or come via rRESPA

  const double dtq = 0.5 * dtv;

  if (mask(i) & groupbit) {
    const double dtfm = dtf / rmass(i);
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);
    x(i,0) += dtv * v(i,0);
    x(i,1) += dtv * v(i,1);
    x(i,2) += dtv * v(i,2);

    // update angular momentum by 1/2 step

    angmom(i,0) += dtf * torque(i,0);
    angmom(i,1) += dtf * torque(i,1);
    angmom(i,2) += dtf * torque(i,2);

    // principal moments of inertia
    
    F_FLOAT inertia[3];

    inertia[0] = 0.2*rmass(i) * // 0.2 is the moment of inertia prefactor for ellipsoid
                 (bonus(ellipsoid(i)).shape[1]*bonus(ellipsoid(i)).shape[1] +
                  bonus(ellipsoid(i)).shape[2]*bonus(ellipsoid(i)).shape[2]);
    inertia[1] = 0.2*rmass(i) *
                (bonus(ellipsoid(i)).shape[0]*bonus(ellipsoid(i)).shape[0] +
                 bonus(ellipsoid(i)).shape[2]*bonus(ellipsoid(i)).shape[2]);
    inertia[2] = 0.2*rmass(i) *
                (bonus(ellipsoid(i)).shape[0]*bonus(ellipsoid(i)).shape[0] +
                 bonus(ellipsoid(i)).shape[1]*bonus(ellipsoid(i)).shape[1]);

    // compute omega at 1/2 step from angmom at 1/2 step and current q
    // update quaternion a full step via Richardson iteration
    // returns new normalized quaternion

    // Expanded MathExtra::mq_to_omega function

    F_FLOAT omega[3];

    F_FLOAT q0q0 = bonus(ellipsoid(i)).quat[0] * bonus(ellipsoid(i)).quat[0];
    F_FLOAT q1q1 = bonus(ellipsoid(i)).quat[1] * bonus(ellipsoid(i)).quat[1];
    F_FLOAT q2q2 = bonus(ellipsoid(i)).quat[2] * bonus(ellipsoid(i)).quat[2];
    F_FLOAT q3q3 = bonus(ellipsoid(i)).quat[3] * bonus(ellipsoid(i)).quat[3];

    F_FLOAT q0q1 = bonus(ellipsoid(i)).quat[0] * bonus(ellipsoid(i)).quat[1];
    F_FLOAT q0q2 = bonus(ellipsoid(i)).quat[0] * bonus(ellipsoid(i)).quat[2];
    F_FLOAT q0q3 = bonus(ellipsoid(i)).quat[0] * bonus(ellipsoid(i)).quat[3];
    F_FLOAT q1q2 = bonus(ellipsoid(i)).quat[1] * bonus(ellipsoid(i)).quat[2];
    F_FLOAT q1q3 = bonus(ellipsoid(i)).quat[1] * bonus(ellipsoid(i)).quat[3];
    F_FLOAT q2q3 = bonus(ellipsoid(i)).quat[2] * bonus(ellipsoid(i)).quat[3];

    omega[0] = (1.0 / inertia[0]) * 
                               (angmom(i,0) * (q0q0 + q1q1 - q2q2 - q3q3) +
                                angmom(i,1) * 2.0 * (q1q2 - q0q3) +
                                angmom(i,2) * 2.0 * (q1q3 + q0q2));

    omega[1] = (1.0 / inertia[1]) * 
                               (angmom(i,0) * 2.0 * (q1q2 + q0q3) +
                                angmom(i,1) * (q0q0 - q1q1 + q2q2 - q3q3) +
                                angmom(i,2) * 2.0 * (q2q3 - q0q1));

    omega[2] = (1.0 / inertia[2]) * 
                               (angmom(i,0) * 2.0 * (q1q3 - q0q2) +
                                angmom(i,1) * 2.0 * (q2q3 + q0q1) +
                                angmom(i,2) * (q0q0 - q1q1 - q2q2 + q3q3));

    // Expanded MathExtra::richardson function

    F_FLOAT q0dot = 0.5 * (-omega[0] * bonus(ellipsoid(i)).quat[1] - 
                            omega[1] * bonus(ellipsoid(i)).quat[2] - 
                            omega[2] * bonus(ellipsoid(i)).quat[3]);

    F_FLOAT q1dot = 0.5 * ( omega[0] * bonus(ellipsoid(i)).quat[0] +
                            omega[1] * bonus(ellipsoid(i)).quat[3] -
                            omega[2] * bonus(ellipsoid(i)).quat[2]);

    F_FLOAT q2dot = 0.5 * (-omega[0] * bonus(ellipsoid(i)).quat[3] +
                            omega[1] * bonus(ellipsoid(i)).quat[0] +
                            omega[2] * bonus(ellipsoid(i)).quat[1]);

    F_FLOAT q3dot = 0.5 * ( omega[0] * bonus(ellipsoid(i)).quat[2] -
                            omega[1] * bonus(ellipsoid(i)).quat[1] +
                            omega[2] * bonus(ellipsoid(i)).quat[0]);

    // full update from dq/dt = 1/2 w q

    F_FLOAT qfull[4];
    qfull[0] = bonus(ellipsoid(i)).quat[0] + dtq * q0dot;
    qfull[1] = bonus(ellipsoid(i)).quat[1] + dtq * q1dot;
    qfull[2] = bonus(ellipsoid(i)).quat[2] + dtq * q2dot;
    qfull[3] = bonus(ellipsoid(i)).quat[3] + dtq * q3dot;

    F_FLOAT qnorm = 1.0 / sqrt(qfull[0]*qfull[0] + qfull[1]*qfull[1] 
                             + qfull[2]*qfull[2] + qfull[3]*qfull[3]);

    qfull[0] *= qnorm;
    qfull[1] *= qnorm;
    qfull[2] *= qnorm;
    qfull[3] *= qnorm;   

    // 1st half update from dq/dt = 1/2 w q

    F_FLOAT qhalf[4];
    qhalf[0] = bonus(ellipsoid(i)).quat[0] + 0.5 * dtq * q0dot;
    qhalf[1] = bonus(ellipsoid(i)).quat[1] + 0.5 * dtq * q1dot;
    qhalf[2] = bonus(ellipsoid(i)).quat[2] + 0.5 * dtq * q2dot;
    qhalf[3] = bonus(ellipsoid(i)).quat[3] + 0.5 * dtq * q3dot;

    qnorm = 1.0 / sqrt(qhalf[0]*qhalf[0] + qhalf[1]*qhalf[1] 
                     + qhalf[2]*qhalf[2] + qhalf[3]*qhalf[3]);

    qhalf[0] *= qnorm;
    qhalf[1] *= qnorm;
    qhalf[2] *= qnorm;
    qhalf[3] *= qnorm;

    // re-compute omega at 1/2 step from m at 1/2 step and q at 1/2 step
    // recompute wq
    
    omega[0] = (1.0 / inertia[0]) * 
                               (angmom(i,0) * (qhalf[0]*qhalf[0] + qhalf[1]*qhalf[1] - 
                                               qhalf[2]*qhalf[2] - qhalf[3]*qhalf[3]) +
                                angmom(i,1) * 2.0 * (qhalf[1]*qhalf[2] - qhalf[0]*qhalf[3]) +
                                angmom(i,2) * 2.0 * (qhalf[1]*qhalf[3] + qhalf[0]*qhalf[2]));

    omega[1] = 1.0 / inertia[1] *
                               (angmom(i,0) * 2.0 * (qhalf[1]*qhalf[2] + qhalf[0]*qhalf[3]) +
                                angmom(i,1) * (qhalf[0]*qhalf[0] - qhalf[1]*qhalf[1] + 
                                               qhalf[2]*qhalf[2] - qhalf[3]*qhalf[3]) +
                                angmom(i,2) * 2.0 * (qhalf[2]*qhalf[3] - qhalf[0]*qhalf[1]));

    omega[2] = 1.0 / inertia[2] *
                                (angmom(i,0) * 2.0 * (qhalf[1]*qhalf[3] - qhalf[0]*qhalf[2]) +
                                 angmom(i,1) * 2.0 * (qhalf[2]*qhalf[3] + qhalf[0]*qhalf[1]) +
                                 angmom(i,2) * (qhalf[0]*qhalf[0] - qhalf[1]*qhalf[1] - 
                                                qhalf[2]*qhalf[2] + qhalf[3]*qhalf[3]));
    
    q0dot = 0.5 * (-omega[0] * qhalf[1] - omega[1] * qhalf[2] - omega[2] * qhalf[3]);
    q1dot = 0.5 * ( omega[0] * qhalf[0] + omega[1] * qhalf[3] - omega[2] * qhalf[2]);
    q2dot = 0.5 * (-omega[0] * qhalf[3] + omega[1] * qhalf[0] + omega[2] * qhalf[1]);
    q3dot = 0.5 * ( omega[0] * qhalf[2] - omega[1] * qhalf[1] + omega[2] * qhalf[0]);

    // 2nd half update from dq/dt = 1/2 w q

    qhalf[0] += 0.5 * dtq * q0dot;
    qhalf[1] += 0.5 * dtq * q1dot;
    qhalf[2] += 0.5 * dtq * q2dot;
    qhalf[3] += 0.5 * dtq * q3dot;

    qnorm = 1.0 / sqrt(qhalf[0]*qhalf[0] + qhalf[1]*qhalf[1] 
                     + qhalf[2]*qhalf[2] + qhalf[3]*qhalf[3]);

    qhalf[0] *= qnorm;
    qhalf[1] *= qnorm;
    qhalf[2] *= qnorm;
    qhalf[3] *= qnorm;

    // update quaternion a full step via Richardson iteration
    
    bonus(ellipsoid(i)).quat[0] = 2.0*qhalf[0] - qfull[0];
    bonus(ellipsoid(i)).quat[1] = 2.0*qhalf[1] - qfull[1];
    bonus(ellipsoid(i)).quat[2] = 2.0*qhalf[2] - qfull[2];
    bonus(ellipsoid(i)).quat[3] = 2.0*qhalf[3] - qfull[3];

    qnorm = 1.0 / sqrt(bonus(ellipsoid(i)).quat[0]*bonus(ellipsoid(i)).quat[0] + 
                       bonus(ellipsoid(i)).quat[1]*bonus(ellipsoid(i)).quat[1] + 
                       bonus(ellipsoid(i)).quat[2]*bonus(ellipsoid(i)).quat[2] + 
                       bonus(ellipsoid(i)).quat[3]*bonus(ellipsoid(i)).quat[3]);

    bonus(ellipsoid(i)).quat[0] *= qnorm;
    bonus(ellipsoid(i)).quat[1] *= qnorm;
    bonus(ellipsoid(i)).quat[2] *= qnorm;
    bonus(ellipsoid(i)).quat[3] *= qnorm;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVEAsphereKokkos<DeviceType>::final_integrate()
{
  atomKK->sync(execution_space, V_MASK | F_MASK | ANGMOM_MASK | TORQUE_MASK | RMASS_MASK | MASK_MASK);

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
    const double dtfm = dtf / rmass(i);
    v(i,0) += dtfm * f(i,0);
    v(i,1) += dtfm * f(i,1);
    v(i,2) += dtfm * f(i,2);

    angmom(i,0) += dtf * torque(i,0);
    angmom(i,1) += dtf * torque(i,1);
    angmom(i,2) += dtf * torque(i,2);
  }
}

/* ---------------------------------------------------------------------- */

/*template<class DeviceType>
void FixNVEAsphereKokkos<DeviceType>::fused_integrate(int /*vflag*)/*
{
  atomKK->sync(execution_space, X_MASK | V_MASK | F_MASK | ANGMOM_MASK | TORQUE_MASK | RMASS_MASK | ELLIPSOID_MASK | MASK_MASK);
  
  auto avec = dynamic_cast<AtomVecEllipsoidKokkos *>(atom->style_match("ellipsoid")); // TODO: check if this is correct, may ask Stan at some point
  bonus = avec->k_bonus.view<DeviceType>();
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

  atomKK->modified(execution_space,  X_MASK | V_MASK | ANGMOM_MASK);
}

/* ---------------------------------------------------------------------- */

/*template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVEAsphereKokkos<DeviceType>::fused_integrate_item(const int i) const
{
  // TODO : Figure out if/how to implement fused kernal integration

  if (mask(i) & groupbit) {

  }
}*/

namespace LAMMPS_NS {
template class FixNVEAsphereKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNVEAsphereKokkos<LMPHostType>;
#endif
}
