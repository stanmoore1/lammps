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

#include "bond_oxdna_fene_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"

#include "pair.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
BondOxdnaFENEKokkos<DeviceType>::BondOxdnaFENEKokkos(LAMMPS *lmp) : BondOxdnaFene(lmp)
{
  atomKK = (AtomKokkos *) atom;
  neighborKK = (NeighborKokkos *) neighbor;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  d_flag = typename AT::t_int_scalar("bond:flag");
  h_flag = HAT::t_int_scalar("bond:flag_mirror");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
BondOxdnaFENEKokkos<DeviceType>::~BondOxdnaFENEKokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
  }
}

/* ----------------------------------------------------------------------
    compute vector COM-sugar-phosphate backbone interaction site in oxDNA
------------------------------------------------------------------------- */
template<class DeviceType>
void BondOxdnaFENEKokkos<DeviceType>::compute_interaction_sites(double e1[3], double /*e2*/[3], double /*e3*/[3],
                                              double r[3]) const
{
  constexpr double d_cs = -0.4;

  r[0] = d_cs * e1[0];
  r[1] = d_cs * e1[1];
  r[2] = d_cs * e1[2];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void BondOxdnaFENEKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"bond:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"bond:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  k_k.template sync<DeviceType>();
  k_r0.template sync<DeviceType>();
  k_Delta.template sync<DeviceType>();

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();

  neighborKK->k_bondlist.template sync<DeviceType>();
  bondlist = neighborKK->k_bondlist.view<DeviceType>();
  int nbondlist = neighborKK->nbondlist;
  nlocal = atom->nlocal;
  newton_bond = force->newton_bond;

  // n(x/y/z)_xtrct = extracted local unit vectors in lab frame from oxdna_excv
  int dim;
  nx_xtrct = (double **) force->pair->extract("nx", dim);
  ny_xtrct = (double **) force->pair->extract("ny", dim);
  nz_xtrct = (double **) force->pair->extract("nz", dim);

  Kokkos::deep_copy(d_flag,0);

  copymode = 1;

  // loop over neighbors of my atoms

  EV_FLOAT ev;

  if (evflag) {
    if (newton_bond) {
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagBondOxdnaFENECompute<1,1> >(0,nbondlist),*this,ev);
    } else {
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagBondOxdnaFENECompute<0,1> >(0,nbondlist),*this,ev);
    }
  } else {
    if (newton_bond) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagBondOxdnaFENECompute<1,0> >(0,nbondlist),*this);
    } else {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagBondOxdnaFENECompute<0,0> >(0,nbondlist),*this);
    }
  }

  Kokkos::deep_copy(h_flag,d_flag);

  if (h_flag() == 1)
    error->warning(FLERR,"FENE bond too long");
  else if (h_flag() == 2)
    error->one(FLERR,"Bad FENE bond");

  if (eflag_global) energy += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  copymode = 0;
}

template<class DeviceType>
template<int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void BondOxdnaFENEKokkos<DeviceType>::operator()(TagBondOxdnaFENECompute<NEWTON_BOND,EVFLAG>, \
  const int &in, EV_FLOAT& ev) const {

  // The f array is atomic
  Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,\
    typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > a_f = f;

  const int a = bondlist(in,1);
  const int b = bondlist(in,0);
  const int type = bondlist(in,2);

  Kokkos::View<F_FLOAT[3]> delf("delf");
  Kokkos::View<F_FLOAT[3]> delta("delta");
  Kokkos::View<F_FLOAT[3]> deltb("deltb");
  Kokkos::View<F_FLOAT[3]> delr("delr");

  // vectors COM-backbone site in lab frame
  Kokkos::View<F_FLOAT[3]> ra_cs("ra_cs");
  Kokkos::View<F_FLOAT[3]> rb_cs("rb_cs");
  // Cartesian unit vectors in lab frame
  Kokkos::View<F_FLOAT[3]> ax("ax");
  Kokkos::View<F_FLOAT[3]> ay("ay");
  Kokkos::View<F_FLOAT[3]> az("az");
  Kokkos::View<F_FLOAT[3]> bx("bx");
  Kokkos::View<F_FLOAT[3]> by("by");
  Kokkos::View<F_FLOAT[3]> bz("bz");

  ax(0) = nx_xtrct[a][0];
  ax(1) = nx_xtrct[a][1];
  ax(2) = nx_xtrct[a][2];
  ay(0) = ny_xtrct[a][0];
  ay(1) = ny_xtrct[a][1];
  ay(2) = ny_xtrct[a][2];
  az(0) = nz_xtrct[a][0];
  az(1) = nz_xtrct[a][1];
  az(2) = nz_xtrct[a][2];
  bx(0) = nx_xtrct[b][0];
  bx(1) = nx_xtrct[b][1];
  bx(2) = nx_xtrct[b][2];
  by(0) = ny_xtrct[b][0];
  by(1) = ny_xtrct[b][1];
  by(2) = ny_xtrct[b][2];
  bz(0) = nz_xtrct[b][0];
  bz(1) = nz_xtrct[b][1];
  bz(2) = nz_xtrct[b][2];

  // vector COM-backbone site a and b
  compute_interaction_sites(ax.data(),ay.data(),ax.data(),ra_cs.data());
  compute_interaction_sites(bx.data(),by.data(),bz.data(),rb_cs.data());
  
  // vector backbone site b to a
  delr(0) = x(a,0) + ra_cs(0) - x(b,0) - rb_cs(0);
  delr(1) = x(a,1) + ra_cs(1) - x(b,1) - rb_cs(1);
  delr(2) = x(a,2) + ra_cs(2) - x(b,2) - rb_cs(2);
  const F_FLOAT rsq = delr(0)*delr(0) + delr(1)*delr(1) + delr(2)*delr(2);
  const F_FLOAT r = sqrt(rsq); //sqrt ok for Kokkos??

  //const F_FLOAT r0 = d_r0[type];
  //const F_FLOAT k = d_k[type];
  //const F_FLOAT Delta = d_Delta[type];

  const F_FLOAT rr0 = r - d_r0[type];
  const F_FLOAT rr0sq = rr0 * rr0;
  const F_FLOAT Deltasq = d_Delta[type]*d_Delta[type];
  F_FLOAT rlogarg = 1.0 - rr0sq/Deltasq;

  // if r -> r0, then rlogarg < 0.0 which is an error
  // issue a warning and reset rlogarg = Delta
  // if r > 2*r0 something serious is wrong, abort

  if (rlogarg < 0.1) {
    if (rlogarg <= -3.0)
      d_flag() = 2;
    else
      d_flag() = 1;
    rlogarg = 0.1;
  }

  F_FLOAT fbond = -d_k[type] * rr0 / rlogarg / Deltasq / r;
  delf(0) = delr(0) * fbond;
  delf(1) = delr(1) * fbond;
  delf(2) = delr(2) * fbond;

  /*// force from LJ term

  F_FLOAT sr6 = 0.0;
  F_FLOAT sigma2 = 1;
  if (rsq < sigma2) {
    const F_FLOAT sr2 = sigma2/rsq;
    sr6 = sr2*sr2*sr2;
    fbond += 48.0*Delta*sr6*(sr6-0.5)/rsq;
  }*/

  // energy

  F_FLOAT ebond = 0.0;
  if (eflag) { ebond = -0.5*d_k[type]*log(rlogarg);}

  // apply force to each of 2 atoms

  if (NEWTON_BOND || a < nlocal) {
    a_f(a,0) += delf(0);
    a_f(a,1) += delf(1);
    a_f(a,2) += delf(3);
    delta(0) = ra_cs(1)*delf(2) - ra_cs(2)*delf(1);
    delta(1) = ra_cs(2)*delf(0) - ra_cs(0)*delf(2);
    delta(2) = ra_cs(0)*delf(1) - ra_cs(1)*delf(0);
    torque(a,0) += delta(0);
    torque(a,1) += delta(1);
    torque(a,2) += delta(2);
  }

  if (NEWTON_BOND || b < nlocal) {
    a_f(b,0) -= delf(0);
    a_f(b,1) -= delf(1);
    a_f(b,2) -= delf(2);
    deltb(0) = rb_cs(1)*delf(2) - rb_cs(2)*delf(1);
    deltb(1) = rb_cs(2)*delf(0) - rb_cs(0)*delf(2);
    deltb(2) = rb_cs(0)*delf(1) - rb_cs(1)*delf(0);
    torque(b,0) -= deltb(0);
    torque(b,1) -= deltb(1);
    torque(b,2) -= deltb(2);
  }

  if (EVFLAG) { ev_tally_xyz(ev, a, b, nlocal, NEWTON_BOND, ebond, delf(0), delf(1), delf(2), \
    x(a,0)-x(b,0), x(a,1)-x(b,1), x(a,2)-x(b,2)); }
}

template<class DeviceType>
template<int NEWTON_BOND, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void BondOxdnaFENEKokkos<DeviceType>::operator()(TagBondOxdnaFENECompute<NEWTON_BOND,EVFLAG>, const int &n) const {
  EV_FLOAT ev;
  this->template operator()<NEWTON_BOND,EVFLAG>(TagBondOxdnaFENECompute<NEWTON_BOND,EVFLAG>(), n, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void BondOxdnaFENEKokkos<DeviceType>::allocate()
{
  BondOxdnaFene::allocate();

  int n = atom->nbondtypes;
  k_k = DAT::tdual_ffloat_1d("BondFene::k",n+1);
  k_r0 = DAT::tdual_ffloat_1d("BondFene::r0",n+1);
  k_Delta = DAT::tdual_ffloat_1d("BondOxdnaFene::Delta",n+1);

  d_k = k_k.template view<DeviceType>();
  d_r0 = k_r0.template view<DeviceType>();
  d_Delta = k_Delta.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

template<class DeviceType>
void BondOxdnaFENEKokkos<DeviceType>::coeff(int narg, char **arg)
{
  BondOxdnaFene::coeff(narg, arg);

  int n = atom->nbondtypes;
  for (int i = 1; i <= n; i++) {
    k_k.h_view[i] = k[i];
    k_r0.h_view[i] = r0[i];
    k_Delta.h_view[i] = Delta[i];
  }

  k_k.template modify<LMPHostType>();
  k_r0.template modify<LMPHostType>();
  k_Delta.template modify<LMPHostType>();
}


/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

template<class DeviceType>
void BondOxdnaFENEKokkos<DeviceType>::read_restart(FILE *fp)
{
  BondOxdnaFene::read_restart(fp);

  int n = atom->nbondtypes;
  for (int i = 1; i <= n; i++) {
    k_k.h_view[i] = k[i];
    k_r0.h_view[i] = r0[i];
    k_Delta.h_view[i] = Delta[i];
  }

  k_k.template modify<LMPHostType>();
  k_r0.template modify<LMPHostType>();
  k_Delta.template modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void BondOxdnaFENEKokkos<DeviceType>::ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,\
      const int &nlocal, const int &newton_bond,\
      const F_FLOAT &ebond, const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,\
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
{
  E_FLOAT ebondhalf;
  F_FLOAT v[6];

  // The eatom and vatom arrays are atomic
  Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,\
    typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > \
    v_eatom = k_eatom.view<DeviceType>();
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,\
    typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > \
    v_vatom = k_vatom.view<DeviceType>();

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) ev.evdwl += ebond;
      else {
        ebondhalf = 0.5*ebond;
        if (i < nlocal) ev.evdwl += ebondhalf;
        if (j < nlocal) ev.evdwl += ebondhalf;
      }
    }
    if (eflag_atom) {
      ebondhalf = 0.5*ebond;
      if (newton_bond || i < nlocal) v_eatom[i] += ebondhalf;
      if (newton_bond || j < nlocal) v_eatom[j] += ebondhalf;
    }
  }

  if (vflag_either) {
    v[0] = delx * fx;
    v[1] = dely * fy;
    v[2] = delz * fz;
    v[3] = delx * fy;
    v[4] = delx * fz;
    v[5] = dely * fz;

    if (vflag_global) {
      if (newton_bond) {
        ev.v[0] += v[0];
        ev.v[1] += v[1];
        ev.v[2] += v[2];
        ev.v[3] += v[3];
        ev.v[4] += v[4];
        ev.v[5] += v[5];
      } else {
        if (i < nlocal) {
          ev.v[0] += 0.5*v[0];
          ev.v[1] += 0.5*v[1];
          ev.v[2] += 0.5*v[2];
          ev.v[3] += 0.5*v[3];
          ev.v[4] += 0.5*v[4];
          ev.v[5] += 0.5*v[5];
        }
        if (j < nlocal) {
          ev.v[0] += 0.5*v[0];
          ev.v[1] += 0.5*v[1];
          ev.v[2] += 0.5*v[2];
          ev.v[3] += 0.5*v[3];
          ev.v[4] += 0.5*v[4];
          ev.v[5] += 0.5*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i < nlocal) {
        v_vatom(i,0) += 0.5*v[0];
        v_vatom(i,1) += 0.5*v[1];
        v_vatom(i,2) += 0.5*v[2];
        v_vatom(i,3) += 0.5*v[3];
        v_vatom(i,4) += 0.5*v[4];
        v_vatom(i,5) += 0.5*v[5];
      }
      if (newton_bond || j < nlocal) {
        v_vatom(j,0) += 0.5*v[0];
        v_vatom(j,1) += 0.5*v[1];
        v_vatom(j,2) += 0.5*v[2];
        v_vatom(j,3) += 0.5*v[3];
        v_vatom(j,4) += 0.5*v[4];
        v_vatom(j,5) += 0.5*v[5];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class BondOxdnaFENEKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class BondOxdnaFENEKokkos<LMPHostType>;
#endif
}
