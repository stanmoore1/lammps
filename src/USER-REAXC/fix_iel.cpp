/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_iel.h"
#include <cstring>
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include <random>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixIEL::FixIEL(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7)
    error->all(FLERR,"Illegal fix iel command");

  atom->XLMDFlag = force->inumeric(FLERR, arg[3]); // Songchen: 0 (Exact) 1 (XLMD) 2 (Ber) 3 (NH) 4 (Lang)
  atom->mLatent = force->numeric(FLERR, arg[4]);  // Songchen: latent mass
  atom->tauLatent = force->numeric(FLERR, arg[5]);  // Songchen: latent thermostat strength
  atom->tLatent = force->numeric(FLERR, arg[6]);  // Songchen: latent temperature

}

/* ---------------------------------------------------------------------- */

int FixIEL::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixIEL::init()
{
  dtv = update->dt;
  dth = update->dt/2;
  dtf = 0.5 * update->dt * force->ftm2v;

  // init XLMD

  double *qLatent;
  double *pLatent;
  double *fLatent;
  get_names("qLatent", qLatent);
  get_names("pLatent", pLatent);
  get_names("fLatent", fLatent);

  for (int i = 0; i < atom->nlocal; i++){
    if (atom->mask[i] & groupbit) {
      qLatent[i] = atom->q[i];
      pLatent[i] = 0;
      fLatent[i] = 0;
    }
  }
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixIEL::initial_integrate(int /*vflag*/)
{
  double dtfm;

  // update v and x of atoms in group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // obtain Latents

  double *qLatent;
  double *pLatent;
  double *fLatent;
  get_names("qLatent", qLatent);
  get_names("pLatent", pLatent);
  get_names("fLatent", fLatent);

  // evolve B(t/2) A(t/2)

  if (atom->XLMDFlag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        pLatent[i] += dth * fLatent[i];
        qLatent[i] += dth * pLatent[i] / atom->mLatent;
      }
    }
  }

  // evolve O(t) for BAOAB Scheme

  //if (update->ntimestep % atom->respaLatent == 0) {
    //double dto = dtv * atom->respaLatent;
    double dto = dtv; /////////////////////////////////////////////
    if (atom->XLMDFlag == 2) {
      if (update->ntimestep > 1000) {
        Berendersen(dto);
      }
    } else if (atom->XLMDFlag == 3) {
      Langevin(dto);
    }
  //}

  if (atom->XLMDFlag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        qLatent[i] += dth * pLatent[i] / atom->mLatent;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixIEL::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Songchen: Obtain Latents

  double *qLatent;
  double *pLatent;
  double *fLatent;
  get_names("qLatent", qLatent);
  get_names("pLatent", pLatent);
  get_names("fLatent", fLatent);

  // Songchen: Evolve B(t/2)


  if (atom->XLMDFlag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        pLatent[i] += dth * fLatent[i];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixIEL::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

void FixIEL::get_names(char *c,double *&ptr)
{
  int index,flag;
  index = atom->find_custom(c,flag);
  
  if (index!=-1)
    ptr = atom->dvector[index];
  else
    error->all(FLERR,"fix iEL-Scf requires fix property/atom ?? command");
}

/* ---------------------------------------------------------------------- */

double FixIEL::kinetic_latent()
{
  double *pLatent;
  get_names("pLatent", pLatent);

  int nlocal = atom->nlocal;
  double sum_p2 = 0.0;
  double sum_p2_mpi = 0.0;
  double avg_p2 = 0.0;

  // find the total kinetic energy and auxiliary temperatures
  for (int i = 0; i < nlocal; i++) {
    sum_p2 = sum_p2 + pLatent[i] * pLatent[i] / atom->mLatent / 2.0;
  }
  MPI_Allreduce(&sum_p2, &sum_p2_mpi, 1, MPI_DOUBLE, MPI_SUM, world);
  avg_p2 = sum_p2_mpi / atom->natoms;
  return avg_p2;
}

/* ---------------------------------------------------------------------- */

void FixIEL::Berendersen(const double dt)
{
  
  int nlocal = atom->nlocal;
  double *pLatent;
  get_names("pLatent", pLatent);   

  double avg_p2 = kinetic_latent();
  double target_p2 = atom->tLatent / 2.0;
  double scale = sqrt(1.0 + (dt/(atom->tauLatent))*(target_p2/avg_p2-1.0));

  for (int i = 0; i < nlocal; i++) {
    pLatent[i] = pLatent[i] * scale;
  }
}

/* ---------------------------------------------------------------------- */

void FixIEL::Langevin(const double dt) {
  int nlocal = atom->nlocal;
  double **v = atom->v;
  int *mask = atom->mask;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  double *pLatent;
  get_names("pLatent",pLatent); 

  double pLatentAvg = sqrt(atom->mLatent * atom->tLatent);
  double dissipationLatent = exp(-dt/atom->tauLatent);
  double fluctuationLatent = sqrt(1 - dissipationLatent * dissipationLatent);

  // change to LAMMPS RNG
  std::default_random_engine gen;
  std::normal_distribution<double> dis(0,1);

  // Log the prior kinetic energy.

  double kineticBefore = kinetic_latent();

  // Perform the Langevin integration.

  for (int i = 0; i < nlocal; i++) {
    pLatent[i] = pLatent[i] * dissipationLatent + pLatentAvg * fluctuationLatent * dis(gen);
  }

  double pDev = parallel_vector_acc(pLatent) / atom->natoms;

  for (int i = 0; i < nlocal; i++) {
    pLatent[i] = pLatent[i] - pDev;
  }
}

/* ---------------------------------------------------------------------- */

double FixIEL::parallel_vector_acc(double *v)
{
  double my_acc, res;
  int nlocal = atom->nlocal;

  my_acc = 0.0;
  res = 0.0;
  for (int i = 0; i < atom->nlocal; i++){
    if (atom->mask[i] & groupbit) {
      my_acc += v[i];
    }
  }

  MPI_Allreduce( &my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}
