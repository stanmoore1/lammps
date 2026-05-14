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

/* ----------------------------------------------------------------------
   Contributing author: Saswat Mishra (USF)
------------------------------------------------------------------------- */

#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include "compute_temp_mwindow.h"
//#include "compute_temp.h" // compute_temp_mwindow already includes compute_temp.h
#include "atom.h"
#include "force.h"
#include "group.h"
#include "modify.h"
#include "fix.h"
#include "domain.h"
#include "lattice.h"
#include "error.h"
#include "utils.h"
#include "update.h"

using namespace LAMMPS_NS;

// No need to define. Clashing with lammps
// #define MIN(A,B) ((A) < (B)) ? (A) : (B)
// #define MAX(A,B) ((A) > (B)) ? (A) : (B)

// INVOKED_XXX way of bitmasking is removed and replaced by update->ntimestep
// #define INVOKED_SCALAR 1
// #define INVOKED_VECTOR 2

/* ---------------------------------------------------------------------- */

ComputeTempMWindow::ComputeTempMWindow(LAMMPS *lmp, int narg, char **arg) :
  ComputeTemp(lmp, 3, arg) // base needs ID, group, style unlike Compute
{
  if (narg != 6) error->all(FLERR, "Illegal compute temp/mwindow command");

  vbias[0] = utils::numeric(FLERR, arg[3], false, lmp);
  vbias[1] = utils::numeric(FLERR, arg[4], false, lmp);
  vbias[2] = utils::numeric(FLERR, arg[5], false, lmp);
  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;
  tempflag = 1;
  tempbias = 1;

  vector = new double[6];
}

/* ---------------------------------------------------------------------- */

// ComputeTempMWindow::~ComputeTempMWindow()
// {
//   delete [] vector;
// }
ComputeTempMWindow::~ComputeTempMWindow() = default;


/* ---------------------------------------------------------------------- */

void ComputeTempMWindow::init()
{
  ComputeTemp::init();  //sets of dof/extra_dof/tfactor from dof_compute()

  // fix_dof = 0;
  // for (int i = 0; i < modify->nfix; i++)
  //  fix_dof += modify->fix[i]->dof(igroup);
  // dof_compute();

  masstotal = group->mass(igroup);
}

/* ---------------------------------------------------------------------- */
// Removing to avoid clash with lammps - check later if there is something special about this function that could not be done by the default.
// void ComputeTempMWindow::dof_compute()
// {
//   double natoms = group->count(igroup);
//   int nper = domain->dimension;
//   dof = nper * natoms;
//   dof -= extra_dof + fix_dof;
//   if (dof > 0) tfactor = force->mvv2e / (dof * force->boltz);
//   else tfactor = 0.0;
// }

/* ---------------------------------------------------------------------- */

double ComputeTempMWindow::compute_scalar()
{
  double vthermal[3];

  // invoked |= INVOKED_SCALAR;
  invoked_scalar = update->ntimestep;

  if (dynamic) masstotal = group->mass(igroup);

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double t = 0.0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      vthermal[0] = v[i][0] - vbias[0];
      vthermal[1] = v[i][1] - vbias[1];
      vthermal[2] = v[i][2] - vbias[2];
      if (mass)
        t += (vthermal[0]*vthermal[0] + vthermal[1]*vthermal[1] +
              vthermal[2]*vthermal[2]) * mass[type[i]];
      else
        t += (vthermal[0]*vthermal[0] + vthermal[1]*vthermal[1] +
              vthermal[2]*vthermal[2]) * rmass[i];
    }

  MPI_Allreduce(&t,&scalar,1,MPI_DOUBLE,MPI_SUM,world);
  if (dynamic) dof_compute();
  scalar *= tfactor;
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeTempMWindow::compute_vector()
{
  int i;
  double vthermal[3];

  // invoked |= INVOKED_VECTOR;
  invoked_vector = update->ntimestep;

  if (dynamic) masstotal = group->mass(igroup);

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double massone,t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      vthermal[0] = v[i][0] - vbias[0];
      vthermal[1] = v[i][1] - vbias[1];
      vthermal[2] = v[i][2] - vbias[2];

      if (mass) massone = mass[type[i]];
      else massone = rmass[i];
      t[0] += massone * vthermal[0]*vthermal[0];
      t[1] += massone * vthermal[1]*vthermal[1];
      t[2] += massone * vthermal[2]*vthermal[2];
      t[3] += massone * vthermal[0]*vthermal[1];
      t[4] += massone * vthermal[0]*vthermal[2];
      t[5] += massone * vthermal[1]*vthermal[2];
    }

  MPI_Allreduce(t,vector,6,MPI_DOUBLE,MPI_SUM,world);
  for (i = 0; i < 6; i++) vector[i] *= force->mvv2e;
}

/* ----------------------------------------------------------------------
   remove velocity bias from atom I to leave thermal velocity
------------------------------------------------------------------------- */

void ComputeTempMWindow::remove_bias(int i, double *v)
{
  v[0] -= vbias[0];
  v[1] -= vbias[1];
  v[2] -= vbias[2];
}

/* ----------------------------------------------------------------------
   remove velocity bias from all atoms to leave thermal velocity
------------------------------------------------------------------------- */

void ComputeTempMWindow::remove_bias_all()
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      v[i][0] -= vbias[0];
      v[i][1] -= vbias[1];
      v[i][2] -= vbias[2];
    }
}

/* ----------------------------------------------------------------------
   add back in velocity bias to atom I removed by remove_bias()
   assume remove_bias() was previously called
------------------------------------------------------------------------- */

void ComputeTempMWindow::restore_bias(int i, double *v)
{
  v[0] += vbias[0];
  v[1] += vbias[1];
  v[2] += vbias[2];
}

/* ----------------------------------------------------------------------
   add back in velocity bias to all atoms removed by remove_bias_all()
   assume remove_bias_all() was previously called
------------------------------------------------------------------------- */

void ComputeTempMWindow::restore_bias_all()
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      v[i][0] += vbias[0];
      v[i][1] += vbias[1];
      v[i][2] += vbias[2];
    }
}
