// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Mark Stevens (SNL)
   Modifying author: You Lin (USF)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "fix_wall_piston.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "modify.h"
#include "output.h"
#include "respa.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixWallPiston::FixWallPiston(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  // Usage: fix ID group_ID wall/piston edgetype pistonposition Energy cutoff <optionalfixMW_ID>
  // edgetype: a string labeling which edge to add potential, choices are xhi, xlo, yhi, ylo, zhi, zlo
  // pistonposition: a float number, the end of piston position, usually corresponds to the edge position.
  // Energy: a float number, the energy height at the edge
  // cutoff: a float number, the potential cutoff distance.
  if (narg < 7) error->all(FLERR,"Illegal fix wall/piston command");
  if (narg == 8) {
     ifix_mw = modify->find_fix(arg[7]);
     if (ifix_mw < 0)
         error->all(FLERR,"fix ID of mwindow/erase for fix wall/piston does not exist");
     fix_mw = modify->fix[ifix_mw];
  } else {
     ifix_mw = -1;
  }

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  extscalar = 1;
  extvector = 1;
  thermo_energy = 1;

  if (strcmp(arg[3],"xlo") == 0) {
    dim = 0;
    side = -1;
  } else if (strcmp(arg[3],"xhi") == 0) {
    dim = 0;
    side = 1;
  } else if (strcmp(arg[3],"ylo") == 0) {
    dim = 1;
    side = -1;
  } else if (strcmp(arg[3],"yhi") == 0) {
    dim = 1;
    side = 1;
  } else if (strcmp(arg[3],"zlo") == 0) {
    dim = 2;
    side = -1;
  } else if (strcmp(arg[3],"zhi") == 0) {
    dim = 2;
    side = 1;
  } else error->all(FLERR,"Illegal fix wall/piston command");

  coord = atof(arg[4]);
  Edeep3 = atof(arg[5]);
  cutoff = atof(arg[6]);

  Edeep3 = Edeep3/(cutoff * cutoff);

  if (dim == 0 && domain->xperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");
  if (dim == 1 && domain->yperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");
  if (dim == 2 && domain->zperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");

  wall_flag = 0;
  wall[0] = wall[1] = wall[2] = wall[3] = 0.0;
}

/* ---------------------------------------------------------------------- */

int FixWallPiston::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallPiston::init()
{
  if (strcmp(update->integrate_style,"respa") == 0)
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixWallPiston::setup(int vflag)
{
  //eflag_enable = 1;
  if (strcmp(update->integrate_style,"verlet") == 0)
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
  //eflag_enable = 0;
}

/* ---------------------------------------------------------------------- */

void FixWallPiston::min_setup(int vflag)
{
  //eflag_enable = 1;
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallPiston::post_force(int vflag)
{
  //bool eflag = false;
  //if (eflag_enable) eflag = true;
  //else if (output->next_thermo == update->ntimestep) eflag = true;

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double delta,delta2,delta3,delnom,Etemp,eng,fpiston;
  double der;
  wall[0] = wall[1] = wall[2] = wall[3] = 0.0;
  wall_flag = 0;
  //if (eflag) eng = 0.0;
  if (ifix_mw < 0)
       der = 0;
  else
       der = fix_mw->compute_scalar();

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (side == -1) delta = x[i][dim] - coord;
      else delta = coord - x[i][dim];
      if (delta <= 0.0) continue;
      if (delta > cutoff) continue;
      delta = cutoff - delta;

      Etemp = Edeep3 * delta;
      if (ifix_mw < 0) {
      } else {
         if (side == -1 && x[i][dim] < der) {
            Etemp = 0.;
         } else if (side == 1 && x[i][dim] > der) {
            Etemp = 0.;
         }
      }

      fpiston = Etemp * side * 2.;
      f[i][dim] -= fpiston;
      wall[0] += Etemp * delta;
      wall[dim+1] += fpiston;
      //if (eflag) eng += Etemp - offset;
    }

  //if (eflag) MPI_Allreduce(&eng,&etotal,1,MPI_DOUBLE,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- */

void FixWallPiston::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallPiston::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of wall interaction
------------------------------------------------------------------------- */

double FixWallPiston::compute_scalar()
{
  // only sum across procs one time

  if (wall_flag == 0)  {
    MPI_Allreduce(wall,wall_all,4,MPI_DOUBLE,MPI_SUM,world);
    wall_flag = 1;
  }
  return wall_all[0];
}

/* ----------------------------------------------------------------------
   components of force on wall
------------------------------------------------------------------------- */

double FixWallPiston::compute_vector(int n)
{
  // only sum across procs one time

  if (wall_flag == 0)  {
    MPI_Allreduce(wall,wall_all,4,MPI_DOUBLE,MPI_SUM,world);
    wall_flag = 1;
  }
  return wall_all[n+1];
}
