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

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "fix_mwindow_erase.h"
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "comm.h"
#include "group.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

void FixMWindowErase::end_of_step()
{

  // qq is a time derivative of average atom energy
  int i, natoms = 0;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double Etnew;

  if ((update->ntimestep % nfreq_u_d) != 1) return;
    MPI_Allreduce(&nlocal,&natoms,1,MPI_INT,MPI_SUM,world);

  if (use_Number_of_Atoms) {
       Etot = natoms;
       Etnew = compute_pe->compute_scalar();
       if (natoms > 0) Etnew /= natoms;
  } else {
       Etot = compute_pe->compute_scalar();
       if (natoms > 0) Etot /= natoms;
       if (update -> ntimestep <= nevery) {
           //Elast = Etot;
       }
  }
  if (me == 0 && update -> ntimestep <= nevery) {
    if (use_Number_of_Atoms) {
       if (screen) fprintf(screen, "MW: Step /Nlast   /Ncur/Eatom/  w/       qq/       bb      /Rx     /A_er/d_er\n");
       if (logfile) fprintf(logfile, "MW: Step /Nlast   /Ncur/Eatom/  w/       qq/       bb      /Rx     /A_er/d_er\n");
    } else {
       if (screen) fprintf(screen, "MW: Step /Elast   /Etot/Natom/  w/       qq/       bb      /Rx     /A_er/d_er\n");
       if (logfile) fprintf(logfile, "MW: Step /Elast   /Etot/Natom/  w/       qq/       bb      /Rx     /A_er/d_er\n");
    }
  }
      
  qq = + (Etot - Elast) /(Ewish - E0);

  // bb is a desired rate dE/dt
  w = (Etot - E0)/(Ewish - E0);
  bb = mw_erase_rate_dwmax * (1. - w);
  Rx = 0.;
  if (mw_erase_rate_b < 0.) {
       if (Ewish < E0)
          Rx = - mw_erase_rate_b * nevery * (qq -bb);
       else
          Rx =  mw_erase_rate_b * nevery * (qq -bb);
       if (Rx > 0.)
          Rx = (double)nevery * Rx / ((double)nevery + Rx);
       else
          Rx = (double)nevery * Rx / ((double)nevery - Rx);
       mw_erase_position_Aerase += Rx; 
       if (mw_erase_position_Aerase < 0.) mw_erase_position_Aerase = 0.;
  }
  mw_erase_position_d = mw_erase_d_max - mw_erase_position_Aerase * (Slope + 1. - w) / Slope;
  if (mw_erase_position_d < mw_erase_d_min) mw_erase_position_d = mw_erase_d_min;
  if (mw_erase_position_d > mw_erase_d_max) mw_erase_position_d = mw_erase_d_max;
  if (me == 0) {
    if (use_Number_of_Atoms) {
        if (screen) fprintf(screen, "MW: %d %d %d %g %g %g %g %g %g %g\n",
             update -> ntimestep, (int)Elast, (int)Etot, Etnew, w, qq, bb, Rx, mw_erase_position_Aerase, mw_erase_position_d);
        if (logfile) fprintf(logfile, "MW: %d %d %d %g %g %g %g %g %g %g\n",
             update -> ntimestep, (int)Elast, (int)Etot, Etnew, w, qq, bb, Rx, mw_erase_position_Aerase, mw_erase_position_d);
    } else {
        if (screen) fprintf(screen, "MW: %d %g %g %d %g %g %g %g %g %g\n",
             update -> ntimestep, Elast, Etot, natoms, w, qq, bb, Rx, mw_erase_position_Aerase, mw_erase_position_d);
        if (logfile) fprintf(logfile, "MW: %d %g %g %d %g %g %g %g %g %g\n",
             update -> ntimestep, Elast, Etot, natoms, w, qq, bb, Rx, mw_erase_position_Aerase, mw_erase_position_d);
    }
  }
  Elast = Etot;
}

FixMWindowErase::FixMWindowErase(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  // Usage: fix ID group_ID mwindow/erase edgetype Aerase Slope d_min d_max rate_b dwmax Elast E0 Ew N N_u_d region-ID Compute_ID
  // edgetype: a string labeling which edge to add potential, choices are xhi, xlo, yhi, ylo, zhi, zlo
  // Aerase   : a float number, starting Aerase.
  // Slope  : The slope of d_er vs Aerase
  // d_min  : a float number, minimum d_er.
  // d_max  : a float number, maximum d_er.
  // rate_b : a float number, damping rate. The change in one timestep.
  // dwmax  : a float number, target damping rate. The change in one timestep.
  // Elast     : a float number, unperturbed energy per atom. If Elast="N=%d", we use the number of atoms as control
  // E0     : a float number, unperturbed energy per atom.
  // Ewf    : a float number, target energy per atom.
  // N      : delete atoms every this many timesteps
  // N_u_d  : update erasing plane every this many timesteps
  use_Number_of_Atoms = 0;
  MPI_Comm_rank(world,&me);
  if (narg != 17) error->all(FLERR,"Illegal fix mwindow/erase command");

  scalar_flag = 1;
  extscalar = 0;

  if (strcmp(arg[3],"xlo") == 0) {
    mw_erase_dim = 0;
    mw_erase_side = -1;
  } else if (strcmp(arg[3],"xhi") == 0) {
    mw_erase_dim = 0;
    mw_erase_side = 1;
  } else if (strcmp(arg[3],"ylo") == 0) {
    mw_erase_dim = 1;
    mw_erase_side = -1;
  } else if (strcmp(arg[3],"yhi") == 0) {
    mw_erase_dim = 1;
    mw_erase_side = 1;
  } else if (strcmp(arg[3],"zlo") == 0) {
    mw_erase_dim = 2;
    mw_erase_side = -1;
  } else if (strcmp(arg[3],"zhi") == 0) {
    mw_erase_dim = 2;
    mw_erase_side = 1;
  } else error->all(FLERR,"Illegal fix wall/mwindow command");

  mw_erase_position_Aerase = atof(arg[4]);
  Slope = atof(arg[5]);
  mw_erase_d_min = atof(arg[6]);
  mw_erase_d_max = atof(arg[7]);
  mw_erase_rate_b = atof(arg[8]);
  mw_erase_rate_dwmax = atof(arg[9]);
  if (strncmp(arg[10], "N=", 2) == 0) {
    Elast = atof(arg[10] + 2);
    use_Number_of_Atoms = 1;
  } else
     Elast = atof(arg[10]);
  E0 = atof(arg[11]);
  Ewf = atof(arg[12]);
     Ewish = Ewf;

  mw_erase_position_d = mw_erase_position_Aerase;

  nfreq = atoi(arg[13]);
  nfreq_u_d = atoi(arg[14]);
  nevery = 1;
  auto region = domain->get_region_by_id(arg[15]);

  if (nfreq <= 0) error->all(FLERR,"Illegal fix mwindow/erase command");
  if (!region) error->all(FLERR,"Fix mwindow/erase region ID does not exist");

  int n = strlen(arg[15]) + 1;
  id_compute_pe = new char[n];
  strcpy(id_compute_pe,arg[16]);

  int icompute_pe = modify->find_compute(id_compute_pe);
  if (icompute_pe < 0)
    error->all(FLERR,"Compute ID of atomic epot for fix mwindow/erase does not exist");

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = (update->ntimestep/nfreq)*nfreq + nfreq;
  ndeleted = 0;

  nmax = 0;
  list = NULL;
  mark = NULL;
}

/* ---------------------------------------------------------------------- */

FixMWindowErase::~FixMWindowErase()
{
  memory->sfree(list);
  memory->sfree(mark);
  delete id_compute_pe;
}

/* ---------------------------------------------------------------------- */

int FixMWindowErase::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMWindowErase::init()
{
  // check that no deletable atoms are in atom->firstgroup
  // deleting such an atom would not leave firstgroup atoms first

  int icompute_pe = modify->find_compute(id_compute_pe);
  if (icompute_pe < 0)
    error->all(FLERR,"Compute ID of atomic epot for fix mwindow/erase does not exist");
  compute_pe = modify->compute[icompute_pe];

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < nlocal; i++)
      if ((mask[i] & groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot mwindow/erase atoms in atom_modify first group");
  }
}

/* ----------------------------------------------------------------------
   perform particle deletion
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixMWindowErase::pre_exchange()
{
  int i,iwhichglobal,iwhichlocal;
  double delta;

  if (update->ntimestep != next_reneighbor && update->ntimestep != 1) return;

  // grow list and mark arrays if necessary

  if (atom->nlocal > nmax) {
    memory->sfree(list);
    memory->sfree(mark);
    nmax = atom->nmax;
    list = (int *) memory->smalloc(nmax*sizeof(int),"mwindow/erase:list");
    mark = (int *) memory->smalloc(nmax*sizeof(int),"mwindow/erase:mark");
  }

  // nall = # of deletable atoms in region
  // nbefore = # on procs before me

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int ncount = 0;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
//      if (region->match(x[i][0],x[i][1],x[i][2]))
	list[ncount++] = i;

  int nall,nwhack;
  
  // nwhack = total number of atoms to delete
  // choose atoms randomly across all procs and mark them for deletion
  // shrink local list of candidates as my atoms get marked for deletion

  for (i = 0; i < nlocal; i++) mark[i] = 0;
  
  nall = 0;
  for (iwhichlocal = 0; iwhichlocal < ncount; iwhichlocal ++) {
    i = list[iwhichlocal];
      if (mw_erase_side == -1) delta = x[i][mw_erase_dim] - mw_erase_position_d;
      else delta = mw_erase_position_d - x[i][mw_erase_dim];
      if (delta <= 0.0) {
         mark[i] = 1;
         nall ++;
      }
  }
  MPI_Allreduce(&nall,&nwhack,1,MPI_INT,MPI_SUM,world);
  
  // delete my marked atoms
  // loop in reverse order to avoid copying marked atoms
  
  AtomVec *avec = atom->avec;
  
  for (i = nlocal-1; i >= 0; i--) {
    if (mark[i]) {
      avec->copy(atom->nlocal-1,i,1);
      atom->nlocal--;
    }
  }

  // reset global natoms
  // if global map exists, reset it now instead of waiting for comm
  // since deleting atoms messes up ghosts

  atom->natoms -= nwhack;
  if (nwhack && atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }

  // statistics
  
  ndeleted = nwhack;
  if (update->ntimestep == next_reneighbor)
      next_reneighbor = update->ntimestep + nfreq;
  if (me == 0) {
/*
    if (screen) fprintf(screen, "MWD: %d %g Deleted\n",
         update -> ntimestep, 1. * nwhack);
    if (logfile) fprintf(logfile, "MWD: %d %g Deleted\n",
         update -> ntimestep, 1.* nwhack);
*/
  }
}

/* ----------------------------------------------------------------------
   return erasing plane position
------------------------------------------------------------------------- */

double FixMWindowErase::compute_scalar()
{
  return 1.0*mw_erase_position_d;
}
double FixMWindowErase::compute_vector(int n)
{
  int i;
  double t[10];
  // t-> step, elast ecurrent, w, qq, bb, Rx, A_er, D_er
  t[0] = update -> ntimestep;
  t[1] =  Elast;
  t[2] = Etot;
  t[3] =  w;
  t[4] = qq;
  t[5] = bb;
  t[6] = Rx;
  t[7] = mw_erase_position_Aerase;
  t[8] = mw_erase_position_d;

  return t[n + 1];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixMWindowErase::memory_usage()
{
  double bytes = 2*nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file 
------------------------------------------------------------------------- */

void FixMWindowErase::write_restart(FILE *fp)
{
  write_restart_settings(fp);

}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void FixMWindowErase::read_restart(FILE *fp)
{
  read_restart_settings(fp);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void FixMWindowErase::write_restart_settings(FILE *fp)
{
  fwrite(&mw_erase_position_Aerase, sizeof(double),1,fp);
  fwrite(&mw_erase_position_d, sizeof(double),1,fp);
  fwrite(&Elast, sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void FixMWindowErase::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&mw_erase_position_Aerase,sizeof(double),1,fp);
    fread(&mw_erase_position_d,sizeof(double),1,fp);
    fread(&Elast, sizeof(double),1,fp);
  }
  MPI_Bcast(&mw_erase_position_Aerase,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mw_erase_position_d,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&Elast,1,MPI_DOUBLE,0,world);
}
