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
   Contributing author: Sai Jayaraman (Sandia)
   Modified by: Alex Pak (UChicago, Aug 2020)
------------------------------------------------------------------------- */

#include "pair_srlr_gauss.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-10

/* ---------------------------------------------------------------------- */

PairSRLRGauss::PairSRLRGauss(LAMMPS *lmp) : Pair(lmp)
{
  nextra = 1;
  pvector = new double[1];
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairSRLRGauss::~PairSRLRGauss()
{
  if (copymode) return;

  delete [] pvector;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(a);
    memory->destroy(b);
    memory->destroy(c);
    memory->destroy(d);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairSRLRGauss::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);
  int occ = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      // define a Gaussian well to be occupied if
      // the site it interacts with is within the force maximum

      if (eflag_global && rsq < 0.5/b[itype][jtype]) occ++;

      if (rsq < cutsq[itype][jtype]) {
        fpair = -2.0*a[itype][jtype]*b[itype][jtype] *
          exp(-b[itype][jtype]*rsq);
	fpair += -2.0*c[itype][jtype]*d[itype][jtype] *
          exp(-d[itype][jtype]*rsq);

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = -(a[itype][jtype]*exp(-b[itype][jtype]*rsq) -
                    offset[itype][jtype]);
          evdwl += -(c[itype][jtype]*exp(-d[itype][jtype]*rsq) -
                    offset[itype][jtype]);
	}
        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (eflag_global) pvector[0] = occ;
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSRLRGauss::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut_gauss");
  memory->create(a,n+1,n+1,"pair:a");
  memory->create(b,n+1,n+1,"pair:b");
  memory->create(c,n+1,n+1,"pair:c");
  memory->create(d,n+1,n+1,"pair:d");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSRLRGauss::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = utils::numeric(FLERR,arg[0],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSRLRGauss::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 7)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double a_one = utils::numeric(FLERR,arg[2],false,lmp);
  double b_one = utils::numeric(FLERR,arg[3],false,lmp);
  double c_one = utils::numeric(FLERR,arg[4],false,lmp);
  double d_one = utils::numeric(FLERR,arg[5],false,lmp);

  double cut_one = cut_global;
  if (narg == 7) cut_one = utils::numeric(FLERR,arg[6],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j<=jhi; j++) {
      a[i][j] = a_one;
      b[i][j] = b_one;
      c[i][j] = c_one;
      d[i][j] = d_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSRLRGauss::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    double sign_bi = (b[i][i] >= 0.0) ? 1.0 : -1.0;
    double sign_bj = (b[j][j] >= 0.0) ? 1.0 : -1.0;
    double si = sqrt(0.5/fabs(b[i][i]));
    double sj = sqrt(0.5/fabs(b[j][j]));
    double sij = mix_distance(si, sj);
    b[i][j] = 0.5 / (sij*sij);
    b[i][j] *= MAX(sign_bi, sign_bj);

    double sign_di = (d[i][i] >= 0.0) ? 1.0 : -1.0;
    double sign_dj = (d[j][j] >= 0.0) ? 1.0 : -1.0;
    double si2 = sqrt(0.5/fabs(d[i][i]));
    double sj2 = sqrt(0.5/fabs(d[j][j]));
    double sij2 = mix_distance(si2, sj2);
    d[i][j] = 0.5 / (sij2*sij2);
    d[i][j] *= MAX(sign_di, sign_dj);

    // Negative "a" values are useful for simulating repulsive particles.
    // If either of the particles is repulsive (a<0), then the
    // interaction between both is repulsive.
    double sign_ai = (a[i][i] >= 0.0) ? 1.0 : -1.0;
    double sign_aj = (a[j][j] >= 0.0) ? 1.0 : -1.0;
    a[i][j] = mix_energy(fabs(a[i][i]), fabs(a[j][j]), si, sj);
    a[i][j] *= MIN(sign_ai, sign_aj);

    double sign_ci = (c[i][i] >= 0.0) ? 1.0 : -1.0;
    double sign_cj = (c[j][j] >= 0.0) ? 1.0 : -1.0;
    c[i][j] = mix_energy(fabs(c[i][i]), fabs(c[j][j]), si2, sj2);
    c[i][j] *= MIN(sign_ci, sign_cj);

    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  // cutoff correction to energy
  if (offset_flag) offset[i][j] = a[i][j]*exp(-b[i][j]*cut[i][j]*cut[i][j]) + c[i][j]*exp(-d[i][j]*cut[i][j]*cut[i][j]);
  else offset[i][j] = 0.0;

  a[j][i] = a[i][j];
  b[j][i] = b[i][j];
  c[j][i] = c[i][j];
  d[j][i] = d[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSRLRGauss::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a[i][j],sizeof(double),1,fp);
        fwrite(&b[i][j],sizeof(double),1,fp);
        fwrite(&c[i][j],sizeof(double),1,fp);
        fwrite(&d[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSRLRGauss::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,NULL,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&a[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&b[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&c[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&d[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,NULL,error);
        }
        MPI_Bcast(&a[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&c[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&d[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSRLRGauss::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSRLRGauss::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,NULL,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,NULL,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,NULL,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairSRLRGauss::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g\n",i,a[i][i],b[i][i],c[i][i],d[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairSRLRGauss::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g\n",i,j,a[i][j],b[i][j],c[i][j],d[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairSRLRGauss::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*factor_coul*/, double /*factor_lj*/,
                         double &fforce)
{
  double philj =
    -(a[itype][jtype]*exp(-b[itype][jtype]*rsq) - offset[itype][jtype]);
  philj += -(c[itype][jtype]*exp(-d[itype][jtype]*rsq) - offset[itype][jtype]);
  fforce = -2.0*a[itype][jtype]*b[itype][jtype] * exp(-b[itype][jtype]*rsq);
  fforce += -2.0*c[itype][jtype]*d[itype][jtype] * exp(-d[itype][jtype]*rsq);
  return philj;
}

/* ---------------------------------------------------------------------- */

void *PairSRLRGauss::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"a") == 0) return (void *) a;
  return NULL;
}
