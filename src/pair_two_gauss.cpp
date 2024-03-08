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
   Contributing authors: Arben Jusufi, Axel Kohlmeyer (Temple U.)
   Modified by Alex Pak (U Chicago, May 2020)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_two_gauss.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include "integrate.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairTwoGauss::PairTwoGauss(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 0;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairTwoGauss::~PairTwoGauss()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(hgauss);
    memory->destroy(sigmah);
    memory->destroy(rmh);
    memory->destroy(hgauss2);
    memory->destroy(sigmah2);
    memory->destroy(rmh2);
    memory->destroy(pgauss);
    memory->destroy(pgauss2);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairTwoGauss::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,rexp,ugauss,factor_lj;
  double ugauss2, rexp2, fpair2; // for second gauss
  double taper, taperRR, taperRR10, taperRR30, taperRR60;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
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
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
	if (r == (cut[itype][jtype] - 0.95)) taper=0.5;
	else {
	  taperRR = r/(cut[itype][jtype]-0.95) * r/(cut[itype][jtype]-0.95);
	  taperRR10 = taperRR*taperRR*taperRR*taperRR*taperRR*taperRR*taperRR*taperRR*taperRR*taperRR;
	  taperRR30 = taperRR10*taperRR10*taperRR10;
	  taperRR60 = taperRR30 * taperRR30;
	  taper=(1-taperRR30)/(1-taperRR60);
	}
        rexp = (r-rmh[itype][jtype])/sigmah[itype][jtype];
        ugauss = pgauss[itype][jtype]*exp(-0.5*rexp*rexp);
        fpair = factor_lj*taper*rexp/r*ugauss/sigmah[itype][jtype];

        rexp2 = (r-rmh2[itype][jtype])/sigmah2[itype][jtype];
        ugauss2 = pgauss2[itype][jtype]*exp(-0.5*rexp2*rexp2);
        fpair2 = factor_lj*taper*rexp2/r*ugauss2/sigmah2[itype][jtype];

        f[i][0] += delx*(fpair+fpair2);
	f[i][1] += dely*(fpair+fpair2);
	f[i][2] += delz*(fpair+fpair2);
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*(fpair+fpair2);
          f[j][1] -= dely*(fpair+fpair2);
          f[j][2] -= delz*(fpair+fpair2);
        }

        if (eflag) {
          evdwl = ugauss*taper + ugauss2*taper - offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,(fpair+fpair2),delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairTwoGauss::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(hgauss,n+1,n+1,"pair:hgauss");
  memory->create(sigmah,n+1,n+1,"pair:sigmah");
  memory->create(rmh,n+1,n+1,"pair:rmh");
  memory->create(pgauss,n+1,n+1,"pair:pgauss");
  memory->create(hgauss2,n+1,n+1,"pair:hgauss2");
  memory->create(sigmah2,n+1,n+1,"pair:sigmah2");
  memory->create(rmh2,n+1,n+1,"pair:rmh2");
  memory->create(pgauss2,n+1,n+1,"pair:pgauss2");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTwoGauss::settings(int narg, char **arg)
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

void PairTwoGauss::coeff(int narg, char **arg)
{
  if (narg < 8 || narg > 9) error->all(FLERR,"Incorrect number of args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double hgauss_one = utils::numeric(FLERR,arg[2],false,lmp);
  double rmh_one = utils::numeric(FLERR,arg[3],false,lmp);
  double sigmah_one = utils::numeric(FLERR,arg[4],false,lmp);
  double hgauss2_one = utils::numeric(FLERR,arg[5],false,lmp);
  double rmh2_one = utils::numeric(FLERR,arg[6],false,lmp);
  double sigmah2_one = utils::numeric(FLERR,arg[7],false,lmp);
  if (sigmah_one <= 0.0 || sigmah2_one <= 0.0)
    error->all(FLERR,"Negative sigma args for pair coefficients");

  double cut_one = cut_global;
  if (narg == 6) cut_one = utils::numeric(FLERR,arg[8],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      hgauss[i][j] = hgauss_one;
      sigmah[i][j] = sigmah_one;
      rmh[i][j] = rmh_one;
      hgauss2[i][j] = hgauss2_one;
      sigmah2[i][j] = sigmah2_one;
      rmh2[i][j] = rmh2_one;
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

double PairTwoGauss::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    hgauss[i][j] = mix_energy(fabs(hgauss[i][i]), fabs(hgauss[j][j]),
			      fabs(sigmah[i][i]), fabs(sigmah[j][j]));

    hgauss2[i][j] = mix_energy(fabs(hgauss2[i][i]), fabs(hgauss2[j][j]),
			      fabs(sigmah2[i][i]), fabs(sigmah2[j][j]));

    // If either of the particles is repulsive (ie, if hgauss > 0),
    // then the interaction between both is repulsive.
    double sign_hi = (hgauss[i][i] >= 0.0) ? 1.0 : -1.0;
    double sign_hj = (hgauss[j][j] >= 0.0) ? 1.0 : -1.0;
    hgauss[i][j] *= MAX(sign_hi, sign_hj);

    sigmah[i][j] = mix_distance(sigmah[i][i], sigmah[j][j]);
    rmh[i][j] = mix_distance(rmh[i][i], rmh[j][j]);

    sigmah2[i][j] = mix_distance(sigmah2[i][i], sigmah2[j][j]);
    rmh2[i][j] = mix_distance(rmh2[i][i], rmh2[j][j]);

    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
  }

  pgauss[i][j] = hgauss[i][j] / sqrt(MY_2PI) / sigmah[i][j];  
  pgauss2[i][j] = hgauss2[i][j] / sqrt(MY_2PI) / sigmah2[i][j];

  if (offset_flag) {
    double rexp = (cut[i][j]-rmh[i][j])/sigmah[i][j];
    offset[i][j] = pgauss[i][j] * exp(-0.5*rexp*rexp);
  } else offset[i][j] = 0.0;
  offset[i][j] = 0.0; 

  hgauss[j][i] = hgauss[i][j];
  sigmah[j][i] = sigmah[i][j];
  rmh[j][i] = rmh[i][j];
  pgauss[j][i] = pgauss[i][j];

  hgauss2[j][i] = hgauss2[i][j];
  sigmah2[j][i] = sigmah2[i][j];
  rmh2[j][i] = rmh2[i][j];
  pgauss2[j][i] = pgauss2[i][j];

  offset[j][i] = offset[i][j];
  cut[j][i] = cut[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTwoGauss::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&hgauss[i][j],sizeof(double),1,fp);
        fwrite(&rmh[i][j],sizeof(double),1,fp);
        fwrite(&sigmah[i][j],sizeof(double),1,fp);
        fwrite(&hgauss2[i][j],sizeof(double),1,fp);
        fwrite(&rmh2[i][j],sizeof(double),1,fp);
        fwrite(&sigmah2[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTwoGauss::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&hgauss[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&rmh[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigmah[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&hgauss2[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&rmh2[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigmah2[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&hgauss[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&rmh[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigmah[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&hgauss2[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&rmh2[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigmah2[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTwoGauss::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTwoGauss::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairTwoGauss::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g\n",i,hgauss[i][i],rmh[i][i],sigmah[i][i],hgauss2[i][i],rmh2[i][i],sigmah2[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairTwoGauss::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g %g %g\n",i,j,hgauss[i][j],rmh[i][j],sigmah[i][j],hgauss2[i][j],rmh2[i][j],sigmah2[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairTwoGauss::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  double r, rexp,ugauss,rexp2,ugauss2,phigauss,taperRR,taperRR10,taperRR30,taperRR60,taper;

  r=sqrt(rsq);
  if (r == (cut[itype][jtype] - 0.95)) taper=0.5;
  else {
    taperRR = r/(cut[itype][jtype]-0.95) * r/(cut[itype][jtype]-0.95);
    taperRR10 = taperRR*taperRR*taperRR*taperRR*taperRR*taperRR*taperRR*taperRR*taperRR*taperRR;
    taperRR30 = taperRR10*taperRR10*taperRR10;
    taperRR60 = taperRR30 * taperRR30;
    taper=(1-taperRR30)/(1-taperRR60);
  }
  rexp = (r-rmh[itype][jtype])/sigmah[itype][jtype];
  ugauss = pgauss[itype][jtype]*exp(-0.5*rexp*rexp);
  rexp2 = (r-rmh2[itype][jtype])/sigmah2[itype][jtype];
  ugauss2 = pgauss2[itype][jtype]*exp(-0.5*rexp2*rexp2);

  fforce = factor_lj*( taper*rexp/r*ugauss/sigmah[itype][jtype] + taper*rexp2/r*ugauss2/sigmah2[itype][jtype]);

  phigauss = ugauss*taper + ugauss2 - offset[itype][jtype];
  return factor_lj*phigauss;
}

/* ---------------------------------------------------------------------- */
double PairTwoGauss::memory_usage()
{
  const int n=atom->ntypes;

  double bytes = Pair::memory_usage();

  bytes += 7*((n+1)*(n+1) * sizeof(double) + (n+1)*sizeof(double *));
  bytes += 1*((n+1)*(n+1) * sizeof(int) + (n+1)*sizeof(int *));

  return bytes;
}
