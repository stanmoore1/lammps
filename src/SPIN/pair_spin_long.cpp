/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)
                         Stan Moore (SNL)
   
   Please cite the related publication:
   Tranchida, J., Plimpton, S. J., Thibaudeau, P., & Thompson, A. P. (2018). 
   Massively parallel symplectic algorithm for coupled magnetic spin dynamics 
   and molecular dynamics. arXiv preprint arXiv:1801.10233.
------------------------------------------------------------------------- */


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pair_spin_long.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "fix_nve_spin.h"
#include "force.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "error.h"
#include "update.h"


using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairSpinLong::PairSpinLong(LAMMPS *lmp) : PairSpin(lmp),
lockfixnvespin(NULL)
{
  single_enable = 0;
  ewaldflag = pppmflag = 1;
  respa_enable = 0;
  no_virial_fdotr_compute = 1;
  lattice_flag = 0;

  hbar = force->hplanck/MY_2PI;		// eV/(rad.THz)
  mub = 5.78901e-5;                	// in eV/T
  mu_0 = 1.2566370614e-6;		// in T.m/A
  mub2mu0 = mub * mub * mu_0;		// in eV
  mub2mu0hbinv = mub2mu0 / hbar;	// in rad.THz

}

/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

PairSpinLong::~PairSpinLong()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairSpinLong::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;  
  double r,rinv,r2inv,rsq,inorm;
  double fdx,fdy,fdz,fmdx,fmdy,fmdz;
  double grij,expm2,t,erfc;
  double gigj,g1,g2,b0,b1,b2,b3,g1b2_g2b3;
  double evdwl,ecoul;
  double xi[3],rij[3],eij[3];
  double spi[3],spj[3],fi[3],fmi[3];
  double sdots,sidotr,sjdotr,pre1,pre2,pre3;
  double local_cut2;
  int *ilist,*jlist,*numneigh,**firstneigh;  

  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double **fm = atom->fm;
  double **sp = atom->sp;	
  int *type = atom->type;  
  int nlocal = atom->nlocal;  
  double *special_coul = force->special_coul;
  double qqrd2e = force->qqrd2e;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  pre1 = 2.0 * g_ewald / MY_PIS;
  pre2 = 4.0 * pow(g_ewald,3.0) / MY_PIS;
  pre3 = 8.0 * pow(g_ewald,5.0) / MY_PIS;

  // computation of the exchange interaction
  // loop over atoms and their neighbors

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xi[0] = x[i][0];
    xi[1] = x[i][1];
    xi[2] = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i]; 
    spi[0] = sp[i][0]; 
    spi[1] = sp[i][1]; 
    spi[2] = sp[i][2];
    itype = type[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      spj[0] = sp[j][0]; 
      spj[1] = sp[j][1]; 
      spj[2] = sp[j][2]; 

      evdwl = 0.0;

      fi[0] = fi[1] = fi[2] = 0.0;
      fmi[0] = fmi[1] = fmi[2] = 0.0;
     
      rij[0] = x[j][0] - xi[0];
      rij[1] = x[j][1] - xi[1];
      rij[2] = x[j][2] - xi[2];
      rsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];
      inorm = 1.0/sqrt(rsq);
      eij[0] = inorm*rij[0]; 
      eij[1] = inorm*rij[1]; 
      eij[2] = inorm*rij[2]; 

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

        if (rsq < cut_spinsq) {
          r = sqrt(rsq);
          grij = g_ewald * r;
          expm2 = exp(-grij*grij);
          t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;

          sdots = sp[i][0]*sp[j][0] + sp[i][1]*sp[j][1] + 
	    sp[i][2]*sp[j][2];
          sidotr = sp[i][0]*rij[0] + sp[i][1]*rij[1] + 
	    sp[i][2]*rij[2];
          sjdotr = sp[j][0]*rij[0] + sp[j][1]*rij[1] + 
	    sp[j][2]*rij[2];

          b0 = erfc * rinv;
          b1 = (b0 + pre1*expm2) * r2inv;
          b2 = (3.0*b1 + pre2*expm2) * r2inv;
          b3 = (5.0*b2 + pre3*expm2) * r2inv;
          
	  g1 = sdots;
	  g2 = -sidotr*sjdotr;

          g1b2_g2b3 = g1*b2 + g2*b3;

	  fdx = rij[0] * g1b2_g2b3 + 
	    b2 * (sjdotr*sp[i][0] + sidotr*sp[j][0]);
          fdy = rij[1] * g1b2_g2b3 + 
	    b2 * (sjdotr*sp[i][1] + sidotr*sp[j][1]);
          fdz = rij[2] * g1b2_g2b3 + 
	    b2 * (sjdotr*sp[i][2] + sidotr*sp[j][2]);

	  fmdx = b2 * sjdotr *rij[0] - b1 * sp[j][0] ;
	  fmdy = b2 * sjdotr *rij[1] - b1 * sp[j][1] ;
	  fmdz = b2 * sjdotr *rij[2] - b1 * sp[j][2] ;

	}
      }

      gigj = sp[i][3]*sp[j][3];

      fi[0] = gigj * mub2mu0 * fdx;
      fi[1] = gigj * mub2mu0 * fdy;
      fi[2] = gigj * mub2mu0 * fdz;
      fmi[0] = gigj * mub2mu0hbinv * fmdx;
      fmi[1] = gigj * mub2mu0hbinv * fmdy;
      fmi[2] = gigj * mub2mu0hbinv * fmdz;
      
      // force accumulation

      f[i][0] += fi[0];	 
      f[i][1] += fi[1];	  	  
      f[i][2] += fi[2];
      fm[i][0] += fmi[0];	 
      fm[i][1] += fmi[1];	  	  
      fm[i][2] += fmi[2];

      if (newton_pair || j < nlocal) {
	f[j][0] -= fi[0];	 
        f[j][1] -= fi[1];	  	  
        f[j][2] -= fi[2];
      }

      if (eflag) {
	if (rsq <= local_cut2) {
	  evdwl -= spi[0]*fmi[0] + spi[1]*fmi[1] + 
	    spi[2]*fmi[2];
	  evdwl *= hbar;
	}
      } else evdwl = 0.0;


      if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
	  evdwl,ecoul,fi[0],fi[1],fi[2],rij[0],rij[1],rij[2]);

    }
  }
}

/* ----------------------------------------------------------------------
   update the pair interaction fmi acting on the spin ii
------------------------------------------------------------------------- */

void PairSpinLong::compute_single_pair(int ii, double fmi[3])
{

}

/* ----------------------------------------------------------------------
   compute exchange interaction between spins i and j
------------------------------------------------------------------------- */

void PairSpinLong::compute_long(int i, int j, double rsq, double fmi[3], double spj[3])
{
  int *type = atom->type;  
  int itype, jtype;
  itype = type[i];
  jtype = type[j];

  fmi[0] += 0.0;
  fmi[1] += 0.0;
  fmi[2] += 0.0;
}

/* ----------------------------------------------------------------------
   compute the mechanical force due to the exchange interaction between atom i and atom j
------------------------------------------------------------------------- */

void PairSpinLong::compute_long_mech(int i, int j, double rsq, double rij[3], double fi[3],  double spi[3], double spj[3])
{
  int *type = atom->type;  
  int itype, jtype;
  itype = type[i];
  jtype = type[j];

  fi[0] -= 0.0;
  fi[1] -= 0.0;
  fi[2] -= 0.0;
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSpinLong::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSpinLong::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2)
    error->all(FLERR,"Incorrect args in pair_style command");

  if (strcmp(update->unit_style,"metal") != 0)
    error->all(FLERR,"Spin simulations require metal unit style");

  cut_spin = force->numeric(FLERR,arg[0]);

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSpinLong::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  // check if args correct

  if (strcmp(arg[2],"long") != 0)
    error->all(FLERR,"Incorrect args in pair_style command");
  if (narg != 3) 
    error->all(FLERR,"Incorrect args in pair_style command");
  
  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSpinLong::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  
  double cut = cut_spin;
  return cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSpinLong::init_style()
{
  if (!atom->sp_flag)
    error->all(FLERR,"Pair spin requires atom/spin style");
  
  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  
  // checking if nve/spin is a listed fix

  int ifix = 0;
  while (ifix < modify->nfix) {
    if (strcmp(modify->fix[ifix]->style,"nve/spin") == 0) break;
    ifix++;
  }
  if (ifix == modify->nfix)
    error->all(FLERR,"pair/spin style requires nve/spin");

  // get the lattice_flag from nve/spin

  for (int i = 0; i < modify->nfix; i++) {
    if (strcmp(modify->fix[i]->style,"nve/spin") == 0) {
      lockfixnvespin = (FixNVESpin *) modify->fix[i];
      lattice_flag = lockfixnvespin->lattice_flag;
    }
  }

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all(FLERR,"Pair style requires a KSpace style");

  g_ewald = force->kspace->g_ewald;

  cut_spinsq = cut_spin * cut_spin;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSpinLong::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSpinLong::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSpinLong::write_restart_settings(FILE *fp)
{
  fwrite(&cut_spin,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSpinLong::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_spin,sizeof(double),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_spin,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

void *PairSpinLong::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut") == 0) {
    dim = 0;
    return (void *) &cut_spin;
  } else if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cut_spin;
  } else if (strcmp(str,"ewald_order") == 0) {
    ewald_order = 0;
    ewald_order |= 1<<1;
    ewald_order |= 1<<3;
    dim = 0;
    return (void *) &ewald_order;
  } else if (strcmp(str,"ewald_mix") == 0) {
    dim = 0;
    return (void *) &mix_flag;
  }
  return NULL;
}
