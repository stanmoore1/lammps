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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_John4.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/*
  See documentation!

  Piecewise interaction composed of two cosine functions:

  0 < r < r0:

  F(r) = A.cos( r.(pi/(2.r0)) )
  U(r) = - (A/a).sin( r.a ), where a = pi/(2.r0)

  r0 < r < rc:

  F(r) = B.cos( pi/2 + (r-r0).(pi/(rc-r0)) )
  U(r) = - (B/b).sin( pi/2 + (r-r0).b ), where b = pi/(rc-r0)

  Plus constants of integration etc.
*/

/* ---------------------------------------------------------------------- */

PairJohn4::PairJohn4(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 0;
  writedata = 0;
}

/* ---------------------------------------------------------------------- */

PairJohn4::~PairJohn4()
{
  if (copymode) return;

  if( allocated )
  {
    //
    // Inherited from base Pair class
    //
    memory->destroy(setflag);
    memory->destroy(cutsq);

    //
    // Specific to this class
    //
    memory->destroy(A_vec);
    memory->destroy(B_vec);
    memory->destroy(r0_vec);
    memory->destroy(rc_vec);
  }
}

/* ---------------------------------------------------------------------- */

void PairJohn4::compute(int eflag, int vflag)
{
  if( eflag || vflag ) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double** x = atom->x;
  double** f = atom->f;
  int* type = atom->type;
  int nlocal = atom->nlocal;
  double* special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  int inum = list->inum;
  int* ilist = list->ilist;
  int* numneigh = list->numneigh;
  int** firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for( int ii=0; ii<inum; ii++ )
  {
    int i = ilist[ii];
    int itype = type[i];

    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];

    for( int jj=0; jj<jnum; jj++ )
    {
      int j = jlist[jj];
      double factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      int jtype = type[j];

      double delx = xtmp - x[j][0];
      double dely = ytmp - x[j][1];
      double delz = ztmp - x[j][2];
      double rsq = delx*delx + dely*dely + delz*delz;
      double r2inv = 1.0/rsq;
      double r6inv = r2inv*r2inv*r2inv;

      // Early skip, if no possible interaction
      if( rsq >= cutsq[itype][jtype] ) continue;

      double r = sqrt( rsq );

      //
      // See documentation
      //
      double U, F;
      {
        double A  = A_vec[itype][jtype];
        double B  = B_vec[itype][jtype];
        double r0 = r0_vec[itype][jtype];
        double rc = rc_vec[itype][jtype];

        double a = M_PI/(2.0*r0);
        double b = M_PI/(rc-r0);
        
        if( r < r0 )
        {
          double C = (A/a) - 2.0*(B/b); // constant of integration, to make energy pieces align correctly.
          F = A * cos( r * a );
          U = -(A/a) * sin( r * a ) + C;
        }
        else
        {
          double C = -(B/b); // constant of integration, so U = 0 at rc.
          F = B * cos( M_PI/2 + (r-r0) * b );
          U = -(B/b) * sin( M_PI/2 + (r-r0) * b ) + C;
        }

        F *= r; // "*r" : so normalized when multiply by r2inv & delx,dely,delz
      }

      double fpair = factor_lj * (F) * r2inv;

      f[i][0] += delx*fpair;
      f[i][1] += dely*fpair;
      f[i][2] += delz*fpair;
      
      if( newton_pair || j<nlocal )
      {
        f[j][0] -= delx*fpair;
        f[j][1] -= dely*fpair;
        f[j][2] -= delz*fpair;
      }

      double evdwl = 0.0;
      if( eflag ) evdwl = factor_lj * (U);
      if( evflag ) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
    }
  }

  if( vflag_fdotr ) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairJohn4::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  //
  // Default arrays from Pair base class
  //
  memory->create( setflag, n+1, n+1, "pair:setflag" );
  for( int i=1; i<=n; i++ )
  {
    for( int j=i; j<=n; j++ ) { setflag[i][j] = 0; }
  }
  memory->create( cutsq, n+1, n+1, "pair:cutsq" );

  //
  // Specific to this interaction type
  // 
  memory->create( A_vec,  n+1, n+1, "pair:A" );
  memory->create( B_vec,  n+1, n+1, "pair:B" );
  memory->create( r0_vec, n+1, n+1, "pair:r0" );
  memory->create( rc_vec, n+1, n+1, "pair:rc" );
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairJohn4::settings(int /*narg*/, char ** /*arg*/) {}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairJohn4::coeff(int narg, char **arg)
{
  // Expect: eps sig r0 onset rcut max_force
  int ilo,ihi, jlo,jhi, count=0;

  if( narg < 6 ) error->all(FLERR,"Incorrect args for pair coefficients");
  if( !allocated ) allocate();

  utils::bounds( FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error );
  utils::bounds( FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error );

  double A   = utils::numeric( FLERR, arg[2], false, lmp );
  double r0  = utils::numeric( FLERR, arg[3], false, lmp );
  double B   = utils::numeric( FLERR, arg[4], false, lmp );
  double rc  = utils::numeric( FLERR, arg[5], false, lmp );

  for( int i=ilo; i<=ihi; i++ )
  {
    for( int j=MAX(jlo,i); j<=jhi; j++ )
    {
        A_vec[i][j] = A;
        B_vec[i][j] = B;
       r0_vec[i][j] = r0;
       rc_vec[i][j] = rc;

      cutsq[i][j] = rc * rc;

      setflag[i][j] = 1;

      count++;
    }
  }

  if( count == 0 ) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairJohn4::init_one(int i, int j)
{
  cutsq[j][i] = cutsq[i][j];

  A_vec[j][i] =  A_vec[i][j];
  B_vec[j][i] =  B_vec[i][j];
  r0_vec[j][i] = r0_vec[i][j];
  rc_vec[j][i] = rc_vec[i][j];

  return sqrt( cutsq[i][j] );
}

/* ---------------------------------------------------------------------- */

double PairJohn4::single(
  int i, int j,
  int itype, int jtype,
  double rsq,
  double factor_coul, double factor_lj, double &fforce )
{
  double r2inv  = 1.0/rsq;
  double r6inv  = r2inv*r2inv*r2inv;
  double r12inv = r6inv*r6inv;

  fforce = 0.0;
  if( rsq >= cutsq[itype][jtype] ) return 0.0;

  double r = sqrt( rsq );

  //
  // See documentation
  //
  double U, F;
  {
    double A  = A_vec[itype][jtype];
    double B  = B_vec[itype][jtype];
    double r0 = r0_vec[itype][jtype];
    double rc = rc_vec[itype][jtype];

    double a = M_PI/(2.0*r0);
    double b = M_PI/(rc-r0);
    
    if( r < r0 )
    {
      double C = (A/a) - 2.0*(B/b); // constant of integration, to make energy pieces align correctly.
      F = A * cos( r * a );
      U = -(A/a) * sin( r * a ) + C;
    }
    else
    {
      double C = -(B/b); // constant of integration, so U = 0 at rc.
      F = B * cos( M_PI/2 + (r-r0) * b );
      U = -(B/b) * sin( M_PI/2 + (r-r0) * b ) + C;
    }

    F *= r; // "*r" : so normalized when multiply by r2inv & delx,dely,delz
  }

  fforce = factor_lj * (F) * r2inv;

  return factor_lj * (U);
}

/* ---------------------------------------------------------------------- */
