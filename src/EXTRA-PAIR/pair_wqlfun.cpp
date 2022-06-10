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
   Contributing author:  Tomas Oppelstrup (LLNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pair_wqlfun.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "math_special.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "modify.h"

#include "stringfunction.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

/* ---------------------------------------------------------------------- */

PairWQLFun::PairWQLFun(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 0;
  writedata = 0;

  parsetree = nullptr;
  parms = nullptr;
  parm_vals = nullptr;
  parm_vals2 = nullptr;
  expression = nullptr;

  yvec = nullptr;
  zvec = nullptr;
  Qlm = nullptr;
  rsqrt = nullptr;
  sqrtfact = nullptr;
  Almvec = nullptr;

  maxshort = 10;
  neighshort = nullptr;

  chunksize = 32768;
}

/* ---------------------------------------------------------------------- */

PairWQLFun::~PairWQLFun()
{
  if (copymode) return;

  // Deallocate energy expression variables

  delete_tree(parsetree);
  for (int i = 0; i<nparms; i++)
    delete[] parms[i];
  delete[] parms;
  delete[] parm_vals;
  delete[] parm_vals2;
  delete[] expression;

  delete[] yvec;
  delete[] zvec;
  delete[] Qlm;
  delete[] rsqrt;
  delete[] sqrtfact;
  delete[] Almvec;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(neighshort);
  }
}

/* ---------------------------------------------------------------------- */

void PairWQLFun::compute(int eflag, int vflag)
{
  int inum;
  int *ilist,*numneigh,**firstneigh;
  double ql,wlscale,wl;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  const int *type = atom->type;
  const int nlocal = atom->nlocal;
  const int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];
    int numshort = 0;
    const int itype = type[i];

    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];

    for (int m = 0; m<=lmax; m++)
      Qlm[m][0] = Qlm[m][1] = 0;

    double nnei_scale = 0; // How many effective neighbors? Used for
                           // scaling ql. Set to 1 if zero neighbors
                           // within cutoff, otherwise to:
                           //    sum fsmooth(r_ij), over j

    // Wl ('energy') section

    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      const int jtype = type[j];
      const double delx = xtmp - x[j][0];
      const double dely = ytmp - x[j][1];
      const double delz = ztmp - x[j][2];
      const double rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq[itype][jtype]) {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
        double df;
        const double rinv = 1.0/sqrt(rsq);
        const double r = rsq*rinv;
        const double xn = delx*rinv;
        const double yn = dely*rinv;
        const double zn = delz*rinv;
        const double fij = fsmooth(r,&df);
        nnei_scale += fij;
        ylmallcompress(lmax,xn,yn,zn,yvec);
        const int idx = lmax*(lmax+1)/2;
        for (int m = 0; m<=lmax; m++) {
          Qlm[m][0] = Qlm[m][0] + fij*yvec[idx+m][0];
          Qlm[m][1] = Qlm[m][1] + fij*yvec[idx+m][1];
        }
      }
    }
    if (nnei_scale <= 0) nnei_scale = 1;

    // Compute Ql for this atom

    ql = Qlm[0][0]*Qlm[0][0] + Qlm[0][1]*Qlm[0][1];
    for (int m = 1; m<=lmax; m++) {
      ql += 2*(Qlm[m][0]*Qlm[m][0] +
                Qlm[m][1]*Qlm[m][1]);
    }

    if (ql > 0)
      wlscale = 1 / sqrt(ql*ql*ql);
    else
      wlscale = 0;

    // Compute Wl

    wl = 0;
    int widx = 0;
    for (int m1 = -lmax; m1<=0; m1++) {
      for (int m2 = 0; m2<=((-m1)>>1); m2++) {
        const int m3 = -(m1 + m2);
        // Loop enforces -lmax<=m1<=0<=m2<=m3<=lmax, and m1+m2+m3=0

        // For even lmax, W3j is invariant under permutation of
        // (m1,m2,m3) and (m1,m2,m3)->(-m1,-m2,-m3). The loop
        // structure enforces visiting only one member of each
        // such symmetry (invariance) group.

        const int sgn = 1 - 2*(m1&1);
        double Q1Q2[3][2],Q1Q2Q3[2];
        // m1 <= 0, and Qlm[-m] = (-1)^m*conjg(Qlm[m]). sgn = (-1)^m.
        Q1Q2[0][0] = (Qlm[-m1][0]*Qlm[ m2][0] + Qlm[-m1][1]*Qlm[ m2][1])*sgn;
        Q1Q2[0][1] = (Qlm[-m1][0]*Qlm[ m2][1] - Qlm[-m1][1]*Qlm[ m2][0])*sgn;
        /*
          Q1Q2[1][0] =   Qlm[ m2][0]*Qlm[ m3][0] - Qlm[ m2][1]*Qlm[ m3][1];
          Q1Q2[1][1] =   Qlm[ m2][0]*Qlm[ m3][1] + Qlm[ m2][1]*Qlm[ m3][0];
          Q1Q2[2][0] = (Qlm[ m3][0]*Qlm[-m1][0] + Qlm[ m3][1]*Qlm[-m1][1])*sgn;
          Q1Q2[2][1] = (-Qlm[ m3][0]*Qlm[-m1][1] + Qlm[ m3][1]*Qlm[-m1][0])*sgn;
        */
        Q1Q2Q3[0] = Q1Q2[0][0]*Qlm[m3][0] - Q1Q2[0][1]*Qlm[m3][1];
        //Q1Q2Q3[1] = Q1Q2[0][0]*Qlm[m3][1] + Q1Q2[0][1]*Qlm[m3][0];

        /*
          //Other permutation: Q2Q3Q1
            Q1Q2Q3[0] = (Q1Q2[1][0]*Qlm[-m1][0] + Q1Q2[1][1]*Qlm[-m1][1])*sgn;
            Q1Q2Q3[1] = (-Q1Q2[1][0]*Qlm[-m1][1] + Q1Q2[1][1]*Qlm[-m1][0])*sgn;
          //Other permutation: Q3Q1Q2
            Q1Q2Q3[0] = Q1Q2[2][0]*Qlm[m2][0] - Q1Q2[2][1]*Qlm[m2][1];
            Q1Q2Q3[1] = Q1Q2[2][0]*Qlm[m2][1] + Q1Q2[2][1]*Qlm[m2][0];
        */

        const double c = w3jlist[widx++];
        wl += Q1Q2Q3[0]*c;
      }
    }
    wl *= wlscale; // wl is really wlhat

    // Set parameters for expression/tree evaluation

    if (eflag_either) {
      parm_vals[qparmidx] = sqrt(ql)/nnei_scale;
      parm_vals[wparmidx] = wl;
      const double uij = eval_tree(parsetree,nparms,parm_vals);
      if (eflag_global) evdwl += uij;
      if (eflag_atom) eatom[i] += uij;
    }

    // Derivative, forces section

    double fitmp[3] = {0.0,0.0,0.0};
    for (int jj = 0; jj < numshort; jj++) {
      const int j = neighshort[jj];
      const double delx = x[i][0] - x[j][0];
      const double dely = x[i][1] - x[j][1];
      const double delz = x[i][2] - x[j][2];
      const double rsq = delx*delx + dely*dely + delz*delz;

      if (rsq > 0) {
        double df;
        const double rinv = 1.0/sqrt(rsq);
        const double r = rsq*rinv;
        const double xn = delx*rinv;
        const double yn = dely*rinv;
        const double zn = delz*rinv;
        const double fij = fsmooth(r,&df);
        double gradQlm[lmax+1][3][2];

        const int lmax1 = lmax + 1; // Need higher degree to enable differentiation
        ylmallcompress(lmax1,xn,yn,zn,yvec);
        ylm2zlmcompress(lmax1,yvec,zvec); // Renormalization to help with differentiation
        for (int m = 0; m<=lmax; m++) {
          const double Alm = Almvec[m];

          // Derivatives of Qlm w.r.t (dx,dy,dz) = r(i) - x(j)

	  double yval[2],dy[3][2];
	  zlmderiv1compress(lmax,m,xn,yn,zn,zvec,yval,dy); // Differentiate re-normalized Ylm (i.e. Zlm)
	  const double dQlmdx0 = Alm*(df*xn*yval[0] + fij*dy[0][0]*rinv); // Alm re-re-normalizes
	  const double dQlmdx1 = Alm*(df*xn*yval[1] + fij*dy[0][1]*rinv); // Zlm to Ylm again...
	
	  const double dQlmdy0 = Alm*(df*yn*yval[0] + fij*dy[1][0]*rinv);
	  const double dQlmdy1 = Alm*(df*yn*yval[1] + fij*dy[1][1]*rinv);
	
	  const double dQlmdz0 = Alm*(df*zn*yval[0] + fij*dy[2][0]*rinv);
	  const double dQlmdz1 = Alm*(df*zn*yval[1] + fij*dy[2][1]*rinv);
	  gradQlm[m][0][0] = dQlmdx0;  // Re(d Qlm/dx)
	  gradQlm[m][0][1] = dQlmdx1;  // Im(d Qlm/dx)
	  gradQlm[m][1][0] = dQlmdy0;  // Re(d Qlm/dy)
	  gradQlm[m][1][1] = dQlmdy1;  // Im(d Qlm/dy)
	  gradQlm[m][2][0] = dQlmdz0;  // Re(d Qlm/dz)
	  gradQlm[m][2][1] = dQlmdz1;  // Im(d Qlm/dz)
        }

        // Compute derivatives of wlscale (one over WQLFun expression denominator) w.r.t atom j

        double gradwlscale[3],gradql[3] = {0,0,0};
        for (int k = 0; k<3; k++)
          gradql[k] += 2*(Qlm[0][0]*gradQlm[0][k][0] +
                           Qlm[0][1]*gradQlm[0][k][1]);
        for (int m = 1; m<=lmax; m++)
          for (int k = 0; k<3; k++)
            gradql[k] += 4*(Qlm[m][0]*gradQlm[m][k][0] +
                             Qlm[m][1]*gradQlm[m][k][1]);

        if (wlscale > 0)
          for (int k = 0; k<3; k++)
            gradwlscale[k] = (-3.0/2)*wlscale/ql * gradql[k];
        else
          for (int k = 0; k<3; k++)
            gradwlscale[k] = 0;

        // Forces
        // Need to accumulate everything in inner-most loop so we can
        //  calculate contribution to forces on 'j' particles...

        double gradwl[3] = {0,0,0};

        int widx = 0;
        for (int m1 = -lmax; m1<=0; m1++) {
          for (int m2 = 0; m2<=((-m1)>>1); m2++) {
            const int m3 = -(m1 + m2);
            // Loop enforces -lmax<=m1<=0<=m2<=m3<=lmax, and m1+m2+m3=0

            // For even lmax, W3j is invariant under permutation of
            // (m1,m2,m3) and (m1,m2,m3)->(-m1,-m2,-m3). The loop
            // structure enforces visiting only one member of each
            // such symmetry (invariance) group.

            const int sgn = 1 - 2*(m1&1);
            double Q1Q2[3][2],Q1Q2Q3[2];
            Q1Q2[0][0] =  (Qlm[-m1][0]*Qlm[ m2][0] + Qlm[-m1][1]*Qlm[ m2][1])*sgn; // m1<0 -> conjugation
            Q1Q2[0][1] =  (Qlm[-m1][0]*Qlm[ m2][1] - Qlm[-m1][1]*Qlm[ m2][0])*sgn;
            Q1Q2[1][0] =   Qlm[ m2][0]*Qlm[ m3][0] - Qlm[ m2][1]*Qlm[ m3][1];
            Q1Q2[1][1] =   Qlm[ m2][0]*Qlm[ m3][1] + Qlm[ m2][1]*Qlm[ m3][0];
            Q1Q2[2][0] =  (Qlm[ m3][0]*Qlm[-m1][0] + Qlm[ m3][1]*Qlm[-m1][1])*sgn;
            Q1Q2[2][1] = (-Qlm[ m3][0]*Qlm[-m1][1] + Qlm[ m3][1]*Qlm[-m1][0])*sgn;
            Q1Q2Q3[0] = Q1Q2[0][0]*Qlm[m3][0] - Q1Q2[0][1]*Qlm[m3][1];
            //Q1Q2Q3[1] = Q1Q2[0][0]*Qlm[m3][1] + Q1Q2[0][1]*Qlm[m3][0];

            const double c = w3jlist[widx++];
            for (int k = 0; k<3; k++) {
              // Since wl is real by summation symmetry, (d/dx) wl
              // and thus grad wl must be also.
              gradwl[k] +=
                ((Q1Q2[0][0]*gradQlm[ m3][k][0] - Q1Q2[0][1]*gradQlm[ m3][k][1] +
                      (Q1Q2[1][0]*gradQlm[-m1][k][0] + Q1Q2[1][1]*gradQlm[-m1][k][1])*sgn +
                     Q1Q2[2][0]*gradQlm[ m2][k][0] - Q1Q2[2][1]*gradQlm[ m2][k][1]) * wlscale +
                  Q1Q2Q3[0] * gradwlscale[k]) * c;
            }
          } // closing m2 loop
        } // closing m1 loop

        // Set up expression/tree evaluator for the i,j atom pair

        // Parameter values
        double dudx[3];
        parm_vals2[qparmidx][0] = sqrt(ql)/nnei_scale;
        parm_vals2[wparmidx][0] = wl;

        for (int k = 0; k<3; k++) {
          // (d/dx) of parameter values
          if (ql > 0)
            parm_vals2[qparmidx][1] = 0.5/(sqrt(ql)*nnei_scale)*gradql[k];
          else
            parm_vals2[qparmidx][1] = 0;
          parm_vals2[wparmidx][1] = gradwl[k];
          const double uij = eval_tree_deriv(parsetree,nparms,parm_vals2);
          dudx[k] = parsetree->dval;

          fitmp[k] -= dudx[k];
          f[j][k] += dudx[k];
        }

        if (vflag_either)
          ev_tally_xyz(i,j,nlocal,newton_pair,0.0,0.0,
                       -dudx[0],-dudx[1],-dudx[2],
                       delx,dely,delz);

      } // closing if (rsq > 0)
    } // closing for (int jj ...)

    for (int k = 0; k<3; k++)
      f[i][k] += fitmp[k];
  }

  if (eflag_global)
    eng_vdwl += evdwl;

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairWQLFun::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(neighshort,maxshort,"pair:neighshort");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairWQLFun::settings(int narg, char **arg)
{
  if (narg < 4 || narg > 8) error->all(FLERR,"Illegal pair_style command");
  if (kokkosable)
    if (narg < 7 || narg > 8) error->all(FLERR,"Pair wqlfun/kk requires including w0, kappa, and renorm as args");

  // Without newton_pair, twice cut-off needed and complicated loop over neighbors' neighbors

  if (!force->newton_pair) error->all(FLERR,"PairWQLFun requires newton on");

  // double kappa,Wzero,rmin,rmax;

  cut_global = utils::numeric(FLERR,arg[0],false,lmp);
  lmax = utils::numeric(FLERR,arg[1],false,lmp);
  rmin = utils::numeric(FLERR,arg[2],false,lmp);
  if (narg > 4)
    wl0 = utils::numeric(FLERR,arg[4],false,lmp);
  if (narg > 5)
    kappa = utils::numeric(FLERR,arg[5],false,lmp);
  if (narg > 6)
    renorm = utils::numeric(FLERR,arg[6],false,lmp);
  if (narg > 7)
    chunksize = utils::numeric(FLERR,arg[7],false,lmp);

  if ((lmax % 2) != 0)
    error->all(FLERR,"lmax must be even in pair wqlfun");

  // Wl relevant variables and tables

  // Deallocate energy expression variables

  delete_tree(parsetree);
  if (parms)
    for (int i = 0; i<nparms; i++)
      delete[] parms[i];
  delete[] parms;
  delete[] parm_vals;
  delete[] parm_vals2;
  delete[] expression;

  delete[] yvec;
  delete[] zvec;
  delete[] Qlm;
  delete[] rsqrt;
  delete[] sqrtfact;
  delete[] Almvec;

  yvec = new double[(lmax+2)*(lmax+3)/2][2];
  zvec = new double[(lmax+2)*(lmax+3)/2][2];
  Qlm = new double[lmax+1][2];
  rsqrt = new double[2*lmax+3];
  sqrtfact = new double[2*(lmax+1)+3];
  Almvec = new double[lmax+1];

  rsqrt[0] = 1.0;
  for (int m=1; m<=lmax+1; m++) {
    rsqrt[2*m-1] = sqrt(1.0/(2*m-1));
    rsqrt[2*m]   = sqrt(1.0/(2*m));
  }

  sqrtfact[0] = 1.0;
  for (int m=0; m<=lmax+1; m++) {
    sqrtfact[2*m+1] = sqrtfact[2*m]   * sqrt((double) (2*m+1));
    sqrtfact[2*m+2] = sqrtfact[2*m+1] * sqrt((double) (2*m+2));
  }

  for (int m = 0; m<=lmax; m++)
    anmsub(lmax,m,&Almvec[m]);

  expression = new char[strlen(arg[3])+1];
  strcpy(expression,arg[3]);
  {
    const char *parmlist[] = {"w","q"};
    nparms = sizeof(parmlist)/sizeof(*parmlist);
    parms = new char*[nparms];
    for (int i = 0; i<nparms; i++) {
      parms[i] = new char[strlen(parmlist[i])+1];
      strcpy(parms[i],parmlist[i]);
    }

    for (int i = 0; i<nparms; i++) {
      // Silly loop, but just in case other parameters are added above
      // in some arbitary order, or they are loaded from the lammps
      // script.
      if (strcmp(parms[i],"w") == 0) wparmidx = i;
      if (strcmp(parms[i],"q") == 0) qparmidx = i;
    }

    parm_vals  = new double[nparms];
    parm_vals2 = new double[nparms][2];
  }

  parsetree = parse_string(expression,nparms,parms);

  rmax = cut_global;
  if (comm->me == 0) {
    printf("WQLFun Potential: rcut = %.3f, L = %d, rmin = %.3f, rmax = %.3f\n",
           cut_global,lmax,rmin,rmax);
    printf("  Energy expression (function of w and q) is %s\n",expression);
  }

  /* Initialize table of Wigner 3j symbols */ {
    w3jlist.clear();
    for (int m1 = -lmax; m1<=0; m1++)
      for (int m2 = 0; m2<=((-m1)>>1); m2++) {
        const int m3 = -(m1 + m2);
        // Loop enforces -lmax<=m1<=0<=m2<=m3<=lmax, and m1+m2+m3=0

        // For even lmax, W3j is invariant under permutation of
        // (m1,m2,m3) and (m1,m2,m3)->(-m1,-m2,-m3). The loop
        // structure enforces visiting only one member of each
        // such symmetry (invariance) group.

        // Determine number of elements in symmetry group of (m1,m2,m3)
        // Concise determination exploiting (m1,m2,m3) loop structure.
        int pfac;
        if (m1 == 0) pfac = 1; // m1 = m2 = m3 = 0
        else if (m2 == 0 || m2 == m3) {
          // reduced group when only 3 permutations, or sign inversion
          // is equivalent to permutation
          pfac = 6;
        } else pfac = 12; // 6 permutations * 2 signs

        w3jlist.push_back(w3j(lmax,m1,m2,m3) * pfac);
      }
  }

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j])
          cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairWQLFun::coeff(int narg, char **arg)
{
  if (narg != 2)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double cut_one = cut_global;
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairWQLFun::init_style()
{
  // need a full neighbor list

  auto req = neighbor->add_request(this, NeighConst::REQ_FULL);
  //if (cutflag) req->set_cutoff(cut_global + neighbor->skin);
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairWQLFun::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairWQLFun::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    cut[i][j] = cut_global;

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairWQLFun::plmallcompress(const int lmax, const double x, double plm[])
{

  double sine;
  int idx1,idx2,idx3;

  if (lmax < 0 || fabs(x) > 1) {
    error->one(FLERR,fmt::format("Input error to PairWQLFun plmall: "
                                 "Lmax = {}, x = {}, abs(x)-1 = {}",
                                 lmax,x,fabs(x)-1));
  }

  // Numerically stable computation (instead of sqrt(1-x*x)), when x is close to 1

  sine = sqrt((1+x)*(1-x));
  plm[0] = 1;
  for (int m=0; m<=lmax; m++) {
    // P(m,m)
    idx1 = m*(m+1)/2 + m;
    if (m > 0) plm[idx1] = -plm[(m-1)*m/2+(m-1)]*(2*m-1)*sine;

    // P(m+1,m)
    idx2 = (m+1)*(m+2)/2 + m;
    if (m < lmax) plm[idx2] = x*(2*m+1)*plm[idx1];

    for (int l=m+2; l<=lmax; l++) {
      // P(l,m)
      idx3 = l*(l+1)/2 + m;
      plm[idx3] = (x*(2*l-1)*plm[idx2] - (l+m-1)*plm[idx1])/(l-m);

      idx1 = idx2;
      idx2 = idx3;
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairWQLFun::ylmallcompress(const int lmax,
                                const double xhat, const double yhat, const double zhat,
                                double ylm[][2])
{
  // Dimensions: complex*16 ylm((lmax+1)*(lmax+2)/2)

  double plm[(lmax+1)*(lmax+2)/2];
  double expi[lmax+1][2];
  //const double im[2] = {0,1};

  double rxy2,rxyinv,c,s,f;
  int idx;

  rxy2 = xhat*xhat + yhat*yhat;
  if (rxy2 > 0.0) {
    rxyinv = 1.0/sqrt(rxy2);
    c = xhat*rxyinv;
    s = yhat*rxyinv;
  } else {
    c = 1.0;
    s = 0.0;
  }
  plmallcompress(lmax,zhat,plm);

  // ylm[0] = (1,0)
  ylm[0][0] = 1.0;
  ylm[0][1] = 0.0;

  expi[0][0] = 1.0;
  expi[0][1] = 0.0;

  for (int l=1; l<=lmax; l++) {
    //expi[l] = expi[l-1] * (c + im*s)
    expi[l][0] = expi[l-1][0]*c - expi[l-1][1]*s;
    expi[l][1] = expi[l-1][1]*c + expi[l-1][0]*s;

    f = 1.0;
    idx = l*(l+1)/2;
    ylm[idx][0] = f * plm[idx];
    ylm[idx][1] = 0.0;

    for (int m=1; m<=l; m++) {
      f = f * rsqrt[l-m+1]*rsqrt[l+m];
      ylm[idx + m][0] = f * plm[idx + m] * expi[m][0];
      ylm[idx + m][1] = f * plm[idx + m] * expi[m][1];
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairWQLFun::ylm2zlmcompress(const int lmax,const double ylm[][2],double zlm[][2])
{
  // Dimensions: complex*16 ylm((lmax+1)*(lmax+2)/2),zlm((lmax+1)*(lmax+2)/2)
  int k = 0;

  double lsign = 1;
  for (int l=0; l<=lmax; l++) {
    double sign = lsign;
    lsign = -lsign;
    for (int m=0; m<=l; m++) {
      //const double ainv = 1.0/Anm(lmax,m);
      const double ainv = sign*sqrtfact[l-m]*sqrtfact[l+m];
      zlm[k][0] = ylm[k][0]*ainv;
      zlm[k][1] = ylm[k][1]*ainv;
      k = k+1;
      sign = -sign;
    }
  }
}

/* ---------------------------------------------------------------------- */

double PairWQLFun::triangle_coeff(const int a, const int b, const int c) {
  return factorial(a+b-c)*factorial(a-b+c)*factorial(-a+b+c) / factorial(a+b+c+1);
}

/* ---------------------------------------------------------------------- */

double PairWQLFun::w3j(const int lmax, const int j1, const int j2, const int j3) {
  const int a = lmax, b = lmax, c = lmax;
  const int alpha = j1, beta = j2, gamma = j3;
  struct {
    double operator() (const int a, const int b, const int c,
		       const int alpha, const int beta,const int gamma,
		       const int t) {
      return factorial(t)*factorial(c-b+t+alpha)*factorial(c-a+t-beta) * factorial(a+b-c-t)*factorial(a-t-alpha)*factorial(b-t+beta);
    }
  } x;
  const double
    sgn = 1 - 2*((a-b-gamma)&1),
    g = sqrt(triangle_coeff(lmax,lmax,lmax)) * sqrt(factorial(a+alpha)*factorial(a-alpha)*
					   factorial(b+beta)*factorial(b-beta)*
					   factorial(c+gamma)*factorial(c-gamma));
  double s = 0;
  int t = 0;
  while(c-b+t+alpha < 0 || c-a+t-beta < 0) t++;
  //     ^^ t>=-j1       ^^ t>=j2
  while(1) {
    if (a+b-c-t < 0) break;   // t<=lmax
    if (a-t-alpha < 0) break; // t<=lmax-j1
    if (b-t+beta < 0) break;  // t<=lmax+j2
    const int m1t = 1 - 2*(t&1);
    s += m1t/x(lmax,lmax,lmax,alpha,beta,gamma,t);
    t++;
  }
  return sgn*g*s;
}

/* ---------------------------------------------------------------------- */

void* PairWQLFun::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "evdwl") == 0) return (void *) &evdwl;
  return nullptr;
}

