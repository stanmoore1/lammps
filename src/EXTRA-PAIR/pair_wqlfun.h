/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */


/*
   Written by Tomas Oppelstrup, at Lawrence Livermore National Laboratory.
   Implements a potential based on the Wl and Ql order parameters.

   The structure of this file is based on pair_lj_cut.cpp by P. Crozier.
*/


#ifdef PAIR_CLASS

PairStyle(wqlfun,PairWQLFun)

#else

#ifndef LMP_PAIR_WQLFun_H
#define LMP_PAIR_WQLFun_H

#include "pair.h"

#include "stringfunction.h"

namespace LAMMPS_NS {

  //class ComputeWQLFun;
class PairWQLFun : public Pair {
 public:
  PairWQLFun(class LAMMPS *);
  virtual ~PairWQLFun();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  void init_list(int, class NeighList *);
  virtual double init_one(int, int);

 protected:
  int lmax,chunksize;
  double wl0,kappa,renorm;
  double cut_global,evdwl;
  double **cut;
  double (*yvec)[2];
  double (*zvec)[2];
  double (*Qlm)[2];
  double *rsqrt;
  double *sqrtfact;
  double *Almvec;

  int maxshort;       // size of short neighbor list array
  int *neighshort;    // short neighbor list array

  double rmin,rmax;
  std::vector<double> w3jlist;

  int nparms,qparmidx,wparmidx;
  char *expression,**parms;
  double *parm_vals,(*parm_vals2)[2];
  node *parsetree;

  virtual void allocate();
  void *extract(const char *, int &);

  // Compute all Plm for lmax<=lmax, 0<=m<=lmax

  void plmallcompress(const int lmax, const double x, double plm[/* (lmax+1)*(lmax+2)/2 */]);

  // Compute all Ylm for lmax<=lmax, 0<=m<=lmax, using Greegard's normalization
  // The vector (xhat,yhat,zhat) is assumed to be a point on the unit sphere,
  //  i.e. sqrt(xhat*xhat + yhat*yhat + zhat*zhat) == 1

  void ylmallcompress(const int lmax,
		      const double xhat,const double yhat,const double zhat,
		      double ylm[/* (lmax+1)*(lmax+2)/2 */][2]);

  void ylm2zlmcompress(const int lmax,
                       const double ylm[/* (lmax+1)*(lmax+2)/2 */][2],
                       double zlm[/* (lmax+1)*(lmax+2)/2 */][2]);

  double fsmooth(double r, double df[1]) {
    df[0] = 0.0;
    if(r < rmin) return 1.0;
    else if(r >= rmax) return 0.0;
    else {
      const double
        scale = M_PI/(rmax - rmin),
        c = cos((r-rmin)*scale),
        s = sin((r-rmin)*scale);
      df[0] = -0.5*s*scale;
      return 0.5*(c + 1.0);
    }
  }

  inline double fact(int n) {
    double f = 1;
    for(int i = 1; i<=n; i++)
      f = f*i;
    return f;
  }

  inline double Anm(int n, int m) {
    const double x = 1.0 / sqrt(fact(n-m)*fact(n+m));
    if (((n+m) & 1) == 1)
      return -x;
    else
      return x;
  }

  inline void anmsub(int n, int m, double a[1]) {
    a[0] = Anm(n,m);
  }

  double triangle_coeff(const int a, const int b, const int c);
  double w3j(const int L, const int j1, const int j2, const int j3);

  static inline void zlmderiv1compress(const int lmax, const int m,
                                       const double xn, const double yn, const double zn,
                                       const double ylm[][2], double Yval[2], double DY[3][2]) {
    // Dimensions: complex*16 ylm((lmax+2)*(lmax+3)/2),Yval,DY(3)

    const double half = 0.5;
  #define Yr(lmax,m) ylm[(lmax)*((lmax)+1)/2 + (m)][0]
  #define Yi(lmax,m) ylm[(lmax)*((lmax)+1)/2 + (m)][1]

    Yval[0] = Yr(lmax,m);
    Yval[1] = Yi(lmax,m);
    if(m == 0) {
      // DY(1) = (lmax+1)*xn*Y(lmax,m) + realpart(Y(lmax+1,1))
      DY[0][0] = (lmax+1)*xn*Yr(lmax,m) + Yr(lmax+1,1);
      DY[0][1] = (lmax+1)*xn*Yi(lmax,m);

      // DY(2) = (lmax+1)*yn*Y(lmax,m) + imagpart(Y(lmax+1,1))
      DY[1][0] = (lmax+1)*yn*Yr(lmax,m) + Yi(lmax+1,1);
      DY[1][1] = (lmax+1)*yn*Yi(lmax,m);
    } else {
      DY[0][0] = (lmax+1)*xn*Yr(lmax,m) +
        ( Yr(lmax+1,m+1) - Yr(lmax+1,m-1) )*half;
      DY[0][1] = (lmax+1)*xn*Yi(lmax,m) +
        ( Yi(lmax+1,m+1) - Yi(lmax+1,m-1) )*half;
      DY[1][0] = (lmax+1)*yn*Yr(lmax,m) +
        ( Yi(lmax+1,m+1) + Yi(lmax+1,m-1) )*half;
      DY[1][1] = (lmax+1)*yn*Yi(lmax,m) -
        ( Yr(lmax+1,m+1) + Yr(lmax+1,m-1) )*half;
    }
    // DY(3) = (lmax+1)*zn*Y(lmax,m) + Y(lmax+1,m)
    DY[2][0] = (lmax+1)*zn*Yr(lmax,m) + Yr(lmax+1,m);
    DY[2][1] = (lmax+1)*zn*Yi(lmax,m) + Yi(lmax+1,m);
  #undef Yr
  #undef Yi
  }

  /*
protected:
  friend class ComputeWQLFun;
  ComputeWQLFun *wl_compute_obj;
  */
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
