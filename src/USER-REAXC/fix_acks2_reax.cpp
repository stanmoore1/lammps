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
   Contributing author: Stan Moore (Sandia)
------------------------------------------------------------------------- */

#include "fix_acks2_reax.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "pair_reaxc.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "force.h"
#include "group.h"
#include "pair.h"
#include "respa.h"
#include "memory.h"
#include "citeme.h"
#include "error.h"
#include "reaxc_defs.h"
#include "reaxc_types.h"
#include "fix_efield.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define SQR(x) ((x)*(x))

static const char cite_fix_acks2_reax[] =
  "fix acks2/reax command:\n\n"
  "@Article{O'Hearn2020,\n"
  " author = {K. A. O'Hearn, A. Alperen, and H. M. Aktulga},\n"
  " title = {Fast Solvers for Charge Distribution Models on Shared Memory Platforms},\n"
  " journal = {SIAM J. Sci. Comput.},\n"
  " year =    2020,\n"
  " volume =  42,\n"
  " pages =   {1--22}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

FixACKS2Reax::FixACKS2Reax(LAMMPS *lmp, int narg, char **arg) :
  FixQEqReax(lmp, narg, arg)
{
  refcharge = NULL;

  bcut = NULL;

  X_diag = NULL;
  Xdia_inv = NULL;

  // BiCGStab
  g = NULL;
  q_hat = NULL;
  r_hat = NULL;
  y = NULL;
  z = NULL;

  // X matrix
  X.firstnbr = NULL;
  X.numnbrs = NULL;
  X.jlist = NULL;
  X.val = NULL;

  // Update comm sizes for this fix
  comm_forward = comm_reverse = 2;

  s_hist_X = s_hist_last = NULL;
}

/* ---------------------------------------------------------------------- */

FixACKS2Reax::~FixACKS2Reax()
{
  if (copymode) return;

  if (!reaxflag)
    memory->destroy(refcharge);

  memory->destroy(bcut);

  if (!reaxflag)
    memory->destroy(b_s_acks2);

  memory->destroy(s_hist_X);
  memory->destroy(s_hist_last);
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::post_constructor()
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_acks2_reax);

  memory->create(s_hist_last,2,nprev,"acks2/reax:s_hist_last");
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < nprev; ++j)
      s_hist_last[i][j] = 0.0;

  grow_arrays(atom->nmax);
  for (int i = 0; i < atom->nmax; i++)
    for (int j = 0; j < nprev; ++j)
      s_hist[i][j] = s_hist_X[i][j] = 0.0;

  pertype_parameters(pertype_option);
  if (dual_enabled)
    error->all(FLERR,"Dual keyword only supported with fix qeq/reax/omp");
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::pertype_parameters(char *arg)
{
  if (strcmp(arg,"reax/c") == 0) {
    reaxflag = 1;
    Pair *pair = force->pair_match("reax/c",0);
    if (pair == NULL) error->all(FLERR,"No pair reax/c for fix acks2/reax");

    int tmp;
    chi = (double *) pair->extract("chi",tmp);
    eta = (double *) pair->extract("eta",tmp);
    gamma = (double *) pair->extract("gamma",tmp);
    b_s_acks2 = (double *) pair->extract("b_s_acks2",tmp);
    refcharge = (double *) pair->extract("refcharge",tmp);
    double* bond_softness_ptr = (double *) pair->extract("bond_softness",tmp);
    if (!reaxc->refcharge_flag)
      error->all(FLERR,"Fix acks2/reax requires specifying refcharges in ReaxFF pair coeffs");

    if (chi == NULL || eta == NULL || gamma == NULL ||
        b_s_acks2 == NULL || refcharge == NULL || bond_softness_ptr == NULL)
      error->all(FLERR,
                 "Fix acks2/reax could not extract params from pair reax/c");
    bond_softness = *bond_softness_ptr;
    return;
  }

  int i,itype,ntypes,rv;
  double v1,v2,v3,v4,v5;
  FILE *pf;

  reaxflag = 0;
  ntypes = atom->ntypes;

  memory->create(chi,ntypes+1,"acks2/reax:chi");
  memory->create(eta,ntypes+1,"acks2/reax:eta");
  memory->create(gamma,ntypes+1,"acks2/reax:gamma");
  memory->create(b_s_acks2,ntypes+1,"acks2/reax:b_s_acks2");
  memory->create(refcharge,ntypes+1,"acks2/reax:refcharge");

  if (comm->me == 0) {
    if ((pf = fopen(arg,"r")) == NULL)
      error->one(FLERR,"Fix acks2/reax parameter file could not be found");

    rv = fscanf(pf,"%lg",&v1);
    if (rv != 1)
      error->one(FLERR,"Fix acks2/reax: Incorrect format of param file");
    bond_softness = v1;

    for (i = 1; i <= ntypes && !feof(pf); i++) {
      rv = fscanf(pf,"%d %lg %lg %lg %lg",&itype,&v1,&v2,&v3,&v4,&v5);
      if (rv != 6)
        error->one(FLERR,"Fix acks2/reax: Incorrect format of param file");
      if (itype < 1 || itype > ntypes)
        error->one(FLERR,"Fix acks2/reax: invalid atom type in param file");
      chi[itype] = v1;
      eta[itype] = v2;
      gamma[itype] = v3;
      b_s_acks2[itype] = v4;
      refcharge[itype] = v5;
    }
    if (i <= ntypes) error->one(FLERR,"Invalid param file for fix acks2/reax");
    fclose(pf);
  }

  MPI_Bcast(&chi[1],ntypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&eta[1],ntypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&gamma[1],ntypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&b_s_acks2[1],ntypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&refcharge[1],ntypes,MPI_DOUBLE,0,world);
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::allocate_storage()
{
  nmax = atom->nmax;
  int size = nmax*2 + 2;

  // 0 to nn-1: owned atoms related to H matrix
  // nn to NN-1: ghost atoms related to H matrix
  // NN to NN+nn-1: owned atoms related to X matrix
  // NN+nn to 2*NN-1: ghost atoms related X matrix
  // 2*NN to 2*NN+1: last two rows, owned by proc 0

  memory->create(s,size,"acks2:s");
  memory->create(b_s,size,"acks2:b_s");

  memory->create(Hdia_inv,nmax,"acks2:Hdia_inv");
  memory->create(chi_field,nmax,"acks2:chi_field");

  memory->create(X_diag,nmax,"acks2:X_diag");
  memory->create(Xdia_inv,nmax,"acks2:Xdia_inv");

  memory->create(p,size,"acks2:p");
  memory->create(q,size,"acks2:q");
  memory->create(r,size,"acks2:r");
  memory->create(d,size,"acks2:d");

  memory->create(g,size,"acks2:g");
  memory->create(q_hat,size,"acks2:q_hat");
  memory->create(r_hat,size,"acks2:r_hat");
  memory->create(y,size,"acks2:y");
  memory->create(z,size,"acks2:z");
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::deallocate_storage()
{
  FixQEqReax::deallocate_storage();

  memory->destroy( X_diag );
  memory->destroy( Xdia_inv );

  memory->destroy( g );
  memory->destroy( q_hat );
  memory->destroy( r_hat );
  memory->destroy( y );
  memory->destroy( z );
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::allocate_matrix()
{
  FixQEqReax::allocate_matrix();

  X.n = n_cap;
  X.m = m_cap;
  memory->create(X.firstnbr,n_cap,"acks2:X.firstnbr");
  memory->create(X.numnbrs,n_cap,"acks2:X.numnbrs");
  memory->create(X.jlist,m_cap,"acks2:X.jlist");
  memory->create(X.val,m_cap,"acks2:X.val");
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::deallocate_matrix()
{
  FixQEqReax::deallocate_matrix();

  memory->destroy( X.firstnbr );
  memory->destroy( X.numnbrs );
  memory->destroy( X.jlist );
  memory->destroy( X.val );
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::init()
{
  FixQEqReax::init();

  init_bondcut();
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::init_bondcut()
{
  int i,j;
  int ntypes;

  ntypes = atom->ntypes;
  if (bcut == NULL)
    memory->create(bcut,ntypes+1,ntypes+1,"acks2:bondcut");

  for (i = 1; i <= ntypes; ++i)
    for (j = 1; j <= ntypes; ++j) {
      bcut[i][j] = 0.5*(b_s_acks2[i] + b_s_acks2[j]);
    }
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::init_storage()
{
  if (field_flag)
    get_chi_field();

  for (int i = 0; i < NN; i++) {
    b_s[i] = -chi[atom->type[i]] - chi_field[i];
    s[i] = 0.0;
  }

  // Reference charges

  for (int i = 0; i < NN; i++) {
    b_s[NN + i] = refcharge[atom->type[i]];
    s[NN + i] = 0.0;
  }

  for (int i = 0; i < 2; i++) {
    b_s[2*NN + i] = 0.0;
    s[2*NN + i] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::pre_force(int /*vflag*/)
{
  double t_start, t_end;

  if (update->ntimestep % nevery) return;
  if (comm->me == 0) t_start = MPI_Wtime();

  int n = atom->nlocal;

  if (reaxc) {
    nn = reaxc->list->inum;
    NN = reaxc->list->inum + reaxc->list->gnum;
    ilist = reaxc->list->ilist;
    numneigh = reaxc->list->numneigh;
    firstneigh = reaxc->list->firstneigh;
  } else {
    nn = list->inum;
    NN = list->inum + list->gnum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
  }

  // grow arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) reallocate_storage();
  if (n > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
    reallocate_matrix();

  if (field_flag)
    get_chi_field();

  init_matvec();

  matvecs = BiCGStab(b_s, s); // BiCGStab on s - parallel

  calculate_Q();

  if (comm->me == 0) {
    t_end = MPI_Wtime();
    qeq_time = t_end - t_start;
  }
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::init_matvec()
{
  /* fill-in H matrix */
  compute_H();

  /* fill-in X matrix */
  compute_X();
  pack_flag = 4;
  comm->reverse_comm_fix(this); //Coll_Vector( X_diag );

  int ii, i;

  for (int i = 0; i < nn; i++) {
    if (X_diag[i] == 0.0)
      Xdia_inv[i] = 1.0;
    else
      Xdia_inv[i] = 1.0 / X_diag[i];
  }

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {

      /* init pre-conditioner for H and init solution vectors */
      Hdia_inv[i] = 1. / eta[ atom->type[i] ];
      b_s[i] = -chi[ atom->type[i] ] - chi_field[i];
      b_s[NN+i] = refcharge[ atom->type[i] ];

      /* cubic extrapolation for s from previous solutions */
      s[i] = 4*(s_hist[i][0]+s_hist[i][2])-(6*s_hist[i][1]+s_hist[i][3]);
      s[NN+i] = 4*(s_hist_X[i][0]+s_hist_X[i][2])-(6*s_hist_X[i][1]+s_hist_X[i][3]);
    }
  }

  // last two rows
  if (comm->me == 0) {
    for (i = 0; i < 2; i++) {
      b_s[2*NN+i] = 0.0;
      s[2*NN+i] = 4*(s_hist_last[i][0]+s_hist_last[i][2])-(6*s_hist_last[i][1]+s_hist_last[i][3]);
    }
  }

  pack_flag = 2;
  comm->forward_comm_fix(this); //Dist_vector( s );
  more_forward_comm(s);
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::compute_X()
{
  int jnum;
  int i, j, ii, jj, flag;
  double dx, dy, dz, r_sqr;
  const double SMALL = 0.0001;

  int *type = atom->type;
  tagint *tag = atom->tag;
  double **x = atom->x;
  int *mask = atom->mask;

  memset(X_diag,0.0,atom->nmax*sizeof(double));

  // fill in the X matrix
  m_fill = 0;
  r_sqr = 0;
  for (ii = 0; ii < nn; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      jlist = firstneigh[i];
      jnum = numneigh[i];
      X.firstnbr[i] = m_fill;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        dx = x[j][0] - x[i][0];
        dy = x[j][1] - x[i][1];
        dz = x[j][2] - x[i][2];
        r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

        flag = 0;
        if (r_sqr <= SQR(swb)) {
          if (j < atom->nlocal) flag = 1;
          else if (tag[i] < tag[j]) flag = 1;
          else if (tag[i] == tag[j]) {
            if (dz > SMALL) flag = 1;
            else if (fabs(dz) < SMALL) {
              if (dy > SMALL) flag = 1;
              else if (fabs(dy) < SMALL && dx > SMALL)
                flag = 1;
            }
          }
        }

        if (flag) {
          double bcutoff = bcut[type[i]][type[j]];
          double bcutoff2 = bcutoff*bcutoff;
          if (r_sqr <= bcutoff2) {
            X.jlist[m_fill] = j;
            double X_val = calculate_X(sqrt(r_sqr), bcutoff);
            X.val[m_fill] = X_val;
            X_diag[i] -= X_val;
            X_diag[j] -= X_val;
            m_fill++;
          }
        }
      }

      X.numnbrs[i] = m_fill - X.firstnbr[i];
    }
  }

  if (m_fill >= X.m) {
    char str[128];
    sprintf(str,"X matrix size has been exceeded: m_fill=%d X.m=%d\n",
             m_fill, X.m);
    error->warning(FLERR,str);
    error->all(FLERR,"Fix acks2/reax has insufficient ACKS2 matrix size");
  }
}

/* ---------------------------------------------------------------------- */

double FixACKS2Reax::calculate_X( double r, double bcut)
{
  double d = r/bcut;
  double d3 = d*d*d;
  double omd = 1.0 - d;
  double omd2 = omd*omd;
  double omd6 = omd2*omd2*omd2;

  return bond_softness*d3*omd6;
}

/* ---------------------------------------------------------------------- */

int FixACKS2Reax::BiCGStab( double *b, double *x)
{
  int  i, j;
  double tmp, alpha, beta, omega, sigma, rho, rho_old, rnorm, bnorm;
  double sig_old, sig_new;

  int jj;

  sparse_matvec_acks2( &H, &X, x, d);
  pack_flag = 1;
  comm->reverse_comm_fix(this); //Coll_Vector( d );
  more_reverse_comm(d);

  vector_sum( r , 1.,  b, -1., d, nn);
  bnorm = parallel_norm( b, nn);
  rnorm = parallel_norm( r, nn);

  if ( bnorm == 0.0 ) bnorm = 1.0;
  vector_copy( r_hat, r, nn);
  omega = 1.0;
  rho = 1.0;

  for (i = 1; i < imax && rnorm / bnorm > tolerance; ++i) {
    rho = parallel_dot( r_hat, r, nn);
    if (rho == 0.0) break;

    if (i > 1) {
      beta = (rho / rho_old) * (alpha / omega);
      vector_sum( q , 1., p, -omega, z, nn);
      vector_sum( p , 1., r, beta, q, nn);     
    } else {
      vector_copy( p, r, nn);
    }

    // pre-conditioning
    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit) {
        d[j] = p[j] * Hdia_inv[j];
        d[NN+j] = p[NN+j] * Xdia_inv[j];
      }
    }
    // last two rows
    if (comm->me == 0) {
      d[2*NN] = p[2*NN];
      d[2*NN + 1] = p[2*NN + 1];
    }

    pack_flag = 1;
    comm->forward_comm_fix(this); //Dist_vector( d );
    more_forward_comm(d);
    sparse_matvec_acks2( &H, &X, d, z );
    pack_flag = 2;
    comm->reverse_comm_fix(this); //Coll_vector( z );
    more_reverse_comm(z);

    tmp = parallel_dot( r_hat, z, nn);
    alpha = rho / tmp;

    vector_sum( q , 1., r, -alpha, z, nn);

    tmp = parallel_dot( q, q, nn);

    // early convergence check
    if (tmp < tolerance) {
      vector_add( x, alpha, d, nn);
      break;
    }

    // pre-conditioning
    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit) {
        q_hat[j] = q[j] * Hdia_inv[j];
        q_hat[NN+j] = q[NN+j] * Xdia_inv[j];
      }
    }
    // last two rows
    if (comm->me == 0) {
      q_hat[2*NN] = q[2*NN];
      q_hat[2*NN + 1] = q[2*NN + 1];
    }

    pack_flag = 3;
    comm->forward_comm_fix(this); //Dist_vector( q_hat );
    more_forward_comm(q_hat);
    sparse_matvec_acks2( &H, &X, q_hat, y );
    pack_flag = 3;
    comm->reverse_comm_fix(this); //Dist_vector( y );
    more_reverse_comm(y);

    sigma = parallel_dot( y, q, nn);
    tmp = parallel_dot( y, y, nn);
    omega = sigma / tmp;

    vector_sum( g , alpha, d, omega, q_hat, nn);
    vector_add( x, 1., g, nn);
    vector_sum( r , 1., q, -omega, y, nn);

    rnorm = parallel_norm( r, nn);
    if (omega == 0) break;
    rho_old = rho;
  }

  if (comm->me == 0) {
    if (omega == 0 || rho == 0) {
      char str[128];
      sprintf(str,"Fix acks2/reax BiCGStab numerical breakdown, omega = %g, rho = %g",omega,rho);
      error->warning(FLERR,str);
    } else if (i >= imax) {
      char str[128];
      sprintf(str,"Fix acks2/reax BiCGStab convergence failed after %d iterations "
              "at " BIGINT_FORMAT " step",i,update->ntimestep);
      error->warning(FLERR,str);
    }
  }

  return i;
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::sparse_matvec_acks2( sparse_matrix *H, sparse_matrix *X, double *x, double *b)
{
  int i, j, itr_j;
  int ii;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      b[i] = eta[ atom->type[i] ] * x[i];
      b[NN + i] = X_diag[i] * x[NN + i];
    }
  }

  for (ii = nn; ii < NN; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      b[i] = 0;
      b[NN + i] = 0;
  }
  // last two rows
  b[2*NN] = 0;
  b[2*NN + 1] = 0;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      // H Matrix
      for (itr_j=H->firstnbr[i]; itr_j<H->firstnbr[i]+H->numnbrs[i]; itr_j++) {
        j = H->jlist[itr_j];
        b[i] += H->val[itr_j] * x[j];
        b[j] += H->val[itr_j] * x[i];
      }

      // X Matrix
      for (itr_j=X->firstnbr[i]; itr_j<X->firstnbr[i]+X->numnbrs[i]; itr_j++) {
        j = X->jlist[itr_j];
        b[NN + i] += X->val[itr_j] * x[NN + j];
        b[NN + j] += X->val[itr_j] * x[NN + i];
      }

      // Identity Matrix
      b[NN + i] += x[i];
      b[i] += x[NN + i];

      // Second-to-last row/column
      b[2*NN] += x[NN + i];
      b[NN + i] += x[2*NN];

      // Last row/column
      b[2*NN + 1] += x[i];
      b[i] += x[2*NN + 1];
    }
  }

}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::calculate_Q()
{
  int i, k;

  for (int ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {

      /* backup s */
      for (k = nprev-1; k > 0; --k) {
        s_hist[i][k] = s_hist[i][k-1];
        s_hist_X[i][k] = s_hist_X[i][k-1];
      }
      s_hist[i][0] = s[i];
      s_hist_X[i][0] = s[NN+i];
    }
  }
  // last two rows
  if (comm->me == 0) {
    for (int i = 0; i < 2; ++i) {
      for (k = nprev-1; k > 0; --k)
        s_hist_last[i][k] = s_hist_last[i][k-1];
      s_hist_last[i][0] = s[2*NN+i];
    }
  }

  pack_flag = 2;
  comm->forward_comm_fix(this); //Dist_vector( s );

  for (int ii = 0; ii < NN; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      atom->q[i] = s[i];
  }
}

/* ---------------------------------------------------------------------- */

int FixACKS2Reax::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
  int m = 0;

  if (pack_flag == 1) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = d[j];
      buf[m++] = d[NN+j];
    }
  } else if (pack_flag == 2) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = s[j];
      buf[m++] = s[NN+j];
    }
  } else if (pack_flag == 3) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = q_hat[j];
      buf[m++] = q_hat[NN+j];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  int last = first + n;
  m = 0;

  if (pack_flag == 1) {
    for(i = first; i < last; i++) {
      d[i] = buf[m++];
      d[NN+i] = buf[m++];
    }
  } else if (pack_flag == 2) {
    for(i = first; i < last; i++) {
      s[i] = buf[m++];
      s[NN+i] = buf[m++];
    }
  } else if (pack_flag == 3) {
    for(i = first; i < last; i++) {
      q_hat[i] = buf[m++];
      q_hat[NN+i] = buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixACKS2Reax::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m;
  m = 0;
  int last = first + n;

  if (pack_flag == 1) {
    for(i = first; i < last; i++) {
      buf[m++] = d[i];
      buf[m++] = d[NN+i];
    }
  } else if (pack_flag == 2) {
    for(i = first; i < last; i++) {
      buf[m++] = z[i];
      buf[m++] = z[NN+i];
    }
  } else if (pack_flag == 3) {
    for(i = first; i < last; i++) {
      buf[m++] = y[i];
      buf[m++] = y[NN+i];
    }
  } else if (pack_flag == 4) {
    for(i = first; i < last; i++)
      buf[m++] = X_diag[i];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::unpack_reverse_comm(int n, int *list, double *buf)
{
  int j;
  int m = 0;
  if (pack_flag == 1) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      d[j] += buf[m++];
      d[NN+j] += buf[m++];
    }
  } else if (pack_flag == 2) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      z[j] += buf[m++];
      z[NN+j] += buf[m++];
    }
  } else if (pack_flag == 3) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      y[j] += buf[m++];
      y[NN+j] += buf[m++];
    }
  } else if (pack_flag == 4) {
    for(int i = 0; i < n; i++) {
      j = list[i];
     X_diag[j] += buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 broadcasts last two rows of vector to everyone else
------------------------------------------------------------------------- */

void FixACKS2Reax::more_forward_comm(double *vec)
{
  MPI_Bcast(&vec[2*NN],2,MPI_DOUBLE,0,world);
}

/* ----------------------------------------------------------------------
   reduce last two rows of vector and give to proc 0
------------------------------------------------------------------------- */

void FixACKS2Reax::more_reverse_comm(double *vec)
{
  if (comm->me == 0)
    MPI_Reduce(MPI_IN_PLACE,&vec[2*NN],2,MPI_DOUBLE,MPI_SUM,0,world);
  else
    MPI_Reduce(&vec[2*NN],NULL,2,MPI_DOUBLE,MPI_SUM,0,world);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixACKS2Reax::memory_usage()
{
  double bytes;

  int size = 2*nmax + 2;

  bytes = size*nprev * sizeof(double); // s_hist
  bytes += nmax*4 * sizeof(double); // storage
  bytes += size*11 * sizeof(double); // storage
  bytes += n_cap*4 * sizeof(int); // matrix...
  bytes += m_cap*2 * sizeof(int);
  bytes += m_cap*2 * sizeof(double);

  return bytes;
}

/* ----------------------------------------------------------------------
   allocate solution history array
------------------------------------------------------------------------- */

void FixACKS2Reax::grow_arrays(int nmax)
{
  memory->grow(s_hist,nmax,nprev,"acks2:s_hist");
  memory->grow(s_hist_X,nmax,nprev,"acks2:s_hist_X");
}

/* ----------------------------------------------------------------------
   copy values within solution history array
------------------------------------------------------------------------- */

void FixACKS2Reax::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int m = 0; m < nprev; m++) {
    s_hist[j][m] = s_hist[i][m];
    s_hist_X[j][m] = s_hist_X[i][m];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixACKS2Reax::pack_exchange(int i, double *buf)
{
  for (int m = 0; m < nprev; m++) buf[m] = s_hist[i][m];
  for (int m = 0; m < nprev; m++) buf[nprev+m] = s_hist_X[i][m];
  return nprev*2;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixACKS2Reax::unpack_exchange(int nlocal, double *buf)
{
  for (int m = 0; m < nprev; m++) s_hist[nlocal][m] = buf[m];
  for (int m = 0; m < nprev; m++) s_hist_X[nlocal][m] = buf[nprev+m];
  return nprev*2;
}

/* ---------------------------------------------------------------------- */

double FixACKS2Reax::parallel_norm( double *v, int n)
{
  int  i;
  double my_sum, norm_sqr;

  int ii;

  my_sum = 0.0;
  norm_sqr = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_sum += SQR( v[i]);
      my_sum += SQR( v[NN+i]);
    }
  }

  // last two rows
  if (comm->me == 0) {
    my_sum += SQR( v[2*NN]);
    my_sum += SQR( v[2*NN + 1]);
  }

  MPI_Allreduce( &my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world);

  return sqrt( norm_sqr);
}

/* ---------------------------------------------------------------------- */

double FixACKS2Reax::parallel_dot( double *v1, double *v2, int n)
{
  int  i;
  double my_dot, res;

  int ii;

  my_dot = 0.0;
  res = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_dot += v1[i] * v2[i];
      my_dot += v1[NN+i] * v2[NN+i];
    }
  }

  // last two rows
  if (comm->me == 0) {
    my_dot += v1[2*NN] * v2[2*NN];
    my_dot += v1[2*NN + 1] * v2[2*NN + 1];
  }

  MPI_Allreduce( &my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ---------------------------------------------------------------------- */

double FixACKS2Reax::parallel_vector_acc( double *v, int n)
{
  int  i;
  double my_acc, res;

  int ii;

  my_acc = 0.0;
  res = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_acc += v[i];
      my_acc += v[NN+i];
    }
  }

  // last two rows
  if (comm->me == 0) {
    my_acc += v[2*NN];
    my_acc += v[2*NN + 1];
  }

  MPI_Allreduce( &my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::vector_sum( double* dest, double c, double* v,
                                double d, double* y, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] = c * v[kk] + d * y[kk];
      dest[NN + kk] = c * v[NN + kk] + d * y[NN + kk];
    }
  }

  // last two rows
  if (comm->me == 0) {
    dest[2*NN] = c * v[2*NN] + d * y[2*NN];
    dest[2*NN + 1] = c * v[2*NN + 1] + d * y[2*NN + 1];
  }
}

/* ---------------------------------------------------------------------- */

void FixACKS2Reax::vector_add( double* dest, double c, double* v, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] += c * v[kk];
      dest[NN + kk] += c * v[NN + kk];
    }
  }

  // last two rows
  if (comm->me == 0) {
    dest[2*NN] += c * v[2*NN];
    dest[2*NN + 1] += c * v[2*NN + 1];
  }
}


/* ---------------------------------------------------------------------- */

void FixACKS2Reax::vector_copy( double* dest, double* v, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] = v[kk];
      dest[NN + kk] = v[NN + kk];
    }
  }

  // last two rows
  if (comm->me == 0) {
    dest[2*NN] = v[2*NN];
    dest[2*NN + 1] = v[2*NN + 1];
  }
}
