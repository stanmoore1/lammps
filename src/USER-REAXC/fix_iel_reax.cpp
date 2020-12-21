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
   Contributing author: Itai Leven (UC Berkeley)
------------------------------------------------------------------------- */

#include "fix_iel_reax.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <random>
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
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define EV_TO_KCAL_PER_MOL 14.4
#define ATOMIC_TO_REAL 23.06
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

static const char cite_fix_iel_reax[] =
  "fix iel/reax command:\n\n"
  "@misc{Tan2020stochastic,\n"
  " title={Stochastic Constrained Extended System Dynamics for Solving Charge Equilibration Models},\n"
  " author={Songchen Tan and Itai Leven and Dong An and Lin Lin and Teresa Head-Gordon},\n"
  " year={2020},\n"
  " eprint={2005.10736},\n"
  " archivePrefix={arXiv},\n"
  " primaryClass={physics.comp-ph}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

FixIELReax::FixIELReax(LAMMPS *lmp, int narg, char **arg) :
  FixQEqReax(lmp, narg, arg)
{
  t_s_flag     = utils::inumeric(FLERR,arg[8],false,lmp); //t_s_flag=1:solve for t and s t_s_flag=0 q solve for q (s only)
  Precon_flag  = utils::inumeric(FLERR,arg[9],false,lmp); //if=0 no CG Pre conditioning://if =1 with CG Pre conditioning
  i_tolerance  = utils::inumeric(FLERR,arg[10],false,lmp);
  thermo_flag  = utils::inumeric(FLERR,arg[11],false,lmp);
  tautemp_aux  = utils::numeric(FLERR,arg[12],false,lmp); //100000.0; //t_S //1000000    //100000
  kelvin_aux_t = utils::numeric(FLERR,arg[13],false,lmp);  //0.000001; //t_S//0.00000001
  kelvin_aux_s = utils::numeric(FLERR,arg[14],false,lmp); //0.00000001; //t_s//0.0000001  //0.0000001
  Omega_t      = utils::numeric(FLERR,arg[15],false,lmp);
  Omega_s      = utils::numeric(FLERR,arg[16],false,lmp);

  if(comm->me == 0)
    printf("\niEL_Scf params:\n t_s_flag= %i Precon_flag= %i thermo_flag= %i tau= %.4f T0_t= %.10f T0_s= %.10f Omega_t= %.5f Omega_s=%.5f\n\n",t_s_flag,Precon_flag,thermo_flag,tautemp_aux,kelvin_aux_t,kelvin_aux_s,Omega_t,Omega_s);

  for (int i = 0; i < 5; i++) {
    tgnhaux[i] = 0.0;
    tvnhaux[i] = 0.0;
    tnhaux[i] = 0.0;
    sgnhaux[i] = 0.0;
    svnhaux[i] = 0.0;
    snhaux[i] = 0.0;
  }

  t_EL_Scf = nullptr;
  vt_EL_Scf = nullptr;
  at_EL_Scf = nullptr;
  s_EL_Scf = nullptr;
  vs_EL_Scf = nullptr;
  as_EL_Scf = nullptr;
}

/* ---------------------------------------------------------------------- */

FixIELReax::~FixIELReax()
{
  if (copymode) return;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::post_constructor()
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_iel_reax);

  grow_arrays(atom->nmax);

  pertype_parameters(pertype_option);
  if (dual_enabled)
    error->all(FLERR,"Dual keyword only supported with fix qeq/reax/omp");
}

/* ---------------------------------------------------------------------- */

int FixIELReax::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::init()
{
  FixQEqReax::init();

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double qterm_aux_t;
  double qterm_aux_s;

  vChi_eq_iEL_Scf = 0.0;
  aChi_eq_iEL_Scf = 0.0; // add by Itai for the iEL_ScF

  for (int i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {
      vs_EL_Scf[i] = 0.0;
      as_EL_Scf[i] = 0.0; // add by Itai for the iEL_ScF
      vt_EL_Scf[i] = 0.0;
      at_EL_Scf[i] = 0.0; // add by Itai for the iEL_ScF
    }
  }

  qterm_aux_t = kelvin_aux_t * tautemp_aux * tautemp_aux;
  qterm_aux_s = kelvin_aux_s * tautemp_aux * tautemp_aux;

  for (int i=0;i < 5; i++)
    {
      if (tgnhaux[i]==0.0)tgnhaux[i] = qterm_aux_t;
      tvnhaux[i] = 0.0;
      if (tnhaux[i]==0.0)tnhaux[i]  = qterm_aux_t;
      if (sgnhaux[i]==0.0)sgnhaux[i] = qterm_aux_s;
      svnhaux[i] = 0.0;
      if (snhaux[i]==0.0)snhaux[i]  = qterm_aux_s;
    }


  b_last = 0.0;   
  x_last = x_hist_2 = x_hist_1 = x_hist_0 = 0.0;
  q_last = 0.0;
  r_last = 0.0;
  d_last = 0.0;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::init_storage()
{
  for (int i = 0; i < NN; i++) {
    if(Precon_flag)
      Hdia_inv[i] = 1. / eta[atom->type[i]];
    else
      Hdia_inv[i] = 1.0;

    b_s[i] = -chi[atom->type[i]];
    b_t[i] = -1.0;
    b_prc[i] = 0;
    b_prm[i] = 0;
    s[i] = t[i] = 0;
  }
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixIELReax::initial_integrate(int /*vflag*/)
{
  double dtfm;

  // update v and x of atoms in group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double term = 2.0/(dtv*dtv);
  double dt = dtv;
  double dt_2 = 0.5*dtv;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      vt_EL_Scf[i] += at_EL_Scf[i]*dt_2*Omega_t;
      t_EL_Scf[i]  += vt_EL_Scf[i]*dtv*Omega_t;    
      vs_EL_Scf[i] += as_EL_Scf[i]*dt_2*Omega_s;
      s_EL_Scf[i]  += vs_EL_Scf[i]*dtv*Omega_s;
    }

  if (t_s_flag == 0) {
    vChi_eq_iEL_Scf = vChi_eq_iEL_Scf + aChi_eq_iEL_Scf*dt_2;
    Chi_eq_iEL_Scf  = Chi_eq_iEL_Scf + vChi_eq_iEL_Scf*dt;    
  }

  if (thermo_flag == 1)
    Nose_Hoover();
  else if (thermo_flag == 2)
    Berendersen();
}

/* ---------------------------------------------------------------------- */

void FixIELReax::pre_force(int /*vflag*/)
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

  init_matvec();

  matvecs_s = CG(b_s, s);       // CG on s - parallel
  if(t_s_flag)
    matvecs_t = CG(b_t, t);       // CG on t - parallel
  else
    matvecs_t = 0;

  matvecs = matvecs_s + matvecs_t;
  printf("matvecs %i %i\n",matvecs_s,matvecs_t);

  calculate_Q();

  if (comm->me == 0) {
    t_end = MPI_Wtime();
    qeq_time = t_end - t_start;
  }
}

/* ---------------------------------------------------------------------- */

void FixIELReax::init_matvec()
{ 
  /* fill-in H matrix */
  compute_H();
  
  r_last = b_last = q_last = d_last = 0.0;
  
  int ii, i;
 
  if (Precon_flag) {
    for (ii = 0; ii < nn; ++ii) {
      i = ilist[ii];
      if (atom->mask[i] & groupbit) {
        /* init pre-conditioner for H and init solution vectors */Hdia_inv[i] = 1. / eta[ atom->type[i] ];
        Hdia_inv[i] = 1. / eta[ atom->type[i] ];
        b_s[i]      = -chi[ atom->type[i] ];
        b_t[i]      = -1.0;
        s[i] = s_EL_Scf[i];
        t[i] = t_EL_Scf[i];
      }
    }
  } else {
    for (ii = 0; ii < nn; ++ii) {
      i = ilist[ii];
      if (atom->mask[i] & groupbit) {
        /* init pre-conditioner for H and init solution vectors */
        Hdia_inv[i] = 1.0;
        b_s[i]      = -chi[ atom->type[i] ];
        b_t[i]      = -1.0;
        s[i] = s_EL_Scf[i];
        t[i] = t_EL_Scf[i];
      }
    }
  }
  //if(update->ntimestep%1000 == 0 and comm->me ==0)printf("x_last=%.16f x_hist_0=%.16f x_hist_1=%.16f x_hist_2=%.16f\n'",atom->x_last,x_hist_0,x_hist_1,x_hist_2);

  
  if (t_s_flag == 0) 
      x_last = Chi_eq_iEL_Scf;
  else
    r_last = b_last = q_last = x_last = d_last = 0.0;
  
  pack_flag = 2;
  comm->forward_comm_fix(this); //Dist_vector( s );
  pack_flag = 3;
  comm->forward_comm_fix(this); //Dist_vector( t );
}

/* ---------------------------------------------------------------------- */

int FixIELReax::CG( double *b, double *x)
{
  int  i, j;
  double tmp, alpha, beta, b_norm;
  double sig_old, sig_new;
  int jj;
  double tolerance_tmp;
  int i_tolerance_tmp;

  pack_flag = 1;
  sparse_matvec( &H, x, q, x_last);
  comm->reverse_comm_fix(this); //Coll_Vector( q );

  vector_sum( r , 1.,  b, -1., q, nn);

  r_last = b_last - q_last;

  for (jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    if (atom->mask[j] & groupbit){
      d[j] = r[j] * Hdia_inv[j]; //pre-condition
    }
  }

  d_last=r_last;

  b_norm = parallel_norm( b, nn);

  sig_new = parallel_dot( r, d, nn);

  sig_new+=r_last*r_last;// For t_s_flag=1

  if (update->ntimestep == 0) {
    tolerance_tmp = 0.0000000001;
    i_tolerance_tmp = 1;
  }
  else {
    tolerance_tmp = tolerance;
    i_tolerance_tmp = i_tolerance;
  }

  for (i = 1; (i < imax && (sqrt(sig_new) / b_norm > tolerance_tmp)) || i < i_tolerance_tmp; ++i) {
    comm->forward_comm_fix(this); //Dist_vector( d );
    sparse_matvec( &H, d, q , d_last);
    comm->reverse_comm_fix(this); //Coll_vector( q );

    tmp = parallel_dot( d, q, nn);

    tmp+=d_last*q_last;

    alpha = sig_new / tmp;

    vector_add( x, alpha, d, nn );

    x_last += alpha*d_last;
    vector_add( r, -alpha, q, nn );

    r_last += -alpha*q_last;

    // pre-conditioning

    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit)
        p[j] = r[j] * Hdia_inv[j];
    }
    sig_old = sig_new;

    sig_new = parallel_dot( r, p, nn);

    sig_new+= r_last*r_last;

    beta = sig_new / sig_old;

    vector_sum( d, 1., p, beta, d, nn );

    d_last = r_last + beta*d_last;
  }

  //if(comm->me ==0) printf("Number of SCF iterations= %i\n",i); // added by Itai EL-Scf

  if (i >= imax && comm->me == 0) {
    char str[128];
    sprintf(str,"Fix qeq/reax CG convergence failed after %d iterations "
            "at " BIGINT_FORMAT " step",i,update->ntimestep);
    error->warning(FLERR,str);
  }

  return i;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::final_integrate()
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double term = 2.0/(dtv*dtv);
  double dt = dtv;
  double dt_2 = 0.5*dtv;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      at_EL_Scf[i] = term * (t[i]-t_EL_Scf[i]);         
      vt_EL_Scf[i] = vt_EL_Scf[i] + at_EL_Scf[i]*dt_2*Omega_t;  
      as_EL_Scf[i] = term * (s[i]-s_EL_Scf[i]);         
      vs_EL_Scf[i] = vs_EL_Scf[i] + as_EL_Scf[i]*dt_2*Omega_s;
    }

  if (thermo_flag == 1)
    Nose_Hoover();
}

/* ---------------------------------------------------------------------- */

void FixIELReax::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::sparse_matvec( sparse_matrix *A, double *x, double *b, double last)
{
  int i, j, itr_j;
  int ii;
  double q_last_tmp;
  q_last=0.0;
  q_last_tmp=0.0;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit){
      b[i] = eta[ atom->type[i] ] * x[i]-last;
    }
  }

  for (ii = nn; ii < NN; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      b[i] = 0;
  }

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      for (itr_j=A->firstnbr[i]; itr_j<A->firstnbr[i]+A->numnbrs[i]; itr_j++) {
        j = A->jlist[itr_j];
        b[i] += A->val[itr_j] * x[j];
        b[j] += A->val[itr_j] * x[i];
      }
    }
    q_last_tmp+=x[i];

  }
  if(t_s_flag==0)
    MPI_Allreduce( &q_last_tmp, &q_last, 1, MPI_DOUBLE, MPI_SUM, world);
}


/* ---------------------------------------------------------------------- */

void FixIELReax::calculate_Q()
{
  int i, k;
  double u, s_sum, t_sum;
  double *q = atom->q;

  int ii;

  if(t_s_flag)
    {
      s_sum = parallel_vector_acc( s, nn);
      t_sum = parallel_vector_acc( t, nn);
      u = s_sum / t_sum;
    }

  //printf("timestep=%i\n",update->ntimestep);
  //printf("u=%.16f\n",u);

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      if(t_s_flag)
        q[i] = s[i] - u * t[i];
      else
        q[i] = s[i];
      if(update->ntimestep==0){
        s_EL_Scf[i]=s[i];
        t_EL_Scf[i]=t[i];
        if(t_s_flag==0)
          Chi_eq_iEL_Scf = x_last;
      }
    }
  }
  if(t_s_flag==0)
    {
      x_hist_2=x_hist_1;
      x_hist_1=x_hist_0;
      x_hist_0=x_last;
    }

  pack_flag = 4;
  comm->forward_comm_fix(this); //Dist_vector( atom->q );
}


/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
------------------------------------------------------------------------- */

void FixIELReax::grow_arrays(int nmax)
{
  memory->grow(s_EL_Scf,nmax,"qeq:s_EL_Scf");
  memory->grow(vs_EL_Scf,nmax,"qeq:vs_EL_Scf");
  memory->grow(as_EL_Scf,nmax,"qeq:as_EL_Scf");

  memory->grow(t_EL_Scf,nmax,"qeq:t_EL_Scf");
  memory->grow(vt_EL_Scf,nmax,"qeq:vt_EL_Scf");
  memory->grow(at_EL_Scf,nmax,"qeq:at_EL_Scf");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

void FixIELReax::copy_arrays(int i, int j, int /*delflag*/)
{
  s_EL_Scf[j] = s_EL_Scf[i];
  vs_EL_Scf[j] = vs_EL_Scf[i];
  as_EL_Scf[j] = as_EL_Scf[i];

  t_EL_Scf[j] = t_EL_Scf[i];
  vt_EL_Scf[j] = vt_EL_Scf[i];
  at_EL_Scf[j] = at_EL_Scf[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixIELReax::pack_exchange(int i, double *buf)
{
  buf[0] = s_EL_Scf[i];
  buf[0] = vs_EL_Scf[i];
  buf[0] = as_EL_Scf[i];

  buf[1] = t_EL_Scf[i];
  buf[1] = vt_EL_Scf[i];
  buf[1] = at_EL_Scf[i];
  return 6;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixIELReax::unpack_exchange(int nlocal, double *buf)
{
  s_EL_Scf[nlocal] = buf[0];
  vs_EL_Scf[nlocal] = buf[0];
  as_EL_Scf[nlocal] = buf[0];

  t_EL_Scf[nlocal] = buf[1];
  vt_EL_Scf[nlocal] = buf[1];
  at_EL_Scf[nlocal] = buf[1];
  return 6;
}

void FixIELReax::kinaux (double &temp_aux_t,double &temp_aux_s)
{
  int i;
  double term;
  double eksum_aux;
  double ekaux_t=0.0;
  double ekaux_s=0.0;
  int nfree_aux =atom->natoms;
  int nlocal = atom->nlocal;
  //zero out the temperature and kinetic energy components
  double ekaux_t_mpi = 0.0;
  double ekaux_s_mpi = 0.0;

  temp_aux_t = 0.0;
  temp_aux_s = 0.0;

  //get the kinetic energy tensor for auxiliary variables
  term = 0.5;

  if(t_s_flag==1)
    {
      for (i = 0; i < nlocal ;i++){
        ekaux_t = ekaux_t + term*vt_EL_Scf[i]*vt_EL_Scf[i];
        ekaux_s = ekaux_s + term*vs_EL_Scf[i]*vs_EL_Scf[i];
      }
      MPI_Allreduce( &ekaux_t, &ekaux_t_mpi, 1, MPI_DOUBLE, MPI_SUM, world);
      temp_aux_t = 2.0 * ekaux_t_mpi / nfree_aux;
    }
  else
    {
      for (i = 0; i < nlocal ;i++){
        ekaux_s = ekaux_s + term*vs_EL_Scf[i]*vs_EL_Scf[i];
      }
      ekaux_t = term*vChi_eq_iEL_Scf*vChi_eq_iEL_Scf;
      temp_aux_t = 2.0 * ekaux_t ;
    }

  MPI_Allreduce( &ekaux_s, &ekaux_s_mpi, 1, MPI_DOUBLE, MPI_SUM, world);

  // find the total kinetic energy and auxiliary temperatures

  //eksum_aux = ekaux;

  //if (nfree_aux =! 0){
  temp_aux_s = 2.0 * ekaux_s_mpi / (nfree_aux);

}

/* ---------------------------------------------------------------------- */

void FixIELReax::Berendersen()
{
  
  int nlocal = atom->nlocal;

  double scale_t = 1.0;
  double scale_s = 1.0;
  double temp_aux_t = 0.0;
  double temp_aux_s = 0.0;
  kinaux (temp_aux_t, temp_aux_s);
  
  if(temp_aux_s != 0)
    scale_s = sqrt(1.0 + (dtv/tautemp_aux)*(kelvin_aux_s/temp_aux_s-1.0));
  
  if(temp_aux_t != 0)
    scale_t = sqrt(1.0 + (dtv/tautemp_aux)*(kelvin_aux_t/temp_aux_t-1.0));
  
  if(t_s_flag){
    for (int i = 0; i < nlocal; i++){
      vt_EL_Scf[i] = scale_t * vt_EL_Scf[i];
      vs_EL_Scf[i] = scale_s * vs_EL_Scf[i];
      if(abs(vt_EL_Scf[i]) > 0.1)vt_EL_Scf[i]*=0.1;
      if(abs(vs_EL_Scf[i]) > 0.1)vs_EL_Scf[i]*=0.1;
    }
  }
  else{
    for (int i = 0; i < nlocal; i++){
      vs_EL_Scf[i] = scale_s * vs_EL_Scf[i];
      if(abs(vs_EL_Scf[i]) > 1.0)vs_EL_Scf[i]*=0.1;
    }
    vChi_eq_iEL_Scf=vChi_eq_iEL_Scf*scale_t;
  }

}
  
void FixIELReax::Nose_Hoover()
{
  
  int nlocal = atom->nlocal;
  
  double scale_t = 1.0;
  double scale_s = 1.0;
  double temp_aux_t = 0.0;
  double temp_aux_s = 0.0;
  kinaux (temp_aux_t, temp_aux_s);
  
  int nc = 5;
  int ns = 3;
  double dtc = dtv / nc;
  double w[3];
  w[0] = 1.0 / (2.0-pow(2.0,(1.0/3.0)));
  w[1] = 1.0 - 2.0*w[0];
  w[2] = w[0];
   
  double expterm;
  double dts,dt2,dt4,dt8;
  int nfree_aux =atom->natoms;
  
  for (int i = 0;i < nc; i++){
    for (int j = 0; j < ns; j++){
      dts = w[j] * dtc;
      dt2 = 0.5 * dts;
      dt4 = 0.25 * dts;
      dt8 = 0.125 * dts;
      
      if(t_s_flag)
	{
	  // t aux calculation
	  tgnhaux[4]=(tnhaux[3]*tvnhaux[3]*tvnhaux[3]-kelvin_aux_t)/tnhaux[4];
	  //printf("tgnhaux[4]= %.16f tnhaux[3]= %.16f tvnhaux[3]=%.16f tnhaux[4]=%.16f kelvin_aux=%.16f\n",tgnhaux[4],tnhaux[3],tvnhaux[3],tnhaux[4],kelvin_aux);
	  tvnhaux[4]=tvnhaux[4]+tgnhaux[4]*dt4;
	  tgnhaux[3]=(tnhaux[2]*tvnhaux[2]*tvnhaux[2]-kelvin_aux_t)/tnhaux[3];
	  expterm=exp(-tvnhaux[4]*dt8);
	  tvnhaux[3]=expterm*(tvnhaux[3]*expterm+tgnhaux[3]*dt4);
	  tgnhaux[2]=(tnhaux[1]*tvnhaux[1]*tvnhaux[1]-kelvin_aux_t)/tnhaux[2];
	  expterm=exp(-tvnhaux[3]*dt8);
	  tvnhaux[2]=expterm*(tvnhaux[2]*expterm+tgnhaux[2]*dt4);
	  tgnhaux[1]=((nfree_aux)*temp_aux_t-(nfree_aux)*kelvin_aux_t)/tnhaux[1];
	  expterm=exp(-tvnhaux[2]*dt8);
	  tvnhaux[1]=expterm*(tvnhaux[1]*expterm+tgnhaux[1]*dt4);
	  scale_t=scale_t*exp(-tvnhaux[1]*dt2);
	  
	  temp_aux_t=temp_aux_t*exp(-tvnhaux[1]*dt2)*exp(-tvnhaux[1]*dt2);
	  tgnhaux[1]=((nfree_aux)*temp_aux_t-(nfree_aux)*kelvin_aux_t)/tnhaux[1];
	  expterm=exp(-tvnhaux[2]*dt8);
	  tvnhaux[1]=expterm*(tvnhaux[1]*expterm+tgnhaux[1]*dt4);
	  tgnhaux[2]=(tnhaux[1]*tvnhaux[1]*tvnhaux[1]-kelvin_aux_t)/tnhaux[2];
	  expterm=exp(-tvnhaux[3]*dt8);
	  tvnhaux[2]=expterm*(tvnhaux[2]*expterm+tgnhaux[2]*dt4);
	  tgnhaux[3]=(tnhaux[2]*tvnhaux[2]*tvnhaux[2]-kelvin_aux_t)/tnhaux[3];
	  expterm=exp(-tvnhaux[4]*dt8);
	  tvnhaux[3]=expterm*(tvnhaux[3]*expterm+tgnhaux[3]*dt4);
	  tgnhaux[4]=(tnhaux[3]*tvnhaux[3]*tvnhaux[3]-kelvin_aux_t)/tnhaux[4];
	  tvnhaux[4]=tvnhaux[4]+tgnhaux[4]*dt4;
	} 
      //printf("scale_t= %.16f exp(tvn)= %.16f dt2=%.16f tvnhaux[1]=%.16f \n",scale_t,exp(-tvnhaux[1]*dt2),dt2,tvnhaux[1]);
      // s aux calculation
      sgnhaux[4]=(snhaux[3]*svnhaux[3]*svnhaux[3]-kelvin_aux_s)/snhaux[4];
      svnhaux[4]=svnhaux[4]+sgnhaux[4]*dt4;
      sgnhaux[3]=(snhaux[2]*svnhaux[2]*svnhaux[2]-kelvin_aux_s)/snhaux[3];
      expterm=exp(-svnhaux[4]*dt8);
      svnhaux[3]=expterm*(svnhaux[3]*expterm+sgnhaux[3]*dt4);
      sgnhaux[2]=(snhaux[1]*svnhaux[1]*svnhaux[1]-kelvin_aux_s)/snhaux[2];
      expterm=exp(-svnhaux[3]*dt8);
      svnhaux[2]=expterm*(svnhaux[2]*expterm+sgnhaux[2]*dt4);
      sgnhaux[1]=((nfree_aux)*temp_aux_s-(nfree_aux)*kelvin_aux_s)/snhaux[1];
      expterm=exp(-svnhaux[2]*dt8);
      svnhaux[1]=expterm*(svnhaux[1]*expterm+sgnhaux[1]*dt4);
      scale_s=scale_s*exp(-svnhaux[1]*dt2);
      temp_aux_s=temp_aux_s*exp(-svnhaux[1]*dt2)*exp(-svnhaux[1]*dt2);
      sgnhaux[1]=((nfree_aux)*temp_aux_s-(nfree_aux)*kelvin_aux_s)/snhaux[1];
      expterm=exp(-svnhaux[2]*dt8);
      svnhaux[1]=expterm*(svnhaux[1]*expterm+sgnhaux[1]*dt4);
      sgnhaux[2]=(snhaux[1]*svnhaux[1]*svnhaux[1]-kelvin_aux_s)/snhaux[2];
      expterm=exp(-svnhaux[3]*dt8);
      svnhaux[2]=expterm*(svnhaux[2]*expterm+sgnhaux[2]*dt4);
      sgnhaux[3]=(snhaux[2]*svnhaux[2]*svnhaux[2]-kelvin_aux_s)/snhaux[3];
      expterm=exp(-svnhaux[4]*dt8);
      svnhaux[3]=expterm*(svnhaux[3]*expterm+sgnhaux[3]*dt4);
      sgnhaux[4]=(snhaux[3]*svnhaux[3]*svnhaux[3]-kelvin_aux_s)/snhaux[4];
      svnhaux[4]=svnhaux[4]+sgnhaux[4]*dt4;
      
    }
  }
  //printf("scale_t= %.16f scale_s= %.16f \n",scale_t,scale_s);
  for (int i = 0; i < nlocal; i++){
    if(t_s_flag)
      vt_EL_Scf[i]=vt_EL_Scf[i]*scale_t;
    vs_EL_Scf[i]=vs_EL_Scf[i]*scale_s;
  }
  if(t_s_flag==0)
    vChi_eq_iEL_Scf=vChi_eq_iEL_Scf*scale_s;

}
