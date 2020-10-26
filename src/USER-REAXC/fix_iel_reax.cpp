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
   Contributing author: Songchen Tan (UC Berkeley)
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
  if (narg<15 || narg>16) error->all(FLERR,"Illegal fix qeq/reax command");

  tolerance_t = force->numeric(FLERR,arg[8]);
  tolerance_s = force->numeric(FLERR,arg[9]);
  t_s_flag = force->numeric(FLERR,arg[10]);
  iEL_Scf_flag = force->numeric(FLERR,arg[11]); //iEL_Scf_flag=0 no iEL-Scf://if iEL_Scf_flag=1 with Scf
  n1_Scf_flag = force->numeric(FLERR,arg[12]); //if=0 no nScf://if =1 with 1 nScf
  Precon_flag = force->numeric(FLERR,arg[13]); //if=0 no CG Pre conditioning://if =1 with CG Pre conditioning
  i_tolerance_t = force->numeric(FLERR,arg[14]);
  i_tolerance_s = force->numeric(FLERR,arg[15]);

  if(comm->me == 0)
    printf("\nQeq params:\ntolerance_t= %.16f tolerance_s= %.16f  t_s_flag= %i iEL_Scf_flag= %i n1_Scf_flag= %i Precon_flag= %i\n\n",tolerance_t,tolerance_s,t_s_flag,iEL_Scf_flag,n1_Scf_flag,Precon_flag);

  xlmd_flag = utils::inumeric(FLERR,arg[8],false,lmp); // 1 (XLMD) 2 (Ber) 3 (NH) 4 (Lang)
  mLatent = utils::numeric(FLERR,arg[9],false,lmp);  // latent mass
  tauLatent = utils::numeric(FLERR,arg[10],false,lmp);  // latent thermostat strength
  tLatent = utils::numeric(FLERR,arg[11],false,lmp);  // latent temperature

  qLatent = pLatent = fLatent = NULL;
}

/* ---------------------------------------------------------------------- */

FixIELReax::~FixIELReax()
{
  if (copymode) return;

  memory->destroy(qLatent);
  memory->destroy(pLatent);
  memory->destroy(fLatent);
}

/* ---------------------------------------------------------------------- */

void FixIELReax::post_constructor()
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_iel_reax);

  grow_arrays(atom->nmax);
  for (int i = 0; i < atom->nmax; i++) {
    for (int j = 0; j < nprev; ++j)
      s_hist[i][j] = t_hist[i][j] = 0;

    qLatent[i] = atom->q[i];
    pLatent[i] = 0;
    fLatent[i] = 0;    
  }

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
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::init()
{
  FixQEqReax::init();

  dtv = update->dt;
  dth = update->dt/2;
  dtf = 0.5 * update->dt * force->ftm2v;

  b_last = 0.0;    //added by itai for the iEL_Scf q
  atom->x_last = x_hist_2 = x_hist_1 = x_hist_0 = 0.0;
  q_last = 0.0; //added by itai for the iEL_Scf q
  r_last = 0.0; //added by itai for the iEL_Scf q
  d_last = 0.0;

  // init XLMD

  for (int i = 0; i < nn; i++){
    if (atom->mask[i] & groupbit) {
      qLatent[i] = atom->q[i];
      pLatent[i] = 0;
      fLatent[i] = 0;
    }
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

  // evolve B(t/2) A(t/2)

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      pLatent[i] += dth * fLatent[i];
      qLatent[i] += dth * pLatent[i] / mLatent;
    }
  }

  // evolve O(t) for BAOAB Scheme

  if (xlmd_flag == 2) {
    if (update->ntimestep > 1000) {
      Berendersen(dtv);
    }
  } else if (xlmd_flag == 3) {
    Langevin(dtv);
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      qLatent[i] += dth * pLatent[i] / mLatent;
    }
  }
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

  if (update->ntimestep == 0) {
    matvecs_s = CG(b_s, s);       // CG on s - parallel
    matvecs_t = CG(b_t, t);       // CG on t - parallel
    matvecs = matvecs_s + matvecs_t;
    calculate_Q();

    // init q

    for (int ii = 0; ii < nn; ++ii) {
      const int i = ilist[ii];
      if (atom->mask[i] & groupbit) {
        qLatent[i] = atom->q[i];
      }
    }
  } else
    calculate_XLMD();

  if (comm->me == 0) {
    t_end = MPI_Wtime();
    qeq_time = t_end - t_start;
  }
}

/* ---------------------------------------------------------------------- */

void FixIELReax::calculate_XLMD() {

  for (int ii = 0; ii < nn; ++ii) {
    const int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      atom->q[i] = qLatent[i];
    }
  }
  pack_flag = 4;
  comm->forward_comm_fix(this); // Dist_vector( atom-> q);

  // Force for Latent
  pack_flag = 1;
  sparse_matvec( &H, atom->q, q);
  comm->reverse_comm_fix(this);
  for (int ii = 0; ii < nn; ++ii) {
    const int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      fLatent[i] = (b_s[i] - q[i]) * ATOMIC_TO_REAL + qConst;
    }
  }

  // Apply constraint on f
  double sumFLatent = parallel_vector_acc(fLatent, nn);
  double fDev = sumFLatent / atom->natoms;
  for (int ii = 0; ii < nn; ++ii) {
    const int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      fLatent[i] = fLatent[i] - fDev;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixIELReax::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Evolve B(t/2)

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      pLatent[i] += dth * fLatent[i];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixIELReax::end_of_step() {
  double qDev = parallel_vector_acc(qLatent, nn) / atom->natoms;
  double KineticLatent = parallel_dot(pLatent, pLatent, nn) / 2 / mLatent;
  // Show charge conservation and latent temperature
  if (update->ntimestep % 100 == 0 && comm->me == 0)
    printf("%d\t%.8f\t%.8f\n", update->ntimestep, qDev, KineticLatent);
}


/* ---------------------------------------------------------------------- */

void FixIELReax::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

double FixIELReax::kinetic_latent()
{
  int nlocal = atom->nlocal;
  double sum_p2 = 0.0;
  double sum_p2_mpi = 0.0;
  double avg_p2 = 0.0;

  // find the total kinetic energy and auxiliary temperatures
  for (int i = 0; i < nlocal; i++) {
    sum_p2 = sum_p2 + pLatent[i] * pLatent[i] / mLatent / 2.0;
  }
  MPI_Allreduce(&sum_p2, &sum_p2_mpi, 1, MPI_DOUBLE, MPI_SUM, world);
  avg_p2 = sum_p2_mpi / atom->natoms;
  return avg_p2;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::Berendersen(const double dt)
{
  
  int nlocal = atom->nlocal;

  double avg_p2 = kinetic_latent();
  double target_p2 = tLatent / 2.0;
  double scale = sqrt(1.0 + (dt/(tauLatent))*(target_p2/avg_p2-1.0));

  for (int i = 0; i < nlocal; i++) {
    pLatent[i] = pLatent[i] * scale;
  }
}

/* ---------------------------------------------------------------------- */

void FixIELReax::Langevin(const double dt) {
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double pLatentAvg = sqrt(mLatent * tLatent);
  double dissipationLatent = exp(-dt/tauLatent);
  double fluctuationLatent = sqrt(1 - dissipationLatent * dissipationLatent);

  // change to LAMMPS RNG
  std::default_random_engine gen;
  std::normal_distribution<double> dis(0,1);

  // Log the prior kinetic energy.

  double kineticBefore = kinetic_latent();

  // Perform the Langevin integration.

  for (int i = 0; i < nlocal; i++) {
    pLatent[i] = pLatent[i] * dissipationLatent + pLatentAvg * fluctuationLatent * dis(gen);
  }

  double pDev = parallel_vector_acc(pLatent, nn) / atom->natoms;

  for (int i = 0; i < nlocal; i++) {
    pLatent[i] = pLatent[i] - pDev;
  }
}


/* ---------------------------------------------------------------------- */

void FixIELReax::calculate_Q()
{
  int i, ii, k;
  double u, s_sum, t_sum;
  double *q = atom->q;

  s_sum = parallel_vector_acc( s, nn);
  t_sum = parallel_vector_acc( t, nn);
  u = s_sum / t_sum;

  // init the chemical potential
  qConst = u * ATOMIC_TO_REAL;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      q[i] = s[i] - u * t[i];
    }
  }

  pack_flag = 4;
  comm->forward_comm_fix(this); //Dist_vector( atom->q );
}

/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
------------------------------------------------------------------------- */

void FixIELReax::grow_arrays(int nmax)
{ 
  memory->grow(s_hist,nmax,nprev,"iel:s_hist");
  memory->grow(t_hist,nmax,nprev,"iel:t_hist");

  memory->grow(qLatent,nmax,"iel:qLatent");
  memory->grow(pLatent,nmax,"iel:pLatent");
  memory->grow(fLatent,nmax,"iel:fLatent");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

void FixIELReax::copy_arrays(int i, int j, int /*delflag*/)
{
  qLatent[j] = qLatent[i];
  pLatent[j] = pLatent[i];
  fLatent[j] = fLatent[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixIELReax::pack_exchange(int i, double *buf)
{
  buf[0] = qLatent[i];
  buf[1] = pLatent[i];
  buf[2] = fLatent[i];

  return 3;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixIELReax::unpack_exchange(int nlocal, double *buf)
{
  qLatent[nlocal] = buf[0];
  pLatent[nlocal] = buf[1];
  fLatent[nlocal] = buf[2];

  return 3;
}

