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

#include "fix_xlmd_reax.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <random>
#include "random_mars.h"
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

static const char cite_fix_xlmd_reax[] =
  "fix xlmd/reax command:\n\n"
  "@misc{Tan2020stochastic,\n"
  " title={Stochastic Constrained Extended System Dynamics for Solving Charge Equilibration Models},\n"
  " author={Songchen Tan and Itai Leven and Dong An and Lin Lin and Teresa Head-Gordon},\n"
  " year={2020},\n"
  " eprint={2005.10736},\n"
  " archivePrefix={arXiv},\n"
  " primaryClass={physics.comp-ph}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

FixXLMDReax::FixXLMDReax(LAMMPS *lmp, int narg, char **arg) :
  FixQEqReax(lmp, narg, arg), random(nullptr)
{
  xlmd_flag = utils::inumeric(FLERR,arg[8],false,lmp); // 1 (Ber) 2 (Lang)
  mLatent = utils::numeric(FLERR,arg[9],false,lmp);  // latent mass
  tauLatent = utils::numeric(FLERR,arg[10],false,lmp);  // latent thermostat strength
  tLatent = utils::numeric(FLERR,arg[11],false,lmp);  // latent temperature
  seed = utils::inumeric(FLERR,arg[12],false,lmp);

  if (seed <= 0) error->all(FLERR,"Illegal fix iel/reax command");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + comm->me);

  qLatent = pLatent = fLatent = NULL;
  setup_flag = 0;
}

/* ---------------------------------------------------------------------- */

FixXLMDReax::~FixXLMDReax()
{
  if (copymode) return;

  memory->destroy(qLatent);
  memory->destroy(pLatent);
  memory->destroy(fLatent);

  delete random;
}

/* ---------------------------------------------------------------------- */

void FixXLMDReax::post_constructor()
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_xlmd_reax);

  grow_arrays(atom->nmax);
  for (int i = 0; i < atom->nmax; i++) {
    for (int j = 0; j < nprev; ++j)
      s_hist[i][j] = t_hist[i][j] = 0.0;
  }

  pertype_parameters(pertype_option);
  if (dual_enabled)
    error->all(FLERR,"Dual keyword only supported with fix qeq/reax/omp");
}

/* ---------------------------------------------------------------------- */

int FixXLMDReax::setmask()
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

void FixXLMDReax::init()
{
  FixQEqReax::init();

  dtv = update->dt;
  dth = update->dt/2;
  dtf = 0.5 * update->dt * force->ftm2v;

  // init XLMD

  for (int i = 0; i < nn; i++) {
    if (atom->mask[i] & groupbit) {
      qLatent[i] = atom->q[i];
      pLatent[i] = 0;
      fLatent[i] = 0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixXLMDReax::setup_pre_force(int vflag)
{
  setup_flag = 1;
  FixQEqReax::setup_pre_force(vflag);
  setup_flag = 0;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixXLMDReax::initial_integrate(int /*vflag*/)
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

  if (xlmd_flag == 1) {
    //if (update->ntimestep > 1000) {
      Berendersen(dtv);
    //}
  } else if (xlmd_flag == 2)
    Langevin(dtv);

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      qLatent[i] += dth * pLatent[i] / mLatent;
}

/* ---------------------------------------------------------------------- */

void FixXLMDReax::pre_force(int /*vflag*/)
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

  if (setup_flag) {
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

void FixXLMDReax::calculate_XLMD() {

  for (int ii = 0; ii < nn; ++ii) {
    const int i = ilist[ii];
    if (atom->mask[i] & groupbit)
      atom->q[i] = qLatent[i];
  }
  pack_flag = 4;
  comm->forward_comm_fix(this); // Dist_vector( atom-> q);

  // Force for Latent
  pack_flag = 1;
  sparse_matvec( &H, atom->q, q);
  comm->reverse_comm_fix(this);
  for (int ii = 0; ii < nn; ++ii) {
    const int i = ilist[ii];
    if (atom->mask[i] & groupbit)
      fLatent[i] = (b_s[i] - q[i]) * ATOMIC_TO_REAL + qConst;
  }

  // Apply constraint on f
  double sumFLatent = parallel_vector_acc(fLatent, nn);
  double fDev = sumFLatent / atom->natoms;
  for (int ii = 0; ii < nn; ++ii) {
    const int i = ilist[ii];
    if (atom->mask[i] & groupbit)
      fLatent[i] = fLatent[i] - fDev;
  }
}

/* ---------------------------------------------------------------------- */

void FixXLMDReax::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Evolve B(t/2)

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      pLatent[i] += dth * fLatent[i];
}

/* ---------------------------------------------------------------------- */

void FixXLMDReax::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

double FixXLMDReax::kinetic_latent()
{
  int nlocal = atom->nlocal;
  double sum_p2 = 0.0;
  double sum_p2_mpi = 0.0;
  double avg_p2 = 0.0;

  // find the total kinetic energy and auxiliary temperatures
  for (int i = 0; i < nlocal; i++)
    sum_p2 = sum_p2 + pLatent[i] * pLatent[i] / mLatent / 2.0;
  MPI_Allreduce(&sum_p2, &sum_p2_mpi, 1, MPI_DOUBLE, MPI_SUM, world);
  avg_p2 = sum_p2_mpi / atom->natoms;
  return avg_p2;
}

/* ---------------------------------------------------------------------- */

void FixXLMDReax::Berendersen(const double dt)
{
  
  int nlocal = atom->nlocal;

  double avg_p2 = kinetic_latent();
  double target_p2 = tLatent / 2.0;
  double scale = sqrt(1.0 + (dt/(tauLatent))*(target_p2/avg_p2-1.0));

  for (int i = 0; i < nlocal; i++)
    pLatent[i] = pLatent[i] * scale;
}

/* ---------------------------------------------------------------------- */

void FixXLMDReax::Langevin(const double dt) {
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double pLatentAvg = sqrt(mLatent * tLatent);
  double dissipationLatent = exp(-dt/tauLatent);
  double fluctuationLatent = sqrt(1 - dissipationLatent * dissipationLatent);

  std::default_random_engine gen;
  std::normal_distribution<double> dis(0,1);

  // Log the prior kinetic energy.

  double kineticBefore = kinetic_latent();

  // Perform the Langevin integration.

  for (int i = 0; i < nlocal; i++) {
   //pLatent[i] = pLatent[i] * dissipationLatent + pLatentAvg * fluctuationLatent * dis(gen);
   pLatent[i] = pLatent[i] * dissipationLatent + pLatentAvg * fluctuationLatent * random->gaussian();
  }

  double pDev = parallel_vector_acc(pLatent, nn) / atom->natoms;

  for (int i = 0; i < nlocal; i++)
    pLatent[i] = pLatent[i] - pDev;
}


/* ---------------------------------------------------------------------- */

void FixXLMDReax::calculate_Q()
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

void FixXLMDReax::grow_arrays(int nmax)
{ 
  memory->grow(s_hist,nmax,nprev,"xlmd:s_hist");
  memory->grow(t_hist,nmax,nprev,"xlmd:t_hist");

  memory->grow(qLatent,nmax,"xlmd:qLatent");
  memory->grow(pLatent,nmax,"xlmd:pLatent");
  memory->grow(fLatent,nmax,"xlmd:fLatent");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

void FixXLMDReax::copy_arrays(int i, int j, int /*delflag*/)
{
  qLatent[j] = qLatent[i];
  pLatent[j] = pLatent[i];
  fLatent[j] = fLatent[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixXLMDReax::pack_exchange(int i, double *buf)
{
  buf[0] = qLatent[i];
  buf[1] = pLatent[i];
  buf[2] = fLatent[i];

  return 3;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixXLMDReax::unpack_exchange(int nlocal, double *buf)
{
  qLatent[nlocal] = buf[0];
  pLatent[nlocal] = buf[1];
  fLatent[nlocal] = buf[2];

  return 3;
}
