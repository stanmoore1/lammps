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
#include <random> ////////////////////////////////
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

using namespace LAMMPS_NS;
using namespace FixConst;

#define EV_TO_KCAL_PER_MOL 14.4
#define ATOMIC_TO_REAL 23.06
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

static const char cite_fix_qeq_reax[] =
  "fix qeq/reax command:\n\n"
  "@Article{Aktulga12,\n"
  " author = {H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama},\n"
  " title = {Parallel reactive molecular dynamics: Numerical methods and algorithmic techniques},\n"
  " journal = {Parallel Computing},\n"
  " year =    2012,\n"
  " volume =  38,\n"
  " pages =   {245--259}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

FixIELReax::FixIELReax(LAMMPS *lmp, int narg, char **arg) :
  FixQEqReax(lmp, narg, arg)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_qeq_reax);

  if (narg<8+1 || narg>9+1) error->all(FLERR,"Illegal fix qeq/reax command");

  strcpy(pertype_option,arg[7]);
  atom->XLMDFlag = force->inumeric(FLERR,arg[8]);

  // dual CG support only available for USER-OMP variant
  // check for compatibility is in Fix::post_constructor()
  dual_enabled = 0;
  if (narg == 9+1) {
    if (strcmp(arg[8+1],"dual") == 0) dual_enabled = 1;
    else error->all(FLERR,"Illegal fix qeq/reax command");
  }

  atom->XLMDFlag = force->inumeric(FLERR, arg[3]); // 0 (Exact) 1 (XLMD) 2 (Ber) 3 (NH) 4 (Lang)
  atom->mLatent = force->numeric(FLERR, arg[4]);  // latent mass
  atom->tauLatent = force->numeric(FLERR, arg[5]);  // latent thermostat strength
  atom->tLatent = force->numeric(FLERR, arg[6]);  // latent temperature
}

/* ---------------------------------------------------------------------- */

FixIELReax::~FixIELReax()
{
  if (copymode) return;
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
  dtv = update->dt;
  dth = update->dt/2;
  dtf = 0.5 * update->dt * force->ftm2v;

  // init XLMD

  double *qLatent;
  double *pLatent;
  double *fLatent;
  get_names("qLatent", qLatent);
  get_names("pLatent", pLatent);
  get_names("fLatent", fLatent);

  for (int i = 0; i < atom->nlocal; i++){
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

  // obtain Latents

  double *qLatent;
  double *pLatent;
  double *fLatent;
  get_names("qLatent", qLatent);
  get_names("pLatent", pLatent);
  get_names("fLatent", fLatent);

  // evolve B(t/2) A(t/2)

  if (atom->XLMDFlag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        pLatent[i] += dth * fLatent[i];
        qLatent[i] += dth * pLatent[i] / atom->mLatent;
      }
    }
  }

  // evolve O(t) for BAOAB Scheme

  //if (update->ntimestep % atom->respaLatent == 0) {
    //double dto = dtv * atom->respaLatent;
    double dto = dtv; /////////////////////////////////////////////
    if (atom->XLMDFlag == 2) {
      if (update->ntimestep > 1000) {
        Berendersen(dto);
      }
    } else if (atom->XLMDFlag == 3) {
      Langevin(dto);
    }
  //}

  if (atom->XLMDFlag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        qLatent[i] += dth * pLatent[i] / atom->mLatent;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixIELReax::setup_pre_force(int vflag)
{
}

/* ---------------------------------------------------------------------- */

void FixIELReax::pre_force(int /*vflag*/)
{
  double t_start, t_end;

  if (update->ntimestep % nevery) return;
  if (comm->me == 0) t_start = MPI_Wtime();

  n = atom->nlocal;
  N = atom->nlocal + atom->nghost;

  // grow arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) reallocate_storage();
  if (n > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
    reallocate_matrix();

  init_matvec();
  if (atom->XLMDFlag) {
    if (update->ntimestep == 0) {
      matvecs_s = CG(b_s, s);       // CG on s - parallel
      matvecs_t = CG(b_t, t);       // CG on t - parallel
      matvecs = matvecs_s + matvecs_t;
      calculate_Q();

      // init q
      int nn, NN, ii, i;
      int *ilist;

      if (reaxc) {
        nn = reaxc->list->inum;
        ilist = reaxc->list->ilist;
        NN = reaxc->list->inum + reaxc->list->gnum;
      } else {
        nn = list->inum;
        ilist = list->ilist;
        NN = list->inum + list->gnum;
      }

      double *qLatent;
      get_names("qLatent", qLatent);
      
      for (ii = 0; ii < nn; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit) {
          qLatent[i] = atom->q[i];
        }
      }

      pack_flag = 6;
      comm->forward_comm_fix(this); //Dist_vector( atom->qLatent );
    } else {
      calculate_XLMD();
    }
  } else {
    matvecs_s = CG(b_s, s);       // CG on s - parallel
    matvecs_t = CG(b_t, t);       // CG on t - parallel
    matvecs = matvecs_s + matvecs_t;
    calculate_Q();
  }

  if (comm->me == 0) {
    t_end = MPI_Wtime();
    qeq_time = t_end - t_start;
  }
}

void FixIELReax::calculate_XLMD() {
  double *qLatent;
  double *pLatent;
  double *fLatent;
  get_names("qLatent", qLatent);
  get_names("pLatent", pLatent);
  get_names("fLatent", fLatent);
  int nn, ii, i;
  int *ilist;

  if (reaxc) {
    nn = reaxc->list->inum;
    ilist = reaxc->list->ilist;
  } else {
    nn = list->inum;
    ilist = list->ilist;
  };

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
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
  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      fLatent[i] = (b_s[i] - q[i]) * ATOMIC_TO_REAL + atom->qConst;
    }
  }
  pack_flag = 8;
  comm->forward_comm_fix(this); //Dist_vector( fLatent );

  // Apply constraint on f
  double sumFLatent = parallel_vector_acc(fLatent, nn);
  double fDev = sumFLatent / atom->natoms;
  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      fLatent[i] = fLatent[i] - fDev;
    }
  }
  pack_flag = 8;
  comm->forward_comm_fix(this); //Dist_vector( fLatent );
}

/* ---------------------------------------------------------------------- */

void FixIELReax::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Obtain Latents

  double *qLatent;
  double *pLatent;
  double *fLatent;
  get_names("qLatent", qLatent);
  get_names("pLatent", pLatent);
  get_names("fLatent", fLatent);

  // Evolve B(t/2)


  if (atom->XLMDFlag) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        pLatent[i] += dth * fLatent[i];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixIELReax::end_of_step() {
  int nn, ii, i;
  int *ilist;

  if (reaxc) {
    nn = reaxc->list->inum;
    ilist = reaxc->list->ilist;
  } else {
    nn = list->inum;
    ilist = list->ilist;
  };

  double *qLatent;
  double *pLatent;
  get_names("qLatent", qLatent);
  get_names("pLatent", pLatent);
  if (atom->XLMDFlag) {
    double qDev = parallel_vector_acc(qLatent, nn) / atom->natoms;
    double KineticLatent = parallel_dot(pLatent, pLatent, nn) / 2 / atom->mLatent;
    // Show charge conservation and latent temperature
    if (update->ntimestep % 100 == 0 && comm->me == 0) {
      printf("%d\t%.8f\t%.8f\n", update->ntimestep, qDev, KineticLatent);
    }
  }
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
  double *pLatent;
  get_names("pLatent", pLatent);

  int nlocal = atom->nlocal;
  double sum_p2 = 0.0;
  double sum_p2_mpi = 0.0;
  double avg_p2 = 0.0;

  // find the total kinetic energy and auxiliary temperatures
  for (int i = 0; i < nlocal; i++) {
    sum_p2 = sum_p2 + pLatent[i] * pLatent[i] / atom->mLatent / 2.0;
  }
  MPI_Allreduce(&sum_p2, &sum_p2_mpi, 1, MPI_DOUBLE, MPI_SUM, world);
  avg_p2 = sum_p2_mpi / atom->natoms;
  return avg_p2;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::Berendersen(const double dt)
{
  
  int nlocal = atom->nlocal;
  double *pLatent;
  get_names("pLatent", pLatent);   

  double avg_p2 = kinetic_latent();
  double target_p2 = atom->tLatent / 2.0;
  double scale = sqrt(1.0 + (dt/(atom->tauLatent))*(target_p2/avg_p2-1.0));

  for (int i = 0; i < nlocal; i++) {
    pLatent[i] = pLatent[i] * scale;
  }
}

/* ---------------------------------------------------------------------- */

void FixIELReax::Langevin(const double dt) {
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  double *pLatent;
  get_names("pLatent",pLatent); 

  double pLatentAvg = sqrt(atom->mLatent * atom->tLatent);
  double dissipationLatent = exp(-dt/atom->tauLatent);
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

void FixIELReax::get_names(char *c,double *&ptr)
{
  int index,flag;
  index = atom->find_custom(c,flag);
  
  if (index!=-1)
    ptr = atom->dvector[index];
  else
    error->all(FLERR,"fix iEL-Scf requires fix property/atom ?? command");
}

/* ---------------------------------------------------------------------- */

void FixIELReax::calculate_Q()
{
  int i, k;
  double u, s_sum, t_sum;
  double *q = atom->q;

  int nn, ii;
  int *ilist;

  if (reaxc) {
    nn = reaxc->list->inum;
    ilist = reaxc->list->ilist;
  } else {
    nn = list->inum;
    ilist = list->ilist;
  }

  s_sum = parallel_vector_acc( s, nn);
  t_sum = parallel_vector_acc( t, nn);
  u = s_sum / t_sum;

  // in SC-XLMD, init the chemical potential
  atom->qConst = u * ATOMIC_TO_REAL;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      q[i] = s[i] - u * t[i];

      /* backup s & t */
      for (k = nprev-1; k > 0; --k) {
        s_hist[i][k] = s_hist[i][k-1];
        t_hist[i][k] = t_hist[i][k-1];
      }
      s_hist[i][0] = s[i];
      t_hist[i][0] = t[i];
    }
  }

  pack_flag = 4;
  comm->forward_comm_fix(this); //Dist_vector( atom->q );
}

/* ---------------------------------------------------------------------- */

int FixIELReax::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
  int m;

  if (pack_flag == 1)
    for(m = 0; m < n; m++) buf[m] = d[list[m]];
  else if (pack_flag == 2)
    for(m = 0; m < n; m++) buf[m] = s[list[m]];
  else if (pack_flag == 3)
    for(m = 0; m < n; m++) buf[m] = t[list[m]];
  else if (pack_flag == 4)
    for(m = 0; m < n; m++) buf[m] = atom->q[list[m]];
  else if (pack_flag == 5) {
    m = 0;
    for(int i = 0; i < n; i++) {
      int j = 2 * list[i];
      buf[m++] = d[j  ];
      buf[m++] = d[j+1];
    }
    return m;
  }
  else if (pack_flag == 6) {
    double *qLatent;
    get_names("qLatent", qLatent);
    for(m = 0; m < n; m++) buf[m] = qLatent[list[m]];
  }
  else if (pack_flag == 7) {
    double *pLatent;
    get_names("pLatent", pLatent);
    for(m = 0; m < n; m++) buf[m] = pLatent[list[m]];
  }
  else if (pack_flag == 8) {
    double *fLatent;
    get_names("fLatent", fLatent);
    for(m = 0; m < n; m++) buf[m] = fLatent[list[m]];
  }
  return n;
}

/* ---------------------------------------------------------------------- */

void FixIELReax::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  if (pack_flag == 1)
    for(m = 0, i = first; m < n; m++, i++) d[i] = buf[m];
  else if (pack_flag == 2)
    for(m = 0, i = first; m < n; m++, i++) s[i] = buf[m];
  else if (pack_flag == 3)
    for(m = 0, i = first; m < n; m++, i++) t[i] = buf[m];
  else if (pack_flag == 4)
    for(m = 0, i = first; m < n; m++, i++) atom->q[i] = buf[m];
  else if (pack_flag == 5) {
    int last = first + n;
    m = 0;
    for(i = first; i < last; i++) {
      int j = 2 * i;
      d[j  ] = buf[m++];
      d[j+1] = buf[m++];
    }
  }
  else if (pack_flag == 6) {
    double *qLatent;
    get_names("qLatent", qLatent);
    for(m = 0, i = first; m < n; m++, i++) qLatent[i] = buf[m];
  }
  else if (pack_flag == 7) {
    double *pLatent;
    get_names("pLatent", pLatent);
    for(m = 0, i = first; m < n; m++, i++) pLatent[i] = buf[m];
  }
  else if (pack_flag == 8) {
    double *fLatent;
    get_names("fLatent", fLatent);
    for(m = 0, i = first; m < n; m++, i++) fLatent[i] = buf[m];
  }
}

