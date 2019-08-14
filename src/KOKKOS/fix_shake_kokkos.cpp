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

#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include "fix_shake_kokkos.h"
#include "fix_rattle.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "update.h"
#include "respa.h"
#include "modify.h"
#include "domain.h"
#include "force.h"
#include "bond.h"
#include "angle.h"
#include "comm.h"
#include "group.h"
#include "fix_respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

#define RVOUS 1   // 0 for irregular, 1 for all2all

#define BIG 1.0e20
#define MASSDELTA 0.1

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixShakeKokkos<DeviceType>::FixShakeKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixShake(lmp, narg, arg), bond_flag(NULL), angle_flag(NULL),
  type_flag(NULL), mass_list(NULL), d_bond_distance(NULL), d_angle_distance(NULL),
  loop_respa(NULL), step_respa(NULL), x(NULL), v(NULL), f(NULL), ftmp(NULL),
  vtmp(NULL), mass(NULL), rmass(NULL), type(NULL), d_shake_flag(NULL),
  d_shake_atom(NULL), d_shake_type(NULL), xshake(NULL), nshake(NULL),
  list(NULL), b_count(NULL), b_count_all(NULL), b_ave(NULL), b_max(NULL),
  b_min(NULL), b_ave_all(NULL), b_max_all(NULL), b_min_all(NULL),
  a_count(NULL), a_count_all(NULL), a_ave(NULL), a_max(NULL), a_min(NULL),
  a_ave_all(NULL), a_max_all(NULL), a_min_all(NULL), atommols(NULL),
  onemols(NULL)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  virial_flag = 1;
  thermo_virial = 1;
  create_attribute = 1;
  dof_flag = 1;

  // error check

  molecular = atom->molecular;
  if (molecular == 0)
    error->all(FLERR,"Cannot use fix shake with non-molecular system");

  // perform initial allocation of atom-based arrays
  // register with Atom class

  d_shake_flag = NULL;
  d_shake_atom = NULL;
  d_shake_type = NULL;
  xshake = NULL;

  ftmp = NULL;
  vtmp = NULL;

  grow_arrays(atom->nmax);
  atom->add_callback(0);

  // set comm size needed by this fix

  comm_forward = 3;

  // parse SHAKE args

  if (narg < 8) error->all(FLERR,"Illegal fix shake command");

  tolerance = force->numeric(FLERR,arg[3]);
  max_iter = force->inumeric(FLERR,arg[4]);
  output_every = force->inumeric(FLERR,arg[5]);

  // parse SHAKE args for bond and angle types
  // will be used by find_clusters
  // store args for "b" "a" "t" as flags in (1:n) list for fast access
  // store args for "m" in list of length nmass for looping over
  // for "m" verify that atom masses have been set

  bond_flag = new int[atom->nbondtypes+1];
  for (int i = 1; i <= atom->nbondtypes; i++) bond_flag[i] = 0;
  angle_flag = new int[atom->nangletypes+1];
  for (int i = 1; i <= atom->nangletypes; i++) angle_flag[i] = 0;
  type_flag = new int[atom->ntypes+1];
  for (int i = 1; i <= atom->ntypes; i++) type_flag[i] = 0;
  mass_list = new double[atom->ntypes];
  nmass = 0;

  char mode = '\0';
  int next = 6;
  while (next < narg) {
    if (strcmp(arg[next],"b") == 0) mode = 'b';
    else if (strcmp(arg[next],"a") == 0) mode = 'a';
    else if (strcmp(arg[next],"t") == 0) mode = 't';
    else if (strcmp(arg[next],"m") == 0) {
      mode = 'm';
      atom->check_mass(FLERR);

    // break if keyword that is not b,a,t,m

    } else if (isalpha(arg[next][0])) break;

    // read numeric args of b,a,t,m

    else if (mode == 'b') {
      int i = force->inumeric(FLERR,arg[next]);
      if (i < 1 || i > atom->nbondtypes)
        error->all(FLERR,"Invalid bond type index for fix shake");
      bond_flag[i] = 1;

    } else if (mode == 'a') {
      int i = force->inumeric(FLERR,arg[next]);
      if (i < 1 || i > atom->nangletypes)
        error->all(FLERR,"Invalid angle type index for fix shake");
      angle_flag[i] = 1;

    } else if (mode == 't') {
      int i = force->inumeric(FLERR,arg[next]);
      if (i < 1 || i > atom->ntypes)
        error->all(FLERR,"Invalid atom type index for fix shake");
      type_flag[i] = 1;

    } else if (mode == 'm') {
      double massone = force->numeric(FLERR,arg[next]);
      if (massone == 0.0) error->all(FLERR,"Invalid atom mass for fix shake");
      if (nmass == atom->ntypes)
        error->all(FLERR,"Too many masses for fix shake");
      mass_d_list[nmass++] = massone;

    } else error->all(FLERR,"Illegal fix shake command");
    next++;
  }

  // parse optional args

  onemols = NULL;

  int iarg = next;
  while (iarg < narg) {
    if (strcmp(arg[next],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix shake command");
      int imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1)
        error->all(FLERR,"Molecule template ID for fix shake does not exist");
      if (atom->molecules[imol]->nset > 1 && comm->me == 0)
        error->warning(FLERR,"Molecule template for "
                       "fix shake has multiple molecules");
      onemols = &atom->molecules[imol];
      nmol = onemols[0]->nset;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix shake command");
  }

  // error check for Molecule template

  if (onemols) {
    for (int i = 0; i < nmol; i++)
      if (onemols[i]->shakeflag == 0)
        error->all(FLERR,"Fix shake molecule template must have shake info");
  }

  // allocate bond and angle distance arrays, indexed from 1 to n

  d_bond_distance = new double[atom->nbondtypes+1];
  d_angle_distance = new double[atom->nangletypes+1];

  // allocate statistics arrays

  if (output_every) {
    int nb = atom->nbondtypes + 1;
    b_count = new int[nb];
    b_count_all = new int[nb];
    b_ave = new double[nb];
    b_ave_all = new double[nb];
    b_max = new double[nb];
    b_max_all = new double[nb];
    b_min = new double[nb];
    b_min_all = new double[nb];

    int na = atom->nangletypes + 1;
    a_count = new int[na];
    a_count_all = new int[na];
    a_ave = new double[na];
    a_ave_all = new double[na];
    a_max = new double[na];
    a_max_all = new double[na];
    a_min = new double[na];
    a_min_all = new double[na];
  }

  // SHAKE vs RATTLE

  rattle = 0;

  // identify all SHAKE clusters

  double time1 = MPI_Wtime();

  find_clusters();

  double time2 = MPI_Wtime();

  if (comm->me == 0) {
    if (screen)
      fprintf(screen,"  find clusters CPU = %g secs\n",time2-time1);
    if (logfile)
      fprintf(logfile,"  find clusters CPU = %g secs\n",time2-time1);
  }

  // initialize list of SHAKE clusters to constrain

  maxlist = 0;
  list = NULL;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixShakeKokkos<DeviceType>::~FixShakeKokkos()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  // set bond_type and angle_type back to positive for SHAKE clusters
  // must set for all SHAKE bonds and angles stored by each atom

  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (d_shake_flag[i] == 0) continue;
    else if (d_shake_flag[i] == 1) {
      bondtype_findset(i,d_shake_atom(i,0),d_shake_atom(i,1),1);
      bondtype_findset(i,d_shake_atom(i,0),d_shake_atom(i,2),1);
      angletype_findset(i,d_shake_atom(i,1),d_shake_atom(i,2),1);
    } else if (d_shake_flag[i] == 2) {
      bondtype_findset(i,d_shake_atom(i,0),d_shake_atom(i,1),1);
    } else if (d_shake_flag[i] == 3) {
      bondtype_findset(i,d_shake_atom(i,0),d_shake_atom(i,1),1);
      bondtype_findset(i,d_shake_atom(i,0),d_shake_atom(i,2),1);
    } else if (d_shake_flag[i] == 4) {
      bondtype_findset(i,d_shake_atom(i,0),d_shake_atom(i,1),1);
      bondtype_findset(i,d_shake_atom(i,0),d_shake_atom(i,2),1);
      bondtype_findset(i,d_shake_atom(i,0),d_shake_atom(i,3),1);
    }
  }

  // delete locally stored arrays

  memory->destroy(d_shake_flag);
  memory->destroy(d_shake_atom);
  memory->destroy(d_shake_type);
  memory->destroy(xshake);
  memory->destroy(ftmp);
  memory->destroy(vtmp);


  delete [] bond_flag;
  delete [] angle_flag;
  delete [] type_flag;
  delete [] mass_list;

  delete [] d_bond_distance;
  delete [] d_angle_distance;

  if (output_every) {
    delete [] b_count;
    delete [] b_count_all;
    delete [] b_ave;
    delete [] b_ave_all;
    delete [] b_max;
    delete [] b_max_all;
    delete [] b_min;
    delete [] b_min_all;

    delete [] a_count;
    delete [] a_count_all;
    delete [] a_ave;
    delete [] a_ave_all;
    delete [] a_max;
    delete [] a_max_all;
    delete [] a_min;
    delete [] a_min_all;
  }

  memory->destroy(list);
}

/* ----------------------------------------------------------------------
   set bond and angle distances
   this init must happen after force->bond and force->angle inits
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::init()
{
  FixShake::init();

  if (strstr(update->integrate_style,"respa")) {
    error->all(FLERR,"Cannot yet use KOKKOS package with respa");
  }

  // set equilibrium bond distances

  for (i = 1; i <= atom->nbondtypes; i++)
    h_bond_distance[i] = bond_distance[i];

  // set equilibrium angle distances

  int nlocal = atom->nlocal;

  for (i = 1; i <= atom->nangletypes; i++) {
    if (angle_flag[i] == 0) continue;

  }
}

/* ----------------------------------------------------------------------
   SHAKE as pre-integrator constraint
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::setup(int vflag)
{
  pre_neighbor();

  if (output_every) stats();

  // setup SHAKE output

  bigint ntimestep = update->ntimestep;
  if (output_every) {
    next_output = ntimestep + output_every;
    if (ntimestep % output_every != 0)
      next_output = (ntimestep/output_every)*output_every + output_every;
  } else next_output = -1;

  // set respa to 0 if verlet is used and to 1 otherwise

  //if (strstr(update->integrate_style,"verlet"))
    respa = 0;
  //else
  //  respa = 1;

  //if (!respa) {
    dtv     = update->dt;
    dtfsq   = 0.5 * update->dt * update->dt * force->ftm2v;
    if (!rattle) dtfsq = update->dt * update->dt * force->ftm2v;
  //} else {
  //  dtv = step_respa[0];
  //  dtf_innerhalf = 0.5 * step_respa[0] * force->ftm2v;
  //  dtf_inner = dtf_innerhalf;
  //}

  // correct geometry of cluster if necessary

  correct_coordinates(vflag);

  // remove velocities along any bonds

  correct_velocities();

  // precalculate constraining forces for first integration step

  shake_end_of_step(vflag);
}

/* ----------------------------------------------------------------------
   build list of SHAKE clusters to constrain
   if one or more atoms in cluster are on this proc,
     this proc lists the cluster exactly once
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::pre_neighbor()
{
  int atom1,atom2,atom3,atom4;

  // local copies of atom quantities
  // used by SHAKE until next re-neighboring

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;

  // extend size of SHAKE list if necessary

  if (nlocal > maxlist) {
    maxlist = nlocal;
    memory->destroy(list);
    memory->create(list,maxlist,"shake:list");
  }

  // build list of SHAKE clusters I compute

  nlist = 0;

  for (int i = 0; i < nlocal; i++) // parallel_for, error flag
    if (d_shake_flag[i]) {
      if (d_shake_flag[i] == 2) {
        atom1 = atom->map(d_shake_atom(i,0));
        atom2 = atom->map(d_shake_atom(i,1));
        if (atom1 == -1 || atom2 == -1) {
          char str[128];
          sprintf(str,"Shake atoms "
                  "missing on proc %d at step " BIGINT_FORMAT,
                  me,update->ntimestep);
          d_error_flag = 1;
        }
        if (i <= atom1 && i <= atom2) d_list[nlist++] = i;
      } else if (d_shake_flag[i] % 2 == 1) {
        atom1 = atom->map(d_shake_atom(i,0));
        atom2 = atom->map(d_shake_atom(i,1));
        atom3 = atom->map(d_shake_atom(i,2));
        if (atom1 == -1 || atom2 == -1 || atom3 == -1) {
          char str[128];
          sprintf(str,"Shake atoms "
                  "missing on proc %d at step " BIGINT_FORMAT,
                  me,update->ntimestep);
          d_error_flag = 1;
        }
        if (i <= atom1 && i <= atom2 && i <= atom3) d_list[nlist++] = i;
      } else {
        atom1 = atom->map(d_shake_atom(i,0));
        atom2 = atom->map(d_shake_atom(i,1));
        atom3 = atom->map(d_shake_atom(i,2));
        atom4 = atom->map(d_shake_atom(i,3));
        if (atom1 == -1 || atom2 == -1 || atom3 == -1 || atom4 == -1) {
          char str[128];
          sprintf(str,"Shake atoms "
                  "missing on proc %d at step " BIGINT_FORMAT,
                  me,update->ntimestep);
          d_error_flag = 1;
        }
        if (i <= atom1 && i <= atom2 && i <= atom3 && i <= atom4)
          d_list[nlist++] = i;
      }
    }
}

  ///KK notes
  // need d_d_shake_flag
  // eflag, vflag
  // atom map (see neigh bond), may need to add KK hash for atom map
  // stats()--sync to host, don't do one device for now
  // comm->forward_comm_fix(this)
  // KK_INLINE_FUNC: shake(), shake3(), shake4(), shake3angle()
  // d_d_bond_distance, d_d_angle_distance
  // domain->minimum_image
  // domain->minimum_image_once
  // v_tally


/* ----------------------------------------------------------------------
   compute the force adjustment for SHAKE constraint
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::post_force(int vflag)
{
  if (update->ntimestep == next_output) stats();

  // xshake = unconstrained move with current v,f
  // communicate results if necessary

  unconstrained_update();
  if (nprocs > 1) comm->forward_comm_fix(this);

  // virial setup

  if (vflag) v_setup(vflag);
  else evflag = 0;

  // loop over clusters to add constraint forces

  int m;
  for (int i = 0; i < nlist; i++) { // parallel_for
    m = d_list[i];
    if (d_shake_flag[m] == 2) shake(m);
    else if (d_shake_flag[m] == 3) shake3(m);
    else if (d_shake_flag[m] == 4) shake4(m);
    else shake3angle(m);
  }

  // store vflag for coordinate_constraints_end_of_step()

  vflag_post_force = vflag;
}

/* ----------------------------------------------------------------------
   count # of degrees-of-freedom removed by SHAKE for atoms in igroup
------------------------------------------------------------------------- */

template<class DeviceType>
int FixShakeKokkos<DeviceType>::dof(int igroup)
{
  int groupbit = group->bitmask[igroup];

  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  // count dof in a cluster if and only if
  // the central atom is in group and atom i is the central atom

  int n = 0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (d_shake_flag[i] == 0) continue;
    if (d_shake_atom(i,0) != tag[i]) continue;
    if (d_shake_flag[i] == 1) n += 3;
    else if (d_shake_flag[i] == 2) n += 1;
    else if (d_shake_flag[i] == 3) n += 2;
    else if (d_shake_flag[i] == 4) n += 3;
  }

  int nall;
  MPI_Allreduce(&n,&nall,1,MPI_INT,MPI_SUM,world);
  return nall;
}

/* ----------------------------------------------------------------------
   update the unconstrained position of each atom
   only for SHAKE clusters, else set to 0.0
   assumes NVE update, seems to be accurate enough for NVT,NPT,NPH as well
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::unconstrained_update()
{
  double dtfmsq;

  if (rmass) {
    for (int i = 0; i < nlocal; i++) { ///// parallel_for
      if (d_shake_flag[i]) {
        dtfmsq = dtfsq / rmass[i];
        d_xshake(i,0) = x(i,0) + dtv*v(i,0) + dtfmsq*f(i,0);
        d_xshake(i,1) = x(i,1) + dtv*v(i,1) + dtfmsq*f(i,1);
        d_xshake(i,2) = x(i,2) + dtv*v(i,2) + dtfmsq*f(i,2);
      } else d_xshake(i,2) = d_xshake(i,1) = d_xshake(i,0) = 0.0;
    }
  } else {
    for (int i = 0; i < nlocal; i++) { ///// parallel_for
      if (d_shake_flag[i]) {
        dtfmsq = dtfsq / mass[type[i]];
        d_xshake(i,0) = x(i,0) + dtv*v(i,0) + dtfmsq*f(i,0);
        d_xshake(i,1) = x(i,1) + dtv*v(i,1) + dtfmsq*f(i,1);
        d_xshake(i,2) = x(i,2) + dtv*v(i,2) + dtfmsq*f(i,2);
      } else d_xshake(i,2) = d_xshake(i,1) = d_xshake(i,0) = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixShakeKokkos<DeviceType>::shake(int m)
{
  int nlist,d_list[2];
  double v[6];
  double invmass0,invmass1;

  // local atom IDs and constraint distances

  int i0 = atom->map(d_shake_atom(m,0));
  int i1 = atom->map(d_shake_atom(m,1));
  double bond1 = d_bond_distance[d_shake_type(m,0)];

  // r01 = distance vec between atoms, with PBC

  double r01[3];
  r01[0] = x(i0,0) - x(i1,0);
  r01[1] = x(i0,1) - x(i1,1);
  r01[2] = x(i0,2) - x(i1,2);
  domain->minimum_image(r01);

  // s01 = distance vec after unconstrained update, with PBC
  // use Domain::minimum_image_once(), not minimum_image()
  // b/c xshake values might be huge, due to e.g. fix gcmc

  double s01[3];
  s01[0] = d_xshake(i0,0) - d_xshake(i1,0);
  s01[1] = d_xshake(i0,1) - d_xshake(i1,1);
  s01[2] = d_xshake(i0,2) - d_xshake(i1,2);
  domain->minimum_image_once(s01);

  // scalar distances between atoms

  double r01sq = r01[0]*r01[0] + r01[1]*r01[1] + r01[2]*r01[2];
  double s01sq = s01[0]*s01[0] + s01[1]*s01[1] + s01[2]*s01[2];

  // a,b,c = coeffs in quadratic equation for lamda

  if (rmass) {
    invmass0 = 1.0/rmass[i0];
    invmass1 = 1.0/rmass[i1];
  } else {
    invmass0 = 1.0/mass[type[i0]];
    invmass1 = 1.0/mass[type[i1]];
  }

  double a = (invmass0+invmass1)*(invmass0+invmass1) * r01sq;
  double b = 2.0 * (invmass0+invmass1) *
    (s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2]);
  double c = s01sq - bond1*bond1;

  // error check

  double determ = b*b - 4.0*a*c;
  if (determ < 0.0) {
    error->warning(FLERR,"Shake determinant < 0.0",0);
    d_error_flag = 2;
    determ = 0.0;
  }

  // exact quadratic solution for lamda

  double lamda,lamda1,lamda2;
  lamda1 = (-b+sqrt(determ)) / (2.0*a);
  lamda2 = (-b-sqrt(determ)) / (2.0*a);

  if (fabs(lamda1) <= fabs(lamda2)) lamda = lamda1;
  else lamda = lamda2;

  // update forces if atom is owned by this processor

  lamda /= dtfsq;

  if (i0 < nlocal) {
    f(i0,0) += lamda*r01[0];
    f(i0,1) += lamda*r01[1];
    f(i0,2) += lamda*r01[2];
  }

  if (i1 < nlocal) {
    f(i1,0) -= lamda*r01[0];
    f(i1,1) -= lamda*r01[1];
    f(i1,2) -= lamda*r01[2];
  }

  if (evflag) {
    nlist = 0;
    if (i0 < nlocal) d_list[nlist++] = i0;
    if (i1 < nlocal) d_list[nlist++] = i1;

    v[0] = lamda*r01[0]*r01[0];
    v[1] = lamda*r01[1]*r01[1];
    v[2] = lamda*r01[2]*r01[2];
    v[3] = lamda*r01[0]*r01[1];
    v[4] = lamda*r01[0]*r01[2];
    v[5] = lamda*r01[1]*r01[2];

    v_tally(nlist,list,2.0,v);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixShakeKokkos<DeviceType>::shake3(int m)
{
  int nlist,d_list[3];
  double v[6];
  double invmass0,invmass1,invmass2;

  // local atom IDs and constraint distances

  int i0 = atom->map(d_shake_atom(m,0));
  int i1 = atom->map(d_shake_atom(m,1));
  int i2 = atom->map(d_shake_atom(m,2));
  double bond1 = d_bond_distance[d_shake_type(m,0)];
  double bond2 = d_bond_distance[d_shake_type(m,1)];

  // r01,r02 = distance vec between atoms, with PBC

  double r01[3];
  r01[0] = x(i0,0) - x(i1,0);
  r01[1] = x(i0,1) - x(i1,1);
  r01[2] = x(i0,2) - x(i1,2);
  domain->minimum_image(r01);

  double r02[3];
  r02[0] = x(i0,0) - x(i2,0);
  r02[1] = x(i0,1) - x(i2,1);
  r02[2] = x(i0,2) - x(i2,2);
  domain->minimum_image(r02);

  // s01,s02 = distance vec after unconstrained update, with PBC
  // use Domain::minimum_image_once(), not minimum_image()
  // b/c xshake values might be huge, due to e.g. fix gcmc

  double s01[3];
  s01[0] = d_xshake(i0,0) - d_xshake(i1,0);
  s01[1] = d_xshake(i0,1) - d_xshake(i1,1);
  s01[2] = d_xshake(i0,2) - d_xshake(i1,2);
  domain->minimum_image_once(s01);

  double s02[3];
  s02[0] = d_xshake(i0,0) - d_xshake(i2,0);
  s02[1] = d_xshake(i0,1) - d_xshake(i2,1);
  s02[2] = d_xshake(i0,2) - d_xshake(i2,2);
  domain->minimum_image_once(s02);

  // scalar distances between atoms

  double r01sq = r01[0]*r01[0] + r01[1]*r01[1] + r01[2]*r01[2];
  double r02sq = r02[0]*r02[0] + r02[1]*r02[1] + r02[2]*r02[2];
  double s01sq = s01[0]*s01[0] + s01[1]*s01[1] + s01[2]*s01[2];
  double s02sq = s02[0]*s02[0] + s02[1]*s02[1] + s02[2]*s02[2];

  // matrix coeffs and rhs for lamda equations

  if (rmass) {
    invmass0 = 1.0/rmass[i0];
    invmass1 = 1.0/rmass[i1];
    invmass2 = 1.0/rmass[i2];
  } else {
    invmass0 = 1.0/mass[type[i0]];
    invmass1 = 1.0/mass[type[i1]];
    invmass2 = 1.0/mass[type[i2]];
  }

  double a11 = 2.0 * (invmass0+invmass1) *
    (s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2]);
  double a12 = 2.0 * invmass0 *
    (s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2]);
  double a21 = 2.0 * invmass0 *
    (s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2]);
  double a22 = 2.0 * (invmass0+invmass2) *
    (s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2]);

  // inverse of matrix

  double determ = a11*a22 - a12*a21;
  if (determ == 0.0) d_error_flag = 3;
  //error->one(FLERR,"Shake determinant = 0.0");
  double determinv = 1.0/determ;

  double a11inv = a22*determinv;
  double a12inv = -a12*determinv;
  double a21inv = -a21*determinv;
  double a22inv = a11*determinv;

  // quadratic correction coeffs

  double r0102 = (r01[0]*r02[0] + r01[1]*r02[1] + r01[2]*r02[2]);

  double quad1_0101 = (invmass0+invmass1)*(invmass0+invmass1) * r01sq;
  double quad1_0202 = invmass0*invmass0 * r02sq;
  double quad1_0102 = 2.0 * (invmass0+invmass1)*invmass0 * r0102;

  double quad2_0202 = (invmass0+invmass2)*(invmass0+invmass2) * r02sq;
  double quad2_0101 = invmass0*invmass0 * r01sq;
  double quad2_0102 = 2.0 * (invmass0+invmass2)*invmass0 * r0102;

  // iterate until converged

  double lamda01 = 0.0;
  double lamda02 = 0.0;
  int niter = 0;
  int done = 0;

  double quad1,quad2,b1,b2,lamda01_new,lamda02_new;

  while (!done && niter < max_iter) {
    quad1 = quad1_0101 * lamda01*lamda01 + quad1_0202 * lamda02*lamda02 +
      quad1_0102 * lamda01*lamda02;
    quad2 = quad2_0101 * lamda01*lamda01 + quad2_0202 * lamda02*lamda02 +
      quad2_0102 * lamda01*lamda02;

    b1 = bond1*bond1 - s01sq - quad1;
    b2 = bond2*bond2 - s02sq - quad2;

    lamda01_new = a11inv*b1 + a12inv*b2;
    lamda02_new = a21inv*b1 + a22inv*b2;

    done = 1;
    if (fabs(lamda01_new-lamda01) > tolerance) done = 0;
    if (fabs(lamda02_new-lamda02) > tolerance) done = 0;

    lamda01 = lamda01_new;
    lamda02 = lamda02_new;

    // stop iterations before we have a floating point overflow
    // max double is < 1.0e308, so 1e150 is a reasonable cutoff

    if (fabs(lamda01) > 1e150 || fabs(lamda02) > 1e150) done = 1;

    niter++;
  }

  // update forces if atom is owned by this processor

  lamda01 = lamda01/dtfsq;
  lamda02 = lamda02/dtfsq;

  if (i0 < nlocal) {
    f(i0,0) += lamda01*r01[0] + lamda02*r02[0];
    f(i0,1) += lamda01*r01[1] + lamda02*r02[1];
    f(i0,2) += lamda01*r01[2] + lamda02*r02[2];
  }

  if (i1 < nlocal) {
    f(i1,0) -= lamda01*r01[0];
    f(i1,1) -= lamda01*r01[1];
    f(i1,2) -= lamda01*r01[2];
  }

  if (i2 < nlocal) {
    f(i2,0) -= lamda02*r02[0];
    f(i2,1) -= lamda02*r02[1];
    f(i2,2) -= lamda02*r02[2];
  }

  if (evflag) {
    nlist = 0;
    if (i0 < nlocal) d_list[nlist++] = i0;
    if (i1 < nlocal) d_list[nlist++] = i1;
    if (i2 < nlocal) d_list[nlist++] = i2;

    v[0] = lamda01*r01[0]*r01[0] + lamda02*r02[0]*r02[0];
    v[1] = lamda01*r01[1]*r01[1] + lamda02*r02[1]*r02[1];
    v[2] = lamda01*r01[2]*r01[2] + lamda02*r02[2]*r02[2];
    v[3] = lamda01*r01[0]*r01[1] + lamda02*r02[0]*r02[1];
    v[4] = lamda01*r01[0]*r01[2] + lamda02*r02[0]*r02[2];
    v[5] = lamda01*r01[1]*r01[2] + lamda02*r02[1]*r02[2];

    v_tally(nlist,list,3.0,v);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixShakeKokkos<DeviceType>::shake4(int m)
{
 int nlist,d_list[4];
  double v[6];
  double invmass0,invmass1,invmass2,invmass3;

  // local atom IDs and constraint distances

  int i0 = atom->map(d_shake_atom(m,0));
  int i1 = atom->map(d_shake_atom(m,1));
  int i2 = atom->map(d_shake_atom(m,2));
  int i3 = atom->map(d_shake_atom(m,3));
  double bond1 = d_bond_distance[d_shake_type(m,0)];
  double bond2 = d_bond_distance[d_shake_type(m,1)];
  double bond3 = d_bond_distance[d_shake_type(m,2)];

  // r01,r02,r03 = distance vec between atoms, with PBC

  double r01[3];
  r01[0] = x(i0,0) - x(i1,0);
  r01[1] = x(i0,1) - x(i1,1);
  r01[2] = x(i0,2) - x(i1,2);
  domain->minimum_image(r01);

  double r02[3];
  r02[0] = x(i0,0) - x(i2,0);
  r02[1] = x(i0,1) - x(i2,1);
  r02[2] = x(i0,2) - x(i2,2);
  domain->minimum_image(r02);

  double r03[3];
  r03[0] = x(i0,0) - x(i3,0);
  r03[1] = x(i0,1) - x(i3,1);
  r03[2] = x(i0,2) - x(i3,2);
  domain->minimum_image(r03);

  // s01,s02,s03 = distance vec after unconstrained update, with PBC
  // use Domain::minimum_image_once(), not minimum_image()
  // b/c xshake values might be huge, due to e.g. fix gcmc

  double s01[3];
  s01[0] = d_xshake(i0,0) - d_xshake(i1,0);
  s01[1] = d_xshake(i0,1) - d_xshake(i1,1);
  s01[2] = d_xshake(i0,2) - d_xshake(i1,2);
  domain->minimum_image_once(s01);

  double s02[3];
  s02[0] = d_xshake(i0,0) - d_xshake(i2,0);
  s02[1] = d_xshake(i0,1) - d_xshake(i2,1);
  s02[2] = d_xshake(i0,2) - d_xshake(i2,2);
  domain->minimum_image_once(s02);

  double s03[3];
  s03[0] = d_xshake(i0,0) - d_xshake(i3,0);
  s03[1] = d_xshake(i0,1) - d_xshake(i3,1);
  s03[2] = d_xshake(i0,2) - d_xshake(i3,2);
  domain->minimum_image_once(s03);

  // scalar distances between atoms

  double r01sq = r01[0]*r01[0] + r01[1]*r01[1] + r01[2]*r01[2];
  double r02sq = r02[0]*r02[0] + r02[1]*r02[1] + r02[2]*r02[2];
  double r03sq = r03[0]*r03[0] + r03[1]*r03[1] + r03[2]*r03[2];
  double s01sq = s01[0]*s01[0] + s01[1]*s01[1] + s01[2]*s01[2];
  double s02sq = s02[0]*s02[0] + s02[1]*s02[1] + s02[2]*s02[2];
  double s03sq = s03[0]*s03[0] + s03[1]*s03[1] + s03[2]*s03[2];

  // matrix coeffs and rhs for lamda equations

  if (rmass) {
    invmass0 = 1.0/rmass[i0];
    invmass1 = 1.0/rmass[i1];
    invmass2 = 1.0/rmass[i2];
    invmass3 = 1.0/rmass[i3];
  } else {
    invmass0 = 1.0/mass[type[i0]];
    invmass1 = 1.0/mass[type[i1]];
    invmass2 = 1.0/mass[type[i2]];
    invmass3 = 1.0/mass[type[i3]];
  }

  double a11 = 2.0 * (invmass0+invmass1) *
    (s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2]);
  double a12 = 2.0 * invmass0 *
    (s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2]);
  double a13 = 2.0 * invmass0 *
    (s01[0]*r03[0] + s01[1]*r03[1] + s01[2]*r03[2]);
  double a21 = 2.0 * invmass0 *
    (s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2]);
  double a22 = 2.0 * (invmass0+invmass2) *
    (s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2]);
  double a23 = 2.0 * invmass0 *
    (s02[0]*r03[0] + s02[1]*r03[1] + s02[2]*r03[2]);
  double a31 = 2.0 * invmass0 *
    (s03[0]*r01[0] + s03[1]*r01[1] + s03[2]*r01[2]);
  double a32 = 2.0 * invmass0 *
    (s03[0]*r02[0] + s03[1]*r02[1] + s03[2]*r02[2]);
  double a33 = 2.0 * (invmass0+invmass3) *
    (s03[0]*r03[0] + s03[1]*r03[1] + s03[2]*r03[2]);

  // inverse of matrix;

  double determ = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 -
    a11*a23*a32 - a12*a21*a33 - a13*a22*a31;
  if (determ == 0.0) d_error_flag = 3;
  //error->one(FLERR,"Shake determinant = 0.0");
  double determinv = 1.0/determ;

  double a11inv = determinv * (a22*a33 - a23*a32);
  double a12inv = -determinv * (a12*a33 - a13*a32);
  double a13inv = determinv * (a12*a23 - a13*a22);
  double a21inv = -determinv * (a21*a33 - a23*a31);
  double a22inv = determinv * (a11*a33 - a13*a31);
  double a23inv = -determinv * (a11*a23 - a13*a21);
  double a31inv = determinv * (a21*a32 - a22*a31);
  double a32inv = -determinv * (a11*a32 - a12*a31);
  double a33inv = determinv * (a11*a22 - a12*a21);

  // quadratic correction coeffs

  double r0102 = (r01[0]*r02[0] + r01[1]*r02[1] + r01[2]*r02[2]);
  double r0103 = (r01[0]*r03[0] + r01[1]*r03[1] + r01[2]*r03[2]);
  double r0203 = (r02[0]*r03[0] + r02[1]*r03[1] + r02[2]*r03[2]);

  double quad1_0101 = (invmass0+invmass1)*(invmass0+invmass1) * r01sq;
  double quad1_0202 = invmass0*invmass0 * r02sq;
  double quad1_0303 = invmass0*invmass0 * r03sq;
  double quad1_0102 = 2.0 * (invmass0+invmass1)*invmass0 * r0102;
  double quad1_0103 = 2.0 * (invmass0+invmass1)*invmass0 * r0103;
  double quad1_0203 = 2.0 * invmass0*invmass0 * r0203;

  double quad2_0101 = invmass0*invmass0 * r01sq;
  double quad2_0202 = (invmass0+invmass2)*(invmass0+invmass2) * r02sq;
  double quad2_0303 = invmass0*invmass0 * r03sq;
  double quad2_0102 = 2.0 * (invmass0+invmass2)*invmass0 * r0102;
  double quad2_0103 = 2.0 * invmass0*invmass0 * r0103;
  double quad2_0203 = 2.0 * (invmass0+invmass2)*invmass0 * r0203;

  double quad3_0101 = invmass0*invmass0 * r01sq;
  double quad3_0202 = invmass0*invmass0 * r02sq;
  double quad3_0303 = (invmass0+invmass3)*(invmass0+invmass3) * r03sq;
  double quad3_0102 = 2.0 * invmass0*invmass0 * r0102;
  double quad3_0103 = 2.0 * (invmass0+invmass3)*invmass0 * r0103;
  double quad3_0203 = 2.0 * (invmass0+invmass3)*invmass0 * r0203;

  // iterate until converged

  double lamda01 = 0.0;
  double lamda02 = 0.0;
  double lamda03 = 0.0;
  int niter = 0;
  int done = 0;

  double quad1,quad2,quad3,b1,b2,b3,lamda01_new,lamda02_new,lamda03_new;

  while (!done && niter < max_iter) {
    quad1 = quad1_0101 * lamda01*lamda01 +
      quad1_0202 * lamda02*lamda02 +
      quad1_0303 * lamda03*lamda03 +
      quad1_0102 * lamda01*lamda02 +
      quad1_0103 * lamda01*lamda03 +
      quad1_0203 * lamda02*lamda03;

    quad2 = quad2_0101 * lamda01*lamda01 +
      quad2_0202 * lamda02*lamda02 +
      quad2_0303 * lamda03*lamda03 +
      quad2_0102 * lamda01*lamda02 +
      quad2_0103 * lamda01*lamda03 +
      quad2_0203 * lamda02*lamda03;

    quad3 = quad3_0101 * lamda01*lamda01 +
      quad3_0202 * lamda02*lamda02 +
      quad3_0303 * lamda03*lamda03 +
      quad3_0102 * lamda01*lamda02 +
      quad3_0103 * lamda01*lamda03 +
      quad3_0203 * lamda02*lamda03;

    b1 = bond1*bond1 - s01sq - quad1;
    b2 = bond2*bond2 - s02sq - quad2;
    b3 = bond3*bond3 - s03sq - quad3;

    lamda01_new = a11inv*b1 + a12inv*b2 + a13inv*b3;
    lamda02_new = a21inv*b1 + a22inv*b2 + a23inv*b3;
    lamda03_new = a31inv*b1 + a32inv*b2 + a33inv*b3;

    done = 1;
    if (fabs(lamda01_new-lamda01) > tolerance) done = 0;
    if (fabs(lamda02_new-lamda02) > tolerance) done = 0;
    if (fabs(lamda03_new-lamda03) > tolerance) done = 0;

    lamda01 = lamda01_new;
    lamda02 = lamda02_new;
    lamda03 = lamda03_new;

    // stop iterations before we have a floating point overflow
    // max double is < 1.0e308, so 1e150 is a reasonable cutoff

    if (fabs(lamda01) > 1e150 || fabs(lamda02) > 1e150
        || fabs(lamda03) > 1e150) done = 1;

    niter++;
  }

  // update forces if atom is owned by this processor

  lamda01 = lamda01/dtfsq;
  lamda02 = lamda02/dtfsq;
  lamda03 = lamda03/dtfsq;

  if (i0 < nlocal) {
    f(i0,0) += lamda01*r01[0] + lamda02*r02[0] + lamda03*r03[0];
    f(i0,1) += lamda01*r01[1] + lamda02*r02[1] + lamda03*r03[1];
    f(i0,2) += lamda01*r01[2] + lamda02*r02[2] + lamda03*r03[2];
  }

  if (i1 < nlocal) {
    f(i1,0) -= lamda01*r01[0];
    f(i1,1) -= lamda01*r01[1];
    f(i1,2) -= lamda01*r01[2];
  }

  if (i2 < nlocal) {
    f(i2,0) -= lamda02*r02[0];
    f(i2,1) -= lamda02*r02[1];
    f(i2,2) -= lamda02*r02[2];
  }

  if (i3 < nlocal) {
    f(i3,0) -= lamda03*r03[0];
    f(i3,1) -= lamda03*r03[1];
    f(i3,2) -= lamda03*r03[2];
  }

  if (evflag) {
    nlist = 0;
    if (i0 < nlocal) d_list[nlist++] = i0;
    if (i1 < nlocal) d_list[nlist++] = i1;
    if (i2 < nlocal) d_list[nlist++] = i2;
    if (i3 < nlocal) d_list[nlist++] = i3;

    v[0] = lamda01*r01[0]*r01[0]+lamda02*r02[0]*r02[0]+lamda03*r03[0]*r03[0];
    v[1] = lamda01*r01[1]*r01[1]+lamda02*r02[1]*r02[1]+lamda03*r03[1]*r03[1];
    v[2] = lamda01*r01[2]*r01[2]+lamda02*r02[2]*r02[2]+lamda03*r03[2]*r03[2];
    v[3] = lamda01*r01[0]*r01[1]+lamda02*r02[0]*r02[1]+lamda03*r03[0]*r03[1];
    v[4] = lamda01*r01[0]*r01[2]+lamda02*r02[0]*r02[2]+lamda03*r03[0]*r03[2];
    v[5] = lamda01*r01[1]*r01[2]+lamda02*r02[1]*r02[2]+lamda03*r03[1]*r03[2];

    v_tally(nlist,list,4.0,v);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixShakeKokkos<DeviceType>::shake3angle(int m)
{
  int nlist,d_list[3];
  double v[6];
  double invmass0,invmass1,invmass2;

  // local atom IDs and constraint distances

  int i0 = atom->map(d_shake_atom(m,0));
  int i1 = atom->map(d_shake_atom(m,1));
  int i2 = atom->map(d_shake_atom(m,2));
  double bond1 = d_bond_distance[d_shake_type(m,0)];
  double bond2 = d_bond_distance[d_shake_type(m,1)];
  double bond12 = d_angle_distance[d_shake_type(m,2)];

  // r01,r02,r12 = distance vec between atoms, with PBC

  double r01[3];
  r01[0] = x(i0,0) - x(i1,0);
  r01[1] = x(i0,1) - x(i1,1);
  r01[2] = x(i0,2) - x(i1,2);
  domain->minimum_image(r01);

  double r02[3];
  r02[0] = x(i0,0) - x(i2,0);
  r02[1] = x(i0,1) - x(i2,1);
  r02[2] = x(i0,2) - x(i2,2);
  domain->minimum_image(r02);

  double r12[3];
  r12[0] = x(i1,0) - x(i2,0);
  r12[1] = x(i1,1) - x(i2,1);
  r12[2] = x(i1,2) - x(i2,2);
  domain->minimum_image(r12);

  // s01,s02,s12 = distance vec after unconstrained update, with PBC
  // use Domain::minimum_image_once(), not minimum_image()
  // b/c xshake values might be huge, due to e.g. fix gcmc

  double s01[3];
  s01[0] = d_xshake(i0,0) - d_xshake(i1,0);
  s01[1] = d_xshake(i0,1) - d_xshake(i1,1);
  s01[2] = d_xshake(i0,2) - d_xshake(i1,2);
  domain->minimum_image_once(s01);

  double s02[3];
  s02[0] = d_xshake(i0,0) - d_xshake(i2,0);
  s02[1] = d_xshake(i0,1) - d_xshake(i2,1);
  s02[2] = d_xshake(i0,2) - d_xshake(i2,2);
  domain->minimum_image_once(s02);

  double s12[3];
  s12[0] = d_xshake(i1,0) - d_xshake(i2,0);
  s12[1] = d_xshake(i1,1) - d_xshake(i2,1);
  s12[2] = d_xshake(i1,2) - d_xshake(i2,2);
  domain->minimum_image_once(s12);

  // scalar distances between atoms

  double r01sq = r01[0]*r01[0] + r01[1]*r01[1] + r01[2]*r01[2];
  double r02sq = r02[0]*r02[0] + r02[1]*r02[1] + r02[2]*r02[2];
  double r12sq = r12[0]*r12[0] + r12[1]*r12[1] + r12[2]*r12[2];
  double s01sq = s01[0]*s01[0] + s01[1]*s01[1] + s01[2]*s01[2];
  double s02sq = s02[0]*s02[0] + s02[1]*s02[1] + s02[2]*s02[2];
  double s12sq = s12[0]*s12[0] + s12[1]*s12[1] + s12[2]*s12[2];

  // matrix coeffs and rhs for lamda equations

  if (rmass) {
    invmass0 = 1.0/rmass[i0];
    invmass1 = 1.0/rmass[i1];
    invmass2 = 1.0/rmass[i2];
  } else {
    invmass0 = 1.0/mass[type[i0]];
    invmass1 = 1.0/mass[type[i1]];
    invmass2 = 1.0/mass[type[i2]];
  }

  double a11 = 2.0 * (invmass0+invmass1) *
    (s01[0]*r01[0] + s01[1]*r01[1] + s01[2]*r01[2]);
  double a12 = 2.0 * invmass0 *
    (s01[0]*r02[0] + s01[1]*r02[1] + s01[2]*r02[2]);
  double a13 = - 2.0 * invmass1 *
    (s01[0]*r12[0] + s01[1]*r12[1] + s01[2]*r12[2]);
  double a21 = 2.0 * invmass0 *
    (s02[0]*r01[0] + s02[1]*r01[1] + s02[2]*r01[2]);
  double a22 = 2.0 * (invmass0+invmass2) *
    (s02[0]*r02[0] + s02[1]*r02[1] + s02[2]*r02[2]);
  double a23 = 2.0 * invmass2 *
    (s02[0]*r12[0] + s02[1]*r12[1] + s02[2]*r12[2]);
  double a31 = - 2.0 * invmass1 *
    (s12[0]*r01[0] + s12[1]*r01[1] + s12[2]*r01[2]);
  double a32 = 2.0 * invmass2 *
    (s12[0]*r02[0] + s12[1]*r02[1] + s12[2]*r02[2]);
  double a33 = 2.0 * (invmass1+invmass2) *
    (s12[0]*r12[0] + s12[1]*r12[1] + s12[2]*r12[2]);

  // inverse of matrix

  double determ = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 -
    a11*a23*a32 - a12*a21*a33 - a13*a22*a31;
  if (determ == 0.0) d_error_flag = 3;
  //error->one(FLERR,"Shake determinant = 0.0");
  double determinv = 1.0/determ;

  double a11inv = determinv * (a22*a33 - a23*a32);
  double a12inv = -determinv * (a12*a33 - a13*a32);
  double a13inv = determinv * (a12*a23 - a13*a22);
  double a21inv = -determinv * (a21*a33 - a23*a31);
  double a22inv = determinv * (a11*a33 - a13*a31);
  double a23inv = -determinv * (a11*a23 - a13*a21);
  double a31inv = determinv * (a21*a32 - a22*a31);
  double a32inv = -determinv * (a11*a32 - a12*a31);
  double a33inv = determinv * (a11*a22 - a12*a21);

  // quadratic correction coeffs

  double r0102 = (r01[0]*r02[0] + r01[1]*r02[1] + r01[2]*r02[2]);
  double r0112 = (r01[0]*r12[0] + r01[1]*r12[1] + r01[2]*r12[2]);
  double r0212 = (r02[0]*r12[0] + r02[1]*r12[1] + r02[2]*r12[2]);

  double quad1_0101 = (invmass0+invmass1)*(invmass0+invmass1) * r01sq;
  double quad1_0202 = invmass0*invmass0 * r02sq;
  double quad1_1212 = invmass1*invmass1 * r12sq;
  double quad1_0102 = 2.0 * (invmass0+invmass1)*invmass0 * r0102;
  double quad1_0112 = - 2.0 * (invmass0+invmass1)*invmass1 * r0112;
  double quad1_0212 = - 2.0 * invmass0*invmass1 * r0212;

  double quad2_0101 = invmass0*invmass0 * r01sq;
  double quad2_0202 = (invmass0+invmass2)*(invmass0+invmass2) * r02sq;
  double quad2_1212 = invmass2*invmass2 * r12sq;
  double quad2_0102 = 2.0 * (invmass0+invmass2)*invmass0 * r0102;
  double quad2_0112 = 2.0 * invmass0*invmass2 * r0112;
  double quad2_0212 = 2.0 * (invmass0+invmass2)*invmass2 * r0212;

  double quad3_0101 = invmass1*invmass1 * r01sq;
  double quad3_0202 = invmass2*invmass2 * r02sq;
  double quad3_1212 = (invmass1+invmass2)*(invmass1+invmass2) * r12sq;
  double quad3_0102 = - 2.0 * invmass1*invmass2 * r0102;
  double quad3_0112 = - 2.0 * (invmass1+invmass2)*invmass1 * r0112;
  double quad3_0212 = 2.0 * (invmass1+invmass2)*invmass2 * r0212;

  // iterate until converged

  double lamda01 = 0.0;
  double lamda02 = 0.0;
  double lamda12 = 0.0;
  int niter = 0;
  int done = 0;

  double quad1,quad2,quad3,b1,b2,b3,lamda01_new,lamda02_new,lamda12_new;

  while (!done && niter < max_iter) {

    quad1 = quad1_0101 * lamda01*lamda01 +
      quad1_0202 * lamda02*lamda02 +
      quad1_1212 * lamda12*lamda12 +
      quad1_0102 * lamda01*lamda02 +
      quad1_0112 * lamda01*lamda12 +
      quad1_0212 * lamda02*lamda12;

    quad2 = quad2_0101 * lamda01*lamda01 +
      quad2_0202 * lamda02*lamda02 +
      quad2_1212 * lamda12*lamda12 +
      quad2_0102 * lamda01*lamda02 +
      quad2_0112 * lamda01*lamda12 +
      quad2_0212 * lamda02*lamda12;

    quad3 = quad3_0101 * lamda01*lamda01 +
      quad3_0202 * lamda02*lamda02 +
      quad3_1212 * lamda12*lamda12 +
      quad3_0102 * lamda01*lamda02 +
      quad3_0112 * lamda01*lamda12 +
      quad3_0212 * lamda02*lamda12;

    b1 = bond1*bond1 - s01sq - quad1;
    b2 = bond2*bond2 - s02sq - quad2;
    b3 = bond12*bond12 - s12sq - quad3;

    lamda01_new = a11inv*b1 + a12inv*b2 + a13inv*b3;
    lamda02_new = a21inv*b1 + a22inv*b2 + a23inv*b3;
    lamda12_new = a31inv*b1 + a32inv*b2 + a33inv*b3;

    done = 1;
    if (fabs(lamda01_new-lamda01) > tolerance) done = 0;
    if (fabs(lamda02_new-lamda02) > tolerance) done = 0;
    if (fabs(lamda12_new-lamda12) > tolerance) done = 0;

    lamda01 = lamda01_new;
    lamda02 = lamda02_new;
    lamda12 = lamda12_new;

    // stop iterations before we have a floating point overflow
    // max double is < 1.0e308, so 1e150 is a reasonable cutoff

    if (fabs(lamda01) > 1e150 || fabs(lamda02) > 1e150
        || fabs(lamda12) > 1e150) done = 1;

    niter++;
  }

  // update forces if atom is owned by this processor

  lamda01 = lamda01/dtfsq;
  lamda02 = lamda02/dtfsq;
  lamda12 = lamda12/dtfsq;

  if (i0 < nlocal) {
    f(i0,0) += lamda01*r01[0] + lamda02*r02[0];
    f(i0,1) += lamda01*r01[1] + lamda02*r02[1];
    f(i0,2) += lamda01*r01[2] + lamda02*r02[2];
  }

  if (i1 < nlocal) {
    f(i1,0) -= lamda01*r01[0] - lamda12*r12[0];
    f(i1,1) -= lamda01*r01[1] - lamda12*r12[1];
    f(i1,2) -= lamda01*r01[2] - lamda12*r12[2];
  }

  if (i2 < nlocal) {
    f(i2,0) -= lamda02*r02[0] + lamda12*r12[0];
    f(i2,1) -= lamda02*r02[1] + lamda12*r12[1];
    f(i2,2) -= lamda02*r02[2] + lamda12*r12[2];
  }

  if (evflag) {
    nlist = 0;
    if (i0 < nlocal) d_list[nlist++] = i0;
    if (i1 < nlocal) d_list[nlist++] = i1;
    if (i2 < nlocal) d_list[nlist++] = i2;

    v[0] = lamda01*r01[0]*r01[0]+lamda02*r02[0]*r02[0]+lamda12*r12[0]*r12[0];
    v[1] = lamda01*r01[1]*r01[1]+lamda02*r02[1]*r02[1]+lamda12*r12[1]*r12[1];
    v[2] = lamda01*r01[2]*r01[2]+lamda02*r02[2]*r02[2]+lamda12*r12[2]*r12[2];
    v[3] = lamda01*r01[0]*r01[1]+lamda02*r02[0]*r02[1]+lamda12*r12[0]*r12[1];
    v[4] = lamda01*r01[0]*r01[2]+lamda02*r02[0]*r02[2]+lamda12*r12[0]*r12[2];
    v[5] = lamda01*r01[1]*r01[2]+lamda02*r02[1]*r02[2]+lamda12*r12[1]*r12[2];

    v_tally(nlist,list,3.0,v);
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::grow_arrays(int nmax)
{
  memory->grow(d_shake_flag,nmax,"shake:d_shake_flag");
  memory->grow(d_shake_atom,nmax,4,"shake:d_shake_atom");
  memory->grow(d_shake_type,nmax,3,"shake:d_shake_type");
  memory->destroy(xshake);
  memory->create(xshake,nmax,3,"shake:xshake");
  memory->destroy(ftmp);
  memory->create(ftmp,nmax,3,"shake:ftmp");
  memory->destroy(vtmp);
  memory->create(vtmp,nmax,3,"shake:vtmp");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::copy_arrays(int i, int j, int /*delflag*/)
{
  int flag = d_shake_flag[j] = d_shake_flag[i];
  if (flag == 1) {
    d_shake_atom(j,0) = d_shake_atom(i,0);
    d_shake_atom(j,1) = d_shake_atom(i,1);
    d_shake_atom(j,2) = d_shake_atom(i,2);
    d_shake_type(j,0) = d_shake_type(i,0);
    d_shake_type(j,1) = d_shake_type(i,1);
    d_shake_type(j,2) = d_shake_type(i,2);
  } else if (flag == 2) {
    d_shake_atom(j,0) = d_shake_atom(i,0);
    d_shake_atom(j,1) = d_shake_atom(i,1);
    d_shake_type(j,0) = d_shake_type(i,0);
  } else if (flag == 3) {
    d_shake_atom(j,0) = d_shake_atom(i,0);
    d_shake_atom(j,1) = d_shake_atom(i,1);
    d_shake_atom(j,2) = d_shake_atom(i,2);
    d_shake_type(j,0) = d_shake_type(i,0);
    d_shake_type(j,1) = d_shake_type(i,1);
  } else if (flag == 4) {
    d_shake_atom(j,0) = d_shake_atom(i,0);
    d_shake_atom(j,1) = d_shake_atom(i,1);
    d_shake_atom(j,2) = d_shake_atom(i,2);
    d_shake_atom(j,3) = d_shake_atom(i,3);
    d_shake_type(j,0) = d_shake_type(i,0);
    d_shake_type(j,1) = d_shake_type(i,1);
    d_shake_type(j,2) = d_shake_type(i,2);
  }
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::set_arrays(int i)
{
  d_shake_flag[i] = 0;
}

/* ----------------------------------------------------------------------
   update one atom's array values
   called when molecule is created from fix gcmc
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::update_arrays(int i, int atom_offset)
{
  int flag = d_shake_flag[i];

  if (flag == 1) {
    d_shake_atom(i,0) += atom_offset;
    d_shake_atom(i,1) += atom_offset;
    d_shake_atom(i,2) += atom_offset;
  } else if (flag == 2) {
    d_shake_atom(i,0) += atom_offset;
    d_shake_atom(i,1) += atom_offset;
  } else if (flag == 3) {
    d_shake_atom(i,0) += atom_offset;
    d_shake_atom(i,1) += atom_offset;
    d_shake_atom(i,2) += atom_offset;
  } else if (flag == 4) {
    d_shake_atom(i,0) += atom_offset;
    d_shake_atom(i,1) += atom_offset;
    d_shake_atom(i,2) += atom_offset;
    d_shake_atom(i,3) += atom_offset;
  }
}

///* ----------------------------------------------------------------------
//   initialize a molecule inserted by another fix, e.g. deposit or pour
//   called when molecule is created
//   nlocalprev = # of atoms on this proc before molecule inserted
//   tagprev = atom ID previous to new atoms in the molecule
//   xgeom,vcm,quat ignored
//------------------------------------------------------------------------- */
//
//void FixShakeKokkos<DeviceType>::set_molecule(int nlocalprev, tagint tagprev, int imol,
//                            double * /*xgeom*/, double * /*vcm*/, double * /*quat*/)
//{
//  int m,flag;
//
//  int nlocal = atom->nlocal;
//  if (nlocalprev == nlocal) return;
//
//  tagint *tag = atom->tag;
//  tagint **mol_d_shake_atom = onemols[imol]->d_shake_atom;
//  int **mol_d_shake_type = onemols[imol]->d_shake_type;
//
//  for (int i = nlocalprev; i < nlocal; i++) {
//    m = tag[i] - tagprev-1;
//
//    flag = d_shake_flag[i] = onemols[imol]->d_shake_flag[m];
//
//    if (flag == 1) {
//      d_shake_atom(i,0) = mol_d_shake_atom(m,0) + tagprev;
//      d_shake_atom(i,1) = mol_d_shake_atom(m,1) + tagprev;
//      d_shake_atom(i,2) = mol_d_shake_atom(m,2) + tagprev;
//      d_shake_type(i,0) = mol_d_shake_type(m,0);
//      d_shake_type(i,1) = mol_d_shake_type(m,1);
//      d_shake_type(i,2) = mol_d_shake_type(m,2);
//    } else if (flag == 2) {
//      d_shake_atom(i,0) = mol_d_shake_atom(m,0) + tagprev;
//      d_shake_atom(i,1) = mol_d_shake_atom(m,1) + tagprev;
//      d_shake_type(i,0) = mol_d_shake_type(m,0);
//    } else if (flag == 3) {
//      d_shake_atom(i,0) = mol_d_shake_atom(m,0) + tagprev;
//      d_shake_atom(i,1) = mol_d_shake_atom(m,1) + tagprev;
//      d_shake_atom(i,2) = mol_d_shake_atom(m,2) + tagprev;
//      d_shake_type(i,0) = mol_d_shake_type(m,0);
//      d_shake_type(i,1) = mol_d_shake_type(m,1);
//    } else if (flag == 4) {
//      d_shake_atom(i,0) = mol_d_shake_atom(m,0) + tagprev;
//      d_shake_atom(i,1) = mol_d_shake_atom(m,1) + tagprev;
//      d_shake_atom(i,2) = mol_d_shake_atom(m,2) + tagprev;
//      d_shake_atom(i,3) = mol_d_shake_atom(m,3) + tagprev;
//      d_shake_type(i,0) = mol_d_shake_type(m,0);
//      d_shake_type(i,1) = mol_d_shake_type(m,1);
//      d_shake_type(i,2) = mol_d_shake_type(m,2);
//    }
//  }
//}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int FixShakeKokkos<DeviceType>::pack_exchange_kokkos(int i, double *buf)
{
  int m = 0;
  d_buf[m++] = d_shake_flag[i];
  int flag = d_shake_flag[i];
  if (flag == 1) {
    d_buf[m++] = d_shake_atom(i,0);
    d_buf[m++] = d_shake_atom[i,1);
    d_buf[m++] = d_shake_atom[i,2);
    d_buf[m++] = d_shake_type[i,0);
    d_buf[m++] = d_shake_type[i,1);
    d_buf[m++] = d_shake_type[i,2);
  } else if (flag == 2) {
    d_buf[m++] = d_shake_atom[i,0);
    d_buf[m++] = d_shake_atom[i,1);
    d_buf[m++] = d_shake_type[i,0);
  } else if (flag == 3) {
    d_buf[m++] = d_shake_atom[i,0);
    d_buf[m++] = d_shake_atom[i,1);
    d_buf[m++] = d_shake_atom[i,2);
    d_buf[m++] = d_shake_type[i,0);
    d_buf[m++] = d_shake_type[i,1);
  } else if (flag == 4) {
    d_buf[m++] = d_shake_atom[i,0);
    d_buf[m++] = d_shake_atom[i,1);
    d_buf[m++] = d_shake_atom[i,2);
    d_buf[m++] = d_shake_atom[i,3);
    d_buf[m++] = d_shake_type[i,0);
    d_buf[m++] = d_shake_type[i,1);
    d_buf[m++] = d_shake_type[i,2);
  }
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int FixShakeKokkos<DeviceType>::unpack_exchange_kokkos(int nlocal, double *buf)
{
  int m = 0;
  int flag = d_shake_flag[nlocal] = static_cast<int> (d_buf[m++]);
  if (flag == 1) {
    d_shake_atom(nlocal,0) = static_cast<tagint> (d_buf[m++]);
    d_shake_atom(nlocal,1) = static_cast<tagint> (d_buf[m++]);
    d_shake_atom(nlocal,2) = static_cast<tagint> (d_buf[m++]);
    d_shake_type(nlocal,0) = static_cast<int> (d_buf[m++]);
    d_shake_type(nlocal,1) = static_cast<int> (d_buf[m++]);
    d_shake_type(nlocal,2) = static_cast<int> (d_buf[m++]);
  } else if (flag == 2) {
    d_shake_atom(nlocal,0) = static_cast<tagint> (d_buf[m++]);
    d_shake_atom(nlocal,1) = static_cast<tagint> (d_buf[m++]);
    d_shake_type(nlocal,0) = static_cast<int> (d_buf[m++]);
  } else if (flag == 3) {
    d_shake_atom(nlocal,0) = static_cast<tagint> (d_buf[m++]);
    d_shake_atom(nlocal,1) = static_cast<tagint> (d_buf[m++]);
    d_shake_atom(nlocal,2) = static_cast<tagint> (d_buf[m++]);
    d_shake_type(nlocal,0) = static_cast<int> (d_buf[m++]);
    d_shake_type(nlocal,1) = static_cast<int> (d_buf[m++]);
  } else if (flag == 4) {
    d_shake_atom(nlocal,0) = static_cast<tagint> (d_buf[m++]);
    d_shake_atom(nlocal,1) = static_cast<tagint> (d_buf[m++]);
    d_shake_atom(nlocal,2) = static_cast<tagint> (d_buf[m++]);
    d_shake_atom(nlocal,3) = static_cast<tagint> (d_buf[m++]);
    d_shake_type(nlocal,0) = static_cast<int> (d_buf[m++]);
    d_shake_type(nlocal,1) = static_cast<int> (d_buf[m++]);
    d_shake_type(nlocal,2) = static_cast<int> (d_buf[m++]);
  }
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixShakeKokkos<DeviceType>::pack_forward_comm_kokkos(int n, int *list, double *buf,
                                int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = d_list[i];
      d_buf[m++] = d_xshake(j,0);
      d_buf[m++] = d_xshake(j,1);
      d_buf[m++] = d_xshake(j,2);
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = d_list[i];
      d_buf[m++] = d_xshake(j,0) + dx;
      d_buf[m++] = d_xshake(j,1) + dy;
      d_buf[m++] = d_xshake(j,2) + dz;
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    d_xshake(i,0) = d_buf[m++];
    d_xshake(i,1) = d_buf[m++];
    d_xshake(i,2) = d_buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::reset_dt()
{
  dtv = update->dt;
  if (rattle) dtfsq   = 0.5 * update->dt * update->dt * force->ftm2v;
  else dtfsq = update->dt * update->dt * force->ftm2v;
}

/* ----------------------------------------------------------------------
   add coordinate constraining forces
   this method is called at the end of a timestep
------------------------------------------------------------------------- */

template<class DeviceType>
void FixShakeKokkos<DeviceType>::shake_end_of_step(int vflag) {
  dtv     = update->dt;
  dtfsq   = 0.5 * update->dt * update->dt * force->ftm2v;
  FixShakeKokkos<DeviceType>::post_force(vflag);
  if (!rattle) dtfsq = update->dt * update->dt * force->ftm2v;
}