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

#include "fix_minimize_kokkos.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMinimizeKokkos::FixMinimizeKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixMinimize(lmp, narg, arg),
  nvector(0), peratom(NULL), vectors(NULL)
{
}

/* ---------------------------------------------------------------------- */

FixMinimizeKokkos::~FixMinimizeKokkos()
{

}

/* ----------------------------------------------------------------------
   allocate/initialize memory for a new vector with N elements per atom
------------------------------------------------------------------------- */

void FixMinimizeKokkos::add_vector(int n)
{
  memory->grow(peratom,nvector+1,"minimize:peratom");
  peratom[nvector] = n;

  vectors = (double **)
    memory->srealloc(vectors,(nvector+1)*sizeof(double *),"minimize:vectors");
  memory->create(vectors[nvector],atom->nmax*n,"minimize:vector");

  int ntotal = n*atom->nlocal;
  for (int i = 0; i < ntotal; i++) vectors[nvector][i] = 0.0;
  nvector++;
}

/* ----------------------------------------------------------------------
   return a pointer to the Mth vector
------------------------------------------------------------------------- */

double *FixMinimizeKokkos::request_vector(int m)
{
  return vectors[m];
}

/* ----------------------------------------------------------------------
   reset x0 for atoms that moved across PBC via reneighboring in line search
   x0 = 1st vector
   must do minimum_image using original box stored at beginning of line search
   swap & set_global_box() change to original box, then restore current box
------------------------------------------------------------------------- */

void FixMinimizeKokkos::reset_coords()
{
  box_swap();
  domain->set_global_box();

  double **x = atom->x;
  double *x0 = vectors[0];
  int nlocal = atom->nlocal;
  double dx,dy,dz,dx0,dy0,dz0;

  int n = 0;
  for (int i = 0; i < nlocal; i++) {
    dx = dx0 = x[i][0] - x0[n];
    dy = dy0 = x[i][1] - x0[n+1];
    dz = dz0 = x[i][2] - x0[n+2];
    domain->minimum_image(dx,dy,dz);
    if (dx != dx0) x0[n] = x[i][0] - dx;
    if (dy != dy0) x0[n+1] = x[i][1] - dy;
    if (dz != dz0) x0[n+2] = x[i][2] - dz;
    n += 3;
  }

  box_swap();
  domain->set_global_box();
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixMinimizeKokkos::grow_arrays(int nmax)
{
  for (int m = 0; m < nvector; m++)
    memory->grow(vectors[m],peratom[m]*nmax,"minimize:vector");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixMinimizeKokkos::copy_arrays(int i, int j, int /*delflag*/)
{
  int m,iper,nper,ni,nj;

  for (m = 0; m < nvector; m++) {
    nper = peratom[m];
    ni = nper*i;
    nj = nper*j;
    for (iper = 0; iper < nper; iper++) vectors[m][nj++] = vectors[m][ni++];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixMinimizeKokkos::pack_exchange(int i, double *buf)
{
  int m,iper,nper,ni;

  int n = 0;
  for (m = 0; m < nvector; m++) {
    nper = peratom[m];
    ni = nper*i;
    for (iper = 0; iper < nper; iper++) buf[n++] = vectors[m][ni++];
  }
  return n;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixMinimizeKokkos::unpack_exchange(int nlocal, double *buf)
{
  int m,iper,nper,ni;

  int n = 0;
  for (m = 0; m < nvector; m++) {
    nper = peratom[m];
    ni = nper*nlocal;
    for (iper = 0; iper < nper; iper++) vectors[m][ni++] = buf[n++];
  }
  return n;
}
