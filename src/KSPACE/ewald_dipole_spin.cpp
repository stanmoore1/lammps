// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)
                         Stan Moore (SNL)

   Please cite the related publication:
   Tranchida, J., Plimpton, S. J., Thibaudeau, P., & Thompson, A. P. (2018).
   Massively parallel symplectic algorithm for coupled magnetic spin dynamics
   and molecular dynamics. Journal of Computational Physics.
------------------------------------------------------------------------- */

#include "ewald_dipole_spin.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr double SMALL = 0.00001;

/* ---------------------------------------------------------------------- */

EwaldDipoleSpin::EwaldDipoleSpin(LAMMPS *lmp) : EwaldDipole(lmp)
{
  ewaldflag = spinflag = 1;
  dipoleflag = 0;

  hbar       = force->hplanck / MY_2PI;    // eV/(rad.THz)
  mub        = 9.274e-4;                   // A.Ang^2
  mu_0       = 784.15;                     // eV/Ang/A^2
  mub2mu0    = mub * mub * mu_0 / (4.0*MY_PI);   // eV.Ang^3
  mub2mu0hbinv = mub2mu0 / hbar;          // rad.THz
}

/* ----------------------------------------------------------------------
   called once before run
------------------------------------------------------------------------- */

void EwaldDipoleSpin::init()
{
  if (comm->me == 0) utils::logmesg(lmp,"EwaldDipoleSpin initialization ...\n");

  // error checks

  if (!atom->sp_flag)
    error->all(FLERR,"Kspace style ewald/dipole/spin requires atom/spin style");

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use EwaldDipoleSpin with 2d simulation");

  triclinic_check();
  triclinic = domain->triclinic;
  if (triclinic)
    error->all(FLERR,"Cannot (yet) use EwaldDipoleSpin with triclinic box");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with EwaldDipoleSpin");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab EwaldDipoleSpin");
  }

  // extract short-range spin dipole cutoff from pair style

  pair_check();

  int itmp;
  auto *p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
  if (p_cutoff == nullptr)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  double cutoff = *p_cutoff;

  // compute spsqsum and mu2 = spsqsum * mub2mu0

  scale = 1.0;
  musum_musq();
  natoms_original = atom->natoms;

  // set accuracy (force units) from accuracy_relative or accuracy_absolute
  // use mub2mu0hbinv as reference force scale for spin systems

  two_charge_force = mub2mu0hbinv;
  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

  // setup K-space resolution

  bigint natoms = atom->natoms;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd * slab_volfactor;

  // make initial g_ewald estimate

  if (!gewaldflag) {
    if (accuracy <= 0.0)
      error->all(FLERR,"KSpace accuracy must be > 0");

    g_ewald = accuracy * sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*mu2);
    if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
    else g_ewald = sqrt(-log(g_ewald)) / cutoff;

    double g_ewald_new =
      NewtonSolve(g_ewald,cutoff,natoms,xprd*yprd*zprd,mu2);
    if (g_ewald_new > 0.0) g_ewald = g_ewald_new;
    else error->warning(FLERR,"Ewald/dipole/spin Newton solver failed, "
                        "using old method to estimate g_ewald");
  }

  // setup EwaldDipoleSpin coefficients

  setup();

  // stats

  if (comm->me == 0) {
    std::string mesg = fmt::format("  G vector (1/distance) = {:.8g}\n",g_ewald);
    mesg += fmt::format("  KSpace vectors: actual max1d max3d = {} {} {}\n",
                        kcount,kmax,kmax3d);
    mesg += fmt::format("                  kxmax kymax kzmax  = {} {} {}\n",
                        kxmax,kymax,kzmax);
    utils::logmesg(lmp,mesg);
  }
}

/* ----------------------------------------------------------------------
   compute the EwaldDipoleSpin long-range force, energy, virial
   the k-space contribution to fm[i] (spin precession field) is:
     fm[i] -= (sp[i][3]*mub/hbar) * muscale * tk[i]
   where tk[i] stores the k-space effective field at atom i
------------------------------------------------------------------------- */

void EwaldDipoleSpin::compute(int eflag, int vflag)
{
  int i,j,k;
  const double g3 = g_ewald*g_ewald*g_ewald;

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = evflag_atom = eflag_global = vflag_global =
         eflag_atom = vflag_atom = 0;

  if (atom->natoms != natoms_original) {
    musum_musq();
    natoms_original = atom->natoms;
  }

  if (musqsum == 0.0) return;

  if (atom->nmax > nmax) {
    memory->destroy(ek);
    memory->destroy(tk);
    memory->destroy(vc);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald_dipole_spin:ek");
    memory->create(tk,nmax,3,"ewald_dipole_spin:tk");
    memory->create(vc,kmax3d,6,"ewald_dipole_spin:vc");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald_dipole_spin:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald_dipole_spin:sn");
    kmax_created = kmax;
  }

  // partial structure factors on each processor
  // total structure factor by summing over procs

  eik_dot_r();

  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  // K-space portion of the effective field
  // double loop over K-vectors and local atoms

  double **f  = atom->f;
  double **fm = atom->fm;
  double **sp = atom->sp;
  int nlocal = atom->nlocal;

  int kx,ky,kz;
  double cypz,sypz,exprl,expim;
  double partial,partial_peratom;
  double vcik[6];
  double mudotk;
  double mux,muy,muz;

  for (i = 0; i < nlocal; i++) {
    ek[i][0] = ek[i][1] = ek[i][2] = 0.0;
    tk[i][0] = tk[i][1] = tk[i][2] = 0.0;
  }

  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    for (j = 0; j < 6; j++) vc[k][j] = 0.0;

    for (i = 0; i < nlocal; i++) {

      for (j = 0; j < 6; j++) vcik[j] = 0.0;

      // effective magnetic moment components: mu_eff = sp[3]*mub * sp[0..2]

      mux = sp[i][0] * sp[i][3] * mub;
      muy = sp[i][1] * sp[i][3] * mub;
      muz = sp[i][2] * sp[i][3] * mub;

      mudotk = mux*kx*unitk[0] + muy*ky*unitk[1] + muz*kz*unitk[2];

      // calculating re and im of exp(i*k*ri)

      cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
      sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
      exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
      expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;

      // taking im of struct_fact x exp(i*k*ri) (for force calc.)

      partial = mudotk*(expim*sfacrl_all[k] - exprl*sfacim_all[k]);
      ek[i][0] += partial * eg[k][0];
      ek[i][1] += partial * eg[k][1];
      ek[i][2] += partial * eg[k][2];

      // compute field for spin precession calculation

      partial_peratom = exprl*sfacrl_all[k] + expim*sfacim_all[k];
      tk[i][0] += partial_peratom * eg[k][0];
      tk[i][1] += partial_peratom * eg[k][1];
      tk[i][2] += partial_peratom * eg[k][2];

      // total and per-atom virial correction

      vc[k][0] += vcik[0] = -(partial_peratom * mux * eg[k][0]);
      vc[k][1] += vcik[1] = -(partial_peratom * muy * eg[k][1]);
      vc[k][2] += vcik[2] = -(partial_peratom * muz * eg[k][2]);
      vc[k][3] += vcik[3] = -(partial_peratom * mux * eg[k][1]);
      vc[k][4] += vcik[4] = -(partial_peratom * mux * eg[k][2]);
      vc[k][5] += vcik[5] = -(partial_peratom * muy * eg[k][2]);

      if (evflag_atom) {
        if (eflag_atom) eatom[i] += mudotk*ug[k]*partial_peratom;
        if (vflag_atom)
          for (j = 0; j < 6; j++)
            vatom[i][j] += (ug[k]*mudotk*vg[k][j]*partial_peratom - vcik[j]);
      }
    }
  }

  // force and spin precession field contribution from k-space
  // muscale = mu_0/(4pi) (magnetic scale factor)

  const double muscale = mub2mu0 / (mub*mub) * scale;

  for (i = 0; i < nlocal; i++) {
    f[i][0] += muscale * ek[i][0];
    f[i][1] += muscale * ek[i][1];
    if (slabflag != 2) f[i][2] += muscale * ek[i][2];

    // fm contribution: fm -= (sp[3]*mub/hbar) * muscale * tk
    // the sign follows from the magnetic torque analogy with EwaldDipole

    const double spinpre = sp[i][3] * mub / hbar * muscale;
    fm[i][0] -= spinpre * tk[i][0];
    fm[i][1] -= spinpre * tk[i][1];
    if (slabflag != 2) fm[i][2] -= spinpre * tk[i][2];
  }

  // global energy: sum over k-vectors, subtract self energy, then scale

  if (eflag_global) {
    for (k = 0; k < kcount; k++) {
      energy += ug[k] * (sfacrl_all[k]*sfacrl_all[k] +
                         sfacim_all[k]*sfacim_all[k]);
    }
    energy -= musqsum * 2.0*g3/3.0/MY_PIS;
    energy *= muscale;
  }

  // global virial

  if (vflag_global) {
    double uk;
    for (k = 0; k < kcount; k++) {
      uk = ug[k] * (sfacrl_all[k]*sfacrl_all[k] + sfacim_all[k]*sfacim_all[k]);
      for (j = 0; j < 6; j++) virial[j] += uk*vg[k][j] - vc[k][j];
    }
    for (j = 0; j < 6; j++) virial[j] *= muscale;
  }

  // per-atom energy/virial including self-energy correction

  if (evflag_atom) {
    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        double musq_i = sp[i][0]*sp[i][3]*mub * sp[i][0]*sp[i][3]*mub
                      + sp[i][1]*sp[i][3]*mub * sp[i][1]*sp[i][3]*mub
                      + sp[i][2]*sp[i][3]*mub * sp[i][2]*sp[i][3]*mub;
        eatom[i] -= musq_i * 2.0*g3/3.0/MY_PIS;
        eatom[i] *= muscale;
      }
    }
    if (vflag_atom)
      for (i = 0; i < nlocal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= muscale;
  }

  // 2d slab correction

  if (slabflag == 1) slabcorr();
}

/* ----------------------------------------------------------------------
   compute the structure factors using spin magnetic moments
   mu_eff[i] = sp[i][3] * mub * sp[i][0..2]
------------------------------------------------------------------------- */

void EwaldDipoleSpin::eik_dot_r()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm;
  double mux,muy,muz;
  double mudotk;

  double **x  = atom->x;
  double **sp = atom->sp;
  int nlocal = atom->nlocal;

  n = 0;
  mux = muy = muz = 0.0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      for (i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
        sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
        mudotk = (sp[i][ic] * sp[i][3] * mub * unitk[ic]);
        cstr1 += mudotk*cs[1][ic][i];
        sstr1 += mudotk*sn[1][ic][i];
      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
          sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];
          mudotk = (sp[i][ic] * sp[i][3] * mub * m*unitk[ic]);
          cstr1 += mudotk*cs[m][ic][i];
          sstr1 += mudotk*sn[m][ic][i];
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          mux = sp[i][0] * sp[i][3] * mub;
          muy = sp[i][1] * sp[i][3] * mub;

          mudotk = (mux*k*unitk[0] + muy*l*unitk[1]);
          cstr1 += mudotk*(cs[k][0][i]*cs[l][1][i]-sn[k][0][i]*sn[l][1][i]);
          sstr1 += mudotk*(sn[k][0][i]*cs[l][1][i]+cs[k][0][i]*sn[l][1][i]);

          mudotk = (mux*k*unitk[0] - muy*l*unitk[1]);
          cstr2 += mudotk*(cs[k][0][i]*cs[l][1][i]+sn[k][0][i]*sn[l][1][i]);
          sstr2 += mudotk*(sn[k][0][i]*cs[l][1][i]-cs[k][0][i]*sn[l][1][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          muy = sp[i][1] * sp[i][3] * mub;
          muz = sp[i][2] * sp[i][3] * mub;

          mudotk = (muy*l*unitk[1] + muz*m*unitk[2]);
          cstr1 += mudotk*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
          sstr1 += mudotk*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);

          mudotk = (muy*l*unitk[1] - muz*m*unitk[2]);
          cstr2 += mudotk*(cs[l][1][i]*cs[m][2][i]+sn[l][1][i]*sn[m][2][i]);
          sstr2 += mudotk*(sn[l][1][i]*cs[m][2][i]-cs[l][1][i]*sn[m][2][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          mux = sp[i][0] * sp[i][3] * mub;
          muz = sp[i][2] * sp[i][3] * mub;

          mudotk = (mux*k*unitk[0] + muz*m*unitk[2]);
          cstr1 += mudotk*(cs[k][0][i]*cs[m][2][i]-sn[k][0][i]*sn[m][2][i]);
          sstr1 += mudotk*(sn[k][0][i]*cs[m][2][i]+cs[k][0][i]*sn[m][2][i]);

          mudotk = (mux*k*unitk[0] - muz*m*unitk[2]);
          cstr2 += mudotk*(cs[k][0][i]*cs[m][2][i]+sn[k][0][i]*sn[m][2][i]);
          sstr2 += mudotk*(sn[k][0][i]*cs[m][2][i]-cs[k][0][i]*sn[m][2][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          for (i = 0; i < nlocal; i++) {
            mux = sp[i][0] * sp[i][3] * mub;
            muy = sp[i][1] * sp[i][3] * mub;
            muz = sp[i][2] * sp[i][3] * mub;

            // dir 1: (k,l,m)
            mudotk = (mux*k*unitk[0] + muy*l*unitk[1] + muz*m*unitk[2]);
            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            cstr1 += mudotk*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr1 += mudotk*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            // dir 2: (k,-l,m)
            mudotk = (mux*k*unitk[0] - muy*l*unitk[1] + muz*m*unitk[2]);
            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            cstr2 += mudotk*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr2 += mudotk*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            // dir 3: (k,l,-m)
            mudotk = (mux*k*unitk[0] + muy*l*unitk[1] - muz*m*unitk[2]);
            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            cstr3 += mudotk*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr3 += mudotk*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            // dir 4: (k,-l,-m)
            mudotk = (mux*k*unitk[0] - muy*l*unitk[1] - muz*m*unitk[2]);
            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            cstr4 += mudotk*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr4 += mudotk*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
          }
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   slab-geometry correction for spin magnetic dipoles
   analogous to EwaldDipole::slabcorr but uses sp instead of mu
   and fm instead of torque
------------------------------------------------------------------------- */

void EwaldDipoleSpin::slabcorr()
{
  // compute local z-component of spin dipole moment sum

  double dipole = 0.0;
  double **sp = atom->sp;
  double **fm = atom->fm;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) dipole += sp[i][2] * sp[i][3] * mub;

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  if (eflag_atom || fabs(qsum) > SMALL)
    error->all(FLERR,"Cannot (yet) use kspace slab correction with "
               "long-range spin dipoles and non-neutral systems or per-atom energy");

  const double muscale = mub2mu0 / (mub*mub) * scale;
  const double e_slabcorr = MY_2PI*(dipole_all*dipole_all/12.0)/volume;

  if (eflag_global) energy += muscale * e_slabcorr;

  // per-atom energy

  if (eflag_atom) {
    double efact = muscale * MY_2PI/volume/12.0;
    for (int i = 0; i < nlocal; i++)
      eatom[i] += efact * sp[i][2]*sp[i][3]*mub * dipole_all;
  }

  // add spin precession field correction (z-direction only)
  // fm[i][2] += (sp[i][3]*mub/hbar) * B_slab_z
  // where B_slab_z = ffact * dipole_all, ffact = muscale * (-4pi/V)

  const double ffact = muscale * (-4.0*MY_PI/volume);
  for (int i = 0; i < nlocal; i++) {
    fm[i][2] += (sp[i][3]*mub/hbar) * ffact * dipole_all;
  }
}

/* ----------------------------------------------------------------------
   compute spsqsum, musqsum (using spin moments), and mu2
   mu2 = spsqsum * mub2mu0  (sum of sp[i][3]^2 * mub2mu0)
------------------------------------------------------------------------- */

void EwaldDipoleSpin::musum_musq()
{
  const int nlocal = atom->nlocal;

  musum = musqsum = mu2 = 0.0;
  if (atom->sp_flag) {
    double **sp = atom->sp;
    double musum_local(0.0), musqsum_local(0.0);

    for (int i = 0; i < nlocal; i++) {
      musum_local += sp[i][3];
      // |mu_eff|^2 = (sp[i][3]*mub)^2 since sp[0..2] is normalized
      musqsum_local += sp[i][3]*sp[i][3] * mub*mub;
    }

    MPI_Allreduce(&musum_local,&musum,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&musqsum_local,&musqsum,1,MPI_DOUBLE,MPI_SUM,world);

    // mu2 = musqsum * (mu_0/4pi) = musqsum * mub2mu0/mub^2
    mu2 = musqsum * mub2mu0 / (mub*mub);
  }

  if (mu2 == 0 && comm->me == 0)
    error->all(FLERR,"Using kspace solver ewald/dipole/spin on system with no spins");
}
