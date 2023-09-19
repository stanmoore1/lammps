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

#include "fix_adapt_kokkos.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "domain.h"
#include "error.h"
#include "fix_store_atom.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{PAIR,KSPACE,ATOM,BOND,ANGLE};
enum{DIAMETER,CHARGE};

/* ---------------------------------------------------------------------- */

FixAdapt::FixAdapt(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), nadapt(0), anypair(0), anybond(0), anyangle(0),
  id_fix_diam(nullptr), id_fix_chg(nullptr), adapt(nullptr)
{

}

/* ---------------------------------------------------------------------- */

FixAdapt::~FixAdapt()
{
  if (copymode) return;


}

/* ----------------------------------------------------------------------
   if need to restore per-atom quantities, create new fix STORE styles
------------------------------------------------------------------------- */

void FixAdapt::post_constructor()
{
  if (!resetflag) return;
  if (!diam_flag && !chgflag) return;

  // new id = fix-ID + FIX_STORE_ATTRIBUTE
  // new fix group = group for this fix

  id_fix_diam = nullptr;
  id_fix_chg = nullptr;

  if (diam_flag && atom->radius_flag) {
    id_fix_diam = utils::strdup(id + std::string("_FIX_STORE_DIAM"));
    fix_diam = dynamic_cast<FixStoreAtom *>(
      modify->add_fix(fmt::format("{} {} STORE/ATOM 1 0 0 1", id_fix_diam,group->names[igroup])));
    if (fix_diam->restart_reset) fix_diam->restart_reset = 0;
    else {
      double *vec = fix_diam->vstore;
      double *radius = atom->radius;
      int *mask = atom->mask;
      int nlocal = atom->nlocal;

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) vec[i] = radius[i];
        else vec[i] = 0.0;
      }
    }
  }

  if (chgflag && atom->q_flag) {
    id_fix_chg = utils::strdup(id + std::string("_FIX_STORE_CHG"));
    fix_chg = dynamic_cast<FixStoreAtom *>(
      modify->add_fix(fmt::format("{} {} STORE/ATOM 1 0 0 1",id_fix_chg,group->names[igroup])));
    if (fix_chg->restart_reset) fix_chg->restart_reset = 0;
    else {
      double *vec = fix_chg->vstore;
      double *q = atom->q;
      int *mask = atom->mask;
      int nlocal = atom->nlocal;

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) vec[i] = q[i];
        else vec[i] = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixAdapt::init()
{

}

/* ----------------------------------------------------------------------
   change pair,bond,angle,kspace,atom parameters based on variable evaluation
------------------------------------------------------------------------- */

void FixAdapt::change_settings()
{
  int i,j;

  // variable evaluation may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];
    double value = input->variable->compute_equal(ad->ivar);

    // set global scalar or type pair array values

    if (ad->which == PAIR) {
      if (ad->pdim == 0) {
        if (scaleflag) *ad->scalar = value * ad->scalar_orig;
        else *ad->scalar = value;
      } else if (ad->pdim == 2) {
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; i++)
            for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
              ad->array[i][j] = value*ad->array_orig[i][j];
        else
          for (i = ad->ilo; i <= ad->ihi; i++)
            for (j = MAX(ad->jlo,i); j <= ad->jhi; j++)
              ad->array[i][j] = value;
      }

    // set bond type array values:

    } else if (ad->which == BOND) {
      if (ad->bdim == 1) {
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value*ad->vector_orig[i];
        else
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value;
      }

    // set angle type array values:

    } else if (ad->which == ANGLE) {
      if (ad->adim == 1) {
        if (scaleflag)
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value*ad->vector_orig[i];
        else
          for (i = ad->ilo; i <= ad->ihi; ++i )
            ad->vector[i] = value;
      }

    // set kspace scale factor

    } else if (ad->which == KSPACE) {
      *kspace_scale = value;

    // set per atom values, also make changes for ghost atoms

    } else if (ad->which == ATOM) {

      // reset radius to new value, for both owned and ghost atoms
      // also reset rmass to new value assuming density remains constant
      // for scaleflag, previous_diam_scale is the scale factor on previous step

      if (ad->atomparam == DIAMETER) {
        double scale;
        double *radius = atom->radius;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        int nall = nlocal + atom->nghost;

        if (scaleflag) scale = value / previous_diam_scale;

        for (i = 0; i < nall; i++) {
          if (mask[i] & groupbit) {
            if (massflag) {
              if (!scaleflag) scale = 0.5*value / radius[i];
              if (discflag) rmass[i] *= scale*scale;
              else rmass[i] *= scale*scale*scale;
            }
            if (scaleflag) radius[i] *= scale;
            else radius[i] = 0.5*value;
          }
        }

        if (scaleflag) previous_diam_scale = value;

      // reset charge to new value, for both owned and ghost atoms
      // for scaleflag, previous_chg_scale is the scale factor on previous step

      } else if (ad->atomparam == CHARGE) {
        double scale;
        double *q = atom->q;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        int nall = nlocal + atom->nghost;

        if (scaleflag) scale = value / previous_chg_scale;

        for (i = 0; i < nall; i++) {
          if (mask[i] & groupbit) {
            if (scaleflag) q[i] *= scale;
            else q[i] = value;
          }
        }

        if (scaleflag) previous_chg_scale = value;
      }
    }
  }

  modify->addstep_compute(update->ntimestep + nevery);

  // re-initialize pair styles if any PAIR settings were changed
  // ditto for bond/angle styles if any BOND/ANGLE settings were changed
  // this resets other coeffs that may depend on changed values,
  //   and also offset and tail corrections
  // we must call force->pair->reinit() instead of the individual
  // adapted pair styles so that also the top-level
  // tail correction values are updated for hybrid pair styles.
  //  same for bond styles

  if (anypair) force->pair->reinit();
  if (anybond) force->bond->reinit();
  if (anyangle) force->angle->reinit();

  // reset KSpace charges if charges have changed

  if (chgflag && force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   restore pair,kspace,atom parameters to original values
------------------------------------------------------------------------- */

void FixAdapt::restore_settings()
{
  for (int m = 0; m < nadapt; m++) {
    Adapt *ad = &adapt[m];
    if (ad->which == PAIR) {
      if (ad->pdim == 0) *ad->scalar = ad->scalar_orig;
      else if (ad->pdim == 2) {
        for (int i = ad->ilo; i <= ad->ihi; i++)
          for (int j = MAX(ad->jlo,i); j <= ad->jhi; j++)
            ad->array[i][j] = ad->array_orig[i][j];
      }

    } else if (ad->which == BOND) {
      if (ad->bdim == 1) {
        for (int i = ad->ilo; i <= ad->ihi; i++)
          ad->vector[i] = ad->vector_orig[i];
      }

    } else if (ad->which == ANGLE) {
      if (ad->adim == 1) {
        for (int i = ad->ilo; i <= ad->ihi; i++)
          ad->vector[i] = ad->vector_orig[i];
      }

    } else if (ad->which == KSPACE) {
      *kspace_scale = 1.0;

    } else if (ad->which == ATOM) {
      if (diam_flag) {
        double scale;

        double *vec = fix_diam->vstore;
        double *radius = atom->radius;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        if (scaleflag) scale = previous_diam_scale;

        for (int i = 0; i < nlocal; i++)
          if (mask[i] & groupbit) {
            if (massflag) {
              if (!scaleflag) scale = vec[i] / radius[i];
              if (discflag) rmass[i] *= scale*scale;
              else rmass[i] *= scale*scale*scale;
            }
            radius[i] = vec[i];
          }
      }
      if (chgflag) {
        double *vec = fix_chg->vstore;
        double *q = atom->q;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++)
          if (mask[i] & groupbit) q[i] = vec[i];
      }
    }
  }

  if (anypair) force->pair->reinit();
  if (anybond) force->bond->reinit();
  if (anyangle) force->angle->reinit();
  if (chgflag && force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   initialize one atom's storage values, called when atom is created
------------------------------------------------------------------------- */

void FixAdapt::set_arrays(int i)
{
  if (fix_diam) fix_diam->vstore[i] = atom->radius[i];
  if (fix_chg) fix_chg->vstore[i] = atom->q[i];
}
