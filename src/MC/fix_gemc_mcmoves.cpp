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

/* ----------------------------------------------------------------------
   Contributing author: Andrew Hong, Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include "fix_gemc.h"

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"

#include <cmath>

using namespace LAMMPS_NS;

// trial energies above this value (or not a number) are always rejected

static constexpr double MAXENERGYTEST = 1.0e50;

/* ----------------------------------------------------------------------
   update box dimensions after changing boxhi, as done by Verlet
   when the box changes
------------------------------------------------------------------------- */

static void reset_box_dims(Domain *domain, Comm *comm, Neighbor *neighbor)
{
  domain->set_global_box();
  domain->set_local_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
}

/* ----------------------------------------------------------------------
  Shrink one box and expand the other one by the same volume
------------------------------------------------------------------------- */

void FixGEMC::attempt_volume_change_full()
{
  nvolume_attempts += 1.0;

  // random walk in logvolratio = log(V1/V2), identical in both boxes
  // V1 = Vtotal/(1+exp(-logvolratio)), V2 = Vtotal/(1+exp(logvolratio))
  // box volumes are always positive and Vtotal is conserved

  double dlogvolratio = max_dlogvolratio * (2.0 * random_universe->uniform() - 1.0);

  // fvolume = Vnew/Vold of my box

  double fvolume;
  if (myworld == 0)
    fvolume = (1.0 + exp(-logvolratio)) / (1.0 + exp(-(logvolratio + dlogvolratio)));
  else
    fvolume = (1.0 + exp(logvolratio)) / (1.0 + exp(logvolratio + dlogvolratio));

  double scale_length = cbrt(fvolume);

  // scale box toward its lower corner and all atom positions with it

  domain->x2lamda(atom->nlocal);
  for (auto &ifix : rfix) ifix->deform(0);

  domain->boxhi[0] = xlo + (xhi - xlo) * scale_length;
  domain->boxhi[1] = ylo + (yhi - ylo) * scale_length;
  domain->boxhi[2] = zlo + (zhi - zlo) * scale_length;
  reset_box_dims(domain, comm, neighbor);

  domain->lamda2x(atom->nlocal);
  for (auto &ifix : rfix) ifix->deform(1);

  // Frenkel & Smit, 3rd Ed. (2023), Eq. (6.6.10) for a random walk in log(V1/V2):
  // acc = (V1new/V1old)^(N1+1) * (V2new/V2old)^(N2+1) * exp(-beta*(dU1+dU2))
  // each box contributes dU - (N+1)*kT*log(Vnew/Vold)

  double energy_after = energy_full();
  double dU =
      energy_after - energy_stored - (atom->natoms + 1) * force->boltz * box_temp * log(fvolume);
  int overflow = !(energy_after < MAXENERGYTEST);

  if (accept_both(dU, overflow)) {
    nvolume_successes += 1.0;
    logvolratio += dlogvolratio;
    energy_stored = energy_after;
    xhi = domain->boxhi[0];
    yhi = domain->boxhi[1];
    zhi = domain->boxhi[2];

  } else {

    // rejected: restore box and atom positions

    domain->x2lamda(atom->nlocal);
    for (auto &ifix : rfix) ifix->deform(0);

    domain->boxhi[0] = xhi;
    domain->boxhi[1] = yhi;
    domain->boxhi[2] = zhi;
    reset_box_dims(domain, comm, neighbor);

    domain->lamda2x(atom->nlocal);
    for (auto &ifix : rfix) ifix->deform(1);
    ghosts_stale = 1;
  }

  // atoms may have migrated between ranks, so the local list must be regenerated

  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
  Move a randomly chosen atom from one box to a random position in the other box
------------------------------------------------------------------------- */

void FixGEMC::attempt_atomic_exchange_full()
{
  nexchange_attempts += 1.0;

  // choose donor box with equal probability, identical in both boxes

  int donor = (random_universe->uniform() < 0.5) ? 0 : 1;
  int sender = (myworld == donor) ? 1 : 0;

  // each box independently decides whether it can use single-atom energies

  int local = use_local();

  double energy_before = energy_stored;
  double volume = (xhi - xlo) * (yhi - ylo) * (zhi - zlo);
  int nold = natom_total;
  double denergy = 0.0;

  // donor box: pick an atom
  // with full energy: remove it and keep its complete state for a possible restore
  // with local energy: only compute its energy, remove it after acceptance
  // donor_info = {box is empty, atom type, atom mask}

  int donor_info[3] = {0, 0, 0};
  int iremove = -1;
  if (sender) {
    if (natom_total == 0) {
      donor_info[0] = 1;
    } else {
      iremove = pick_random_gas_atom();
      int info[2] = {0, 0};
      double eatom = 0.0;
      if (iremove >= 0) {
        info[0] = atom->type[iremove];
        info[1] = atom->mask[iremove];
        if (local) {
          eatom = energy_local(iremove, info[0], atom->tag[iremove], atom->x[iremove]);
        } else {
          size_t nbuf = atom->avec->maxexchange + 1024;
          for (const auto &ifix : modify->get_fix_list()) nbuf += ifix->maxexchange;
          if (exchange_buf.size() < nbuf) exchange_buf.resize(nbuf);
          atom->avec->pack_exchange(iremove, exchange_buf.data());
          atom->avec->copy(atom->nlocal - 1, iremove, 1);
          atom->nlocal--;
        }
      }
      MPI_Allreduce(info, &donor_info[1], 2, MPI_INT, MPI_MAX, world);
      if (local) {
        MPI_Allreduce(&eatom, &denergy, 1, MPI_DOUBLE, MPI_SUM, world);
        denergy = -denergy;
      } else {
        atom->natoms--;
      }
    }
  }

  // receiver box needs the donor info

  if (me == 0) {
    int other[3];
    MPI_Sendrecv(donor_info, 3, MPI_INT, 1 - myworld, 0, other, 3, MPI_INT, 1 - myworld, 0,
                 comm_replica, MPI_STATUS_IGNORE);
    if (!sender)
      for (int k = 0; k < 3; k++) donor_info[k] = other[k];
  }
  MPI_Bcast(donor_info, 3, MPI_INT, 0, world);

  // donor box is empty: reject without doing anything

  if (donor_info[0]) return;

  // receiver box: insert an atom of the same type and group membership at a random position
  // with local energy: compute its energy in a scratch slot and insert it after acceptance

  int itype = donor_info[1];
  double coord[3];
  int proc_flag = 0;
  tagint newtag = 0;
  if (!sender) {
    if (me == 0) {
      coord[0] = xlo + random_proc->uniform() * (xhi - xlo);
      coord[1] = ylo + random_proc->uniform() * (yhi - ylo);
      coord[2] = zlo + random_proc->uniform() * (zhi - zlo);
    }
    MPI_Bcast(coord, 3, MPI_DOUBLE, 0, world);
    if ((coord[0] >= sublo[0]) && (coord[0] < subhi[0]) && (coord[1] >= sublo[1]) &&
        (coord[1] < subhi[1]) && (coord[2] >= sublo[2]) && (coord[2] < subhi[2]))
      proc_flag = 1;

    if (local) {
      double eatom = 0.0;
      if (proc_flag) {
        int ii = atom->nlocal + atom->nghost;
        if (ii >= atom->nmax) atom->avec->grow(0);
        atom->type[ii] = itype;
        atom->mask[ii] = donor_info[2];
        atom->tag[ii] = 0;
        if (atom->q_flag) atom->q[ii] = 0.0;
        atom->x[ii][0] = coord[0];
        atom->x[ii][1] = coord[1];
        atom->x[ii][2] = coord[2];
        eatom = energy_local(ii, itype, 0, coord);
      }
      MPI_Allreduce(&eatom, &denergy, 1, MPI_DOUBLE, MPI_SUM, world);
    } else {
      newtag = insert_atom(itype, donor_info[2], coord, proc_flag);
    }
  }
  if (!local) {
    if (atom->map_style != Atom::MAP_NONE) atom->map_init();
    if (force->pair && force->pair->tail_flag) force->pair->reinit();
  }

  // Frenkel & Smit, 3rd Ed. (2023), Eq. (6.6.11)
  // acc = (Ndonor,old * Vreceiver) / (Vdonor * (Nreceiver,old+1)) * exp(-beta*(dU1+dU2))
  // each box contributes dU - kT*log(its factor)

  double energy_after = local ? energy_before + denergy : energy_full();
  double logfactor;
  if (sender)
    logfactor = log(nold / volume);
  else
    logfactor = log(volume / (nold + 1));
  double dU = energy_after - energy_before - force->boltz * box_temp * logfactor;
  int overflow = !(energy_after < MAXENERGYTEST);

  if (accept_both(dU, overflow)) {
    nexchange_successes += 1.0;
    energy_stored = energy_after;

    // with local energy the move has not been applied yet

    if (local) {
      if (sender) {
        if (iremove >= 0) {
          atom->avec->copy(atom->nlocal - 1, iremove, 1);
          atom->nlocal--;
        }
        atom->natoms--;
      } else {
        insert_atom(itype, donor_info[2], coord, proc_flag);
      }
      if (atom->map_style != Atom::MAP_NONE) atom->map_init();
      refresh_ghosts();
    }

  } else if (!local) {

    // rejected: put the removed atom back or delete the inserted atom

    if (sender) {
      if (iremove >= 0) atom->avec->unpack_exchange(exchange_buf.data());
      atom->natoms++;
    } else {
      tagint *tag = atom->tag;
      for (int i = 0; i < atom->nlocal; i++) {
        if (tag[i] == newtag) {
          atom->avec->copy(atom->nlocal - 1, i, 1);
          atom->nlocal--;
          break;
        }
      }
      atom->natoms--;
    }
    if (atom->map_style != Atom::MAP_NONE) atom->map_init();
    if (force->pair && force->pair->tail_flag) force->pair->reinit();
    energy_stored = energy_before;
    ghosts_stale = 1;
  }

  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
   insert an atom of type itype with group mask imask at coord on the owning
   rank (proc_flag = 1), assign a new atom ID and thermal velocity.
   must be called by all ranks of a box, returns the new atom ID.
------------------------------------------------------------------------- */

tagint FixGEMC::insert_atom(int itype, int imask, double *coord, int proc_flag)
{
  tagint mytag = 0;
  if (proc_flag) {
    atom->avec->create_atom(itype, coord);
    int m = atom->nlocal - 1;
    atom->mask[m] = imask;
    double sigma = sqrt(force->boltz * box_temp / atom->mass[itype] / force->mvv2e);
    atom->v[m][0] = random_proc->gaussian() * sigma;
    atom->v[m][1] = random_proc->gaussian() * sigma;
    atom->v[m][2] = random_proc->gaussian() * sigma;
    modify->create_attribute(m);
    atom->tag_extend();
    mytag = atom->tag[m];
  } else {
    atom->tag_extend();
  }
  tagint newtag;
  MPI_Allreduce(&mytag, &newtag, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  if (newtag == 0) error->all(FLERR, "Fix gemc failed to insert atom");
  atom->natoms++;
  return newtag;
}

/* ----------------------------------------------------------------------
   Displace a randomly chosen atom within a box
------------------------------------------------------------------------- */

void FixGEMC::attempt_atomic_translation_full()
{
  ntranslation_attempts += 1.0;

  if (natom_total == 0) return;

  int local = use_local();
  double energy_before = energy_stored;

  int i = pick_random_gas_atom();

  double xold[3] = {0.0, 0.0, 0.0};
  double coord[3] = {0.0, 0.0, 0.0};
  imageint imageold = 0;
  tagint tagold = 0;
  double denergy = 0.0;

  if (i >= 0) {
    double **x = atom->x;
    double rsq = 1.1;
    double rx, ry, rz;
    rx = ry = rz = 0.0;
    while (rsq > 1.0) {
      rx = 2.0 * random_proc->uniform() - 1.0;
      ry = 2.0 * random_proc->uniform() - 1.0;
      rz = 2.0 * random_proc->uniform() - 1.0;
      rsq = rx * rx + ry * ry + rz * rz;
    }
    xold[0] = x[i][0];
    xold[1] = x[i][1];
    xold[2] = x[i][2];
    imageold = atom->image[i];
    tagold = atom->tag[i];
    coord[0] = x[i][0] + displace * rx;
    coord[1] = x[i][1] + displace * ry;
    coord[2] = x[i][2] + displace * rz;

    if (local) {
      int itype = atom->type[i];
      double energy_old;
      double energy_new = energy_local(i, itype, tagold, coord, xold, &energy_old);
      denergy = energy_new - energy_old;
    } else {
      x[i][0] = coord[0];
      x[i][1] = coord[1];
      x[i][2] = coord[2];
    }
  }

  double energy_after;
  if (local) {
    double denergy_all;
    MPI_Allreduce(&denergy, &denergy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    energy_after = energy_before + denergy_all;
  } else {
    energy_after = energy_full();
  }

  if ((energy_after < MAXENERGYTEST) &&
      (random_world->uniform() < exp(beta * (energy_before - energy_after)))) {
    energy_stored = energy_after;
    ntranslation_successes += 1.0;

    // with local energy the move has not been applied yet

    if (local) {
      if (i >= 0) {
        atom->x[i][0] = coord[0];
        atom->x[i][1] = coord[1];
        atom->x[i][2] = coord[2];
      }
      refresh_ghosts();
    }

  } else if (!local) {

    // rejected: restore position and image flags of the atom,
    // which may have moved to a different rank or across a periodic boundary

    tagint tagold_all;
    MPI_Allreduce(&tagold, &tagold_all, 1, MPI_LMP_TAGINT, MPI_MAX, world);
    double xold_all[3];
    MPI_Allreduce(xold, xold_all, 3, MPI_DOUBLE, MPI_SUM, world);
    imageint imageold_all;
    MPI_Allreduce(&imageold, &imageold_all, 1, MPI_LMP_IMAGEINT, MPI_SUM, world);

    double **x = atom->x;
    tagint *tag = atom->tag;
    for (int j = 0; j < atom->nlocal; j++) {
      if (tag[j] == tagold_all) {
        x[j][0] = xold_all[0];
        x[j][1] = xold_all[1];
        x[j][2] = xold_all[2];
        atom->image[j] = imageold_all;
        break;
      }
    }
    energy_stored = energy_before;
    ghosts_stale = 1;
  }
  update_gas_atoms_list();
}
