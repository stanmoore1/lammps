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
#include "math_const.h"
#include "math_extra.h"
#include "modify.h"
#include "molecule.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"

#include <algorithm>
#include <cmath>

using namespace LAMMPS_NS;
using MathConst::MY_2PI;

// trial energies above this value (or not a number) are always rejected

static constexpr double MAXENERGYTEST = 1.0e50;

// values per atom in gather_molecule() and in the exchanged molecule data

static constexpr int NMOLDATA = 10;
static constexpr int NMOLINFO = 8;

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

  double s = cbrt(fvolume);

  // scale the box uniformly about its lower corner (keeping its shape)
  // atoms are scaled individually, molecules with their center of mass

  // moves that make a box narrower than twice the size of a molecule are
  // rejected in both boxes

  bigint nunits = scale_positions(s, 1);
  if (nunits < 0) return;
  set_box(xlo + (xhi - xlo) * s, ylo + (yhi - ylo) * s, zlo + (zhi - zlo) * s, boxxy * s, boxxz * s,
          boxyz * s);

  // Frenkel & Smit, 3rd Ed. (2023), Eq. (6.6.10) for a random walk in log(V1/V2):
  // acc = (V1new/V1old)^(N1+1) * (V2new/V2old)^(N2+1) * exp(-beta*(dU1+dU2))
  // N = number of independent units (molecules or atoms) in a box
  // each box contributes dU - (N+1)*kT*log(Vnew/Vold)

  double energy_after = energy_full();
  double dU = energy_after - energy_stored - (nunits + 1) * force->boltz * box_temp * log(fvolume);
  int overflow = !(energy_after < MAXENERGYTEST);

  if (accept_both(dU, overflow)) {
    nvolume_successes += 1.0;
    logvolratio += dlogvolratio;
    energy_stored = energy_after;
    xhi = domain->boxhi[0];
    yhi = domain->boxhi[1];
    zhi = domain->boxhi[2];
    boxxy = domain->xy;
    boxxz = domain->xz;
    boxyz = domain->yz;

  } else {

    // rejected: scale positions back while the trial box is still set, restore box

    scale_positions(1.0 / s, 0);
    set_box(xhi, yhi, zhi, boxxy, boxxz, boxyz);
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
  double volume = box_volume();
  int nold = natom_total;
  double denergy = 0.0;

  // donor box: pick an atom
  // with full energy: remove it and keep its complete state for a possible restore
  // with local energy: only compute its energy, remove it after acceptance
  // donor_info = {box is empty, atom type, atom mask, atom charge, velocity}

  double donor_info[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int iremove = -1;
  if (sender) {
    if (natom_total == 0) {
      donor_info[0] = 1.0;
    } else {
      iremove = pick_random_gas_atom();
      double info[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      double eatom = 0.0;
      if (iremove >= 0) {
        info[0] = atom->type[iremove];
        info[1] = atom->mask[iremove];
        info[2] = atom->q_flag ? atom->q[iremove] : 0.0;
        info[3] = atom->v[iremove][0];
        info[4] = atom->v[iremove][1];
        info[5] = atom->v[iremove][2];
        if (local) {
          eatom = energy_local(iremove, atom->type[iremove], atom->tag[iremove], atom->x[iremove]);
        } else {
          size_t nbuf = atom->avec->maxexchange + 1024;
          for (const auto &ifix : modify->get_fix_list()) nbuf += ifix->maxexchange;
          if (exchange_buf.size() < nbuf) exchange_buf.resize(nbuf);
          atom->avec->pack_exchange(iremove, exchange_buf.data());
          atom->avec->copy(atom->nlocal - 1, iremove, 1);
          atom->nlocal--;
        }
      }
      // exactly one rank owns the atom, all others contribute zeros
      MPI_Allreduce(info, &donor_info[1], 6, MPI_DOUBLE, MPI_SUM, world);
      if (local) {
        MPI_Allreduce(&eatom, &denergy, 1, MPI_DOUBLE, MPI_SUM, world);
        denergy = -denergy;
      } else {
        atom->natoms--;
        changed_atoms();
      }
    }
  }

  // receiver box needs the donor info

  if (me == 0) {
    double other[7];
    MPI_Sendrecv(donor_info, 7, MPI_DOUBLE, 1 - myworld, 0, other, 7, MPI_DOUBLE, 1 - myworld, 0,
                 comm_replica, MPI_STATUS_IGNORE);
    if (!sender)
      for (int k = 0; k < 7; k++) donor_info[k] = other[k];
  }
  MPI_Bcast(donor_info, 7, MPI_DOUBLE, 0, world);

  // donor box is empty: reject without doing anything

  if (donor_info[0] != 0.0) return;

  // receiver box: insert an atom of the same type, group membership, and charge
  // at a random position
  // with local energy: compute its energy in a scratch slot and insert it after acceptance

  int itype = static_cast<int>(donor_info[1]);
  int imask = static_cast<int>(donor_info[2]);
  double iq = donor_info[3];
  double *iv = &donor_info[4];
  double coord[3];
  int proc_flag = 0;
  tagint newtag = 0;
  if (!sender) {
    random_point(coord);
    proc_flag = owns(coord);

    if (local) {
      double eatom = 0.0;
      if (proc_flag) {
        int ii = atom->nlocal + atom->nghost;
        if (ii >= atom->nmax) atom->avec->grow(0);
        atom->type[ii] = itype;
        atom->mask[ii] = imask;
        atom->tag[ii] = 0;
        if (atom->q_flag) atom->q[ii] = iq;
        atom->x[ii][0] = coord[0];
        atom->x[ii][1] = coord[1];
        atom->x[ii][2] = coord[2];
        eatom = energy_local(ii, itype, 0, coord);
      }
      MPI_Allreduce(&eatom, &denergy, 1, MPI_DOUBLE, MPI_SUM, world);
    } else {
      newtag = insert_atom(itype, imask, iq, iv, coord, proc_flag);
      changed_atoms();
    }
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
        insert_atom(itype, imask, iq, iv, coord, proc_flag);
      }
      changed_atoms();
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
    changed_atoms();
    energy_stored = energy_before;
    ghosts_stale = 1;
  }

  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
   insert an atom of type itype with group mask imask, charge iq, and velocity iv
   at coord on the owning rank (proc_flag = 1) and assign a new atom ID.
   the velocity of the removed atom is kept, so that time integration
   continues consistently in hybrid MD/MC simulations.
   must be called by all ranks of a box, returns the new atom ID.
------------------------------------------------------------------------- */

tagint FixGEMC::insert_atom(int itype, int imask, double iq, double *iv, double *coord,
                            int proc_flag)
{
  tagint mytag = 0;
  if (proc_flag) {
    atom->avec->create_atom(itype, coord);
    int m = atom->nlocal - 1;
    atom->mask[m] = imask;
    if (atom->q_flag) atom->q[m] = iq;
    atom->v[m][0] = iv[0];
    atom->v[m][1] = iv[1];
    atom->v[m][2] = iv[2];
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

/* ----------------------------------------------------------------------
   return molecule ID of a randomly chosen molecule in my box
   all molecules have the same number of atoms, so picking a random group
   atom picks each molecule with equal probability
------------------------------------------------------------------------- */

tagint FixGEMC::pick_random_molecule()
{
  int i = pick_random_gas_atom();
  tagint molid = (i >= 0) ? atom->molecule[i] : 0;
  tagint molid_all;
  MPI_Allreduce(&molid, &molid_all, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  return molid_all;
}

/* ----------------------------------------------------------------------
   gather data of all atoms of molecule molid from all ranks, sorted by atom ID
   NMOLDATA values per atom: atom ID, unwrapped x,y,z, charge, mask, mass, velocity
------------------------------------------------------------------------- */

void FixGEMC::gather_molecule(tagint molid, std::vector<double> &data)
{
  std::vector<double> mine;
  double xu[3];
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->molecule[i] != molid) continue;
    domain->unmap(atom->x[i], atom->image[i], xu);
    mine.push_back(ubuf(atom->tag[i]).d);
    mine.push_back(xu[0]);
    mine.push_back(xu[1]);
    mine.push_back(xu[2]);
    mine.push_back(atom->q_flag ? atom->q[i] : 0.0);
    mine.push_back(atom->mask[i]);
    mine.push_back(atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]]);
    mine.push_back(atom->v[i][0]);
    mine.push_back(atom->v[i][1]);
    mine.push_back(atom->v[i][2]);
  }

  int nprocs = comm->nprocs;
  std::vector<int> counts(nprocs), displs(nprocs);
  int nsend = mine.size();
  MPI_Allgather(&nsend, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  int ntotal = 0;
  for (int iproc = 0; iproc < nprocs; iproc++) {
    displs[iproc] = ntotal;
    ntotal += counts[iproc];
  }
  std::vector<double> all(ntotal + 1);
  MPI_Allgatherv(mine.data(), nsend, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                 MPI_DOUBLE, world);

  int n = ntotal / NMOLDATA;
  std::vector<int> order(n);
  for (int k = 0; k < n; k++) order[k] = k;
  std::sort(order.begin(), order.end(), [&all](int a, int b) {
    return (tagint) ubuf(all[NMOLDATA * a]).i < (tagint) ubuf(all[NMOLDATA * b]).i;
  });
  data.resize(ntotal);
  for (int k = 0; k < n; k++)
    for (int j = 0; j < NMOLDATA; j++) data[NMOLDATA * k + j] = all[NMOLDATA * order[k] + j];
}

/* ----------------------------------------------------------------------
   Displace a randomly chosen molecule within a box
------------------------------------------------------------------------- */

void FixGEMC::attempt_molecule_translation_full()
{
  ntranslation_attempts += 1.0;

  if (natom_total == 0) return;

  double energy_before = energy_stored;
  tagint molid = pick_random_molecule();

  double d[3];
  double rsq = 1.1;
  while (rsq > 1.0) {
    d[0] = 2.0 * random_world->uniform() - 1.0;
    d[1] = 2.0 * random_world->uniform() - 1.0;
    d[2] = 2.0 * random_world->uniform() - 1.0;
    rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  }
  d[0] *= displace;
  d[1] *= displace;
  d[2] *= displace;

  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->molecule[i] == molid) {
      atom->x[i][0] += d[0];
      atom->x[i][1] += d[1];
      atom->x[i][2] += d[2];
    }
  }

  double energy_after = energy_full();

  if ((energy_after < MAXENERGYTEST) &&
      (random_world->uniform() < exp(beta * (energy_before - energy_after)))) {
    energy_stored = energy_after;
    ntranslation_successes += 1.0;
  } else {

    // rejected: shifting back is consistent with the image flags,
    // even if energy_full() wrapped atoms around periodic boundaries

    double **x = atom->x;
    for (int i = 0; i < atom->nlocal; i++) {
      if (atom->molecule[i] == molid) {
        x[i][0] -= d[0];
        x[i][1] -= d[1];
        x[i][2] -= d[2];
      }
    }
    energy_stored = energy_before;
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
   Rotate a randomly chosen molecule about its center of mass
------------------------------------------------------------------------- */

void FixGEMC::attempt_molecule_rotation_full()
{
  nrotation_attempts += 1.0;

  if (natom_total == 0) return;

  double energy_before = energy_stored;
  tagint molid = pick_random_molecule();

  // center of mass from unwrapped coordinates

  std::vector<double> data;
  gather_molecule(molid, data);
  int n = data.size() / NMOLDATA;
  double com[3] = {0.0, 0.0, 0.0};
  double mtotal = 0.0;
  for (int k = 0; k < n; k++) {
    double m = data[NMOLDATA * k + 6];
    com[0] += m * data[NMOLDATA * k + 1];
    com[1] += m * data[NMOLDATA * k + 2];
    com[2] += m * data[NMOLDATA * k + 3];
    mtotal += m;
  }
  com[0] /= mtotal;
  com[1] /= mtotal;
  com[2] /= mtotal;

  // random axis and angle in [-maxangle,maxangle], a symmetric proposal

  double r[3], quat[4], rotmat[3][3];
  double rsq = 1.1;
  while ((rsq > 1.0) || (rsq < 1.0e-6)) {
    r[0] = 2.0 * random_world->uniform() - 1.0;
    r[1] = 2.0 * random_world->uniform() - 1.0;
    r[2] = 2.0 * random_world->uniform() - 1.0;
    rsq = MathExtra::dot3(r, r);
  }
  MathExtra::norm3(r);
  double theta = maxangle * (2.0 * random_world->uniform() - 1.0);
  MathExtra::axisangle_to_quat(r, theta, quat);
  MathExtra::quat_to_mat(quat, rotmat);

  // save old positions and image flags, apply rotation to unwrapped coordinates

  const imageint imagezero =
      ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  std::vector<double> saved;
  double **x = atom->x;
  imageint *image = atom->image;
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->molecule[i] != molid) continue;
    saved.push_back(ubuf(atom->tag[i]).d);
    saved.push_back(x[i][0]);
    saved.push_back(x[i][1]);
    saved.push_back(x[i][2]);
    saved.push_back(ubuf(image[i]).d);
    double xu[3], dx[3];
    domain->unmap(x[i], image[i], xu);
    dx[0] = xu[0] - com[0];
    dx[1] = xu[1] - com[1];
    dx[2] = xu[2] - com[2];
    MathExtra::matvec(rotmat, dx, x[i]);
    x[i][0] += com[0];
    x[i][1] += com[1];
    x[i][2] += com[2];
    image[i] = imagezero;
    domain->remap(x[i], image[i]);
  }

  double energy_after = energy_full();

  if ((energy_after < MAXENERGYTEST) &&
      (random_world->uniform() < exp(beta * (energy_before - energy_after)))) {
    energy_stored = energy_after;
    nrotation_successes += 1.0;
  } else {

    // rejected: energy_full() may have moved atoms of the molecule to other
    // ranks, so restore saved positions and image flags by atom ID

    int nprocs = comm->nprocs;
    std::vector<int> counts(nprocs), displs(nprocs);
    int nsend = saved.size();
    MPI_Allgather(&nsend, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
    int ntotal = 0;
    for (int iproc = 0; iproc < nprocs; iproc++) {
      displs[iproc] = ntotal;
      ntotal += counts[iproc];
    }
    std::vector<double> all(ntotal + 1);
    MPI_Allgatherv(saved.data(), nsend, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, world);
    x = atom->x;
    image = atom->image;
    for (int k = 0; k < ntotal; k += 5) {
      int i = atom->map((tagint) ubuf(all[k]).i);
      if ((i >= 0) && (i < atom->nlocal)) {
        x[i][0] = all[k + 1];
        x[i][1] = all[k + 2];
        x[i][2] = all[k + 3];
        image[i] = (imageint) ubuf(all[k + 4]).i;
      }
    }
    energy_stored = energy_before;
  }
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
  Move a randomly chosen molecule from one box to the other box.
  the molecule keeps its conformation and is inserted at a random position
  with a random orientation, so the move is symmetric for flexible molecules
------------------------------------------------------------------------- */

void FixGEMC::attempt_molecule_exchange_full()
{
  nexchange_attempts += 1.0;

  const int n = natoms_per_molecule;

  // choose donor box with equal probability, identical in both boxes

  int donor = (random_universe->uniform() < 0.5) ? 0 : 1;
  int sender = (myworld == donor) ? 1 : 0;

  double energy_before = energy_stored;
  double volume = box_volume();
  int nold = natom_total / n;

  // donor box: pick a molecule, record its conformation relative to its
  // center of mass, then remove its atoms and keep their state for a restore
  // molinfo = {box is empty, n x (dx, dy, dz, charge, mask, vx, vy, vz)}

  std::vector<double> molinfo(1 + NMOLINFO * n, 0.0);
  int nremoved = 0;
  if (sender) {
    if (nold == 0) {
      molinfo[0] = 1.0;
    } else {
      tagint molid = pick_random_molecule();
      std::vector<double> data;
      gather_molecule(molid, data);
      if ((int) data.size() != NMOLDATA * n)
        error->all(FLERR, "Fix gemc found molecule {} with {} atoms instead of {}", molid,
                   data.size() / NMOLDATA, n);
      double com[3] = {0.0, 0.0, 0.0};
      double mtotal = 0.0;
      for (int k = 0; k < n; k++) {
        double m = data[NMOLDATA * k + 6];
        com[0] += m * data[NMOLDATA * k + 1];
        com[1] += m * data[NMOLDATA * k + 2];
        com[2] += m * data[NMOLDATA * k + 3];
        mtotal += m;
      }
      for (int j = 0; j < 3; j++) com[j] /= mtotal;
      for (int k = 0; k < n; k++) {
        double *info = &molinfo[1 + NMOLINFO * k];
        info[0] = data[NMOLDATA * k + 1] - com[0];
        info[1] = data[NMOLDATA * k + 2] - com[1];
        info[2] = data[NMOLDATA * k + 3] - com[2];
        info[3] = data[NMOLDATA * k + 4];
        info[4] = data[NMOLDATA * k + 5];
        info[5] = data[NMOLDATA * k + 7];
        info[6] = data[NMOLDATA * k + 8];
        info[7] = data[NMOLDATA * k + 9];
      }

      // remove atoms of the molecule, save their complete state

      size_t nbuf = atom->avec->maxexchange + 1024 + 2 * atom->bond_per_atom +
          4 * atom->angle_per_atom + 5 * atom->dihedral_per_atom + 5 * atom->improper_per_atom +
          atom->maxspecial;
      for (const auto &ifix : modify->get_fix_list()) nbuf += ifix->maxexchange;
      size_t used = 0;
      int i = 0;
      while (i < atom->nlocal) {
        if (atom->molecule[i] == molid) {
          if (exchange_buf.size() < used + nbuf) exchange_buf.resize(used + nbuf);
          used += atom->avec->pack_exchange(i, &exchange_buf[used]);
          atom->avec->copy(atom->nlocal - 1, i, 1);
          atom->nlocal--;
          nremoved++;
        } else {
          i++;
        }
      }
      atom->natoms -= n;
      atom->nbonds -= onemol->nbonds;
      atom->nangles -= onemol->nangles;
      atom->ndihedrals -= onemol->ndihedrals;
      atom->nimpropers -= onemol->nimpropers;
      changed_atoms();
    }
  }

  // receiver box needs the donor info

  if (me == 0) {
    std::vector<double> other(1 + NMOLINFO * n);
    MPI_Sendrecv(molinfo.data(), 1 + NMOLINFO * n, MPI_DOUBLE, 1 - myworld, 0, other.data(),
                 1 + NMOLINFO * n, MPI_DOUBLE, 1 - myworld, 0, comm_replica, MPI_STATUS_IGNORE);
    if (!sender) molinfo = other;
  }
  MPI_Bcast(molinfo.data(), 1 + NMOLINFO * n, MPI_DOUBLE, 0, world);

  // donor box is empty: reject without doing anything

  if (molinfo[0] != 0.0) return;

  // receiver box: insert the molecule at a random position with a uniformly
  // distributed random orientation (random unit quaternion).
  // velocities are rotated with the molecule, so that time integration
  // continues consistently in hybrid MD/MC simulations

  tagint newmol = 0;
  if (!sender) {
    tagint maxtag = 0, maxmol = 0;
    for (int i = 0; i < atom->nlocal; i++) {
      maxtag = MAX(maxtag, atom->tag[i]);
      maxmol = MAX(maxmol, atom->molecule[i]);
    }
    tagint maxtag_all, maxmol_all;
    MPI_Allreduce(&maxtag, &maxtag_all, 1, MPI_LMP_TAGINT, MPI_MAX, world);
    MPI_Allreduce(&maxmol, &maxmol_all, 1, MPI_LMP_TAGINT, MPI_MAX, world);
    if ((maxtag_all + n >= MAXTAGINT) || (maxmol_all + 1 >= MAXTAGINT))
      error->all(FLERR, "Fix gemc ran out of atom or molecule IDs");
    newmol = maxmol_all + 1;

    double com[3];
    random_point(com);
    double u1 = random_world->uniform();
    double u2 = random_world->uniform();
    double u3 = random_world->uniform();
    double quat[4] = {sqrt(1.0 - u1) * sin(MY_2PI * u2), sqrt(1.0 - u1) * cos(MY_2PI * u2),
                      sqrt(u1) * sin(MY_2PI * u3), sqrt(u1) * cos(MY_2PI * u3)};
    double rotmat[3][3];
    MathExtra::quat_to_mat(quat, rotmat);

    const imageint imagezero =
        ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;
    int ncreated = 0;
    for (int k = 0; k < n; k++) {
      double xnew[3];
      double *info = &molinfo[1 + NMOLINFO * k];
      MathExtra::matvec(rotmat, info, xnew);
      xnew[0] += com[0];
      xnew[1] += com[1];
      xnew[2] += com[2];
      imageint imagenew = imagezero;
      domain->remap(xnew, imagenew);
      if (!owns(xnew)) continue;
      ncreated++;

      int itype = onemol->type[k];
      atom->avec->create_atom(itype, xnew);
      int m = atom->nlocal - 1;
      atom->mask[m] = static_cast<int>(info[4]);
      atom->image[m] = imagenew;
      atom->molecule[m] = newmol;
      atom->tag[m] = maxtag_all + k + 1;
      MathExtra::matvec(rotmat, &info[5], atom->v[m]);
      atom->add_molecule_atom(onemol, k, m, maxtag_all);
      if (atom->q_flag) atom->q[m] = info[3];
      modify->create_attribute(m);
    }
    int ncreated_all = 0;
    MPI_Allreduce(&ncreated, &ncreated_all, 1, MPI_INT, MPI_SUM, world);
    if (ncreated_all != n)
      error->all(FLERR, "Fix gemc inserted {} instead of {} atoms of a molecule", ncreated_all, n);
    atom->natoms += n;
    atom->nbonds += onemol->nbonds;
    atom->nangles += onemol->nangles;
    atom->ndihedrals += onemol->ndihedrals;
    atom->nimpropers += onemol->nimpropers;
    changed_atoms();
  }

  // same acceptance rule as for atoms, with N = number of molecules

  double energy_after = energy_full();
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

  } else {

    // rejected: put the removed atoms back or delete the inserted molecule

    if (sender) {
      size_t pos = 0;
      for (int k = 0; k < nremoved; k++) pos += atom->avec->unpack_exchange(&exchange_buf[pos]);
      atom->natoms += n;
      atom->nbonds += onemol->nbonds;
      atom->nangles += onemol->nangles;
      atom->ndihedrals += onemol->ndihedrals;
      atom->nimpropers += onemol->nimpropers;
    } else {
      int i = 0;
      while (i < atom->nlocal) {
        if (atom->molecule[i] == newmol) {
          atom->avec->copy(atom->nlocal - 1, i, 1);
          atom->nlocal--;
        } else {
          i++;
        }
      }
      atom->natoms -= n;
      atom->nbonds -= onemol->nbonds;
      atom->nangles -= onemol->nangles;
      atom->ndihedrals -= onemol->ndihedrals;
      atom->nimpropers -= onemol->nimpropers;
    }
    changed_atoms();
    energy_stored = energy_before;
  }

  update_gas_atoms_list();
}
