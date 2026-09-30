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

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "universe.h"
#include "update.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_PI;

static constexpr double RESTART_VERSION = 3.0;
static constexpr double MAXDLOGVOLRATIO = 1.0;
static constexpr double DEG2RAD = MY_PI / 180.0;
static constexpr double BIG = 1.0e20;

/* ---------------------------------------------------------------------- */

FixGEMC::FixGEMC(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), idmol(nullptr), onemol(nullptr), c_pe(nullptr), local_gas_list(nullptr),
    comm_replica(MPI_COMM_NULL), random_universe(nullptr), random_world(nullptr),
    random_proc(nullptr)
{
  if (narg < 11) utils::missing_cmd_args(FLERR, "fix gemc", error);

  // must have exactly two boxes

  if (universe->nworlds != 2)
    error->universe_all(FLERR, "Must use exactly two partitions with fix gemc");

  // various fix flags

  time_integrate = 0;
  global_freq = 1;
  time_depend = 1;
  restart_global = 1;
  vector_flag = 1;
  size_vector = 11;
  extvector = 0;

  // box size changes with volume MC moves

  box_change |= BOX_CHANGE_SIZE;
  if (domain->triclinic) box_change |= BOX_CHANGE_SHAPE;

  // required user args

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  ntranslate = utils::inumeric(FLERR, arg[4], false, lmp);
  nexchange = utils::inumeric(FLERR, arg[5], false, lmp);
  nvolume = utils::inumeric(FLERR, arg[6], false, lmp);
  box_temp = utils::numeric(FLERR, arg[7], false, lmp);
  displace = utils::numeric(FLERR, arg[8], false, lmp);
  max_dlogvolratio = utils::numeric(FLERR, arg[9], false, lmp);
  seed = utils::inumeric(FLERR, arg[10], false, lmp);

  if (nevery <= 0) error->all(FLERR, 3, "Illegal fix gemc N value {}: must be > 0", nevery);
  if (ntranslate < 0) error->all(FLERR, 4, "Illegal fix gemc M value {}: must be >= 0", ntranslate);
  if (nexchange < 0) error->all(FLERR, 5, "Illegal fix gemc X value {}: must be >= 0", nexchange);
  if (nvolume < 0) error->all(FLERR, 6, "Illegal fix gemc V value {}: must be >= 0", nvolume);
  if (ntranslate + nexchange + nvolume <= 0)
    error->all(FLERR, "Fix gemc requires at least one type of Monte Carlo move");
  if (box_temp <= 0.0)
    error->all(FLERR, 7, "Illegal fix gemc temperature {}: must be > 0", box_temp);
  if (displace < 0.0)
    error->all(FLERR, 8, "Illegal fix gemc displace value {}: must be >= 0", displace);
  if (max_dlogvolratio <= 0.0)
    error->all(FLERR, 9, "Illegal fix gemc maxdlogvolratio value {}: must be > 0",
               max_dlogvolratio);
  if ((seed <= 0) || (seed > MAXSMALLINT - 3 - universe->nprocs))
    error->all(FLERR, 10, "Illegal fix gemc random number seed {}", seed);

  // optional keywords

  full_flag = 0;
  molflag = 0;
  maxangle = 30.0 * DEG2RAD;
  tune_every = 0;
  tune_trans = tune_vol = 0.4;
  int iarg = 11;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "full_energy") == 0) {
      full_flag = 1;
      iarg++;
    } else if (strcmp(arg[iarg], "mol") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix gemc mol", error);
      if (atom->find_molecule(arg[iarg + 1]) == -1)
        error->all(FLERR, iarg + 1, "Molecule template ID {} for fix gemc does not exist",
                   arg[iarg + 1]);
      delete[] idmol;
      idmol = utils::strdup(arg[iarg + 1]);
      molflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "maxangle") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix gemc maxangle", error);
      maxangle = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if ((maxangle <= 0.0) || (maxangle > 180.0))
        error->all(FLERR, iarg + 1, "Illegal fix gemc maxangle value {}: must be > 0 and <= 180",
                   maxangle);
      maxangle *= DEG2RAD;
      iarg += 2;
    } else if (strcmp(arg[iarg], "tune") == 0) {
      if (iarg + 4 > narg) utils::missing_cmd_args(FLERR, "fix gemc tune", error);
      tune_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      tune_trans = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      tune_vol = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      if (tune_every <= 0)
        error->all(FLERR, iarg + 1, "Illegal fix gemc tune N value {}: must be > 0", tune_every);
      if ((tune_trans <= 0.0) || (tune_trans >= 1.0))
        error->all(FLERR, iarg + 2, "Illegal fix gemc tune translation acceptance ratio {}",
                   tune_trans);
      if ((tune_vol <= 0.0) || (tune_vol >= 1.0))
        error->all(FLERR, iarg + 3, "Illegal fix gemc tune volume acceptance ratio {}", tune_vol);
      iarg += 4;
    } else {
      error->all(FLERR, iarg, "Unknown fix gemc keyword: {}", arg[iarg]);
    }
  }
  local_flag = 0;
  ghosts_stale = 0;
  charge_warned = 0;
  group_charged = 0;
  natoms_per_molecule = 1;

  if (molflag) {
    if (!atom->molecule_flag)
      error->all(FLERR, "Fix gemc keyword mol requires an atom style with molecule IDs");
    if (atom->molecular == Atom::TEMPLATE)
      error->all(FLERR, "Fix gemc does not support atom style template");
  }

  // set up comm_replica = communicator between the same ranks of both boxes
  // only rank 0 of each box uses it, rank 0 of box 1 is rank 0 of comm_replica

  MPI_Comm_rank(world, &me);
  myworld = universe->iworld;
  MPI_Comm_split(universe->uworld, me, myworld, &comm_replica);

  // random number generators: unique to each rank, synchronized across one box,
  // and synchronized across both boxes. all seeds are different.

  random_universe = new RanPark(lmp, seed);
  random_world = new RanPark(lmp, seed + 1 + myworld);
  random_proc = new RanPark(lmp, seed + 3 + universe->me);

  ntranslation_attempts = ntranslation_successes = 0.0;
  nrotation_attempts = nrotation_successes = 0.0;
  nvolume_attempts = nvolume_successes = 0.0;
  nexchange_attempts = nexchange_successes = 0.0;
  for (auto &n : nlast) n = 0.0;
  for (auto &n : tune_last) n = 0.0;
  ntune_calls = 0;

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  gemc_nmax = 0;
  natom_lower = natom_local = natom_total = 0;
  logvolratio = voltot = 0.0;
  triclinic = 0;
  xlo = ylo = zlo = xhi = yhi = zhi = boxxy = boxxz = boxyz = 0.0;
}

/* ---------------------------------------------------------------------- */

FixGEMC::~FixGEMC()
{
  delete random_proc;
  delete random_world;
  delete random_universe;
  memory->destroy(local_gas_list);
  delete[] idmol;
  if (comm_replica != MPI_COMM_NULL) MPI_Comm_free(&comm_replica);
}

/* ---------------------------------------------------------------------- */

int FixGEMC::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGEMC::init()
{
  if (domain->dimension != 3) error->all(FLERR, "Fix gemc requires a 3d system");
  if (domain->nonperiodic) error->all(FLERR, "Fix gemc requires a fully periodic box");
  if (!atom->tag_enable) error->all(FLERR, "Fix gemc requires atom IDs");
  if (!atom->mass) error->all(FLERR, "Fix gemc requires per-type masses");
  if (atom->rmass_flag)
    error->all(FLERR, "Fix gemc does not support atom styles with per-atom masses");

  // exchanges transfer only type, charge, group membership, velocity, and
  // topology, so other per-atom properties would be lost

  if (atom->mu_flag || atom->quat_flag || atom->omega_flag || atom->angmom_flag ||
      atom->ellipsoid_flag || atom->line_flag || atom->tri_flag || atom->body_flag ||
      atom->superellipsoid_flag || atom->sp_flag || atom->temperature_flag || atom->dpd_flag ||
      atom->edpd_flag || atom->tdpd_flag || atom->rho_flag)
    error->all(FLERR, "Fix gemc does not support atom style {}", atom->atom_style);
  if (atom->nivector || atom->ndvector || atom->niarray || atom->ndarray)
    error->all(FLERR, "Fix gemc does not support custom per-atom properties");
  if (force->pair && force->pair->tail_flag && !force->pair->reinitflag)
    error->all(FLERR, "Fix gemc with pair_modify tail yes is not supported by pair style {}",
               force->pair_style);
  for (const auto &ifix : modify->get_fix_list())
    if (ifix->rigid_flag || utils::strmatch(ifix->style, "^shake") ||
        utils::strmatch(ifix->style, "^rattle"))
      error->all(FLERR, "Fix gemc does not support constraints like fix {}", ifix->style);

  triclinic = domain->triclinic;

  // molecule template

  if (molflag) {
    int imol = atom->find_molecule(idmol);
    if (imol == -1) error->all(FLERR, "Molecule template ID {} for fix gemc does not exist", idmol);
    onemol = atom->molecules[imol];
    if ((onemol->nset > 1) && (comm->me == 0))
      error->warning(FLERR,
                     "Molecule template {} for fix gemc has multiple molecules; "
                     "using only the first",
                     idmol);
    if (!onemol->xflag || !onemol->typeflag)
      error->all(FLERR, "Molecule template {} for fix gemc must define coordinates and types",
                 idmol);
    if (onemol->ntypes > atom->ntypes)
      error->all(FLERR, "Molecule template {} for fix gemc has invalid atom types", idmol);
    if (atom->molecular == Atom::MOLECULAR) {
      if ((onemol->bondflag && !atom->avec->bonds_allow) ||
          (onemol->angleflag && !atom->avec->angles_allow) ||
          (onemol->dihedralflag && !atom->avec->dihedrals_allow) ||
          (onemol->improperflag && !atom->avec->impropers_allow))
        error->all(FLERR,
                   "Molecule template {} for fix gemc has topology not allowed by the "
                   "atom style",
                   idmol);
      if (onemol->specialflag && (onemol->maxspecial > atom->maxspecial))
        error->all(FLERR,
                   "Molecule template {} for fix gemc has too many special neighbors; "
                   "use extra/special/per/atom",
                   idmol);
    }
    onemol->check_attributes();
    natoms_per_molecule = onemol->natoms;
  } else {
    natoms_per_molecule = 1;
  }
  check_molecules();

  // decide whether single-atom energies can replace full energy evaluations
  // for translation and exchange moves. requires a pair style that
  // provides single() and has no many-body terms, no tail corrections,
  // no neighbor list exclusions, and no fixes contributing to the potential energy.
  // displacements must stay within the neighbor skin so that ghost atoms
  // cover all interactions of a displaced atom.

  local_flag = 0;
  if (!full_flag) {
    local_flag = 1;
    std::string reason;
    if (!force->pair) {
      reason = "no pair style";
    } else if (molflag) {
      reason = "molecules are exchanged";
    } else if (force->kspace) {
      reason = "a long-range solver is used";
    } else if (group_charged) {
      reason = "exchanged atoms are charged";
    } else if (!force->pair->single_enable) {
      reason = fmt::format("pair style {} does not support single()", force->pair_style);
    } else if (force->pair->manybody_flag) {
      reason = fmt::format("pair style {} is a many-body potential", force->pair_style);
    } else if (force->pair->tail_flag) {
      reason = "pair_modify tail yes is used";
    } else if (neighbor->exclude) {
      reason = "neighbor list exclusions are used";
    } else if (displace > neighbor->skin) {
      reason = "the displace value is larger than the neighbor skin";
    } else {
      for (const auto &ifix : modify->get_fix_list())
        if (ifix->energy_global_flag && ifix->thermo_energy)
          reason = fmt::format("fix {} contributes to the potential energy", ifix->id);
    }
    if (!reason.empty()) {
      local_flag = 0;
      if (comm->me == 0)
        error->warning(FLERR, "Fix gemc uses full energy evaluations for all moves because {}",
                       reason);
    }
  }

  // total energy is taken from the thermo_pe compute

  c_pe = modify->get_compute_by_id("thermo_pe");
  if (!c_pe) error->all(FLERR, "Fix gemc could not find thermo_pe compute");

  // move type selection probabilities

  nmoves = nvolume + nexchange + ntranslate;
  pc_exchange = static_cast<double>(nexchange) / nmoves;
  pc_volume = static_cast<double>(nexchange + nvolume) / nmoves;

  beta = 1.0 / (force->boltz * box_temp);

  // update box dimensions and list of atoms in the fix group

  xlo = domain->boxlo[0];
  xhi = domain->boxhi[0];
  ylo = domain->boxlo[1];
  yhi = domain->boxhi[1];
  zlo = domain->boxlo[2];
  zhi = domain->boxhi[2];
  boxxy = domain->xy;
  boxxz = domain->xz;
  boxyz = domain->yz;

  update_gas_atoms_list();
  progress = 0;
}

/* ----------------------------------------------------------------------
   check that atoms in the fix group can be moved and exchanged:
   without mol keyword, the group atoms must not belong to molecules;
   with mol keyword, each group molecule must match the template:
   consecutive atom IDs in the order of the template atoms, same types.
   also warn once if exchanged atoms or molecules carry a net charge.
------------------------------------------------------------------------- */

void FixGEMC::check_molecules()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  double *q = atom->q;

  int flag = 0;
  int charged = 0;
  group_charged = 0;

  if (!molflag) {
    int anyq = 0;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (molecule && (molecule[i] > 0)) flag = 1;
      if (q && (q[i] != 0.0)) charged = anyq = 1;
    }
    MPI_Allreduce(&anyq, &group_charged, 1, MPI_INT, MPI_MAX, world);
    int flag_all;
    MPI_Allreduce(&flag, &flag_all, 1, MPI_INT, MPI_MAX, world);
    if (flag_all)
      error->all(FLERR, "Fix gemc group contains atoms with molecule IDs; use the mol keyword");
  } else {
    // gather (molecule ID, atom ID, type, charge) of all group atoms

    std::vector<double> mine;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (molecule[i] <= 0) flag = 1;
      mine.push_back(ubuf(molecule[i]).d);
      mine.push_back(ubuf(atom->tag[i]).d);
      mine.push_back(atom->type[i]);
      mine.push_back(q ? q[i] : 0.0);
    }
    int flag_all;
    MPI_Allreduce(&flag, &flag_all, 1, MPI_INT, MPI_MAX, world);
    if (flag_all)
      error->all(FLERR, "Fix gemc group atoms must all belong to molecules with the mol keyword");

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

    struct Entry {
      tagint mol, tag;
      int type;
      double q;
    };
    std::vector<Entry> entries(ntotal / 4);
    for (size_t k = 0; k < entries.size(); k++)
      entries[k] = {(tagint) ubuf(all[4 * k]).i, (tagint) ubuf(all[4 * k + 1]).i,
                    (int) all[4 * k + 2], all[4 * k + 3]};
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
      return (a.mol < b.mol) || ((a.mol == b.mol) && (a.tag < b.tag));
    });

    size_t k = 0;
    while (k < entries.size()) {
      size_t kend = k;
      while ((kend < entries.size()) && (entries[kend].mol == entries[k].mol)) kend++;
      if ((int) (kend - k) != natoms_per_molecule) flag = 1;
      double qmol = 0.0;
      for (size_t n = k; n < kend; n++) {
        int iatom = n - k;
        if (iatom < natoms_per_molecule) {
          if (entries[n].type != onemol->type[iatom]) flag = 1;
          if (entries[n].tag != entries[k].tag + iatom) flag = 1;
        }
        qmol += entries[n].q;
      }
      if (fabs(qmol) > 1.0e-6) charged = 1;
      k = kend;
    }
    if (flag)
      error->all(FLERR,
                 "Molecules in fix gemc group must match molecule template {}: same number and "
                 "types of atoms, with consecutive atom IDs in template order",
                 idmol);
    if (onemol->qflag) {
      double qmol = 0.0;
      for (int i = 0; i < onemol->natoms; i++) qmol += onemol->q[i];
      if (fabs(qmol) > 1.0e-6) charged = 1;
    }
  }

  int charged_all;
  MPI_Allreduce(&charged, &charged_all, 1, MPI_INT, MPI_MAX, world);
  if (charged_all && !charge_warned && (comm->me == 0))
    error->warning(FLERR,
                   "Fix gemc exchanges {} with a net charge, so the boxes will not be "
                   "charge neutral",
                   molflag ? "molecules" : "atoms");
  if (charged_all) charge_warned = 1;
}

/* ----------------------------------------------------------------------
   checks and initialization that require communication between the boxes.
   done in setup() and not in init(), since only setup() is guaranteed to be
   called at the same time in both partitions (at the start of a run).
------------------------------------------------------------------------- */

void FixGEMC::setup(int /*vflag*/)
{
  // both boxes must use the same settings, or they would fall out of step and hang

  // this includes the state of the synchronized random number generator
  // and the group definitions, since group bitmasks are transferred between boxes

  uint32_t hash = 2166136261u;
  for (int igroup = 0; igroup < Group::MAX_GROUP; igroup++) {
    std::string name = group->names[igroup] ? group->names[igroup] : "";
    name += '\n';
    for (const auto c : name) hash = (hash ^ (uint32_t) (unsigned char) c) * 16777619u;
  }

  // step size adjustments only use moves attempted during this run

  tune_last[0] = ntranslation_attempts;
  tune_last[1] = ntranslation_successes;
  tune_last[2] = nvolume_attempts;
  tune_last[3] = nvolume_successes;
  tune_last[4] = nrotation_attempts;
  tune_last[5] = nrotation_successes;
  ntune_calls = 0;

  // the maximum displacement and rotation may differ between the boxes

  // signature of atom style and molecule template: types, topology, charges

  double molsig = atom->molecular + 3.0 * atom->q_flag;
  if (molflag) {
    molsig += 7.0 * onemol->nbonds + 11.0 * onemol->nangles + 13.0 * onemol->ndihedrals +
        17.0 * onemol->nimpropers;
    for (int i = 0; i < onemol->natoms; i++) {
      molsig += 19.0 * (i + 1) * onemol->type[i];
      if (onemol->qflag) molsig += 23.0 * (i + 1) * onemol->q[i];
    }
  }

  constexpr int NCHECK = 18;
  int mismatch = 0;
  if (me == 0) {
    double mine[NCHECK] = {(double) nevery,
                           (double) ntranslate,
                           (double) nexchange,
                           (double) nvolume,
                           box_temp,
                           (double) tune_every,
                           tune_trans,
                           tune_vol,
                           max_dlogvolratio,
                           (double) seed,
                           (double) atom->ntypes,
                           (double) update->ntimestep,
                           (double) next_reneighbor,
                           (double) random_universe->state(),
                           (double) hash,
                           (double) molflag,
                           (double) natoms_per_molecule,
                           molsig};
    double other[NCHECK];
    MPI_Sendrecv(mine, NCHECK, MPI_DOUBLE, 1 - myworld, 0, other, NCHECK, MPI_DOUBLE, 1 - myworld,
                 0, comm_replica, MPI_STATUS_IGNORE);
    for (int i = 0; i < NCHECK; i++)
      if (mine[i] != other[i]) mismatch = 1;
  }
  MPI_Bcast(&mismatch, 1, MPI_INT, 0, world);
  if (mismatch)
    error->universe_all(FLERR,
                        "Fix gemc settings, groups, number of atom types, atom style, molecule "
                        "template, timestep, and restart status must be the same in both "
                        "partitions");

  // initialize log volume ratio and total volume

  double vol_i = box_volume();
  double vol_j = 0.0;
  if (me == 0)
    MPI_Sendrecv(&vol_i, 1, MPI_DOUBLE, 1 - myworld, 0, &vol_j, 1, MPI_DOUBLE, 1 - myworld, 0,
                 comm_replica, MPI_STATUS_IGNORE);
  MPI_Bcast(&vol_j, 1, MPI_DOUBLE, 0, world);

  voltot = vol_i + vol_j;
  if (myworld == 0)
    logvolratio = log(vol_i / vol_j);
  else
    logvolratio = log(vol_j / vol_i);
}

/* ----------------------------------------------------------------------
   attempt Monte Carlo translations, exchanges, and volume changes
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixGEMC::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  xlo = domain->boxlo[0];
  xhi = domain->boxhi[0];
  ylo = domain->boxlo[1];
  yhi = domain->boxhi[1];
  zlo = domain->boxlo[2];
  zhi = domain->boxhi[2];
  boxxy = domain->xy;
  boxxz = domain->xz;
  boxyz = domain->yz;

  next_reneighbor = update->ntimestep + nevery;

  // the move type sequence comes from the synchronized RNG,
  // so both boxes always attempt the same type of move

  energy_stored = energy_full();
  update_gas_atoms_list();

  for (int i = 0; i < nmoves; i++) {
    double imove = random_universe->uniform();
    if (imove < pc_exchange) {
      if (molflag)
        attempt_molecule_exchange_full();
      else
        attempt_atomic_exchange_full();
    } else if (imove < pc_volume) {
      attempt_volume_change_full();
    } else if (molflag) {
      // translation or rotation, chosen independently in each box
      if (random_world->uniform() < 0.5)
        attempt_molecule_translation_full();
      else
        attempt_molecule_rotation_full();
    } else {
      attempt_atomic_translation_full();
    }
  }

  if (tune_every && (++ntune_calls % tune_every == 0)) tune_steps();

  print_progress();
}

/* ----------------------------------------------------------------------
   adjust maximum displacement (separately for each box) and maximum
   change of log(V1/V2) (identical in both boxes, since volume moves are
   accepted jointly) towards the target acceptance ratios, based on the
   moves attempted since the last adjustment
------------------------------------------------------------------------- */

void FixGEMC::tune_steps()
{
  static constexpr int MINATTEMPTS = 20;

  double dtrans = ntranslation_attempts - tune_last[0];
  if (dtrans >= MINATTEMPTS) {
    double ratio = (ntranslation_successes - tune_last[1]) / dtrans;
    double factor = MIN(MAX(ratio / tune_trans, 0.5), 1.5);
    double maxdisp = 0.5 * min_box_width();
    if (local_flag) maxdisp = MIN(maxdisp, neighbor->skin);
    displace = MIN(displace * factor, maxdisp);
    displace = max_translation();
    displace = MAX(displace, 1.0e-6 * maxdisp);
    tune_last[0] = ntranslation_attempts;
    tune_last[1] = ntranslation_successes;
  }

  double dvol = nvolume_attempts - tune_last[2];
  if (dvol >= MINATTEMPTS) {
    double ratio = (nvolume_successes - tune_last[3]) / dvol;
    double factor = MIN(MAX(ratio / tune_vol, 0.5), 1.5);
    max_dlogvolratio = MIN(max_dlogvolratio * factor, MAXDLOGVOLRATIO);
    max_dlogvolratio = MAX(max_dlogvolratio, 1.0e-6);
    tune_last[2] = nvolume_attempts;
    tune_last[3] = nvolume_successes;
  }

  double drot = nrotation_attempts - tune_last[4];
  if (drot >= MINATTEMPTS) {
    double ratio = (nrotation_successes - tune_last[5]) / drot;
    double factor = MIN(MAX(ratio / tune_trans, 0.5), 1.5);
    maxangle = MIN(maxangle * factor, MY_PI);
    maxangle = MAX(maxangle, 1.0e-6);
    tune_last[4] = nrotation_attempts;
    tune_last[5] = nrotation_successes;
  }
}

/* ----------------------------------------------------------------------
   print progress info to universe screen/logfile, about every 1% of a run
------------------------------------------------------------------------- */

void FixGEMC::print_progress()
{
  // world 0 needs the atom count of world 1 for the message

  int n_other = 0;
  if (me == 0)
    MPI_Sendrecv(&natom_total, 1, MPI_INT, 1 - myworld, 0, &n_other, 1, MPI_INT, 1 - myworld, 0,
                 comm_replica, MPI_STATUS_IGNORE);

  if (universe->me != 0) return;

  double delta = update->ntimestep - update->beginstep;
  if ((delta != 0.0) && (update->beginstep != update->endstep))
    delta /= update->endstep - update->beginstep;
  int status = static_cast<int>(delta * 100.0);
  if (status <= progress) return;
  progress = status;

  double now[6] = {ntranslation_successes, ntranslation_attempts, nvolume_successes,
                   nvolume_attempts,       nexchange_successes,   nexchange_attempts};
  double d[6];
  for (int i = 0; i < 6; i++) {
    d[i] = now[i] - nlast[i];
    nlast[i] = now[i];
  }

  double vol1 = voltot / (1.0 + exp(-logvolratio));
  double vol2 = voltot / (1.0 + exp(logvolratio));
  int n1 = natom_total / natoms_per_molecule;
  int n2 = n_other / natoms_per_molecule;

  auto msg =
      fmt::format(" GEMC run progress: {:>3d}% \n  Trans: {:g}/{:g}\n"
                  "  Vol: {:g}/{:g}\n  Ex: {:g}/{:g}\n"
                  "  Replica Volume N{} Number-density:\n"
                  "   1: {:g} {:d} {:g}\n"
                  "   2: {:g} {:d} {:g}\n",
                  progress, d[0], d[1], d[2], d[3], d[4], d[5], molflag ? "molecules" : "particles",
                  vol1, n1, n1 / vol1, vol2, n2, n2 / vol2);
  if (universe->uscreen) utils::print(universe->uscreen, msg);
  if (universe->ulogfile) utils::print(universe->ulogfile, msg);
}

/* ----------------------------------------------------------------------
   update list of local atoms in fix group and their counts
------------------------------------------------------------------------- */

void FixGEMC::update_gas_atoms_list()
{
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  if (nlocal > gemc_nmax) {
    memory->sfree(local_gas_list);
    gemc_nmax = atom->nmax;
    local_gas_list = (int *) memory->smalloc(gemc_nmax * sizeof(int), "GEMC:local_gas_list");
  }

  natom_local = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) local_gas_list[natom_local++] = i;

  MPI_Allreduce(&natom_local, &natom_total, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&natom_local, &natom_lower, 1, MPI_INT, MPI_SUM, world);
  natom_lower -= natom_local;
}

/* ----------------------------------------------------------------------
   return local index of a randomly chosen group atom in my box
   or -1 if not owned by this rank. must be called by all ranks of the box.
------------------------------------------------------------------------- */

int FixGEMC::pick_random_gas_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int>(natom_total * random_world->uniform());
  if ((iwhichglobal >= natom_lower) && (iwhichglobal < natom_lower + natom_local))
    i = local_gas_list[iwhichglobal - natom_lower];
  return i;
}

/* ----------------------------------------------------------------------
   return 1 if flag (valid on rank 0) is set in either box, on all ranks of both boxes
------------------------------------------------------------------------- */

int FixGEMC::any_box(int flag)
{
  int result = 0;
  if (me == 0) {
    int other;
    MPI_Sendrecv(&flag, 1, MPI_INT, 1 - myworld, 0, &other, 1, MPI_INT, 1 - myworld, 0,
                 comm_replica, MPI_STATUS_IGNORE);
    result = (flag || other) ? 1 : 0;
  }
  MPI_Bcast(&result, 1, MPI_INT, 0, world);
  return result;
}

/* ----------------------------------------------------------------------
   joint Metropolis decision for a move that changes both boxes
   dU = my box's contribution to the effective energy change (valid on rank 0)
   overflow = 1 if my box's trial energy is unusable (valid on rank 0)
   both boxes compute the same sum and draw the same random number,
   so they always reach the same decision
------------------------------------------------------------------------- */

int FixGEMC::accept_both(double dU, int overflow)
{
  double rnd = random_universe->uniform();
  int success = 0;
  if (me == 0) {
    double mine[2] = {dU, (double) overflow};
    double other[2];
    MPI_Sendrecv(mine, 2, MPI_DOUBLE, 1 - myworld, 0, other, 2, MPI_DOUBLE, 1 - myworld, 0,
                 comm_replica, MPI_STATUS_IGNORE);
    double all_dU = mine[0] + other[0];
    if ((mine[1] == 0.0) && (other[1] == 0.0) && std::isfinite(all_dU))
      success = (all_dU <= 0.0) || (rnd < exp(-beta * all_dU));
  }
  MPI_Bcast(&success, 1, MPI_INT, 0, world);
  return success;
}

/* ----------------------------------------------------------------------
   move atoms to their owning ranks, rebuild ghost atoms and neighbor lists
------------------------------------------------------------------------- */

void FixGEMC::reset_comm()
{
  refresh_ghosts();
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);
}

/* ----------------------------------------------------------------------
   move atoms to their owning ranks and rebuild ghost atoms, no neighbor lists
   atoms are reordered, so the list of local group atoms is regenerated
------------------------------------------------------------------------- */

void FixGEMC::refresh_ghosts()
{
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  ghosts_stale = 0;
  update_gas_atoms_list();
}

/* ----------------------------------------------------------------------
   return 1 if a translation or exchange move in my box can use energy_local()
   the box must be larger than the pair cutoff, so that an atom does
   not interact with its own periodic images
------------------------------------------------------------------------- */

int FixGEMC::use_local()
{
  if (!local_flag) return 0;
  if (min_box_width() <= force->pair->cutforce) return 0;
  if (ghosts_stale) refresh_ghosts();
  return 1;
}

/* ----------------------------------------------------------------------
   pair energy of atom with local index i, type itype, and atom ID itag
   placed at coord with all owned and ghost atoms, except for its own images.
   i may be a scratch index beyond the ghost atoms for an atom not yet inserted.
   if coord2 is not null, also return the energy at coord2 in energy2
------------------------------------------------------------------------- */

double FixGEMC::energy_local(int i, int itype, tagint itag, double *coord, double *coord2,
                             double *energy2)
{
  double **x = atom->x;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nall = atom->nlocal + atom->nghost;
  Pair *pair = force->pair;
  double **cutsq = pair->cutsq;
  double *cutsqi = cutsq[itype];
  double fpair;

  double total_energy = 0.0;
  double total_energy2 = 0.0;
  for (int j = 0; j < nall; j++) {
    if (tag[j] == itag) continue;
    int jtype = type[j];
    double delx = coord[0] - x[j][0];
    double dely = coord[1] - x[j][1];
    double delz = coord[2] - x[j][2];
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq < cutsqi[jtype]) total_energy += pair->single(i, j, itype, jtype, rsq, 1.0, 1.0, fpair);
    if (coord2) {
      delx = coord2[0] - x[j][0];
      dely = coord2[1] - x[j][1];
      delz = coord2[2] - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < cutsqi[jtype])
        total_energy2 += pair->single(i, j, itype, jtype, rsq, 1.0, 1.0, fpair);
    }
  }
  if (energy2) *energy2 = total_energy2;
  return total_energy;
}

/* ----------------------------------------------------------------------
   compute system potential energy
------------------------------------------------------------------------- */

double FixGEMC::energy_full()
{
  reset_comm();
  int eflag = 1;
  int vflag = 0;

  // clear forces so they don't accumulate over multiple
  // calls within fix gemc timestep, e.g. for fix shake

  size_t nbytes = sizeof(double) * (atom->nlocal + atom->nghost);
  if (nbytes) memset(&atom->f[0][0], 0, 3 * nbytes);

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag, vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (force->kspace) force->kspace->compute(eflag, vflag);

  if (modify->n_post_force_any) modify->post_force(vflag);

  // NOTE: all fixes with energy_global_flag set and which
  //   operate at pre_force() or post_force()
  //   and which user has enabled via fix_modify energy yes,
  //   will contribute to total MC energy via pe->compute_scalar()

  update->eflag_global = update->ntimestep;
  return c_pe->compute_scalar();
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixGEMC::write_restart(FILE *fp)
{
  int n = 0;
  double list[14];
  list[n++] = -RESTART_VERSION;
  list[n++] = random_proc->state();
  list[n++] = random_world->state();
  list[n++] = random_universe->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = ntranslation_attempts;
  list[n++] = ntranslation_successes;
  list[n++] = nexchange_attempts;
  list[n++] = nexchange_successes;
  list[n++] = nvolume_attempts;
  list[n++] = nvolume_successes;
  list[n++] = nrotation_attempts;
  list[n++] = nrotation_successes;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list, sizeof(double), n, fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
   restart files written before the version marker was added start with the
   (positive) random number generator state and contain rotation counters
------------------------------------------------------------------------- */

void FixGEMC::restart(char *buf)
{
  int n = 0;
  auto *list = (double *) buf;

  int oldformat = (list[0] > 0.0);
  int version = oldformat ? 1 : static_cast<int>(-list[0]);
  if (!oldformat) n++;

  // only the state of rank 0 was saved, so give each rank a distinct stream

  double coord[3] = {(double) universe->me, (double) myworld, 0.0};
  random_proc->reset(static_cast<int>(list[n++]), coord);
  random_world->reset(static_cast<int>(list[n++]));
  random_universe->reset(static_cast<int>(list[n++]));

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  ntranslation_attempts = list[n++];
  ntranslation_successes = list[n++];
  if (oldformat) n += 2;
  nexchange_attempts = list[n++];
  nexchange_successes = list[n++];
  nvolume_attempts = list[n++];
  nvolume_successes = list[n++];
  if (version >= 3) {
    nrotation_attempts = list[n++];
    nrotation_successes = list[n++];
  }

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR, "Must not reset timestep when restarting fix gemc");
}

/* ----------------------------------------------------------------------
   return cumulative attempt and success counts of this box
------------------------------------------------------------------------- */

double FixGEMC::compute_vector(int n)
{
  if (n == 0) return ntranslation_attempts;
  if (n == 1) return ntranslation_successes;
  if (n == 2) return nexchange_attempts;
  if (n == 3) return nexchange_successes;
  if (n == 4) return nvolume_attempts;
  if (n == 5) return nvolume_successes;
  if (n == 6) return displace;
  if (n == 7) return max_dlogvolratio;
  if (n == 8) return nrotation_attempts;
  if (n == 9) return nrotation_successes;
  if (n == 10) return maxangle / DEG2RAD;
  return 0.0;
}

/* ----------------------------------------------------------------------
   volume of the box (also for triclinic boxes)
------------------------------------------------------------------------- */

double FixGEMC::box_volume()
{
  return domain->xprd * domain->yprd * domain->zprd;
}

/* ----------------------------------------------------------------------
   smallest distance between opposite faces of the box
------------------------------------------------------------------------- */

double FixGEMC::min_box_width()
{
  double w[3];
  box_widths(w);
  return MIN(MIN(w[0], w[1]), w[2]);
}

/* ----------------------------------------------------------------------
   distances between opposite faces of the box
------------------------------------------------------------------------- */

void FixGEMC::box_widths(double *w)
{
  if (!triclinic) {
    w[0] = domain->xprd;
    w[1] = domain->yprd;
    w[2] = domain->zprd;
    return;
  }

  // rows of the inverse box matrix are normal to the box faces,
  // their lengths are the inverse face distances

  double *h_inv = domain->h_inv;
  w[0] = 1.0 / sqrt(h_inv[0] * h_inv[0] + h_inv[5] * h_inv[5] + h_inv[4] * h_inv[4]);
  w[1] = 1.0 / sqrt(h_inv[1] * h_inv[1] + h_inv[3] * h_inv[3]);
  w[2] = 1.0 / h_inv[2];
}

/* ----------------------------------------------------------------------
   largest distance an atom may move in a translation or rotation.
   with more than one rank per box, comm->exchange() in energy_full() can
   only move an atom to a neighboring subdomain, so the distance must stay
   below the smallest subdomain width
------------------------------------------------------------------------- */

double FixGEMC::max_move()
{
  if (comm->nprocs == 1) return BIG;
  double w[3];
  box_widths(w);
  double wsub = BIG;
  for (int k = 0; k < 3; k++) {
    double frac = triclinic ? domain->subhi_lamda[k] - domain->sublo_lamda[k]
                            : (domain->subhi[k] - domain->sublo[k]) / domain->prd[k];
    wsub = MIN(wsub, frac * w[k]);
  }
  double wsub_all;
  MPI_Allreduce(&wsub, &wsub_all, 1, MPI_DOUBLE, MPI_MIN, world);
  return 0.5 * wsub_all;
}

/* ----------------------------------------------------------------------
   maximum displacement actually used for translations. the box does not
   change during a translation, so the move remains symmetric
------------------------------------------------------------------------- */

double FixGEMC::max_translation()
{
  return MIN(displace, max_move());
}

/* ----------------------------------------------------------------------
   uniformly distributed random point in my box, identical on all ranks of the box
------------------------------------------------------------------------- */

void FixGEMC::random_point(double *coord)
{
  double lamda[3];
  lamda[0] = random_world->uniform();
  lamda[1] = random_world->uniform();
  lamda[2] = random_world->uniform();
  domain->lamda2x(lamda, coord);
  domain->remap(coord);
}

/* ----------------------------------------------------------------------
   1 if a point inside the box belongs to the subdomain of this rank
------------------------------------------------------------------------- */

int FixGEMC::owns(double *coord)
{
  double *lo, *hi;
  double c[3];
  if (triclinic) {
    domain->x2lamda(coord, c);
    lo = domain->sublo_lamda;
    hi = domain->subhi_lamda;

    // round-off may place a point just outside the unit cube

    for (int k = 0; k < 3; k++) {
      if (c[k] >= 1.0) c[k] -= 1.0;
      if (c[k] < 0.0) c[k] += 1.0;
      if (c[k] >= 1.0) c[k] = 0.0;
    }
  } else {
    c[0] = coord[0];
    c[1] = coord[1];
    c[2] = coord[2];
    lo = domain->sublo;
    hi = domain->subhi;
  }
  return (c[0] >= lo[0]) && (c[0] < hi[0]) && (c[1] >= lo[1]) && (c[1] < hi[1]) &&
      (c[2] >= lo[2]) && (c[2] < hi[2]);
}

/* ----------------------------------------------------------------------
   return local index of the owned atom with atom ID itag or -1 if not owned
------------------------------------------------------------------------- */

int FixGEMC::local_index(tagint itag)
{
  if (atom->map_style != Atom::MAP_NONE) {
    int i = atom->map(itag);
    return (i < atom->nlocal) ? i : -1;
  }
  for (int i = 0; i < atom->nlocal; i++)
    if (atom->tag[i] == itag) return i;
  return -1;
}

/* ----------------------------------------------------------------------
   update after atoms were added or removed or charges changed
   ghost atoms are rebuilt by the next energy_full() or use_local()
------------------------------------------------------------------------- */

void FixGEMC::changed_atoms()
{
  // new or restored atoms were appended after the owned atoms and have
  // overwritten ghost atoms, so the ghost atoms are invalid until rebuilt

  atom->nghost = 0;
  ghosts_stale = 1;
  if (atom->map_style != Atom::MAP_NONE) atom->map_init();
  if (force->kspace) force->kspace->qsum_qsq();
  if (force->pair && force->pair->tail_flag) force->pair->reinit();
}

/* ----------------------------------------------------------------------
   set box to lower corner lo (unchanged) with given upper bounds and tilts
------------------------------------------------------------------------- */

void FixGEMC::set_box(double xhi_new, double yhi_new, double zhi_new, double xy_new, double xz_new,
                      double yz_new)
{
  domain->boxhi[0] = xhi_new;
  domain->boxhi[1] = yhi_new;
  domain->boxhi[2] = zhi_new;
  if (triclinic) {
    domain->xy = xy_new;
    domain->xz = xz_new;
    domain->yz = yz_new;
  }
  domain->set_global_box();
  domain->set_local_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  if (force->kspace) force->kspace->setup();
}

/* ----------------------------------------------------------------------
   scale positions for a change of all box lengths by factor s about the
   lower box corner lo, before the box itself is changed.
   atoms that belong to a molecule are displaced together with the center of
   mass of their molecule, so molecules are not deformed:
     unwrapped x += (s-1) (xcm - lo)
   for the wrapped coordinate x of an atom with unwrapped coordinate xu
   this is x += (s-1) (x + xcm - xu - lo).  atoms without a molecule ID
   are scaled individually, x += (s-1) (x - lo).
   returns the number of independent units (molecules plus single atoms).
   if check is set and the scaled box would be narrower than twice the largest
   molecule, nothing is changed and -1 is returned in both boxes, since bonded
   interactions use the closest periodic image of bond partners.
   must be called at the same time in both boxes if check is set.
------------------------------------------------------------------------- */

bigint FixGEMC::scale_positions(double s, int check)
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  imageint *image = atom->image;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  double *lo = domain->boxlo;
  double sm1 = s - 1.0;

  // centers of mass of all molecules with atoms on this rank, from all ranks

  std::map<tagint, std::array<double, 4>> com;
  bigint nfree = 0;
  if (molecule) {
    std::map<tagint, std::array<double, 4>> partial;
    double xu[3];
    for (int i = 0; i < nlocal; i++) {
      if (molecule[i] <= 0) continue;
      double m = rmass ? rmass[i] : mass[type[i]];
      domain->unmap(x[i], image[i], xu);
      auto &p = partial[molecule[i]];
      p[0] += m;
      p[1] += m * xu[0];
      p[2] += m * xu[1];
      p[3] += m * xu[2];
    }
    std::vector<double> mine;
    mine.reserve(5 * partial.size());
    for (const auto &p : partial) {
      mine.push_back(ubuf(p.first).d);
      for (int k = 0; k < 4; k++) mine.push_back(p.second[k]);
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
    for (int k = 0; k < ntotal; k += 5) {
      auto &c = com[(tagint) ubuf(all[k]).i];
      for (int n = 0; n < 4; n++) c[n] += all[k + 1 + n];
    }
    for (auto &c : com) {
      c.second[1] /= c.second[0];
      c.second[2] /= c.second[0];
      c.second[3] /= c.second[0];
    }
    for (int i = 0; i < nlocal; i++)
      if (molecule[i] <= 0) nfree++;
  } else {
    nfree = nlocal;
  }
  bigint nfree_all;
  MPI_Allreduce(&nfree, &nfree_all, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  // largest distance of an atom from the center of mass of its molecule

  if (check) {
    double rmax = 0.0;
    if (molecule) {
      double xu[3];
      for (int i = 0; i < nlocal; i++) {
        if (molecule[i] <= 0) continue;
        const auto &c = com[molecule[i]];
        domain->unmap(x[i], image[i], xu);
        double dx = xu[0] - c[1];
        double dy = xu[1] - c[2];
        double dz = xu[2] - c[3];
        rmax = MAX(rmax, dx * dx + dy * dy + dz * dz);
      }
    }
    double rmax_all;
    MPI_Allreduce(&rmax, &rmax_all, 1, MPI_DOUBLE, MPI_MAX, world);
    if (any_box(min_box_width() * s <= 4.0 * sqrt(rmax_all))) return -1;
  }

  double xu[3];
  for (int i = 0; i < nlocal; i++) {
    if (molecule && (molecule[i] > 0)) {
      const auto &c = com[molecule[i]];
      domain->unmap(x[i], image[i], xu);
      x[i][0] += sm1 * (x[i][0] + c[1] - xu[0] - lo[0]);
      x[i][1] += sm1 * (x[i][1] + c[2] - xu[1] - lo[1]);
      x[i][2] += sm1 * (x[i][2] + c[3] - xu[2] - lo[2]);
    } else {
      x[i][0] += sm1 * (x[i][0] - lo[0]);
      x[i][1] += sm1 * (x[i][1] - lo[1]);
      x[i][2] += sm1 * (x[i][2] - lo[2]);
    }
  }

  return nfree_all + (bigint) com.size();
}
