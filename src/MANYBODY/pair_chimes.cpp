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
   Contributing author: Rebecca K. Lindsey (LLNL)
------------------------------------------------------------------------- */

#include "pair_chimes.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "math.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "mpi.h"
#include "my_page.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "update.h"

#include <iostream>
#include <vector>

using namespace LAMMPS_NS;

/*    Functions required by LAMMPS:


settings     (done)        reads the input script line with arguments defined here
coeff        (done)        set coefficients for one i,j pair type
compute        (done)        workhorse routine that computes pairwise interactions
init_one    (done)        perform initalization for one i,j type pair
init_style     (done)        initialization specific to this pair style

write_restart            write i,j pair coeffs to restart file
read_restart            read i,j pair coeffs from restart file
write_restart_settings            write global settings to restart file
read_restart_settings            read global settings from restart file
single                force and energy fo a single pairwise interaction between two atoms
*/

PairCHIMES::PairCHIMES(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;

  const int me = comm->me;

  chimes_calculator = new chimesFF();

  chimes_calculator->init(me);

  // 2, 3, and 4-body vars for chimesFF access

  dr.resize(CHDIM);
  dr_3b.resize(3 * CHDIM);
  dr_4b.resize(6 * CHDIM);

  dist_3b.resize(3);
  dist_4b.resize(6);

  // CHDIM is the number of spatial dimensions (usually 3).
  force_2b.resize(2 * CHDIM);
  force_3b.resize(3 * CHDIM);
  force_4b.resize(4 * CHDIM);

  typ_idxs_2b.resize(2);
  typ_idxs_3b.resize(3);
  typ_idxs_4b.resize(4);

  n_3mers = 0;
  n_4mers = 0;

  if (chimes_calculator->rank == 0) {
    std::cout << std::endl;
    std::cout << "************************* WARNING (pair_style chimesFF) ************************"
              << std::endl;
    std::cout << "Assuming n-body interactions have longer cutoffs than all (n+1)-body interactions"
              << std::endl;
    std::cout << "************************* WARNING (pair_style chimesFF) ************************"
              << std::endl;
    std::cout << std::endl;
  }
}

PairCHIMES::~PairCHIMES()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }

  delete chimes_calculator;
}

void PairCHIMES::settings(int narg, char **arg)
{
  if (narg != 0)
    error->all(FLERR, "Illegal pair_style command. Expects no arguments beyond pair_style name.");

  return;
}

void PairCHIMES::coeff(int narg, char **arg)
{
  // Expect: pair_coeff * * <parameter file name>

  if (narg != 3)
    error->all(FLERR,
               "Illegal pair_style command. Expects \"pair_coeff * * <parameter file name>\" ");

  chimesFF_paramfile = arg[2];

  chimes_calculator->read_parameters(chimesFF_paramfile);

  set_chimes_type();

  //chimes_calculator->set_atomtypes(chimes_type);
  chimes_calculator->build_pair_int_trip_map();
  chimes_calculator->build_pair_int_quad_map();
  chimes_calculator->build_interaction_tables();

  // Set special LAMMPS flags/cutoffs

  if (!allocated) allocate();

  vector<vector<double>> cutoff_2b;
  chimes_calculator->get_cutoff_2B(cutoff_2b);

  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      setflag[i][j] = 1;
      setflag[j][i] = 1;

      cutsq[i][j] = cutoff_2b[chimes_calculator->get_atom_pair_index(
          chimes_type[i - 1] * chimes_calculator->natmtyps + chimes_type[j - 1])][1];
      cutsq[i][j] *= cutsq[i][j];

      if (i != j) {
        cutsq[j][i] = cutoff_2b[chimes_calculator->get_atom_pair_index(
            chimes_type[j - 1] * chimes_calculator->natmtyps + chimes_type[i - 1])][1];
        cutsq[j][i] *= cutsq[j][i];
      }
    }
  }

  maxcut_2b = chimes_calculator->max_cutoff_2B();
  maxcut_3b = chimes_calculator->max_cutoff_3B();
  maxcut_4b = chimes_calculator->max_cutoff_4B();
}

void PairCHIMES::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes + 1, atom->ntypes + 1, "pair:setflag");

  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes + 1, atom->ntypes + 1, "pair:cutsq");
}

void PairCHIMES::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style ChIMES requires atom IDs");

  if (force->newton_pair == 0) error->all(FLERR, "Pair style ChIMES requires newton pair on");

  // Set up neighbor lists... borrowing this from pair_airebo:
  // need a full neighbor list, including neighbors of ghosts

  // int irequest = neighbor->request(this, instance_me);
  // neighbor->requests[irequest]->half = 0;
  // neighbor->requests[irequest]->full = 1;
  // neighbor->requests[irequest]->ghost = 1;

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
}

double PairCHIMES::init_one(int i, int j)
{
  // Sets the cutoff for each pair interaction.
  // The maximum of the returned values are used to set outer cutoff for neighbor lists
  // WARNING: This means linking won't work properly if 2-b interactions do not have larger cutoffs than all other
  // higher bodied interactions!!

  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return sqrt(cutsq[i][j]);
}

inline double PairCHIMES::get_dist(int i, int j, double *dr)
{
  double **x = atom->x;    // Access to system coordinates

  dr[0] = x[j][0] - x[i][0];
  dr[1] = x[j][1] - x[i][1];
  dr[2] = x[j][2] - x[i][2];

  return sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
}

inline double PairCHIMES::get_dist(int i, int j)
{
  double dummy_dr[3];

  return get_dist(i, j, dummy_dr);
}

void PairCHIMES::build_mb_neighlists()
{

  if ((chimes_calculator->poly_orders[1] == 0) && (chimes_calculator->poly_orders[2] == 0)) return;

  // List gets built based on atoms owned by calling proc.
  // resize(0) rather than clear() so the capacity survives to the next rebuild.

  neighborlist_3mers.resize(0);
  neighborlist_4mers.resize(0);
  n_3mers = 0;
  n_4mers = 0;

  int i, j, k, l, inum, jnum, ii, jj, kk, ll;     // Local iterator vars
  int *ilist, *jlist, *numneigh, **firstneigh;    // Local neighborlist vars
  tagint *tag = atom->tag;                        // Access to global atom indices
  tagint itag, jtag, ktag, ltag;                  // holds tags
  double **x = atom->x;                           // Access to system coordinates

  const double maxcut_3b_padded = maxcut_3b + neighbor->skin;
  const double maxcut_4b_padded = maxcut_4b + neighbor->skin;

  // Every comparison below is against a padded cutoff, so the distances
  // themselves are never needed -- only their ordering against those cutoffs.
  // Working in squared distances therefore removes every sqrt from the build.
  // A pair can only land on the wrong side of a squared comparison when it sits
  // within one ulp of the padded cutoff, and such a cluster is beyond every
  // unpadded cutoff, so compute_3B/compute_4B reject it and it contributes
  // exactly zero either way.

  const double cutsq_3b = maxcut_3b_padded * maxcut_3b_padded;
  const double cutsq_4b = maxcut_4b_padded * maxcut_4b_padded;
  const double cutsq_max = MAX(cutsq_3b, cutsq_4b);

  const bool do_4b = (chimes_calculator->poly_orders[2] > 0);

  ////////////////////////////////////////
  // Access to neighbor list vars
  ////////////////////////////////////////

  inum = list->inum;                // length of the list
  ilist = list->ilist;              // list of i atoms for which neighbor list exists
  numneigh = list->numneigh;        // length of each of the ilist neighbor lists
  firstneigh = list->firstneigh;    // point to the list of neighbors of i

  for (ii = 0; ii < inum; ii++)    // Loop over real atoms (ai)
  {
    i = ilist[ii];
    itag = tag[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    const double xi = x[i][0];
    const double yi = x[i][1];
    const double zi = x[i][2];

    // One pass over i's neighbors collects every atom that can take part in a
    // cluster owned by i, together with its distance to i.  The old code
    // rescanned firstneigh[i] from the start for k and again for l, so the ik
    // distance was recomputed once per j and the il distance once per (j,k)
    // pair.  Filtering preserves the neighbor list order, so iterating these
    // candidate arrays visits the same clusters in the same sequence.

    cand_3b.resize(0);
    cand_4b.resize(0);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj] & NEIGHMASK;

      if (j == i) continue;

      jtag = tag[j];

      if (jtag < itag) continue;

      const double dx = x[j][0] - xi;
      const double dy = x[j][1] - yi;
      const double dz = x[j][2] - zi;
      const double rsq = dx * dx + dy * dy + dz * dz;

      if (rsq >= cutsq_max) continue;

      cand_3b.push_back({j, jtag, rsq});

      if (do_4b && (rsq < cutsq_4b)) cand_4b.push_back({j, jtag, rsq});
    }

    ////////////////////////////////////////
    // 3-body clusters
    ////////////////////////////////////////

    if (chimes_calculator->poly_orders[1] > 0) {
      const int n3 = cand_3b.size();

      for (jj = 0; jj < n3; jj++) {
        if (cand_3b[jj].rsq >= cutsq_3b) continue;

        j = cand_3b[jj].idx;
        jtag = cand_3b[jj].tag;

        for (kk = 0; kk < n3; kk++) {
          k = cand_3b[kk].idx;

          if (k == j) continue;

          ktag = cand_3b[kk].tag;

          if (ktag < jtag) continue;
          if (cand_3b[kk].rsq >= cutsq_3b) continue;

          if (dist_sq(x, j, k) >= cutsq_3b) continue;

          neighborlist_3mers.push_back(i);
          neighborlist_3mers.push_back(j);
          neighborlist_3mers.push_back(k);
          n_3mers++;
        }
      }
    }

    ////////////////////////////////////////
    // 4-body clusters
    ////////////////////////////////////////

    if (!do_4b) continue;

    const int n4 = cand_4b.size();

    if (n4 < 3) continue;

    // Tabulate the squared distances among the 4-body candidates once.  The
    // jl and kl distances were previously recomputed for every (j,k,l) triple;
    // there are only n4*(n4-1)/2 distinct values, and for this benchmark that
    // is roughly a factor of forty fewer evaluations.

    cand_rsq.resize((size_t) n4 * n4);

    for (int a = 0; a < n4; a++) {
      cand_rsq[(size_t) a * n4 + a] = 0.0;

      for (int b = a + 1; b < n4; b++) {
        const double rsq = dist_sq(x, cand_4b[a].idx, cand_4b[b].idx);

        cand_rsq[(size_t) a * n4 + b] = rsq;
        cand_rsq[(size_t) b * n4 + a] = rsq;
      }
    }

    for (jj = 0; jj < n4; jj++) {
      j = cand_4b[jj].idx;
      jtag = cand_4b[jj].tag;

      const double *rsq_j = &cand_rsq[(size_t) jj * n4];

      for (kk = 0; kk < n4; kk++) {
        k = cand_4b[kk].idx;

        if (k == j) continue;

        ktag = cand_4b[kk].tag;

        if (ktag < jtag) continue;
        if (rsq_j[kk] >= cutsq_4b) continue;

        const double *rsq_k = &cand_rsq[(size_t) kk * n4];

        for (ll = 0; ll < n4; ll++) {
          l = cand_4b[ll].idx;

          if ((l == j) || (l == k)) continue;

          ltag = cand_4b[ll].tag;

          if ((ltag < jtag) || (ltag < ktag)) continue;
          if (rsq_j[ll] >= cutsq_4b) continue;
          if (rsq_k[ll] >= cutsq_4b) continue;

          neighborlist_4mers.push_back(i);
          neighborlist_4mers.push_back(j);
          neighborlist_4mers.push_back(k);
          neighborlist_4mers.push_back(l);
          n_4mers++;
        }
      }
    }
  }

  sort_3mers_by_type();
}

/* ----------------------------------------------------------------------
   Group the triplets by cluster type so that the force loop can hand a run of
   same-typed clusters to the batched evaluator.  A counting sort over the
   packed type index, which is small (ntypes^3), keeps this linear; it runs once
   per neighbor rebuild rather than once per step.
------------------------------------------------------------------------- */

void PairCHIMES::sort_3mers_by_type()
{
  if ((chimes_calculator->poly_orders[1] == 0) || (n_3mers == 0)) return;

  const int nt = chimes_calculator->natmtyps;
  const int nkey = nt * nt * nt;
  int *type = atom->type;

  mer_key.resize(n_3mers);
  type_count.assign(nkey + 1, 0);

  for (int c = 0; c < n_3mers; c++) {
    const int *m = &neighborlist_3mers[3 * c];
    const int key = chimes_calculator->type_index_3B(chimes_type[type[m[0]] - 1],
                                                     chimes_type[type[m[1]] - 1],
                                                     chimes_type[type[m[2]] - 1]);
    mer_key[c] = key;
    type_count[key + 1]++;
  }

  for (int t = 0; t < nkey; t++) type_count[t + 1] += type_count[t];

  mer_scratch.resize((size_t) n_3mers * 3);

  for (int c = 0; c < n_3mers; c++) {
    const int dst = type_count[mer_key[c]]++;

    mer_scratch[3 * dst + 0] = neighborlist_3mers[3 * c + 0];
    mer_scratch[3 * dst + 1] = neighborlist_3mers[3 * c + 1];
    mer_scratch[3 * dst + 2] = neighborlist_3mers[3 * c + 2];
  }

  neighborlist_3mers.swap(mer_scratch);
}

void PairCHIMES::compute(int eflag, int vflag)
{
  // Vars for access to chimesFF compute_XB functions

  std::vector<double> stensor(6);    // pointers to system stress tensor

  // Atom indices of the current cluster, passed on to ev_tally_mb

  int atmlist[4];

  // General LAMMPS compute vars

  int i, j, k, l, inum, jnum, ii, jj;             // Local iterator vars
  int *ilist, *jlist, *numneigh, **firstneigh;    // Local neighborlist vars
  int idx;

  double **x = atom->x;    // Access to system coordinates
  double **f = atom->f;    // Access to system forces

  int *type =
      atom->type;    // Acces to system atom types (countng starts from 1, chimesFF class expects counting from 0!)
  tagint *tag = atom->tag;    // Access to global atom indices (sort of like "parent" indices)
  int itag, jtag;             // holds tags
  double energy;              // pair energy

  // Set up vars controlling if energy/pressure (virial) contributions are computed

  if (eflag || vflag) {
    ev_setup(eflag, vflag);
  } else {
    evflag = 0;
    vflag_fdotr = 0;
    vflag_atom = 0;
  }

  ////////////////////////////////////////
  // Access to (2-body) neighbor list vars
  ////////////////////////////////////////

  inum = list->inum;                // length of the list
  ilist = list->ilist;              // list of i atoms for which neighbor list exists
  numneigh = list->numneigh;        // length of each of the ilist neighbor lists
  firstneigh = list->firstneigh;    // point to the list of neighbors of i

  chimes2BTmp chimes_2btmp(chimes_calculator->poly_orders[0]);
  chimes3BTmp chimes_3btmp(chimes_calculator->poly_orders[1]);
  chimes3BBatch chimes_3bbatch(chimes_calculator->poly_orders[1]);
  chimes4BTmp chimes_4btmp(chimes_calculator->poly_orders[2]);

  // Build the ChIMES many-body neighbor lists.. only do so when LAMMPS neighborlist has been updated

  if (neighbor->ago == 0) build_mb_neighlists();


  ////////////////////////////////////////
  // Compute 1- and 2-body interactions
  ////////////////////////////////////////

  for (ii = 0; ii < inum; ii++)    // Loop over the atoms owned by the current process
  {
    i = ilist[ii];    // Index of the current atom
    itag = tag[i];    // Get i's global atom index (sort of like its "parent")

    jlist = firstneigh[i];    // Neighborlist for atom i
    jnum = numneigh[i];       // Number of neighbors of atom i

    // Type (index) of the current atom... subtract 1 to account for chimesFF
    // vs LAMMPS numbering convention

    typ_idxs_2b[0] = chimes_type[type[i] - 1];

    // First, get the single-atom energy contribution

    energy = 0.0;

    chimes_calculator->compute_1B(type[i] - 1, energy);

    atmlist[0] = i;

    if (evflag) ev_tally_mb(1, atmlist, energy, stensor.data());

    // Now move on to two-body force, stress, and energy

    for (jj = 0; jj < jnum; jj++)    // Loop over neighbors of i
    {
      j = jlist[jj] & NEIGHMASK;    // Index of the jj atom, extra bits stripped
      jtag = tag[j];                // Get j's global atom index (sort of like its "parent")

      if (jtag <=
          itag)    // only allow calculation for j<i, since we've requested a full neighbor list
        continue;

      // Get distance using ghost atoms... don't need MIC since we're using ghost atoms

      dist = get_dist(i, j, &dr[0]);

      typ_idxs_2b[1] = chimes_type[type[j] - 1];

      // Using std::fill for maximum efficiency.
      std::fill(force_2b.begin(), force_2b.end(), 0.0);

      // Do the same for stress tensors, but only when a virial was asked for
      if (vflag_either) std::fill(stensor.begin(), stensor.end(), 0.0);

      energy = 0.0;

      chimes_calculator->compute_2B(dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp,
                                    vflag_either);

      for (idx = 0; idx < 3; idx++) {
        f[i][idx] += force_2b[0 * CHDIM + idx];
        f[j][idx] += force_2b[1 * CHDIM + idx];
      }

      // "Save"/tally up the energy and stresses to the global virial/energy data objects (see pair.cpp ~ line 1000)
      // Compute pressure, (in contrast to chimes_md) AFTER penalty has been added

      atmlist[1] = j;

      if (evflag) ev_tally_mb(2, atmlist, energy, stensor.data());
    }
  }

  if (chimes_calculator->poly_orders[1] > 0) {
    ////////////////////////////////////////
    // Compute 3-body interactions
    ////////////////////////////////////////

    // The triplets arrive grouped by cluster type, so clusters that survive the
    // cutoff test are collected into a batch of CHIMES_VLEN and evaluated
    // together.  Rejected clusters never enter a lane, so no lane is wasted on
    // them, and the batch is flushed whenever it fills or the type changes.

    double bdx[3][CHIMES_VLEN];
    double bdr[3][CHDIM][CHIMES_VLEN];
    int batom[3][CHIMES_VLEN];
    int nb = 0;
    int batch_type = -1;

    for (ii = 0; ii <= n_3mers; ii++) {
      int this_type = -1;
      const chimesSlotConst *sc3 = nullptr;

      if (ii < n_3mers) {
        const int *mer = &neighborlist_3mers[3 * ii];
        i = mer[0];
        j = mer[1];
        k = mer[2];

        typ_idxs_3b[0] = chimes_type[type[i] - 1];
        typ_idxs_3b[1] = chimes_type[type[j] - 1];
        typ_idxs_3b[2] = chimes_type[type[k] - 1];

        sc3 = chimes_calculator->slots_3B(typ_idxs_3b[0], typ_idxs_3b[1], typ_idxs_3b[2]);

        if (sc3) {
          if (within(x, i, j, sc3[0].outer_sq, &dr_3b[0 * CHDIM], dist_3b[0]) &&
              within(x, i, k, sc3[1].outer_sq, &dr_3b[1 * CHDIM], dist_3b[1]) &&
              within(x, j, k, sc3[2].outer_sq, &dr_3b[2 * CHDIM], dist_3b[2]))
            this_type = chimes_calculator->type_index_3B(typ_idxs_3b[0], typ_idxs_3b[1],
                                                         typ_idxs_3b[2]);
        }
      }

      // A cluster outside the cutoffs is simply skipped: it must not disturb
      // the batch in hand, or the 44% of clusters that fail would flush it
      // constantly and leave every batch a fraction full.

      const bool at_end = (ii == n_3mers);

      if (!at_end && (this_type < 0)) continue;

      // Flush first if this cluster cannot join the batch in hand.

      if ((nb > 0) && (at_end || (this_type != batch_type) || (nb == CHIMES_VLEN))) {
        for (int p = 0; p < 3; p++)
          for (int l = nb; l < CHIMES_VLEN; l++) bdx[p][l] = bdx[p][0];

        chimes_calculator->compute_3B_batch(nb, batch_type, bdx, chimes_3bbatch);

        for (int l = 0; l < nb; l++) {
          const double fc0 = chimes_3bbatch.fcut[0][l];
          const double fc1 = chimes_3bbatch.fcut[1][l];
          const double fc2 = chimes_3bbatch.fcut[2][l];
          const double fcut_all = fc0 * fc1 * fc2;
          const double poly = chimes_3bbatch.poly[l];

          double fs[3];

          fs[0] = (fcut_all * chimes_3bbatch.dpoly[0][l] +
                   chimes_3bbatch.fcutderiv[0][l] * fc1 * fc2 * poly) /
              bdx[0][l];
          fs[1] = (fcut_all * chimes_3bbatch.dpoly[1][l] +
                   chimes_3bbatch.fcutderiv[1][l] * fc0 * fc2 * poly) /
              bdx[1][l];
          fs[2] = (fcut_all * chimes_3bbatch.dpoly[2][l] +
                   chimes_3bbatch.fcutderiv[2][l] * fc0 * fc1 * poly) /
              bdx[2][l];

          // Pair p acts between the two atoms of the cluster it connects:
          // 0 = ij, 1 = ik, 2 = jk.

          static const int pa[3] = {0, 0, 1}, pb[3] = {1, 2, 2};

          if (vflag_either) std::fill(stensor.begin(), stensor.end(), 0.0);

          for (int p = 0; p < 3; p++) {
            const int a = batom[pa[p]][l], b = batom[pb[p]][l];

            for (idx = 0; idx < CHDIM; idx++) {
              f[a][idx] += fs[p] * bdr[p][idx][l];
              f[b][idx] -= fs[p] * bdr[p][idx][l];
            }

            if (vflag_either) {
              stensor[0] -= fs[p] * bdr[p][0][l] * bdr[p][0][l];
              stensor[1] -= fs[p] * bdr[p][0][l] * bdr[p][1][l];
              stensor[2] -= fs[p] * bdr[p][0][l] * bdr[p][2][l];
              stensor[3] -= fs[p] * bdr[p][1][l] * bdr[p][1][l];
              stensor[4] -= fs[p] * bdr[p][1][l] * bdr[p][2][l];
              stensor[5] -= fs[p] * bdr[p][2][l] * bdr[p][2][l];
            }
          }

          if (evflag) {
            atmlist[0] = batom[0][l];
            atmlist[1] = batom[1][l];
            atmlist[2] = batom[2][l];

            ev_tally_mb(3, atmlist, poly * fcut_all, stensor.data());
          }
        }

        nb = 0;
      }

      if (at_end) break;

      batch_type = this_type;
      batom[0][nb] = i;
      batom[1][nb] = j;
      batom[2][nb] = k;

      for (int p = 0; p < 3; p++) {
        bdx[p][nb] = dist_3b[p];

        for (idx = 0; idx < CHDIM; idx++) bdr[p][idx][nb] = dr_3b[p * CHDIM + idx];
      }

      nb++;
    }
  }

  if (chimes_calculator->poly_orders[2] > 0) {
    ////////////////////////////////////////
    // Compute 4-body interactions
    ////////////////////////////////////////

    for (ii = 0; ii < n_4mers; ii++) {
      const int *mer = &neighborlist_4mers[4 * ii];
      i = mer[0];
      j = mer[1];
      k = mer[2];
      l = mer[3];

      typ_idxs_4b[0] = chimes_type[type[i] - 1];
      typ_idxs_4b[1] = chimes_type[type[j] - 1];
      typ_idxs_4b[2] = chimes_type[type[k] - 1];
      typ_idxs_4b[3] = chimes_type[type[l] - 1];

      // As for the triplets, and it matters more here: two thirds of the
      // enumerated quadruplets are outside the real cutoffs on a given step,
      // and each one used to cost six square roots before being discarded.

      const chimesSlotConst *sc4 = chimes_calculator->slots_4B(
          typ_idxs_4b[0], typ_idxs_4b[1], typ_idxs_4b[2], typ_idxs_4b[3]);

      if (!sc4) continue;

      if (!within(x, i, j, sc4[0].outer_sq, &dr_4b[0 * CHDIM], dist_4b[0])) continue;
      if (!within(x, i, k, sc4[1].outer_sq, &dr_4b[1 * CHDIM], dist_4b[1])) continue;
      if (!within(x, i, l, sc4[2].outer_sq, &dr_4b[2 * CHDIM], dist_4b[2])) continue;
      if (!within(x, j, k, sc4[3].outer_sq, &dr_4b[3 * CHDIM], dist_4b[3])) continue;
      if (!within(x, j, l, sc4[4].outer_sq, &dr_4b[4 * CHDIM], dist_4b[4])) continue;
      if (!within(x, k, l, sc4[5].outer_sq, &dr_4b[5 * CHDIM], dist_4b[5])) continue;

      std::fill(force_4b.begin(), force_4b.end(), 0.0);

      if (vflag_either) std::fill(stensor.begin(), stensor.end(), 0.0);

      energy = 0.0;

      chimes_calculator->compute_4B(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy,
                                    chimes_4btmp, vflag_either);

      for (idx = 0; idx < 3; idx++) {
        f[i][idx] += force_4b[0 * CHDIM + idx];
        f[j][idx] += force_4b[1 * CHDIM + idx];
        f[k][idx] += force_4b[2 * CHDIM + idx];
        f[l][idx] += force_4b[3 * CHDIM + idx];
      }

      atmlist[0] = i;
      atmlist[1] = j;
      atmlist[2] = k;
      atmlist[3] = l;

      if (evflag) ev_tally_mb(4, atmlist, energy, stensor.data());
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

  return;
}

void PairCHIMES::set_chimes_type()
{
  int nmatches = 0;

  for (int i = 1; i <= atom->ntypes; i++)    // Lammps indexing starts at 1
  {
    for (int j = 0; j < chimes_calculator->natmtyps; j++)    // ChIMES indexing starts at 0
    {
      if (abs(atom->mass[i] - chimes_calculator->masses[j]) <
          1e-3)    // Masses should match to at least 3 decimal places
      {
        chimes_type.push_back(j);
        nmatches++;
      }
    }
  }

  if (nmatches < atom->ntypes) {
    std::cout << "ERROR: LAMMPS coordinate file has " << atom->ntypes << " atom type masses"
              << std::endl;
    std::cout << "       but only found " << nmatches << " matches with the ChIMES parameter file."
              << std::endl;
    exit(0);
  }
}

/* ----------------------------------------------------------------------
   general ev tally function for many-body models where per-atom assignments
   do not make sense. Expects newton_pair = 1.
 ------------------------------------------------------------------------- */

void PairCHIMES::ev_tally_mb(int ninteractionatoms, const int *atmlist, double evdwl,
                             const double *stress)
{
  // Assumes newton pair is always true
  // Assumes a full neighbor list is always true (hard coded in pair_chimes.cpp)
  // Modeled after ev_tally_full and ev_tally3 (to get MB handling)
  // atmlist holds the ninteractionatoms local atom indices of the cluster.

  if (eflag_global) eng_vdwl += evdwl;

  if (eflag_atom)
    for (int atm = 0; atm < ninteractionatoms; atm++)
      eatom[atmlist[atm]] += evdwl / ninteractionatoms;

  if (ninteractionatoms < 2) return;

  if (!vflag_either) return;

  // FYI, stress calculations follow strategy described here: https://docs.lammps.org/compute_stress_atom.html

  if (vflag_global) {
    virial[0] += stress[0];
    virial[1] += stress[3];
    virial[2] += stress[5];
    virial[3] += stress[1];
    virial[4] += stress[2];
    virial[5] += stress[4];
  }

  if (vflag_atom) {
    for (int a = 0; a < ninteractionatoms; a++) {
      vatom[atmlist[a]][0] += stress[0] / ninteractionatoms;
      vatom[atmlist[a]][1] += stress[3] / ninteractionatoms;
      vatom[atmlist[a]][2] += stress[5] / ninteractionatoms;
      vatom[atmlist[a]][3] += stress[1] / ninteractionatoms;
      vatom[atmlist[a]][4] += stress[2] / ninteractionatoms;
      vatom[atmlist[a]][5] += stress[4] / ninteractionatoms;
    }
  }
}

void PairCHIMES::write_restart() {}
void PairCHIMES::read_restart() {}
void PairCHIMES::write_restart_settings() {}
void PairCHIMES::read_restart_settings() {}
void PairCHIMES::single() {}
