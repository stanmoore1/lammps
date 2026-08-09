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

  // A full list, and only for the atoms this processor owns.
  //
  // The request used to ask for neighbors of ghosts as well, copied from
  // pair_airebo, which needs them because its bond order reaches through a
  // neighbor to that neighbor's own neighbors.  Nothing here does: both the
  // many-body list build and the force loop iterate ii < inum, which is the
  // owned atoms, and read firstneigh only for those.  A full list already
  // carries ghosts as neighbors of owned atoms; the flag additionally builds a
  // list for every ghost, and this model's cutoff makes ghosts the majority of
  // the atoms in the box.

  neighbor->add_request(this, NeighConst::REQ_FULL);
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

/* ----------------------------------------------------------------------
   Point the neighbor list members at the list this style requested.  An
   accelerated variant overrides this to supply its own storage instead.
------------------------------------------------------------------------- */

void PairCHIMES::setup_neighlist_ptrs()
{
  nl_inum = list->inum;
  nl_ilist = list->ilist;
  nl_numneigh = list->numneigh;
  nl_firstneigh = list->firstneigh;
}

/* ---------------------------------------------------------------------- */

PairCHIMES::MBContext PairCHIMES::mb_context() const
{
  MBContext ctx;

  ctx.x = atom->x;
  ctx.tag = atom->tag;

  // Every comparison in the enumeration is against a padded cutoff, so the
  // distances themselves are never needed -- only their ordering against those
  // cutoffs.  Working in squared distances therefore removes every sqrt from
  // the build.  A pair can only land on the wrong side of a squared comparison
  // when it sits within one ulp of the padded cutoff, and such a cluster is
  // beyond every unpadded cutoff, so compute_3B/compute_4B reject it and it
  // contributes exactly zero either way.

  const double maxcut_3b_padded = maxcut_3b + neighbor->skin;
  const double maxcut_4b_padded = maxcut_4b + neighbor->skin;

  ctx.cutsq_3b = maxcut_3b_padded * maxcut_3b_padded;
  ctx.cutsq_4b = maxcut_4b_padded * maxcut_4b_padded;
  ctx.cutsq_max = MAX(ctx.cutsq_3b, ctx.cutsq_4b);

  ctx.do_3b = (chimes_calculator->poly_orders[1] > 0);
  ctx.do_4b = (chimes_calculator->poly_orders[2] > 0);

  return ctx;
}

/* ----------------------------------------------------------------------
   Enumerate the clusters owned by one atom.
------------------------------------------------------------------------- */

void PairCHIMES::mb_clusters_for_atom(int i, const MBContext &ctx, MBScratch &s,
                                      std::vector<int> &out3, std::vector<int> &out4) const
{
  double **x = ctx.x;
  tagint *tag = ctx.tag;

  const tagint itag = tag[i];
  const int *const jlist = nl_firstneigh[i];
  const int jnum = nl_numneigh[i];

  const double xi = x[i][0];
  const double yi = x[i][1];
  const double zi = x[i][2];

  // One pass over i's neighbors collects every atom that can take part in a
  // cluster owned by i, together with its distance to i.  The old code
  // rescanned firstneigh[i] from the start for k and again for l, so the ik
  // distance was recomputed once per j and the il distance once per (j,k)
  // pair.  Filtering preserves the neighbor list order, so iterating these
  // candidate arrays visits the same clusters in the same sequence.

  s.cand_3b.resize(0);
  s.cand_4b.resize(0);

  for (int jj = 0; jj < jnum; jj++) {
    const int j = jlist[jj] & NEIGHMASK;

    if (j == i) continue;

    const tagint jtag = tag[j];

    if (jtag < itag) continue;

    const double dx = x[j][0] - xi;
    const double dy = x[j][1] - yi;
    const double dz = x[j][2] - zi;
    const double rsq = dx * dx + dy * dy + dz * dz;

    if (rsq >= ctx.cutsq_max) continue;

    s.cand_3b.push_back({j, jtag, rsq});

    if (ctx.do_4b && (rsq < ctx.cutsq_4b)) s.cand_4b.push_back({j, jtag, rsq});
  }

  ////////////////////////////////////////
  // 3-body clusters
  ////////////////////////////////////////

  if (ctx.do_3b) {
    const int n3 = s.cand_3b.size();

    for (int jj = 0; jj < n3; jj++) {
      if (s.cand_3b[jj].rsq >= ctx.cutsq_3b) continue;

      const int j = s.cand_3b[jj].idx;
      const tagint jtag = s.cand_3b[jj].tag;

      for (int kk = 0; kk < n3; kk++) {
        const int k = s.cand_3b[kk].idx;

        if (k == j) continue;

        if (s.cand_3b[kk].tag < jtag) continue;
        if (s.cand_3b[kk].rsq >= ctx.cutsq_3b) continue;

        if (dist_sq(x, j, k) >= ctx.cutsq_3b) continue;

        out3.push_back(i);
        out3.push_back(j);
        out3.push_back(k);
      }
    }
  }

  ////////////////////////////////////////
  // 4-body clusters
  ////////////////////////////////////////

  if (!ctx.do_4b) return;

  const int n4 = s.cand_4b.size();

  if (n4 < 3) return;

  // Reduce the candidates to two bitmaps in one pass over the n4*(n4-1)/2
  // candidate pairs.  Everything the cluster loops below ask about a pair is
  // a yes/no question -- is it inside the cutoff, does it order the right way
  // -- so a bit answers it, and sixty-four candidates are answered by one
  // machine word.  That turns the innermost loop from a scan over every
  // candidate into an intersection of three words: the l that can join (j,k)
  // are exactly those adjacent to both and not ordering before k.
  //
  // Tags are not unique among candidates, because a periodic image carries
  // the tag of the atom it images, so cand_ge is built from the same "does
  // not order before" test the scan used rather than from a strict rank.

  s.cand_words = (n4 + 63) / 64;

  s.cand_adj.assign((size_t) n4 * s.cand_words, 0);
  s.cand_ge.assign((size_t) n4 * s.cand_words, 0);

  for (int a = 0; a < n4; a++) {
    uint64_t *const adj_a = &s.cand_adj[(size_t) a * s.cand_words];
    uint64_t *const ge_a = &s.cand_ge[(size_t) a * s.cand_words];

    for (int b = a + 1; b < n4; b++) {
      uint64_t *const ge_b = &s.cand_ge[(size_t) b * s.cand_words];

      if (dist_sq(x, s.cand_4b[a].idx, s.cand_4b[b].idx) < ctx.cutsq_4b) {
        adj_a[b >> 6] |= (uint64_t) 1 << (b & 63);
        s.cand_adj[(size_t) b * s.cand_words + (a >> 6)] |= (uint64_t) 1 << (a & 63);
      }

      // Both tests, not one and its negation: two candidates can carry the
      // same tag when they are periodic images of one atom, and the scan
      // this replaces accepted that pair in either order.

      if (s.cand_4b[b].tag >= s.cand_4b[a].tag) ge_a[b >> 6] |= (uint64_t) 1 << (b & 63);

      if (s.cand_4b[a].tag >= s.cand_4b[b].tag) ge_b[a >> 6] |= (uint64_t) 1 << (a & 63);
    }
  }

  for (int jj = 0; jj < n4; jj++) {
    const int j = s.cand_4b[jj].idx;

    const uint64_t *const adj_j = &s.cand_adj[(size_t) jj * s.cand_words];
    const uint64_t *const ge_j = &s.cand_ge[(size_t) jj * s.cand_words];

    for (int wk = 0; wk < s.cand_words; wk++) {
      uint64_t mk = adj_j[wk] & ge_j[wk];

      while (mk) {
        const int kk = (wk << 6) + lowest_bit(mk);
        mk &= mk - 1;

        const int k = s.cand_4b[kk].idx;

        const uint64_t *const adj_k = &s.cand_adj[(size_t) kk * s.cand_words];
        const uint64_t *const ge_k = &s.cand_ge[(size_t) kk * s.cand_words];

        for (int wl = 0; wl < s.cand_words; wl++) {
          uint64_t ml = adj_j[wl] & adj_k[wl] & ge_k[wl];

          while (ml) {
            const int ll = (wl << 6) + lowest_bit(ml);
            ml &= ml - 1;

            out4.push_back(i);
            out4.push_back(j);
            out4.push_back(k);
            out4.push_back(s.cand_4b[ll].idx);
          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairCHIMES::build_mb_neighlists()
{

  if ((chimes_calculator->poly_orders[1] == 0) && (chimes_calculator->poly_orders[2] == 0)) return;

  // List gets built based on atoms owned by calling proc.
  // resize(0) rather than clear() so the capacity survives to the next rebuild.

  neighborlist_3mers.resize(0);
  neighborlist_4mers.resize(0);

  setup_neighlist_ptrs();

  const MBContext ctx = mb_context();

  for (int ii = 0; ii < nl_inum; ii++)    // Loop over real atoms (ai)
    mb_clusters_for_atom(nl_ilist[ii], ctx, mb_scratch, neighborlist_3mers, neighborlist_4mers);

  n_3mers = neighborlist_3mers.size() / 3;
  n_4mers = neighborlist_4mers.size() / 4;

  if (ctx.do_3b) sort_mers_by_type<3>(neighborlist_3mers, n_3mers, mer_type_3b);

  if (ctx.do_4b) sort_mers_by_type<4>(neighborlist_4mers, n_4mers, mer_type_4b);
}

/* ----------------------------------------------------------------------
   Group the triplets by cluster type so that the force loop can hand a run of
   same-typed clusters to the batched evaluator.  A counting sort over the
   packed type index, which is small (ntypes^3), keeps this linear; it runs once
   per neighbor rebuild rather than once per step.
------------------------------------------------------------------------- */

template <int WIDTH>
void PairCHIMES::sort_mers_by_type(std::vector<int> &mers, int nmers, std::vector<int> &mer_type)
{
  if (nmers == 0) return;

  const int nt = chimes_calculator->natmtyps;
  int nkey = 1;

  for (int w = 0; w < WIDTH; w++) nkey *= nt;

  int *type = atom->type;

  mer_key.resize(nmers);
  type_count.assign(nkey + 1, 0);

  for (int c = 0; c < nmers; c++) {
    const int *m = &mers[(size_t) WIDTH * c];
    int key = 0;

    for (int w = 0; w < WIDTH; w++) key = key * nt + chimes_type[type[m[w]] - 1];

    mer_key[c] = key;
    type_count[key + 1]++;
  }

  for (int t = 0; t < nkey; t++) type_count[t + 1] += type_count[t];

  mer_scratch.resize((size_t) nmers * WIDTH);
  mer_type.resize(nmers);

  // A counting sort is stable, so clusters that share a type keep the order the
  // build gave them.  That matters: the build emits the last atom fastest, so
  // clusters that are adjacent already share most of their pairs, and keeping
  // them adjacent keeps the distances a batch works on close together in
  // memory.

  for (int c = 0; c < nmers; c++) {
    const int dst = type_count[mer_key[c]]++;

    mer_type[dst] = mer_key[c];

    for (int w = 0; w < WIDTH; w++)
      mer_scratch[(size_t) WIDTH * dst + w] = mers[(size_t) WIDTH * c + w];
  }

  mers.swap(mer_scratch);
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

  setup_neighlist_ptrs();

  inum = nl_inum;                // length of the list
  ilist = nl_ilist;              // list of i atoms for which neighbor list exists
  numneigh = nl_numneigh;        // length of each of the ilist neighbor lists
  firstneigh = nl_firstneigh;    // point to the list of neighbors of i

  chimes2BTmp chimes_2btmp(chimes_calculator->poly_orders[0]);
  chimes3BTmp chimes_3btmp(chimes_calculator->poly_orders[1]);
  chimes3BBatch chimes_3bbatch(chimes_calculator->poly_orders[1]);
  chimes4BBatch chimes_4bbatch(chimes_calculator->poly_orders[2]);

  // Build the ChIMES many-body neighbor lists.. only do so when LAMMPS neighborlist has been updated

  if (neighbor->ago == 0) build_mb_neighlists();


  // Per-key staging for the batched 2-body path.

  const int nchem = chimes_calculator->natmtyps;

  b2_cnt.assign(nchem * nchem, 0);
  b2_i.resize(nchem * nchem);
  b2_j.resize(nchem * nchem);
  b2_dist.resize(nchem * nchem);
  b2_dr.resize(nchem * nchem);

  auto flush_2b = [&](const int key2) {
    const int nb = b2_cnt[key2];
    double bd[CHIMES_VLEN], be[CHIMES_VLEN], bfs[CHIMES_VLEN];

    for (int l = 0; l < nb; l++) bd[l] = b2_dist[key2][l];

    for (int l = nb; l < CHIMES_VLEN; l++) bd[l] = bd[0];

    chimes_calculator->compute_2B_batch(key2, bd, be, bfs);

    for (int l = 0; l < nb; l++) {
      const int ai = b2_i[key2][l], aj = b2_j[key2][l];
      const double *const pdr = b2_dr[key2][l].data();

      for (int idx2 = 0; idx2 < CHDIM; idx2++) {
        const double fc = bfs[l] * pdr[idx2];

        f[ai][idx2] += fc;
        f[aj][idx2] -= fc;
      }

      if (evflag) {
        int alist[2] = {ai, aj};
        double st[6];

        st[0] = -bfs[l] * pdr[0] * pdr[0];
        st[1] = -bfs[l] * pdr[0] * pdr[1];
        st[2] = -bfs[l] * pdr[0] * pdr[2];
        st[3] = -bfs[l] * pdr[1] * pdr[1];
        st[4] = -bfs[l] * pdr[1] * pdr[2];
        st[5] = -bfs[l] * pdr[2] * pdr[2];

        ev_tally_mb(2, alist, be[l], st);
      }
    }

    b2_cnt[key2] = 0;
  };

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

    // Now move on to two-body force, stress, and energy.  Pairs that sit in
    // the plain middle of the potential -- outside the outer cutoff test,
    // clear of the inner cutoff and the penalty region, with a monomial form
    // of their series available -- are collected into per-key batches of
    // CHIMES_VLEN and evaluated together: one vector exponential and one
    // broadcast-coefficient Horner descent instead of eight scalar ones.
    // Everything else takes the scalar path unchanged.

    for (jj = 0; jj < jnum; jj++)    // Loop over neighbors of i
    {
      j = jlist[jj] & NEIGHMASK;    // Index of the jj atom, extra bits stripped
      jtag = tag[j];                // Get j's global atom index (sort of like its "parent")

      if (jtag <=
          itag)    // only allow calculation for j<i, since we've requested a full neighbor list
        continue;

      // Get distance using ghost atoms... don't need MIC since we're using ghost atoms

      dist = get_dist(i, j, &dr[0]);

      const int jchem = chimes_type[type[j] - 1];
      const int key2 = typ_idxs_2b[0] * nchem + jchem;

      if (chimes_calculator->fast_2b(key2, dist)) {
        const int nb = b2_cnt[key2];

        b2_i[key2][nb] = i;
        b2_j[key2][nb] = j;
        b2_dist[key2][nb] = dist;

        for (idx = 0; idx < CHDIM; idx++) b2_dr[key2][nb][idx] = dr[idx];

        if (++b2_cnt[key2] == CHIMES_VLEN) flush_2b(key2);

        continue;
      }

      typ_idxs_2b[1] = jchem;

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

  for (int key2 = 0; key2 < nchem * nchem; key2++)
    if (b2_cnt[key2]) flush_2b(key2);

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

        const int cand_type = mer_type_3b[ii];

        sc3 = chimes_calculator->slots_3B_idx(cand_type);

        if (sc3) {
          if (within(x, i, j, sc3[0].outer_sq, &dr_3b[0 * CHDIM], dist_3b[0]) &&
              within(x, i, k, sc3[1].outer_sq, &dr_3b[1 * CHDIM], dist_3b[1]) &&
              within(x, j, k, sc3[2].outer_sq, &dr_3b[2 * CHDIM], dist_3b[2]))
            this_type = cand_type;
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
                   chimes_3bbatch.fcutderiv[0][l] * fc1 * fc2 * poly) *
              chimes_3bbatch.inv_dx[0][l];
          fs[1] = (fcut_all * chimes_3bbatch.dpoly[1][l] +
                   chimes_3bbatch.fcutderiv[1][l] * fc0 * fc2 * poly) *
              chimes_3bbatch.inv_dx[1][l];
          fs[2] = (fcut_all * chimes_3bbatch.dpoly[2][l] +
                   chimes_3bbatch.fcutderiv[2][l] * fc0 * fc1 * poly) *
              chimes_3bbatch.inv_dx[2][l];

          // Pair p acts between the two atoms of the cluster it connects:
          // 0 = ij, 1 = ik, 2 = jk.

          static const int pa[3] = {0, 0, 1}, pb[3] = {1, 2, 2};

          if (vflag_either) std::fill(stensor.begin(), stensor.end(), 0.0);

          // Three pairs but only three atoms, so each is written twice.  As for
          // the quadruplets, the cluster's own contribution is summed per atom
          // before it touches the global force array.

          double fatom[3][CHDIM] = {{0.0}};

          for (int p = 0; p < 3; p++) {
            for (idx = 0; idx < CHDIM; idx++) {
              const double fpair = fs[p] * bdr[p][idx][l];

              fatom[pa[p]][idx] += fpair;
              fatom[pb[p]][idx] -= fpair;
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

          for (int a = 0; a < 3; a++) {
            const int ga = batom[a][l];

            for (idx = 0; idx < CHDIM; idx++) f[ga][idx] += fatom[a][idx];
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

    // Quadruplets are batched exactly as the triplets are, and for the same
    // reason: the list is sorted by cluster type, so a run of clusters shares
    // one coefficient set and can be evaluated lane by lane.  Sorting is what
    // makes this work -- an earlier attempt that batched the list in build
    // order averaged 2.6 of 8 lanes, because a run of quadruplets sharing
    // (i,j,k) holds barely more than one surviving cluster.

    double bdx[6][CHIMES_VLEN];
    double bdr[6][CHDIM][CHIMES_VLEN];
    int batom[4][CHIMES_VLEN];

    int nb = 0;
    int batch_type = -1;

    // Pair p acts between the two atoms of the cluster it connects.

    static const int pa[6] = {0, 0, 0, 1, 1, 2}, pb[6] = {1, 2, 3, 2, 3, 3};

    for (ii = 0; ii <= n_4mers; ii++) {
      int this_type = -1;

      if (ii < n_4mers) {
        const int *mer = &neighborlist_4mers[4 * ii];
        i = mer[0];
        j = mer[1];
        k = mer[2];
        l = mer[3];

        // As for the triplets, and it matters more here: two thirds of the
        // enumerated quadruplets are outside the real cutoffs on a given step,
        // and each one used to cost six square roots before being discarded.

        const int cand_type = mer_type_4b[ii];
        const chimesSlotConst *sc4 = chimes_calculator->slots_4B_idx(cand_type);

        if (sc4) {
          if (within(x, i, j, sc4[0].outer_sq, &dr_4b[0 * CHDIM], dist_4b[0]) &&
              within(x, i, k, sc4[1].outer_sq, &dr_4b[1 * CHDIM], dist_4b[1]) &&
              within(x, i, l, sc4[2].outer_sq, &dr_4b[2 * CHDIM], dist_4b[2]) &&
              within(x, j, k, sc4[3].outer_sq, &dr_4b[3 * CHDIM], dist_4b[3]) &&
              within(x, j, l, sc4[4].outer_sq, &dr_4b[4 * CHDIM], dist_4b[4]) &&
              within(x, k, l, sc4[5].outer_sq, &dr_4b[5 * CHDIM], dist_4b[5]))
            this_type = cand_type;
        }
      }

      const bool at_end = (ii == n_4mers);

      // A rejected cluster must not disturb the batch in hand.

      if (!at_end && (this_type < 0)) continue;

      if ((nb > 0) && (at_end || (this_type != batch_type) || (nb == CHIMES_VLEN))) {
        for (int p = 0; p < 6; p++)
          for (int lane = nb; lane < CHIMES_VLEN; lane++) bdx[p][lane] = bdx[p][0];

        chimes_calculator->compute_4B_batch(nb, batch_type, bdx, chimes_4bbatch);

        for (int lane = 0; lane < nb; lane++) {
          double fc[6], fcut_5[6];

          for (int p = 0; p < 6; p++) fc[p] = chimes_4bbatch.fcut[p][lane];

          const double fcut_all = fc[0] * fc[1] * fc[2] * fc[3] * fc[4] * fc[5];
          const double poly = chimes_4bbatch.poly[lane];

          // Products of the other five cutoffs, from one pass each way.

          double pre[6], suf[6];

          pre[0] = 1.0;
          suf[5] = 1.0;

          for (int p = 1; p < 6; p++) pre[p] = pre[p - 1] * fc[p - 1];

          for (int p = 4; p >= 0; p--) suf[p] = suf[p + 1] * fc[p + 1];

          for (int p = 0; p < 6; p++) fcut_5[p] = pre[p] * suf[p];

          if (vflag_either) std::fill(stensor.begin(), stensor.end(), 0.0);

          // Six pairs but only four atoms, so every atom is written three
          // times.  Summing per atom first turns thirty-six read-modify-writes
          // scattered through the global force array into twelve.

          double fatom[4][CHDIM] = {{0.0}};

          for (int p = 0; p < 6; p++) {
            const double fs = (fcut_all * chimes_4bbatch.dpoly[p][lane] +
                               chimes_4bbatch.fcutderiv[p][lane] * fcut_5[p] * poly) *
                chimes_4bbatch.inv_dx[p][lane];

            for (idx = 0; idx < CHDIM; idx++) {
              const double fpair = fs * bdr[p][idx][lane];

              fatom[pa[p]][idx] += fpair;
              fatom[pb[p]][idx] -= fpair;
            }

            if (vflag_either) {
              stensor[0] -= fs * bdr[p][0][lane] * bdr[p][0][lane];
              stensor[1] -= fs * bdr[p][0][lane] * bdr[p][1][lane];
              stensor[2] -= fs * bdr[p][0][lane] * bdr[p][2][lane];
              stensor[3] -= fs * bdr[p][1][lane] * bdr[p][1][lane];
              stensor[4] -= fs * bdr[p][1][lane] * bdr[p][2][lane];
              stensor[5] -= fs * bdr[p][2][lane] * bdr[p][2][lane];
            }
          }

          for (int a = 0; a < 4; a++) {
            const int ga = batom[a][lane];

            for (idx = 0; idx < CHDIM; idx++) f[ga][idx] += fatom[a][idx];
          }

          if (evflag) {
            for (int a = 0; a < 4; a++) atmlist[a] = batom[a][lane];

            ev_tally_mb(4, atmlist, poly * fcut_all, stensor.data());
          }
        }

        nb = 0;
      }

      if (at_end) break;

      batch_type = this_type;
      batom[0][nb] = i;
      batom[1][nb] = j;
      batom[2][nb] = k;
      batom[3][nb] = l;

      for (int p = 0; p < 6; p++) {
        bdx[p][nb] = dist_4b[p];

        for (idx = 0; idx < CHDIM; idx++) bdr[p][idx][nb] = dr_4b[p * CHDIM + idx];
      }

      nb++;
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
