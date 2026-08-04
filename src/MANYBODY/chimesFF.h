/*
    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
    Contributing Author:  Rebecca K. Lindsey (2020)
*/

#ifndef _chimesFF_h
#define _chimesFF_h

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define CHIMES_PI 3.14159265359

using namespace std;

// Notes:
//
// 1. A Morse-style coordinate transformation is hard-coded (see set_cheby_polys)
// 2. Polynomials are hard-coded over the domain [-1,1]
// 3. A cubic style cutoff is assumed, and Tersoff is the only other style considered (see get_fcut)

#define CHDIM 3                  // The number of spatial dimensions.

// Temporary storage for ChIMES interaction.
class chimes2BTmp {
 public:
  inline chimes2BTmp(int poly_order);
  inline void resize(int poly_order);
  vector<double> Tn;
  vector<double> Tnd;
};

inline chimes2BTmp::chimes2BTmp(int poly_order) : Tn(poly_order + 1), Tnd(poly_order + 1)
{
  ;
}

inline void chimes2BTmp::resize(int poly_order)
{

  if (Tn.size() < poly_order + 1) Tn.resize(poly_order + 1);

  if (Tnd.size() < poly_order + 1) Tnd.resize(poly_order + 1);
}

// Records what the last cluster left in a pair slot, so that a slot whose pair
// is unchanged can reuse it.  The many-body lists are generated with the last
// atom varying fastest, so consecutive clusters share most of their pairs: all
// of ij, ik and jk are constant while l runs, and ij is constant while k runs.
// Keying on the slot's constant record together with the distance means a hit
// guarantees an identical result -- same cutoffs, same Morse lambda, same
// separation -- without the caller having to say anything about atom indices.

struct chimesPairCache {
  double morse, inner, outer;    // the slot constants the cached setup was built from
  double dx;
  double fcut, fcutderiv;
};

class chimes3BTmp {
 public:
  inline chimes3BTmp(int poly_order);
  inline void resize(int poly_order);

  vector<double> Tn_ij, Tn_ik, Tn_jk;       // The Chebyshev polymonials
  vector<double> Tnd_ij, Tnd_ik, Tnd_jk;    // The Chebyshev polymonial derivatives
  chimesPairCache cache[3];
};

inline chimes3BTmp::chimes3BTmp(int poly_order) :
    Tn_ij(poly_order + 1), Tn_ik(poly_order + 1), Tn_jk(poly_order + 1), Tnd_ij(poly_order + 1),
    Tnd_ik(poly_order + 1), Tnd_jk(poly_order + 1)
{
  for (int p = 0; p < 3; p++) cache[p] = chimesPairCache{-1.0, -1.0, -1.0, -1.0, 0.0, 0.0};
}

class chimes4BTmp {
 public:
  inline chimes4BTmp(int poly_order);
  inline void resize(int poly_order);

  vector<double> Tn_ij, Tn_ik, Tn_il, Tn_jk, Tn_jl, Tn_kl;    // The Chebyshev polymonials
  vector<double> Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
      Tnd_kl;    // The Chebyshev polymonial derivatives
  chimesPairCache cache[6];
};

inline chimes4BTmp::chimes4BTmp(int poly_order) :
    Tn_ij(poly_order + 1), Tn_ik(poly_order + 1), Tn_il(poly_order + 1), Tn_jk(poly_order + 1),
    Tn_jl(poly_order + 1), Tn_kl(poly_order + 1), Tnd_ij(poly_order + 1), Tnd_ik(poly_order + 1),
    Tnd_il(poly_order + 1), Tnd_jk(poly_order + 1), Tnd_jl(poly_order + 1), Tnd_kl(poly_order + 1)
{
  for (int p = 0; p < 6; p++) cache[p] = chimesPairCache{-1.0, -1.0, -1.0, -1.0, 0.0, 0.0};
}

inline void chimes3BTmp::resize(int poly_order)
{

  if (Tn_ij.size() < poly_order + 1) Tn_ij.resize(poly_order + 1);

  if (Tnd_ij.size() < poly_order + 1) Tnd_ij.resize(poly_order + 1);

  if (Tn_ik.size() < poly_order + 1) Tn_ik.resize(poly_order + 1);

  if (Tnd_ik.size() < poly_order + 1) Tnd_ik.resize(poly_order + 1);

  if (Tn_jk.size() < poly_order + 1) Tn_jk.resize(poly_order + 1);

  if (Tnd_jk.size() < poly_order + 1) Tnd_jk.resize(poly_order + 1);
}

enum class fcutType {
  CUBIC,
  TERSOFF,
};

// Everything about one pair slot of one cluster type that does not depend on
// the configuration.  The Morse transform bounds and the cutoff-function
// constants were previously re-derived on every single interaction, which cost
// two of the three exp() calls in set_cheby_polys plus a handful of divisions.
// They are functions of the atom types alone, so they are tabulated at setup.

struct chimesSlotConst {
  double morse;         // Morse lambda of the pair type occupying this slot
  double inner, outer;  // inner and outer cutoff of this slot
  double outer_sq;      // outer*outer, so callers can reject before taking a sqrt
  double x_avg;         // Morse transform: x = (exp(-r/morse) - x_avg) / x_diff
  double x_diff;
  double fcut_thresh;   // TERSOFF: distance at which the cutoff function starts
  double fcut_denom;    // TERSOFF: outer - fcut_thresh
  double fcut_dscale;   // TERSOFF: pi/fcut_denom.  CUBIC: -3/outer
};

// The polynomial coefficients of one cluster type, in a form the inner loop
// can stream.  powers is contiguous with stride npairs and already permuted
// into runtime pair order, so evaluating a coefficient no longer costs a walk
// through vector<vector<int>> plus a mapped_pair_idx indirection per pair.

struct chimesPolySet {
  int ncoeffs;
  const double *params;    // [ncoeffs]
  const int *powers;       // [ncoeffs*npairs], runtime pair order
  const struct chimesGroupedPoly *grouped;    // nullptr if not built
};

// The same coefficients arranged as a tree over the leading npairs-1 power
// indices, so that the Chebyshev factors shared by a whole subtree are applied
// once instead of once per coefficient.  Each node at level d carries its power
// value, and level_start[d] gives its children's range in level d+1 -- or, at
// the last level, in the leaf arrays.  Sibling subtrees are contiguous, so the
// evaluator walks all three arrays sequentially.
//
// At the leaf only two sums are accumulated; every level up multiplies those by
// its own Chebyshev value and carries one more accumulator, so the whole
// derivative set falls out of a single pass.

struct chimesGroupedPoly {
  int nlevels;                 // npairs - 1
  vector<int> level_pow[5];    // [nlevels][nodes at that level]
  vector<int> level_start[5];  // [nlevels][nodes + 1]
  vector<int> leaf_pow;
  vector<double> leaf_c;
};

class chimesFF {
 public:
  ////////////////////////
  // General parameters
  ////////////////////////

  int rank;        // Used to prevent multiple cout statements when accessed from MPI
  int natmtyps;    // How many atom types are defined for this force field?

  vector<int>
      poly_orders;    // [bodiedness-1]; i.e. 12 = 2-body only, 12th order; 12 5 = 2+3-body, 0 5 = 3-body only, 5th order
  vector<string> atmtyps;    // Atom types
  vector<double> masses;     // Atom masses

  ////////////////////////
  // Functions
  ////////////////////////

  chimesFF();
  ~chimesFF();

  void init(int mpi_rank);

  virtual void read_parameters(string paramfile);

  void compute_1B(const int typ_idx, double &energy);

  // vflag selects whether the stress tensor is accumulated.  LAMMPS asks for
  // the virial only on the steps a pressure or per-atom stress is needed, and
  // on every other step the stress arithmetic -- twelve operations per pair
  // slot, so 36 per 3-mer and 72 per 4-mer -- is discarded by the caller.

  void compute_2B(const double dx, const vector<double> &dr, const vector<int> &typ_idxs,
                  vector<double> &force, vector<double> &stress, double &energy, chimes2BTmp &tmp,
                  const bool vflag = true);

  void compute_3B(const vector<double> &dx, const vector<double> &dr, const vector<int> &typ_idxs,
                  vector<double> &force, vector<double> &stress, double &energy, chimes3BTmp &tmp,
                  const bool vflag = true);

  void compute_4B(const vector<double> &dx, const vector<double> &dr, const vector<int> &typ_idxs,
                  vector<double> &force, vector<double> &stress, double &energy, chimes4BTmp &tmp,
                  const bool vflag = true);

  void get_cutoff_2B(vector<vector<double>> &cutoff_2b);    // Populates the 2b cutoffs

  double max_cutoff_2B(bool silent = false);    // Returns the largest 2B cutoff
  double max_cutoff_3B(bool silent = false);    // Returns the largest 3B cutoff
  double max_cutoff_4B(bool silent = false);    // Returns the largest 4B cutoff

  void set_atomtypes(vector<string> &type_list);

  // The per-slot constants for a cluster of the given atom types, or nullptr if
  // the force field excludes that combination.  Callers use these to reject a
  // cluster on squared distances, before paying for any sqrt: on a typical
  // model most enumerated clusters are outside the real cutoffs, because the
  // lists are built with the neighbor skin added.

  inline const chimesSlotConst *slots_3B(const int t0, const int t1, const int t2) const
  {
    const int idx = (t0 * natmtyps + t1) * natmtyps + t2;

    if (atom_int_trip_map[idx] < 0) return nullptr;

    return &slot_3b[idx * 3];
  }

  inline const chimesSlotConst *slots_4B(const int t0, const int t1, const int t2,
                                         const int t3) const
  {
    const int idx = ((t0 * natmtyps + t1) * natmtyps + t2) * natmtyps + t3;

    if (atom_int_quad_map[idx] < 0) return nullptr;

    return &slot_4b[idx * 6];
  }

  int get_atom_pair_index(int pair_id);
  virtual void build_pair_int_trip_map();
  virtual void build_pair_int_quad_map();

  // Tabulates the configuration-independent per-slot constants.  Must be called
  // after read_parameters() and the two build_pair_int_*_map() calls.
  void build_interaction_tables();

 protected:
  void fill_slot(chimesSlotConst &sc, int pair_idx, double inner, double outer);
  int permuted_powers(map<pair<int, vector<int>>, int> &pool_slot, int cluster_idx, int npairs,
                      const vector<int> &map, const vector<vector<int>> &powers, int ncoeffs);
  int build_grouped(int npairs, const vector<int> &flatpow, const double *params, int ncoeffs);

 public:

 protected:
  bool dense_coeffs;

  string xform_style;               //  Morse, direct, inverse, etc...
  fcutType fcut_type;               // cutoff function style (tersoff/cubic)
  double fcut_var;                  // tersoff distance (if fcut_type)
  double inner_smooth_distance;     // Used in smoothing the cutoff interaction.
  vector<double> morse_var;         // [npairs]; morse_lambda
  vector<double> penalty_params;    // [2];  Second dimension: [0] = A_pen, [1] = d_pen
  vector<double> energy_offsets;    // [natmtyps]; Single atom ChIMES energies

  // Names (chemical symbols for constituent atoms) .. handled differently for 2-body versus >2-body interactions

  vector<string> pair_params_atm_chem_1;    //[npairs]; // first atom in pair
  vector<string> pair_params_atm_chem_2;    //[npairs]; // second atom in pair

  vector<vector<string>>
      trip_params_atm_chems;    //[ntrips][3]    // Gives chemical symbol  for each ATOM in the triplet (i.e. "Si")
  vector<vector<string>>
      trip_params_pair_typs;    //[ntrips][3]    // Gives chemical symbols for each PAIR in the triplet (i.e. "SiO")

  vector<vector<string>>
      quad_params_atm_chems;    //[quads][3]    // Gives chemical symbol  for each ATOM in the quadruplet (i.e. "Si")
  vector<vector<string>>
      quad_params_pair_typs;    //[quads][3]    // Gives chemical symbols for each PAIR in the quadruplet (i.e. "SiO")

  int n_pair_maps;    // Number of pair maps entries
  int n_trip_maps;    // Number of trip maps entries
  int n_quad_maps;    // Number of quad maps entries

  int pair_type_idx;
  int trip_type_idx;
  int quad_type_idx;

  ////////////////////////
  // Definitions for pair, triplet, and quadruplet types
  ////////////////////////

  // 2-body maps

  vector<string>
      atom_typ_pair_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiO")
  vector<int>
      atom_idx_pair_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 5)
  vector<int> atom_int_pair_map;    // [nmaps] "fast" maps, based on atom type index
  vector<string>
      atom_int_prpr_map;    // [nmaps] "fast" maps, based on atom type index ... returns the "proper" pair type instead of an index

  // 3-body maps

  vector<string>
      atom_typ_trip_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiOSiOOO")
  vector<int>
      atom_idx_trip_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 3)
  vector<int>
      atom_int_trip_map;    // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of three atoms of arbitrary order
  vector<vector<int>>
      pair_int_trip_map;    // Gives the atom pair indices for an arbitrary triplet of atom types.

  // 4-body maps

  vector<string>
      atom_typ_quad_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiOSiOOO")
  vector<int>
      atom_idx_quad_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 3)
  vector<int>
      atom_int_quad_map;    // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of four atoms of arbitrary order
  vector<vector<int>>
      pair_int_quad_map;    // Gives the atom pair indices for an arbitrary quad of atom types.

  ////////////////////////
  // Polynomial parameters
  ////////////////////////

  // number of coefficients for the pair/triplet/quadruplet type

  vector<int> ncoeffs_2b;    // [npairs]

  vector<vector<int>> chimes_2b_pows;    // [npairs][npowers] power for the coresponding parameter
  vector<vector<double>> chimes_2b_params;    // [npairs][npowers] 2-body polynomial coefficients
  vector<vector<double>> chimes_2b_cutoff;    // [npairs][2] inner and outer cutoff for pair

  vector<int> ncoeffs_3b;                          // [ntrips]
  vector<vector<vector<int>>> chimes_3b_powers;    // [ntrips][nparams][constit. pair]
  vector<vector<double>> chimes_3b_params;         // [ntrips][nparams]
  vector<vector<vector<double>>>
      chimes_3b_cutoff;    // [ntrips][2][constit. pair] inner and outer cutoff for pair 1

  vector<int> ncoeffs_4b;                          // [nquads]
  vector<vector<vector<int>>> chimes_4b_powers;    // [nquads][nparams][constit. pair]
  vector<vector<double>> chimes_4b_params;         // [nquads][nparams]
  vector<vector<vector<double>>>
      chimes_4b_cutoff;    // [nquads][2][constit. pair] inner and outer cutoff for pair 1

  // Per-slot constants, indexed by the packed atom-type index of the cluster:
  //   2-body: [t0*natmtyps + t1]
  //   3-body: [type_idx*3 + pair],  pair order (ij, ik, jk)
  //   4-body: [type_idx*6 + pair],  pair order (ij, ik, il, jk, jl, kl)
  // Entries for excluded cluster types are left zeroed and never read.

  vector<chimesSlotConst> slot_2b;
  vector<chimesSlotConst> slot_3b;
  vector<chimesSlotConst> slot_4b;

  // Pre-permuted coefficient sets, indexed by the packed atom-type index.
  // Many type indices share a (parameter set, permutation) pair, so the power
  // blocks themselves live in a pool and the sets point into it.

  vector<chimesPolySet> poly_3b_set;    // [natmtyps^3]
  vector<chimesPolySet> poly_4b_set;    // [natmtyps^4]
  vector<vector<int>> powers_pool;
  vector<chimesGroupedPoly> grouped_pool;

  // Tools for compute functions

  inline void set_cheby_polys(vector<double> &Tn, vector<double> &Tnd, double dx,
                              const chimesSlotConst &sc, const int bodiedness_idx);

  void poly_2B(double *e, double *f0, int ncoeffs_2b, vector<double> &chimes_2b_params,
               vector<int> &chimes_2b_pows, vector<double> &Tn, vector<double> &Tnd);

  void poly_3B(double *e, double *f, const chimesPolySet &ps, vector<double> &Tn_ij,
               vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
               vector<double> &Tnd_ik, vector<double> &Tnd_jk);

  void poly_3B_grouped(double *e, double *f, const chimesGroupedPoly &g, vector<double> &Tn_ij,
                       vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
                       vector<double> &Tnd_ik, vector<double> &Tnd_jk);

  void poly_4B_grouped(double *e, double *f, const chimesGroupedPoly &g, vector<double> &Tn_ij,
                       vector<double> &Tn_ik, vector<double> &Tn_il, vector<double> &Tn_jk,
                       vector<double> &Tn_jl, vector<double> &Tn_kl, vector<double> &Tnd_ij,
                       vector<double> &Tnd_ik, vector<double> &Tnd_il, vector<double> &Tnd_jk,
                       vector<double> &Tnd_jl, vector<double> &Tnd_kl);

  // Evaluates the 3-Body chebyshev polynomial in dense format.
  void poly_3B_dense(double &e, double &f0, double &f1, double &f2, int ncoeffs_3b,
                     vector<double> &params_3b, vector<double> &Tn_ij, vector<double> &Tn_ik,
                     vector<double> &Tn_jk, vector<double> &Tnd_ij, vector<double> &Tnd_ik,
                     vector<double> &Tnd_jk);

  // Loop evaluators for poly_3B_dense.
  void poly_3B_dense_loop1(int max_poly, double &e, double &f0, double &f1, double &f2,
                           int ncoeffs_3b, vector<double> &chimes_3b_params, vector<double> &Tn_ij,
                           vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
                           vector<double> &Tnd_ik, vector<double> &Tnd_jk);

  void poly_3B_dense_loop2(int max_poly, double &e, double &f0, double &f1, double &f2,
                           int ncoeffs_3b, vector<double> &chimes_3b_params, vector<double> &Tn_ij,
                           vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
                           vector<double> &Tnd_ik, vector<double> &Tnd_jk);

  void poly_3B_dense_loop3(int max_poly, double &e, double &f0, double &f1, double &f2,
                           int ncoeffs_3b, vector<double> &chimes_3b_params, vector<double> &Tn_ij,
                           vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
                           vector<double> &Tnd_ik, vector<double> &Tnd_jk);

  // Transforms 3-body input ChIMES model into a "dense" format where all
  // possible coefficients are allocated.
  void densify_3B(int &ncoeffs3, vector<vector<int>> &powers_3b, vector<double> &params_3b);

  // Transforms 4-body input ChIMES model into a "dense" format where all
  // possible coefficients are allocated.
  void densify_4B(int &ncoeffs4, vector<vector<int>> &powers_4b, vector<double> &params_4b);

  void poly_4B(double *e, double *f, const chimesPolySet &ps, vector<double> &Tn_ij,
               vector<double> &Tn_ik, vector<double> &Tn_il, vector<double> &Tn_jk,
               vector<double> &Tn_jl, vector<double> &Tn_kl, vector<double> &Tnd_ij,
               vector<double> &Tnd_ik, vector<double> &Tnd_il, vector<double> &Tnd_jk,
               vector<double> &Tnd_jl, vector<double> &Tnd_kl);

  // Loop1 uses a flat loop to evaluate a dense 4-body polynomial.
  void poly_4B_dense_loop1(int max_poly, double &e, double &f0, double &f1, double &f2, double &f3,
                           double &f4, double &f5, int ncoeffs_4b, vector<double> &params_4b,
                           vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_il,
                           vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
                           vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_il,
                           vector<double> &Tnd_jk, vector<double> &Tnd_jl, vector<double> &Tnd_kl);

  // Innver evaluation loop for dense 4 body poly.  2nd. variant.
  // loop2 is a templated loop.
  void poly_4B_dense_loop2(int max_poly, double &e, double &f0, double &f1, double &f2, double &f3,
                           double &f4, double &f5, int ncoeffs_4b, vector<double> &params_4b,
                           vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_il,
                           vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
                           vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_il,
                           vector<double> &Tnd_jk, vector<double> &Tnd_jl, vector<double> &Tnd_kl);

  // Loop3 is a templated nested loop
  void poly_4B_dense_loop3(int max_poly, double &e, double &f0, double &f1, double &f2, double &f3,
                           double &f4, double &f5, int ncoeffs_4b, vector<double> &params_4b,
                           vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_il,
                           vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
                           vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_il,
                           vector<double> &Tnd_jk, vector<double> &Tnd_jl, vector<double> &Tnd_kl);

  // Evaluates the 4-body Chebyshev polynomial in dense format.
  void poly_4B_dense(double &e, double &f0, double &f1, double &f2, double &f3, double &f4,
                     double &f5, int ncoeffs_4b, vector<double> &params_4b, vector<double> &Tn_ij,
                     vector<double> &Tn_ik, vector<double> &Tn_il, vector<double> &Tn_jk,
                     vector<double> &Tn_jl, vector<double> &Tn_kl, vector<double> &Tnd_ij,
                     vector<double> &Tnd_ik, vector<double> &Tnd_il, vector<double> &Tnd_jk,
                     vector<double> &Tnd_jl, vector<double> &Tnd_kl);

  void set_polys_out_of_range(vector<double> &Tn, vector<double> &Tnd, double dx, double x,
                              int poly_order, double inner_cutoff, double exprlen, double dx_dr);

  inline void get_fcut(const double dx, const chimesSlotConst &sc, double &fcut, double &fcutderiv);

  // True when this slot already holds the setup for exactly this pair, so the
  // Chebyshev arrays and the cutoff function can be left alone.  Otherwise the
  // cache is claimed for the new pair and the caller refills it.
  //
  // The comparison is against the slot's values rather than its address on
  // purpose: the slot record is selected by the cluster's full atom-type index,
  // so consecutive clusters that genuinely share a pair still land on different
  // records as soon as one of the *other* atoms changes type.  Matching on
  // (Morse lambda, inner, outer) instead recognizes those, which for a model
  // whose cutoffs do not vary by cluster type is nearly every one of them.
  // Everything else in the record is derived from these three.

  static inline bool pair_cached(chimesPairCache &c, const chimesSlotConst &sc, const double dx)
  {
    if ((c.dx == dx) && (c.morse == sc.morse) && (c.inner == sc.inner) && (c.outer == sc.outer))
      return true;

    c.dx = dx;
    c.morse = sc.morse;
    c.inner = sc.inner;
    c.outer = sc.outer;

    return false;
  }

  inline void get_penalty(const double dx, const int &pair_idx, const double inner_cutoff,
                          double &E_penalty, double &force_scalar);

  inline void build_atom_and_pair_mappers(const int natoms, const int npairs,
                                          const vector<int> &typ_idxs,
                                          const vector<string> &clu_params_atm_chems,
                                          vector<int> &mapped_pair_idx);

  inline void build_atom_and_pair_mappers(const int natoms, const int npairs,
                                          const vector<int> &typ_idxs,
                                          const vector<string> &clu_params_atm_chems,
                                          int *mapped_pair_idx);

  int get_proper_pair(string ty1, string ty2);

  double max_cutoff(int ntypes, vector<vector<vector<double>>> &cutoff_list);

  // Tools for reading the input file

  int split_line(string line, vector<string> &items);

  string get_next_line(istream &str);

  // Fun stuff

  void print_pretty_stuff();

  inline double dr2_3B(const double *dr2, int i, int j, int k, int l);
  inline double dr2_4B(const double *dr2, int i, int j, int k, int l);
  inline void init_distance_tensor(double *dr2, const vector<double> &dr, int natoms);

  template <int max_poly>
  void poly_3B_dense_template(double &e, double &f0, double &f1, double &f2, int ncoeffs_3b,
                              vector<double> &chimes_3b_params, vector<double> &Tn_ij,
                              vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
                              vector<double> &Tnd_ik, vector<double> &Tnd_jk)
  // Templated loop to evaluate the ChIMES polynomial.
  {
    int count = 0;
    for (int i = 0; i < max_poly; i++) {
      const double tn_ij = Tn_ij[i];
      const double tnd_ij = Tnd_ij[i];

      for (int j = 0; j < max_poly; j++) {
        const double tn_ik = Tn_ik[j];
        const double tnd_ik = Tnd_ik[j];
        const double tn_ij_ik = tn_ij * tn_ik;

        for (int k = 0; k < max_poly; k++) {
          if (chimes_3b_params[count] != 0.0) {
            const double tn_jk = Tn_jk[k];
            const double tnd_jk = Tnd_jk[k];
            const double coeff = chimes_3b_params[count];

            e += coeff * tn_ij_ik * tn_jk;
            f0 += coeff * tnd_ij * tn_ik * tn_jk;
            f1 += coeff * tnd_ik * tn_ij * tn_jk;
            f2 += coeff * tnd_jk * tn_ij_ik;
          }
          count++;
        }
      }
    }
  }

  template <int max_poly>
  void poly_4B_dense_template(double &e, double &f0, double &f1, double &f2, double &f3, double &f4,
                              double &f5, int ncoeffs_4b, vector<double> &params_4b,
                              vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_il,
                              vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
                              vector<double> &Tnd_ij, vector<double> &Tnd_ik,
                              vector<double> &Tnd_il, vector<double> &Tnd_jk,
                              vector<double> &Tnd_jl, vector<double> &Tnd_kl)
  {
    double coeff;

    int count = 0;
    for (int i = 0; i < max_poly; i++) {
      const double tn_ij = Tn_ij[i];
      const double tnd_ij = Tnd_ij[i];
      for (int j = 0; j < max_poly; j++) {
        const double tn_ik = Tn_ik[j];
        const double tnd_ik = Tnd_ik[j];
        for (int l = 0; l < max_poly; l++) {
          const double tn_il = Tn_il[l];
          const double tnd_il = Tnd_il[l];
          const double Tn_ij_ik_il = tn_ij * tn_ik * tn_il;
          for (int m = 0; m < max_poly; m++) {
            const double tn_jk = Tn_jk[m];
            const double tnd_jk = Tnd_jk[m];
            for (int n = 0; n < max_poly; n++) {
              const double tn_jl = Tn_jl[n];
              const double tnd_jl = Tnd_jl[n];
              const double Tn_jk_jl = tn_jk * tn_jl;
              for (int o = 0; o < max_poly; o++) {
                const double tn_kl = Tn_kl[o];
                const double tnd_kl = Tnd_kl[o];

                if (params_4b[count] != 0.0) {
                  const double coeff = params_4b[count];

                  e += coeff * Tn_ij_ik_il * Tn_jk_jl * tn_kl;

                  f0 += coeff * tnd_ij * tn_ik * tn_il * Tn_jk_jl * tn_kl;

                  f1 += coeff * tnd_ik * tn_ij * tn_il * Tn_jk_jl * tn_kl;

                  f2 += coeff * tnd_il * tn_ij * tn_ik * Tn_jk_jl * tn_kl;

                  f3 += coeff * tnd_jk * Tn_ij_ik_il * tn_jl * tn_kl;

                  f4 += coeff * tnd_jl * Tn_ij_ik_il * tn_jk * tn_kl;

                  f5 += coeff * tnd_kl * Tn_ij_ik_il * Tn_jk_jl;
                }
                count++;
              }
            }
          }
        }
      }
    }
  }
};

inline void chimesFF::get_fcut(const double dx, const chimesSlotConst &sc, double &fcut,
                               double &fcutderiv)
{

  double fcut0;

  if (fcut_type == fcutType::CUBIC) {
    fcut0 = (1.0 - dx / sc.outer);
    fcut = pow(fcut0, 3.0);
    fcutderiv = pow(fcut0, 2.0);
    fcutderiv *= sc.fcut_dscale;

  } else if (fcut_type == fcutType::TERSOFF) {

    if (dx < sc.fcut_thresh)    // Case 1: Our pair distance is less than the fcut kick-in distance
    {
      fcut = 1.0;
      fcutderiv = 0.0;
    } else if (dx > sc.outer)    // Case 2: Our pair distance is greater than the cutoff
    {
      fcut = 0.0;
      fcutderiv = 0.0;
    } else    // Case 3: We'll use our modified sin function
    {
      fcut0 = (dx - sc.fcut_thresh) / sc.fcut_denom * CHIMES_PI + CHIMES_PI / 2.0;

      fcut = 0.5 + 0.5 * sin(fcut0);
      fcutderiv = 0.5 * cos(fcut0) * sc.fcut_dscale;
    }
  }
}

inline void chimesFF::get_penalty(const double dx, const int &pair_idx, const double inner_cutoff,
                                  double &E_penalty, double &force_scalar)
{
  double r_penalty = 0.0;

  E_penalty = 0.0;
  force_scalar = 1.0;

  if (dx - penalty_params[0] < inner_cutoff) r_penalty = inner_cutoff + penalty_params[0] - dx;

  if (r_penalty > 0.0) {
    E_penalty = r_penalty * r_penalty * r_penalty * penalty_params[1];

    force_scalar = -3.0 * r_penalty * r_penalty * penalty_params[1];

    cout << "chimesFF: " << "Warning: Adding penalty in 2B Cheby calc, r < rmin+penalty_dist "
         << fixed << dx << " " << inner_cutoff + penalty_params[0] << " pair type: " << pair_idx
         << endl;
    cout << "chimesFF: " << "\t...Penalty potential = " << E_penalty << endl;
  }
}

inline void chimesFF::build_atom_and_pair_mappers(const int natoms, const int npairs,
                                                  const vector<int> &typ_idxs,
                                                  const vector<string> &clu_params_pair_typs,
                                                  vector<int> &mapped_pair_idx)
// Interface to array-based version.
{
  build_atom_and_pair_mappers(natoms, npairs, typ_idxs, clu_params_pair_typs,
                              mapped_pair_idx.data());
}

inline void chimesFF::build_atom_and_pair_mappers(const int natoms, const int npairs,
                                                  const vector<int> &typ_idxs,
                                                  const vector<string> &clu_params_pair_typs,
                                                  int *mapped_pair_idx)
{
  // Generate permutations for atoms... all we are doing is permuting the possible indices for typ_idxs

  // build a copy of the atom type vector for permuting

  vector<int> tmp_typ_idxs;
  int nelements;

  nelements = typ_idxs.size();
  tmp_typ_idxs.resize(nelements);

  for (int i = 0; i < nelements; i++) tmp_typ_idxs[i] = i;

  // Build a copy of the original pairs for comparison against permuted pairs

  vector<vector<int>> tmp_pairs;
  tmp_pairs.resize(npairs, vector<int>(2));
  vector<vector<int>> runtime_pairs;
  runtime_pairs.resize(npairs, vector<int>(2));

  int idx = 0;

  for (int i = 0; i < natoms; i++) {

    for (int j = i + 1; j < natoms; j++) {
      tmp_pairs[idx][0] = i;
      tmp_pairs[idx][1] = j;

      idx++;
    }
  }

  vector<string> runtime_pair_typs(npairs);

  do {
    // Check if the permutation leads to pair types that match the order specified by the force field type

    idx = 0;

    for (int i = 0; i < natoms;
         i++)    // Associate the current atom pairs with a "proper" 2-body force field name
    {
      for (int j = i + 1; j < natoms; j++) {
        runtime_pair_typs[idx] =
            atom_int_prpr_map[typ_idxs[tmp_typ_idxs[i]] * natmtyps + typ_idxs[tmp_typ_idxs[j]]];

        idx++;
      }
    }

    bool match = true;

    for (int i = 0; i < npairs; i++) {
      if (clu_params_pair_typs[i] != runtime_pair_typs[i]) {
        match = false;
        break;
      }
    }

    if (match)    // Then we've found an appropriate atom ordering... now what?
    {
      idx = 0;

      for (int i = 0; i < natoms;
           i++)    // Associate the current atom pairs with a "proper" 2-body force field name
      {
        for (int j = i + 1; j < natoms; j++) {
          runtime_pairs[idx][0] = tmp_typ_idxs[i];
          runtime_pairs[idx][1] = tmp_typ_idxs[j];

          idx++;
        }
      }

      break;
    }

  } while (next_permutation(tmp_typ_idxs.begin(), tmp_typ_idxs.begin() + nelements));

  // Once we've found a re-ordering of atoms that properly maps to the force field pair types, need to figure out how to convert that to a map between *pairs*

  idx = 0;

  for (int i = 0; i < npairs; i++)
    for (int j = 0; j < npairs; j++)
      if (((runtime_pairs[i][0] == tmp_pairs[j][0]) && (runtime_pairs[i][1] == tmp_pairs[j][1])) ||
          ((runtime_pairs[i][0] == tmp_pairs[j][1]) && (runtime_pairs[i][1] == tmp_pairs[j][0])))
        mapped_pair_idx[j] = i;
}

inline void chimesFF::set_cheby_polys(vector<double> &Tn, vector<double> &Tnd, double dx,
                                      const chimesSlotConst &sc, const int bodiedness_idx)
{
  // Currently assumes a Morse-style transformation has been requested

  // Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd).  Tnd is the derivative
  // with respect to the interatomic distance, not the transformed distance (x).

  // The Morse transformation bounds (x_avg, x_diff) depend only on the pair
  // type and this slot's cutoffs, so they come from the precomputed table
  // rather than from two exp() calls per interaction.

  bool out_of_range;
  double dx_orig = dx;

  //  The case dx > outer_cutoff is not treated, because it is assumed that the outer smoothing
  //  function will be zero for dx > outer_cutoff.
  if (dx < sc.inner) {
    out_of_range = true;
    dx = sc.inner;
  } else
    out_of_range = false;

  double exprlen = exp(-1 * dx / sc.morse);
  double x = (exprlen - sc.x_avg) / sc.x_diff;
  double dx_dr = (-exprlen / sc.morse) / sc.x_diff;

  double *const tn = Tn.data();
  double *const tnd = Tnd.data();

  if (!out_of_range) {
    // Generate Chebyshev polynomials by recursion.
    //
    // What we're doing here. Want to fit using Cheby polynomials of the 1st kinD[i]. "T_n(x)."
    // We need to calculate the derivative of these polynomials.
    // Derivatives are defined through use of Cheby polynomials of the 2nd kind "U_n(x)", as:
    //
    // d/dx[ T_n(x) = n * U_n-1(x)]
    //
    // So we need to first set up the 1st-kind polynomials ("Tn[]")
    // Then, to compute the derivatives ("Tnd[]"), first set equal to the 2nd-kind, then multiply by n to get the der's

    // First two 1st-kind Chebys:

    tn[0] = 1.0;
    tn[1] = x;

    // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

    tnd[0] = 1.0;
    tnd[1] = 2.0 * x;

    // Use recursion to set up the higher n-value Tn and Tnd's

    const double x2 = 2.0 * x;

    for (int i = 2; i <= poly_orders[bodiedness_idx]; i++) {
      tn[i] = x2 * tn[i - 1] - tn[i - 2];
      tnd[i] = x2 * tnd[i - 1] - tnd[i - 2];
    }

    // Now multiply by n to convert Tnd's to actual derivatives of Tn

    // The following dx_dr compuation assumes a Morse transformation
    // DERIV_CONST is no longer used. (old way: dx_dr = DERIV_CONST*cheby_var_deriv(x_diff, rlen, ff_2body.LAMBDA, ff_2body.CHEBY_TYPE, exprlen);)

    for (int i = poly_orders[bodiedness_idx]; i >= 1; i--) tnd[i] = i * dx_dr * tnd[i - 1];

    tnd[0] = 0.0;
  } else    // out_of_range == true
  {
    cout << "Warning: An intermolecular distance less than the inner cutoff = " << sc.inner
         << " was found\n ";
    cout << "         Distance = " << dx_orig << endl;

    set_polys_out_of_range(Tn, Tnd, dx_orig, x, poly_orders[bodiedness_idx], sc.inner, exprlen,
                           dx_dr);
  }
}

#endif
