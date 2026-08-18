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

#include "datamask_audit_kokkos.h"

#ifdef LMP_KOKKOS_DEBUG_SYNC

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "kokkos_type.h"
#include "update.h"

#include <cstring>
#include <map>

using namespace LAMMPS_NS;

static int audit_enabled = 0;

// masks that the style being audited marked itself, by calling
// AtomKokkos::modified() rather than declaring them in datamask_modify
static uint64_t audit_self_declared = 0;

// one entry per style and array, so that a wrong declaration in the inner loop
// is reported once instead of on every step

static std::map<std::string, bigint> audit_found;

/* ---------------------------------------------------------------------- */

void DatamaskAudit::enable(int flag)
{
  audit_enabled = flag;
}

/* ---------------------------------------------------------------------- */

void DatamaskAudit::note_modified(uint64_t mask)
{
  audit_self_declared |= mask;
}

/* ----------------------------------------------------------------------
   Where the per-atom arrays live and how much of each belongs to the atoms
   that exist.  The device side is the one to watch: that is what the kernels
   write, and in a build without a GPU it is also the copy that the host does
   not see.  Collected fresh on both ends of the comparison, because a style may
   reallocate an array and the old pointer would then be dangling.
------------------------------------------------------------------------- */

static void collect(AtomKokkos *atomKK, int nall, std::vector<DatamaskAudit::Array> &out)
{
  out.clear();

  auto take = [&](uint64_t bit, const char *name, const char *data, size_t span, size_t esz,
                  size_t n0) {
    if (!data || n0 == 0 || nall <= 0 || nall > (int) n0) return;
    const size_t per = span * esz / n0;
    if (per == 0) return;
    out.push_back({bit, name, data, (size_t) nall * per, (int) per});
  };

#define LMP_AUDIT_ARRAY(BIT, NAME, KV)                                                    \
  {                                                                                       \
    auto v = (KV).view_device();                                                          \
    take(BIT, NAME, (const char *) v.data(), v.span(),                                    \
         sizeof(typename decltype(v)::value_type), v.extent(0));                          \
  }

  LMP_AUDIT_ARRAY(X_MASK, "x", atomKK->k_x)
  LMP_AUDIT_ARRAY(V_MASK, "v", atomKK->k_v)
  LMP_AUDIT_ARRAY(F_MASK, "f", atomKK->k_f)
  LMP_AUDIT_ARRAY(TAG_MASK, "tag", atomKK->k_tag)
  LMP_AUDIT_ARRAY(TYPE_MASK, "type", atomKK->k_type)
  LMP_AUDIT_ARRAY(MASK_MASK, "mask", atomKK->k_mask)
  LMP_AUDIT_ARRAY(IMAGE_MASK, "image", atomKK->k_image)
  if (atomKK->q_flag) LMP_AUDIT_ARRAY(Q_MASK, "q", atomKK->k_q)
  if (atomKK->molecule_flag) LMP_AUDIT_ARRAY(MOLECULE_MASK, "molecule", atomKK->k_molecule)
  if (atomKK->rmass_flag) LMP_AUDIT_ARRAY(RMASS_MASK, "rmass", atomKK->k_rmass)
  if (atomKK->radius_flag) LMP_AUDIT_ARRAY(RADIUS_MASK, "radius", atomKK->k_radius)
  if (atomKK->mu_flag) LMP_AUDIT_ARRAY(MU_MASK, "mu", atomKK->k_mu)
  if (atomKK->omega_flag) LMP_AUDIT_ARRAY(OMEGA_MASK, "omega", atomKK->k_omega)
  if (atomKK->angmom_flag) LMP_AUDIT_ARRAY(ANGMOM_MASK, "angmom", atomKK->k_angmom)
  if (atomKK->torque_flag) LMP_AUDIT_ARRAY(TORQUE_MASK, "torque", atomKK->k_torque)
  if (atomKK->ellipsoid_flag) LMP_AUDIT_ARRAY(ELLIPSOID_MASK, "ellipsoid", atomKK->k_ellipsoid)

#undef LMP_AUDIT_ARRAY
}

/* ---------------------------------------------------------------------- */

DatamaskAudit::DatamaskAudit(LAMMPS *lmp_in, const char *what_in, const char *style_in,
                             uint64_t datamask_modify) :
    lmp(lmp_in), what(what_in), style(style_in ? style_in : "(unnamed)"),
    declared(datamask_modify), nall(0), active(false)
{
  if (!audit_enabled) return;

  auto *atomKK = (AtomKokkos *) lmp->atom;

  // compare only over the atoms that exist: the allocation runs to nmax, and
  // that tail is genuinely uninitialised

  nall = atomKK->nlocal + atomKK->nghost;
  if (nall <= 0) return;

  audit_self_declared = 0;

  collect(atomKK, nall, arrays);
  if (arrays.empty()) return;

  before.resize(arrays.size());
  for (size_t i = 0; i < arrays.size(); i++) {
    if (declared & arrays[i].bit) continue;    // free to change it, do not copy
    before[i].assign(arrays[i].data, arrays[i].data + arrays[i].bytes);
  }

  active = true;
}

/* ---------------------------------------------------------------------- */

DatamaskAudit::~DatamaskAudit()
{
  if (!active || !audit_enabled) return;

  auto *atomKK = (AtomKokkos *) lmp->atom;

  // whatever the style marked itself while it ran is declared just as much as
  // what it named in datamask_modify
  const uint64_t covered = declared | audit_self_declared;
  audit_self_declared = 0;

  // an exchange or a growth in the middle leaves nothing comparable
  if (atomKK->nlocal + atomKK->nghost != nall) return;

  std::vector<Array> now;
  collect(atomKK, nall, now);

  for (size_t i = 0; i < arrays.size(); i++) {
    if (covered & arrays[i].bit) continue;
    if (before[i].empty()) continue;

    // find the same array again rather than trusting the old pointer
    const Array *cur = nullptr;
    for (auto &n : now)
      if (n.bit == arrays[i].bit) { cur = &n; break; }
    if (!cur || cur->data != arrays[i].data || cur->bytes != before[i].size()) continue;

    if (memcmp(cur->data, before[i].data(), before[i].size()) == 0) continue;

    int iatom = -1;
    for (size_t b = 0; b < before[i].size(); b++)
      if (cur->data[b] != before[i][b]) { iatom = (int) ((int) b / cur->stride); break; }

    const std::string key = style + " changed " + arrays[i].name;
    if (audit_found.count(key)) { audit_found[key]++; continue; }
    audit_found[key] = 1;

    lmp->error->warning(FLERR,
                        "datamask audit: {} {} changed {} without declaring it in "
                        "datamask_modify or marking it modified, first at atom {} of {} "
                        "on step {}",
                        what, style, arrays[i].name, iatom, nall, lmp->update->ntimestep);
  }
}

/* ---------------------------------------------------------------------- */

void DatamaskAudit::report(LAMMPS *lmp)
{
  if (lmp->comm->me != 0) return;

  if (audit_found.empty()) {
    utils::logmesg(lmp, "datamask audit: no undeclared changes to per-atom arrays\n");
    return;
  }

  utils::logmesg(lmp, "datamask audit: undeclared changes to per-atom arrays\n");
  for (auto &f : audit_found)
    utils::logmesg(lmp, "  {} on {} step(s)\n", f.first, f.second);
}

#endif    // LMP_KOKKOS_DEBUG_SYNC
