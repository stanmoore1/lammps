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

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "fix_shock_info_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "update.h"

using namespace LAMMPS_NS;

#define cpnts_of_rho  1
#define cpnts_of_v    3
#define cpnts_of_epot 1
#define cpnts_of_ekin 4
#define cpnts_of_etot 1
#define cpnts_of_T    4

#define densfactor  1.66053
#define velfactor   0.1

enum{LOWER,CENTER,UPPER,COORD};
enum{BOX,LATTICE,REDUCED};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixShockInfoKokkos<DeviceType>::FixShockInfoKokkos(
    LAMMPS *lmp, int narg, char **arg)
  : FixShockInfo(lmp, narg, arg), maxlayer_kk(0)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read  = X_MASK | V_MASK | MASK_MASK | TYPE_MASK | RMASS_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixShockInfoKokkos<DeviceType>::~FixShockInfoKokkos()
{
  if (copymode) return;

  // parent destructor handles all raw-pointer arrays
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixShockInfoKokkos<DeviceType>::init()
{
  FixShockInfo::init();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixShockInfoKokkos<DeviceType>::end_of_step()
{
  int j, m;

  if ((update->ntimestep - 1) % nfreq < nfreq - nevery * nrepeat) return;

  // ---- Host-only block: set up layer geometry (runs only when nsum == 0) ----
  if (nsum == 0) {
    double *boxlo, *boxhi, *prd;
    if (scaleflag == REDUCED) {
      boxlo = domain->boxlo_lamda;
      boxhi = domain->boxhi_lamda;
      prd   = domain->prd_lamda;
    } else {
      boxlo = domain->boxlo;
      boxhi = domain->boxhi;
      prd   = domain->prd;
    }

    if      (originflag == LOWER)  origin = boxlo[dim];
    else if (originflag == UPPER)  origin = boxhi[dim];
    else if (originflag == CENTER) origin = 0.5 * (boxlo[dim] + boxhi[dim]);

    double lo, hi;
    if (origin < boxlo[dim]) {
      m  = static_cast<int>((boxlo[dim] - origin) * invdelta);
      lo = origin + m * delta;
    } else {
      m  = static_cast<int>((origin - boxlo[dim]) * invdelta);
      lo = origin - m * delta;
      if (lo > boxlo[dim]) lo -= delta;
    }

    if (origin < boxhi[dim]) {
      m  = static_cast<int>((boxhi[dim] - origin) * invdelta);
      hi = origin + m * delta;
      if (hi < boxhi[dim]) hi += delta;
    } else {
      m  = static_cast<int>((origin - boxhi[dim]) * invdelta);
      hi = origin - m * delta;
    }

    offset  = origin + delta;
    nlayers = static_cast<int>((hi - lo) * invdelta + 0.5);
    if (nlayers < 0) nlayers = 0;

    const double volume = domain->xprd * domain->yprd * domain->zprd;
    layer_volume = delta * volume / prd[dim];

    if (nlayers + 1 > maxlayer) {
      maxlayer = nlayers + 1;
      coord        = (double *) memory->srealloc(coord,        maxlayer * sizeof(double), "shock/info:coord");
      count_one    = (double *) memory->srealloc(count_one,    maxlayer * sizeof(double), "shock/info:count_one");
      count_many   = (double *) memory->srealloc(count_many,   maxlayer * sizeof(double), "shock/info:count_many");
      count_total  = (double *) memory->srealloc(count_total,  maxlayer * sizeof(double), "shock/info:count_total");
      values_one   = memory->grow(values_one,   maxlayer, nvalues, "shock/info:values_one");
      values_many  = memory->grow(values_many,  maxlayer, nvalues, "shock/info:values_many");
      values_total = memory->grow(values_total, maxlayer, nvalues, "shock/info:values_total");
      variable_bin = (double *) memory->srealloc(variable_bin, maxlayer * sizeof(double), "shock/info:variable_bin");
    }

    for (m = 0; m < nlayers + 1; m++) {
      coord[m]       = offset + (m + 0.5) * delta;
      variable_bin[m] = 0.0;
      count_many[m]  = count_total[m] = 0.0;
      for (j = 0; j < nvalues; j++) {
        values_many[m][j]  = 0.0;
        values_total[m][j] = 0.0;
      }
    }
  }

  // ---- Zero per-step accumulators on host ----
  nsum++;
  for (m = 0; m < nlayers + 1; m++) {
    count_one[m] = 0.0;
    for (j = 0; j < nvalues; j++) values_one[m][j] = 0.0;
  }

  // ---- Coordinate transform and per-atom computes (host only) ----
  const int nlocal = atom->nlocal;

  if (scaleflag == REDUCED) {
    atomKK->sync(Host, X_MASK);
    domain->x2lamda(nlocal);
    atomKK->modified(Host, X_MASK);
  }

  compute_pe->compute_peratom();
  compute_stress->compute_peratom();

  double *pe_atom      = compute_pe->vector_atom;
  double **stress_atom = compute_stress->array_atom;

  if (!pe_atom)
    error->all(FLERR, "pe/atom compute did not provide vector_atom");
  if (!stress_atom || compute_stress->size_peratom_cols < 6)
    error->all(FLERR, "stress/atom compute did not provide 6 columns");

  // ---- Sync atom data to execution space ----
  atomKK->sync(execution_space, datamask_read);
  if (!atomKK->rmass) atomKK->k_mass.sync<DeviceType>();

  d_x    = atomKK->k_x.view<DeviceType>();
  d_v    = atomKK->k_v.view<DeviceType>();
  d_mask = atomKK->k_mask.view<DeviceType>();
  d_type = atomKK->k_type.view<DeviceType>();
  if (atomKK->rmass) {
    d_rmass    = atomKK->k_rmass.view<DeviceType>();
    d_has_rmass = 1;
  } else {
    d_mass     = atomKK->k_mass.view<DeviceType>();
    d_has_rmass = 0;
  }

  // ---- Copy pe_atom and stress_atom to device ----
  {
    Kokkos::View<double*, Kokkos::LayoutRight, LMPHostType>
      h_pe(pe_atom, nlocal);
    d_pe_atom = Kokkos::View<double*, Kokkos::LayoutRight, DeviceType>("pe_atom", nlocal);
    Kokkos::deep_copy(d_pe_atom, h_pe);
  }
  {
    // stress_atom is a contiguous 2D C-array: row-major [nlocal][6]
    Kokkos::View<double**, Kokkos::LayoutRight, LMPHostType>
      h_stress(&stress_atom[0][0], nlocal, stress_size_peratom);
    d_stress_atom = Kokkos::View<double**, Kokkos::LayoutRight, DeviceType>("stress_atom", nlocal, stress_size_peratom);
    Kokkos::deep_copy(d_stress_atom, h_stress);
  }

  // ---- Resize per-layer device accumulators if needed ----
  if (nlayers + 1 > maxlayer_kk) {
    maxlayer_kk  = nlayers + 1;
    d_count_kk  = Kokkos::View<double*,  Kokkos::LayoutRight, DeviceType>("count_kk",  maxlayer_kk);
    d_values_kk = Kokkos::View<double**, Kokkos::LayoutRight, DeviceType>("values_kk", maxlayer_kk, nvalues);
  }
  Kokkos::deep_copy(d_count_kk,  0.0);
  Kokkos::deep_copy(d_values_kk, 0.0);

  // ---- Cache scalars for the Kokkos kernel ----
  d_offset         = offset;
  d_invdelta       = invdelta;
  d_mvv2e          = force->mvv2e;
  d_dim            = dim;
  d_nlayers_kk     = nlayers;
  d_cpnts_all_kk   = cpnts_all;
  d_stress_size_kk = stress_size_peratom;

  // ---- Run per-atom Kokkos kernel ----
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,
    TagFixShockInfoAtomLoop>(0, nlocal), *this);
  copymode = 0;

  // ---- Copy per-layer results back to host ----
  {
    auto h_count  = Kokkos::create_mirror_view(d_count_kk);
    auto h_values = Kokkos::create_mirror_view(d_values_kk);
    Kokkos::deep_copy(h_count,  d_count_kk);
    Kokkos::deep_copy(h_values, d_values_kk);
    for (m = 0; m < nlayers + 1; m++) {
      count_one[m] = h_count(m);
      for (j = 0; j < nvalues; j++) values_one[m][j] = h_values(m, j);
    }
  }

  // ---- Post-loop host processing (scale kinetic energy, compute totals) ----
  for (m = 0; m < nlayers + 1; m++) {
    values_one[m][5] *= 0.5;
    values_one[m][6] *= 0.5;
    values_one[m][7] *= 0.5;
    values_one[m][8] = values_one[m][5] + values_one[m][6] + values_one[m][7];
    values_one[m][9] = values_one[m][4] + values_one[m][8];
  }

  for (m = 0; m < nlayers + 1; m++)
    for (j = cpnts_all + 2 * stress_size_peratom; j < nvalues; j++)
      values_one[m][j] = values_one[m][j - 2 * stress_size_peratom]
                       + values_one[m][j -     stress_size_peratom];

  if (scaleflag == REDUCED) domain->lamda2x(nlocal);

  // ---- Accumulate into count_many / values_many ----
  for (m = 0; m < nlayers + 1; m++) {
    count_many[m] += count_one[m];
    for (j = 0; j < cpnts_noT; j++)   values_many[m][j] += values_one[m][j];
    for (j = cpnts_all; j < nvalues; j++) values_many[m][j] += values_one[m][j];
  }

  // ---- Output block (every nfreq steps) ----
  if (update->ntimestep % nfreq == 0) {
    MPI_Allreduce(count_many, count_total, nlayers + 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&values_many[0][0], &values_total[0][0],
                  nvalues * (nlayers + 1), MPI_DOUBLE, MPI_SUM, world);

    for (m = 0; m < nlayers + 1; m++) {
      count_one[m] = count_many[m] = 0.0;
      for (j = 0; j < cpnts_noT; j++)   values_one[m][j] = values_many[m][j] = 0.0;
      for (j = cpnts_all; j < nvalues; j++) values_one[m][j] = values_many[m][j] = 0.0;
    }

    int layer, bin, coordswitch;
    double firstcoord, lastcoord, frac;
    layer = bin = coordswitch = 0;
    firstcoord = lastcoord = frac = 0.0;

    for (layer = 0; layer < nlayers; layer++) {
      count_one[bin] += count_total[layer];
      for (j = 0; j < cpnts_noT; j++)      values_one[bin][j] += values_total[layer][j];
      for (j = cpnts_all; j < nvalues; j++) values_one[bin][j] += values_total[layer][j];

      if (!coordswitch) {
        coordswitch = 1;
        firstcoord  = coord[layer];
      }
      if (coordswitch) variable_bin[bin] += 1.0;

      if ((count_one[bin] + count_total[layer + 1] > nmin * nsum + 1.e-9)
          || (layer == nlayers - 1)) {
        if (bin == 0) coord[bin] = 0.0;

        if ((fabs(count_one[bin] - nmin * nsum) < 1.e-9) || (layer == nlayers - 1))
          frac = 0.0;
        else
          frac = (nmin * nsum - count_one[bin]) / count_total[layer + 1];

        lastcoord    = coord[layer];
        coord[bin]  += 0.5 * (lastcoord + firstcoord);
        variable_bin[bin] += frac;
        count_one[bin]    += frac * count_total[layer + 1];
        coord[bin]        += 0.5 * frac * delta;

        for (j = 0; j < cpnts_noT; j++)
          values_one[bin][j] += frac * values_total[layer + 1][j];
        for (j = cpnts_all; j < nvalues; j++)
          values_one[bin][j] += frac * values_total[layer + 1][j];

        variable_bin[bin + 1] -= frac;
        count_one[bin + 1]    -= frac * count_total[layer + 1];
        coord[bin + 1]         = 0.5 * frac * delta;

        for (j = 0; j < cpnts_noT; j++)
          values_one[bin + 1][j] -= frac * values_total[layer + 1][j];
        for (j = cpnts_all; j < nvalues; j++)
          values_one[bin + 1][j] -= frac * values_total[layer + 1][j];

        bin++;
        coordswitch = 0;
      }
    }

    nlayers = bin;

    const double boltz  = force->boltz;
    const double mvv2e_ = force->mvv2e;

    for (m = 0; m < nlayers; m++) {
      count_total[m] = count_one[m] / nsum;
      if (count_one[m] > 1.e-12) {
        values_total[m][0]  = values_one[m][0] / (nsum * layer_volume * variable_bin[m]);
        values_total[m][0] *= densfactor;
      } else values_total[m][0] = 0.0;

      for (j = 1; j < cpnts_noT; j++) {
        values_total[m][j] = (count_one[m] > 1.e-12)
          ? values_one[m][j] / count_one[m] : 0.0;
      }
      for (j = cpnts_all; j < nvalues; j++) {
        if (count_one[m] > 1.e-12) {
          values_total[m][j]  = values_one[m][j] / (nsum * layer_volume * variable_bin[m]);
          values_total[m][j] *= pressfactor;
        } else values_total[m][j] = 0.0;
      }

      if (count_one[m] > 1.e-12) {
        values_total[m][10] = values_one[m][5]
          - 0.5 * mvv2e_ * values_one[m][0] * values_total[m][1] * values_total[m][1];
        values_total[m][10] /= 0.5 * boltz * count_one[m];
        values_total[m][11] = values_one[m][6]
          - 0.5 * mvv2e_ * values_one[m][0] * values_total[m][2] * values_total[m][2];
        values_total[m][11] /= 0.5 * boltz * count_one[m];
        values_total[m][12] = values_one[m][7]
          - 0.5 * mvv2e_ * values_one[m][0] * values_total[m][3] * values_total[m][3];
        values_total[m][12] /= 0.5 * boltz * count_one[m];
        values_total[m][13] = (values_total[m][10] + values_total[m][11]
                               + values_total[m][12]) / 3.0;
      } else {
        values_total[m][10] = values_total[m][11] =
        values_total[m][12] = values_total[m][13] = 0.0;
      }

      for (j = 1; j <= 3; j++) values_total[m][j] *= velfactor;
    }

    if (comm->me == 0) {
      // energy/info file
      {
        const std::string fname = fmt::format("{}.{}", einfo_fileprefix, update->ntimestep);
        fp = fopen(fname.c_str(), "w");
        if (!fp) error->one(FLERR, "Cannot open output file {} for shock/info", fname);
        utils::print(fp, "Spatial-averaged (Variable bins) data for Shock Wave Simulation with REBO potential:\n");
        utils::print(fp, "TimeStep \tNumber-of-layers (one per snapshot)\n");
        utils::print(fp, fmt::format("{} \t\t{}\n", update->ntimestep, nlayers));
        utils::print(fp, "Layer# \t      Coord \t      #Atoms \t      #deltas \t      Dens "
                         "\t      Vx \t      Vy \t      Vz \t      Epot \t      Ekinx "
                         "\t      Ekiny \t      Ekinz \t      Ekin \t      Etot "
                         "\t      Tx \t      Ty \t      Tz \t      T \n");
        for (m = 1; m < nlayers; m++) {
          utils::print(fp, fmt::format(" {} \t {:10.5f}\t {:11.5f}\t {:10.4f}\t",
                     m + 1, coord[m], count_total[m], variable_bin[m]));
          for (j = 0; j < cpnts_all; j++)
            utils::print(fp, fmt::format(" {:14.8f}\t", values_total[m][j]));
          utils::print(fp, "\n");
        }
        fclose(fp);
      }

      // stress file
      {
        const std::string fname = fmt::format("{}.{}", stress_fileprefix, update->ntimestep);
        fp = fopen(fname.c_str(), "w");
        if (!fp) error->one(FLERR, "Cannot open output file {} for shock/info", fname);
        utils::print(fp, "Spatial-averaged (Variable bins) data for Shock Wave Simulation with REBO potential:\n");
        utils::print(fp, "TimeStep \tNumber-of-layers (one per snapshot)\n");
        utils::print(fp, fmt::format("{} \t\t{}\n", update->ntimestep, nlayers));
        utils::print(fp, "Layer# \t      Coord \t      #Atoms \t      #deltas "
                         "\t      Dens \t    PxxPot \t    PyyPot \t    PzzPot "
                         "\t    PxyPot \t    PxzPot \t    PyzPot "
                         "\t    PxxKin \t    PyyKin \t    PzzKin "
                         "\t    PxyKin \t    PxzKin \t    PyzKin "
                         "\t    PxxTot \t    PyyTot \t    PzzTot "
                         "\t    PxyTot \t    PxzTot \t    PyzTot \n");
        for (m = 1; m < nlayers; m++) {
          utils::print(fp, fmt::format(" {} \t {:10.5f}\t {:11.5f}\t {:10.4f}\t",
                     m + 1, coord[m], count_total[m], variable_bin[m]));
          utils::print(fp, fmt::format(" {:14.8f}\t", values_total[m][0]));
          for (j = cpnts_all; j < nvalues; j++)
            utils::print(fp, fmt::format(" {:14.8f}\t", -values_total[m][j]));
          utils::print(fp, "\n");
        }
        fclose(fp);
      }
    }

    nsum = 0;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixShockInfoKokkos<DeviceType>::operator()(
    TagFixShockInfoAtomLoop, const int &i) const
{
  if (!(d_mask[i] & groupbit)) return;

  const int ilayer = static_cast<int>((d_x(i, d_dim) - d_offset) * d_invdelta);
  if (ilayer < 0) return;
  if (ilayer > d_nlayers_kk) return;

  const KK_FLOAT m = d_has_rmass ? d_rmass[i] : d_mass[d_type[i]];
  const KK_FLOAT vx = d_v(i, 0), vy = d_v(i, 1), vz = d_v(i, 2);
  const double   Smass = d_mvv2e * (double)m;

  Kokkos::atomic_add(&d_count_kk(ilayer),    1.0);
  Kokkos::atomic_add(&d_values_kk(ilayer, 0), (double)m);
  Kokkos::atomic_add(&d_values_kk(ilayer, 1), (double)vx);
  Kokkos::atomic_add(&d_values_kk(ilayer, 2), (double)vy);
  Kokkos::atomic_add(&d_values_kk(ilayer, 3), (double)vz);
  Kokkos::atomic_add(&d_values_kk(ilayer, 4), d_pe_atom(i));
  Kokkos::atomic_add(&d_values_kk(ilayer, 5), Smass * (double)(vx*vx));
  Kokkos::atomic_add(&d_values_kk(ilayer, 6), Smass * (double)(vy*vy));
  Kokkos::atomic_add(&d_values_kk(ilayer, 7), Smass * (double)(vz*vz));

  // virial (potential part) from stress compute
  for (int js = 0; js < d_stress_size_kk; js++)
    Kokkos::atomic_add(&d_values_kk(ilayer, d_cpnts_all_kk + js),
                       d_stress_atom(i, js));

  // kinetic contribution to virial (subtracted into the next block)
  const double ks0 = Smass * (double)(vx*vx);
  const double ks1 = Smass * (double)(vy*vy);
  const double ks2 = Smass * (double)(vz*vz);
  const double ks3 = Smass * (double)(vx*vy);
  const double ks4 = Smass * (double)(vx*vz);
  const double ks5 = Smass * (double)(vy*vz);
  Kokkos::atomic_add(&d_values_kk(ilayer, d_cpnts_all_kk + d_stress_size_kk + 0), -ks0);
  Kokkos::atomic_add(&d_values_kk(ilayer, d_cpnts_all_kk + d_stress_size_kk + 1), -ks1);
  Kokkos::atomic_add(&d_values_kk(ilayer, d_cpnts_all_kk + d_stress_size_kk + 2), -ks2);
  Kokkos::atomic_add(&d_values_kk(ilayer, d_cpnts_all_kk + d_stress_size_kk + 3), -ks3);
  Kokkos::atomic_add(&d_values_kk(ilayer, d_cpnts_all_kk + d_stress_size_kk + 4), -ks4);
  Kokkos::atomic_add(&d_values_kk(ilayer, d_cpnts_all_kk + d_stress_size_kk + 5), -ks5);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixShockInfoKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixShockInfoKokkos<LMPHostType>;
#endif
}
