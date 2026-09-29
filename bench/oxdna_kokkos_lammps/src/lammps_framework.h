#pragma once

// LAMMPS framework of a single-rank KOKKOS run with ghost atoms
// (lammps_ghosts = 1): owns the communication (lammps_comm.h) and neighbor
// (lammps_neigh.h) state and runs the per-step pieces of VerletKokkos::run()
// that surround the force kernels:
//
//   rebuild():     domain->pbc(), atom->map_clear(), comm->exchange(),
//                  comm->borders() (incl. map_set), neighbor->build():
//                  xhold, bins, half/bin/newton pair list, bond topology
//   forward():     comm->forward_comm()          (steps without a rebuild)
//   force_clear(): VerletKokkos::force_clear()   (Zero f, Zero torque, nall)
//   reverse():     comm->reverse_comm()          (newton on, every step)
//   virial():      pair->virial_fdotr_compute()  (energy/virial steps)
//   decide():      neighbor->decide() (every / delay 0 / check)
//
// The fix OXDNA/NPAIR screen (pre_force) is NeighborList::build_screen().

#include "types.h"
#include "particles.h"
#include "neighbor_list.h"
#include "lammps_comm.h"
#include "lammps_neigh.h"
#include <Kokkos_Core.hpp>
#include <algorithm>

struct LammpsFramework {
    LammpsComm  comm;
    LammpsNeigh neigh;
    double box_lo[3] = {0, 0, 0};
    int every = 1;          // neigh_modify every
    bool check = true;      // neigh_modify check
    int ago = 0;            // steps since the last rebuild
    long long nbuilds = 0, ndanger = 0;

    // list_radius: cutforce + skin; lmp_skin: LAMMPS skin (2 * verlet_skin);
    // comm_cutoff: comm_modify cutoff (0 = the list radius)
    void setup(ParticleArrays &p, NeighborList &nl, SimBox &box, double list_radius,
               double lmp_skin, double comm_cutoff, int every_in = 1, bool check_in = true) {
        box.min_image = false;
        every = std::max(1, every_in);
        check = check_in;
        // Box origin: [0, L) when the configuration is already inside it (as
        // the oxDNA confs and the LAMMPS data files), otherwise [-L/2, L/2)
        // (the bench's folding convention).
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.poss);
        const double L[3] = {(double)box.Lx, (double)box.Ly, (double)box.Lz};
        for (int d = 0; d < 3; d++) {
            bool inside = true;
            for (int i = 0; i < p.N && inside; i++) {
                const double x = static_cast<double>(h(i, d));
                if (x < 0 || x >= L[d]) inside = false;
            }
            box_lo[d] = inside ? 0.0 : -0.5 * L[d];
        }
        const double cutghost = std::max(list_radius, comm_cutoff);
        comm.setup(box, box_lo, cutghost, p.N);
        neigh.setup(comm, list_radius, lmp_skin);
        nl.max_neigh = 0;
        p.init_topology_arrays();
    }

    // neighbor->decide(): rebuild now?
    bool decide(const ParticleArrays &p) {
        ago++;
        if (ago % every != 0) return false;
        if (!check) return true;
        const bool flag = neigh.check_distance(p);
        if (flag && ago == every) ndanger++;
        return flag;
    }

    void rebuild(ParticleArrays &p, NeighborList &nl) {
        comm.domain_pbc(p);
        comm.map_clear(p);
        comm.exchange(p);
        comm.borders(p);
        neigh.ncalls++;
        if (check) neigh.store_xhold(p);    // dist_check only
        neigh.bin_atoms(p);
        neigh.build_pairs(p, nl);
        neigh.bond_all(p);
        ago = 0;          // (setup build: not counted in nbuilds, as LAMMPS)
    }

    void forward(ParticleArrays &p) const { comm.forward_comm(p); }
    void reverse(ParticleArrays &p) const { comm.reverse_comm(p); }

    static void force_clear(ParticleArrays &p) {
        const int nall = p.N + p.nghost;
        auto f = p.forces; auto t = p.torques;
        Kokkos::parallel_for("VerletKokkos::force_clear (f)", nall, KOKKOS_LAMBDA(int i) {
            f(i, 0) = 0; f(i, 1) = 0; f(i, 2) = 0;
        });
        Kokkos::parallel_for("VerletKokkos::force_clear (torque)", nall, KOKKOS_LAMBDA(int i) {
            t(i, 0) = 0; t(i, 1) = 0; t(i, 2) = 0;
        });
    }

    // PairVirialFDotRCompute: global virial from f . r over nall (energy /
    // virial steps; the value is not printed by the bench)
    static double virial(const ParticleArrays &p) {
        auto x = p.poss; auto f = p.forces;
        double v = 0;
        Kokkos::parallel_reduce("PairVirialFDotRCompute", p.N + p.nghost, KOKKOS_LAMBDA(int i, double &s) {
            s += static_cast<double>(x(i, 0)) * static_cast<double>(f(i, 0))
               + static_cast<double>(x(i, 1)) * static_cast<double>(f(i, 1))
               + static_cast<double>(x(i, 2)) * static_cast<double>(f(i, 2));
        }, v);
        return v;
    }
};
