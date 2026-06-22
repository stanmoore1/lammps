#pragma once

#include "types.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

// All per-particle arrays live here. Kokkos::View<c_number*[4]> gives 4-component
// SoA storage: particles are the outer dimension, components are inner. A warp
// reading 32 consecutive particles loads all x-components coalesced, then y, etc.
// The [4] shape also naturally matches GPU float4/double2 alignment.

struct ParticleArrays {
    // Position: (x, y, z, w). The .w component stores a float-encoded integer
    // packing particle index (lower 22 bits) and base type btype (upper bits),
    // following the standalone oxDNA convention for a single coalesced load.
    Vec4 poss;

    // Linear velocity (x, y, z, 0)
    Vec4 vels;

    // Angular momentum (x, y, z, 0)
    Vec4 Ls;

    // Net force (x, y, z, 0) — zeroed before each force evaluation
    Vec4 forces;

    // Net torque (x, y, z, 0) — zeroed before each force evaluation
    Vec4 torques;

    // Orientation as unit quaternion (w, x, y, z) stored in .x/.y/.z/.w
    Vec4 orientations;

    // Precomputed body-frame basis vectors (LAMMPS-faithful "fix oxdna/lrf").
    // nx=a1, ny=a2, nz=a3 are the rows of the rotation matrix from the quaternion.
    // A dedicated LRF precompute kernel fills these once per step; every force
    // kernel then READS them instead of recomputing from the quaternion.
    // Stored as (x, y, z, 0) per particle.
    Vec4 nx, ny, nz;

    // Bonded strand neighbours (n3, n5)
    Kokkos::View<LR_bonds *> bonds;

    // Integer base type: A=0, C=1, G=2, T=3 (same as LAMMPS btype convention)
    Kokkos::View<int *> btype;

    // LAMMPS-overhead mode scratch: the per-bond "prime-neigh" table that LAMMPS
    // re-derives every step (TagPair...PrecomputeBondPrimeNeighs). Unused unless
    // the lammps_overhead toggle is on.
    Kokkos::View<int *[4]> bond_prime_neighs;

    // Number of particles
    int N = 0;

    void allocate(int n) {
        N = n;
        poss        = Vec4("poss",        n);
        vels        = Vec4("vels",        n);
        Ls          = Vec4("Ls",          n);
        forces      = Vec4("forces",      n);
        torques     = Vec4("torques",     n);
        orientations= Vec4("orientations",n);
        nx          = Vec4("nx",          n);
        ny          = Vec4("ny",          n);
        nz          = Vec4("nz",          n);
        bonds       = Kokkos::View<LR_bonds *>   ("bonds",       n);
        btype       = Kokkos::View<int *>        ("btype",       n);
        bond_prime_neighs = Kokkos::View<int *[4]>("bond_prime_neighs", n);
    }

    void zero_forces() {
        Kokkos::deep_copy(forces,  c_number(0));
        Kokkos::deep_copy(torques, c_number(0));
    }
};

// Host-resident particle arrays for I/O (always in HostSpace)
struct ParticleArraysHost {
    Vec4::host_mirror_type poss;
    Vec4::host_mirror_type vels;
    Vec4::host_mirror_type Ls;
    Vec4::host_mirror_type forces;
    Vec4::host_mirror_type torques;
    Vec4::host_mirror_type orientations;
    Kokkos::View<LR_bonds *>::host_mirror_type bonds;
    Kokkos::View<int *>::host_mirror_type btype;
    int N = 0;

    void allocate(int n) {
        N            = n;
        poss         = Vec4::host_mirror_type("poss",         n);
        vels         = Vec4::host_mirror_type("vels",         n);
        Ls           = Vec4::host_mirror_type("Ls",           n);
        forces       = Vec4::host_mirror_type("forces",       n);
        torques      = Vec4::host_mirror_type("torques",      n);
        orientations = Vec4::host_mirror_type("orientations", n);
        bonds        = Kokkos::View<LR_bonds *>::host_mirror_type("bonds",        n);
        btype        = Kokkos::View<int *>::host_mirror_type("btype",        n);
    }
};

// Deep-copy device ↔ host
inline void copy_to_device(const ParticleArraysHost &h, ParticleArrays &d) {
    Kokkos::deep_copy(d.poss,         h.poss);
    Kokkos::deep_copy(d.vels,         h.vels);
    Kokkos::deep_copy(d.Ls,           h.Ls);
    Kokkos::deep_copy(d.forces,       h.forces);
    Kokkos::deep_copy(d.torques,      h.torques);
    Kokkos::deep_copy(d.orientations, h.orientations);
    Kokkos::deep_copy(d.bonds,        h.bonds);
    Kokkos::deep_copy(d.btype,        h.btype);
}

inline void copy_to_host(const ParticleArrays &d, ParticleArraysHost &h) {
    Kokkos::deep_copy(h.poss,         d.poss);
    Kokkos::deep_copy(h.vels,         d.vels);
    Kokkos::deep_copy(h.Ls,           d.Ls);
    Kokkos::deep_copy(h.forces,       d.forces);
    Kokkos::deep_copy(h.torques,      d.torques);
    Kokkos::deep_copy(h.orientations, d.orientations);
    Kokkos::deep_copy(h.bonds,        d.bonds);
    Kokkos::deep_copy(h.btype,        d.btype);
}
