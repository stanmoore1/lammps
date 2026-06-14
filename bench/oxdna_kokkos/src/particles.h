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
    Kokkos::View<c_number *[4]> poss;

    // Linear velocity (x, y, z, 0)
    Kokkos::View<c_number *[4]> vels;

    // Angular momentum (x, y, z, 0)
    Kokkos::View<c_number *[4]> Ls;

    // Net force (x, y, z, 0) — zeroed before each force evaluation
    Kokkos::View<c_number *[4]> forces;

    // Net torque (x, y, z, 0) — zeroed before each force evaluation
    Kokkos::View<c_number *[4]> torques;

    // Orientation as unit quaternion (w, x, y, z) stored in .x/.y/.z/.w
    Kokkos::View<c_number *[4]> orientations;

    // Bonded strand neighbours (n3, n5)
    Kokkos::View<LR_bonds *> bonds;

    // Integer base type: A=0, C=1, G=2, T=3 (same as LAMMPS btype convention)
    Kokkos::View<int *> btype;

    // Number of particles
    int N = 0;

    void allocate(int n) {
        N = n;
        poss        = Kokkos::View<c_number *[4]>("poss",        n);
        vels        = Kokkos::View<c_number *[4]>("vels",        n);
        Ls          = Kokkos::View<c_number *[4]>("Ls",          n);
        forces      = Kokkos::View<c_number *[4]>("forces",      n);
        torques     = Kokkos::View<c_number *[4]>("torques",     n);
        orientations= Kokkos::View<c_number *[4]>("orientations",n);
        bonds       = Kokkos::View<LR_bonds *>   ("bonds",       n);
        btype       = Kokkos::View<int *>        ("btype",       n);
    }

    void zero_forces() {
        Kokkos::deep_copy(forces,  c_number(0));
        Kokkos::deep_copy(torques, c_number(0));
    }
};

// Host-resident particle arrays for I/O (always in HostSpace)
struct ParticleArraysHost {
    Kokkos::View<c_number *[4], Kokkos::HostSpace> poss;
    Kokkos::View<c_number *[4], Kokkos::HostSpace> vels;
    Kokkos::View<c_number *[4], Kokkos::HostSpace> Ls;
    Kokkos::View<c_number *[4], Kokkos::HostSpace> forces;
    Kokkos::View<c_number *[4], Kokkos::HostSpace> torques;
    Kokkos::View<c_number *[4], Kokkos::HostSpace> orientations;
    Kokkos::View<LR_bonds *,    Kokkos::HostSpace> bonds;
    Kokkos::View<int *,         Kokkos::HostSpace> btype;
    int N = 0;

    void allocate(int n) {
        N            = n;
        poss         = Kokkos::View<c_number *[4], Kokkos::HostSpace>("poss",         n);
        vels         = Kokkos::View<c_number *[4], Kokkos::HostSpace>("vels",         n);
        Ls           = Kokkos::View<c_number *[4], Kokkos::HostSpace>("Ls",           n);
        forces       = Kokkos::View<c_number *[4], Kokkos::HostSpace>("forces",       n);
        torques      = Kokkos::View<c_number *[4], Kokkos::HostSpace>("torques",      n);
        orientations = Kokkos::View<c_number *[4], Kokkos::HostSpace>("orientations", n);
        bonds        = Kokkos::View<LR_bonds *,    Kokkos::HostSpace>("bonds",        n);
        btype        = Kokkos::View<int *,         Kokkos::HostSpace>("btype",        n);
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
