#pragma once

#include "types.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

// All per-particle arrays live here, stored AoS like the oxDNA CUDA c_number4
// arrays (see types.h). Names follow MD_CUDABackend / CUDAMixedBackend.

struct ParticleArrays {
    // Position (x, y, z, w). The .w component stores a float-encoded integer
    // packing particle index (lower 22 bits) and base type btype (upper bits),
    // following the standalone oxDNA convention (_d_poss).
    Vec4 poss;

    // Linear velocity (x, y, z, K = v^2/2 after second_step) (_d_vels)
    Vec4 vels;

    // Angular momentum in the BODY frame, as upstream (_d_Ls)
    Vec4 Ls;

    // Net force (x, y, z, per-particle energy) (_d_forces)
    Vec4 forces;

    // Net torque, BODY frame after the force kernels (_d_torques)
    Vec4 torques;

    // Orientation as unit quaternion (w, x, y, z) (_d_orientations)
    Vec4 orientations;

    // Mixed precision only (CUDAMixedBackend): double copies used by the
    // integrator (_d_possd, _d_velsd, _d_Lsd, _d_orientationsd). Empty otherwise.
    Vec4m possd, velsd, Lsd, orientationsd;

    // Bonded strand neighbours (n3, n5) (_d_bonds)
    Kokkos::View<LR_bonds *> bonds;

    // Integer base type: A=0, C=1, G=2, T=3 (host bookkeeping; the kernels read
    // the copy in poss.w)
    Kokkos::View<int *> btype;

    // Strand-end flag per particle (CUDADNAInteraction::_d_is_strand_end,
    // init_DNA_strand_ends), read by the oxDNA2 edge kernel for Debye-Hueckel
    Kokkos::View<int *> is_strand_end;

    // oxDNA particle type (A=0, G=1, C=2, T=3), the CUDA `particle_types`
    // array; used by oxDNA3 for the sequence-dependent (tetramer) lookups
    Kokkos::View<uint8_t *> ptype;

    // Strand (molecule) id of each particle (_d_particles_to_mols)
    Kokkos::View<int *> strand;

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
        if (OXDNA_MIXED) {
            possd         = Vec4m("possd",         n);
            velsd         = Vec4m("velsd",         n);
            Lsd           = Vec4m("Lsd",           n);
            orientationsd = Vec4m("orientationsd", n);
        }
        bonds       = Kokkos::View<LR_bonds *>   ("bonds",       n);
        btype       = Kokkos::View<int *>        ("btype",       n);
        ptype       = Kokkos::View<uint8_t *>    ("ptype",       n);
        strand      = Kokkos::View<int *>        ("strand",      n);
        is_strand_end = Kokkos::View<int *>      ("is_strand_end", n);
    }

    // init_DNA_strand_ends (CUDADNAInteraction::_init_strand_ends)
    void init_strand_ends() {
        auto b = bonds;
        auto e = is_strand_end;
        Kokkos::parallel_for("init_DNA_strand_ends", OxPolicy(0, N), KOKKOS_LAMBDA(int i) {
            const LR_bonds pb = b(i);
            e(i) = (pb.n3 < 0 || pb.n5 < 0) ? 1 : 0;
        });
    }

    // Two memsets; used by the validation tools. The MD loop uses
    // set_external_forces() instead (one kernel, as upstream).
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
    Kokkos::View<uint8_t *>::host_mirror_type ptype;
    Kokkos::View<int *>::host_mirror_type strand;
    int N = 0;
    int N_strands = 0;

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
        ptype        = Kokkos::View<uint8_t *>::host_mirror_type("ptype",    n);
        strand       = Kokkos::View<int *>::host_mirror_type("strand",   n);
    }
};

// ---------------------------------------------------------------------------
// Mixed precision conversion kernels (CUDA_mixed.cuh float4_to_LR_double4 /
// LR_double4_to_float4 and their quaternion overloads)
// ---------------------------------------------------------------------------
inline void float4_to_double4(const Vec4 &src, const Vec4m &dest) {
    const int N = src.extent(0);
    Kokkos::parallel_for("float4_to_LR_double4", OxPolicy(0, N), KOKKOS_LAMBDA(int i) {
        dest(i, 0) = static_cast<m_number>(src(i, 0));
        dest(i, 1) = static_cast<m_number>(src(i, 1));
        dest(i, 2) = static_cast<m_number>(src(i, 2));
        dest(i, 3) = static_cast<m_number>(src(i, 3));
    });
}

inline void double4_to_float4(const Vec4m &src, const Vec4 &dest) {
    const int N = src.extent(0);
    Kokkos::parallel_for("LR_double4_to_float4", OxPolicy(0, N), KOKKOS_LAMBDA(int i) {
        dest(i, 0) = static_cast<c_number>(src(i, 0));
        dest(i, 1) = static_cast<c_number>(src(i, 1));
        dest(i, 2) = static_cast<c_number>(src(i, 2));
        dest(i, 3) = static_cast<c_number>(src(i, 3));
    });
}

// CUDAMixedBackend::apply_changes_to_simulation_data (float -> double copies)
inline void mixed_float_to_double(ParticleArrays &p) {
    if (!OXDNA_MIXED) return;
    float4_to_double4(p.poss, p.possd);
    float4_to_double4(p.orientations, p.orientationsd);
    float4_to_double4(p.vels, p.velsd);
    float4_to_double4(p.Ls, p.Lsd);
}

// CUDAMixedBackend::apply_simulation_data_changes (double -> float copies)
inline void mixed_double_to_float(ParticleArrays &p) {
    if (!OXDNA_MIXED) return;
    double4_to_float4(p.possd, p.poss);
    double4_to_float4(p.orientationsd, p.orientations);
    double4_to_float4(p.velsd, p.vels);
    double4_to_float4(p.Lsd, p.Ls);
}

// ---------------------------------------------------------------------------
// set_external_forces (CUDA_MD.cuh) with no external forces: one kernel that
// zeroes F and T of every particle before the force kernels, every step.
// ---------------------------------------------------------------------------
inline void set_external_forces(ParticleArrays &p) {
    auto F = p.forces;
    auto T = p.torques;
    Kokkos::parallel_for("set_external_forces", OxPolicy(0, p.N), KOKKOS_LAMBDA(int i) {
        F(i, 0) = 0; F(i, 1) = 0; F(i, 2) = 0; F(i, 3) = 0;
        T(i, 0) = 0; T(i, 1) = 0; T(i, 2) = 0; T(i, 3) = 0;
    });
}

// Deep-copy device <-> host (_host_to_gpu / _gpu_to_host)
inline void copy_to_device(const ParticleArraysHost &h, ParticleArrays &d) {
    Kokkos::deep_copy(d.poss,         h.poss);
    Kokkos::deep_copy(d.vels,         h.vels);
    Kokkos::deep_copy(d.Ls,           h.Ls);
    Kokkos::deep_copy(d.forces,       h.forces);
    Kokkos::deep_copy(d.torques,      h.torques);
    Kokkos::deep_copy(d.orientations, h.orientations);
    Kokkos::deep_copy(d.bonds,        h.bonds);
    Kokkos::deep_copy(d.btype,        h.btype);
    Kokkos::deep_copy(d.ptype,        h.ptype);
    Kokkos::deep_copy(d.strand,       h.strand);
    d.init_strand_ends();
    mixed_float_to_double(d);
}

// MD_CUDABackend::_gpu_to_host: poss, bonds, orientations, particles_to_mols,
// vels, Ls, forces, torques (in mixed precision preceded by the four
// double -> float conversions of CUDAMixedBackend::apply_simulation_data_changes).
// btype/ptype are copied too (upstream: packed in poss.w / not copied).
inline void copy_to_host(ParticleArrays &d, ParticleArraysHost &h) {
    mixed_double_to_float(d);
    Kokkos::deep_copy(h.poss,         d.poss);
    Kokkos::deep_copy(h.bonds,        d.bonds);
    Kokkos::deep_copy(h.orientations, d.orientations);
    Kokkos::deep_copy(h.strand,       d.strand);
    Kokkos::deep_copy(h.vels,         d.vels);
    Kokkos::deep_copy(h.Ls,           d.Ls);
    Kokkos::deep_copy(h.forces,       d.forces);
    Kokkos::deep_copy(h.torques,      d.torques);
    Kokkos::deep_copy(h.btype,        d.btype);
    Kokkos::deep_copy(h.ptype,        d.ptype);
}
