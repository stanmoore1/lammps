#pragma once

#include "types.h"
#include "forces/params.h"
#include "forces/tables.h"
#include <cstring>
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

    // Net force / torque (x, y, z, 0) -- zeroed before each force evaluation.
    // Stored in the accumulation precision c_acc (LAMMPS KK_ACC_FLOAT).
    VecA4 forces;
    VecA4 torques;

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

    // oxDNA particle type (A=0, G=1, C=2, T=3), used by oxDNA3 for the
    // sequence-dependent (tetramer) table lookups (LAMMPS: atom type)
    Kokkos::View<uint8_t *> ptype;

    // Per-atom Debye-Huckel charge (LAMMPS atom->qeff): 1, or 0.5 at a strand
    // end when half-charged ends are on. The dh kernel reads qeff(a) once per
    // atom and qeff(b) per in-range pair. Filled by ensure_qeff() (qeff_mode
    // records which convention the view currently holds; -1 = not yet filled).
    Kokkos::View<c_number *> qeff;
    int qeff_mode = -1;

    // LAMMPS-style bond list (neighbor->bondlist): one entry per bond:
    // column 0 = the atom that stores the bond (its 5' end: the atom whose n3
    // is set), column 1 = its 3' partner, column 2 = bond type (1). Built once
    // from the topology (minimum-image mode), or on every neighbor-list
    // rebuild by NeighBond bond_all with the closest image of the partner
    // (lammps_ghosts). Used by the lammps_overhead per-bond kernels.
    Kokkos::View<int **, Kokkos::LayoutRight> bondlist;
    int nbonds = 0;

    // atom->map_array (tag -> local index; identity without ghosts) and
    // atom->sametag (next image of the same tag, -1 = none); the prime-neighbor
    // precomputes resolve the 3'/5' flank tags through the map as LAMMPS does
    Kokkos::View<int *> map_array, sametag;

    // lammps_tables: per-type coefficient tables (oxDNA1/2 kernels), built by
    // ensure_tables() from the DNAParams they were last filled with
    bool use_tables = false;
    OxdnaTables tab;
    DNAParams tab_key{};
    bool tab_key_valid = false;

    // LAMMPS fix OXDNA/PRIME_NEIGHS bond table (d_prime_neighs_bond, t_int_1d_4,
    // LayoutLeft on GPU): per bond (a, b, a3p, b5p) with a = 3' end, b = 5' end,
    // a3p = 3' neighbour of a, b5p = 5' neighbour of b (-1 if none). Rebuilt on
    // neighbor-rebuild steps in lammps_overhead mode; the per-bond stk/fene
    // kernels read their atoms and tetramer context only from this table.
    Kokkos::View<int *[4], Kokkos::LayoutLeft> prime_bond;

    // lammps_overhead mode: LAMMPS' 4D (5^4) base-base excluded-volume tables
    // used by excv's bonded (tetramer) branch. Uniform for oxDNA1/2 (physics
    // unchanged); tet_excv_key caches the 2D values they were filled from.
    Kokkos::View<ExcvParams *> tet_excv_bsbs;
    ExcvParams tet_excv_key{};

    // LAMMPS-overhead mode: a 256-entry tetramer coefficient table, filled with
    // 1.0, used by the per-bond stk/fene kernels to model LAMMPS's 4D
    // sequence-dependent coefficient indexing (4 type reads + table lookups per
    // bond). Uniform value => physics unchanged; only the memory traffic is modelled.
    Kokkos::View<c_number *> tetramer_tbl;

    // LAMMPS-overhead mode: a 1-int "bond overstretched" flag. LAMMPS bond/fene
    // writes this on device and copies it back to the host every step, which is
    // a per-step device->host sync point. We model the exact round-trip with a
    // 0-D device scalar and its host mirror -- the same d_flag / h_flag pair
    // bond_oxdna_fene_kokkos uses. A 0-D scalar deep_copy is contiguous and
    // always valid across spaces (unlike a strided subview).
    Kokkos::View<int>                  overstretch_flag;
    Kokkos::View<int>::host_mirror_type overstretch_flag_host;

    // LAMMPS ghost-atom mode (lammps_ghosts): per-atom data of the periodic
    // images within the ghost cutoff live after the N local atoms, as in
    // LAMMPS (indices N .. N+nghost-1). The arrays that ghosts need (x, the
    // bonus quaternion, the LRF frames, f/torque, type, id3p/id5p, qeff, tag,
    // ...) are sized nmax >= N + nghost; velocities and angular momenta stay
    // local. nghost = 0 and nmax = N in the lean / minimum-image mode.
    int nghost = 0;
    int nmax   = 0;

    // LAMMPS atom->tag (here: the index of the owning local atom, tag - 1):
    // tag(i) = i for local atoms, tag(ghost) = its owner.
    Kokkos::View<int *> tag;
    // atom->ellipsoid: index of the atom's bonus entry (shape, quat). The
    // quaternions are the bonus data (orientations), so ellipsoid(i) = i; the
    // LAMMPS-mode kernels (LRF, comm, integrator) read it as LAMMPS does.
    Kokkos::View<int *> ellipsoid;
    // bonus shape (constant; carried by the border communication only)
    Vec4 shape;
    // atom->mask / atom->molecule (border communication traffic only)
    Kokkos::View<int *> mask, molecule;
    // image flags of the local atoms (updated by the rebuild-step pbc())
    Kokkos::View<int *[3], Kokkos::LayoutRight> image;
    // atom->num_bond / bond_atom (newton_bond on: each bond stored once, on
    // its 5' atom, partner = tag of the 3' atom). NeighBond bond_all builds
    // the bond list from these on every neighbor-list rebuild.
    Kokkos::View<int *> num_bond;
    Kokkos::View<int **, Kokkos::LayoutRight> bond_atom;
    // rmass (LAMMPS ellipsoid atoms carry a per-atom mass; border traffic
    // and the optional nve/asphere integrator)
    Kokkos::View<c_number *> rmass;

    // Number of particles
    int N = 0;

    void allocate(int n) {
        N = n;
        nmax = n;
        nghost = 0;
        poss        = Vec4("poss",        n);
        vels        = Vec4("vels",        n);
        Ls          = Vec4("Ls",          n);
        forces      = VecA4("forces",      n);
        torques     = VecA4("torques",     n);
        orientations= Vec4("orientations",n);
        nx          = Vec4("nx",          n);
        ny          = Vec4("ny",          n);
        nz          = Vec4("nz",          n);
        bonds       = Kokkos::View<LR_bonds *>   ("bonds",       n);
        btype       = Kokkos::View<int *>        ("btype",       n);
        ptype       = Kokkos::View<uint8_t *>    ("ptype",       n);
        qeff        = Kokkos::View<c_number *>("qeff", n);
        qeff_mode   = -1;
        tetramer_tbl = Kokkos::View<c_number *>("tetramer_tbl", 256);
        Kokkos::deep_copy(tetramer_tbl, c_number(1));
        overstretch_flag      = Kokkos::View<int>("overstretch_flag");
        overstretch_flag_host = Kokkos::create_mirror_view(overstretch_flag);
        tag         = Kokkos::View<int *>("tag", n);
        ellipsoid   = Kokkos::View<int *>("ellipsoid", n);
        shape       = Vec4("shape", n);
        mask        = Kokkos::View<int *>("mask", n);
        molecule    = Kokkos::View<int *>("molecule", n);
        image       = Kokkos::View<int *[3], Kokkos::LayoutRight>("image", n);
        num_bond    = Kokkos::View<int *>("num_bond", n);
        bond_atom   = Kokkos::View<int **, Kokkos::LayoutRight>("bond_atom", n, 1);
        rmass       = Kokkos::View<c_number *>("rmass", n);
        map_array   = Kokkos::View<int *>("atom:map_array", n);
        sametag     = Kokkos::View<int *>("atom:sametag", n);
        auto tg = tag, el = ellipsoid, mk = mask; auto sh = shape; auto rm = rmass;
        auto mp = map_array, stg = sametag;
        Kokkos::parallel_for("atom_init", n, KOKKOS_LAMBDA(int i) {
            tg(i) = i; el(i) = i; mk(i) = 1; mp(i) = i; stg(i) = -1;
            sh(i,0) = sh(i,1) = sh(i,2) = c_number(1); sh(i,3) = 0;
            rm(i) = c_number(1);
        });
    }

    // Set the LAMMPS topology arrays (num_bond / bond_atom) from bonds():
    // each bond is stored on its 5' atom (the one whose n3 is set).
    void init_topology_arrays() {
        auto b = bonds; auto nb = num_bond; auto ba = bond_atom;
        Kokkos::parallel_for("atom_init_bonds", N, KOKKOS_LAMBDA(int i) {
            nb(i) = (b(i).n3 >= 0) ? 1 : 0;
            ba(i, 0) = b(i).n3;
        });
    }

    // Grow the per-atom arrays that hold ghost atoms to n entries, keeping the
    // local data (LAMMPS AtomVec::grow).
    void grow(int n) {
        if (n <= nmax) return;
        Kokkos::resize(poss, n);         Kokkos::resize(orientations, n);
        Kokkos::resize(nx, n);           Kokkos::resize(ny, n);          Kokkos::resize(nz, n);
        Kokkos::resize(forces, n);       Kokkos::resize(torques, n);
        Kokkos::resize(bonds, n);        Kokkos::resize(btype, n);       Kokkos::resize(ptype, n);
        Kokkos::resize(qeff, n);         Kokkos::resize(tag, n);         Kokkos::resize(ellipsoid, n);
        Kokkos::resize(shape, n);        Kokkos::resize(mask, n);        Kokkos::resize(molecule, n);
        Kokkos::resize(rmass, n);        Kokkos::resize(sametag, n);
        nmax = n;
    }

    int nall() const { return N + nghost; }

    void zero_forces() {
        Kokkos::deep_copy(forces,  c_acc(0));
        Kokkos::deep_copy(torques, c_acc(0));
    }

    // Build the LAMMPS-style bond list from bonds(i).n3 (topology is static,
    // so this runs once, on the host, like LAMMPS' initial bond-list build).
    void build_bondlist() {
        auto h_bonds = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bonds);
        int nb = 0;
        for (int i = 0; i < N; i++) if (h_bonds(i).n3 >= 0) nb++;
        nbonds     = nb;
        bondlist   = Kokkos::View<int **, Kokkos::LayoutRight>("bondlist", nb, 3);
        prime_bond = Kokkos::View<int *[4], Kokkos::LayoutLeft>("prime_bond", nb);
        auto h_bl  = Kokkos::create_mirror_view(bondlist);
        nb = 0;
        for (int i = 0; i < N; i++)
            if (h_bonds(i).n3 >= 0) { h_bl(nb,0) = i; h_bl(nb,1) = h_bonds(i).n3; h_bl(nb,2) = 1; nb++; }
        Kokkos::deep_copy(bondlist, h_bl);
    }

    // (Re)build the lammps_tables coefficient tables if the parameters changed.
    void ensure_tables(const DNAParams &par) {
        if (tab_key_valid && tab.built && std::memcmp(&tab_key, &par, sizeof(DNAParams)) == 0) return;
        build_oxdna_tables(tab, par);
        std::memcpy(&tab_key, &par, sizeof(DNAParams));
        tab_key_valid = true;
    }

    // Fill qeff for the requested half-charged-ends convention (device kernel;
    // only re-run when the convention changes).
    void ensure_qeff(bool half_ends) {
        const int mode = half_ends ? 1 : 0;
        if (qeff_mode == mode) return;
        auto q = qeff; auto b = bonds;
        Kokkos::parallel_for("oxdna_qeff", N, KOKKOS_LAMBDA(int i) {   // local atoms; ghosts get it by border comm
            const bool end = (b(i).n3 < 0 || b(i).n5 < 0);
            q(i) = (half_ends && end) ? c_number(0.5) : c_number(1);
        });
        qeff_mode = mode;
    }
};

// Host-resident particle arrays for I/O (always in HostSpace)
struct ParticleArraysHost {
    Vec4::host_mirror_type poss;
    Vec4::host_mirror_type vels;
    Vec4::host_mirror_type Ls;
    VecA4::host_mirror_type forces;
    VecA4::host_mirror_type torques;
    Vec4::host_mirror_type orientations;
    Kokkos::View<LR_bonds *>::host_mirror_type bonds;
    Kokkos::View<int *>::host_mirror_type btype;
    Kokkos::View<uint8_t *>::host_mirror_type ptype;
    int N = 0;

    void allocate(int n) {
        N            = n;
        poss         = Vec4::host_mirror_type("poss",         n);
        vels         = Vec4::host_mirror_type("vels",         n);
        Ls           = Vec4::host_mirror_type("Ls",           n);
        forces       = VecA4::host_mirror_type("forces",       n);
        torques      = VecA4::host_mirror_type("torques",      n);
        orientations = Vec4::host_mirror_type("orientations", n);
        bonds        = Kokkos::View<LR_bonds *>::host_mirror_type("bonds",        n);
        btype        = Kokkos::View<int *>::host_mirror_type("btype",        n);
        ptype        = Kokkos::View<uint8_t *>::host_mirror_type("ptype",    n);
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
    Kokkos::deep_copy(d.ptype,        h.ptype);
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
    Kokkos::deep_copy(h.ptype,        d.ptype);
}
