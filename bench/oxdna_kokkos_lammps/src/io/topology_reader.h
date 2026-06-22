#pragma once

#include "../particles.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// Reads an oxDNA topology file (.top).
//
// Format:
//   Line 1: <N_particles> <N_strands>
//   Lines 2..N+1: <strand_id> <base_letter> <n3_idx> <n5_idx>
//     base_letter: A C G T (or a c g t)
//     n3_idx: index of 3' neighbour (-1 at terminus)
//     n5_idx: index of 5' neighbour (-1 at terminus)
//
// Base type mapping (matches LAMMPS/LAMMPS oxDNA convention):
//   A=0, C=1, G=2, T=3
inline int base_letter_to_type(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:
            throw std::runtime_error(std::string("Unknown base letter: ") + c);
    }
}

inline void read_topology(const std::string &filename, ParticleArraysHost &host, int &N) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open topology file: " + filename);

    int N_strands;
    f >> N >> N_strands;
    if (N <= 0) throw std::runtime_error("Invalid particle count in topology");

    host.allocate(N);

    for (int i = 0; i < N; i++) {
        int strand_id, n3, n5;
        char base;
        f >> strand_id >> base >> n3 >> n5;
        host.btype(i)     = base_letter_to_type(base);
        host.bonds(i).n3  = n3;
        host.bonds(i).n5  = n5;
    }
    if (!f) throw std::runtime_error("Topology file truncated or malformed");
}
