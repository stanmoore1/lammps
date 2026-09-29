#pragma once

#include "../particles.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Reads an oxDNA topology file (.top), in either of the two standalone-oxDNA
// formats (upstream TopologyParser):
//
// Old ("3'->5'") format:
//   Line 1: <N_particles> <N_strands>
//   Lines 2..N+1: <strand_id> <base_letter> <n3_idx> <n5_idx>
//     base_letter: A C G T (or a c g t; U is read as T)
//     n3_idx: index of 3' neighbour (-1 at terminus)
//     n5_idx: index of 5' neighbour (-1 at terminus)
//
// New ("5->3") format:
//   Line 1: <N_particles> <N_strands> 5->3
//   One line per strand: <sequence written 5'->3'> [key=value ...]
//     e.g. "ACGTTGCA type=DNA circular=false". Consecutive particles of a
//     strand are bonded: particle i has n5 = i-1 and n3 = i+1 (circular=true
//     closes the strand). Custom integer base types "(...)" are not supported.
//
// Base type mapping (matches LAMMPS/LAMMPS oxDNA convention):
//   btype: A=0, C=1, G=2, T=3
// and, for oxDNA3 (tetramer tables), the oxDNA particle type
//   ptype: A=0, G=1, C=2, T=3   (upstream N_A, N_G, N_C, N_T)
inline int base_letter_to_type(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        case 'U': case 'u': return 3;
        default:
            throw std::runtime_error(std::string("Unknown base letter: ") + c);
    }
}

inline uint8_t btype_to_ptype(int btype) {
    static const uint8_t m[4] = {0, 2, 1, 3};   // A C G T -> A G C T order
    return m[btype & 3];
}

inline void read_topology(const std::string &filename, ParticleArraysHost &host, int &N) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open topology file: " + filename);

    std::string header;
    std::getline(f, header);
    std::istringstream hs(header);
    int N_strands = 0;
    std::string token3;
    hs >> N >> N_strands;
    hs >> token3;
    if (!hs.fail() && !token3.empty() && token3 != "5->3")
        throw std::runtime_error("Topology header must be '<N> <N_strands>' or '<N> <N_strands> 5->3'");
    const bool new_format = (token3 == "5->3");
    if (N <= 0) throw std::runtime_error("Invalid particle count in topology");

    host.allocate(N);
    host.N_strands = N_strands;

    if (!new_format) {
        int i = 0;
        std::string line;
        while (i < N && std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            int strand_id, n3, n5;
            std::string base;
            if (!(ls >> strand_id >> base >> n3 >> n5))
                throw std::runtime_error("Topology file truncated or malformed (line " +
                                         std::to_string(i + 2) + ")");
            if (base.size() != 1)
                throw std::runtime_error("Custom (integer) base types are not supported: " + base);
            host.btype(i)    = base_letter_to_type(base[0]);
            host.ptype(i)    = btype_to_ptype(host.btype(i));
            host.strand(i)   = strand_id - 1;   // 1-based in the file (as upstream)
            if (host.strand(i) < 0 || host.strand(i) >= N_strands)
                throw std::runtime_error("Topology: strand id out of range (line " + std::to_string(i + 2) + ")");
            host.bonds(i).n3 = n3;
            host.bonds(i).n5 = n5;
            i++;
        }
        if (i != N) throw std::runtime_error("Topology file truncated or malformed");
        return;
    }

    // new format
    int idx = 0;
    for (int ns = 0; ns < N_strands; ns++) {
        std::string line;
        do {
            if (!std::getline(f, line))
                throw std::runtime_error("Not enough strand lines in topology file");
        } while (line.find_first_not_of(" \t\r") == std::string::npos);
        std::istringstream ls(line);
        std::string seq, kvtok;
        ls >> seq;
        bool circular = false;
        while (ls >> kvtok) {
            auto eq = kvtok.find('=');
            if (eq == std::string::npos) continue;
            std::string k = kvtok.substr(0, eq), v = kvtok.substr(eq + 1);
            if (k == "circular") circular = (v == "true" || v == "1" || v == "yes");
            if (k == "type" && v != "DNA")
                throw std::runtime_error("topology: only type=DNA strands are supported");
        }
        const int n_in = static_cast<int>(seq.size());
        if (idx + n_in > N) throw std::runtime_error("Too many particles in topology file");
        for (int i = 0; i < n_in; i++) {
            if (seq[i] == '(') throw std::runtime_error("Custom (integer) base types are not supported");
            const int p = idx + i;
            host.btype(p)    = base_letter_to_type(seq[i]);
            host.ptype(p)    = btype_to_ptype(host.btype(p));
            host.strand(p)   = ns;
            host.bonds(p).n5 = (i > 0) ? p - 1 : -1;
            host.bonds(p).n3 = (i < n_in - 1) ? p + 1 : -1;
        }
        if (circular && n_in > 1) {
            host.bonds(idx + n_in - 1).n3 = idx;
            host.bonds(idx).n5 = idx + n_in - 1;
        }
        idx += n_in;
    }
    if (idx != N) throw std::runtime_error("Topology: particle count does not match the header");
}
