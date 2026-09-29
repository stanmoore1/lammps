#pragma once

#include "../particles.h"
#include "../types.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// Read an oxDNA configuration file (.conf / .dat).
//
// Format:
//   t = <step>
//   b = <Lx> <Ly> <Lz>
//   E = <Etot> <Ekin> <Epot>
//   (per nucleotide, one line each):
//   <x> <y> <z>  <a1x> <a1y> <a1z>  <a3x> <a3y> <a3z>
//   <vx> <vy> <vz>  <Lx> <Ly> <Lz>
//
// a1 is the "xhat" (nx) vector, a3 is the "zhat" (nz) vector of the body frame;
// they are orthonormalised as upstream and a2 = a3 x a1. The angular momentum
// L is the BODY-frame angular momentum (as written and read by oxDNA).
// We convert the 3x3 rotation matrix (a1, a2, a3) to a unit quaternion.

// Convert a 3x3 rotation matrix (stored as rows a1, a2, a3) to a unit quaternion
// following the Shepperd method. The quaternion is (q0, q1, q2, q3) = (w, x, y, z).
inline void rot_to_quat(const double a1[3], const double a2[3], const double a3[3],
                        c_number &q0, c_number &q1, c_number &q2, c_number &q3) {
    // Store a1, a2, a3 as the COLUMNS of R, so that get_vectors_from_quat()
    // (which extracts the rotation-matrix columns) reconstructs nx=a1, ny=a2,
    // nz=a3 exactly. (a1/a2/a3 are the lab-frame body axes from the conf.)
    double R[3][3];
    R[0][0] = a1[0]; R[1][0] = a1[1]; R[2][0] = a1[2];
    R[0][1] = a2[0]; R[1][1] = a2[1]; R[2][1] = a2[2];
    R[0][2] = a3[0]; R[1][2] = a3[1]; R[2][2] = a3[2];

    double trace = R[0][0] + R[1][1] + R[2][2];
    double w, x, y, z;
    if (trace > 0) {
        double s = 0.5 / std::sqrt(trace + 1.0);
        w = 0.25 / s;
        x = (R[2][1] - R[1][2]) * s;
        y = (R[0][2] - R[2][0]) * s;
        z = (R[1][0] - R[0][1]) * s;
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        double s = 2.0 * std::sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]);
        w = (R[2][1] - R[1][2]) / s;
        x = 0.25 * s;
        y = (R[0][1] + R[1][0]) / s;
        z = (R[0][2] + R[2][0]) / s;
    } else if (R[1][1] > R[2][2]) {
        double s = 2.0 * std::sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]);
        w = (R[0][2] - R[2][0]) / s;
        x = (R[0][1] + R[1][0]) / s;
        y = 0.25 * s;
        z = (R[1][2] + R[2][1]) / s;
    } else {
        double s = 2.0 * std::sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]);
        w = (R[1][0] - R[0][1]) / s;
        x = (R[0][2] + R[2][0]) / s;
        y = (R[1][2] + R[2][1]) / s;
        z = 0.25 * s;
    }
    // normalise
    double norm = std::sqrt(w*w + x*x + y*y + z*z);
    q0 = static_cast<c_number>(w / norm);
    q1 = static_cast<c_number>(x / norm);
    q2 = static_cast<c_number>(y / norm);
    q3 = static_cast<c_number>(z / norm);
}

// Shift every strand so that its centre of mass lies inside [0, L)
// (SimBackend::fix_diffusion / read_next_configuration).
inline void shift_strands_into_box(ParticleArraysHost &host, const SimBox &box) {
    const int NS = std::max(host.N_strands, 1);
    std::vector<double> com(3 * NS, 0.0);
    std::vector<int> n(NS, 0);
    for (int i = 0; i < host.N; i++) {
        const int s = host.strand(i);
        for (int d = 0; d < 3; d++) com[3 * s + d] += (double)host.poss(i, d);
        n[s]++;
    }
    const double L[3] = {(double)box.Lx, (double)box.Ly, (double)box.Lz};
    for (int i = 0; i < host.N; i++) {
        const int s = host.strand(i);
        for (int d = 0; d < 3; d++) {
            const double c = com[3 * s + d] / n[s];
            host.poss(i, d) = static_cast<c_number>((double)host.poss(i, d) - L[d] * std::floor(c / L[d]));
        }
    }
}

inline void read_config(const std::string &filename, ParticleArraysHost &host,
                        SimBox &box, long long &step, bool fix_diffusion = true) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open config file: " + filename);

    // Header
    std::string token;
    char eq;
    f >> token >> eq >> step;            // t = <step>
    f >> token >> eq >> box.Lx >> box.Ly >> box.Lz;  // b = Lx Ly Lz
    // skip E line
    std::string eline;
    std::getline(f, eline); // consume rest of b-line
    std::getline(f, eline); // skip E line

    const int N = host.N;
    for (int i = 0; i < N; i++) {
        // oxDNA conf lines have 9 mandatory columns (position, a1, a3) and
        // optionally 6 more (velocity, angular momentum). Velocity-less confs
        // (9 columns) are valid; the missing v/L default to zero. Parse per
        // line so a short line never bleeds into the next particle.
        std::string pline;
        if (!std::getline(f, pline))
            throw std::runtime_error("Config file truncated (too few particle lines)");
        std::istringstream ls(pline);

        double x, y, z;
        double a1x, a1y, a1z, a3x, a3y, a3z;
        double vx = 0, vy = 0, vz = 0, Lx = 0, Ly = 0, Lz = 0;

        if (!(ls >> x >> y >> z
                 >> a1x >> a1y >> a1z
                 >> a3x >> a3y >> a3z))
            throw std::runtime_error("Malformed config line (need >= 9 columns)");
        ls >> vx >> vy >> vz >> Lx >> Ly >> Lz;  // optional; stay 0 if absent

        // .w carries the base type, so that the oxDNA1/2 kernels get it with
        // the position load, as upstream (get_particle_type(poss.w)). Upstream
        // bit-packs btype << 22 | index into the float; here the type is
        // stored as a plain number (the packed pattern is a denormal float,
        // which flush-to-zero / fast math would destroy), the index is not
        // needed (no host-side reordering of output).
        host.poss(i, 0) = static_cast<c_number>(x);
        host.poss(i, 1) = static_cast<c_number>(y);
        host.poss(i, 2) = static_cast<c_number>(z);
        host.poss(i, 3) = static_cast<c_number>(host.btype(i));

        host.vels(i, 0) = static_cast<c_number>(vx);
        host.vels(i, 1) = static_cast<c_number>(vy);
        host.vels(i, 2) = static_cast<c_number>(vz);
        host.vels(i, 3) = 0;

        host.Ls(i, 0) = static_cast<c_number>(Lx);
        host.Ls(i, 1) = static_cast<c_number>(Ly);
        host.Ls(i, 2) = static_cast<c_number>(Lz);
        host.Ls(i, 3) = 0;

        // SimBackend::read_next_configuration: normalise v1 and v3, remove the
        // v3 component of v1, normalise, v2 = v3 x v1
        double a1[3] = {a1x, a1y, a1z};
        double a3[3] = {a3x, a3y, a3z};
        auto normalize = [](double v[3]) {
            const double n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            v[0] /= n; v[1] /= n; v[2] /= n;
        };
        normalize(a1);
        normalize(a3);
        const double a13 = a1[0]*a3[0] + a1[1]*a3[1] + a1[2]*a3[2];
        for (int d = 0; d < 3; d++) a1[d] -= a3[d] * a13;
        normalize(a1);
        double a2d[3] = {a3[1]*a1[2] - a3[2]*a1[1], a3[2]*a1[0] - a3[0]*a1[2], a3[0]*a1[1] - a3[1]*a1[0]};
        normalize(a2d);

        c_number q0, q1, q2, q3;
        rot_to_quat(a1, a2d, a3, q0, q1, q2, q3);

        // Store quaternion as (w=q0, x=q1, y=q2, z=q3) in .x .y .z .w
        host.orientations(i, 0) = q0;  // w
        host.orientations(i, 1) = q1;  // x
        host.orientations(i, 2) = q2;  // y
        host.orientations(i, 3) = q3;  // z
    }

    if (!f && !f.eof()) throw std::runtime_error("Config file truncated or malformed");

    // fix_diffusion (default true upstream): shift every strand by whole box
    // vectors so that its centre of mass lies in [0, L) (OrthogonalBox::
    // shift_particle with the strand COM, computed in double). Positions are
    // otherwise kept as read (not folded), as upstream.
    if (fix_diffusion) shift_strands_into_box(host, box);
}
