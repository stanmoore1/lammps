#pragma once

#include "../particles.h"
#include "../types.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

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
// a1 is the "xhat" (nx) vector, a3 is the "zhat" (nz) vector of the body frame.
// a2 = a3 × a1 gives the yhat vector.
// We convert the 3×3 rotation matrix (a1, a2, a3) to a unit quaternion.

// Convert a 3×3 rotation matrix (stored as rows a1, a2, a3) to a unit quaternion
// following the Shepperd method. The quaternion is (q0, q1, q2, q3) = (w, x, y, z).
inline void rot_to_quat(const double a1[3], const double a2[3], const double a3[3],
                        c_number &q0, c_number &q1, c_number &q2, c_number &q3) {
    // Build R as column-major: R[col][row]
    // a1 = first row = (R[0][0], R[1][0], R[2][0]) etc.
    double R[3][3];
    R[0][0] = a1[0]; R[0][1] = a1[1]; R[0][2] = a1[2];
    R[1][0] = a2[0]; R[1][1] = a2[1]; R[1][2] = a2[2];
    R[2][0] = a3[0]; R[2][1] = a3[1]; R[2][2] = a3[2];

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

inline void read_config(const std::string &filename, ParticleArraysHost &host,
                        SimBox &box, long long &step) {
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
        double x, y, z;
        double a1x, a1y, a1z, a3x, a3y, a3z;
        double vx, vy, vz, Lx, Ly, Lz;

        f >> x >> y >> z
          >> a1x >> a1y >> a1z
          >> a3x >> a3y >> a3z
          >> vx >> vy >> vz
          >> Lx >> Ly >> Lz;

        // Pack index and btype into .w: lower 22 bits = index, upper bits = btype
        int btype_i = host.btype(i);
        unsigned int packed = ((unsigned int)(btype_i) << 22)
                            | (0x003FFFFFu & (unsigned int)(i));
        float w_f;
        std::memcpy(&w_f, &packed, sizeof(float));

        host.poss(i, 0) = static_cast<c_number>(x);
        host.poss(i, 1) = static_cast<c_number>(y);
        host.poss(i, 2) = static_cast<c_number>(z);
        host.poss(i, 3) = static_cast<c_number>(w_f);

        host.vels(i, 0) = static_cast<c_number>(vx);
        host.vels(i, 1) = static_cast<c_number>(vy);
        host.vels(i, 2) = static_cast<c_number>(vz);
        host.vels(i, 3) = 0;

        host.Ls(i, 0) = static_cast<c_number>(Lx);
        host.Ls(i, 1) = static_cast<c_number>(Ly);
        host.Ls(i, 2) = static_cast<c_number>(Lz);
        host.Ls(i, 3) = 0;

        // Compute a2 = a3 × a1
        double a2[3];
        a2[0] = a3y * a1z - a3z * a1y;
        a2[1] = a3z * a1x - a3x * a1z;
        a2[2] = a3x * a1y - a3y * a1x;

        double a1[3] = {a1x, a1y, a1z};
        double a2d[3] = {a2[0], a2[1], a2[2]};
        double a3[3] = {a3x, a3y, a3z};

        c_number q0, q1, q2, q3;
        rot_to_quat(a1, a2d, a3, q0, q1, q2, q3);

        // Store quaternion as (w=q0, x=q1, y=q2, z=q3) in .x .y .z .w
        host.orientations(i, 0) = q0;  // w
        host.orientations(i, 1) = q1;  // x
        host.orientations(i, 2) = q2;  // y
        host.orientations(i, 3) = q3;  // z
    }

    if (!f && !f.eof()) throw std::runtime_error("Config file truncated or malformed");
}
