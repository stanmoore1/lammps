#pragma once

// Reader for standalone-oxDNA-style input files (key = value), so the same
// input that drives the reference oxDNA can drive this code. Only the keys that
// affect this standalone are interpreted; everything else (backend, CUDA_list,
// trajectory_file, ensemble, external_forces, data_output blocks, ...) is
// ignored, exactly as the reference tolerates unknown keys.
//
// Recognized keys:
//   topology, conf_file                          (mandatory)
//   interaction_type   DNA|DNA1 -> oxDNA1, DNA2 -> oxDNA2
//   salt_concentration                           (oxDNA2)
//   T                  e.g. "20C", "300K", or a number in oxDNA units
//   dt, steps, verlet_skin, print_energy_every, seed
//   thermostat         brownian|john (enables NVT); anything else -> NVE
//   newtonian_steps, diff_coeff, pt
//   timing             0|1  (Kokkos-specific: per-kernel timing breakdown)

#include "../simulation.h"
#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace inp_detail {

inline std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Temperature parse following oxDNA: "<x>C" Celsius, "<x>K" Kelvin, else
// already in oxDNA units (1 unit = 3000 K, so 300 K ~ 0.1).
inline double parse_T(const std::string &v) {
    std::string s = trim(v);
    if (s.empty()) throw std::runtime_error("empty temperature");
    char last = s.back();
    if (last == 'C' || last == 'c') return (std::stod(s.substr(0, s.size() - 1)) + 273.15) / 3000.0;
    if (last == 'K' || last == 'k') return std::stod(s.substr(0, s.size() - 1)) / 3000.0;
    return std::stod(s);
}

} // namespace inp_detail

// Parse top-level "key = value" pairs, skipping comments (#...) and the
// contents of { ... } blocks (e.g. data_output_N). Values that themselves open
// a block or contain a computed ${...} expression are not stored as scalars.
inline std::map<std::string, std::string> parse_input_kv(const std::string &file) {
    std::ifstream f(file);
    if (!f) throw std::runtime_error("Cannot open input file: " + file);

    std::map<std::string, std::string> kv;
    int depth = 0;
    std::string line;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);

        std::string t = inp_detail::trim(line);
        if (depth == 0 && !t.empty()) {
            auto eq = t.find('=');
            if (eq != std::string::npos) {
                std::string key = inp_detail::trim(t.substr(0, eq));
                std::string val = inp_detail::trim(t.substr(eq + 1));
                if (!key.empty() && val.find('{') == std::string::npos)
                    kv[key] = val;
            }
        }
        for (char c : line) {
            if (c == '{') depth++;
            else if (c == '}') depth--;
        }
        if (depth < 0) depth = 0;
    }
    return kv;
}

inline void read_input(const std::string &file, SimConfig &cfg) {
    auto kv  = parse_input_kv(file);
    auto has = [&](const char *k) { return kv.count(k) > 0; };
    auto get = [&](const char *k) { return kv[k]; };

    if (!has("topology"))  throw std::runtime_error("input: missing mandatory key 'topology'");
    if (!has("conf_file")) throw std::runtime_error("input: missing mandatory key 'conf_file'");
    cfg.topology_file = get("topology");
    cfg.config_file   = get("conf_file");
    if (has("energy_file")) cfg.energy_file = get("energy_file");

    if (has("interaction_type")) {
        std::string it = inp_detail::lower(get("interaction_type"));
        cfg.model = (it == "dna2") ? 2 : 1;   // DNA / DNA1 -> 1, DNA2 -> 2
    }
    if (has("salt_concentration")) cfg.salt = std::stod(get("salt_concentration"));

    if (has("T"))                  cfg.T = inp_detail::parse_T(get("T"));
    if (has("dt"))                 cfg.dt = std::stod(get("dt"));
    if (has("steps"))              cfg.nsteps = static_cast<long long>(std::stod(get("steps")));
    if (has("verlet_skin"))        cfg.skin = std::stod(get("verlet_skin"));
    if (has("print_energy_every")) cfg.output_freq = static_cast<int>(std::stod(get("print_energy_every")));
    if (has("seed"))               cfg.seed = std::stoull(get("seed"));

    // Thermostat: this code implements the Brownian ("John") refresh thermostat.
    bool thermo_on = false;
    if (has("thermostat")) {
        std::string th = inp_detail::lower(get("thermostat"));
        thermo_on = (th == "brownian" || th == "john");
    }
    if (has("newtonian_steps")) cfg.newtonian_steps = static_cast<int>(std::stod(get("newtonian_steps")));
    if (has("diff_coeff"))      cfg.diff_coeff = std::stod(get("diff_coeff"));
    if (has("pt"))              cfg.pt = std::stod(get("pt"));
    if (!thermo_on) cfg.newtonian_steps = 0;   // NVE unless a supported thermostat is requested

    // Kokkos-specific extension (not a standalone-oxDNA key).
    if (has("timing")) {
        std::string v = inp_detail::lower(get("timing"));
        cfg.timing = (v == "1" || v == "yes" || v == "true" || v == "on");
    }
}
