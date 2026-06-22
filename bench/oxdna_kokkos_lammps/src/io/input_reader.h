#pragma once

// Reader for standalone-oxDNA-style input files (key = value), so the same
// input that drives the reference oxDNA can drive this code. Only the keys that
// affect this standalone are interpreted; everything else (backend, CUDA_list,
// trajectory_file, ensemble, external_forces, data_output blocks, ...) is
// ignored, exactly as the reference tolerates unknown keys.
//
// oxDNA value expressions are supported: $(key) substitutes another key's value
// and ${ ... } evaluates a + - * / () arithmetic expression, e.g.
//   print_energy_every = ${$(steps) / 100}
//
// Recognized keys:
//   topology, conf_file, energy_file
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

// Minimal recursive-descent evaluator for + - * / and parentheses over doubles.
struct ExprParser {
    const std::string &s;
    size_t i = 0;
    explicit ExprParser(const std::string &str) : s(str) {}
    void skip() { while (i < s.size() && std::isspace((unsigned char)s[i])) i++; }
    double parse() { double v = expr(); skip();
        if (i != s.size()) throw std::runtime_error("bad expression: " + s);
        return v; }
    double expr() {
        double v = term();
        for (;;) { skip();
            if (i < s.size() && s[i] == '+') { i++; v += term(); }
            else if (i < s.size() && s[i] == '-') { i++; v -= term(); }
            else break; }
        return v;
    }
    double term() {
        double v = factor();
        for (;;) { skip();
            if (i < s.size() && s[i] == '*') { i++; v *= factor(); }
            else if (i < s.size() && s[i] == '/') { i++; v /= factor(); }
            else break; }
        return v;
    }
    double factor() {
        skip();
        if (i < s.size() && s[i] == '+') { i++; return factor(); }
        if (i < s.size() && s[i] == '-') { i++; return -factor(); }
        if (i < s.size() && s[i] == '(') { i++; double v = expr(); skip();
            if (i < s.size() && s[i] == ')') i++; return v; }
        size_t start = i;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.' ||
               s[i] == 'e' || s[i] == 'E' ||
               ((s[i] == '+' || s[i] == '-') && i > start && (s[i-1] == 'e' || s[i-1] == 'E'))))
            i++;
        if (i == start) throw std::runtime_error("bad number in expression: " + s);
        return std::stod(s.substr(start, i - start));
    }
};

// Evaluate an oxDNA value to a number, resolving $(key) and ${ ... }.
inline double eval_num(const std::string &val,
                       const std::map<std::string, std::string> &kv, int depth = 0) {
    if (depth > 16) throw std::runtime_error("input: expression nested too deeply / cyclic");
    std::string e = trim(val);
    if (e.size() >= 3 && e[0] == '$' && e[1] == '{' && e.back() == '}')
        e = e.substr(2, e.size() - 3);            // strip ${ ... }
    // substitute $(name) with the numeric value of that key
    size_t p;
    while ((p = e.find("$(")) != std::string::npos) {
        size_t q = e.find(')', p);
        if (q == std::string::npos) throw std::runtime_error("input: unmatched $( in " + val);
        std::string name = trim(e.substr(p + 2, q - (p + 2)));
        auto it = kv.find(name);
        if (it == kv.end()) throw std::runtime_error("input: $(" + name + ") refers to a missing key");
        double sub = eval_num(it->second, kv, depth + 1);
        e = e.substr(0, p) + std::to_string(sub) + e.substr(q + 1);
    }
    return ExprParser(e).parse();
}

// Temperature parse following oxDNA: "<x>C" Celsius, "<x>K" Kelvin, else
// already in oxDNA units (1 unit = 3000 K, so 300 K ~ 0.1).
inline double parse_T(const std::string &v, const std::map<std::string, std::string> &kv) {
    std::string s = trim(v);
    if (s.empty()) throw std::runtime_error("empty temperature");
    char last = s.back();
    if (last == 'C' || last == 'c') return (eval_num(s.substr(0, s.size() - 1), kv) + 273.15) / 3000.0;
    if (last == 'K' || last == 'k') return eval_num(s.substr(0, s.size() - 1), kv) / 3000.0;
    return eval_num(s, kv);
}

} // namespace inp_detail

// Parse top-level "key = value" pairs, skipping comments (#...) and the
// contents of multi-line { ... } blocks (e.g. data_output_N). A value with
// balanced braces on its own line (e.g. a ${...} expression) is kept; a line
// that opens a block (net unbalanced '{') is not stored as a scalar.
inline std::map<std::string, std::string> parse_input_kv(const std::string &file) {
    std::ifstream f(file);
    if (!f) throw std::runtime_error("Cannot open input file: " + file);

    std::map<std::string, std::string> kv;
    int depth = 0;
    std::string line;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);

        int net = 0;
        for (char c : line) { if (c == '{') net++; else if (c == '}') net--; }

        std::string t = inp_detail::trim(line);
        if (depth == 0 && !t.empty()) {
            auto eq = t.find('=');
            if (eq != std::string::npos && net == 0) {   // balanced line -> scalar
                std::string key = inp_detail::trim(t.substr(0, eq));
                std::string val = inp_detail::trim(t.substr(eq + 1));
                if (!key.empty()) kv[key] = val;
            }
        }
        depth += net;
        if (depth < 0) depth = 0;
    }
    return kv;
}

inline void read_input(const std::string &file, SimConfig &cfg) {
    auto kv  = parse_input_kv(file);
    auto has = [&](const char *k) { return kv.count(k) > 0; };
    auto str = [&](const char *k) { return kv[k]; };
    auto num = [&](const char *k) { return inp_detail::eval_num(kv[k], kv); };

    if (!has("topology"))  throw std::runtime_error("input: missing mandatory key 'topology'");
    if (!has("conf_file")) throw std::runtime_error("input: missing mandatory key 'conf_file'");
    cfg.topology_file = str("topology");
    cfg.config_file   = str("conf_file");
    if (has("energy_file")) cfg.energy_file = str("energy_file");

    if (has("interaction_type")) {
        std::string it = inp_detail::lower(str("interaction_type"));
        cfg.model = (it == "dna2") ? 2 : 1;   // DNA / DNA1 -> 1, DNA2 -> 2
    }
    if (has("salt_concentration")) cfg.salt = num("salt_concentration");

    if (has("T"))                  cfg.T = inp_detail::parse_T(str("T"), kv);
    if (has("dt"))                 cfg.dt = num("dt");
    if (has("steps"))              cfg.nsteps = static_cast<long long>(num("steps"));
    if (has("verlet_skin"))        cfg.skin = num("verlet_skin");
    if (has("print_energy_every")) {
        int f = static_cast<int>(num("print_energy_every"));
        if (f > 0) cfg.output_freq = f;
    }
    if (has("seed"))               cfg.seed = static_cast<uint64_t>(num("seed"));

    // Thermostat: this code implements the Brownian ("John") refresh thermostat.
    bool thermo_on = false;
    if (has("thermostat")) {
        std::string th = inp_detail::lower(str("thermostat"));
        thermo_on = (th == "brownian" || th == "john");
    }
    if (has("newtonian_steps")) cfg.newtonian_steps = static_cast<int>(num("newtonian_steps"));
    if (has("diff_coeff"))      cfg.diff_coeff = num("diff_coeff");
    if (has("pt"))              cfg.pt = num("pt");
    if (!thermo_on) cfg.newtonian_steps = 0;   // NVE unless a supported thermostat is requested

    if (has("refresh_vel")) {
        std::string v = inp_detail::lower(str("refresh_vel"));
        cfg.refresh_vel = (v == "1" || v == "yes" || v == "true" || v == "on");
    }

    // Kokkos-specific extension (not a standalone-oxDNA key).
    if (has("timing")) {
        std::string v = inp_detail::lower(str("timing"));
        cfg.timing = (v == "1" || v == "yes" || v == "true" || v == "on");
    }
    if (has("lammps_overhead")) { std::string v = str("lammps_overhead"); cfg.lammps_overhead = (v=="1"||v=="yes"||v=="true"||v=="on"); }
}
