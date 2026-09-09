#!/usr/bin/env python3
"""Check pair styles for energy / force / virial consistency by numerical differentiation.

The force-style unit tests in unittest/force-styles/ compare against stored
reference data that was generated with the very same code, so a pair style whose
energy is not the integral of its force passes them forever (pair lj/expand/sphere
shipped that way).  This script closes that gap without touching the unit tests.

For every pair style covered by a YAML reference file in
unittest/force-styles/tests/ it builds a small LAMMPS input from the same recipe
the unit-test driver uses, adds "fix numdiff" and "fix numdiff/virial", and
compares

    the analytic forces  against  -dE/dx
    the analytic virial  against  -dE/d(strain)

Each deck measures both derivatives at delta and at delta/2 in a single run.  A
central difference truncates as O(delta^2), so halving delta should cut the
discrepancy by about four, and the Richardson extrapolation (4*D(d/2)-D(d))/3
should be accurate to O(delta^4).  A residual that survives the extrapolation is
not truncation: the energy and the force genuinely disagree.  That scaling, not a
hand-tuned per-style tolerance, is what decides the verdict.

The same generated deck is replayed for the accelerated variants of the style
(/omp, /opt, /intel, /kk), so a defect present in only one variant is caught.

Usage:
    python3 tools/numdiff-check/check_pair_numdiff.py --lmp-bin build/lmp
    python3 tools/numdiff-check/check_pair_numdiff.py --lmp-bin build/lmp \\
            --variants base,omp --filter lj_ --keep-all --verbose

Requires: python3 with PyYAML, and a LAMMPS binary built with the EXTRA-FIX
package (which provides both numdiff fixes).
"""

import os
import re
import shutil
import signal
import subprocess
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

try:
    import yaml
except ImportError:
    sys.exit("This script requires PyYAML: pip install pyyaml")

LAMMPS_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
TESTS_DIR = os.path.join(LAMMPS_DIR, 'unittest', 'force-styles', 'tests')
POTENTIALS_DIR = os.path.join(LAMMPS_DIR, 'potentials')
PYTHON_DIR = os.path.join(LAMMPS_DIR, 'python')

# ----------------------------------------------------------------------------
# Styles for which F = -dE/dx does not hold by construction.  "hard" entries are
# never run, "soft" ones are run and reported but cannot fail the sweep.  Keys
# are YAML basenames or "re:" regular expressions matched against the pair style
# name (including hybrid sub-style names).
# ----------------------------------------------------------------------------

SKIPS = [
    ('re:^dpd(/|$)', 'hard',
     'dissipative and random pairwise terms that no energy accounts for'),
    ('re:^(sph|sdpd)/', 'hard',
     'force comes from a density and viscosity field; the tallied energy is a literal zero'),
    ('edpd', 'hard', 'temperature-dependent dissipative and random forces'),
    ('tdpd', 'hard', 'concentration-dependent dissipative and random forces'),
    ('re:^table( |$)', 'hard',
     'energy and force are splined from independent columns of a table file'),
    ('atomic-pair-sw_angle_table', 'hard', 'hybrid/overlay includes a table sub-style'),
    ('atomic-pair-threebody_table', 'hard', 'hybrid/overlay includes a table sub-style'),
    ('re:^ldd$', 'soft',
     'some local-density potential modes tabulate energy and force independently'),
    ('atomic-pair-reaxff-acks2_efield', 'hard',
     'fix efield adds a force in post_force that fix numdiff never re-evaluates'),
    ('re:^(granular|gran/|lubricate|brownian|rheo|bpm/|srp)', 'hard',
     'velocity-, history- or noise-dependent by construction'),
    ('manybody-pair-ilp-graphene-hbn_notaper', 'hard',
     'the energy is discontinuous at the cutoff by design (taper disabled)'),
    ('manybody-pair-kolmogorov_crespi_full_notaper', 'hard',
     'the energy is discontinuous at the cutoff by design (taper disabled)'),
    ('mol-pair-ldd-LD-noforce', 'hard', 'this potential mode deliberately applies no force'),
    ('re:/dielectric$', 'soft',
     'the force is scaled by the local permittivity and is not pairwise antisymmetric'),
    ('re:^(reaxff|comb|comb3)', 'soft',
     'charges are equilibrated by a fix, which fix numdiff never re-invokes'),
    ('re:qeq', 'soft', 'charges are equilibrated by a fix, which fix numdiff never re-invokes'),
    ('re:^coul/(streitz|ctip)', 'soft', 'variable-charge model; the charges stay frozen'),
    ('re:^hbond/dreiding', 'soft',
     'the style rebuilds its vectors with domain->minimum_image(), which the '
     'unstrained box of fix numdiff/virial cannot follow: only the force channel '
     'is meaningful here'),
    ('re:^(gayberne|resquared|ylz)', 'soft',
     'orientations are held fixed, so only the translational gradient is tested'),
    ('re:^spin/', 'soft',
     'spin styles split the interaction into a mechanical force and a magnetic field'),
]

# ----------------------------------------------------------------------------
# Accelerator variants.  "-sf X" silently falls back to the plain style when no
# /X variant exists, so a variant is only run when the binary really has it.
# ----------------------------------------------------------------------------

VARIANTS = {
    'base': {'suffix': None, 'package': None, 'args': [], 'threads': 1},
    # mixed precision produces a delta-independent error that looks exactly like
    # a real bug, so the INTEL package is pinned to double precision
    'intel': {'suffix': 'intel', 'package': 'INTEL',
              'args': ['-sf', 'intel', '-pk', 'intel', '0', 'mode', 'double'], 'threads': 1},
    'omp': {'suffix': 'omp', 'package': 'OPENMP',
            'args': ['-sf', 'omp', '-pk', 'omp', '4'], 'threads': 4},
    'opt': {'suffix': 'opt', 'package': 'OPT', 'args': ['-sf', 'opt'], 'threads': 1},
    'kk': {'suffix': 'kk', 'package': 'KOKKOS',
           'args': ['-k', 'on', 't', '1', '-sf', 'kk'], 'threads': 1},
}

# skip_tests tokens in the YAML files that rule out a variant
SKIP_TOKENS = {
    'omp': 'omp', 'extract_omp': 'omp', 'intel': 'intel',
    'kokkos_omp': 'kk', 'kokkos_serial': 'kk', 'kokkos_gpu': 'kk',
    'kokkos_omp_single': 'kk', 'kokkos_serial_single': 'kk', 'kokkos_gpu_single': 'kk',
    'kokkos_omp_mixed': 'kk', 'kokkos_serial_mixed': 'kk', 'kokkos_gpu_mixed': 'kk',
}

# the finite-difference step balances truncation, O(delta^2), against the
# round-off floor of the energy, O(eps*E/delta); all pair decks use one of these
DELTAS = {
    'real': (1.0e-5, 1.0e-6),
    'metal': (1.0e-5, 1.0e-6),
    'lj': (1.0e-5, 1.0e-6),
}

# smallest force / pressure magnitude worth calling a denominator
ABS_FLOOR = {'real': 1.0e-9, 'metal': 1.0e-11, 'lj': 1.0e-11}

NDC_RE = re.compile(r'^NDC\s+(\S+)\s+(.*)$', re.MULTILINE)

# ----------------------------------------------------------------------------


def block_lines(value):
    """Split a YAML block scalar into a list of non-empty, stripped lines."""
    if not value:
        return []
    return [line.strip() for line in str(value).split('\n') if line.strip()]


def build_configuration(lmp_bin):
    """Read the style and package inventory out of the binary's help output."""
    try:
        out = subprocess.run([lmp_bin, '-h'], capture_output=True, text=True,
                             timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as err:
        sys.exit("cannot run '{}': {}".format(lmp_bin, err))

    styles = {}
    packages = set()
    section = None
    for line in out.split('\n'):
        match = re.match(r'^\* (\w[\w ]*) styles:', line)
        if match:
            section = match.group(1).strip().lower()
            styles.setdefault(section, set())
            continue
        if line.startswith('Installed packages:'):
            section = 'packages'
            continue
        if line.startswith('*') or line.startswith('-'):
            section = None
            continue
        if not line.strip():
            continue
        if section == 'packages':
            packages.update(line.split())
        elif section in styles:
            # plugin styles are flagged with a trailing asterisk
            styles[section].update(tok.rstrip('*') for tok in line.split())
    return styles, packages


def classify_skip(name, style_names):
    """Return (scope, reason) if this deck is in the skip table, else None."""
    for key, scope, reason in SKIPS:
        if key.startswith('re:'):
            pattern = re.compile(key[3:])
            if any(pattern.search(s) for s in style_names):
                return scope, reason
        elif key == name or key in style_names:
            return scope, reason
    return None


def needs_atom_map(cfg, tests_dir):
    """True when neither the template nor the pre_commands provide an atom map.

    Atom::create_avec() turns the map on unconditionally for molecular atom
    styles, so this is only about atomic ones.
    """
    blob = '\n'.join(block_lines(cfg.get('pre_commands')))
    path = os.path.join(tests_dir, cfg['input_file'])
    if os.path.exists(path):
        with open(path) as handle:
            blob += '\n' + handle.read()
    if re.search(r'^\s*atom_modify\b.*\bmap\b', blob, re.M):
        return False
    return not re.search(r'^\s*atom_style\s+(full|molecular|template|bond|angle|hybrid)',
                         blob, re.M)


def make_deck(cfg, name, opts, fdelta, vdelta, tests_dir):
    """Build the LAMMPS input deck for one YAML reference file.

    The setup mirrors init_lammps() in unittest/force-styles/test_pair_style.cpp
    command for command, so the pair style sees the configuration its reference
    data was generated for.
    """
    pre = block_lines(cfg.get('pre_commands'))
    post = block_lines(cfg.get('post_commands'))
    injected = {}

    lines = [
        '# generated by tools/numdiff-check/check_pair_numdiff.py -- do not edit',
        '# source: ' + name + '.yaml',
        'variable newton_pair index on',
        'variable input_dir index ' + tests_dir,
    ]
    if needs_atom_map(cfg, tests_dir):
        # fix numdiff requires an atom map, and atom_modify has to come before
        # the simulation box is created
        lines.append('atom_modify map array')
        injected['atom_map'] = True

    lines += pre
    lines.append('include ' + os.path.join(tests_dir, cfg['input_file']))
    lines.append('pair_style ' + cfg['pair_style'].strip())
    lines += ['pair_coeff ' + line for line in block_lines(cfg.get('pair_coeff'))]
    lines += ['pair_modify table 0', 'pair_modify table/disp 0']
    lines += post

    # Coulomb tabulation splines energy and force separately, which produces a
    # delta-independent mismatch indistinguishable from a real bug.  Several
    # decks turn it back on in post_commands, so turn it off again afterwards.
    if any(re.match(r'^\s*pair_modify\b.*\btable(/disp)?\s+[1-9]', line) for line in post):
        lines += ['pair_modify table 0', 'pair_modify table/disp 0']
        injected['tabulation_disabled'] = True

    # numdiff cannot differentiate a grid-based solver, and fix numdiff/virial
    # strains the coordinates without straining the box, which is meaningless
    # for a reciprocal-space sum.  82 of the 83 kspace decks already do this.
    if any('kspace_style' in line for line in pre + post):
        if not any(re.search(r'kspace_modify\b.*\bcompute\s+no\b', line) for line in pre + post):
            lines.append('kspace_modify compute no')
            injected['kspace_off'] = True

    group = 'all'
    if opts.max_diff_atoms > 0:
        # the force check costs 6*N energy evaluations per delta; all decks have
        # consecutive IDs starting at 1, which fix numdiff requires anyway
        group = 'ndcgrp'
        lines.append('group ndcgrp id <= {}'.format(opts.max_diff_atoms))

    lines += [
        '',
        '# ---- energy / force / virial consistency check ----',
        'compute ndc_peall all pe',
        'compute ndc_pepair all pe pair',
        # fix numdiff differentiates the TOTAL potential energy, so the analytic
        # counterpart is the total force and the total non-kinetic virial
        'compute ndc_vir all pressure NULL virial',
        'fix ndc_f1 {} numdiff 1 {:.12g}'.format(group, fdelta),
        'fix ndc_f2 {} numdiff 1 {:.12g}'.format(group, 0.5 * fdelta),
        # fix numdiff/virial rejects any group but all
        'fix ndc_v1 all numdiff/virial 1 {:.12g}'.format(vdelta),
        'fix ndc_v2 all numdiff/virial 1 {:.12g}'.format(0.5 * vdelta),
        '',
        'variable ndc_a2 atom (f_ndc_f1[1]-fx)^2+(f_ndc_f1[2]-fy)^2+(f_ndc_f1[3]-fz)^2',
        'variable ndc_b2 atom (f_ndc_f2[1]-fx)^2+(f_ndc_f2[2]-fy)^2+(f_ndc_f2[3]-fz)^2',
        # Richardson extrapolation of the two central differences: O(delta^4)
        'variable ndc_r2 atom ((4.0*f_ndc_f2[1]-f_ndc_f1[1])/3.0-fx)^2+'
        '((4.0*f_ndc_f2[2]-f_ndc_f1[2])/3.0-fy)^2+'
        '((4.0*f_ndc_f2[3]-f_ndc_f1[3])/3.0-fz)^2',
        'variable ndc_fm atom fx^2+fy^2+fz^2',
        'compute ndc_ra {0} reduce ave v_ndc_a2'.format(group),
        'compute ndc_rb {0} reduce ave v_ndc_b2'.format(group),
        'compute ndc_rr {0} reduce ave v_ndc_r2'.format(group),
        'compute ndc_rf {0} reduce ave v_ndc_fm'.format(group),
        'compute ndc_mr {0} reduce max v_ndc_r2'.format(group),
        'compute ndc_mf {0} reduce max v_ndc_fm'.format(group),
        '',
        'thermo_style custom step pe press',
        'run 0 post no',
        'info system',
        '',
        'print "NDC natoms $(count(all):%.17g)"',
        'print "NDC ndiff $(count({}):%.17g)"'.format(group),
        'print "NDC pe_all $(c_ndc_peall:%.17g)"',
        'print "NDC pe_pair $(c_ndc_pepair:%.17g)"',
        'print "NDC frms $(sqrt(c_ndc_rf):%.17g)"',
        'print "NDC fmax $(sqrt(c_ndc_mf):%.17g)"',
        'print "NDC ferr1 $(sqrt(c_ndc_ra):%.17g)"',
        'print "NDC ferr2 $(sqrt(c_ndc_rb):%.17g)"',
        'print "NDC ferrR $(sqrt(c_ndc_rr):%.17g)"',
        'print "NDC ferrmaxR $(sqrt(c_ndc_mr):%.17g)"',
        # the two commands order the tensor differently: fix numdiff/virial is
        # xx yy zz yz xz xy, compute pressure is xx yy zz xy xz yz.  The raw
        # components are printed and reordered in python, so the swap lives in
        # exactly one place.
        'print "NDC vref ' + ' '.join(
            '$(c_ndc_vir[{}]:%.17g)'.format(i) for i in range(1, 7)) + '"',
        'print "NDC vnd1 ' + ' '.join(
            '$(f_ndc_v1[{}]:%.17g)'.format(i) for i in range(1, 7)) + '"',
        'print "NDC vnd2 ' + ' '.join(
            '$(f_ndc_v2[{}]:%.17g)'.format(i) for i in range(1, 7)) + '"',
        'print "NDC done 1"',
    ]
    return '\n'.join(lines) + '\n', injected


def run_deck(lmp_bin, deck, variant, workdir, timeout, tests_dir):
    """Run one deck in its own directory.  Returns (values, output, failure)."""
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, 'in.numdiff')
    with open(path, 'w') as handle:
        handle.write(deck)

    env = dict(os.environ)
    # about a quarter of the decks name a potential file that lives either in
    # potentials/ or next to the YAML files themselves
    env['LAMMPS_POTENTIALS'] = POTENTIALS_DIR + os.pathsep + tests_dir
    env['PYTHONPATH'] = os.pathsep.join(
        [tests_dir, PYTHON_DIR, env.get('PYTHONPATH', '')])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['OMP_PROC_BIND'] = 'false'
    env['OMP_NUM_THREADS'] = str(VARIANTS[variant]['threads'])

    cmd = [lmp_bin, '-in', 'in.numdiff', '-log', 'log.numdiff', '-nocite']
    cmd += VARIANTS[variant]['args']
    try:
        proc = subprocess.Popen(cmd, cwd=workdir, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            out = proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.communicate()
            return None, '', 'TIMEOUT after {:.0f} s'.format(timeout)
    except OSError as err:
        return None, '', 'cannot launch LAMMPS: {}'.format(err)

    values = {key: val.strip() for key, val in NDC_RE.findall(out)}
    if 'done' not in values:
        return None, out, lammps_error(out, proc.returncode)
    return values, out, None


def lammps_error(out, returncode):
    """Condense a failed run's output into one line."""
    if returncode is not None and returncode < 0:
        return 'died on signal {}'.format(-returncode)
    for line in out.split('\n'):
        if re.match(r'^ERROR(\s+on\s+proc\s+\d+)?:', line):
            return line.strip()[:200]
    for line in reversed(out.split('\n')):
        if line.strip():
            return 'incomplete output, last line: ' + line.strip()[:160]
    return 'no output'


def parse_values(values):
    """Turn the printed NDC lines into floats and vectors."""
    out = {}
    for key, raw in values.items():
        parts = raw.split()
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        out[key] = nums[0] if len(nums) == 1 else nums
    return out


def virial_errors(res):
    """Residuals of the numerical virial against compute pressure.

    fix numdiff/virial orders the tensor xx yy zz yz xz xy while compute
    pressure uses xx yy zz xy xz yz, so components 4 and 6 are swapped.
    """
    ref = res.get('vref')
    nd1 = res.get('vnd1')
    nd2 = res.get('vnd2')
    if not (ref and nd1 and nd2) or len(ref) != 6:
        return None
    order = [0, 1, 2, 5, 4, 3]
    ref = [ref[i] for i in order]
    err1 = err2 = errR = scale = 0.0
    for i in range(6):
        rich = (4.0 * nd2[i] - nd1[i]) / 3.0
        err1 += (nd1[i] - ref[i]) ** 2
        err2 += (nd2[i] - ref[i]) ** 2
        errR += (rich - ref[i]) ** 2
        scale += ref[i] ** 2
    root = lambda x: (x / 6.0) ** 0.5
    return root(err1), root(err2), root(errR), root(scale)


def channel_verdict(e1, e2, eR, scale, units, opts):
    """Classify one channel from its errors at delta, delta/2 and extrapolated."""
    denom = max(scale, ABS_FLOOR.get(units, 1.0e-11))
    rel = {'e1': e1, 'e2': e2, 'eR': eR, 'scale': scale,
           'rel1': e1 / denom, 'rel2': e2 / denom, 'relR': eR / denom}
    rel['ratio'] = e1 / e2 if e2 > 0.0 else float('inf')

    if scale <= ABS_FLOOR.get(units, 1.0e-11) and e1 <= ABS_FLOOR.get(units, 1.0e-11):
        return 'TRIVIAL', rel
    if rel['relR'] <= opts.tol_ok:
        return 'OK', rel
    ratio = rel['ratio']
    if ratio < 0.7:
        # halving delta made things worse: either the round-off floor of the
        # energy, or a discontinuity sitting inside the displacement ball (an
        # unshifted cutoff, a table kink, a branch on the distance).  A small
        # residual is the former and is nothing to report.
        return ('OK' if rel['relR'] <= opts.tol_bad else 'NOISE_LIMITED'), rel
    if 2.5 <= ratio <= 6.0:
        # truncation is behaving like O(delta^2).  A residual that survives the
        # extrapolation anyway is not truncation, but only once it is well clear
        # of the round-off floor -- an expensive potential evaluated at a step
        # size near its own noise level leaves a small one behind either way
        return ('INCONSISTENT' if rel['relR'] > opts.tol_bad else 'SUSPECT'), rel
    if ratio < 2.5:
        # the error does not shrink with delta at all
        return 'INCONSISTENT' if rel['relR'] > opts.tol_bad else 'SUSPECT', rel
    return 'SUSPECT', rel


def worst(*statuses):
    order = ['INCONSISTENT', 'ERROR', 'NOISE_LIMITED', 'SUSPECT', 'OK', 'TRIVIAL']
    for status in order:
        if status in statuses:
            return status
    return 'OK'


REPORT_ORDER = ['INCONSISTENT', 'SUSPECT', 'NOISE_LIMITED', 'ERROR', 'ADVISORY',
                'SKIPPED', 'TRIVIAL', 'OK']


def main():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--lmp-bin', required=True, help='LAMMPS binary to check')
    parser.add_argument('--tests-dir', default=TESTS_DIR,
                        help='directory holding the force-style YAML files')
    parser.add_argument('--work-dir', default='numdiff-check-work',
                        help='scratch directory for the generated decks')
    parser.add_argument('--filter', default='',
                        help='only check YAML files whose name matches this regex')
    parser.add_argument('--variants', default='',
                        help='comma separated accelerator variants to run '
                             '(default: all the binary supports): ' + ','.join(VARIANTS))
    parser.add_argument('--fdelta', type=float, default=0.0,
                        help='displacement for fix numdiff (default: per units)')
    parser.add_argument('--vdelta', type=float, default=0.0,
                        help='strain for fix numdiff/virial (default: per units)')
    parser.add_argument('--max-diff-atoms', type=int, default=64,
                        help='differentiate only the first N atoms (0 = all); the '
                             'force check costs 6N energy evaluations per delta')
    parser.add_argument('--timeout', type=float, default=300.0,
                        help='seconds allowed per run (x4 for decks tagged slow)')
    parser.add_argument('--tol-ok', type=float, default=1.0e-8,
                        help='relative error below which a style is reported OK')
    parser.add_argument('--tol-bad', type=float, default=1.0e-6,
                        help='relative error above which a delta-independent '
                             'residual is reported INCONSISTENT')
    parser.add_argument('--output', default='numdiff-results.yaml',
                        help='machine readable report')
    parser.add_argument('--keep-all', action='store_true',
                        help='keep the working directory of successful runs too')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print the LAMMPS output of failing runs and list OK styles')
    opts = parser.parse_args()

    lmp_bin = os.path.realpath(opts.lmp_bin)
    if not os.access(lmp_bin, os.X_OK):
        sys.exit("'{}' is not an executable".format(lmp_bin))
    tests_dir = os.path.realpath(opts.tests_dir)

    styles, packages = build_configuration(lmp_bin)
    pair_styles = styles.get('pair', set())
    if not pair_styles:
        sys.exit('could not read the pair style list from the binary')
    if 'numdiff' not in styles.get('fix', set()) \
            or 'numdiff/virial' not in styles.get('fix', set()):
        sys.exit('this binary has no fix numdiff; rebuild it with -D PKG_EXTRA-FIX=on')

    if opts.variants:
        variants = [v.strip() for v in opts.variants.split(',') if v.strip()]
        for variant in variants:
            if variant not in VARIANTS:
                sys.exit("unknown variant '{}'".format(variant))
    else:
        variants = [v for v in VARIANTS
                    if not VARIANTS[v]['package'] or VARIANTS[v]['package'] in packages]

    names = sorted(f[:-5] for f in os.listdir(tests_dir)
                   if f.endswith('.yaml') and '-pair-' in f)
    if opts.filter:
        pattern = re.compile(opts.filter)
        names = [n for n in names if pattern.search(n)]

    workroot = os.path.realpath(opts.work_dir)
    os.makedirs(workroot, exist_ok=True)

    print('# binary   : {}'.format(lmp_bin))
    print('# decks    : {} in {}'.format(len(names), tests_dir))
    print('# variants : {}'.format(', '.join(variants)))
    print()

    results = []
    report = open(opts.output, 'w')
    report.write('# generated by tools/numdiff-check/check_pair_numdiff.py\n')

    def emit(entry):
        results.append(entry)
        yaml.safe_dump([entry], report, default_flow_style=False, sort_keys=False)
        report.flush()

    def say(status, name, variant, detail):
        print('{:14s} {:<42s} {:<6s} {}'.format(status, name, variant, detail))

    for name in names:
        with open(os.path.join(tests_dir, name + '.yaml')) as handle:
            cfg = yaml.safe_load(handle)
        if not cfg or 'pair_style' not in cfg or 'input_file' not in cfg:
            continue

        tags = str(cfg.get('tags') or '').split()
        skipped_variants = {SKIP_TOKENS[tok]
                            for tok in str(cfg.get('skip_tests') or '').split()
                            if tok in SKIP_TOKENS}
        # every token of the pair_style line that the binary knows as a style;
        # this picks up hybrid sub-styles without having to parse their arguments
        style_names = [tok for tok in cfg['pair_style'].split() if tok in pair_styles]
        base = {'name': name, 'pair_style': cfg['pair_style'].strip(),
                'styles': style_names, 'tags': tags}

        skip = classify_skip(name, style_names + [cfg['pair_style'].split()[0]])
        if skip and skip[0] == 'hard':
            emit(dict(base, status='SKIPPED', reason=skip[1]))
            say('SKIPPED', name, '-', skip[1])
            continue
        advisory = bool(skip)

        missing = [p for p in block_lines(cfg.get('prerequisites'))
                   if len(p.split()) == 2 and p.split()[0] in styles
                   and p.split()[1] not in styles[p.split()[0]]]
        if missing:
            reason = 'not in this binary: ' + ', '.join(missing)
            emit(dict(base, status='SKIPPED', reason=reason))
            say('SKIPPED', name, '-', reason)
            continue

        timeout = opts.timeout * (4.0 if 'slow' in tags else 1.0)

        for variant in variants:
            suffix = VARIANTS[variant]['suffix']
            if suffix:
                if variant in skipped_variants:
                    continue
                if not any(s + '/' + suffix in pair_styles for s in style_names):
                    continue
            entry = dict(base, variant=variant, advisory=advisory)
            if advisory:
                entry['advisory_reason'] = skip[1]

            workdir = os.path.join(workroot, name, variant)
            # units decide the step size, and pre_commands may override the
            # template, so take them from a first pass and re-run if they differ
            units = "real"
            failure = None
            values = None
            for attempt in range(2):
                fdelta = opts.fdelta or DELTAS[units][0]
                vdelta = opts.vdelta or DELTAS[units][1]
                deck, injected = make_deck(cfg, name, opts, fdelta, vdelta, tests_dir)
                values, out, failure = run_deck(lmp_bin, deck, variant, workdir,
                                                timeout, tests_dir)
                if failure:
                    break
                found = re.search(r'^Units\s+=\s+(\S+)', out, re.M)
                seen = found.group(1) if found else units
                if seen == units or seen not in DELTAS:
                    break
                units = seen
            entry['units'] = units

            if failure:
                entry['status'] = 'ERROR'
                entry['reason'] = failure
                if opts.verbose:
                    print(out)
                say('ERROR', name, variant, failure)
                emit(entry)
                continue

            res = parse_values(values)
            entry['natoms'] = int(res.get('natoms', 0))
            entry['ndiff'] = int(res.get('ndiff', 0))
            entry['injected'] = injected or None
            # a pair bug is diluted when most of the energy is not the pair's
            pe_all, pe_pair = res.get('pe_all', 0.0), res.get('pe_pair', 0.0)
            entry['sensitivity'] = abs(pe_pair) / max(abs(pe_all), 1.0e-30)

            fstatus, frel = channel_verdict(res.get('ferr1', 0.0), res.get('ferr2', 0.0),
                                            res.get('ferrR', 0.0), res.get('frms', 0.0),
                                            units, opts)
            entry['force'] = frel
            vals = virial_errors(res)
            if vals:
                vstatus, vrel = channel_verdict(*vals, units=units, opts=opts)
                entry['virial'] = vrel
            else:
                vstatus = 'ERROR'
            status = worst(fstatus, vstatus)
            entry['status'] = status
            entry['channels'] = {'force': fstatus, 'virial': vstatus}

            detail = 'force rel={:.2e} ratio={:.2f} | virial rel={:.2e} ratio={:.2f}'.format(
                frel['relR'], frel['ratio'],
                entry.get('virial', {}).get('relR', float('nan')),
                entry.get('virial', {}).get('ratio', float('nan')))
            if advisory and status not in ('OK', 'TRIVIAL'):
                entry['status'] = 'ADVISORY'
                status = 'ADVISORY'
            if status != 'OK' or opts.verbose:
                say(status, name, variant, detail)
            emit(entry)

            if status in ('OK', 'TRIVIAL') and not opts.keep_all:
                shutil.rmtree(workdir, ignore_errors=True)

    report.close()

    print()
    print('# summary')
    counts = {}
    for entry in results:
        counts[entry['status']] = counts.get(entry['status'], 0) + 1
    for status in REPORT_ORDER:
        if status in counts:
            print('#   {:14s} {}'.format(status, counts[status]))
    print('# full report written to {}'.format(opts.output))
    if not opts.keep_all:
        print('# decks of non-OK runs kept under {}'.format(workroot))

    return 1 if counts.get('INCONSISTENT') else 0


if __name__ == '__main__':
    sys.exit(main())
