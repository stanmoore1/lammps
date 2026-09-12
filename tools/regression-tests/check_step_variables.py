#!/usr/bin/env python3
"""Check that a regression config can shorten every parameterized example.

parameterize_steps.py replaces a literal step count with an index-style
variable, and a test run turns the count down by naming that variable in the
config's "args" with a "-var" switch.  The two halves are easy to let drift
apart: an input with three runs gets nsteps1, nsteps2 and nsteps3, and if the
config only sets nsteps the input still runs at full length and times out with
no sign of why.  This reports any variable used by an example that the config
does not override.

Usage:
  check_step_variables.py [--examples examples] config.yaml [config.yaml ...]
"""

import argparse
import os
import re
import sys

# the variable names parameterize_steps.py generates
STEP_VARIABLE_RE = re.compile(r'\$\{(nsteps[0-9]*|nequil|nprod|maxiter|maxeval)\}')
ARGS_RE = re.compile(r'^\s*args:\s*"([^"]*)"', re.MULTILINE)
VAR_SWITCH_RE = re.compile(r'-var\s+(\S+)')


def variables_used(examples_dir):
    """Return {variable name: [input scripts using it]}."""
    used = {}
    for root, _, files in os.walk(examples_dir):
        for name in files:
            if not name.startswith('in.'):
                continue
            path = os.path.join(root, name)
            with open(path, errors='replace') as src:
                for match in STEP_VARIABLE_RE.finditer(src.read()):
                    used.setdefault(match.group(1), []).append(path)
    return used


def variables_set(config_file):
    with open(config_file) as src:
        args = ARGS_RE.search(src.read())
    if not args:
        return None
    return set(VAR_SWITCH_RE.findall(args.group(1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--examples', default='examples',
                        help='top level of the example tree (default: examples)')
    parser.add_argument('configs', nargs='+', help='regression config files to check')
    args = parser.parse_args()

    used = variables_used(args.examples)
    failed = False
    for config in args.configs:
        covered = variables_set(config)
        if covered is None:
            print(f'{config}: no "args:" setting found', file=sys.stderr)
            failed = True
            continue
        missing = sorted(set(used) - covered)
        if not missing:
            print(f'{config}: sets all {len(used)} step variables used by {args.examples}')
            continue
        failed = True
        print(f'{config}: does not set {len(missing)} step variable(s):')
        for name in missing:
            scripts = used[name]
            shown = ', '.join(scripts[:3])
            more = f' (and {len(scripts) - 3} more)' if len(scripts) > 3 else ''
            print(f'  {name}: used by {shown}{more}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
