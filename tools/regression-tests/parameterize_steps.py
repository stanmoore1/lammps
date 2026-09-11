#!/usr/bin/env python3
"""Make the length of an example run settable from the command line.

Example inputs that run for minutes are the main cost of a full regression
sweep, but editing their step counts changes what the example does and
invalidates its reference log files.  Instead this script turns the literal
number into an index-style variable that keeps the original value as its
default:

    run             40000            ->   variable        nsteps index 40000
                                          ...
                                          run             ${nsteps}

An index-style variable is the only kind that a "-var" command-line switch
can override (see the note in doc/src/variable.rst), so the example behaves
exactly as before unless a test run asks for something shorter with
"-var nsteps 200".

Usage:
  parameterize_steps.py --check in.foo [in.bar ...]
  parameterize_steps.py --apply in.foo [in.bar ...]
"""

import argparse
import os
import re
import sys

# "run 10000", "run 10000 upto", "run ${nsteps} post no", ...
RUN_RE = re.compile(r'^(\s*run\s+)(\S+)(.*)$', re.IGNORECASE)
# "minimize 0.0 1.0e-8 1000 100000"
MIN_RE = re.compile(r'^(\s*minimize\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(.*)$', re.IGNORECASE)
# any variable definition, used to find where it is safe to insert ours
VAR_RE = re.compile(r'^\s*variable\s+(\S+)\s+(\S+)', re.IGNORECASE)

# a run of this many steps or fewer is already cheap and is left alone, and
# "run 0" is a single point evaluation that must not be turned into a real run
MIN_STEPS_TO_PARAMETERIZE = 10

# names follow the convention already used by the example inputs: a single run
# is "nsteps", an equilibration followed by production is "nequil"/"nprod"
NAMES_BY_COUNT = {1: ['nsteps'], 2: ['nequil', 'nprod']}


def run_targets(lines):
    """Return [(index, steps)] for the run commands worth parameterizing."""
    targets = []
    for i, line in enumerate(lines):
        match = RUN_RE.match(line)
        if match and match.group(2).isdigit():
            steps = int(match.group(2))
            if steps > MIN_STEPS_TO_PARAMETERIZE:
                targets.append((i, steps))
    return targets


def minimize_targets(lines):
    """Return [(index, maxiter, maxeval)] for minimize commands with literals."""
    targets = []
    for i, line in enumerate(lines):
        match = MIN_RE.match(line)
        if match and match.group(6).isdigit() and match.group(8).isdigit():
            targets.append((i, int(match.group(6)), int(match.group(8))))
    return targets


def names_for(count):
    if count in NAMES_BY_COUNT:
        return NAMES_BY_COUNT[count]
    return [f'nsteps{n + 1}' for n in range(count)]


def existing_variables(lines):
    return {m.group(1): m.group(2).lower() for m in
            (VAR_RE.match(line) for line in lines) if m}


def insert_before(lines, index, new_lines):
    """Insert new_lines above the command at index, skipping its comment block.

    A command is usually preceded by a blank line and often by a comment
    describing it; the definitions belong above both, so that the comment stays
    attached to the command it documents.  A blank line is added after the
    definitions only when the following line is not already blank, so that
    re-running this does not pile up empty lines.
    """
    at = index
    while at > 0 and (lines[at - 1].strip().startswith('#') or not lines[at - 1].strip()):
        at -= 1
    block = list(new_lines)
    if at < len(lines) and lines[at].strip():
        block.append('')
    return lines[:at] + block + lines[at:]


def parameterize(path, apply_changes):
    with open(path, 'r', errors='ignore') as src:
        lines = src.read().split('\n')

    runs = run_targets(lines)
    minimizes = minimize_targets(lines)
    if not runs and not minimizes:
        return None

    defined = existing_variables(lines)
    report = []

    # Collect every edit first, then apply them from the bottom of the file
    # upwards.  Inserting the variable definitions shifts every line below the
    # insertion point, so an edit must not be planned against line numbers that
    # an earlier edit has already invalidated.
    edits = []
    for (index, steps), name in zip(runs, names_for(len(runs))):
        edits.append(('run', index, name, steps, None))
    for index, maxiter, maxeval in minimizes:
        edits.append(('minimize', index, None, maxiter, maxeval))

    for kind, index, name, first, second in sorted(edits, key=lambda e: -e[1]):
        if kind == 'run':
            if name in defined:
                report.append(f'  run {first}: variable {name} already defined, skipped')
                continue
            match = RUN_RE.match(lines[index])
            lines[index] = f'{match.group(1)}${{{name}}}{match.group(3)}'
            lines = insert_before(lines, index,
                                  [f'variable        {name} index {first}'])
            report.append(f'  run {first} -> run ${{{name}}} (variable {name} index {first})')
        else:
            if ('maxiter' in defined) or ('maxeval' in defined):
                report.append(f'  minimize {first} {second}: variables already defined, skipped')
                continue
            match = MIN_RE.match(lines[index])
            lines[index] = (f'{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}'
                            f'{match.group(5)}${{maxiter}}{match.group(7)}${{maxeval}}'
                            f'{match.group(9)}')
            lines = insert_before(lines, index,
                                  [f'variable        maxiter index {first}',
                                   f'variable        maxeval index {second}'])
            report.append(f'  minimize ... {first} {second} -> ${{maxiter}} ${{maxeval}}')

    if apply_changes:
        with open(path, 'w') as dst:
            dst.write('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true',
                       help='report what would change without writing')
    group.add_argument('--apply', action='store_true',
                       help='rewrite the input scripts in place')
    parser.add_argument('inputs', nargs='+', help='input scripts to process')
    args = parser.parse_args()

    changed = 0
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f'{path}: not a file', file=sys.stderr)
            continue
        report = parameterize(path, args.apply)
        if not report:
            print(f'{path}: nothing to parameterize')
            continue
        changed += 1
        print(f'{path}:')
        for line in report:
            print(line)
    print(f'\n{changed} input script(s) {"changed" if args.apply else "would change"}')


if __name__ == '__main__':
    main()
