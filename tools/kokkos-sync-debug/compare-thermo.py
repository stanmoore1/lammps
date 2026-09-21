#!/usr/bin/env python3
"""Compare the thermo output of two LAMMPS logs (plain or gzipped).

    compare-thermo.py a.log b.log

Prints one word on stdout -- SAME, DIFFER or NO-OUTPUT -- and a line of detail
on stderr.  Callers test the word, so keep it a single token.

This is the only thermo extractor in this directory, on purpose.  There used to
be a second one inside diff-divergence.sh, and when the wall-clock column
problem below was fixed here and not there, PACKAGES/fep/CH4-CF4/bar10 was
written up as a bug it never was.  Anything that needs to compare thermo calls
this.

Three things it gets right that a plain diff of the two files does not, each
because getting it wrong produced a finding that was not real:

  - Wall-clock columns are dropped.  S/CPU and CPU differ on every pair of runs,
    so an input whose thermo_style names one differs from itself.  Three inputs
    here have such a column and are identical in every other.

  - Both thermo formats are read.  "thermo_style multi" prints a Step banner and
    named values rather than a row under a header, and 26 inputs use it; an
    extractor that returns nothing for both sides makes them all look equal.

  - Only the part both runs reached is compared.  A sweep that bounds runs with
    "timer timeout" stops them wherever the wall clock landed, so two runs of
    one input end at different steps -- 90000 and 110000, say.  Comparing the
    whole output calls that a difference, which it is not.  Comparing the common
    prefix asks the question actually worth asking: do they agree as far as both
    got?  The stderr line reports how many rows that was, because agreement over
    a short prefix is weaker evidence than agreement over a long one, and a run
    that died early shows up there as a very short one.
"""

import gzip
import re
import sys

TIMING = {"S/CPU", "CPU", "CPUleft", "CPULeft", "T/CPU", "Elapsed", "WallTime"}

# thermo_style multi: "------- Step 50 ----- CPU = 0.21 (sec) ------" then
# "Name = value Name = value ..." lines.  The banner's CPU is dropped with it.
MULTI_STEP = re.compile(r"^-+ Step\s+(\d+)\s+-+\s+CPU\s*=")
MULTI_PAIR = re.compile(r"([A-Za-z_][\w/]*)\s*=\s*(-?[\d.eE+-]+)")


def read(path):
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", errors="ignore") as fh:
            return fh.read().split("\n")
    except OSError:
        return None


def extract(path):
    """Return a list of per-step tuples, comparable across runs."""
    lines = read(path)
    if lines is None:
        return None
    out = []
    keep = None          # column indices of the current standard-style header
    ncol = 0
    multi = None         # values collected for the multi-style step in progress
    for line in lines:
        m = MULTI_STEP.match(line)
        if m:
            if multi is not None:
                out.append(tuple(multi))
            multi = [("step", m.group(1))]
            continue
        if multi is not None:
            pairs = MULTI_PAIR.findall(line)
            if pairs:
                multi.extend((k, v) for k, v in pairs if k not in TIMING)
                continue
            out.append(tuple(multi))
            multi = None
        if re.match(r"^\s+Step\s", line):
            cols = line.split()
            keep = [i for i, c in enumerate(cols) if c not in TIMING]
            ncol = len(cols)
            continue
        if line.startswith("Loop time"):
            keep = None
            continue
        if keep and re.match(r"^\s+[0-9]+\s+[-0-9.]", line):
            f = line.split()
            if len(f) == ncol:            # a partial row is not a thermo line
                out.append(tuple(f[i] for i in keep))
    if multi is not None:
        out.append(tuple(multi))
    return out


def errored(path):
    """The first LAMMPS error in the log, or None.  Not every early stop is an
    error -- a timer timeout is not -- so this asks for the message itself."""
    lines = read(path)
    if lines is None:
        return None
    for line in lines:
        if re.match(r"^ERROR(\s|:|\s+on\s+proc)", line):
            return line.strip()[:90]
    return None


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: compare-thermo.py a.log b.log")
    # One side stopping with an error is a difference in its own right, whether
    # or not it got far enough to print thermo.  Asked first because both of the
    # later answers would bury it: the prefix comparison would call a run that
    # errors after two rows equal to a run of 101 over those two, and a side
    # that errors before the first run command has no thermo at all and would
    # read as NO-OUTPUT.  Either way a deliberate new error on one build stops
    # showing up.
    ea, eb = errored(sys.argv[1]), errored(sys.argv[2])
    if (ea is None) != (eb is None):
        print("DIFFER")
        print("one side stopped with an error and the other did not: %s"
              % (ea or eb), file=sys.stderr)
        return

    a, b = extract(sys.argv[1]), extract(sys.argv[2])
    if not a or not b:
        print("NO-OUTPUT")
        print("no thermo extracted from %s" % ("either side" if not a and not b
              else sys.argv[1] if not a else sys.argv[2]), file=sys.stderr)
        return

    n = min(len(a), len(b))
    same = a[:n] == b[:n]
    print("SAME" if same else "DIFFER")
    note = "compared %d rows" % n
    if len(a) != len(b):
        note += " (of %d and %d -- the runs stopped at different points)" % (len(a), len(b))
    if ea and eb:
        note += "; both sides stopped with an error"
    if not same:
        first = next(i for i in range(n) if a[i] != b[i])
        note += "; first difference at row %d" % first
    print(note, file=sys.stderr)


main()
