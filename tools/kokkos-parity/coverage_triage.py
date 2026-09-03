#!/usr/bin/env python3
"""Per-style coverage triage for the LAMMPS KOKKOS package.

Combines several sources of information into one table that tells which
KOKKOS styles are exercised by the test suite and how well:

  * the style inventory parsed out of the style macros in src/KOKKOS/*.h
  * line and branch coverage from a gcovr JSON report of src/KOKKOS
  * the YAML fixtures of unittest/force-styles/tests that reference a style,
    including whether they skip the "kokkos_serial" test case
  * optionally the result of the "kokkos_serial" gtest case per ctest test,
    read from a CTest JUnit XML report (ctest --output-junit)

Only the Python standard library is used.  Run with --help for options.
"""

import argparse
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# style inventory
# ---------------------------------------------------------------------------

# style macro name -> short category label used throughout the report
STYLE_MACROS = {
    "PairStyle": "pair",
    "BondStyle": "bond",
    "AngleStyle": "angle",
    "DihedralStyle": "dihedral",
    "ImproperStyle": "improper",
    "FixStyle": "fix",
    "ComputeStyle": "compute",
    "KSpaceStyle": "kspace",
    "AtomStyle": "atom",
    "RegionStyle": "region",
    "MinimizeStyle": "min",
    "IntegrateStyle": "integrate",
    "CommandStyle": "command",
    "NBinStyle": "nbin",
    "NPairStyle": "npair",
}

# the macros may span several lines (NPairStyle, NBinStyle do), so the scan
# runs over the whole file content instead of line by line
STYLE_RE = re.compile(r"\b(\w+Style)\(\s*([^,\s]+)\s*,\s*([A-Za-z_][\w:<>]*)")

# keyword suffixes that only select the device or host back end of a style
KK_VARIANT_SUFFIXES = ("/kk/device", "/kk/host")


class Style(object):
    """One KOKKOS style keyword with its files and variants."""

    def __init__(self, keyword, category, classname, header):
        self.keyword = keyword
        self.category = category
        self.classname = classname
        self.header = header
        self.source = None
        self.variants = set()
        self.base = base_style_name(keyword)

    def covfile(self):
        """File whose coverage represents this style."""
        return self.source if self.source else self.header


def base_style_name(keyword):
    """Strip the trailing /kk from a KOKKOS style keyword."""
    if keyword.lower().endswith("/kk"):
        stripped = keyword[:-3]
        if stripped:
            return stripped
    return keyword


def collapse_keyword(keyword):
    """Fold /kk/device and /kk/host keywords into the plain /kk keyword.

    Internal styles spell their keyword in upper case (NEIGH_HISTORY/KK),
    so the suffixes are matched without regard to case and the keyword
    itself keeps the case it was written in.
    """
    lowered = keyword.lower()
    for suffix in KK_VARIANT_SUFFIXES:
        if lowered.endswith(suffix):
            # replace "/kk/device" or "/kk/host" by the leading "/kk"
            return keyword[:-len(suffix) + 3]
    if lowered in ("kk/device", "kk/host"):
        return keyword[:2]
    return keyword


def clean_classname(name):
    """Drop template arguments from a class name captured by STYLE_RE."""
    pos = name.find("<")
    if pos >= 0:
        return name[:pos]
    return name


def read_text(path):
    """Read a file as text, never raising on odd bytes."""
    try:
        fp = open(path, "r", encoding="utf-8", errors="replace")
    except (IOError, OSError):
        return ""
    try:
        return fp.read()
    finally:
        fp.close()


def scan_styles(kokkos_dir):
    """Build the KOKKOS style inventory from the style macros in the headers."""
    styles = {}
    try:
        names = sorted(os.listdir(kokkos_dir))
    except OSError as err:
        sys.stderr.write("cannot read %s: %s\n" % (kokkos_dir, err))
        return styles

    for name in names:
        if not name.endswith(".h"):
            continue
        header = os.path.join(kokkos_dir, name)
        content = read_text(header)
        if not content:
            continue
        for match in STYLE_RE.finditer(content):
            macro, keyword, classname = match.groups()
            category = STYLE_MACROS.get(macro)
            if category is None:
                continue
            collapsed = collapse_keyword(keyword)
            key = (category, collapsed)
            style = styles.get(key)
            if style is None:
                style = Style(collapsed, category, clean_classname(classname),
                              header)
                source = os.path.join(kokkos_dir, name[:-2] + ".cpp")
                if os.path.isfile(source):
                    style.source = source
                styles[key] = style
            style.variants.add(keyword)
    return styles


# ---------------------------------------------------------------------------
# coverage data
# ---------------------------------------------------------------------------

class FileCoverage(object):
    """Line and branch coverage of a single source file."""

    def __init__(self, path):
        self.path = path
        self.lines_total = 0
        self.lines_covered = 0
        self.branches_total = 0
        self.branches_covered = 0
        self.partial_branch_lines = []
        self.uncovered_lines = []

    def line_pct(self):
        if self.lines_total == 0:
            return None
        return 100.0 * self.lines_covered / self.lines_total

    def branch_pct(self):
        if self.branches_total == 0:
            return None
        return 100.0 * self.branches_covered / self.branches_total


def load_gcovr_json(path, include_throw=False):
    """Parse a gcovr JSON report into FileCoverage objects.

    Returns two dicts, keyed by the report relative path and by the plain
    file name, so lookups work no matter which root gcovr was run with.
    """
    by_path = {}
    by_name = {}
    try:
        fp = open(path, "r", encoding="utf-8", errors="replace")
    except (IOError, OSError) as err:
        sys.stderr.write("cannot read gcovr JSON %s: %s\n" % (path, err))
        return by_path, by_name
    try:
        data = json.load(fp)
    except ValueError as err:
        sys.stderr.write("cannot parse gcovr JSON %s: %s\n" % (path, err))
        return by_path, by_name
    finally:
        fp.close()

    for entry in data.get("files", []):
        name = entry.get("file")
        if not name:
            continue
        cov = FileCoverage(name)
        uncovered = []
        for line in entry.get("lines", []):
            if line.get("gcovr/noncode"):
                continue
            number = line.get("line_number")
            count = line.get("count", 0)
            cov.lines_total += 1
            if count:
                cov.lines_covered += 1
            elif number is not None:
                uncovered.append(number)
            untaken = False
            for branch in line.get("branches", []):
                if branch.get("throw") and not include_throw:
                    continue
                cov.branches_total += 1
                if branch.get("count", 0):
                    cov.branches_covered += 1
                else:
                    untaken = True
            if untaken and count and (number is not None):
                cov.partial_branch_lines.append(number)
        cov.uncovered_lines = uncovered
        by_path[name] = cov
        by_name.setdefault(os.path.basename(name), cov)
    return by_path, by_name


def lookup_coverage(by_path, by_name, path):
    """Find the coverage record belonging to a source file path."""
    if path is None:
        return None
    normalized = path.replace(os.sep, "/")
    if normalized in by_path:
        return by_path[normalized]
    return by_name.get(os.path.basename(path))


def compress_ranges(numbers):
    """Turn a list of line numbers into a list of (first, last) pairs."""
    ranges = []
    for number in sorted(numbers):
        if ranges and (number == ranges[-1][1] + 1):
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return [(first, last) for first, last in ranges]


def format_ranges(ranges, limit=0):
    """Render (first, last) pairs as a compact 12,15-19,23 style string."""
    parts = []
    for first, last in ranges:
        if first == last:
            parts.append(str(first))
        else:
            parts.append("%d-%d" % (first, last))
    if limit and (len(parts) > limit):
        return ",".join(parts[:limit]) + ",... (+%d)" % (len(parts) - limit)
    return ",".join(parts)


# ---------------------------------------------------------------------------
# YAML fixture inventory
# ---------------------------------------------------------------------------

TOPLEVEL_STYLE_KEYS = {
    "pair_style": "pair",
    "bond_style": "bond",
    "angle_style": "angle",
    "dihedral_style": "dihedral",
    "improper_style": "improper",
    "kspace_style": "kspace",
}

KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(.*)$")
BLOCK_RE = re.compile(r"\|[0-9]*-?\s*$")
FIX_RE = re.compile(r"^\s*(fix|compute)\s+\S+\s+\S+\s+(\S+)")
ONEARG_RE = re.compile(r"^\s*(min_style|atom_style|run_style)\s+(\S+)")
REGION_RE = re.compile(r"^\s*region\s+\S+\s+(\S+)")
NUMBER_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


class Fixture(object):
    """One force-style YAML reference file."""

    def __init__(self, name):
        self.name = name
        self.skips_kokkos_serial = False
        self.styles = set()          # set of (category, style name) pairs


def parse_yaml_lite(text):
    """Small YAML subset reader: top level scalars and block scalars.

    The force-style reference files are machine generated and stick to a
    fixed shape, so a full YAML parser is not needed here.
    """
    scalars = {}
    blocks = {}
    lines = text.splitlines()
    index = 0
    count = len(lines)
    while index < count:
        line = lines[index]
        index += 1
        match = KEY_RE.match(line)
        if not match:
            continue
        key = match.group(1)
        rest = match.group(2).strip()
        if BLOCK_RE.search(rest):
            body = []
            while index < count:
                nxt = lines[index]
                if nxt.strip() and not nxt[:1].isspace():
                    break
                body.append(nxt)
                index += 1
            blocks[key] = body
        else:
            if rest.startswith("!"):
                rest = rest[1:].strip()
            if (len(rest) > 1) and rest.startswith('"') and rest.endswith('"'):
                rest = rest[1:-1]
            scalars[key] = rest
    return scalars, blocks


def substyles_of(spec):
    """Style names named by a style command argument list.

    The first word is always taken.  For hybrid styles every further word
    that does not look like a number is a candidate sub-style name.
    """
    words = spec.split()
    if not words:
        return []
    names = [words[0]]
    if (words[0] == "hybrid") or words[0].startswith("hybrid/"):
        for word in words[1:]:
            if NUMBER_RE.match(word):
                continue
            if word.startswith("$") or word.startswith("#"):
                continue
            names.append(word)
    return names


def scan_fixtures(tests_dir):
    """Build the map (category, style) -> list of YAML fixtures."""
    fixtures = []
    index = {}
    if not os.path.isdir(tests_dir):
        sys.stderr.write("no such fixture directory: %s\n" % tests_dir)
        return fixtures, index

    for name in sorted(os.listdir(tests_dir)):
        if not name.endswith(".yaml"):
            continue
        text = read_text(os.path.join(tests_dir, name))
        if not text:
            continue
        scalars, blocks = parse_yaml_lite(text)
        fixture = Fixture(name)
        if "kokkos_serial" in scalars.get("skip_tests", "").split():
            fixture.skips_kokkos_serial = True

        for key, category in TOPLEVEL_STYLE_KEYS.items():
            spec = scalars.get(key)
            if (spec is None) and (key in blocks):
                body = [item.strip() for item in blocks[key] if item.strip()]
                spec = body[0] if body else None
            if not spec:
                continue
            for style in substyles_of(spec):
                fixture.styles.add((category, style))

        for key in ("pre_commands", "post_commands"):
            for line in blocks.get(key, []):
                match = FIX_RE.match(line)
                if match:
                    fixture.styles.add((match.group(1), match.group(2)))
                    continue
                match = ONEARG_RE.match(line)
                if match:
                    category = match.group(1).replace("_style", "")
                    if category == "run":
                        category = "integrate"
                    fixture.styles.add((category, match.group(2)))
                    continue
                match = REGION_RE.match(line)
                if match:
                    fixture.styles.add(("region", match.group(1)))

        fixtures.append(fixture)
        for entry in fixture.styles:
            index.setdefault(entry, []).append(fixture)
    return fixtures, index


# ---------------------------------------------------------------------------
# CTest JUnit report
# ---------------------------------------------------------------------------

# ctest test name prefix -> YAML file name prefix, mirroring the name
# rewriting rules in unittest/force-styles/CMakeLists.txt
CTEST_PREFIXES = [
    ("MolPairStyle", "mol-pair-"),
    ("AtomicPairStyle", "atomic-pair-"),
    ("ManybodyPairStyle", "manybody-pair-"),
    ("EllipsoidPairStyle", "ellipsoid-pair-"),
    ("SpinPairStyle", "spin-pair-"),
    ("SPHPairStyle", "sph-pair-"),
    ("MesoPairStyle", "meso-pair-"),
    ("BondStyle", "bond-"),
    ("AngleStyle", "angle-"),
    ("DihedralStyle", "dihedral-"),
    ("ImproperStyle", "improper-"),
    ("KSpaceStyle", "kspace-"),
    ("FixTimestep", "fix-timestep-"),
    ("MinStyle", "min-"),
    ("OutputStyle", ""),
]

GTEST_RE = re.compile(r"^\s*\[\s*(OK|SKIPPED|FAILED)\s*\]\s+(\w+)\.(\w+)\b")

STATUS_RANK = {"none": 0, "skipped": 1, "ran": 2, "failed": 3}

GTEST_STATUS = {"OK": "ran", "SKIPPED": "skipped", "FAILED": "failed"}


def gtest_status(system_out, case_name):
    """Status of one gtest case inside captured ctest output."""
    status = "none"
    for line in system_out.splitlines():
        match = GTEST_RE.match(line)
        if not match:
            continue
        if match.group(3) != case_name:
            continue
        found = GTEST_STATUS[match.group(1)]
        if STATUS_RANK[found] > STATUS_RANK[status]:
            status = found
    return status


def ctest_name_to_yaml(name):
    """Guess the YAML file a ctest test name was generated from."""
    if ":" not in name:
        return None
    suite, rest = name.split(":", 1)
    for prefix, yaml_prefix in CTEST_PREFIXES:
        if suite == prefix:
            return yaml_prefix + rest + ".yaml"
    return rest + ".yaml"


def parse_junit(path, case_name):
    """Map YAML fixture name -> status of case_name in that ctest test."""
    result = {}
    try:
        tree = ET.parse(path)
    except (IOError, OSError, ET.ParseError) as err:
        sys.stderr.write("cannot parse JUnit XML %s: %s\n" % (path, err))
        return result
    for testcase in tree.getroot().iter("testcase"):
        name = testcase.get("name")
        if not name:
            continue
        chunks = []
        for out in testcase.iter("system-out"):
            if out.text:
                chunks.append(out.text)
        status = gtest_status("\n".join(chunks), case_name)
        yaml_name = ctest_name_to_yaml(name)
        if yaml_name is None:
            continue
        old = result.get(yaml_name, "none")
        if STATUS_RANK[status] > STATUS_RANK[old]:
            result[yaml_name] = status
    return result


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

class Row(object):
    """One line of the triage table."""

    def __init__(self, style):
        self.style = style
        self.coverage = None
        self.fixtures = []
        self.skipped = 0
        self.kk_status = "none"

    def sort_pct(self):
        if self.coverage is None:
            return -1.0
        pct = self.coverage.line_pct()
        return -1.0 if pct is None else pct

    def line_cell(self):
        cov = self.coverage
        if (cov is None) or (cov.lines_total == 0):
            return "n/a"
        return "%d/%d (%.1f%%)" % (cov.lines_covered, cov.lines_total,
                                   cov.line_pct())

    def branch_cell(self):
        cov = self.coverage
        if (cov is None) or (cov.branches_total == 0):
            return "n/a"
        return "%d/%d (%.1f%%)" % (cov.branches_covered, cov.branches_total,
                                   cov.branch_pct())

    def fixture_cell(self):
        if not self.fixtures:
            return "0"
        names = [item.name for item in self.fixtures[:2]]
        text = "%d: %s" % (len(self.fixtures), " ".join(names))
        if len(self.fixtures) > 2:
            text += " ..."
        return text

    def file_cell(self):
        path = self.style.covfile()
        return os.path.basename(path) if path else "n/a"


def build_rows(styles, by_path, by_name, fixture_index, junit_status):
    """Assemble and sort the table rows: category first, coldest first."""
    rows = []
    for key in sorted(styles):
        style = styles[key]
        row = Row(style)
        row.coverage = lookup_coverage(by_path, by_name, style.covfile())
        row.fixtures = fixture_index.get((style.category, style.base), [])
        row.skipped = len([f for f in row.fixtures if f.skips_kokkos_serial])
        status = "none"
        for fixture in row.fixtures:
            found = junit_status.get(fixture.name, "none")
            if STATUS_RANK[found] > STATUS_RANK[status]:
                status = found
        row.kk_status = status
        rows.append(row)
    rows.sort(key=lambda r: (r.style.category, r.sort_pct(), r.style.keyword))
    return rows


HEADERS = ["keyword", "category", "base", "file", "lines", "branches",
           "fixtures", "skip_kk_serial", "kk_case"]


def row_cells(row):
    return [row.style.keyword, row.style.category, row.style.base,
            row.file_cell(), row.line_cell(), row.branch_cell(),
            row.fixture_cell(), str(row.skipped), row.kk_status]


def source_line(src_dir, kokkos_dir, path, number):
    """Text of one source line, for annotating untaken branches."""
    cache = source_line.cache
    lines = cache.get(path)
    if lines is None:
        candidates = [path, os.path.join(src_dir, path),
                      os.path.join(kokkos_dir, os.path.basename(path))]
        text = ""
        for candidate in candidates:
            if os.path.isfile(candidate):
                text = read_text(candidate)
                break
        lines = text.splitlines()
        cache[path] = lines
    if 1 <= number <= len(lines):
        return lines[number - 1].strip()
    return ""


source_line.cache = {}


def collect_branch_report(rows, max_lines):
    """Per-file untaken-branch and uncovered-line details, de-duplicated."""
    seen = set()
    report = []
    for row in rows:
        cov = row.coverage
        if (cov is None) or (cov.lines_total == 0):
            continue
        if cov.path in seen:
            continue
        seen.add(cov.path)
        if not cov.partial_branch_lines and not cov.uncovered_lines:
            continue
        report.append((cov, cov.partial_branch_lines[:max_lines]))
    report.sort(key=lambda item: item[0].path)
    return report


def summarize(rows):
    """Per-category totals: style count, mean line coverage, cold styles."""
    order = []
    data = {}
    for row in rows:
        category = row.style.category
        if category not in data:
            data[category] = {"styles": 0, "pcts": [], "cold": 0, "nodata": 0}
            order.append(category)
        entry = data[category]
        entry["styles"] += 1
        cov = row.coverage
        if (cov is None) or (cov.lines_total == 0):
            entry["nodata"] += 1
            entry["cold"] += 1
            continue
        pct = cov.line_pct()
        entry["pcts"].append(pct)
        if pct == 0.0:
            entry["cold"] += 1
    summary = []
    for category in order:
        entry = data[category]
        pcts = entry["pcts"]
        mean = (sum(pcts) / len(pcts)) if pcts else None
        summary.append((category, entry["styles"], mean, entry["cold"],
                        entry["nodata"]))
    return summary


def mean_cell(mean):
    return "n/a" if mean is None else "%.1f%%" % mean


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def write_aligned(out, table):
    """Write a list of string rows as an aligned plain text table."""
    widths = [0] * len(table[0])
    for cells in table:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))
    for index, cells in enumerate(table):
        out.write("  ".join(cell.ljust(widths[i])
                            for i, cell in enumerate(cells)).rstrip() + "\n")
        if index == 0:
            out.write("  ".join("-" * width for width in widths) + "\n")


def render_text(out, rows, branch_report, summary, args, src_dir, kokkos_dir):
    out.write("KOKKOS style coverage triage\n")
    out.write("=" * 76 + "\n\n")
    write_aligned(out, [HEADERS] + [row_cells(row) for row in rows])

    out.write("\nUntaken branches\n")
    out.write("=" * 76 + "\n")
    if not branch_report:
        out.write("(no coverage data)\n")
    for cov, lines in branch_report:
        out.write("\n%s\n" % cov.path)
        if cov.uncovered_lines:
            out.write("  never executed: %s\n"
                      % format_ranges(compress_ranges(cov.uncovered_lines),
                                      args.max_ranges))
        for number in lines:
            text = source_line(src_dir, kokkos_dir, cov.path, number)
            out.write("  %6d  %s\n" % (number, text[:80]))
        extra = len(cov.partial_branch_lines) - len(lines)
        if extra > 0:
            out.write("  ... %d more lines with untaken branches\n" % extra)

    out.write("\nSummary per category\n")
    out.write("=" * 76 + "\n")
    out.write("%-12s %8s %12s %8s %8s\n"
              % ("category", "styles", "mean_lines", "cold", "no_data"))
    for category, count, mean, cold, nodata in summary:
        out.write("%-12s %8d %12s %8d %8d\n"
                  % (category, count, mean_cell(mean), cold, nodata))


def render_md(out, rows, branch_report, summary, args, src_dir, kokkos_dir):
    out.write("# KOKKOS style coverage triage\n\n")
    out.write("| " + " | ".join(HEADERS) + " |\n")
    out.write("|" + "|".join(["---"] * len(HEADERS)) + "|\n")
    for row in rows:
        out.write("| " + " | ".join(row_cells(row)) + " |\n")

    out.write("\n## Untaken branches\n")
    if not branch_report:
        out.write("\n(no coverage data)\n")
    for cov, lines in branch_report:
        out.write("\n### %s\n\n" % cov.path)
        if cov.uncovered_lines:
            out.write("never executed: `%s`\n\n"
                      % format_ranges(compress_ranges(cov.uncovered_lines),
                                      args.max_ranges))
        for number in lines:
            text = source_line(src_dir, kokkos_dir, cov.path, number)
            out.write("- %d: `%s`\n" % (number, text[:80].replace("`", "'")))
        extra = len(cov.partial_branch_lines) - len(lines)
        if extra > 0:
            out.write("- ... %d more lines with untaken branches\n" % extra)

    out.write("\n## Summary per category\n\n")
    out.write("| category | styles | mean_lines | cold | no_data |\n")
    out.write("|---|---|---|---|---|\n")
    for category, count, mean, cold, nodata in summary:
        out.write("| %s | %d | %s | %d | %d |\n"
                  % (category, count, mean_cell(mean), cold, nodata))


def render_csv(out, rows, branch_report, summary, args, src_dir, kokkos_dir):
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["section", "styles"])
    writer.writerow(HEADERS)
    for row in rows:
        writer.writerow(row_cells(row))
    writer.writerow([])
    writer.writerow(["section", "untaken_branches"])
    writer.writerow(["file", "line", "text"])
    for cov, lines in branch_report:
        for number in lines:
            text = source_line(src_dir, kokkos_dir, cov.path, number)
            writer.writerow([cov.path, number, text[:80]])
    writer.writerow([])
    writer.writerow(["section", "uncovered_ranges"])
    writer.writerow(["file", "ranges"])
    for cov, lines in branch_report:
        if cov.uncovered_lines:
            writer.writerow([cov.path,
                             format_ranges(compress_ranges(cov.uncovered_lines),
                                           args.max_ranges)])
    writer.writerow([])
    writer.writerow(["section", "summary"])
    writer.writerow(["category", "styles", "mean_lines", "cold", "no_data"])
    for category, count, mean, cold, nodata in summary:
        writer.writerow([category, count, mean_cell(mean), cold, nodata])


INVENTORY_HEADERS = ["keyword", "category", "base", "class", "header",
                     "source", "variants"]


def render_inventory(out, styles, fmt):
    table = []
    for key in sorted(styles):
        style = styles[key]
        table.append([style.keyword, style.category, style.base,
                      style.classname, os.path.basename(style.header),
                      os.path.basename(style.source) if style.source else "-",
                      " ".join(sorted(style.variants))])
    if fmt == "csv":
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(INVENTORY_HEADERS)
        for cells in table:
            writer.writerow(cells)
        return
    if fmt == "md":
        out.write("| " + " | ".join(INVENTORY_HEADERS) + " |\n")
        out.write("|" + "|".join(["---"] * len(INVENTORY_HEADERS)) + "|\n")
        for cells in table:
            out.write("| " + " | ".join(cells) + " |\n")
        return
    write_aligned(out, [INVENTORY_HEADERS] + table)
    out.write("\n%d style keywords in %d categories\n"
              % (len(table), len(set(cells[1] for cells in table))))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def default_src():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, os.pardir, os.pardir, "src"))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Triage table of KOKKOS style test coverage.")
    parser.add_argument("--src", default=None, metavar="DIR",
                        help="LAMMPS src directory (default: the src directory "
                             "of the repository holding this script)")
    parser.add_argument("--kokkos-dir", default=None, metavar="DIR",
                        help="KOKKOS package directory (default: <src>/KOKKOS)")
    parser.add_argument("--gcovr-json", default=None, metavar="FILE",
                        help="gcovr JSON report covering the KOKKOS sources")
    parser.add_argument("--tests-dir", default=None, metavar="DIR",
                        help="force-style YAML fixtures (default: "
                             "unittest/force-styles/tests)")
    parser.add_argument("--junit", default=None, metavar="FILE",
                        help="CTest JUnit XML from ctest --output-junit")
    parser.add_argument("--gtest-case", default="kokkos_serial", metavar="NAME",
                        help="gtest case looked for in the JUnit output "
                             "(default: kokkos_serial)")
    parser.add_argument("--format", default="text",
                        choices=("text", "md", "csv"),
                        help="output format (default: text)")
    parser.add_argument("-o", "--output", default=None, metavar="FILE",
                        help="write the report to FILE instead of stdout")
    parser.add_argument("--category", action="append", default=None,
                        metavar="NAME",
                        help="restrict the report to this style category "
                             "(may be repeated)")
    parser.add_argument("--max-branch-lines", type=int, default=20,
                        metavar="N",
                        help="untaken-branch lines listed per file "
                             "(default: 20)")
    parser.add_argument("--max-ranges", type=int, default=12, metavar="N",
                        help="uncovered line ranges listed per file "
                             "(default: 12)")
    parser.add_argument("--include-throw-branches", action="store_true",
                        help="also count the compiler generated exception "
                             "branches, which are never taken in normal runs")
    parser.add_argument("--inventory-only", action="store_true",
                        help="print the style inventory and exit")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    src_dir = os.path.abspath(args.src if args.src else default_src())
    if args.kokkos_dir:
        kokkos_dir = args.kokkos_dir
    else:
        kokkos_dir = os.path.join(src_dir, "KOKKOS")
    kokkos_dir = os.path.abspath(kokkos_dir)
    if args.tests_dir:
        tests_dir = args.tests_dir
    else:
        tests_dir = os.path.join(os.path.dirname(src_dir), "unittest",
                                 "force-styles", "tests")
    tests_dir = os.path.abspath(tests_dir)

    if not os.path.isdir(kokkos_dir):
        sys.stderr.write("no such KOKKOS directory: %s\n" % kokkos_dir)
        return 1

    styles = scan_styles(kokkos_dir)
    if not styles:
        sys.stderr.write("no style macros found in %s\n" % kokkos_dir)
        return 1
    if args.category:
        wanted = set(args.category)
        styles = dict((key, value) for key, value in styles.items()
                      if value.category in wanted)
        if not styles:
            sys.stderr.write("no styles in categories: %s\n"
                             % " ".join(sorted(wanted)))
            return 1

    if args.output:
        out = open(args.output, "w", encoding="ascii", errors="replace")
    else:
        out = sys.stdout

    try:
        if args.inventory_only:
            render_inventory(out, styles, args.format)
            return 0

        by_path = {}
        by_name = {}
        if args.gcovr_json:
            by_path, by_name = load_gcovr_json(args.gcovr_json,
                                               args.include_throw_branches)
        fixtures, fixture_index = scan_fixtures(tests_dir)
        junit_status = {}
        if args.junit:
            junit_status = parse_junit(args.junit, args.gtest_case)

        rows = build_rows(styles, by_path, by_name, fixture_index, junit_status)
        branch_report = collect_branch_report(rows, args.max_branch_lines)
        summary = summarize(rows)
        renderer = {"text": render_text, "md": render_md,
                    "csv": render_csv}[args.format]
        renderer(out, rows, branch_report, summary, args, src_dir, kokkos_dir)
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
