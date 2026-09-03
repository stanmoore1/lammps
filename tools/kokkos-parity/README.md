# KOKKOS parity tools

Helper scripts for the KOKKOS test-coverage effort: finding out which styles of
the KOKKOS package are exercised by the test suite, and how well.

## coverage_triage.py

Turns a gcovr coverage report of `src/KOKKOS` into a per-style triage table.
Python 3, standard library only, no external dependencies.

The script combines four sources of information:

1. **Style inventory.**  All style macros (`PairStyle(`, `FixStyle(`,
   `NPairStyle(`, ...) in `src/KOKKOS/*.h` are parsed, including the ones that
   span several lines.  The `/kk/device` and `/kk/host` keywords are folded into
   the plain `/kk` keyword, so every style appears once.  Each keyword is mapped
   to its category, its base style name (the keyword without the trailing
   `/kk`), its header, and the matching `.cpp` file if there is one.
2. **Coverage.**  Line and branch percentages per style, taken from the `.cpp`
   file of the style (header-only styles fall back to the `.h` file).  Compiler
   generated exception branches are ignored unless `--include-throw-branches` is
   given, because they are never taken in a normal run and would otherwise
   dominate the branch numbers.
3. **Fixtures.**  Every YAML file below `unittest/force-styles/tests` is scanned
   for the styles it exercises: the `pair_style:`, `bond_style:`, `angle_style:`,
   `dihedral_style:`, `improper_style:` and `kspace_style:` keys, plus the
   `fix`, `compute`, `min_style`, `atom_style` and `region` commands inside the
   `pre_commands:` and `post_commands:` blocks.  Sub-styles of `hybrid` and
   `hybrid/overlay` are matched as well.  Fixtures whose `skip_tests:` line
   contains the exact word `kokkos_serial` are counted separately (note that
   `kokkos_serial_single` skips only single-precision builds and does not
   count).
4. **Test results** (optional).  A CTest JUnit report holds one `<testcase>` per
   ctest test with the captured output in `<system-out>`.  That output is
   searched for the gtest result lines of the `kokkos_serial` case, so the table
   can say whether the KOKKOS case actually ran, was skipped, or failed.

## Generating the inputs

Build with coverage instrumentation and the KOKKOS package enabled, for example:

```bash
cmake -S cmake -B build-kk -C cmake/presets/gcc.cmake -C cmake/presets/most.cmake \
      -D PKG_KOKKOS=on -D Kokkos_ENABLE_SERIAL=on -D ENABLE_TESTING=on \
      -D CMAKE_CXX_FLAGS="--coverage -O0 -g" -D CMAKE_EXE_LINKER_FLAGS="--coverage" \
      -D DOWNLOAD_POTENTIALS=off -G Ninja
cmake --build build-kk -j 4
```

Run the tests, capturing the per-test output:

```bash
cd build-kk && ctest --output-junit kk-tests.xml
```

Collect the coverage data.  gcovr writes temporary files into the current
working directory, so run it from a scratch directory:

```bash
mkdir -p /tmp/gcovr-wd && cd /tmp/gcovr-wd
gcovr -r <lammps>/src --object-directory <lammps>/build-kk \
      --filter '.*src/KOKKOS/.*' --gcov-ignore-parse-errors \
      --json -o /tmp/kk-coverage.json
```

If gcovr cannot make sense of the object directory, `--gcov-executable gcov`
(or the version-matched `gcov-NN` of the compiler used for the build) usually
helps.

## Usage

```bash
tools/kokkos-parity/coverage_triage.py \
    --gcovr-json /tmp/kk-coverage.json \
    --junit build-kk/kk-tests.xml \
    --format text -o kk-triage.txt
```

Common options:

| Option | Meaning |
|---|---|
| `--src DIR` | LAMMPS `src` directory (default: the `src` of this repository) |
| `--kokkos-dir DIR` | KOKKOS package directory (default: `<src>/KOKKOS`) |
| `--gcovr-json FILE` | gcovr JSON report; without it all coverage cells read `n/a` |
| `--tests-dir DIR` | YAML fixtures (default: `unittest/force-styles/tests`) |
| `--junit FILE` | CTest JUnit XML from `ctest --output-junit` |
| `--gtest-case NAME` | gtest case to look for (default: `kokkos_serial`) |
| `--category NAME` | restrict the report to a category, may be repeated |
| `--format {text,md,csv}` | output format (default: `text`) |
| `-o FILE` | write to a file instead of standard output |
| `--inventory-only` | print just the style inventory and exit |
| `--max-branch-lines N` | untaken-branch lines listed per file (default: 20) |
| `--max-ranges N` | uncovered line ranges listed per file (default: 12) |
| `--include-throw-branches` | also count exception branches |

## Report layout

The report has three sections.

**Triage table**, sorted by category and then by line coverage ascending, so the
coldest styles of each category come first:

```
keyword    category  base    file                      lines           branches       fixtures              skip_kk_serial  kk_case
harmonic/kk  angle   harmonic  angle_harmonic_kokkos.cpp  0/191 (0.0%)  0/134 (0.0%)  2: angle-harmonic.yaml ...  0          skipped
```

`fixtures` gives the number of YAML files that reference the base style plus the
first two file names; `skip_kk_serial` how many of them skip the KOKKOS serial
case; `kk_case` is the aggregated gtest result (`ran`, `skipped`, `failed`, or
`none` when no JUnit report was given or no case was found).

**Untaken branches**, per source file with coverage data: the compressed ranges
of lines that never executed at all, followed by the lines that did execute but
still have a branch that was never taken, annotated with the source text.  These
are the concrete places a new test fixture has to reach.

**Summary per category**: number of styles, mean line coverage over the styles
that have data, how many styles are cold (0% or no data at all), and how many
have no coverage data.
