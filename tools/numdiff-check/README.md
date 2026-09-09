# Pair style energy / force / virial consistency check

`check_pair_numdiff.py` verifies by numerical differentiation that a pair style's
forces are the negative gradient of its energy, and that its virial is the
negative strain derivative of its energy:

```
  F = -dE/dx            virial = -dE/d(strain)
```

## Why this exists

The force-style unit tests in `unittest/force-styles/` compare a style against
reference data that was generated with the same code.  A style whose energy is
not the integral of its force is perfectly self-consistent from their point of
view and passes indefinitely -- `pair lj/expand/sphere` shipped that way.  Only an
independent check of the derivative catches such a defect.

This is a developer tool, run on demand.  It is deliberately *not* part of the
unit test suite: the force check alone costs `6N` energy evaluations per style
per step size, which is far too much to add to every CI run.

## What it does

For every `*-pair-*.yaml` reference file in `unittest/force-styles/tests/` it

1. rebuilds the exact simulation the unit-test driver would set up -- the same
   `pre_commands`, input template, `pair_style`, `pair_coeff` and `post_commands`,
   in the same order as `init_lammps()` in `test_pair_style.cpp`;
2. adds `fix numdiff` and `fix numdiff/virial` (EXTRA-FIX package) at two step
   sizes, `delta` and `delta/2`, plus a `compute pressure` for the analytic
   virial;
3. compares both against the analytic force and virial, and classifies the style
   from how the discrepancy scales.

Step 3 is what makes the verdict trustworthy without per-style tolerances.  A
central difference has an `O(delta^2)` truncation error, so halving `delta`
should cut the discrepancy by about four and the Richardson extrapolation
`(4*D(d/2) - D(d))/3` should be accurate to `O(delta^4)`.  A residual that
survives the extrapolation is not truncation -- the energy and the force
genuinely disagree.  Both step sizes are measured in the *same* run, so the
comparison is not polluted by re-seeded random numbers or by thread-scheduling
differences between two processes.

The two commands order the stress tensor differently (`fix numdiff/virial` uses
`xx yy zz yz xz xy`, `compute pressure` uses `xx yy zz xy xz yz`).  The script
prints the raw components and swaps 4 and 6 in python, so the reordering lives in
exactly one place.

## Usage

```bash
# build LAMMPS with EXTRA-FIX (cmake/presets/most.cmake includes it)
cmake -S cmake -B build -C cmake/presets/gcc.cmake -C cmake/presets/most.cmake \
      -D BUILD_TOOLS=off -D DOWNLOAD_POTENTIALS=off -G Ninja
cmake --build build -j 4

# check every pair style the binary has, in every variant it supports
python3 tools/numdiff-check/check_pair_numdiff.py --lmp-bin build/lmp

# a KOKKOS build with the Serial backend checks the /kk variants
python3 tools/numdiff-check/check_pair_numdiff.py --lmp-bin build-kk/lmp --variants kk

# narrow down while debugging one style and keep the generated input
python3 tools/numdiff-check/check_pair_numdiff.py --lmp-bin build/lmp \
        --filter lj_expand --keep-all --verbose
```

`--variants` accepts `base`, `omp`, `opt`, `intel` and `kk`; by default every
variant whose package the binary contains is run.  Because `-sf X` falls back to
the plain style without saying so, a suffixed variant is only run when the binary
really contains that `<style>/<suffix>`, so the report never claims coverage it
does not have.  The `skip_tests` and `single_thread` entries of each YAML are
honoured.

`--intel` is pinned to `mode double`: the INTEL package defaults to mixed
precision, whose ~1e-7 step-size-independent error is indistinguishable from a
real inconsistency.

Useful options: `--max-diff-atoms N` bounds the force check to the first `N`
atoms (default 64), `--timeout` bounds a single run (x4 for decks tagged `slow`),
`--output` names the machine-readable YAML report, and `--keep-all` retains the
working directory of successful runs as well as failing ones.

## Verdicts

| verdict | meaning |
|---|---|
| `OK` | the extrapolated residual is below `--tol-ok`, or the error is at the round-off floor |
| `SUSPECT` | a small step-size-independent residual, between `--tol-ok` and `--tol-bad` |
| `INCONSISTENT` | a residual that survives Richardson extrapolation and does not shrink with `delta`: the energy and the force disagree |
| `NOISE_LIMITED` | the error *grew* when `delta` was halved and is large -- usually a discontinuity inside the displacement ball (an unshifted cutoff, a table kink, a branch on the distance) |
| `TRIVIAL` | no force and no energy, so there is nothing to compare |
| `ADVISORY` | matched a `soft` entry of the skip table: reported but never a failure |
| `ERROR` | the run failed or timed out; the reason is in the report |
| `SKIPPED` | not applicable, or a prerequisite is missing from the binary |

The exit status is non-zero when anything came back `INCONSISTENT`.

## What is skipped, and why

`F = -dE/dx` does not hold for every pair style.  The `SKIPS` table at the top of
the script lists each case with its reason; `hard` entries are never run, `soft`
ones are run and reported as `ADVISORY` but cannot fail the sweep:

- **stochastic** -- the DPD family and SDPD add a random force no energy accounts for;
- **velocity dependent** -- the SPH styles tally a literal zero energy while
  applying a real force;
- **tabulated** -- `pair table` splines energy and force from independent columns
  of a file, so the interpolated force is not the derivative of the interpolated
  energy by construction;
- **discontinuous by design** -- the `notaper` interlayer variants;
- **auxiliary degrees of freedom** (soft) -- charge equilibration and variable-charge
  models are solved by a fix, and `fix numdiff` re-invokes only the pair, bonded
  and kspace styles, so those charges stay frozen at the unperturbed geometry;
- **orientations** (soft) -- ellipsoid and spin styles, where a position-only
  sweep holds the orientation fixed.

Long-range styles are checked against their real-space part alone, which is
itself a conservative function of the coordinates: 82 of the 83 decks with a
`kspace_style` already set `kspace_modify compute no`, and the script adds it for
the remaining one.  Coulomb tabulation is likewise turned off (`pair_modify
table 0`, re-issued after `post_commands` for the decks that switch it back on),
because it interpolates energy and force independently.

## Reading the results

Some inconsistencies the sweep reports are documented behaviour or a limit of the
method, not defects, and the report is meant to be read with that in mind:

- `coul/charmm` applies the energy switching function to the Coulomb force
  without the product-rule term.  This reproduces what CHARMM historically did
  and is exactly why the `charmmfsw` / `charmmfsh` styles exist -- and indeed
  `lj/charmmfsw/coul/charmmfsw` comes back clean.
- Wolf and damped-shifted-force styles subtract a constant from the force by
  construction, so `F = -dE/dr` is deliberately violated.
- The `coul/long` family shows a residual around `1e-6`.  With tabulation off the
  energy uses the Abramowitz-Stegun rational approximation of `erfc`, while the
  force term is the exact derivative of the true `erfc`; the mismatch is the
  approximation error, not a coding defect.  It is an *absolute* deviation of
  about `3.6e-5`, so how large it looks depends on how big the forces are: the
  same number is `6e-6` relative for `coul/long` and `6e-8` for
  `lj/cut/coul/long`.
- Dipole and spin styles pass on the force channel but not on the virial: the
  strain derivative holds the dipole and spin vectors fixed, which is a different
  quantity from the virial LAMMPS tallies for a non-central force.
- Some styles evaluate their functions from an internal grid and then derive the
  force from the interpolated value, so the force is not the derivative of the
  interpolant: `edip` (grid density 8000 per distance unit, giving about `2e-4`,
  while the ungridded `edip/multi` is clean), `tersoff/table`, `vashishta/table`.

Two limits of the method itself are worth knowing:

- **A style that calls `domain->minimum_image()`** cannot be virial-checked.
  `fix numdiff/virial` strains the atom coordinates but not the box, so a pair
  reconstructed across a periodic boundary picks up an unstrained box length.
  `hbond/dreiding` does this; with a box three times larger, so no triplet
  wraps, its virial residual falls from `3.5e-1` to `1.2e-9`.  The force channel
  is unaffected, because displacing one atom by `delta` never changes which
  image is nearest.
- **An expensive potential may sit above the default step size's noise floor.**
  `pair pace` looks inconsistent at `delta = 1e-5` but converges cleanly at
  `1e-3` (`3.7e-8`, ratio `3.73`).  When a verdict comes with a ratio *below*
  one -- the error grew when the step was halved -- re-run with `--fdelta 1e-3
  --vdelta 1e-4` before believing it.

A real defect looks like `lj/expand/sphere` did before it was fixed: a relative
error of order one on *both* channels, with a step-size ratio of exactly `1.00`,
unchanged whatever step size you choose.
