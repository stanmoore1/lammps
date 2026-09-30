# Recipe: remove silent fp32/fp64 conversions in a LAMMPS KOKKOS file

Goal: eliminate all `-Wimplicit-float-conversion` and `-Wdouble-promotion`
warnings in the assigned `src/KOKKOS` files.  In the SINGLE build
`KK_FLOAT = KK_ACC_FLOAT = float`; in the MIXED build `KK_FLOAT = float` but
`KK_ACC_FLOAT = double`.  Make each conversion explicit, or keep the math in
the intended precision, WITHOUT changing algorithm results.  Every cast must
be a no-op in the default double build (`KK_FLOAT = KK_ACC_FLOAT = double`).

This is the canonical rule set.  It is recovered version 4 of the recipe
that was handed to every subagent (stanmoore1/private:lammps/kokkos-precision/history/tooling/RECIPE.md.v4), plus the
header and mixed-build sections of version 3, plus the conventions the
subagents converged on in their reports (stanmoore1/private:lammps/kokkos-precision/history/agents/notifications.md).

## Checker (use constantly; a file is done when it prints NOTHING)

`$KKP_TOOLS` is `tools/kokkos-precision` in the LAMMPS checkout; run from
anywhere inside the checkout:

    $KKP_TOOLS/chk.sh single src/KOKKOS/<TU>.cpp
    $KKP_TOOLS/chk.sh mixed  src/KOKKOS/<TU>.cpp

It prints every flagged `src/KOKKOS/<file>:line:col` reached by the TU (the
`.cpp` AND the `src/KOKKOS` headers it includes) plus any `ERROR:` lines.
BOTH single and mixed must be empty.  An `ERROR:` line means the file does not
compile; a file with errors is never clean, even if it lists no warnings.

To choose the cast direction you need the types clang reports ("'KK_FLOAT'
(aka 'float') to 'double'"); print the full warning text with source and
caret lines for the TU, or for one header it reaches:

    $KKP_TOOLS/warntext.sh single src/KOKKOS/<TU>.cpp
    $KKP_TOOLS/warntext.sh mixed  src/KOKKOS/<TU>.cpp <header>.h

## Rules (change ONLY lines the checker flags)

1. **Bare math on KK_FLOAT promotes.**  Unqualified `sqrt`, `pow`, `exp`,
   `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `erfc`,
   `fabs`, `cbrt`, `sinh`, `cosh`, `tanh`, `log10`, ... call the double C
   library version.  Qualify with `Kokkos::` (it has float overloads):
   `Kokkos::sqrt(x)`.
   - Caution: even `Kokkos::pow(x, 2)` computes in double, because the
     exponent is an **int**.  Make the exponent KK_FLOAT: `Kokkos::pow(x, static_cast<KK_FLOAT>(2))`.
   - `pow(int_or_double_base, 0.23)` on genuine double data is legitimate
     double math: leave `pow` alone and cast the result at the KK_FLOAT
     boundary (rule 6).

2. **Double literal or double constant mixed with KK_FLOAT** (`1.0`, `0.5`,
   `THIRD`, `MY_PIS`, `EWALD_P`, `EWALD_F`, `A1..A5`, `MY_EPSILON`, ...)
   promotes the KK_FLOAT operand.  Wrap the literal/constant:
   `static_cast<KK_FLOAT>(0.5)*y`, `c > static_cast<KK_FLOAT>(2.0)`.
   Constant sub-expressions fold at compile time, so
   `static_cast<KK_FLOAT>(1.0/3.0)` is fine.  Clang does NOT flag exactly
   representable literal assignments (`KK_FLOAT r = 0.0;`, `= 0.5;`) nor
   comparisons such as `? 1.0 : -1.0` that the checker does not report; only
   fix what the checker reports.

3. **Parent-class `double` scalar members** (declared in the non-Kokkos base
   class) used in KK_FLOAT math: make one local copy near the top of the
   function, `const KK_FLOAT foo_kk = static_cast<KK_FLOAT>(foo);`, and use
   it on the flagged lines.  The `_kk` suffix is the established convention.
   The copy is a new helper line, which the Discipline section allows; the
   copy itself needs the cast, or it is flagged in turn.
   Check the `_kokkos.h` header first: many such members (`qqrd2e`,
   `special_lj[]`, `ftm2v`, ...) are already KK_FLOAT in the Kokkos class,
   and then only their host-side initialization needs a cast.

4. **Accumulators use KK_ACC_FLOAT, everything else KK_FLOAT.**  If a KK_FLOAT
   value feeds a target that ACCUMULATES (`+=`, summed over a loop):
   `EV_FLOAT` fields (`evdwl`, `ecoul`, `v[6]`), per-atom views (`d_eatom`,
   `d_vatom`, `a_eatom`, `a_vatom`), force/torque views (`f`, `a_f`,
   `torque`), wrap it in `static_cast<KK_ACC_FLOAT>(...)`.  Decide by how the
   destination is USED and by its declared type in the `.h` file:
   `t_kkacc_*` views are KK_ACC_FLOAT, `t_kkfloat_*` views are KK_FLOAT.
   This cast is a no-op in single, so it cannot regress single.
   REVERSE case: a KK_ACC_FLOAT value (double in mixed) read into KK_FLOAT
   math or a KK_FLOAT destination: `static_cast<KK_FLOAT>(...)` so the math
   stays float.

5. **Host reductions into base-class `double`** (`eng_vdwl`, `eng_coul`,
   `virial[]`, `energy`, `result[]`, `double` reduction structs such as
   `s_CTEMP`, `double &` reduction arguments, anything that feeds
   `MPI_DOUBLE`): `static_cast<double>(...)`, NOT KK_ACC_FLOAT.

6. **Genuinely double sub-expressions** (double-only helpers, double table
   views, `Few<double,N>` domain math, bonus `shape[]` data) meeting KK_FLOAT
   at a boundary: keep the double math, make the widening of KK_FLOAT
   operands explicit (`static_cast<double>(x(i,0))`) where flagged, and wrap
   the whole sub-expression in `static_cast<KK_FLOAT>(...)` at the boundary.

7. **Double-only helpers.**  `MathSpecialKokkos::fm_exp(double)` has no float
   overload: `static_cast<KK_FLOAT>(MathSpecialKokkos::fm_exp(static_cast<double>(arg)))`.
   `MathSpecialKokkos::square()`, `cube()`, `powint()` are templated and
   already KK_FLOAT safe.  RNG calls (`drand()`, `normal()`, `uniform()`)
   return double: narrow the result, e.g.
   `static_cast<KK_FLOAT>(rand_gen.drand() - 0.5)` (the established idiom
   keeps the subtraction in double, then narrows once).

8. **Exchange and communication buffers are `double`.**  Pack with
   `static_cast<double>(...)`, unpack with `static_cast<KK_FLOAT>(...)` into
   KK_FLOAT views (or `static_cast<KK_ACC_FLOAT>(...)` when the unpack
   accumulates into a `t_kkacc_*` view, e.g. reverse communication of
   forces).  Never unpack coordinates through an integer type.

## Recurring conventions (from the subagents' judgment calls)

- **A literal takes the type of what it multiplies.**  `0.5*v[k]` with
  KK_FLOAT `v[]` becomes `static_cast<KK_FLOAT>(0.5)*v[k]`.  When the product
  then feeds a KK_ACC_FLOAT accumulator, widen the finished product:
  `ev.v[k] += static_cast<KK_ACC_FLOAT>(static_cast<KK_FLOAT>(0.5)*v[k]);`
  Do not cast the literal to KK_ACC_FLOAT while the other operand stays
  KK_FLOAT: in mixed that promotes the operand again.
- **The declared type of the left-hand side decides.**  A local declared
  `KK_FLOAT` (for example `ebondhalf`, `eanglethird`, `edihedralquarter`,
  `uCG_i`) gets KK_FLOAT casts even if it later feeds an accumulator; do not
  store a KK_ACC_FLOAT expression into it.  Views such as `uCG`, `rho`,
  `dpdTheta`, `d_result` are `t_kkfloat_*`: their `+=` uses KK_FLOAT.
- **In `MAX(1.0 - x*x, 0.0)` both literals need the cast.**  The ternary inside `MAX`
  re-promotes the result to double through the `0.0` branch.
- **Leave unflagged lines alone**, e.g. the plain `ev.v[N] += v[N]` in the
  `newton_bond` branch next to a flagged `ev.v[N] += 0.5*v[N]`.  A later fix
  can expose a new flag on a neighboring line (e.g. `phi *= -1.0` after
  fixing `dx > 0.0`); fix it when the checker reports it.
- **DualView host copies over parent double arrays are double.**
  `k_cutsq.view_host()(i,j) = cutone*cutone` needs no cast; only the KK_FLOAT
  stack copies (`m_cutsq[i][j]`) and KK_FLOAT device views do.  A pre-existing
  `static_cast<KK_FLOAT>` into such a double view is wrong and is removed.
- **Host-side double math stays unqualified.**  Bare `erfc`, `sqrt`, `cos`
  on double base-class data in `init_one()`, `coeff()`, `compute()` host
  code stays as is; cast only at the narrowing boundary.  Use `Kokkos::`
  in device kernels on KK_FLOAT operands.
- **Runtime-indexed double parameters** (`d_params[i].rm`, `param.lam1`,
  base-class `Param` structs with `double` members, `s_coeff[k]` arrays)
  cannot have a single `_kk` copy: cast at the boundary,
  `static_cast<KK_FLOAT>(d_params[i].rm)`.  Frequently reused scalars from
  such a struct may still get a local `_kk` copy.
- **File-scope constants** (a new helper line, see Discipline): a
  `constexpr` KK_FLOAT constant at file scope, e.g.
  `static constexpr KK_FLOAT SMALL = static_cast<KK_FLOAT>(0.001);`, avoids
  repeating the same cast when a double constant is used on many flagged
  lines.  *Learned in later rounds (Sep 2026):* the CG-DNA port uses
  `constexpr KK_FLOAT MY_PI_KK = static_cast<KK_FLOAT>(MY_PI);` in
  `pair_oxdna_coaxstk_kokkos.cpp`; in a post-compaction run a pattern
  rewriter turned an unrelated `using MathConst::MY_PI;` line into a
  reference to a `MY_PI_KK` that did not exist in that file.  Use such a
  constant only where it is actually declared, and never edit `using`
  declarations.
- **Textually identical lines may need different fixes** (a host function
  on `double*` and a device kernel on KK_FLOAT views; a non-instantiated
  template copy).  Edit by line number, never with a global replace.
- **Double reduction locals** that only participate in double math and
  `MPI_DOUBLE` reductions (e.g. `fdotf` in `min_fire`) were retyped to
  `double` in the original work where the declaration itself was flagged.
  If the declaration is not flagged, this is a retyping proposal: report it,
  do not apply it (see Discipline).
- **Whole-expression wrapping at an accumulator.**  For
  `a_f(j,0) -= f1[0] + f3[0]` wrap the whole sum once, instead of casting
  one operand and leaving a promotion on the other.
- **Macros**: fix a narrowing that appears at every use of a file-local macro
  (e.g. the `saru` RNG macro in `pair_dpd_kokkos.cpp`) at its definition.
- **Coulomb table lookup**: `union_int_float_t::f` is a real `float` by design
  (coulomb table bit trick): use `static_cast<float>`/`static_cast<double>`
  there, not KK_FLOAT.  Its warning in the double build is out of scope.
- **Report, do not fix, latent bugs** found in passing (e.g. an MPI_DOUBLE
  reduction into a KK_FLOAT scalar, coordinates unpacked through
  `static_cast<tagint>`).  They belong in a separate change.

## HEADER FILES

A header's warnings appear only when a `.cpp` that includes it is compiled,
and they are attributed to the HEADER path.  `chk.sh` prints all flagged
`src/KOKKOS` files reached by the TU; restrict to one header with a third
argument:

    $KKP_TOOLS/chk.sh single src/KOKKOS/<including_tu>.cpp pair_tip4p_kokkos.h

Edit the HEADER file itself.  What a header reports varies from TU to TU,
since each TU instantiates its own set of templates, so check a header
through more than one including TU (the prompt names them;
`$KKP_TOOLS/hdrorigin.sh <header>.h` lists the including TUs with their
counts), keep every previously clean TU at zero, and rely on the full
rebuild for the final word.  One agent owns a shared header; other agents
ignore its warnings.

## MIXED-BUILD PASS (KK_ACC_FLOAT = double)

In mixed, KK_FLOAT is float but KK_ACC_FLOAT is double, so KK_FLOAT values
meeting a KK_ACC_FLOAT target now promote.  The DOMINANT fix is rule 4: wrap
the KK_FLOAT value feeding a KK_ACC_FLOAT accumulator (EV_FLOAT
`evdwl`/`ecoul`/`v[6]`, per-atom `d_eatom`/`d_vatom`, `a_f` force sums,
`s_EV_FLOAT` fields, KK_ACC_FLOAT locals such as `fxtmp`) in
`static_cast<KK_ACC_FLOAT>(...)`.  This is a no-op in single, so it will not
regress single.  Also watch for KK_ACC_FLOAT (double) narrowing back to
KK_FLOAT (`a_rho[i] += rhotmp` where `rhotmp` is a KK_ACC_FLOAT sum and
`rho` a KK_FLOAT view; integrators reading `f`/`torque` into `v`/`omega`):
cast the KK_ACC_FLOAT source down, `static_cast<KK_FLOAT>(f(i,0))`, so the
multiply stays in float.  Warnings may name the target type as plain
`double` rather than `KK_ACC_FLOAT` when it is reached through an atomic or
scatter view; the declared view type in the `.h` decides.
Always re-run BOTH checkers: a fix for one precision must not regress the
other.

## OUT OF SCOPE

- `-Wimplicit-int-float-conversion` (int, `bigint`, `tagint` to float or
  double), e.g. `double delta = update->ntimestep - update->beginstep;`, and
  any other warning that also appears in the default double build.  Confirm
  with a DOUBLE_DOUBLE compile:
      KKP_WFLAGS='Wimplicit-int-float-conversion|Wimplicit-float-conversion|Wdouble-promotion' \
        $KKP_TOOLS/chk.sh double src/KOKKOS/<TU>.cpp
  If it is reported there, it is not a precision-mode issue: leave it.
- Kokkos library headers (`lib/kokkos/...`), other third-party code in `lib/`,
  non-KOKKOS sources, and `-Wall`/`-Wextra`/`-pedantic` warnings.
- REAXFF and ML-IAP (skipped by instruction; ML-IAP is double-only).

## Discipline

- Change ONLY lines the compiler flags; never cast pre-emptively, and never
  edit an unflagged line "for consistency".  This was an explicit rule of
  the original work, and it holds even when a neighboring unflagged line
  looks similar.
- The one exception: a fix of a flagged line may require a NEW helper line
  that did not exist before, such as the `_kk` local copy of rule 3 or a
  file-scope `constexpr` KK_FLOAT constant.  Adding such a line is allowed;
  list every added line in the report.
- Retyping an existing unflagged declaration (e.g. a local `v0..v5` or a
  double reduction local) to remove many flags at once is NOT covered by the
  exception.  Do not do it on your own: stop and report it as a proposal
  with the line numbers; the main thread decides and asks the user if
  needed.  (In the original work this was allowed only in a few nested
  prompts of the mixed pass, and each case was reported.)
- Re-run the checker after each batch of edits; confirm the count drops and
  NO new warnings or ERROR lines appear.
- Preserve formatting, indentation, and trailing comments; keep lines within
  the file's existing length conventions.
- Never replace an expression wholesale: an edit that empties or rewrites a
  right-hand side (e.g. `static_cast<KK_FLOAT>()`) is a bug even if the
  warning disappears.
- Casts must be no-ops in the double build.  Call out anything that is not
  bit-identical in single or mixed (e.g. moving an RNG cast, forming a
  virial product in float instead of double).

## Reference already-cleaned files

`src/KOKKOS/meam_funcs_kokkos.h` (first exemplar: `Kokkos::sqrt`/`pow`,
`_kk` copies, `fm_exp`), `src/KOKKOS/pair_lj_cut_coul_cut_kokkos.cpp`,
`src/KOKKOS/pair_coul_wolf_kokkos.cpp` (erfc/EWALD constants),
`src/KOKKOS/pair_lj_cut_coul_long_kokkos.cpp`,
`src/KOKKOS/dihedral_harmonic_kokkos.cpp` and `dihedral_class2_kokkos.cpp`
(accumulator patterns, `MAX` literals), `src/KOKKOS/fix_langevin_kokkos.cpp`
and `fix_gjf_kokkos.cpp` (RNG narrowing, `_kk` copies),
`src/KOKKOS/fix_setforce_kokkos.cpp` (per-atom force views).

Report the final single AND mixed counts per file (both must be 0) and
every judgment call.
