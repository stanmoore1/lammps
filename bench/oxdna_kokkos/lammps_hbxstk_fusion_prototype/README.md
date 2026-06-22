# Prototype: fused oxDNA hbond+xstk Kokkos kernel + `acosf` (FP32)

This directory holds a self-contained prototype (as a patch) of two
GPU-oriented optimizations for the **LAMMPS-KOKKOS** oxDNA pair styles, plus the
Kokkos-Serial verification data that was used to confirm correctness.

The patch applies on top of
[`lrussell676/lammps@oxdna3KK`](https://github.com/lrussell676/lammps/tree/oxdna3KK)
(base commit `dc8b51b`, "update last_allocate condition to use
neighbor->lastcall"). It is **not** part of the standalone `bench/oxdna_kokkos`
code — it modifies the in-tree `src/KOKKOS` oxDNA pair styles of that branch.

## The two optimizations

### #1 — Fuse the hbond + xstk screened-pair kernels

`oxdna2/hbond` (F1 / hydrogen bonding) and `oxdna2/xstk` (F2 / cross stacking)
both operate on the **same** base-site separation
`delr_hb = (x_a + 0.4·nx_a) − (x_b + 0.4·nx_b)` and share the **same six
interaction angles** `theta1, theta2, theta3, theta4, theta7, theta8`. In the
`oxdna3KK` branch they run as two separate edge-parallel ("screened pair")
kernels that each:

* re-load the per-atom local frames (`nx`, `nz`) for `a` and `b`,
* re-compute the base-site separation, its norm and `delr_hb_norm`,
* re-compute the six direction cosines and their six `acos()` angles,

over the **same** `d_pairs_screened` list built by `fix oxdna/npair/kk`.

The prototype adds a fused kernel `TagPairOxdnaHbXstkFused` (hosted in
`pair_oxdna_hbond_kokkos`) that computes the shared geometry and the six
`acosf()` angles **once**, then evaluates both interactions:

* HB branch: `F1` (hbond radial cutoff) × six `F4` (hbond angular params).
* XST branch: `F2` (xstk radial cutoff) × six `F4` (xstk angular params, with
  the `theta4/7/8` → `pi − theta` symmetrization).

**Cutoffs (checked carefully):** hbond and xstk use *different* radial cutoffs
(`cut_hb_*` vs `cut_xst_*`), but both act on the *same* base-site distance
`r_hb`, and both already iterate the *same* shared screened pair list (a COM
superset, `r_screen = 2.0`). The fusion keeps each term self-cutting through its
own `F1`/`F2`, so the differing cutoffs are honored — no shared early-return is
introduced. When fusion is on, the `xstk` style skips its own screened pass and
the hbond style tallies the combined energy (total PE unchanged; only the
per-style energy attribution differs).

### #2 — `acos` → `acosf` in the screened-pair kernels

The screened/GPU kernels already use `sqrtf`/`expf` FP32 intrinsics. The six
per-pair `acos()` calls in the screened operators of `oxdna2/hbond`,
`oxdna2/xstk` and `oxdna2/coaxstk` are switched to `acosf()` for consistency.
The per-atom (host) operators are left on double `acos()`.

## Verification harness (prototype only, default OFF)

The screened/GPU pair path is normally gated to device execution spaces, so it
is never exercised by the Kokkos Serial backend. Two environment variables
(added by the patch) make Serial verification possible without touching normal
GPU execution or the default host path:

| variable | effect |
| --- | --- |
| `OXDNA_KK_FORCE_SCREENED=1` | run the screened/GPU pair path on the host |
| `OXDNA_KK_FUSE_HBXSTK=1`    | enable the fused hbond+xstk kernel |

## Reproducing

```bash
# apply on a fresh oxdna3KK checkout
git clone -b oxdna3KK https://github.com/lrussell676/lammps lammps-oxdna3KK
cd lammps-oxdna3KK
git am < 0001-fuse-hbond-xstk-and-acosf.patch

# build (Kokkos Serial)
cmake -S cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DPKG_KOKKOS=on -DKokkos_ENABLE_SERIAL=on \
  -DPKG_MOLECULE=on -DPKG_ASPHERE=on -DPKG_CG-DNA=on
cmake --build build -j

cd examples/PACKAGES/cgdna/examples/lj_units/oxDNA2/duplex2
RUN="../../../../../../../build/lmp -k on t 1 -sf kk -pk kokkos newton on neigh half -in in.duplex2"

$RUN                                                   # per-atom double baseline
OXDNA_KK_FORCE_SCREENED=1 $RUN                         # separate screened kernels (FP32+acosf)
OXDNA_KK_FORCE_SCREENED=1 OXDNA_KK_FUSE_HBXSTK=1 $RUN  # FUSED kernel
```

## Results (duplex2, `run 1000`, total potential energy)

Captured traces are in this directory:
`baseline_peratom_double.txt`, `screened_separate_acosf.txt`, `screened_fused.txt`.

| step | per-atom (double) | separate screened (FP32+acosf) | **fused** | fused − separate | fused − per-atom |
| ---- | ----------------- | ------------------------------ | --------- | ---------------- | ---------------- |
| 0    | -21.0482844625272 | -21.0482842632964 | -21.0482842632964 | 0 | +2.0e-07 |
| 200  | -21.0198586323319 | -21.0198585645771 | -21.0198585645771 | 0 | +6.8e-08 |
| 500  | -20.9759517611524 | -20.9759517952759 | -20.9759517952759 | 0 | -3.4e-08 |
| 1000 | -20.9025739544833 | -20.9025742482550 | -20.9025742482550 | 0 | -2.9e-07 |

* **Fusion is exact:** the fused kernel reproduces the two separate screened
  kernels **bit-for-bit** at every printed step.
* **`acosf`/FP32 is benign:** the screened path (and the fused kernel) tracks the
  original double-precision per-atom path to ~1e-7 over 1000 steps — only the
  expected FP32-intrinsic difference, no drift.

## Notes / next steps for productionizing

* The fused kernel currently tallies the combined hbond+xstk energy under the
  hbond style. For a per-style energy/virial breakdown, split the tally.
* `OXDNA_KK_FORCE_SCREENED` is purely a verification aid; in production the
  fused kernel would simply replace the two screened kernels on the device path,
  gated by a normal pair-style/package option rather than an env var.
* `coaxstk` was intentionally **not** fused: it uses the stacking-site
  separation (a different base vector), so it shares no geometry with hbond/xstk.
  It does receive the `acosf` change (#2).
