# KOKKOS static analysis triage

Triage of the CodeChecker / clang-tidy run published at
<https://download.lammps.org/analysis/index.html>.

Every finding below was re-checked by reading the source on the `bugfixes`
branch (head `22d3173c7`), not on the tree the analyzer ran against.  The
analyzer tree is older, so line numbers in the report do not line up; findings
were matched by content.

## Dedup

| measure | count |
|---|---|
| raw findings | 7162 |
| unique report hash | 6570 |
| unique (file, line, message) | 7054 |
| unique (file, checker) | 1291 |
| distinct checkers | 38 |
| distinct root-cause idioms | ~25 |
| files touched | 613 |

7156 of 7162 are under `src/KOKKOS/`; the remaining six are in headers pulled in
from `src/`, `src/BOCS/`, `src/ML-IAP/` and `src/GRANULAR/`.  The raw-to-unique
inflation comes from two sources: templates instantiated for both `LMPHostType`
and `LMPDeviceType`, and headers re-analyzed once per including translation unit.

Deduplicated severity split: 6222 STYLE, 475 LOW, 357 MEDIUM.  Nothing was
reported as HIGH or CRITICAL.

## Verdict summary

The run is dominated by mechanical style noise.  **Three checkers account for
6183 of 7162 findings (86%)** and none of them describes a defect.  After
screening, the actionable set is 11 items, none of which is a live crash or
wrong-results bug on `bugfixes`.

The one genuinely serious defect the analyzer found (a null-pointer dereference
in `pppm/kk`) is **already fixed** on `bugfixes` by commit `490ba8b9d`.

---

## Tier 1 -- confirmed, act on these

Ranked by severity.

### 1. MEDIUM -- `remap_kokkos.cpp`: `malloc(0)` result treated as an allocation failure

`optin.portability.UnixAPI`, "Call to 'malloc' has an allocation size of 0 bytes".

In the collective (`Alltoallv`) branch of `remap_3d_create_plan_kokkos()` the
guard is `if (nsend || nrecv)`, so a rank that only receives enters the block
with `nsend == 0`.  It then calls `malloc(nsend*sizeof(int))` for
`plan->send_offset` and `malloc(nsend*sizeof(struct pack_plan_3d))` for
`plan->packplan`, and the very next statement is

```cpp
if (plan->send_offset == nullptr || plan->send_size == nullptr ||
    plan->sendcnts == nullptr || plan->sdispls == nullptr ||
    plan->packplan == nullptr) return nullptr;
```

C leaves `malloc(0)` free to return a null pointer, in which case plan creation
aborts on a rank that is otherwise perfectly healthy.  glibc returns a unique
non-null pointer, so this is latent on the platforms we test.

Fix: size these with `nsend*sizeof(...) + 1`, or skip the null check for the
zero-length allocations.

**Also fix the CPU sibling.**  `src/KSPACE/remap.cpp` has the identical pattern.
It already carries `+ 1` on the `commringlen` allocations but not on the two
`nsend` ones, so the same latent failure exists there.

### 2. MEDIUM -- `mliap_model_python_kokkos.cpp`: type punning through `void *`

`bugprone-casting-through-void`, 4 sites (destructor, `read_coeffs`,
`connect_param_counts`, and one more).

```cpp
auto nontemplated_this = static_cast<MLIAPModelPythonKokkosDevice*>((void*)this);
```

`MLIAPModelPythonKokkosDevice` is not a typedef -- it is a distinct empty class
*derived from* `MLIAPModelPythonKokkos<LMPDeviceType>`:

```cpp
class MLIAPModelPythonKokkosDevice : public MLIAPModelPythonKokkos<LMPDeviceType> {};
```

`MLIAPModelPythonKokkos<LMPHostType>` is explicitly instantiated under
`LMP_KOKKOS_GPU`, so on a GPU build `this` can point at an object that is not a
`MLIAPModelPythonKokkosDevice` and is not even related to it.  The cast is
undefined behaviour.

Most of the Cython glue only uses the pointer as an opaque identity key
(`int(<uintptr_t> c_model)`), which is harmless.  But
`load_from_python()` in `mliap_model_python_couple_kokkos.pyx` calls
`lmp_model.connect_param_counts()` through it, executing the
`LMPDeviceType` instantiation's member function on a possibly host-instantiated
object.  It happens to work today only because `MLIAPModelPython` is the first
base and its layout does not depend on `DeviceType`.

Fix: make the couple API take an opaque `void *` handle, or at minimum use
`reinterpret_cast` and document why the layout assumption holds.

### 3. LOW/MEDIUM -- `DynamicalMatrixKokkos::setup()` and `ThirdOrderKokkos::setup()` are unreachable

`bugprone-derived-method-shadowing-base-method`, 2 sites.

`DynamicalMatrix::setup()` is **not** virtual.  `DynamicalMatrixKokkos::command()`
delegates:

```cpp
void DynamicalMatrixKokkos::command(int narg, char **arg)
{
  atomKK->sync(Host, X_MASK|RMASS_MASK|TYPE_MASK);
  DynamicalMatrix::command(narg, arg);
}
```

and `DynamicalMatrix::command()` calls `setup()`, which binds statically to the
base.  `DynamicalMatrixKokkos::setup()` is therefore dead code.  `ThirdOrder`
is identical.

Functional impact today is nil: the sync-critical work lives in the virtual
`update_force()` override, which the base `setup()` does call, and
`update_force()` manages `lmp->kokkos->auto_sync` itself.  In fact the dead
`setup()` contains a *duplicate* `force->pair->compute()` block that would
double-count pair forces if it ever ran.

Fix: delete both overrides (they are redundant with `update_force()`), or make
the base `setup()` virtual and drop the duplicated compute block.  Leaving an
unreachable method that looks like an override is the trap here.

### 4. LOW -- `KokkosBase` / `KokkosBaseFFT`: virtual functions, public non-virtual destructor

`clang-diagnostic-non-virtual-dtor` (2) + `cppcoreguidelines-virtual-class-destructor` (2).

No live defect: both are mixins reached only via
`dynamic_cast<KokkosBase *>(...)`, and the objects are always owned and deleted
through `Fix *` / `Pair *` / `Region *`.  Worth hardening anyway --
`virtual ~KokkosBase() = default;` costs nothing and removes the trap for the
next person who writes `delete kkbase;`.

### 5. LOW -- `comm_kokkos.cpp`: `count_bonus` read and discarded

`clang-diagnostic-unused-but-set-variable`.

```cpp
int count = k_count.view_host()(0);
int count_bonus = k_count.view_host()(1);   // never read
...
if (count >= (int)k_exchange_sendlist_bonus.view_host().extent(0)) {   // should be count_bonus
```

Not a correctness bug: bonus atoms leaving a subdomain are a subset of atoms
leaving, so `count_bonus <= count`, and the bonus list is pre-sized to at least
the main sendlist -- the check is conservative in the safe direction.  It is
still a copy-paste residue that grows the bonus lists more often than needed and
leaves a dead variable.

### 6. LOW -- `comm_kokkos.cpp`: dead `mlo` / `mhi` declarations in `borders_device()`

`clang-diagnostic-unused-but-set-variable`.  `CommKokkos::borders()` routes
`comm_style multi` to `CommBrick::borders()`, so the device path never needs the
per-collection cutoffs.  Delete the declarations.

### 7. LOW -- `fix_shardlow_kokkos.h`: 12 virtual overrides not marked `override`

`modernize-use-override`, 17 findings, 12 of them in this one header (plus
`fix_rigid_small_kokkos.h` and `fix_wall_flow_kokkos.h`).  The header still uses
the pre-C++11 `virtual void init();` form.  Without `override` a signature drift
in `Fix` silently creates a new function instead of an override, and the style
stops being called.  This is the one style checker in the run that buys real
compiler checking.

### 8. LOW -- global `using` in headers

`google-global-names-in-headers`, 10 findings.

* `meam_dens_final_kokkos.h`, `meam_dens_init_kokkos.h`, `meam_force_kokkos.h`,
  `meam_funcs_kokkos.h`, `meam_impl_kokkos.h`, `meam_setup_done_kokkos.h`:
  `using namespace LAMMPS_NS;` and `using namespace MathSpecialKokkos;` at
  global scope.  Blast radius is small (only the MEAM translation units include
  them) but it is a genuine hygiene defect.
* `kokkos_type.h:1203-1205`: `using LAMMPS_NS::bigint; ... tagint; ... imageint;`
  at global scope in a header that **every** KOKKOS translation unit includes.
  Same class of problem, much wider reach.

### 9. LOW -- `pair_uf3_kokkos.cpp`: inconsistent signed/unsigned comparison

`clang-diagnostic-sign-compare`.  Line 784 casts (`(int)d_neighbors_short.extent(0) != ignum`),
line 787 does not (`d_numneigh_short.extent(0) != ignum`).  Benign while
`ignum >= 0`; fix for consistency.  The other 8 sign-compare findings, in
`atom_vec_kokkos.cpp` and `comm_kokkos.cpp`, are the same shape
(`extent()` vs. an `int` buffer count) and equally benign.

### 10. LOW -- leftovers

* `pair_sw_kokkos.cpp` and `pair_sw_mod_kokkos.cpp`: duplicate
  `#include "neighbor.h"` (`readability-duplicate-include`).
* `compute_orientorder_atom_kokkos.cpp`: unused `using MathSpecial::factorial;`
* `pair_pod_kokkos.cpp`: unused `using MathSpecial::powint;`
  (`misc-unused-using-decls`).  Worth removing rather than ignoring -- the
  KOKKOS rules explicitly forbid host-only `powint()` in device code, so a live
  `using` for it is a loaded gun.
* `fix_qeq_reaxff_kokkos.cpp`: commented-out `// calculate_Q();` should be
  deleted per the project's own review rules.
* `misc-unconventional-assign-operator` (6): `SplineInterpolatorKokkos::operator=`
  returns `void` in `pair_pace_kokkos.h` and `pair_pace_extrapolation_kokkos.h`.
* `FixRigidSmallKokkos::unpack_exchange_kokkos()` ignores its
  `ExecutionSpace space` parameter and syncs to `DeviceType` instead.  Benign,
  because `CommKokkos` calls the instantiation matching `space`, but the
  parameter is misleading and other KOKKOS fixes honour it.

### 11. Note, not a defect -- two conflicting file-scope enums with the same names

`readability-enum-initial-value` pointed at
`src/KOKKOS/fix_wall_gran_kokkos.cpp:23`:

```cpp
enum{XPLANE=0,YPLANE=1,ZPLANE=2,ZCYLINDER,REGION};      // REGION == 4
```

while `src/GRANULAR/fix_wall_gran.cpp` has

```cpp
enum {NOSTYLE=-1,XPLANE=0,YPLANE=1,ZPLANE=2,REGION=3};  // REGION == 3, no ZCYLINDER
```

**Verified not a bug.**  `FixWallGranKokkos` inherits from `FixWallGranOld`, the
KOKKOS-package-local copy of the old style, whose enum
(`fix_wall_gran_old.cpp:53`) matches the KOKKOS one.  Recorded here only because
two same-named, differently-valued enums for the same concept in one package is
exactly the sort of thing that bites during a future merge.

---

## Already fixed on `bugfixes`

### HIGH -- `PPPMKokkos` had no `reset_grid()` override: null-pointer dereference

`bugprone-derived-method-shadowing-base-method` on `PPPMKokkos::compute_rho_coeff`.

On the analyzed tree, `PPPM::reset_grid()` was not overridden by `PPPMKokkos`.
`fix balance` and the `balance` command call `force->kspace->reset_grid()`, which
reached `PPPM::reset_grid()` on a `PPPMKokkos` object.  Every other callee there
is virtual and resolved correctly, but `compute_rho_coeff()` is **not** virtual,
so `PPPM::compute_rho_coeff()` ran and wrote `rho_coeff[l][m]` / `drho_coeff[l-1][m]`.
`PPPMKokkos::allocate()` allocates the `k_rho_coeff` dual view and never touches
the base raw pointers, which stay `nullptr` from the `PPPM` constructor.

That is `fix balance` + `kspace_style pppm/kk` -> segfault.

Commit `490ba8b9d` ("KOKKOS: fix pppm, tip4p and the phonon drivers") adds
`void reset_grid() override;` with a Kokkos-aware body.  Confirmed fixed; the
remaining three PPPM shadowing findings (`setup_triclinic`,
`compute_gf_ik_triclinic`, `poisson_ik_triclinic`) are reached only from
`PPPM::setup()` and `PPPM::poisson_ik()`, both of which `PPPMKokkos` overrides.

---

## Tier 2 -- screened out as false positives

Verified individually, not dismissed by category.

| checker | raw | verdict |
|---|---|---|
| `readability-redundant-typename` | 3943 | Valid only under C++20. `cmake/CMakeLists.txt:156` forces `CMAKE_CXX_STANDARD 20` when `PKG_KOKKOS` is on, so the check is technically right -- but this is ~3.9k mechanical edits with zero functional value that break any KOKKOS build pinned to C++17. **Suppress the check.** 55% of the entire run. |
| `modernize-use-using` | 1301 | `typedef` -> `using`. Style only. |
| `modernize-use-auto` | 1018 | Style only. |
| `performance-unnecessary-value-param` | 301 | Not actionable. The flagged parameters are Kokkos `DualView`/`View` handles in the `KokkosBase` virtual interface (`pack_exchange_kokkos`, `pack_forward_comm_kokkos`, ...). By-value is the Kokkos idiom and the signatures are fixed by the base-class contract -- one cannot change without every implementer changing in lockstep. Cost is one atomic refcount bump on host-side setup paths, never in a kernel. |
| `misc-override-with-different-visibility` | 159 | FP as a defect class. KOKKOS subclasses group their overrides into one access section; access is checked statically at the base, so no call site breaks. Roughly a third are widening (protected -> public), harmless by definition. |
| `clang-diagnostic-float-conversion` | 135 | All intentional truncation. Largest single idiom is `int step = MAX(DELTA, nmax*0.01);` in `AtomVec*Kokkos::grow()`, copied across 13 files -- a 1% growth heuristic. The rest are Kokkos View extents computed from a double expression. `region_block_kokkos.h`'s `double->bool` is `inside_face()`, which returns `double` but only ever `0` or `1` (the CPU sibling returns `int`); behaviourally identical, worth narrowing the return type for clarity. |
| `readability-qualified-auto` | 64 | Style only. |
| `misc-use-internal-linkage` | 44 | Kokkos functor structs at namespace scope; could move into anonymous namespaces. Style only. |
| `bugprone-derived-method-shadowing-base-method` | 22 | All FP except the two `setup()` cases (item 3) and PPPM (already fixed). Traced every one: each base-class call site of the shadowed non-virtual method sits inside a function the KOKKOS class overrides (`compute`, `pre_force`, `setup_pre_force`, `init`, `init_style`, `energy_force`), or the caller already holds a correctly typed pointer (`min_kokkos.cpp` uses `fix_minimize_kk`), or the overload is a deliberate name-hiding forwarder (`domain_kokkos.h`, which carries the explaining comment). |
| `optin.cplusplus.VirtualCall` | 18 | All FP. Every `grow_arrays()` / `deallocate()` / `read_coeffs()` call is in the leaf class's own constructor or destructor, where the dynamic type is already correct. Checked the two classes that *are* further derived: `FixRigidNHSmallKokkos` does not override `grow_arrays`, and `PPPMTIP4PKokkos` overrides neither `allocate` nor `deallocate`. |
| `modernize-use-nodiscard` | 20 | Style only. |
| `modernize-use-equals-default` | 18 | Style only. |
| `clang-diagnostic-sign-compare` | 10 | Benign; see item 9. |
| `misc-header-include-cycle` | 9 | FP. `meam_kokkos.h` includes `meam_impl_kokkos.h` at the end (template-impl idiom) and each impl header re-includes the guard-protected parent. Same for `sna_kokkos_impl.h`. |
| `misc-unused-parameters` / `clang-diagnostic-unused-parameter` | 18 | FP functionally. `CommTiledKokkos::grow_list` grows the whole rectangular sendlist at once so `iswap`/`iwhich` are genuinely unneeded; `ComputeAveSphereAtomKokkos::pack_forward_comm_kokkos` packs velocities, which need no PBC shift (the CPU sibling comments the names out). Cosmetic fix: comment out the names. |
| `bugprone-virtual-near-miss` | 3 | FP. `FixQEqReaxFFKokkos::calculate_q` vs `FixQEqReaxFF::calculate_Q`: the KOKKOS class overrides `pre_force()`, the only caller of the base `calculate_Q()`, and calls its own device routine directly. Recommend renaming the KOKKOS method to something unambiguous rather than making it an override. |
| `bugprone-parent-virtual-call` | 3 | FP. `fix_rigid_nh_small_kokkos.cpp` calls `FixRigidSmall::setup()` deliberately -- there is a six-line comment explaining why -- and then re-implements the Nose-Hoover half inline, mirroring `FixRigidNHSmall::setup()`. The two `memory_usage` cases affect reporting only. |
| `performance-unnecessary-copy-initialization` | 4 | **Do not act.** In `*_hybrid_kokkos.cpp::compute()` the surrounding code deliberately copies the DualView handle (`auto k_anglelist_orig = neighborKK->k_anglelist;`) so it survives the sub-style's rebinding of `k_anglelist`. Converting to a reference risks reintroducing exactly the aliasing bug the copy prevents. |
| `readability-static-accessed-through-instance` | 4 | `atomKK->k_x.NEED_TRANSFORM`. Style only. |
| `cppcoreguidelines-virtual-class-destructor` | 3 | See item 4. |
| `modernize-use-nullptr` | 3 | Style only. |
| `readability-uppercase-literal-suffix` | 5 | Style only. |
| remaining checkers | <5 each | Style only. |

## Recommended checker configuration

To make future runs useful, suppress the checkers that generate volume without
signal in a Kokkos code base:

* `readability-redundant-typename` -- 55% of the run, C++20-only, and acting on
  it would break C++17 KOKKOS builds.
* `modernize-use-using`, `modernize-use-auto`, `readability-qualified-auto`,
  `modernize-use-nodiscard` -- 2400+ findings, pure churn.
* `performance-unnecessary-value-param` -- by-value View/DualView parameters are
  the Kokkos idiom and are pinned by the `KokkosBase` interface.
* `misc-override-with-different-visibility` -- collides with the LAMMPS
  convention of grouping overrides by access section.

That leaves roughly 200 findings per run, of which the bug-prone families
(`bugprone-*`, `optin.*`, `clang-diagnostic-*`) are ~90 -- a reviewable set.
