# KOKKOS neighflag / newton reachability audit

Purpose: for every KOKKOS style whose reported defect depends on a particular
`neighflag` / `newton` configuration, decide whether that configuration can
actually be reached at run time.  Repo root `/home/user/lammps`.

## 1. The rules that govern neighflag and newton

All line numbers are `src/KOKKOS/kokkos.cpp` unless stated otherwise.

| # | Rule | Evidence |
|---|---|---|
| R1 | `package kokkos` is executed unconditionally at startup whenever KOKKOS exists, so the block below always runs — there is no "no package command" escape. | `src/lammps.cpp:916` (`input->one("package kokkos")`) |
| R2 | GPU defaults (`ngpus > 0`): `neighflag = FULL`, `neighflag_qeq = FULL`, `newtonflag = 0`. | `kokkos.cpp:345-348` |
| R3 | CPU defaults: `neighflag = HALFTHREAD` (nthreads>1) or `HALF`; `newtonflag = 1`. | `kokkos.cpp:358-366` |
| R4 | `package kokkos neigh full|half` sets `neighflag`; `half` becomes `HALFTHREAD` when `nthreads > 1 || ngpus > 0`. | `kokkos.cpp:532-542` |
| R5 | `package kokkos newton on|off` sets `newtonflag`. | `kokkos.cpp:557-560` |
| R6 | At the end of `accelerator()`: `force->newton = force->newton_pair = force->newton_bond = newtonflag;` then `newton_check()`. | `kokkos.cpp:846-848` |
| R7 | **`newton_check()`: `neighflag == FULL && force->newton` is a fatal error.** Note it tests `force->newton` (the OR of pair and bond), so FULL requires *both* newton pair off *and* newton bond off. | `kokkos.cpp:859-860` |
| R8 | `neigh_thread && force->newton` is likewise fatal. | `kokkos.cpp:862-863` |
| R9 | The `newton` command re-runs `newton_check()`, so the invariant cannot be broken after the `package` command. | `src/input.cpp:1729` |
| R10 | `neighflag_qeq` is a **separate** flag. `newton_check()` does not constrain it, and `package kokkos neigh half` does **not** change it (only `neigh/qeq half` does). | `kokkos.cpp:347`, `543-553`, `859-863` |
| R11 | The small-system auto-heuristic only turns `neigh_thread` on when the configuration is already `FULL` or newton-off, so it cannot violate R8. | `src/KOKKOS/pair_kokkos.h:900-903` |

### Corollaries used throughout

* **C1 — `neighflag == FULL` implies `force->newton_pair == 0`, always.**
  Therefore *any* code branch of the shape `neighflag == FULL` combined with
  `newton_pair == 1` is **dead code**, in every KOKKOS style.
* **C2 — reaching `FULL` on a CPU-only build requires an explicit
  `package kokkos neigh full newton off`.**  `neigh full` alone fails R7 because
  the CPU `newtonflag` default is 1 (R3), and issuing `newton off` before the
  `package` command does not help, because R6 overwrites `force->newton` from
  `newtonflag`.  On a GPU build FULL is the default (R2).
* **C3 — reaching `newton on` on a GPU build requires an explicit
  `package kokkos neigh half newton on`.**  `neigh half` alone leaves
  `newtonflag = 0`; `newton on` alone fails R7 because `neighflag` is still FULL.
* **C4 — the decisive question for a KOKKOS style is whether its `init_style()`
  calls a CPU base `init_style()` that errors on newton off *before* the KOKKOS
  neighbor request is adjusted.**  If it does, FULL is unreachable for that style
  even though `lmp->kokkos->neighflag` is FULL by default on a GPU.

## 2. The three named findings

### 2.1 pair vashishta/kk — UNREACHABLE (worked example, confirmed)

* `PairVashishtaKokkos::init_style()` calls the CPU base first:
  `src/KOKKOS/pair_vashishta_kokkos.cpp:585-587`.
* The base errors on newton off: `src/MANYBODY/pair_vashishta.cpp:268-269`
  (`"Pair style Vashishta requires newton pair on"`).
* By C1, `neighflag == FULL` implies `newton_pair == 0`, so that base call always
  aborts first.  The `FULL` dispatch at
  `src/KOKKOS/pair_vashishta_kokkos.cpp:153-165` (TagPairVashishtaComputeFullA /
  FullB) is therefore never entered.
* The only reachable configuration is `HALF`/`HALFTHREAD` + newton on, i.e. on a
  GPU it needs an explicit `package kokkos neigh half newton on` (C3).
* Note the header's `EnabledNeighFlags = FULL` (`pair_vashishta_kokkos.h`) is
  vestigial: this style has its own `compute()` and never calls `pair_compute`,
  so the enum constrains nothing.  Also note `init_style` requests a *full
  neighbour list* unconditionally (`pair_vashishta_kokkos.cpp:596`) — that is the
  neighbour-list shape, not the force-accumulation mode, and is unrelated to
  `neighflag`.

### 2.2 pair brownian/kk — REACHABLE (verified; GPU default)

* `PairBrownianKokkos::init_style()` calls the base at
  `src/KOKKOS/pair_brownian_kokkos.cpp:80-82`.
* The base only **warns**: `src/COLLOID/pair_brownian.cpp:445-446`
  (`error->warning(FLERR, "Pair brownian needs newton pair on for momentum
  conservation")`) — no `error->all`, nothing else in it constrains newton.
* So `FULL` + newton off is reachable and is the **default on any GPU build**
  (R2), with no `package kokkos` option needed.
* The reported defect is real on that path: the j-side force is guarded by
  `if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && ...)` at
  `src/KOKKOS/pair_brownian_kokkos.cpp:404-407` (same shape for the torque at
  432 and 462).
* Severity caveat: with a full list each pair is visited from both sides with
  *independent* random draws (`rand_gen.drand()`, lines 379/385), which is
  exactly the behaviour the CPU code warns about for newton off.  The warning
  does fire on the GPU default, since `force->newton_pair == 0` there.
* The `neighflag == FULL` + `newton_pair` sub-branches at
  `pair_brownian_kokkos.cpp:207-208, 218-219, 231-232, 242-243` are dead (C1).

### 2.3 pair multi/lucy/rx/kk — REACHABLE (verified; newton off is the GPU default)

* `PairMultiLucyRXKokkos::init_style()` calls
  `PairMultiLucyRX::init_style()` at
  `src/KOKKOS/pair_multi_lucy_rx_kokkos.cpp:110-112`.
* `PairMultiLucyRX` declares **no** `init_style` at all
  (`src/DPD-REACT/pair_multi_lucy_rx.{h,cpp}` — no override), so the call
  resolves to `Pair::init_style()` (`src/pair.cpp`), whose entire body is
  `neighbor->add_request(this);`.  No newton constraint whatsoever.
* Therefore newton off is reachable — and it is the default on GPU (R2) and
  available on CPU via `package kokkos newton off`.
* The reported defect is on a live path: the self-energy term is halved when
  `NEWTON_PAIR == 0` at `src/KOKKOS/pair_multi_lucy_rx_kokkos.cpp:442`
  (the line carries an in-tree `FIXME???` comment).  This is a *self* energy, so
  the half-vs-full-pair reasoning that justifies the 0.5 for pair terms does not
  apply.
* The `neighflag == FULL` + `newton_pair` branches at
  `pair_multi_lucy_rx_kokkos.cpp:228-229` and `511-512` are dead (C1).

## 3. Full sweep — FULL reachability per KOKKOS style

### 3.1 FULL rejected explicitly inside the KOKKOS `init_style()` — UNREACHABLE

These abort with a clear message, so any finding in their `FULL` kernels is
latent, not live.

| Style | Decisive line |
|---|---|
| `sw/kk` | `pair_sw_kokkos.cpp:432-433` |
| `sw/mod/kk` | `pair_sw_mod_kokkos.cpp:432-433` |
| `tersoff/kk` | `pair_tersoff_kokkos.cpp:134-135` |
| `tersoff/mod/kk` | `pair_tersoff_mod_kokkos.cpp:120-121` |
| `tersoff/mod/c/kk` | `pair_tersoff_mod_c_kokkos.cpp:120-121` |
| `tersoff/zbl/kk` | `pair_tersoff_zbl_kokkos.cpp:133-134` |
| `meam/kk` | `pair_meam_kokkos.cpp:359-360` (base also errors: `src/MEAM/pair_meam.cpp:326`) |
| `snap/kk` | `pair_snap_kokkos_impl.h:83-84`, plus its own newton check at `:72-73` |
| `reaxff/kk` | `pair_reaxff_kokkos.cpp:194-195` (base also errors: `src/REAXFF/pair_reaxff.cpp:365`) |
| `pace/kk` | `pair_pace_kokkos.cpp:780-781`, newton check at `:770` |
| `pace/extrapolation/kk` | `pair_pace_extrapolation_kokkos.cpp:465-466`, newton check at `:455` |
| `pod/kk` | `pair_pod_kokkos.cpp:107-108`, newton check at `:99` |
| `dpd/kk` | `pair_dpd_kokkos.cpp:139-140` (rejects FULL **and** newton off) |
| `dpd/tstat/kk` | `pair_dpd_tstat_kokkos.cpp:93-94` |
| `dpd/ext/kk` | `pair_dpd_ext_kokkos.cpp:94-95` |
| `dpd/ext/tstat/kk` | `pair_dpd_ext_tstat_kokkos.cpp:93-94` |
| `gran/hooke/kk` | `pair_gran_hooke_kokkos.cpp:77-78` |
| `gran/hooke/history/kk` | `pair_gran_hooke_history_kokkos.cpp:91-92` |
| `gran/hertz/history/kk` | `pair_gran_hertz_history_kokkos.cpp:91-92` |

`sw/kk`, `sw/mod/kk` and the four `tersoff` variants additionally declare
`EnabledNeighFlags = HALF|HALFTHREAD` (no FULL) in their headers, so the FULL
instantiation of `pair_compute` does not even exist for them.

### 3.2 FULL blocked *only* by a CPU base that requires newton on — UNREACHABLE

No KOKKOS-level message; the base aborts first.  These are the cases the review's
"this is the GPU default" reasoning gets wrong.

| Style | KOKKOS base call | CPU base error |
|---|---|---|
| `vashishta/kk` | `pair_vashishta_kokkos.cpp:587` | `src/MANYBODY/pair_vashishta.cpp:268-269` |
| `uf3/kk` | `pair_uf3_kokkos.cpp:146` | `src/ML-UF3/pair_uf3.cpp:1041` |
| `mliap/kk` | `pair_mliap_kokkos.cpp:333` | `src/ML-IAP/pair_mliap.cpp:340-341` |
| `tip4p/cut/kk` | `pair_tip4p_kokkos.h:89-91` (`PairCPUBase::init_style()`) | `src/MOLECULE/pair_tip4p_cut.cpp:434-435` |
| `lj/cut/tip4p/cut/kk` | `pair_tip4p_kokkos.h:89-91` | `src/MOLECULE/pair_lj_cut_tip4p_cut.cpp:508-509` |

Notes:
* `uf3/kk` declares `EnabledNeighFlags = FULL` and its `compute()` dispatches the
  `TagPairUF3ComputeFullA<FULL,...>` kernel **unconditionally** (hard-coded
  `FULL` template argument, `pair_uf3_kokkos.cpp:794-800`) regardless of
  `neighflag`.  The only thing `neighflag == FULL` actually gates is
  `no_virial_fdotr_compute = 1` at `pair_uf3_kokkos.cpp:723` — and that line is
  unreachable, so `uf3/kk` always takes the `pair_virial_fdotr_compute` path
  while running a full-list kernel.  Worth a separate look, but it is a
  *consequence* of unreachability, not an example of it.
* The `tip4p` KOKKOS styles never read `neighflag` at all (no occurrence in
  `pair_tip4p_kokkos.h`), so they have no FULL path to reach.

### 3.3 FULL reachable — no newton constraint anywhere on the path

For all of these, `PairXxxKokkos::init_style()` calls a CPU base whose
`init_style()` contains no newton test (or only a warning), and the KOKKOS
`init_style()` merely does `if (neighflag == FULL) request->enable_full();`.
FULL + newton off is therefore the **GPU default** with no `package` option.

This covers the whole "plain pair style" family with
`EnabledNeighFlags = FULL|HALFTHREAD|HALF` — lj/cut, coul/*, born/*, buck/*,
morse, gauss, yukawa, table, zbl, eam (and eam/alloy, eam/fs, eam/he), adp,
colloid, brownian, yukawa/colloid, ylz, wf/cut, momb, ufm, soft, beck,
lj/charmm*, lj/class2*, nm/cut*, lj/spica*, exp6/rx, table/rx, multi/lucy/rx,
dpd/fdt/energy, bondval, bondval/vec, lj/cut/dipole/cut, coul/wolf, coul/dsf, …

Spot-checked bases with no newton requirement:
`src/pair.cpp Pair::init_style()` (bare `neighbor->add_request(this)`),
`src/MANYBODY/pair_adp.cpp PairADP::init_style()`,
`src/ASPHERE/pair_ylz.cpp PairYLZ::init_style()`.
Bases that only **warn** (so FULL stays reachable):
`src/COLLOID/pair_brownian.cpp:445-446`,
`src/DPD-REACT/pair_dpd_fdt_energy.cpp:409-411`.

Within this group, the following contain `neighflag == FULL` + `newton_pair`
branches that are **dead code** by C1 — a finding located inside one of these is
not live:

`pair_adp_kokkos.cpp:238,258` · `pair_bondval_kokkos.cpp:230,262` ·
`pair_bondval_vec_kokkos.cpp:232,264` · `pair_brownian_kokkos.cpp:207,218,231,242` ·
`pair_coul_dsf_kokkos.cpp:133,153` · `pair_coul_wolf_kokkos.cpp:137,157` ·
`pair_coul_wolf_cs_kokkos.cpp:142,162` ·
`pair_dpd_fdt_energy_kokkos.cpp:210,236,284,310` ·
`pair_eam_kokkos.cpp:235,267` · `pair_exp6_rx_kokkos.cpp:267,305` ·
`pair_lj_cut_dipole_cut_kokkos.cpp:152,163,176` ·
`pair_multi_lucy_rx_kokkos.cpp:228,511` · `pair_table_rx_kokkos.cpp:741,791` ·
`pair_ylz_kokkos.cpp:133,144,157,168` · `fix_rx_kokkos.cpp:1382`.

### 3.4 Special cases

**`neighflag_qeq == FULL` is reachable *and* is the ReaxFF GPU default.**
`fix qeq/reaxff/kk` and `fix acks2/reaxff/kk` read `lmp->kokkos->neighflag_qeq`
(`fix_qeq_reaxff_kokkos.cpp:108`, `fix_acks2_reaxff_kokkos.cpp:113`), which
`newton_check()` never constrains (R10).  Because `pair reaxff/kk` rejects
`neighflag == FULL` (`pair_reaxff_kokkos.cpp:194-195`) and its base requires
newton on (`src/REAXFF/pair_reaxff.cpp:365`), a GPU ReaxFF run must pass
`package kokkos neigh half newton on` — which leaves `neighflag_qeq` at its GPU
default of FULL (`kokkos.cpp:347`).  So the FULL qeq/acks2 kernels
(`fix_qeq_reaxff_kokkos.cpp:242,255,428,476,589`;
`fix_acks2_reaxff_kokkos.cpp:304,321,344,363,386,1241,1305,1350,1428`) are live
on exactly the standard GPU ReaxFF configuration, and they are FULL **together
with newton pair on** — the one place where the "FULL implies newton off"
corollary does *not* hold.  Findings in those kernels must not be dismissed by C1.

**`fix neigh/history/kk` requires newton off.**
`fix_neigh_history_kokkos.cpp:92-94` aborts in `pre_exchange()` when
`newton_pair` is set (`newton_pair` is copied from `force->newton_pair`,
`src/fix_neigh_history.cpp:48`).  Combined with 3.1, the granular history styles
need `neighflag != FULL` **and** newton off — i.e. `package kokkos neigh half`
on a GPU (newton is already off), or an explicit `package kokkos newton off` on
a CPU build, where the default HALF+newton-on configuration aborts.

**`fix shake/kk` never sees FULL.**  It downgrades FULL to HALFTHREAD/HALF
itself (`fix_shake_kokkos.cpp:411-414` and `694-703`), so it has no FULL path.

**`neigh_thread` (and hence `threads/per/atom` / `pair/team/size`) requires
newton off** (`kokkos.cpp:862-867`), so it carries the same reachability
restriction as FULL, except that it is additionally enabled automatically for
small GPU systems (`pair_kokkos.h:900-903`).

## 4. Configurations that are reachable only via an explicit `package kokkos`

Relevant because it lowers severity: a user has to opt in.

| Configuration | How to reach it | Never a default on |
|---|---|---|
| `FULL` + newton off | GPU: default. CPU-only build: `package kokkos neigh full newton off` (C2) | CPU-only builds |
| `HALF`/`HALFTHREAD` + newton **on** | CPU: default. GPU: `package kokkos neigh half newton on` (C3) | GPU builds |
| `HALF`/`HALFTHREAD` + newton off | GPU: `package kokkos neigh half`. CPU: `package kokkos newton off` | neither is the default |
| `neigh_thread on` | `package kokkos neigh/thread on newton off`, or automatic on GPU when `inum <= 16000` and FULL/newton-off | CPU builds |
| `neighflag_qeq FULL` with `neighflag HALF*` | GPU + `package kokkos neigh half newton on` (the standard ReaxFF GPU recipe) | CPU builds |

Consequence for the review: **every** KOKKOS many-body style
(sw, tersoff*, vashishta, meam, snap, reaxff, pace, pod, uf3, mliap) can only run
on a GPU with an explicit `package kokkos neigh half newton on`.  A finding in
one of their kernels is never "the GPU default path"; it is the path a user
reaches after opting in, and their FULL kernels are unreachable outright.
