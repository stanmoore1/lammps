# Targeted regression inputs

Each input exercises one verified defect.  A finding counts as PROVEN only when the
pristine `origin/develop` build shows kk diverging from cpu AND the fixed build shows them
agreeing -- cpu-vs-kk agreement on the fixed build alone proves nothing, because a test that
exercises nothing looks identical to a passing test.

Run four ways and compare:

    lmp -in <input> -log none                                # cpu reference
    lmp -k on t 1 -sf kk -pk kokkos -in <input> -log none     # kk

against both `origin/develop` (pristine) and this branch (fixed).

## Results: pristine origin/develop (d71abe6102) vs this branch

Last thermo line of each run.  "upstream kk" is the defect; every fixed-kk run reproduces
the cpu reference exactly.

| input | quantity | cpu reference | upstream kk | fixed kk |
|---|---|---|---|---|
| in.gravity_disable | temp | 0.53177514 | 0.61527548 | 0.53177514 |
| in.morse_shift | E_pair | -5.0248849 | -12.055493 | -5.0248849 |
| in.dipole_split | E_pair / press | -6.1665552 / -1.4057913 | -5.7197887 / +5.4435332 | -6.1665552 / -1.4057913 |
| in.dsf_special | E_coul | -0.76902174 | +0.046435449 | -0.76902174 |
| in.hexorder_nnn_null | c_hxa | 0.001817125 | -nan | 0.001817125 |
| in.temp_region_bias | temp / c_t | 0.54286473 / 0.28128204 | SIGSEGV in `ComputeTempRegion::dof_remove` | 0.54286473 / 0.28128204 |
| in.map_hash_dedup | E_total | 6.1118502 | `Kokkos::abort`, "Concurrent modification of host and device hashes" | 6.1118502 |

`in.map_hash_dedup` NEEDS the extra run option `-pk kokkos atom/map device`.  That is the
DEFAULT on a GPU build, so upstream `develop` aborts on the ordinary path there for any
molecular system with special bonds; a CPU-only build has to ask for it explicitly.  This one
is an upstream regression rather than a long-standing defect -- `map_clear()` gained its
device branch without a matching `clear_sync_state()`.

`in.gravity_none` is the control for `in.gravity_disable`: it agrees everywhere, confirming
that the divergence above comes from the `disable` keyword and not from `fix gravity` itself.

### Seed sensitivity: in.brownian_seed1 / in.brownian_seed2

`pair brownian/kk` built its RNG pool in the constructor, before `settings()` had parsed the
seed, so the seed was ignored.  The proof is a comparison between the two inputs, which
differ only in the seed:

| build | seed 12345 | seed 98765 | |
|---|---|---|---|
| upstream kk | 234248.18 232703.34 216032.87 | 234248.18 232703.34 216032.87 | identical -- seed ignored |
| fixed kk | 185788.75 167311.44 214435.96 | 192571.59 184419.40 191304.83 | differs -- seed honoured |
| cpu (both builds) | 205170.72 194227.17 211379.44 | 216260.85 179485.73 222963.54 | differs, as it always did |

kk and cpu are not expected to agree here: they draw from different generators.  Only
seed sensitivity is under test.

### CPU-side defects: pristine vs fixed, both builds

These three are not KOKKOS defects; the KOKKOS variant mirrors a defective CPU parent, so kk
and cpu agree with each other in BOTH builds and only a pristine-vs-fixed comparison shows
anything.  See ../NON_KOKKOS_FINDINGS.md.

`in.lj_expand_sphere_equiv` is self-checking: its two thermo lines are the same potential
computed two ways and must agree.  Sum of squared forces, cpu and kk alike:

| | lj/expand | lj/expand/sphere | |
|---|---|---|---|
| upstream | 1.0682863e+11 3.5069984e+09 9.3367809e+10 | 3.0885079e+09 1.6353346e+08 2.785233e+09 | disagree |
| fixed | 1.0682863e+11 3.5069984e+09 9.3367809e+10 | 1.0682863e+11 3.5069984e+09 9.3367809e+10 | agree exactly |

Energy is 980.35705 in every one of those runs: same potential, different gradient.

`in.deform_remap_subgroup` (temp / c_tm / press at step 50):

| | upstream | fixed |
|---|---|---|
| cpu and kk (identical within each build) | 0.66710764 / 0.39908086 / 1.3382274 | 0.66731388 / 0.39942296 / 1.3389962 |

This one shows that the code path is live and that the change is real and deterministic; it
does not by itself say which answer is right.  That rests on the code: `j = list[i]` is the
atom, every other field in the loop is packed from `j`, and `mask[j]` is packed on the
adjacent line.  Note the three conditions in the input header -- drop any one of them and the
defect is invisible, which is why it survived.

The `compute sna/grid` inner-cutoff fix has no input here.  It needs a SNAP potential file
with `switchinnerflag 1`, and the shipped parameter sets do not exercise it.

## Traps worth remembering

* `special_bonds coul 0.0` makes LAMMPS REMOVE those pairs from the neighbour list, so they
  never reach the kernel and a special-bonds bug cannot fire.  Use a fractional factor.
* `create_bonds many ... 0.9 1.1` adds zero bonds on an fcc lattice at rho=0.8442, where the
  nearest-neighbour distance is a/sqrt(2) ~ 1.188.  Always check the "Added N bonds" line.
* A defect can present as a crash rather than as a wrong number.  An extractor that reads the
  last thermo line reports a segfaulting run as empty output, which reads like "no
  divergence".  Check for a missing result, not just for a different one.
* `pair_style brownian` needs seven or nine arguments (mu flaglog flagfld cut_inner cut
  t_target seed); a short list is rejected with a bare "Illegal pair_style command".
* Packing the wrong value into a ghost buffer shows nothing unless something READS it.
  `in.deform_remap_subgroup` was silent with `pair lj/cut`, which never looks at ghost
  velocities; it needed a granular style before the defect could reach a number.
* Prefer a deterministic style when the point is to compare two builds.  A stochastic style
  (dpd) does show the divergence, but it cannot tell you which of the two answers is correct.
* An extractor that reads "the last thermo line" silently reports only the second half of an
  input with two `run` commands.
