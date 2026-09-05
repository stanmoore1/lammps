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
