# Targeted regression inputs

Each input exercises one verified defect.  A finding counts as PROVEN only when the
pristine `origin/develop` build shows kk diverging from cpu AND the fixed build shows them
agreeing — cpu-vs-kk agreement on the fixed build alone proves nothing, because a test that
exercises nothing looks identical to a passing test.

Run three ways:

    lmp -in <input> -log none                                     # cpu reference
    lmp -k on t 1 -sf kk -pk kokkos -in <input> -log none          # kk, default
    lmp -k on t 1 -sf kk -pk kokkos neigh full newton off ...      # kk, GPU-like

## Results against pristine origin/develop (d71abe6102)

| input | upstream cpu | upstream kk | proven |
|---|---|---|---|
| in.gravity_disable | temp 0.53177514 | temp 0.61527548 | yes - `disable` ignored |
| in.morse_shift | E_pair -5.0248849 | E_pair -12.055493 | yes - offset never subtracted |
| in.dipole_split | E_pair -2.7509396 | E_pair -2.7534785 | yes - LJ force and virial dropped |
| in.dsf_special | E_coul -0.76902174 | E_coul +0.046435449 | yes - wrong sign |
| in.temp_region_bias | - | - | no divergence observed |
| in.gravity_none | control for in.gravity_disable | | |

## Two traps worth remembering

* `special_bonds coul 0.0` makes LAMMPS REMOVE those pairs from the neighbour list, so they
  never reach the kernel and a special-bonds bug cannot fire.  Use a fractional factor.
* `create_bonds many ... 0.9 1.1` adds zero bonds on an fcc lattice at rho=0.8442, where the
  nearest-neighbour distance is a/sqrt(2) ~ 1.188.  Always check the "Added N bonds" line.
