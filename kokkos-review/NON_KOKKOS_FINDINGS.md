# Defects found outside src/KOKKOS

This review targeted `src/KOKKOS`, but four defects surfaced that are not KOKKOS defects at
all.  Two were found because the KOKKOS variant faithfully mirrors a defective CPU parent, so
a kk-vs-cpu comparison can never see them; two were found by reading the CPU parent while
checking a KOKKOS claim.

They are fixed on this branch because the project's own rule is to fix a style, its parent,
and all its suffix variants together.  A maintainer may reasonably want them split into their
own pull request -- they touch `src/`, `src/EXTRA-PAIR/`, `src/ML-SNAP/` and `src/OPENMP/`,
and one of them invalidates a checked-in reference fixture.

## 1. `pair_style lj/expand/sphere` computes the wrong force  (all three variants)

    src/EXTRA-PAIR/pair_lj_expand_sphere.cpp:122   compute()
    src/EXTRA-PAIR/pair_lj_expand_sphere.cpp:396   single()
    src/OPENMP/pair_lj_expand_sphere_omp.cpp:149
    src/KOKKOS/pair_lj_expand_sphere_kokkos.cpp:154

All four read

    fpair = factor_lj * forcelj * rshift / r;

where the gradient of the shifted Lennard-Jones potential requires a division:

    fpair = factor_lj * forcelj / rshift / r;

The parent style has it right (`src/pair_lj_expand.cpp:115`); the sphere variant inverted the
operator and the /omp and /kk variants inherited it by copy-adapt.  Because all three agree,
no accelerator-vs-reference comparison can catch this.

Proof without relying on the derivation: `lj/expand/sphere` with every radius set to `d/2` is
by construction the same potential as `lj/expand` with `shift = d`.  Set up both on identical
coordinates and compare (`run 0`, atoms displaced off-lattice so forces are non-zero):

| | E_pair | sum fx^2 | sum fy^2 | sum fz^2 |
|---|---|---|---|---|
| `lj/expand` shift 0.5 | 980.35705 | 1.0682863e+11 | 3.5069984e+09 | 9.3367809e+10 |
| `lj/expand/sphere` radius 0.25 | 980.35705 | 3.0885079e+09 | 1.6353346e+08 | 2.785233e+09 |

Identical potential energy, forces differing by more than an order of magnitude: same
potential, different gradient.  Only one of the two can be right, and `lj/expand` is.

**This invalidates `unittest/force-styles/tests/atomic-pair-lj_expand_sphere.yaml`**, which
was generated from the defective code and therefore encodes the wrong reference forces.  It
has to be regenerated, which means the fixture's authority for this style is only as good as
the fix -- worth a maintainer's eye before it is trusted again.

## 2. `compute sna/grid` reads its inner cutoffs out of bounds

    src/ML-SNAP/compute_sna_grid.cpp:266-267
    src/ML-SNAP/compute_sna_grid_local.cpp:263-264

`sinnerelem` and `dinnerelem` are allocated with `ntypes + 1` entries and filled at indices
`1..ntypes` (line 144), i.e. indexed by atom type.  Both files then read them with `jelem`,
the 0-based element index.  In a single-element run `jelem` is 0 and the read lands on the
never-written slot; with several elements it is simply the wrong entry.

Every sibling in the package -- `compute_snap`, `compute_sna_atom`, `compute_snad_atom`,
`compute_snav_atom` -- reads `sinnerelem[itype]`/`[jtype]`, and the surrounding lines in these
very files already use `jtype` for `wjelem` and `radelem`.  Only these two lines use `jelem`.
Now `sinnerelem[jtype]` / `dinnerelem[jtype]`.

The KOKKOS grid computes had a different defect on the same line: they index by type
correctly but average in `d_sinnerelem[itype]` where `itype` is a hardcoded `1` (a grid point
has no atom type), which is the pair-snap convention applied where there is no central atom.
Fixed to match the CPU.

## 3. `AtomVec::pack_comm_vel` / `pack_border_vel` test the wrong atom's mask

    src/atom_vec.cpp:490    pack_comm_vel()
    src/atom_vec.cpp:951    pack_border_vel()
    src/KOKKOS/atom_vec_kokkos.cpp:1851  (mirrors the CPU, same defect)

Both loops run `j = list[i]`, pack every field from `j` -- including `mask[j]` on the line
immediately above -- and then decide whether to apply the `fix deform` velocity remap with

    if (mask[i] & deform_groupbit)

`i` is the position in the send list, not an atom index.  The remap is therefore applied to a
set of ghost atoms unrelated to the deform group: for a send list of length n it tests the
masks of atoms 0..n-1 whatever the list actually contains.  Only `fix deform` with `remap v`
and a deform group that is not `all` is affected, which is why it has survived; with
`group all` every mask matches and the bug is invisible.

`domain.cpp`, `domain_omp.cpp` and `domain_kokkos.cpp` all use `mask[i]` correctly -- there
`i` really is the atom index.  `RHEO/compute_rheo_grad.cpp:345` shows the correct form for a
packing loop.  Now `mask[j]`.

## 4. Unported upstream fix: `cutsq_trim` fallback  (ported on this branch)

Upstream commit `47cea8e1ba` added a fallback to the pairwise neighbour cutoff when a trim
request carries no custom cutoff, touching `src/npair.cpp`, `npair_halffull.cpp`,
`npair_skip.cpp` and `npair_trim.cpp`.  It touched no KOKKOS mirror, so the KOKKOS neighbour
builds kept trimming against `cutsq_custom == 0` and discarded every pair.

This is the one class of defect a line-by-line review structurally cannot find: both sides
are internally coherent and neither looks wrong on its own.  Only the history reveals it.
Worth a periodic `git log` sweep of `src/npair*.cpp` against `src/KOKKOS/npair*_kokkos.cpp`.
