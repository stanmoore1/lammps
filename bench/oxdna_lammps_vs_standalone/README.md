# LAMMPS vs standalone oxDNA: smoothing widths, cross-stacking gate, coaxial stacking

Reproducers for three physics differences between the LAMMPS CG-DNA styles
(`oxdna3KK-kk-fixes`, 2afff57fb5, potential file `potentials/oxdna3_lj.cgdna`)
and standalone oxDNA (c2c74cc0, `DNA2_nomesh` / `DNA3_nomesh`).

- `cases/`: eight single configurations. Each has a standalone-oxDNA input
  (`input_oxdna`: `potential_energy split` + `pair_energy`), a LAMMPS input
  (`in.lammps`: one `compute pair` per style) and the same configuration in
  both formats (`top.top` / `conf.dat`, `data.lmp`).
  `OXDNA=... LMP=... ./run_cases.sh` prints the hbond (hb), cross-stacking
  (xstk) and coaxial-stacking (cx) energies of both codes. The reference output
  is `expected_output.txt`, and the pair-level breakdown is `expected_pairs.txt`.
- `reproduce_md.sh`: the MD statistics below (`results_md_statistics.txt`).
  - Five structures are built with the geometry of oxDNA's `generate-sa.py`
    (`tools/build.py`): a 12-bp duplex, a 16-bp nicked duplex, two 8-bp duplexes
    stacked at a blunt end, a 12-bp duplex with a one-nucleotide bulge, and a
    20-nt single strand.
  - Each structure gets 1e6 steps of standalone Brownian MD at T = 0.1 (300 K);
    the 12-bp duplex also runs at T = 0.12 (fraying).
  - All 199 frames per run are evaluated with both codes.
- `tools/lammps_verification.patch`: verification-only environment switches for
  the CPU styles:
  - `OXDNA3_XSTK_GATE` applies the standalone cross-stacking gate;
  - `OXDNA_COAX_NOTERM` drops the terminal criterion;
  - `OXDNA_COAX_NOMIRROR` drops the mirrored theta4 lobe.

  With these switches and `oxdna3_lj_ts.cgdna` (column `lmpV`), LAMMPS equals
  standalone oxDNA in hb, xstk and cx to <= 2e-4 on every one of the 2388
  frames. So the three items below account for every difference in these terms
  beyond that. The 2e-4 residual is in the oxDNA3 nicked-duplex coaxial
  stacking and is the same in every LAMMPS variant.

Energies are totals in oxDNA units (kT = 0.1 at 300 K).

## Q1: oxDNA3 hbond / xstk smoothing widths (0.7 vs 0.7359)

**Where the difference comes from.** Standalone oxDNA3 does not use the
smoothing points `*_TS` of `model.h` or of the parameter file.
`DNA3Interaction::init()` recomputes them, and it does so for every one of the
21 sequence-dependent f4 modulations (stacking, hbond, cross stacking) and
every tetramer, in the loop commented "Set enslaved parameters. Enslaved = set
by continuity and differentiability" (`src/Interactions/DNA3Interaction.cpp:945`,
added with the oxDNA3 merge, 88403457):

    F4_SD_THETA_TS[m](i, j, k, l) = sqrt(0.81225/F4_SD_THETA_A[m](i, j, k, l));
    F4_SD_THETA_TC[m](i, j, k, l) = 1. / F4_SD_THETA_A[m](i, j, k, l) / F4_SD_THETA_TS[m](i, j, k, l);

So in oxDNA3 every f4 reaches the same value 1 - 0.81225 = 0.18775 at its
smoothing point. oxDNA2 (`DNA2Interaction`) does not do this and keeps the
`model.h` values, so for oxDNA2 both codes agree.

`oxdna3_lj.cgdna` applies the recomputation to the entries that were made
sequence-dependent (hbond theta4, xstk theta4 33/55, stacking). The entries
copied from oxDNA2 keep the `model.h` values instead (`tools/ts_compare.py`):

| term | A | ts in `oxdna3_lj.cgdna` (= `model.h`) | standalone oxDNA3 sqrt(0.81225/A) | f4 differs for dtheta in (rad) | max diff of f4 |
|---|---:|---:|---:|---|---:|
| hbond theta1, theta2, theta3 | 1.5 | 0.70 | 0.735867 | 0.700 - 0.952 | 0.025 |
| hbond theta4 (A-T / C-G) | 1.3 / 2.0 | 0.790448 / 0.637279 | same | - | 0 |
| hbond theta7, theta8 | 4.0 | 0.45 | 0.450625 | 0.450 - 0.556 | 0.0007 |
| xstk theta1 | 2.25 | 0.58 | 0.600833 | 0.580 - 0.766 | 0.018 |
| xstk theta2, theta3, theta7, theta8 | 1.7 | 0.68 | 0.691227 | 0.680 - 0.865 | 0.008 |
| xstk theta4 (33 / 55, tetramer tables) | 1.30 - 2.37 | sqrt(0.81225/A) | same | - | 0 |

The two f4 curves are identical up to the LAMMPS smoothing point, because both
are 1 - A dtheta^2 there. They differ only in the smoothing tail. LAMMPS starts
the tail later and reaches zero later (hbond theta1: at 0.952 rad instead of
0.906), so LAMMPS is slightly more attractive. That is why a single
configuration can look "the same".

Size in MD (oxDNA3, T = 0.1, compare `lmp` with `lmpT`):
- The mean hb shift is -0.0004 to -0.0005 per system, about 0.01 % of E_hb.
- The largest single-frame difference is 0.013 (0.13 kT).
- 10 - 17 % of the frames of the duplex systems differ by more than 1e-3.

Reproducer: `cases/q1_hbond_smoothing_oxdna3` (16-bp helix with a blunt
interface, one MD frame). E_hb is -7.785143 in standalone oxDNA, -7.798547 in
LAMMPS with the shipped file, and -7.785143 with `oxdna3_lj_ts.cgdna`. Fix:
write sqrt(0.81225/A) for these entries (`tools/patch_potfile.py` does this).

## Q2: oxDNA3 cross-stacking gate

**Correction to the earlier question.** Standalone oxDNA does not pick one
channel. `DNA3Interaction::_cross_stacking` computes the 3'3' **and** the 5'5'
channel, but only when

    (cos7 > 0 && cos8 > 0 && rclow_33 < r < rchigh_33) || (cos7 < 0 && cos8 < 0 && rclow_55 < r < rchigh_55)

Otherwise the pair gets exactly 0. LAMMPS evaluates both channels
unconditionally.

**When it matters.**
- The 3'3' channel is centred at theta7 = theta8 = 50.1 deg and extends to
  98.9 deg. The 5'5' channel is centred at 129.9 deg and extends down to
  81.1 deg.
- A pair with one angle just past 90 deg and the other one well inside its
  channel therefore has a nonzero LAMMPS energy but 0 in standalone oxDNA.
  f4(theta7 = 90 deg) = 0.177, so this is a tail contribution.
- The largest possible jump at the gate (all other factors 1) is
  0.38 x 0.177 = 0.067.
- In every MD frame with a difference, the cause was a cross-strand pair in the
  3'3' channel with theta7 or theta8 between 90 and 93 deg ("mixed signs").
  These pairs sat near a duplex end, the nick or the blunt interface.
  `tools/xstk_pairs.py` lists them per frame.

Size in MD (oxDNA3; compare `lmpT` with standalone, which isolates the gate):
- The mean xstk shift is -0.0001 to -0.0007, about 0.01 % of E_xstk.
- The largest single-frame difference is 0.020 (0.2 kT).
- 2.5 % (12-bp duplex, 300 K) to 14 % (bulge) of the frames differ by more
  than 1e-3.

So on average energies it is negligible, as Oliver said. But it is reached
routinely, and in standalone oxDNA it is a discontinuity: the energy jumps by
up to 0.02 (at most 0.067) whenever a cos changes sign, which shows up as
energy drift in NVE.

Reproducers:
- `cases/q2_xstk_gate_nick_oxdna3` (nicked duplex, 300 K): pair (7, 24),
  theta7 = 92.7 deg, E = -0.0195 in LAMMPS, 0 in standalone oxDNA.
- `cases/q2_xstk_gate_hot_duplex_oxdna3` (duplex at T = 0.12): pair (10, 13),
  theta7 = 90.1 deg, E = -0.0190.

## Q3: coaxial stacking terminal criterion and mirrored theta4 lobe

The LAMMPS rules (3c86796749) apply to oxDNA1/2/3:
- **Terminal criterion.** The pair is skipped unless *both* nucleotides are
  strand ends (`id3p` or `id5p` unset).
- **Mirrored lobe.** theta4 gets a second lobe, `F4(theta4; pi - theta4_0)`.

Standalone oxDNA (`_coaxial_stacking`) evaluates every non-bonded pair, with a
symmetrised K for non-terminal pairs (`F2_SD_K_SYMM`), and has a single theta4
lobe.

**Nick and blunt end: no difference, as Oliver expected.** Both nucleotides are
terminal there, and the mirrored lobe was never active in these runs:
- the 16-bp nicked duplex (oxDNA2 and oxDNA3);
- the two duplexes stacked at a blunt end (oxDNA2 and oxDNA3).

LAMMPS and standalone oxDNA agree to < 2e-4 on all frames of these systems
(`cases/control_coax_nick_oxdna2`: -0.068345 in both).

**Terminal criterion: large differences whenever a nucleotide is looped out.**
- If nucleotide i+1 leaves the helix (a bulge, or a base flipped out of a
  duplex), nucleotides i and i+2 of the same strand stack coaxially across the
  gap.
- They are neither bonded nor terminal, so standalone oxDNA counts them and
  LAMMPS skips them.
- One such pair is worth -0.55 to -1.07, i.e. 5 - 11 kT.

| system (MD, 199 frames) | standalone mean E_cx | LAMMPS mean E_cx | frames with a difference | largest single-frame difference |
|---|---:|---:|---:|---:|
| bulge, oxDNA2, 300 K | -0.345 | 0 | 115 | 0.888 |
| bulge, oxDNA3, 300 K | -0.110 | 0 | 34 | 1.341 |
| 12-bp duplex, oxDNA3, T = 0.12 (fraying, flipped bases) | -0.057 | 0 | 16 | 1.068 |
| 12-bp duplex, 300 K; nicked duplex; blunt-end stacking | same | same | 0 | < 2e-4 |

Reproducers (`expected_pairs.txt` has the pairs):
- `cases/q3_coax_bulge_oxdna2`: pair (5, 7) across the bulge nucleotide 6,
  E_cx = -0.888 standalone vs 0 in LAMMPS.
- `cases/q3_coax_bulge_oxdna3`: pairs (6, 8) and (7, 9) across a broken stacking
  step next to the bulge, -1.341 vs 0.
- `cases/q3_coax_flipped_base_oxdna3`: nucleotide 7 of a plain 12-bp duplex
  flipped out at T = 0.12 (no stacking, no H-bond); pair (6, 8), -1.068 vs 0.

The same geometry occurs at bulges, internal loops, flipped or mismatched
bases, and across DNA-origami crossovers, where the stacked helix continues
through non-terminal nucleotides.

**Mirrored theta4 lobe: occasional LAMMPS-only energy.** When the two ends of a
single strand meet antiparallel, LAMMPS stacks them coaxially through the
mirrored lobe; standalone oxDNA gives 0. `cases/q3_coax_mirror_lobe_ss_ends_oxdna2`:
- 20-nt strand whose 5' and 3' ends meet with theta4 = 143 deg;
- E_cx = -0.1195 in LAMMPS, 0 in standalone oxDNA;
- the difference disappears with `OXDNA_COAX_NOMIRROR` and is unchanged by
  `OXDNA_COAX_NOTERM`.

It occurred in 2 of 199 frames of that run.
