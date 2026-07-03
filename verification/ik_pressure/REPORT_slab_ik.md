# Verification: IK pressure profile for the smooth-damped slab dispersion method

Verified `ewald/disp/planar`, `pppm/disp/planar`, `pppm/disp/planar/kk` (pair
`lj/cut/dispswitch`) and fixed two gaps in the local IK pressure-profile support.

## Method
The pair does full LJ to `rcut` and fades the 1/r^6 dispersion out by `(1-S)` over
`[rcut, rcut+Delta]`; the kspace reciprocal represents the complementary smooth
tail `u_smooth(r) = S*(-4/r^6)` (0 at rcut, full beyond `rcut+Delta`), with the
slab correction folded into the reciprocal coefficients (merged corr, no separate
shell). By the Ewald identity the kspace IK profile equals a direct real-space IK
pair sum of `u_smooth` (no shell subtraction).

## Fixes (committed to the slab source)
1. **ewald/disp/planar had no `pressure_profile_long`** (dropped in the smooth-damped
   prune, only re-ported to pppm) -> `compute stress/cartesian ... kspace` errored.
   Restored, adapted to merged-damped (K = kcount-1, no shell).
2. **pppm/disp/planar/kk had no `pressure_profile_long` override** -> inherited the
   host base reading `atom->x` with no device sync (GPU hazard). Added the sync.

## Results (CPP2 rerun, rcut=4.0)
| check | result |
|---|---|
| ewald/disp/planar IK hook runs | yes (was erroring) |
| ewald vs pppm IK (geometric) | rms 3e-7 |
| ewald vs pppm IK (arith nchan=7) | rms 1.8e-9 |
| pppm/disp/planar/kk vs host | bit-identical (max\|d\|=0), GPU-safe |
| IK vs brute IK of u_smooth (Ewald id) | converges (ratio 0.63->0.94, RMAX 8->14) |
| LB mixing IK vs brute (xy-disordered fluid) | converges (ratio->0.94) |
| full pair+kspace = full-LJ IK | P_N-P_T ratio 1.0002, P_N converges |
| H vs IK contour-invariant gamma | match (0.07024 vs 0.07036) |
| ewald vs pppm global energy/pressure/gamma | match to ~1e-7 |

Reproduce: `in.slab_*`, `verify_slab_ik.py`, `verify_full_workflow.py`,
`verify_arith.py`, `verify_slab_plots.py`; figures `fig_slab_crosscode.png`,
`fig_slab_lb.png`, `fig_full_workflow.png`.
