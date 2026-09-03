# Orchestrator state (read this first after any restart)

Task: full line-by-line review of src/KOKKOS for the LAMMPS stable release, using Opus agents
in waves, then independent verification of findings, then a consolidated report for the user.

Groups: group_00..group_29 (file lists). Group G is done when progress_G/COMPLETE exists;
its findings are in findings_G.json (or progress_G/findings.jsonl if the agent died late).
Rule audits: rules_A (copymode/virtual), rules_B (DAT/AT, device math, instantiation),
rules_C (datamask/sync), rules_D (packaging/docs) -> findings_rules_X.json, progress_rules_X/.

Relaunch protocol: for any group without COMPLETE, launch an Opus agent with the standard
prompt in LAUNCH.md; the agent resumes from progress_G/. Keep at most 4-6 agents in flight
(16 at once tripped "session limit, resets 8:30am UTC" on 2026-09-03 ~12:20 UTC; a probe at
12:35 UTC worked again, so it behaved like a burst limit).

Earlier result: /code-review skill pass on the PR #5150 diff only (not the full package)
found 1 confirmed bug: atom_map_kokkos.cpp:404 map_one() modify_host() + later
modify_device() -> dual_hash_type abort "Concurrent modification of host and device hashes"
with `package kokkos atom/map device` + hash map (Special::build, FixShake rendezvous).
Plus cleanup items: modify_kokkos.cpp:66 ALL_MASK modified claim in setup() for non-kk
computes, fix_shake_kokkos.cpp:1739 grow_arrays comment wrong about DualView::resize,
comm_kokkos.cpp:736 stale comment about hybrid/scaled, pair_pace neigh_scratch_level_select
duplicated in pair_pace_extrapolation, fix_deform_kokkos dead pre_exchange override,
unparenthesized || operands comm_kokkos.cpp:742 / comm_tiled_kokkos.cpp:320.

Phase log:
- 2026-09-03 12:40 UTC: wave 1 (groups 00-03) launched; checkpoint protocol sent to them at 12:50.
- 2026-09-03 12:50 UTC: rule audits A and B launched (6 agents in flight). C and D still to launch.
- Orchestrator-verified findings so far: findings_orchestrator.json (atom_map hash abort = high, confirmed).
- 13:05 UTC: group 02 COMPLETE (16 findings, 6 high: atom_vec_kokkos PackBorderVel deform branch, PackCommVel _mu/_sp uninit, unpack_border_vel ncomm_vel vs nborder_vel, field2size num_improper, RADIUS/RMASS forward comm missing, ellipsoid bonus comm stride). Group 04 launched.
- 13:10 UTC: group 00 COMPLETE (11 findings, 2 high: angle_gaussian coeff ndihedraltypes bound, angle_cosine_shift_exp SMALL=1e-12). Group 05 launched.
- 13:25 UTC: group 01 COMPLETE (11 findings, 2 high: angle_hybrid_kokkos cvatom null deref with centroid/stress/atom, angle_spica_kokkos init_style null lj tables when repflag=0; medium: spica missing kokkosable=1). Group 06 launched. In flight: 03, 04, 05, 06, rules A, rules B.
- 13:35 UTC: group 03 COMPLETE (15 findings, 1 high: bond_quartic_kokkos k_brokenflag never modify<Device>() -> no bonds break on GPU; medium: bond_hybrid subview modify_device clobbers quartic host writes, missing partial_flag write-back, FENE d_flag races). Group 07 launched. In flight: 04, 05, 06, 07, rules A, rules B.
- 13:45 UTC: rules_B COMPLETE (2 low/medium: fix_eos_table_rx std::isnan in device fn, npair_ssa debug fprintf; DAT/AT, instantiation, tokens clean). rules_C launched. In flight: 04, 05, 06, 07, rules A, rules C.
- 13:55 UTC: rules_A COMPLETE (6: dynamical_matrix/third_order setup() non-virtual -> kk setup dead, fix_wall_gran_kokkos copymode leak on error path, FixShardlow/PairPOD dtor guards, copymode=1 before error->all in many styles). rules_D launched. In flight: 04, 05, 06, 07, rules C, rules D.
- ~14:05 UTC: second usage-limit kill (6 agents), reset 17:30 UTC. Checkpoints held (04: 37/48, 05: 12/26, 06: 11/17, 07: 11/20, rules C/D partial).
- 17:35 UTC: relaunched 04, 05, 06, 07, rules C, rules D as resumers. Observed budget: roughly 3M Opus tokens (~10 agent-runs) per 5-hour window.
- Orchestrator (Fable) reviews group 08 files directly meanwhile, checkpointing into progress_08/ (verlet_kokkos first).
- 18:40 UTC: orchestrator finished hand review of group 08 except pack_kokkos.h/sna_kokkos*.h (agent launched for those). Orchestrator findings in progress_08/findings.jsonl (neighbor_kokkos build_topology host sync gap = medium; tune_kokkos rank-0 error->all = medium; rest low). 7 agents in flight: 04, 05, 06, 07, 08-rest, rules C, rules D.
- 18:55 UTC: rules_D COMPLETE (13 findings; HIGH npair_halffull_kokkos.h:221 wrong class for triclinic trim/skip, verified by orchestrator; rest low doc/packaging/ascii). Group 09 launched. In flight: 04, 05, 06, 07, 08-rest, 09, rules C.
- 19:00 UTC: group 04 COMPLETE (38 findings, 18 high: compute_sna_grid*/gaussian_grid_local kokkos many defects (group filter dropped, cutsq[1][1] for all types, radelem sized nelements filled ntypes, PreUi type(iatom) on grid index, max_neighs=100 unbounded, host_flag fallbacks broken, alocal double-alloc); coord/atom multi-column group filter; hexorder /nnn; inertia sphere; orientorder stale ncount). Group 10 launched. In flight: 05, 06, 07, 08-rest, 09, 10, rules C.
- 19:05 UTC: group 07 COMPLETE (25 findings; high: meam/kk d_scale int view + missing force scaling block + scale(i,i) + eatom phi vs phi_sc; medium: kokkos.cpp:824 pair/only guard inverted, SLURM_LOCALID g 0 div0, memory_kokkos printf + create4d_offset no offsets + 3d overload 2 extents, kissfft bfly_generic shared scratch (latent)). Group 11 launched. In flight: 05, 06, 08-rest, 09, 10, 11, rules C.
- 19:10 UTC: group 06 COMPLETE (15 findings; high: atom_map_kokkos map_set_device sametag realloc before map_init (verified), comm_tiled_kokkos self-send never unpacked (verified), pppm_kokkos reset_grid not overridden (likely); medium: group_kokkos angmom no deform velocity, comm_kokkos device MAP_ARRAY never cleared, fft3d norm int). Group 12 launched (wave 3 begins). In flight: 05, 08-rest, 09, 10, 11, 12, rules C.
- 19:20 UTC: rules_C COMPLETE (16: high fix_wall_flow V_MASK read missing, fix_wall_region RADIUS_MASK, fix_nve_limit k_mass unsynced; medium k_mass family (shake, gravity, momentum, dt_reset, electron_stopping, shardlow), pair_dpd TAG_MASK, multi_lucy_rx/exp6_rx cutsq modify only in allocate). All 4 rule audits done.
- 19:25 UTC: group 05 COMPLETE (12: high compute_temp_sphere strcmp vs /kk suffix (CPU file), verlet_kokkos overlap merge drops legacy Host forces in transform builds (likely), min_linesearch xvec flat alias layout; medium s_KK_double2 float accumulators, cg/sd unsynced f dot products). Groups 13, 14 launched. In flight: 08-rest, 09, 10, 11, 12, 13, 14.
- 19:40 UTC: group 08 COMPLETE (16 incl. orchestrator's; new high: sna_kokkos_impl.h:1118 bzero subtraction on every diagonal triple vs only ielem; medium: pack_kokkos permute*_n nqty factor on stride (latent), uncoalesced pack index order). Group 15 launched. In flight: 09, 10, 11, 12, 13, 14, 15.
- 19:45 UTC: group 09 COMPLETE (4: high dihedral_hybrid_kokkos cvatom null deref (same as angle/improper hybrid); medium dihedral_fourier allocate_kokkos resize zeroes host mirror). Group 16 launched. In flight: 10, 11, 12, 13, 14, 15, 16.
- 19:55 UTC: group 10 COMPLETE (8: medium pair_zbl_kokkos applies special_lj unlike CPU; dihedral_nharmonic/spherical resize wipes host mirror; Install.sh item duplicates rules_D and overstates severity (legacy build refused) -> dedupe). Group 17 launched. In flight: 11, 12, 13, 14, 15, 16, 17.
- ~20:00 UTC: third usage-limit kill (7 agents), reset 22:30 UTC. Checkpoints: 11: 20/40, 12: 22/58, 13: 20/24, 14: 6/8, 15: 2/32, 16: 6/34, 17: 8/24.
- 22:35 UTC: relaunched 11-17 as resumers. Remaining unlaunched: 18 (npair/pppm), 19-29 (pair). Orchestrator verifying group 00-03 highs directly meanwhile (results -> findings_orchestrator.json).
- 22:50 UTC: orchestrator verified (direct read) 21 findings into findings_orchestrator.json: atom_vec x5, angle x4, spica kokkosable, bond_quartic, sna_grid cluster, meam scale cluster, kokkos.cpp pair/only, coord_atom, hexorder (+ earlier core ones). NOTE: group 04 agent line numbers for coord_atom/hexorder/inertia were off by ~70 lines; verification must locate by symbol.
- 23:05 UTC: findings_orchestrator.json now holds 34 orchestrator-verified findings (all highs from groups 00-10 + rules + core). Remaining verification (phase 6) for medium/low items and groups 11-29 will use independent agents once the group reviews finish.
- 23:15 UTC: group 17 COMPLETE (7: high improper_hybrid cvatom (same family), improper_cvff host sign[] in kernel (verified); medium distance/distharm/sqdistharm no minimum_image). Group 18 launched. In flight: 11, 12, 13, 14, 15, 16, 18.
- 23:20 UTC: USER INSTRUCTION: stop launching agents. Do NOT launch any further agents (no group 19-29, no verification agents). Let the 7 in flight (11-16, 18) finish and collect their output; then consolidate and write the report from what exists. Pair styles (groups 19-29) remain unreviewed unless the user says otherwise.
- 22:50 UTC: groups 12 (17 findings, 9 high: fix_gravity disable ignored, fix_langevin scale reset + host omega_thermostat, fix_neigh_history retry stale views + no kk restart, fix_nh isochoric dropped, fix_nve_sphere dipole/dlm ignored, fix_nve_asphere superellipsoid, fix_nve_limit k_mass) and 13 (14 findings, 1 high: fix_property_atom_kokkos dtor use-after-free; medium reaxff/bonds NULL file, reaxff/species && vs ||, fix_recenter firstgroup, qeq/reaxff exchange space) COMPLETE. Still running: 11, 14, 15, 16, 18. NO NEW LAUNCHES (user instruction).
- 22:55 UTC: snapshot committed to the repo as kokkos-review/ on branch claude/kokkos-code-review-6lcg8d (af823198e) and pushed. Re-snapshot (cp -r scratchpad review -> kokkos-review/, drop scan.json/scan4.json/scan_raw.json, commit, push) after each remaining group (11, 14, 15, 16, 18) reports. If the container resets: the scratchpad is gone; resume from kokkos-review/ in the repo.
- 23:10 UTC: group 11 COMPLETE (15 findings, 6 high: fix_deform_kokkos update_box host bracket reverts device remap with rigid fixes; fix_dt_reset/fix_electron_stopping k_mass unsynced; fix_eos_table_rx lo/hi int views (+ int locals); fix_efield variables evaluated on stale host x; medium acks2 '|| 1' dead GPU path, cmap ctor leaks base arrays, addtorque/atom storque null). Still running: 14, 15, 16, 18.
- 23:30 UTC: group 14 COMPLETE (8; likely-high fix_rigid_small grow_arrays discards host bookkeeping on host exchange path). Group 16 COMPLETE (21; high: mliap_so3 int weight, mliap_unified double Py_DECREF, fix_wall_region host region pointer in kernel; medium mliap_data elems over nmax, wall/gran/old vwall +=, wall/reflect modified() before kernels, mliap linear default exec space, so3 m_dclist not regrown). Still running: 15, 18.
- 23:35 UTC: group 15 COMPLETE (21; high: fix_shake min_post_force ghost index + no-op reverse_comm, fix_spring_self modify_host clobbers device xoriginal, fix_wall_gran pack/unpack_exchange wrong offsets (x2) and no force for granular/region, fix_spring stale host image/type/rmass; medium: shake stats never reduced, shardlow k_mass, store_force array_atom dangling, wall_flow init() drops checks, wall_gran limit_damping ignored + contacts arrays). Only group 18 still running.
- Next: as each group completes, launch the next group in order 04..29, then rules C, D; then verification agents.
