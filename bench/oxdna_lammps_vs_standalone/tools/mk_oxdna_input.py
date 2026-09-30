#!/usr/bin/env python3
"""Write a standalone-oxDNA input file.
usage: mk_oxdna_input.py <model 2|3> <mode md|eval> <top> <conf> <tag> [steps] [T] [seed] [seq_dep_file]
  md:   Brownian (john) MD, trajectory traj_<tag>.dat (every steps/200)
  eval: one single-point evaluation of <conf>: potential_energy split -> pe_<tag>.dat
"""
import sys
model, mode, top, conf, tag = sys.argv[1:6]
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 200000
T = sys.argv[7] if len(sys.argv) > 7 else '0.1'
seed = sys.argv[8] if len(sys.argv) > 8 else '4242'
seqf = sys.argv[9] if len(sys.argv) > 9 else ''
L = ['backend = CPU', 'sim_type = MD', f'interaction_type = DNA{model}_nomesh' if mode == 'eval' else f'interaction_type = DNA{model}',
     f'T = {T}', 'salt_concentration = 0.5', f'topology = {top}', f'conf_file = {conf}', 'refresh_vel = 1',
     'restart_step_counter = 1', 'time_scale = linear', 'verlet_skin = 0.2']
if mode == 'eval': L = [l.replace('refresh_vel = 1', 'refresh_vel = 0') for l in L]
if model == '3':
    L += ['use_average_seq = 0', f'seq_dep_file = {seqf}']
else:
    L += ['use_average_seq = 1']
if mode == 'md':
    L += ['dt = 0.003', f'steps = {steps}', 'thermostat = john', 'newtonian_steps = 103', 'diff_coeff = 2.5',
          f'seed = {seed}', f'trajectory_file = traj_{tag}.dat', f'lastconf_file = last_{tag}.dat',
          f'energy_file = en_{tag}.dat', f'print_energy_every = {steps // 200}', f'print_conf_interval = {steps // 200}']
else:
    L += ['dt = 1e-9', 'steps = 1', 'thermostat = no', 'trajectory_file = /dev/null', 'lastconf_file = /dev/null',
          'energy_file = /dev/null', 'print_energy_every = 1', 'print_conf_interval = 100000',
          'data_output_1 = {', f'  name = pe_{tag}.dat', '  print_every = 1', '  only_last = 1', '  col_1 = {',
          '    type = potential_energy', '    split = true', '    precision = 15', '  }', '}']
print('\n'.join(L))
