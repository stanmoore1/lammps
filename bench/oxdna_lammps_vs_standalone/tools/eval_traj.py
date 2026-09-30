#!/usr/bin/env python3
"""Evaluate every frame of a standalone-oxDNA trajectory with standalone oxDNA and LAMMPS.

usage: eval_traj.py <model 2|3> <top> <traj> <workdir> [first_frame] [nproc]
env:   OXDNA  standalone oxDNA binary          LMP  LAMMPS binary (CG-DNA, ASPHERE, MOLECULE)
       SEQ    oxDNA3 sequence-dependence file  (model 3)

Per frame k (frames/f<k>/): conf.dat, data.lmp (tools/ox2lmp.py) and
  standalone: DNA<m>_nomesh, potential_energy split (pe_ox.dat) and pair_energy (pairs_ox.dat)
  LAMMPS:     lammps/in.eval<m> (compute pair per style), variants
    m=2: lmp   stock oxDNA2 coefficients
         lmpV  + OXDNA_COAX_NOTERM OXDNA_COAX_NOMIRROR   (verification build only)
    m=3: lmp   stock potentials/oxdna3_lj.cgdna
         lmpT  oxdna3_lj_ts.cgdna (theta smoothing points = sqrt(0.81225/a), tools/patch_potfile.py)
         lmpV  oxdna3_lj_ts.cgdna + OXDNA3_XSTK_GATE + OXDNA_COAX_NOTERM + OXDNA_COAX_NOMIRROR
Writes <workdir>/summary.txt: one line per frame with the total hb, xstk and coaxstk energies.
(The V variants need the verification-patched LAMMPS; with a stock LAMMPS they equal lmpT / lmp.)
"""
import os, sys, subprocess, shutil
from multiprocessing import Pool
H = os.path.dirname(os.path.abspath(__file__))
model, top, traj, work = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3], os.path.abspath(sys.argv[4])
first = int(sys.argv[5]) if len(sys.argv) > 5 else 1
nproc = int(sys.argv[6]) if len(sys.argv) > 6 else 4
OX, LMP, SEQ = os.environ['OXDNA'], os.environ['LMP'], os.environ.get('SEQ', '')
N = int(open(top).readline().split()[0])
L = open(traj).read().split('\n')
frames = [L[k:k + 3 + N] for k in range(0, len(L) - 3, 3 + N) if L[k].startswith('t =')]
VAR = {'2': [('lmp', {}, None), ('lmpV', {'OXDNA_COAX_NOTERM': '1', 'OXDNA_COAX_NOMIRROR': '1'}, None)],
       '3': [('lmp', {}, 'oxdna3_lj.cgdna'), ('lmpT', {}, 'oxdna3_lj_ts.cgdna'),
             ('lmpV', {'OXDNA3_XSTK_GATE': '1', 'OXDNA_COAX_NOTERM': '1', 'OXDNA_COAX_NOMIRROR': '1'}, 'oxdna3_lj_ts.cgdna')]}


def one(k):
    d = f'{work}/frames/f{k}'; os.makedirs(d, exist_ok=True)
    open(f'{d}/conf.dat', 'w').write('\n'.join(frames[k]) + '\n')
    shutil.copy(top, f'{d}/top.top')
    subprocess.run(['python3', f'{H}/ox2lmp.py', 'top.top', 'conf.dat', 'data.lmp'], cwd=d, check=True)
    inp = subprocess.run(['python3', f'{H}/mk_oxdna_input.py', model, 'eval', 'top.top', 'conf.dat', 'ox', '1', '0.1', '1', SEQ],
                         capture_output=True, text=True, check=True).stdout
    inp += '\ndata_output_2 = {\n  name = pairs_ox.dat\n  print_every = 1\n  only_last = 1\n  col_1 = {\n    type = pair_energy\n  }\n}\n'
    open(f'{d}/in_ox', 'w').write(inp)
    subprocess.run([OX, 'in_ox'], cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    ox = [float(x) * N for x in open(f'{d}/pe_ox.dat').read().split()]   # fene bexc stck nexc hb crst cxst dh
    res = {'ox': (ox[4], ox[5], ox[6])}
    for tag, env, pot in VAR[model]:
        cmd = [LMP, '-in', f'{H}/../lammps/in.eval{model}', '-var', 'data', 'data.lmp', '-var', 'tag', tag,
               '-log', 'none', '-screen', 'none']
        if pot:
            shutil.copy(f'{H}/../lammps/{pot}', d); cmd += ['-var', 'potfile', pot]
        subprocess.run(cmd, cwd=d, env={**os.environ, **env}, check=True)
        e = [float(x) for x in open(f'{d}/energy_{tag}.txt').read().split()[2:]]   # bond excv stk hb xstk cx dh pe
        res[tag] = (e[3], e[4], e[5])
    return k, frames[k][0].split('=')[1].strip(), res


if __name__ == '__main__':
    os.makedirs(work, exist_ok=True)
    with Pool(nproc) as pool:
        out = pool.map(one, range(first, len(frames)))
    tags = ['ox'] + [t for t, _, _ in VAR[model]]
    with open(f'{work}/summary.txt', 'w') as f:
        f.write('# frame time ' + ' '.join(f'hb_{t} xstk_{t} cx_{t}' for t in tags) + '\n')
        for k, t, r in out:
            f.write(f'{k} {t} ' + ' '.join('%.10f %.10f %.10f' % r[x] for x in tags) + '\n')
    print(f'{len(out)} frames -> {work}/summary.txt')
