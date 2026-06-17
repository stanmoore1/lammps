#!/usr/bin/env python3
"""P0(rho) from the IK and Harasima pressure contours with block-bootstrap error
bands, vs BOTH LJTS EOS (PeTS + Thol 2015).  Usage: cube100_p0_boot.py <tag> <dumax>"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, pets_eos as pets, thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else 'cube100'
dumax = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
T = float(sys.argv[3]) if len(sys.argv) > 3 else 1.198   # 1.089 for the Tc runs
L = 6.8582414181223398941; Lz = L; area = L*L
nboot = 200; rng = np.random.default_rng(0); grid = np.linspace(0.10, 0.57, 45)
sm = lambda y: oz.fourier_cosine_smooth(y, 6)


def chunk_blocks(fn):
    bl, cur = [], None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 3:
            cur = []; bl.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return [np.array(b) for b in bl]


def vec_blocks(fn):
    bl, cur = [], None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 2:
            cur = []; bl.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return [np.array(b) for b in bl]


def h_p0(st, de):
    nb = len(st); V = area*(Lz/nb); rho = sm(de[:, 3])
    PN = sm(-st[:, 5]/V) + rho*T
    PT = sm(-0.5*(st[:, 3]+st[:, 4])/V) + rho*T
    return rho, 1.5*PT - 0.5*PN


def ik_p0(ik):
    rho = sm(ik[:, 2])
    PN = sm(ik[:, 5] + ik[:, 8])
    PT = sm(0.5*((ik[:, 3]+ik[:, 6]) + (ik[:, 4]+ik[:, 7])))
    return rho, 1.5*PT - 0.5*PN


def band(blocks, fn, paired=None):
    nb = len(blocks) if paired is None else min(len(blocks), len(paired))
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, nb, nb)
        if paired is None:
            r, p = fn(np.mean([blocks[i] for i in idx], 0))
        else:
            r, p = fn(np.mean([blocks[i] for i in idx], 0), np.mean([paired[i] for i in idx], 0))
        o = np.argsort(r); out.append(np.interp(grid, r[o], p[o]))
    out = np.array(out); return out.mean(0), out.std(0)


hb = chunk_blocks('%s_hstress.out' % tag); db = chunk_blocks('%s_dens.out' % tag)
ikb = vec_blocks('%s_ikstress.out' % tag)
hm, hs = band(hb, h_p0, paired=db)
im, iss = band(ikb, ik_p0)
print('blocks: H=%d IK=%d ; bootstrap sigma IK=%.4f H=%.4f'
      % (min(len(hb), len(db)), len(ikb), np.median(iss), np.median(hs)))

rr = np.linspace(0.05, 0.58, 200)
plt.figure(figsize=(8, 5.5))
plt.plot(rr, [pets.properties(T, x)['p'] for x in rr], 'k-', lw=2.5, label='PeTS EOS')
plt.plot(rr, [thol.properties(T, x)['p'] for x in rr], '--', color='dimgray', lw=2, label='Thol 2015 EOS')
plt.fill_between(grid, im-iss, im+iss, color='tab:blue', alpha=.25)
plt.plot(grid, im, 'o-', ms=3, color='tab:blue', label=r'IK contour $P_0$')
plt.fill_between(grid, hm-hs, hm+hs, color='tab:green', alpha=.25)
plt.plot(grid, hm, 's-', ms=3, color='tab:green', label=r'Harasima contour $P_0$')
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$P_0^*$'); plt.legend(); plt.grid(alpha=.3)
plt.title(r'%s ($\Delta U=%g$): contour $P_0$ vs both LJTS EOS (bootstrap bands)' % (tag, dumax))
plt.tight_layout(); plt.savefig('%s_p0_vs_rho.png' % tag, dpi=140)
print('wrote %s_p0_vs_rho.png' % tag)
