#!/usr/bin/env python3
"""All-methods mu0(rho) recovery WITH block-bootstrap error bars, for the cubic
N=100 CPP torture runs.  Usage: cube100_boot.py <tag> <dumax>
(tag = cube100 or cube100u4).  Each fix ave/* window is a block; we resample the
blocks with replacement, recompute mu0(rho) on a fixed grid per resample, and band
= +/- std over resamples.  The dump-based fluctuation method blocks the frames."""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, field_coupling as fc
import pets_eos as pets
import thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else 'cube100'
dumax = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
T = float(sys.argv[3]) if len(sys.argv) > 3 else 1.198   # 1.089 for the Tc runs
L = 6.8582414181223398941; Lz = L; area = L*L
NB = 200
nboot = 200
rng = np.random.default_rng(0)
grid = np.linspace(0.14, 0.56, 40)
sc = grid; mp = np.array([T*np.log(x) + pets.properties(T, x)['mu_res'] for x in sc])


def read_chunk_blocks(fn):
    blocks, cur = [], None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 3:
            cur = []; blocks.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return [np.array(b) for b in blocks]


def anchor_rms(rg, mu):
    o = np.argsort(rg); rg, mu = rg[o], mu[o]
    mua = mu + (np.interp(np.median(sc), sc, mp) - np.interp(np.median(sc), rg, mu))
    return np.interp(grid, rg, mua), np.sqrt(np.mean((np.interp(sc, rg, mua) - mp)**2))


def p0_to_mu0_rhospace(rho, P0):
    """rho-space contour integration: smooth P0(rho), mu0 = INT dP0/rho."""
    o = np.argsort(rho); rs, ps = rho[o], P0[o]
    m = (rs > 0.10) & (rs < 0.58)
    rs, ps = rs[m], ps[m]
    if len(rs) < 8:
        return None
    cf = np.polyfit(rs, ps, 4); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(max(rs.min(), 0.12), min(rs.max(), 0.57), 120)
    mu0 = np.concatenate([[0], np.cumsum(0.5*(dP(rg[1:])/rg[1:]+dP(rg[:-1])/rg[:-1])*np.diff(rg))])
    return rg, mu0


def h_mu0(st, de):
    nb = len(st); Vbin = area*(Lz/nb)
    rho = oz.fourier_cosine_smooth(de[:, 3], 6)
    PN = oz.fourier_cosine_smooth(-st[:, 5]/Vbin, 6) + rho*T
    PT = oz.fourier_cosine_smooth(-0.5*(st[:, 3]+st[:, 4])/Vbin, 6) + rho*T
    return p0_to_mu0_rhospace(rho, 1.5*PT - 0.5*PN)


def read_vec_blocks_keepidx(fn):     # ave/time vector, keep the row-index column
    blocks, cur = [], None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 2:
            cur = []; blocks.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    return [np.array(b) for b in blocks]


def ik_mu0(ik):                       # IK stress/cartesian block (9 cols incl. index)
    rho = oz.fourier_cosine_smooth(ik[:, 2], 6)
    PN = oz.fourier_cosine_smooth(ik[:, 5] + ik[:, 8], 6)
    PT = oz.fourier_cosine_smooth(0.5*((ik[:, 3]+ik[:, 6]) + (ik[:, 4]+ik[:, 7])), 6)
    return p0_to_mu0_rhospace(rho, 1.5*PT - 0.5*PN)


SFBIN = 16
def ozkb_mu0(sf):                     # DCF Kirkwood-Buff route from S_ij blocks
    qs, Sm, rho = oz.assemble_matrices(sf, SFBIN)
    Sm, rho = oz.mirror_symmetrize(Sm, rho)
    dz = Lz/SFBIN
    active = np.where(rho > 0.10)[0]
    Carr = {q: oz.invert_oz(Sm[q], rho, dz, area, active=active, ridge=1e-3)[0] for q in qs}
    chat0 = oz.local_chat0(qs, Carr, active, SFBIN, 2.5, dz)
    return oz.kb_chemical_potential(rho, chat0, T)


def fc_grad_mu0(de):
    rho = oz.fourier_cosine_smooth(de[:, 3], 6)
    amps = np.array([0.0, dumax/2.0])
    prof = np.array([np.full_like(rho, rho.mean()), rho])
    e = fc.local_eos(amps, prof, T, Lz, deg=6, smooth=6, grad_spec={2: 0, 4: 0})
    return e['rho'], e['mu0']


def boot(blocks_list, fn):
    """blocks_list: list of per-block arrays (or tuple of lists for multi-input);
    fn: averaged-block(s) -> (rho, mu0).  Returns (mean_on_grid, std_on_grid, rms)."""
    multi = isinstance(blocks_list, tuple)
    nb = len(blocks_list[0]) if multi else len(blocks_list)
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, nb, nb)
        if multi:
            args = tuple(np.mean([bl[i] for i in idx], axis=0) for bl in blocks_list)
            res = fn(*args)
        else:
            res = fn(np.mean([blocks_list[i] for i in idx], axis=0))
        if res is None:
            continue
        g, _ = anchor_rms(*res)
        out.append(g)
    out = np.array(out)
    full = fn(*[np.mean(bl, axis=0) for bl in blocks_list]) if multi else fn(np.mean(blocks_list, axis=0))
    gmean, rms = anchor_rms(*full)
    return gmean, out.std(0), rms


curves = {}
# Harasima contour (rho-space) -- the winner
try:
    hb = read_chunk_blocks('%s_hstress.out' % tag)
    db = read_chunk_blocks('%s_dens.out' % tag)
    n = min(len(hb), len(db)); hb, db = hb[:n], db[:n]
    curves['Harasima contour'] = (boot((hb, db), h_mu0), 'tab:green')
    print('Harasima: %d blocks' % n)
except Exception as e:
    print('Harasima failed:', e)
# IK contour (rho-space)
try:
    ikb = read_vec_blocks_keepidx('%s_ikstress.out' % tag)
    curves['IK contour'] = (boot(ikb, ik_mu0), 'tab:blue')
    print('IK: %d blocks' % len(ikb))
except Exception as e:
    print('IK failed:', e)
# OZ-KB (DCF)
try:
    sfb = oz.read_ave_time_blocks('%s_sf.out' % tag)
    curves['OZ-KB (DCF)'] = (boot(sfb, ozkb_mu0), 'tab:purple')
    print('OZ-KB: %d blocks' % len(sfb))
except Exception as e:
    print('OZ-KB failed:', e)
# FC-gradient
try:
    db = read_chunk_blocks('%s_dens.out' % tag)
    curves['FC-gradient'] = (boot(db, fc_grad_mu0), 'tab:orange')
except Exception as e:
    print('FC-gradient failed:', e)

# plot
plt.figure(figsize=(8.5, 6))
rr = np.linspace(0.06, 0.58, 200)
plt.plot(rr, [T*np.log(x)+pets.properties(T, x)['mu_res'] for x in rr], 'k-', lw=2.5, label='PeTS EOS')
plt.plot(rr, [T*np.log(x)+thol.properties(T, x)['mu_res'] for x in rr], '--', color='dimgray', lw=2.0, label='Thol 2015 EOS')
for lab, ((gm, gs, rms), col) in curves.items():
    plt.fill_between(grid, gm-gs, gm+gs, color=col, alpha=0.25)
    plt.plot(grid, gm, 'o-', ms=3, color=col, label='%s (RMS %.3f$\\pm$%.3f)' % (lab, rms, np.median(gs)))
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored)')
plt.title(r'%s ($\Delta U=%g$): mu0 vs PeTS with block-bootstrap errors' % (tag, dumax))
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('%s_boot.png' % tag, dpi=140)
print('wrote %s_boot.png' % tag)
for lab, ((gm, gs, rms), col) in curves.items():
    print('  %-18s RMS=%.4f   median bootstrap sigma=%.4f' % (lab, rms, np.median(gs)))
