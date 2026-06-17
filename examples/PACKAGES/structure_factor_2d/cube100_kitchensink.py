#!/usr/bin/env python3
"""Kitchen-sink mu0(rho) recovery over the FULL density range (0.02-0.67) for the
N=100 cubic dUmax=4 torture run, with block-bootstrap bands and BOTH LJTS EOS, plus
a per-rho deviation diagnosis of where/why each method breaks.  Usage: <tag> <dumax>"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, field_coupling as fc, contour_pressure as cp
import pets_eos as pets, thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else 'cube100u4'
dumax = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
T = 1.198; L = 6.8582414181223398941; Lz = L; area = L*L
SFBIN = 16; nboot = 120; rng = np.random.default_rng(0)
grid = np.linspace(0.03, 0.65, 80)
sm = lambda y, n=6: oz.fourier_cosine_smooth(y, n)
pmu = lambda r: T*np.log(r) + pets.properties(T, r)['mu_res']
mp_grid = np.array([pmu(x) for x in grid])
tmu = lambda r: T*np.log(r) + thol.properties(T, r)['mu_res']


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


def anchor(rg, mu):
    o = np.argsort(rg); rg, mu = rg[o], mu[o]
    rm = (grid > rg.min()) & (grid < rg.max())
    g = np.interp(grid, rg, mu)
    g[~rm] = np.nan
    sh = np.nanmedian(mp_grid - g)
    return g + sh


# --- methods: each maps averaged-block(s) -> (rho, mu0) over the FULL z-profile ---
def h_contour(st, de):                 # z-space integration -> full rho coverage
    nb = len(st); V = area*(Lz/nb); rho = sm(de[:, 3])
    PN = sm(-st[:, 5]/V) + rho*T; PT = sm(-0.5*(st[:, 3]+st[:, 4])/V) + rho*T
    r, mu, _ = cp.mu0_from_p0(rho, 1.5*PT-0.5*PN, Lz, 6)
    return r, mu


def ik_contour(ik):
    rho = sm(ik[:, 2]); PN = sm(ik[:, 5]+ik[:, 8])
    PT = sm(0.5*((ik[:, 3]+ik[:, 6])+(ik[:, 4]+ik[:, 7])))
    r, mu, _ = cp.mu0_from_p0(rho, 1.5*PT-0.5*PN, Lz, 6)
    return r, mu


def lda(de):                           # mu0(rho(z)) = -U(z) (no inhomogeneity term)
    rho = sm(de[:, 3]); z = (np.arange(len(rho))+0.5)*Lz/len(rho)
    return rho, -0.5*dumax*np.cos(2*np.pi*z/Lz)


def fc_grad(de):
    rho = sm(de[:, 3])
    e = fc.local_eos(np.array([0.0, dumax/2]), np.array([np.full_like(rho, rho.mean()), rho]),
                     T, Lz, deg=6, smooth=6, grad_spec={2: 0, 4: 0})
    return e['rho'], e['mu0']


def ozkb(sf):
    qs, Sm, rho = oz.assemble_matrices(sf, SFBIN); Sm, rho = oz.mirror_symmetrize(Sm, rho)
    dz = Lz/SFBIN; active = np.where(rho > 0.06)[0]
    Carr = {q: oz.invert_oz(Sm[q], rho, dz, area, active=active, ridge=1e-3)[0] for q in qs}
    chat0 = oz.local_chat0(qs, Carr, active, SFBIN, 2.5, dz)
    return oz.kb_chemical_potential(rho, chat0, T)


def boot(fn, *blks):
    nb = min(len(b) for b in blks); out = []
    for _ in range(nboot):
        idx = rng.integers(0, nb, nb)
        r, mu = fn(*[np.mean([b[i] for i in idx], 0) for b in blks])
        out.append(anchor(r, mu))
    out = np.array(out)
    full = fn(*[np.mean(b, 0) for b in blks])
    return anchor(*full), np.nanstd(out, 0)


db = chunk_blocks('%s_dens.out' % tag)
hb = chunk_blocks('%s_hstress.out' % tag)
ikb = vec_blocks('%s_ikstress.out' % tag)
sfb = oz.read_ave_time_blocks('%s_sf.out' % tag)

methods = {
    'Harasima contour': (boot(h_contour, hb, db), 'tab:green'),
    'IK contour':       (boot(ik_contour, ikb), 'tab:blue'),
    'OZ-KB (DCF)':      (boot(ozkb, sfb), 'tab:purple'),
    'FC-gradient':      (boot(fc_grad, db), 'tab:orange'),
    'LDA':              (boot(lda, db), 'tab:gray'),
}

# --- plot full range ---
plt.figure(figsize=(9, 6))
rr = np.linspace(0.03, 0.66, 200)
plt.plot(rr, [pmu(x) for x in rr], 'k-', lw=2.5, label='PeTS EOS')
plt.plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2, label='Thol 2015 EOS')
print('# method            break-rho(|dev|>0.05)    RMS[0.12,0.55]')
for lab, ((gm, gs), c) in methods.items():
    plt.fill_between(grid, gm-gs, gm+gs, color=c, alpha=0.2)
    plt.plot(grid, gm, '-', color=c, lw=1.4, label=lab)
    dev = np.abs(gm - mp_grid)
    core = (grid > 0.12) & (grid < 0.55)
    rms = np.sqrt(np.nanmean(dev[core]**2))
    bad = grid[(dev > 0.05) & ~np.isnan(dev)]
    brk = ('<%.2f or >%.2f' % (bad[bad < 0.3].max() if (bad < 0.3).any() else 0.03,
                               bad[bad > 0.3].min() if (bad > 0.3).any() else 0.66)) if len(bad) else 'none'
    print('  %-16s  %-22s  %.4f' % (lab, brk, rms))
plt.ylim(-3.7, -1.9); plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored)')
plt.title(r'%s ($\Delta U=%g$): every method over the FULL range, vs both EOS' % (tag, dumax))
plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('%s_fullrange.png' % tag, dpi=140)
print('wrote %s_fullrange.png' % tag)
