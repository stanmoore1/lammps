#!/usr/bin/env python3
"""Method #1: local-RDF virial.  A third, independent 2nd-derivative route to the EOS.
Bin atom PAIRS by the local density of the central atom, accumulate g(r;rho), then
the virial pressure
    P0(rho) = rho*kT - (2*pi/3) rho^2 INT r^3 u'(r) g(r;rho) dr,
and mu0 by Gibbs-Duhem.  Uses a config dump (full xyz); the local density per atom is
rho(z_i) from the measured profile.  Independent estimator from the per-atom stress."""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, field_coupling as fc
import pets_eos as pets, thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.198; L = 6.8582414181223398941; Lz = L; rc = 2.5
pmu = lambda r: T*np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T*np.log(r) + thol.properties(T, r)['mu_res']


def parse_xyz(path):
    lines = open(path).read().split('\n'); fr = []; i = 0; n = len(lines)
    while i < n:
        if lines[i].startswith('ITEM: TIMESTEP'):
            na = int(lines[i+3])
            xyz = np.array([[float(v) for v in lines[i+9+j].split()] for j in range(na)])
            fr.append(xyz); i += 9 + na
        else:
            i += 1
    return fr


frames = parse_xyz('cube100u4_xyz.lammpstrj')
print('frames=%d' % len(frames))

# local density per atom from the measured profile rho(z)
prof = oz.fourier_cosine_smooth(fc._read_density('cube100u4_dens.out'), 6)
nz = len(prof); zc = (np.arange(nz)+0.5)*Lz/nz
def rho_of_z(z): return np.interp(z % Lz, zc, prof, period=Lz)

rho_edges = np.linspace(0.04, 0.66, 22)
rho_cen = 0.5*(rho_edges[1:]+rho_edges[:-1])
nr = 120; r_edges = np.linspace(0.0, rc, nr+1); r_cen = 0.5*(r_edges[1:]+r_edges[:-1]); dr = r_edges[1]
pair_hist = np.zeros((len(rho_cen), nr))
n_central = np.zeros(len(rho_cen))

for xyz in frames:
    rho_i = rho_of_z(xyz[:, 2])
    rb = np.clip(np.digitize(rho_i, rho_edges)-1, 0, len(rho_cen)-1)
    # minimum-image pair distances
    d = xyz[:, None, :] - xyz[None, :, :]
    d -= L*np.round(d/L)
    rij = np.sqrt((d**2).sum(-1))
    np.fill_diagonal(rij, 1e9)
    for i in range(len(xyz)):
        rs = rij[i]; m = rs < rc
        if m.any():
            hb = np.clip((rs[m]/dr).astype(int), 0, nr-1)
            np.add.at(pair_hist[rb[i]], hb, 1.0)
        n_central[rb[i]] += 1

# g(r;rho) = counts / (rho * 4 pi r^2 dr * n_central)
shell = 4.0*np.pi*r_cen**2*dr
g = np.zeros_like(pair_hist)
for k in range(len(rho_cen)):
    if n_central[k] > 100:
        g[k] = pair_hist[k]/(rho_cen[k]*shell*n_central[k])

# virial pressure per density:  P0 = rho kT - (2 pi/3) rho^2 INT r^3 u'(r) g dr
def du(r):                                    # LJ u'(r) (shift doesn't change u')
    return -48.0*r**-13 + 24.0*r**-7
mask = r_cen > 0.8
integ = (r_cen**3*du(r_cen)*g)[:, mask]
P0 = rho_cen*T - (2.0*np.pi/3.0)*rho_cen**2*np.trapezoid(integ, r_cen[mask], axis=1)
good = n_central > 1000

# mu0 by Gibbs-Duhem in rho-space
rr = rho_cen[good]; pp = P0[good]; o = np.argsort(rr); rr, pp = rr[o], pp[o]
m = (rr > 0.10) & (rr < 0.60); rr, pp = rr[m], pp[m]
cf = np.polyfit(rr, pp, 4); dP = np.polyder(np.poly1d(cf))
rg = np.linspace(rr.min(), rr.max(), 120)
mu0 = np.concatenate([[0], np.cumsum(0.5*(dP(rg[1:])/rg[1:]+dP(rg[:-1])/rg[:-1])*np.diff(rg))])
sc = np.linspace(0.14, 0.55, 40); mp = np.array([pmu(x) for x in sc])
mua = mu0 + (np.interp(np.median(sc), sc, mp) - np.interp(np.median(sc), rg, mu0))
rms = np.sqrt(np.mean((np.interp(sc, rg, mua)-mp)**2))
print('local-RDF virial:  mu0 RMS vs PeTS = %.4f' % rms)

# also P0(rho) directly vs PeTS
rr2 = rho_cen[good]; pe = np.array([pets.properties(T, x)['p'] for x in rr2])
mm = (rr2 > 0.12) & (rr2 < 0.56)
print('local-RDF virial:  P0 RMS vs PeTS  = %.4f' % np.sqrt(np.mean((P0[good][mm]-pe[mm])**2)))

plt.figure(figsize=(8, 5.5)); rr3 = np.linspace(0.05, 0.62, 200)
plt.plot(rr3, [pmu(x) for x in rr3], 'k-', lw=2.5, label='PeTS EOS')
plt.plot(rr3, [tmu(x) for x in rr3], '--', color='dimgray', lw=2, label='Thol 2015 EOS')
plt.plot(rg, mua, 'o-', ms=3, color='tab:brown', label='local-RDF virial (RMS %.3f)' % rms)
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$'); plt.legend(); plt.grid(alpha=0.3)
plt.title(r'Local-RDF virial EOS ($\Delta U=4$): g(r;$\rho$) -> P0 -> $\mu_0$')
plt.tight_layout(); plt.savefig('cube100_local_rdf.png', dpi=140)
print('wrote cube100_local_rdf.png')
