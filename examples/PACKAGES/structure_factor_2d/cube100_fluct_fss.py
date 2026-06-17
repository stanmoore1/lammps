#!/usr/bin/env python3
"""Occupancy-fluctuation EOS with finite-size scaling (the compressibility route done
right).  d mu0/d rho = kT/S(0), and the per-z-slab occupancy gives S_app(w)=Var(N)/<N>
for slab width w.  S_app(w) = S(0) - C/w (surface deficit), so extrapolate 1/w -> 0
per density to remove the finite-bin bias that caps the naive method, then integrate.
This is the k=0 compressibility WITHOUT the OZ inversion or the large-kmin extrapolation
that break the DCF route.  Usage: cube100_fluct_fss.py <zdump> <dumax>"""
import sys
import numpy as np
sys.path.insert(0, '/home/user/lammps/ljts_eos')
import pets_eos as pets, thol2015_ljts_eos as thol
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

dumpfile = sys.argv[1] if len(sys.argv) > 1 else 'cube100u4_zdump.lammpstrj'
dumax = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
T = 1.198; L = 6.8582414181223398941; Lz = L


def parse(path):
    lines = open(path).read().split('\n'); fr = []; i = 0; n = len(lines)
    while i < n:
        if lines[i].startswith('ITEM: TIMESTEP'):
            na = int(lines[i+3]); fr.append([float(lines[i+9+j]) for j in range(na)]); i += 9+na
        else:
            i += 1
    return np.array(fr)


Z = parse(dumpfile) % Lz
nf = len(Z)
print('frames=%d' % nf)
pmu = lambda r: T*np.log(r) + pets.properties(T, r)['mu_res']
tmu = lambda r: T*np.log(r) + thol.properties(T, r)['mu_res']

# dmu0/drho(rho) for several slab widths w = Lz/Nbins
rg = np.linspace(0.06, 0.62, 80)
Nbins_list = [10, 12, 14, 16, 20, 24, 30]
widths, dmu_w = [], []
for Nb in Nbins_list:
    v = L*L*(Lz/Nb); edges = np.linspace(0, Lz, Nb+1)
    cnt = np.array([np.histogram(z, bins=edges)[0] for z in Z])
    Nm = cnt.mean(0); Nv = cnt.var(0)
    good = (Nm > 0.5) & (Nv > 0)
    rho = Nm[good]/v; dmu = T*Nm[good]/Nv[good]        # kT<N>/Var(N) = kT/S_app
    o = np.argsort(rho)
    widths.append(Lz/Nb)
    dmu_w.append(np.interp(rg, rho[o], dmu[o], left=np.nan, right=np.nan))
widths = np.array(widths); dmu_w = np.array(dmu_w)     # (nw, nrho)

# per-rho linear extrapolation of dmu0/drho in 1/w -> 0  (Kruger-Vlugt)
inv_w = 1.0/widths
dmu_inf = np.full_like(rg, np.nan)
for j in range(len(rg)):
    y = dmu_w[:, j]; m = ~np.isnan(y)
    if m.sum() >= 3:
        a, b = np.polyfit(inv_w[m], y[m], 1)        # y = a*(1/w) + b ; b = w->inf value
        dmu_inf[j] = b

# integrate dmu0/drho -> mu0(rho), both naive (finest bin) and FSS-extrapolated
def integrate(dmu):
    m = ~np.isnan(dmu); r, d = rg[m], dmu[m]
    return r, np.concatenate([[0], np.cumsum(0.5*(d[1:]+d[:-1])*np.diff(r))])


def score(r, mu):
    sc = np.linspace(0.14, 0.55, 40); mp = np.array([pmu(x) for x in sc])
    mua = mu + (np.interp(np.median(sc), sc, mp) - np.interp(np.median(sc), r, mu))
    return mua, np.sqrt(np.mean((np.interp(sc, r, mua)-mp)**2))


r_naive, mu_naive = integrate(dmu_w[-2])      # Nbins=24 (fine), no extrapolation
r_fss, mu_fss = integrate(dmu_inf)
m_naive, rms_naive = score(r_naive, mu_naive)
m_fss, rms_fss = score(r_fss, mu_fss)
print('naive fluctuation (fine bin): RMS=%.4f' % rms_naive)
print('FSS-extrapolated fluctuation: RMS=%.4f' % rms_fss)

plt.figure(figsize=(8, 5.5))
rr = np.linspace(0.05, 0.62, 200)
plt.plot(rr, [pmu(x) for x in rr], 'k-', lw=2.5, label='PeTS EOS')
plt.plot(rr, [tmu(x) for x in rr], '--', color='dimgray', lw=2, label='Thol 2015 EOS')
plt.plot(r_naive, m_naive, 'o-', ms=3, color='tab:blue', alpha=0.6, label='fluctuation, fine bin (RMS %.3f)' % rms_naive)
plt.plot(r_fss, m_fss, 's-', ms=3, color='tab:red', label='fluctuation + 1/w extrapolation (RMS %.3f)' % rms_fss)
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored)')
plt.title(r'Compressibility route done right: occupancy fluctuation + finite-size scaling ($\Delta U=%g$)' % dumax)
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('cube100_fluct_fss.png', dpi=140)
print('wrote cube100_fluct_fss.png')
