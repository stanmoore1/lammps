#!/usr/bin/env python3
"""Occupancy-fluctuation EOS recovery (outside-the-box): from the per-z-bin number
fluctuations of the inhomogeneous CPP run, the local compressibility gives

    d mu0/d rho |_{rho_i} = kT <N_i> / Var(N_i)            (rho_i = <N_i>/v_bin)

Integrate over rho (the field spreads the bins across the density range) to get
mu0(rho).  No stress, no gradient expansion, no matrix inversion -- pure
thermodynamic fluctuations.  Compared to PeTS.  (A finite-bin correction is probed
by scanning the bin width.)"""
import sys
import numpy as np
sys.path.insert(0, '/home/user/lammps/ljts_eos')
import pets_eos as pets
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T = 1.198; L = 6.8582414181223398941; Lz = L


def parse_dump(path):
    lines = open(path).read().split('\n')
    frames = []; i = 0; n = len(lines)
    while i < n:
        if lines[i].startswith('ITEM: TIMESTEP'):
            natoms = int(lines[i+3])
            z = np.array([float(lines[i+9+j]) for j in range(natoms)])
            frames.append(z); i += 9 + natoms
        else:
            i += 1
    return np.array(frames)


Z = parse_dump('cube100_zdump.lammpstrj') % Lz
print('frames=%d, atoms=%d' % Z.shape)


def pmu(r): return T*np.log(r) + pets.properties(T, r)['mu_res']


plt.figure(figsize=(8.5, 6))
rr = np.linspace(0.05, 0.6, 200)
plt.plot(rr, np.array([pmu(x) for x in rr]), 'k-', lw=2.5, label='PeTS EOS')
sc = np.linspace(0.15, 0.55, 40); mp = np.array([pmu(x) for x in sc])
for Nbins, col in [(12, 'tab:blue'), (16, 'tab:green'), (24, 'tab:red')]:
    v = L*L*(Lz/Nbins)
    edges = np.linspace(0, Lz, Nbins+1)
    counts = np.array([np.histogram(z, bins=edges)[0] for z in Z])   # (nframes, Nbins)
    Nmean = counts.mean(0); Nvar = counts.var(0)
    rho = Nmean/v
    good = (Nmean > 1.0) & (Nvar > 0)
    dmu = T*Nmean[good]/Nvar[good]                                   # kT <N>/Var(N)
    rg = rho[good]; o = np.argsort(rg); rg, dmu = rg[o], dmu[o]
    # collapse duplicate densities, integrate dmu0/drho over rho
    mu0 = np.concatenate([[0], np.cumsum(0.5*(dmu[1:]+dmu[:-1])*np.diff(rg))])
    mua = mu0 + (np.interp(np.median(sc), sc, mp) - np.interp(np.median(sc), rg, mu0))
    rms = np.sqrt(np.mean((np.interp(sc, rg, mua) - mp)**2))
    print('Nbins=%2d (dz=%.2f, <N> up to %.1f):  mu0 RMS vs PeTS = %.4f'
          % (Nbins, Lz/Nbins, Nmean.max(), rms))
    plt.plot(rg, mua, 'o-', ms=3, color=col, label='fluctuation, %d bins (RMS %.3f)' % (Nbins, rms))
plt.xlabel(r'$\rho^*$'); plt.ylabel(r'$\mu_0^*$ (anchored)')
plt.title(r'Occupancy-fluctuation EOS recovery, N=100 cubic $T^*=1.198$')
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('cube100_fluct.png', dpi=140)
print('wrote cube100_fluct.png')
