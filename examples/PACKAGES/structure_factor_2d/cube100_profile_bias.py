#!/usr/bin/env python3
"""Check the PROFILE fits (cosine-series / savgol smoothing of rho(z), PN(z), PT(z))
for bias.  A good smoother should: (a) conserve the mean (integral), (b) leave a
zero-mean, structure-free residual (fit-raw), and (c) not systematically flatten the
extrema.  We test all three vs the number of cosine modes and savgol, and propagate
the result to mu0 to see the smoothing-induced EOS bias.
Usage: cube100_profile_bias.py <tag> <T>"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/user/lammps/ljts_eos')
import oz_invert as oz, contour_pressure as cp, pets_eos as pets
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else 'cube100Tc2'
T = float(sys.argv[2]) if len(sys.argv) > 2 else 1.089
L = 6.8582414181223398941; Lz = L; area = L * L
pmu = lambda r: T * np.log(r) + pets.properties(T, r)['mu_res']

de = cp.read_chunk('%s_dens.out' % tag); hs = cp.read_chunk('%s_hstress.out' % tag)
nb = len(de); z = (np.arange(nb) + 0.5) / nb * Lz; Vbin = area * (Lz / nb)
rho_raw = de[:, 3]
PN_raw = -hs[:, 5] / Vbin + rho_raw * T
PT_raw = -0.5 * (hs[:, 3] + hs[:, 4]) / Vbin + rho_raw * T
P0_raw = 1.5 * PT_raw - 0.5 * PN_raw


def runs_test(resid):
    """Wald-Wolfowitz: z-score of the # of sign runs. |z|>2 => residual is structured
    (autocorrelated) = the smoother left/created systematic (biased) structure."""
    s = np.sign(resid - np.median(resid)); s = s[s != 0]
    n = len(s); runs = 1 + np.sum(s[1:] != s[:-1])
    n1 = np.sum(s > 0); n2 = np.sum(s < 0)
    if n1 == 0 or n2 == 0:
        return 0.0
    mu = 1 + 2 * n1 * n2 / n; var = (mu - 1) * (mu - 2) / (n - 1)
    return (runs - mu) / np.sqrt(var) if var > 0 else 0.0


print('%s (T=%g): profile-fit bias (cosine modes & savgol)\n' % (tag, T))
print('%-12s %-10s | mean-shift   resid-rms  runs-z   peak-flatten' % ('profile', 'smoother'))
for name, raw in [('rho', rho_raw), ('PN', PN_raw), ('PT', PT_raw), ('P0', P0_raw)]:
    pk = raw.max() - raw.min()
    for lab, fit in ([('cos m=%d' % m, oz.fourier_cosine_smooth(raw, m)) for m in (4, 6, 8, 12)] +
                     [('savgol w=%d' % w, savgol_filter(raw, w, 3, mode='wrap')) for w in (9, 15)]):
        resid = fit - raw
        meanshift = fit.mean() - raw.mean()                  # integral/mean conservation
        peakflat = (fit.max() - fit.min()) - pk              # negative => extrema flattened
        print('%-12s %-10s | %+9.2e  %8.4f  %+6.2f   %+8.4f'
              % (name, lab, meanshift, np.sqrt(np.mean(resid**2)), runs_test(resid), peakflat))
    print()

# propagate to mu0: smoothing-induced EOS bias (vs the rho-binning-corrected reference)
sc = np.linspace(0.16, 0.50, 40); mp = np.array([pmu(x) for x in sc])


def mu0(rho, P0):
    m = z <= Lz / 2; r, p = rho[m], P0[m]; o = np.argsort(r); r, p = r[o], p[o]
    k = (r > 0.10) & (r < 0.56); r, p = r[k], p[k]
    cf = np.polyfit(r, p, 4); dP = np.polyder(np.poly1d(cf))
    rg = np.linspace(max(r.min(), 0.13), min(r.max(), 0.54), 120)
    mu = np.concatenate([[0], np.cumsum(0.5 * (dP(rg[1:]) / rg[1:] + dP(rg[:-1]) / rg[:-1]) * np.diff(rg))])
    mu += np.mean(mp - np.interp(sc, rg, mu)); return rg, mu


fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].plot(z, rho_raw, 'k.', ms=4, label='raw bins')
ax[1].plot(z, P0_raw, 'k.', ms=4, label='raw bins')
muP = np.array([pmu(x) for x in sc])
ax[2].plot(sc, np.zeros_like(sc), 'k-', lw=1, label='PeTS (ref)')
for m, c in [(4, 'tab:blue'), (6, 'tab:orange'), (8, 'tab:green'), (12, 'tab:red')]:
    rf = oz.fourier_cosine_smooth(rho_raw, m); pnf = oz.fourier_cosine_smooth(PN_raw, m)
    ptf = oz.fourier_cosine_smooth(PT_raw, m); p0f = 1.5 * ptf - 0.5 * pnf
    ax[0].plot(z, rf, '-', color=c, lw=1.4, label='cos m=%d' % m)
    ax[1].plot(z, p0f, '-', color=c, lw=1.4, label='cos m=%d' % m)
    rg, mu = mu0(rf, p0f)
    ax[2].plot(sc, np.interp(sc, rg, mu) - muP, '-', color=c, lw=1.6, label='cos m=%d' % m)
ax[0].set_title('density profile fit'); ax[0].set_xlabel('z'); ax[0].set_ylabel(r'$\rho$'); ax[0].legend(fontsize=7)
ax[1].set_title(r'$P_0(z)$ fit'); ax[1].set_xlabel('z'); ax[1].set_ylabel(r'$P_0$'); ax[1].legend(fontsize=7)
ax[2].set_title(r'smoothing bias in $\mu_0$ (fit $-$ PeTS)'); ax[2].set_xlabel(r'$\rho^*$')
ax[2].set_ylabel(r'$\mu_0$ residual'); ax[2].axhline(0, color='gray', lw=0.6); ax[2].legend(fontsize=7)
plt.suptitle('Profile-fit bias: %s (T=%g)' % (tag, T)); plt.tight_layout()
plt.savefig('%s_profile_bias.png' % tag, dpi=140)
print('wrote %s_profile_bias.png' % tag)
