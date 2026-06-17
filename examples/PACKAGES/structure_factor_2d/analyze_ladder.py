#!/usr/bin/env python3
"""Per-temperature comparison of mu0(rho) for the LJTS CPP field ladder:
  PeTS EOS, pressure tensor (IK and H, P0=3/2PT-1/2PN -> mu0), and field coupling
  (nonlocal kernel and gradient expansion).  At the lowest temperature, Maxwell
  binodal of each loop vs the PeTS vle.  Reuses field_coupling, kernel_fit,
  contour_pressure, phase_diagram, and the PeTS EOS (ljts_eos/pets_eos.py).

Usage: python3 analyze_ladder.py --temps 0.980 1.089 1.198 --rho 0.31
"""
import argparse, sys, glob
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
sys.path.insert(0, '/home/user/lammps/ljts_eos')
import field_coupling as fc, kernel_fit as kf, contour_pressure as cp, phase_diagram as pd
import oz_invert as oz
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
try:
    import pets_eos as pets
except Exception:
    pets = None


def box_dims(rho, N=1000, aspect=3.0):
    V = N / rho; Lx = (V / aspect) ** (1.0 / 3.0); return Lx, aspect * Lx


def load_manifest(T):
    rows = [l.split(',') for l in open('ladder_T%s.csv' % T) if l.strip() and not l.startswith('#')]
    dus = [float(r[0]) for r in rows]
    files = [r[2].strip() for r in rows]
    profs = [fc._read_density(f) for f in files]
    amps = np.array([0.0] + [d / 2.0 for d in dus])
    profiles = np.array([np.full_like(profs[0], np.mean(profs[0]))] + profs)
    return amps, profiles, dus, files


def anchor(rho, mu, rho0, mu0_ref):
    return mu + (mu0_ref - np.interp(rho0, rho, mu))


def pets_mu0(T, rho):
    return T * np.log(rho) + np.array([pets.properties(T, r)['mu_res'] for r in rho])


def analyze_T(T, rho_avg, smooth=10):
    Lx, Lz = box_dims(rho_avg)
    amps, profiles, dus, files = load_manifest(T)
    # field coupling: gradient expansion and nonlocal kernel
    eg = fc.local_eos(amps, profiles, T, Lz, deg=6, smooth=smooth, grad_spec={2: 0, 4: 0})
    ek = kf.kernel_eos(amps, profiles, T, Lz, deg=6, smax=2.5, nmodes=3, ridge=1e-3, smooth=smooth)
    # pressure tensor (IK, H) from the strongest-field run (widest density range)
    tag = files[-1].replace('_dens.out', '')
    rik, PNik, PTik = cp.ik_profile(tag + '_ikstress.out', Lz, smooth)
    rh, PNh, PTh = cp.h_profile(tag + '_hstress.out', tag + '_dens.out', Lz, Lx * Lx, T, smooth)
    rik2, muik = cp.mu0_from_p0(rik, cp.p0_ik(PNik, PTik), Lz, smooth)
    rh2, muh = cp.mu0_from_p0(rh, cp.p0_ik(PNh, PTh), Lz, smooth)
    out = dict(T=T, Lz=Lz, eg=eg, ek=ek,
               ik=(rik2, muik), h=(rh2, muh), rho_avg=rho_avg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--temps', nargs='+', default=['0.980', '1.089', '1.198'])
    ap.add_argument('--rho', type=float, default=0.31)
    ap.add_argument('--smooth', type=int, default=10)
    args = ap.parse_args()
    res = [analyze_T(T, args.rho, args.smooth) for T in args.temps]

    fig, axes = plt.subplots(1, len(res), figsize=(6 * len(res), 5), squeeze=False)
    for ax, r in zip(axes[0], res):
        T = float(r['T']); ra = r['rho_avg']
        rr = np.linspace(0.05, 0.65, 200)
        muref = pets_mu0(T, rr) if pets else None
        m0 = np.interp(ra, rr, muref) if pets else 0.0
        if pets:
            ax.plot(rr, muref, 'k-', lw=2.5, label='PeTS EOS')
        for key, lab, c, ls in [('eg', 'FC gradient', 'tab:orange', '--'),
                                ('ek', 'FC kernel', 'tab:red', '-')]:
            e = r[key]; mu = anchor(e['rho'], e['mu0'], ra, m0)
            ax.plot(e['rho'], mu, ls, color=c, lw=1.8, label=lab)
        for key, lab, c in [('ik', 'PT (IK)', 'tab:blue'), ('h', 'PT (H)', 'tab:green')]:
            rho2, mu2 = r[key]; mu2 = anchor(rho2, mu2, ra, m0)
            ax.plot(rho2, mu2, ':', color=c, lw=1.8, label=lab)
        ax.axvline(ra, color='gray', ls=':', lw=0.7)
        ax.set_xlabel(r'$\rho^*$'); ax.set_ylabel(r'$\mu_0^*$')
        ax.set_title('T*=%.3f' % T); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('ladder_mu0.png', dpi=130)
    print('wrote ladder_mu0.png')

    # Maxwell binodal at the lowest temperature
    rlow = res[int(np.argmin([float(r['T']) for r in res]))]
    Tlow = float(rlow['T'])
    print('\n=== Binodal at lowest T*=%.3f ===' % Tlow)
    for key, lab in [('ek', 'FC kernel'), ('eg', 'FC gradient')]:
        e = rlow[key]
        bi = pd.binodal(e['rho'], e['P0'], e['mu0'])
        if bi:
            print('  %-12s rho_v=%.3f rho_l=%.3f' % (lab, bi['rho_v'], bi['rho_l']))
        else:
            print('  %-12s no loop found' % lab)
    if pets:
        rv, rl, ps = pets.vle(Tlow)
        print('  %-12s rho_v=%.3f rho_l=%.3f (P_sat=%.4f)' % ('PeTS vle', rv, rl, ps))


if __name__ == '__main__':
    main()
