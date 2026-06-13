#!/usr/bin/env python3
"""Build one homogeneous CPP isotherm (mu0(rho), P0(rho), the van der Waals loop)
from a (rho_avg x field-amplitude) grid of density profiles, via the field-coupling
local-EOS route, and hand it to phase_diagram.py.

Manifest (CSV, '#' comments) -- one line per CPP run at this temperature:
    dumax,rho_avg,density_file
All runs are POOLED into field_coupling.local_eos: the rho_avg axis gives density
coverage and identifies the density-dependent influence parameters kappa_g(rho), the
field axis separates mu0 from the gradient correction (see field_coupling._selftest).
A uniform A=0 reference is prepended per distinct rho_avg.

The surface tension gamma is computed per fixed-rho_avg dense sub-ladder (the coupling
integral needs the dA' charging path; field_coupling.interfacial).

Outputs a "rho  mu0  P0" table (consumable by phase_diagram.py --iso) and the fitted
kappa_g(rho) / gamma.  Optional --lrc-rcut applies the local LJ tail correction to
mu0/P0 (for full-LJ runs done with a truncated force; same formula as plot_muih.py).

Self-test: python3 fc_isotherm.py --selftest  (synthetic 2D grid, recovers the EOS).
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import field_coupling as fc
import oz_invert as oz


def load_manifest(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        du, ra, fn = ln.split(',')
        rows.append((float(du), float(ra), fn.strip()))
    return rows


def lj_tail(rho, rcut):
    """Local standard LJ long-range correction (truncated-force runs, full-LJ EOS
    comparison): a = (16/3) pi [(2/3) rc^-9 - rc^-3];  P0 += a rho^2,  mu0 += 2 a rho
    (dmu=dP/rho).  Same convention as plot_muih.py / dissertation Eq. 7.9-7.10."""
    a = (16.0 / 3.0) * np.pi * ((2.0 / 3.0) * rcut ** -9 - rcut ** -3)
    return a * rho ** 2, 2.0 * a * rho                       # dP0, dmu0


def fit_isotherm(amps, profiles, temp, Lz, grad_spec, smooth=10, lrc_rcut=0.0):
    """Pool the whole grid into local_eos; optionally add the LJ tail correction."""
    eos = fc.local_eos(np.asarray(amps), np.asarray(profiles), temp, Lz,
                       deg=6, smooth=smooth, grad_spec=grad_spec)
    if lrc_rcut > 0.0:
        dP, dmu = lj_tail(eos['rho'], lrc_rcut)
        eos['P0'] = eos['P0'] + dP
        eos['mu0'] = eos['mu0'] + dmu
    return eos


def gamma_by_rho_avg(rows, profiles_by_row, temp, Lz, eos, smooth=10):
    """gamma per fixed-rho_avg sub-ladder via the exact coupling integral."""
    out = {}
    groups = {}
    for (du, ra, _), prof in zip(rows, profiles_by_row):
        groups.setdefault(round(ra, 6), []).append((du / 2.0, prof))
    for ra, items in groups.items():
        items = sorted(items)
        amps = np.array([0.0] + [a for a, _ in items])
        profs = np.array([np.full_like(items[0][1], ra)] + [p for _, p in items])
        if len(amps) >= 3:
            itf = fc.interfacial(amps, profs, eos, temp, Lz, smooth=smooth)
            out[ra] = float(itf['gamma'][-1])
    return out


def build_grid(rows, profiles_by_row):
    """Assemble the pooled (amps, profiles) with one uniform A=0 rung per rho_avg."""
    amps, profiles = [], []
    seen = set()
    for (du, ra, _), prof in zip(rows, profiles_by_row):
        key = round(ra, 6)
        if key not in seen:                                  # uniform reference rung
            amps.append(0.0); profiles.append(np.full_like(prof, ra)); seen.add(key)
        amps.append(du / 2.0); profiles.append(prof)
    return amps, profiles


def run(rows, profiles_by_row, temp, Lz, grad_spec, smooth=10, lrc_rcut=0.0):
    amps, profiles = build_grid(rows, profiles_by_row)
    eos = fit_isotherm(amps, profiles, temp, Lz, grad_spec, smooth, lrc_rcut)
    gam = gamma_by_rho_avg(rows, profiles_by_row, temp, Lz, eos, smooth)
    return eos, gam


# ----------------------------------------------------------------------------

def _selftest():
    """Synthetic 2D (rho_avg x field) grid from a known supercritical EOS + density-
    dependent kappa2(rho); confirm the pooled fit recovers mu0(rho) and that the LJ
    tail correction shifts mu0/P0 by the analytic amount."""
    Lz = 30.0; T = 1.0; nb = 200
    a = {2: 2.5, 3: -1.0, 4: 0.6}
    mu_ex = lambda r: sum(a[m] * m * r ** (m - 1) for m in a)
    mu0 = lambda r: T * np.log(r) + mu_ex(r)
    grad_spec = {2: 1, 4: 0}
    acoef = {(2, 0): 1.5, (2, 1): 0.9, (4, 0): 0.8}
    rows, profs = [], []
    for ra in (0.40, 0.45, 0.50):
        for du in (0.0, 1.6, 3.2, 4.8):
            rows.append((du, ra, 'synthetic'))
            profs.append(fc._solve_el(mu0, acoef, grad_spec, du / 2.0, T, ra, Lz, nb))
    eos, gam = run(rows, profs, T, Lz, grad_spec, smooth=12)
    rg = eos['rho']; mt = mu_ex(rg)
    de = (eos['mu_ex'] - eos['mu_ex'].mean()) - (mt - mt.mean())
    k2err = np.max(np.abs(eos['kappa2'](rg) - (1.5 + 0.9 * rg)))
    # LRC check
    dP, dmu = lj_tail(rg, 2.5)
    eos2 = fit_isotherm(*build_grid(rows, profs), T, Lz, grad_spec, 12, lrc_rcut=2.5)
    lrc_ok = np.allclose(eos2['mu0'] - eos['mu0'], dmu, atol=1e-9)
    print("pooled isotherm:  max|mu_ex-ref| %.2e ; kappa2(rho) err %.2e ; "
          "gamma keys %s" % (np.max(np.abs(de)), k2err, sorted(gam)))
    print("LRC mu0/P0 shift matches analytic: %s" % lrc_ok)
    ok = np.max(np.abs(de)) < 3e-2 and k2err < 0.05 and lrc_ok
    print("  -> isotherm pooling + kappa_g(rho) + LRC OK" if ok else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', help='CSV: dumax,rho_avg,density_file per line')
    ap.add_argument('--lz', type=float)
    ap.add_argument('--temp', type=float)
    ap.add_argument('--grad-spec', default='2:0,4:0')
    ap.add_argument('--smooth', type=int, default=10)
    ap.add_argument('--lrc-rcut', type=float, default=0.0)
    ap.add_argument('--out', default='isotherm.txt', help='output rho mu0 P0 table')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return

    rows = load_manifest(args.manifest)
    profs = [oz.fourier_cosine_smooth(fc._read_density(fn), args.smooth)
             for _, _, fn in rows]
    eos, gam = run(rows, profs, args.temp, args.lz, fc.parse_grad_spec(args.grad_spec),
                   args.smooth, args.lrc_rcut)
    np.savetxt(args.out, np.c_[eos['rho'], eos['mu0'], eos['P0']],
               header='rho  mu0  P0', comments='# ')
    print('# wrote', args.out, '(feed to: phase_diagram.py --iso %.4f:%s)'
          % (args.temp, args.out))
    print('# kappa2(rho_avg)=%.4f kappa4=%.4f' % (
        eos['kappa2'](eos['rho_avg']), eos['kappa4'](eos['rho_avg'])))
    print('# gamma per rho_avg:', {k: round(v, 4) for k, v in gam.items()})


if __name__ == '__main__':
    main()
