#!/usr/bin/env python3
"""Exact homogeneous internal energy, entropy, and enthalpy from a TEMPERATURE
ladder of field-coupling isotherms.

Field-coupling gives the homogeneous Helmholtz free-energy density (and mu0, P0) at
each temperature; the internal energy needs its temperature dependence.  Given
isotherms {T: (rho, mu0, P0)} from fc_isotherm.py at several T, this forms
    f0(rho,T) = rho mu0 - P0
and returns, exactly (no gradient expansion, no k = k' approximation -- cf. the
dissertation Eq. 4.15 / Appendix B):
    phi0 = f0 - T (d f0/dT)_rho      internal energy density (Gibbs-Helmholtz)
    s0   = -(d f0/dT)_rho            entropy density
    eta0 = phi0 + P0                 enthalpy density
This is the recommended route; it reuses the multi-temperature runs already needed
for the phase diagram and avoids the influence-parameter mismatch k != k'.

Self-test: python3 energy_from_isotherms.py --selftest  (van der Waals + monatomic
ideal gas, whose phi0 = (3/2) rho T - a rho^2 is known analytically).
"""
import argparse
import numpy as np


def thermo_energy(isos, rho_grid=None, ngrid=200):
    """isos: dict {T: (rho, mu0, P0)}.  Returns dict T -> dict(rho, f0, phi0, s0,
    eta0, P0, mu0), the internal-energy/entropy/enthalpy densities from the
    temperature derivative of f0 = rho mu0 - P0."""
    Ts = np.array(sorted(isos), float)
    if len(Ts) < 3:
        raise ValueError("need >=3 temperatures for a stable dT derivative")
    if rho_grid is None:
        lo = max(isos[T][0].min() for T in Ts)
        hi = min(isos[T][0].max() for T in Ts)
        rho_grid = np.linspace(lo, hi, ngrid)
    F = np.empty((len(Ts), len(rho_grid)))      # f0 on the common rho grid
    P = np.empty_like(F); MU = np.empty_like(F)
    for i, T in enumerate(Ts):
        rho, mu0, P0 = isos[T][0], isos[T][1], isos[T][2]
        f0 = rho * mu0 - P0
        o = np.argsort(rho)
        F[i] = np.interp(rho_grid, rho[o], f0[o])
        P[i] = np.interp(rho_grid, rho[o], P0[o])
        MU[i] = np.interp(rho_grid, rho[o], mu0[o])
    dFdT = np.gradient(F, Ts, axis=0)            # (d f0/dT)_rho
    out = {}
    for i, T in enumerate(Ts):
        phi0 = F[i] - T * dFdT[i]
        s0 = -dFdT[i]
        out[T] = dict(rho=rho_grid, f0=F[i], phi0=phi0, s0=s0,
                      eta0=phi0 + P[i], P0=P[i], mu0=MU[i])
    return out


# ----------------------------------------------------------------------------

def _vdw_ideal(rho, T, a=1.0):
    """Monatomic ideal gas + van der Waals attraction (b=0 for simplicity).
    f0 = rho T [ln rho - (3/2) ln T - 1] - a rho^2  (reduced, k_B=1).
    Known: phi0 = (3/2) rho T - a rho^2 ; s0 = rho[(5/2) - ln rho + (3/2) ln T]."""
    f0 = rho * T * (np.log(rho) - 1.5 * np.log(T) - 1.0) - a * rho ** 2
    mu0 = T * (np.log(rho) - 1.5 * np.log(T)) - 2.0 * a * rho       # df0/drho
    P0 = rho * mu0 - f0                                              # = rho T - a rho^2
    return mu0, P0, f0


def _selftest():
    a = 1.5
    Ts = [1.20, 1.30, 1.40, 1.50, 1.60]
    rho = np.linspace(0.05, 0.70, 300)
    isos = {T: (rho,) + _vdw_ideal(rho, T, a)[:2] for T in Ts}
    out = thermo_energy(isos, rho_grid=np.linspace(0.08, 0.68, 250))
    Tmid = Ts[len(Ts) // 2]
    r = out[Tmid]['rho']
    phi_true = 1.5 * r * Tmid - a * r ** 2
    s_true = r * (2.5 - np.log(r) + 1.5 * np.log(Tmid))
    ephi = np.max(np.abs(out[Tmid]['phi0'] - phi_true))
    es = np.max(np.abs(out[Tmid]['s0'] - s_true))
    # f0 consistency: phi0 - T s0 should equal f0
    ef = np.max(np.abs(out[Tmid]['phi0'] - Tmid * out[Tmid]['s0'] - out[Tmid]['f0']))
    print("phi0 (internal energy): max err %.2e  [true (3/2)rho T - a rho^2]" % ephi)
    print("s0   (entropy):         max err %.2e" % es)
    print("f0 = phi0 - T s0 consistency: %.2e" % ef)
    ok = ephi < 2e-3 and es < 3e-3 and ef < 1e-12
    print("  -> internal energy, entropy, enthalpy recovered from the T-ladder"
          if ok else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iso', nargs='+',
                    help='per-temperature tables "T:rho_mu0_P0.txt" (cols rho mu0 P0)')
    ap.add_argument('--at', type=float, help='report at this temperature (default: middle)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    isos = {}
    for spec in args.iso:
        T, fn = spec.split(':')
        d = np.loadtxt(fn)
        isos[float(T)] = (d[:, 0], d[:, 1], d[:, 2])
    out = thermo_energy(isos)
    T = args.at if args.at in out else sorted(out)[len(out) // 2]
    o = out[T]
    print("# T=%.4f   rho      phi0(energy)   s0(entropy)   eta0(enthalpy)   P0" % T)
    for i in range(0, len(o['rho']), 12):
        print("  %6.3f   % .4f      % .4f       % .4f      % .4f"
              % (o['rho'][i], o['phi0'][i], o['s0'][i], o['eta0'][i], o['P0'][i]))


if __name__ == '__main__':
    main()
