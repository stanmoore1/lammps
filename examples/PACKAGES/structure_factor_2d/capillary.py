#!/usr/bin/env python3
"""Interface-width / capillary-wave analysis for the field-pinning study (paper
thesis point 3): the external field pins the liquid-vapor interface and suppresses
long-wavelength capillary waves, narrowing the interface and reducing finite-size
effects near the critical point.

Routines (all robust direct estimators -- no nonlinear solver required):
  * interface_width(z, rho, Lz)   -- 10-90 width of a two-interface slab profile,
                                     from the plateau densities and the slope at the
                                     mid-density crossings.
  * position_fluctuation(z, blocks, Lz) -- RMS wandering of the interface position
                                     across time blocks (the capillary drift the
                                     field suppresses).
  * gamma_from_width_scaling(L, w, kT)  -- surface tension from the capillary-wave
                                     broadening  w^2 = w0^2 + (kT/2 pi gamma) ln L.

Use on the near-Tc subset (field-pinned vs field-off, several box sizes): plot
width and position fluctuation vs field strength and L; a strong field flattens the
L-dependence (the capillary divergence is cut off).

Self-test: python3 capillary.py --selftest  (synthetic tanh slabs, known width and
jitter, and a synthetic w(L) capillary spectrum).
"""
import argparse
import numpy as np


def _crossings(z, rho, level):
    """z-locations and local slopes where rho(z) crosses `level` (periodic)."""
    out = []
    n = len(z)
    for i in range(n):
        a, b = rho[i], rho[(i + 1) % n]
        if (a - level) * (b - level) < 0:
            t = (level - a) / (b - a)
            dz = z[(i + 1) % n] - z[i]
            if dz <= 0:
                dz += z[-1] - z[0] + (z[1] - z[0])           # wrap
            zc = z[i] + t * dz
            slope = (b - a) / dz
            out.append((zc, slope))
    return out


def interface_width(z, rho, Lz, plateau_frac=0.15):
    """10-90 interface width of a two-interface (liquid slab) profile.  Uses the
    plateau densities (mean of the lowest/highest `plateau_frac` of bins) and the
    slope at the mid-density crossing: for a tanh interface of intrinsic width w the
    central slope is (rho_l-rho_v)/(2w) and the 10-90 width is w*ln(81)."""
    rs = np.sort(rho)
    k = max(1, int(plateau_frac * len(rho)))
    rho_v, rho_l = rs[:k].mean(), rs[-k:].mean()
    level = 0.5 * (rho_v + rho_l)
    cr = _crossings(z, rho, level)
    if len(cr) < 2:
        return None
    slopes = [abs(s) for _, s in cr]
    w_intrinsic = (rho_l - rho_v) / (2.0 * np.mean(slopes))
    return dict(rho_v=rho_v, rho_l=rho_l, w=w_intrinsic,
                w_1090=w_intrinsic * np.log(81.0),
                z_interfaces=[c for c, _ in cr])


def position_fluctuation(z, blocks, Lz, plateau_frac=0.15):
    """RMS fluctuation of the (rising-edge) interface position across time blocks.
    blocks: array (nblocks, nbins) of per-block density profiles."""
    pos = []
    for rho in blocks:
        rs = np.sort(rho); k = max(1, int(plateau_frac * len(rho)))
        level = 0.5 * (rs[:k].mean() + rs[-k:].mean())
        cr = _crossings(z, rho, level)
        rising = [c for c, s in cr if s > 0]
        if rising:
            pos.append(rising[0])
    pos = np.array(pos)
    # unwrap around the periodic box before taking the std
    if len(pos) > 1:
        pos = pos[0] + np.unwrap((pos - pos[0]) * 2 * np.pi / Lz) * Lz / (2 * np.pi)
    return float(np.std(pos)) if len(pos) else float('nan')


def gamma_from_width_scaling(L, w, kT):
    """Capillary-wave surface tension from w^2 = w0^2 + (kT/2 pi gamma) ln L.
    Returns dict(gamma, w0).  L, w arrays over box sizes at fixed T and field."""
    L = np.asarray(L, float); w2 = np.asarray(w, float) ** 2
    slope, intercept = np.polyfit(np.log(L), w2, 1)
    return dict(gamma=kT / (2.0 * np.pi * slope) if slope > 0 else np.inf,
                w0=np.sqrt(max(intercept, 0.0)), slope=slope)


def read_chunk_blocks(fn):
    """Per-block density profiles from a fix ave/chunk file (each timestep block is a
    'Nrows' header then the rows); returns (z, blocks[nblocks, nbins])."""
    blocks, cur, z = [], None, None
    for l in open(fn):
        if l.startswith('#'):
            continue
        p = l.split()
        if len(p) == 3:
            cur = []; blocks.append(cur)
        elif cur is not None:
            cur.append([float(x) for x in p])
    arr = [np.array(b) for b in blocks]
    z = arr[0][:, 1]                                          # coord column
    dens = np.array([b[:, 3] for b in arr])                   # density column
    return z, dens


# ----------------------------------------------------------------------------

def _tanh_slab(z, rho_v, rho_l, z1, z2, w):
    return rho_v + 0.5 * (rho_l - rho_v) * (np.tanh((z - z1) / w) - np.tanh((z - z2) / w))


def _selftest():
    Lz = 30.0; nb = 300
    z = (np.arange(nb) + 0.5) / nb * Lz
    rho_v, rho_l, w_true = 0.05, 0.70, 0.8
    z1, z2 = 7.5, 22.5
    rho = _tanh_slab(z, rho_v, rho_l, z1, z2, w_true)
    iw = interface_width(z, rho, Lz)
    werr = abs(iw['w'] - w_true)
    print("width: w fit %.4f (true %.4f)  10-90 %.4f  rho_v=%.3f rho_l=%.3f"
          % (iw['w'], w_true, iw['w_1090'], iw['rho_v'], iw['rho_l']))

    # position fluctuation: jitter z1,z2 by a known sigma across blocks
    rng = np.random.default_rng(0); sig = 0.6
    blocks = []
    for _ in range(400):
        d = rng.normal(0, sig)
        blocks.append(_tanh_slab(z, rho_v, rho_l, z1 + d, z2 + d, w_true))
    pf = position_fluctuation(z, np.array(blocks), Lz)
    print("position fluctuation: fit %.3f (true sigma %.3f)" % (pf, sig))

    # capillary spectrum: synthesize w(L) = sqrt(w0^2 + (kT/2 pi gamma) ln L)
    kT, gamma_true, w0 = 1.0, 0.5, 0.8
    Ls = np.array([10., 20., 40., 80., 160.])
    ws = np.sqrt(w0 ** 2 + (kT / (2 * np.pi * gamma_true)) * np.log(Ls))
    g = gamma_from_width_scaling(Ls, ws, kT)
    print("gamma from width scaling: fit %.4f (true %.4f)  w0 %.3f"
          % (g['gamma'], gamma_true, g['w0']))

    ok = (werr < 0.05 and abs(pf - sig) < 0.1
          and abs(g['gamma'] - gamma_true) < 1e-3)
    print("  -> interface width, position fluctuation, capillary gamma recovered"
          if ok else "  -> MISMATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--density', help='fix ave/chunk density file (multi-block)')
    ap.add_argument('--lz', type=float)
    ap.add_argument('--kt', type=float, default=1.0)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    z, blocks = read_chunk_blocks(args.density)
    rho = blocks.mean(axis=0)
    iw = interface_width(z, rho, args.lz)
    pf = position_fluctuation(z, blocks, args.lz)
    print("interface width (10-90) = %.4f ; intrinsic w = %.4f" % (iw['w_1090'], iw['w']))
    print("interface position fluctuation (RMS) = %.4f" % pf)


if __name__ == '__main__':
    main()
