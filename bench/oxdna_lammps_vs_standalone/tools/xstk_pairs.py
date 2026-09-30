#!/usr/bin/env python3
"""Q2: per-pair oxDNA3 cross-stacking energies with the LAMMPS rule (both channels, no gate)
and the standalone rule (DNA3Interaction::_cross_stacking: both channels, but only if
cos7 > 0 and cos8 > 0 with r inside the 3'3' radial range, or cos7 < 0 and cos8 < 0 with r
inside the 5'5' radial range; 0 otherwise). Parameters: LAMMPS potential file with the
standalone smoothing points (oxdna3_lj_ts.cgdna), flanks as LAMMPS (terminal flank = average).
usage: xstk_pairs.py <top> <conf> <oxdna3_lj_ts.cgdna> [min |dE| to list, default 1e-4]"""
import sys
import numpy as np
top, conf, pot = sys.argv[1:4]; thr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
cl = open(conf).read().split('\n'); box = float(cl[1].split()[2])
d = np.array([list(map(float, l.split()[:9])) for l in cl[3:] if l.strip()])
tl = [l.split() for l in open(top).read().split('\n')[1:] if l.strip()]
N = len(d); tmap = {'A': 1, 'C': 2, 'G': 3, 'T': 4}; pb = {1: .43, 3: .43, 2: .37, 4: .37}
strand = [int(t[0]) for t in tl]; typ = [tmap[t[1]] for t in tl]; n3 = [int(t[2]) for t in tl]; n5 = [int(t[3]) for t in tl]
for l in open(pot):
    p = l.split()
    if len(p) > 2 and p[2] == 'xstk': v = np.array(list(map(float, p[3:])))
K = v[0]
r0_33, rc_33, rlo_33, rhi_33, r0_55, rc_55, rlo_55, rhi_55 = [v[1 + 256 * k:1 + 256 * (k + 1)].reshape(4, 4, 4, 4) for k in range(8)]
o = 1 + 8 * 256; th123 = v[o:o + 9].reshape(3, 3); o += 9
a4_33, t0_33, ts4_33, a4_55, t0_55, ts4_55 = [v[o + 256 * k:o + 256 * (k + 1)].reshape(4, 4, 4, 4) for k in range(6)]; o += 1536
t7 = v[o:o + 4]; t8 = v[o + 4:o + 8]


def avg(A, i, j, k, l):
    s = A[:, j-1, k-1, :] if (i == 0 and l == 0) else (A[:, j-1, k-1, l-1] if i == 0 else (A[i-1, j-1, k-1, :] if l == 0 else A[i-1, j-1, k-1, l-1]))
    return float(np.mean(s))


def f2range(r0, rc, rlo, rhi):
    t1, t2, t3 = rlo - r0, rhi - r0, rc - r0
    return rlo - t1 + t3 * t3 / t1, rhi - t2 + t3 * t3 / t2


def f2(r, r0, rc, rlo, rhi):
    t1, t2, t3 = rlo - r0, rhi - r0, rc - r0
    rclo, rchi = f2range(r0, rc, rlo, rhi)
    blo = -0.5 * t1 / (rclo - rlo); bhi = -0.5 * t2 / (rchi - rhi)
    if r < rclo or r > rchi: return 0.
    if r < rlo: return K * blo * (r - rclo) ** 2
    if r > rhi: return K * bhi * (r - rchi) ** 2
    return K / 2 * ((r - r0) ** 2 - (rc - r0) ** 2)


def f4(t, a, t0, ts):
    tc = 1 / (a * ts); b = a * ts / (tc - ts); dt = abs(t - t0)
    return 1 - a * dt * dt if dt < ts else (b * (tc - dt) ** 2 if dt < tc else 0.)


ac = lambda x: np.arccos(np.clip(x, -1, 1))
E_l = E_s = 0.; rows = []
for i in range(N):
    for j in range(i + 1, N):
        if n3[i] == j or n5[i] == j: continue
        a, b = i, j
        ra = d[a, :3] + pb[typ[a]] * d[a, 3:6]; rb = d[b, :3] + pb[typ[b]] * d[b, 3:6]
        r = ra - rb; r -= box * np.round(r / box); rm = np.linalg.norm(r); u = r / rm
        if rm > 1.0: continue
        a3 = typ[n3[a]] if n3[a] >= 0 else 0; b3 = typ[n3[b]] if n3[b] >= 0 else 0
        a5 = typ[n5[a]] if n5[a] >= 0 else 0; b5 = typ[n5[b]] if n5[b] >= 0 else 0
        ta, tb = typ[a], typ[b]
        P33 = [avg(A, a3, ta, tb, b3) for A in (r0_33, rc_33, rlo_33, rhi_33)]
        P55 = [avg(A, a5, ta, tb, b5) for A in (r0_55, rc_55, rlo_55, rhi_55)]
        g33, g55 = f2(rm, *P33), f2(rm, *P55)
        if g33 == 0 and g55 == 0: continue
        ax, az, bx, bz = d[a, 3:6], d[a, 6:9], d[b, 3:6], d[b, 6:9]
        th1, th2, th3, th4 = ac(-ax @ bx), ac(-ax @ u), ac(bx @ u), ac(az @ bz)
        c7, c8 = -az @ u, bz @ u
        f123 = np.prod([f4(t, *th123[k]) for k, t in enumerate((th1, th2, th3))])
        e33 = g33 * f4(th4, avg(a4_33, a3, ta, tb, b3), avg(t0_33, a3, ta, tb, b3), avg(ts4_33, a3, ta, tb, b3)) * f4(ac(c7), t7[0], t7[1], t7[3]) * f4(ac(c8), t8[0], t8[1], t8[3])
        e55 = g55 * f4(th4, avg(a4_55, a5, ta, tb, b5), avg(t0_55, a5, ta, tb, b5), avg(ts4_55, a5, ta, tb, b5)) * f4(ac(c7), t7[0], t7[2], t7[3]) * f4(ac(c8), t8[0], t8[2], t8[3])
        e = f123 * (e33 + e55)
        lo33, hi33 = f2range(*P33); lo55, hi55 = f2range(*P55)
        gate = (c7 > 0 and c8 > 0 and lo33 < rm < hi33) or (c7 < 0 and c8 < 0 and lo55 < rm < hi55)
        E_l += e; E_s += e if gate else 0.
        if not gate and abs(e) > thr:
            why = 'mixed signs of cos7, cos8' if c7 * c8 < 0 else ('cos7, cos8 > 0 but r outside the 3\'3\' range' if c7 > 0 else 'cos7, cos8 < 0 but r outside the 5\'5\' range')
            rows.append((e, a, b, strand[a] == strand[b], np.degrees(ac(c7)), np.degrees(ac(c8)), rm, f123 * e33, f123 * e55, why))
print(f'E_xstk LAMMPS rule {E_l:.6f}   standalone rule {E_s:.6f}   difference {E_l - E_s:+.6f}')
for e, a, b, same, t7d, t8d, rm, e33, e55, why in sorted(rows):
    print(f'  pair ({a:3d},{b:3d}) {"same strand" if same else "cross strand"}: E = {e:+.5f} (3\'3\' {e33:+.5f}, 5\'5\' {e55:+.5f}), '
          f'theta7 = {t7d:5.1f} deg, theta8 = {t8d:5.1f} deg, r = {rm:.3f}; standalone 0: {why}')
