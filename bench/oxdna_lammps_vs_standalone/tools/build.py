#!/usr/bin/env python3
"""Build ideal B-DNA starting structures (old-style oxDNA topology + configuration)
with the geometry of oxDNA's utils/generate-sa.py (rise 0.3897628551303122, twist
35.9 deg, COM 0.6 from the helix axis along -a1).

usage: build.py <structure> <outdir> [box]
  duplex12  12-bp duplex
  nick16    16-bp duplex, one strand nicked between bp 8 and 9
  blunt16   16-bp helix cut on both strands between bp 8 and 9 (two 8-bp duplexes
            stacked end to end at a blunt interface)
  bulge12   12-bp duplex with one extra unpaired T in strand 1 between bp 6 and 7
            (placed outside the helix)
  ss20      20-nt single strand in helical geometry
"""
import sys, os
import numpy as np

BASE_BASE = 0.3897628551303122
CM = 0.6                              # POS_BASE + 0.2
TW = np.deg2rad(35.9)


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    x, y, z = axis; c, s = np.cos(ang), np.sin(ang); o = 1 - c
    return np.array([[o*x*x+c, o*x*y-s*z, o*x*z+s*y], [o*x*y+s*z, o*y*y+c, o*y*z-s*x], [o*x*z-s*y, o*y*z+s*x, o*z*z+c]])


def helix(nbp):
    """strand 1 (generated 3' -> 5' along +z, as generate-sa.py) and its complement, as (pos, a1, a3) lists"""
    a1 = np.array([1.0, 0, 0]); a3 = np.array([0, 0, 1.0]); R = rot(a3, TW)
    rb = np.array([0, 0, 0.0]); s1 = []
    for i in range(nbp):
        s1.append((rb - CM * a1, a1.copy(), a3.copy()))
        if i != nbp - 1: a1 = R @ a1; rb = rb + a3 * BASE_BASE
    a1 = -a1; a3 = -a3; s2 = []
    for i in range(nbp):
        s2.append((rb - CM * a1, a1.copy(), a3.copy()))
        a1 = R.T @ a1; rb = rb + a3 * BASE_BASE
    return s1, s2


def write(outdir, strands, seqs, box=30.0):
    """strands: lists of (pos, a1, a3) in generation order, which is 3' -> 5' as in
    generate-sa.py (old-style topology: n3 = previous nucleotide of the strand)"""
    os.makedirs(outdir, exist_ok=True)
    top, conf = [], []
    allpos = np.array([p for s in strands for p, _, _ in s]); shift = box / 2 - allpos.mean(axis=0)
    idx = 0
    for k, (s, seq) in enumerate(zip(strands, seqs)):
        n = len(s)
        for j, ((p, a1, a3), b) in enumerate(zip(s, seq)):
            n3 = idx - 1 if j > 0 else -1
            n5 = idx + 1 if j < n - 1 else -1
            top.append(f'{k+1} {b} {n3} {n5}')
            conf.append(' '.join(f'{v:.10f}' for v in list(p + shift) + list(a1) + list(a3)) + ' 0 0 0 0 0 0')
            idx += 1
    open(f'{outdir}/top.top', 'w').write(f'{idx} {len(strands)}\n' + '\n'.join(top) + '\n')
    open(f'{outdir}/conf.dat', 'w').write(f't = 0\nb = {box} {box} {box}\nE = 0 0 0\n' + '\n'.join(conf) + '\n')


def comp(s):
    return ''.join({'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}[c] for c in reversed(s))


def main():
    what, out = sys.argv[1], sys.argv[2]
    box = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    if what == 'duplex12':
        seq = 'GCTAGCATCGAC'; s1, s2 = helix(12); write(out, [s1, s2], [seq, comp(seq)], box)
    elif what == 'nick16':
        seq = 'GCTAGCATCGACTGAC'; s1, s2 = helix(16)
        write(out, [s1[:8], s1[8:], s2], [seq[:8], seq[8:], comp(seq)], box)
    elif what == 'blunt16':
        seq = 'GCTAGCATCGACTGAC'; s1, s2 = helix(16); c = comp(seq)
        write(out, [s1[:8], s1[8:], s2[:8], s2[8:]], [seq[:8], seq[8:], c[:8], c[8:]], box)
    elif what == 'bulge12':
        seq = 'GCTAGCATCGAC'; s1, s2 = helix(12)
        p6, p7 = s1[5][0], s1[6][0]
        mid = 0.5 * (p6 + p7); rad = mid.copy(); rad[2] = 0; rad /= np.linalg.norm(rad)
        pb = mid + 0.7 * rad                      # outside the helix (backbone bonds 0.89 / 0.71)
        a1b = -rad; a3b = np.array([0, 0, 1.0])
        s1b = s1[:6] + [(pb, a1b, a3b)] + s1[6:]
        write(out, [s1b, s2], [seq[:6] + 'T' + seq[6:], comp(seq)], box)
    elif what == 'ss20':
        seq = 'GCTAGCATCGACTGACATGC'; s1, _ = helix(20); write(out, [s1], [seq], box)
    else:
        sys.exit(__doc__)


main()
