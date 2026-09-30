#!/usr/bin/env python3
"""Replicate an oxDNA system (old-style topology + configuration) on an n x n x n grid.
The box grows by n in every direction; strands and nucleotide indices are renumbered.
usage: tile.py <top> <conf> <n> <out_top> <out_conf>"""
import sys
top, conf, n, otop, oconf = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
tl = open(top).read().split('\n')
N, ns = map(int, tl[0].split()[:2])
if len(tl[0].split()) > 2:
    sys.exit('tile.py: only the old-style topology (N N_strands, then strand base n3 n5) is supported')
nuc = [l.split() for l in tl[1:1 + N]]
cl = open(conf).read().split('\n')
box = [float(x) for x in cl[1].split('=')[1].split()]
rows = [l.split() for l in cl[3:3 + N]]
T, C = [], []
k = 0
for ix in range(n):
    for iy in range(n):
        for iz in range(n):
            off = (ix * box[0], iy * box[1], iz * box[2])
            for s, b, n3, n5 in nuc:
                n3, n5 = int(n3), int(n5)
                T.append(f'{int(s) + k * ns} {b} {n3 + k * N if n3 >= 0 else -1} {n5 + k * N if n5 >= 0 else -1}')
            for r in rows:
                x = [float(r[d]) + off[d] for d in range(3)]
                C.append(' '.join([f'{v:.10f}' for v in x] + r[3:]))
            k += 1
open(otop, 'w').write(f'{N * k} {ns * k}\n' + '\n'.join(T) + '\n')
open(oconf, 'w').write(f't = 0\nb = {box[0]*n} {box[1]*n} {box[2]*n}\nE = 0 0 0\n' + '\n'.join(C) + '\n')
print(f'{N * k} nucleotides, {ns * k} strands, box {box[0]*n} x {box[1]*n} x {box[2]*n}')
