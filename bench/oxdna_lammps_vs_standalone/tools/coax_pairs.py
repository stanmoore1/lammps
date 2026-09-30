#!/usr/bin/env python3
"""Q3: list the coaxial-stacking pairs of standalone oxDNA (pair_energy observable, file
pairs_ox.dat) with the strand-terminal status of both nucleotides. LAMMPS (oxdna*/coaxstk)
only evaluates pairs of two terminal nucleotides (id3p or id5p unset), so every pair with a
non-terminal nucleotide is missing there.
usage: coax_pairs.py <top> <pairs_ox.dat>"""
import sys
top, pf = sys.argv[1:3]
tl = [l.split() for l in open(top).read().split('\n')[1:] if l.strip()]
s = [int(t[0]) for t in tl]; n3 = [int(t[2]) for t in tl]; n5 = [int(t[3]) for t in tl]
pos = []
cnt = {}
for i in range(len(tl)):
    cnt[s[i]] = cnt.get(s[i], 0); pos.append(cnt[s[i]]); cnt[s[i]] += 1
term = lambda i: n3[i] == -1 or n5[i] == -1
tot = {True: 0.0, False: 0.0}
for l in open(pf):
    if l.startswith('#') or not l.strip(): continue
    p = l.split(); a, b = int(p[0]), int(p[1]); cx = float(p[8])
    if cx == 0: continue
    both = term(a) and term(b); tot[both] += cx
    def desc(i):
        e = (", 3' end" if n3[i] == -1 else "") + (", 5' end" if n5[i] == -1 else "")
        return f'{i} (strand {s[i]}, nt {pos[i]}{e})'
    print(f'  pair {desc(a)} - {desc(b)}: E_cxst = {cx:+.5f}  -> {"evaluated in LAMMPS" if both else "NOT evaluated in LAMMPS (non-terminal nucleotide)"}')
print(f'  total standalone coaxial stacking {tot[True] + tot[False]:+.5f}: terminal-terminal pairs {tot[True]:+.5f}, pairs with a non-terminal nucleotide {tot[False]:+.5f}')
