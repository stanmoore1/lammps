#!/usr/bin/env python3
"""Convert old-style oxDNA topology + configuration to a LAMMPS data file
(atom_style hybrid bond ellipsoid oxdna, units lj).
Conventions:
  - LAMMPS atom id = oxDNA index + 1, mol = strand id
  - type: A=1 C=2 G=3 T=4 (LAMMPS CG-DNA convention)
  - bond (i, j): atom i is the 3' neighbour of atom j  (oxDNA n5[i] == j)
  - quaternion: body x axis = a1, body z axis = a3, y = a3 x a1
  - positions wrapped into [0,L) with image flags; velocities zero
usage: ox2lmp.py top conf out.data
"""
import sys
import numpy as np

top, conf, out = sys.argv[1:4]
tl = open(top).read().split('\n')
N, ns = map(int, tl[0].split())
nuc = []
for l in tl[1:1 + N]:
    s, b, n3, n5 = l.split()
    nuc.append((int(s), b, int(n3), int(n5)))
cl = open(conf).read().split('\n')
box = np.array(list(map(float, cl[1].split('=')[1].split())))
dat = np.array([list(map(float, l.split()[:9])) for l in cl[3:3 + N]])
tmap = {'A': 1, 'C': 2, 'G': 3, 'T': 4}


def mat2quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


bonds = [(i + 1, n[3] + 1) for i, n in enumerate(nuc) if n[3] >= 0]
with open(out, 'w') as f:
    f.write(f"LAMMPS data from {top} {conf}\n\n{N} atoms\n4 atom types\n{len(bonds)} bonds\n1 bond types\n{N} ellipsoids\n\n")
    for d, L in zip('xyz', box):
        f.write(f"0.0 {L:.16g} {d}lo {d}hi\n")
    f.write("\nMasses\n\n" + "".join(f"{t} 3.1575\n" for t in range(1, 5)))
    f.write("\nAtoms # hybrid\n\n")
    for i, n in enumerate(nuc):
        r = dat[i, :3]
        img = np.floor(r / box).astype(int)
        rw = r - img * box
        f.write(f"{i+1} {tmap[n[1]]} {rw[0]:.17g} {rw[1]:.17g} {rw[2]:.17g} {n[0]} 1 3.7269849963023267 {img[0]} {img[1]} {img[2]}\n")
    f.write("\nBonds\n\n")
    for k, (a, b) in enumerate(bonds):
        f.write(f"{k+1} 1 {a} {b}\n")
    f.write("\nEllipsoids\n\n")
    for i in range(N):
        a1 = dat[i, 3:6]; a3 = dat[i, 6:9]
        a1 = a1 / np.linalg.norm(a1)
        a3 = a3 - np.dot(a3, a1) * a1; a3 /= np.linalg.norm(a3)
        a2 = np.cross(a3, a1)
        q = mat2quat(np.column_stack([a1, a2, a3]))
        f.write(f"{i+1} 1.173984503142341 1.173984503142341 1.173984503142341 {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g}\n")
