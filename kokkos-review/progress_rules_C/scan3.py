#!/usr/bin/env python3
"""Per-function datamask/sync analysis for src/KOKKOS (block split on column-0 '}')."""
import os, re, json

SRC = "/home/user/lammps/src/KOKKOS"
OUT = "/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_C"

FIELD2MASK = {
 'x':'X_MASK','v':'V_MASK','f':'F_MASK','tag':'TAG_MASK','type':'TYPE_MASK',
 'mask':'MASK_MASK','image':'IMAGE_MASK','q':'Q_MASK','molecule':'MOLECULE_MASK',
 'rmass':'RMASS_MASK','num_bond':'BOND_MASK','bond_type':'BOND_MASK','bond_atom':'BOND_MASK',
 'num_angle':'ANGLE_MASK','angle_type':'ANGLE_MASK','angle_atom1':'ANGLE_MASK',
 'angle_atom2':'ANGLE_MASK','angle_atom3':'ANGLE_MASK',
 'num_dihedral':'DIHEDRAL_MASK','dihedral_type':'DIHEDRAL_MASK','dihedral_atom1':'DIHEDRAL_MASK',
 'dihedral_atom2':'DIHEDRAL_MASK','dihedral_atom3':'DIHEDRAL_MASK','dihedral_atom4':'DIHEDRAL_MASK',
 'num_improper':'IMPROPER_MASK','improper_type':'IMPROPER_MASK','improper_atom1':'IMPROPER_MASK',
 'improper_atom2':'IMPROPER_MASK','improper_atom3':'IMPROPER_MASK','improper_atom4':'IMPROPER_MASK',
 'nspecial':'SPECIAL_MASK','special':'SPECIAL_MASK','mu':'MU_MASK','ellipsoid':'ELLIPSOID_MASK',
 'sp':'SP_MASK','fm':'FM_MASK','fm_long':'FML_MASK','rho':'DPDRHO_MASK','dpdTheta':'DPDTHETA_MASK',
 'uCond':'UCOND_MASK','uMech':'UMECH_MASK','uChem':'UCHEM_MASK','uCG':'UCG_MASK',
 'uCGnew':'UCGNEW_MASK','duChem':'DUCHEM_MASK','radius':'RADIUS_MASK','omega':'OMEGA_MASK',
 'torque':'TORQUE_MASK','angmom':'ANGMOM_MASK','dvector':'DVECTOR_MASK','ivector':'IVECTOR_MASK',
 'iarray':'IARRAY_MASK','darray':'DARRAY_MASK',
}
HOSTFIELDS = set(FIELD2MASK.keys())

def strip_comments(txt):
    txt = re.sub(r'/\*.*?\*/', lambda m: '\n'*m.group(0).count('\n'), txt, flags=re.S)
    txt = re.sub(r'//[^\n]*', '', txt)
    return txt

def masks_in(s):
    return set(re.findall(r'\b[A-Z][A-Z_0-9]*_MASK\b', s))

def split_functions(path):
    raw = open(path, errors='replace').read()
    clean = strip_comments(raw).split('\n')
    raws = raw.split('\n')
    # block boundaries: lines that are exactly '}' at col0
    bounds = [i for i,l in enumerate(clean) if l.startswith('}')]
    funcs = []
    start = 0
    for b in bounds:
        blk = list(range(start, b+1))
        # find signature: last line index <= b with '::' and '(' starting at col 0-ish
        sig = None; sigidx = None
        for i in blk:
            l = clean[i]
            if re.match(r'^[A-Za-z_~]', l) or re.match(r'^\s*[A-Za-z_]', l):
                pass
        # join whole block, search for the FIRST '::name(' occurrence at line start region
        for i in blk:
            m = re.search(r'([A-Za-z_][A-Za-z_0-9]*)\s*::\s*(~?[A-Za-z_][A-Za-z_0-9]*)\s*\(', clean[i])
            if m and not clean[i].lstrip().startswith(('return','if','}','//')):
                sig = m.group(1)+'::'+m.group(2); sigidx = i; break
        if sig:
            funcs.append({'name': sig, 'start': sigidx+1, 'end': b+1,
                          'lines': [(i+1, raws[i], clean[i]) for i in range(sigidx, b+1)]})
        start = b+1
    return funcs

results = {}
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('_kokkos.cpp'): continue
    path = os.path.join(SRC, fn)
    funcs = split_functions(path)
    finfo = []
    for f in funcs:
        kuse = {}; auses = {}; syncs = []; mods = []
        for ln, raw, cl in f['lines']:
            for m in re.finditer(r'atomKK->k_([A-Za-z_0-9]+)', cl):
                kuse.setdefault(m.group(1), []).append(ln)
            for m in re.finditer(r'\batom->([A-Za-z_0-9]+)\b', cl):
                if m.group(1) in HOSTFIELDS: auses.setdefault(m.group(1), []).append(ln)
            if re.search(r'->sync(_overlapping_device|_pinned)?\s*[<(]', cl): syncs.append((ln, cl.strip()))
            if re.search(r'->modified\s*[<(]', cl): mods.append((ln, cl.strip()))
        if not (kuse or auses or syncs or mods): continue
        sm = set(); 
        for ln,s in syncs: sm |= masks_in(s)
        mm = set()
        for ln,s in mods: mm |= masks_in(s)
        need = {}; hneed = {}
        for fld,l in kuse.items():
            mk = FIELD2MASK.get(fld)
            if mk: need.setdefault(mk, []).extend(l)
        for fld,l in auses.items():
            mk = FIELD2MASK.get(fld)
            if mk: hneed.setdefault(mk, []).extend(l)
        finfo.append({'func': f['name'], 'start': f['start'], 'end': f['end'],
            'kuse': {k: sorted(set(v)) for k,v in kuse.items()},
            'atomuse': {k: sorted(set(v)) for k,v in auses.items()},
            'need': {k: sorted(set(v)) for k,v in need.items()},
            'hneed': {k: sorted(set(v)) for k,v in hneed.items()},
            'syncs': syncs, 'mods': mods,
            'sync_masks': sorted(sm), 'mod_masks': sorted(mm)})
    txt = strip_comments(open(path, errors='replace').read())
    hp = path[:-4]+'.h'
    if os.path.exists(hp): txt += strip_comments(open(hp, errors='replace').read())
    dmr = re.findall(r'datamask_read\s*\|?=\s*([^;]*);', txt, re.S)
    dmm = re.findall(r'datamask_modify\s*\|?=\s*([^;]*);', txt, re.S)
    results[fn] = {'funcs': finfo,
        'dmr': [' '.join(x.split()) for x in dmr], 'dmm': [' '.join(x.split()) for x in dmm],
        'dmr_masks': sorted(set().union(*[masks_in(x) for x in dmr]) if dmr else set()),
        'dmm_masks': sorted(set().union(*[masks_in(x) for x in dmm]) if dmm else set())}

json.dump(results, open(os.path.join(OUT,'scan3.json'),'w'), indent=1)
print("files", len(results), "funcs", sum(len(v['funcs']) for v in results.values()))
