#!/usr/bin/env python3
"""Per-function datamask/sync analysis for src/KOKKOS."""
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
 'iarray':'IARRAY_MASK','darray':'DARRAY_MASK','mass':None,
}
# plain host arrays of interest (atom->foo)
HOSTFIELDS = set(FIELD2MASK.keys())

def strip_comments(txt):
    txt = re.sub(r'/\*.*?\*/', lambda m: '\n'*m.group(0).count('\n'), txt, flags=re.S)
    txt = re.sub(r'//[^\n]*', '', txt)
    return txt

def split_functions(path):
    raw = open(path, errors='replace').read()
    src = strip_comments(raw)
    lines = src.split('\n')
    rawlines = raw.split('\n')
    funcs = []
    depth = 0
    cur = None
    for i, l in enumerate(lines):
        if depth == 0 and cur is None:
            if '::' in l and '(' in l and not l.strip().startswith('#') and '{' not in l.strip()[:0]:
                # candidate start; find the opening brace within next lines
                pass
        opens = l.count('{'); closes = l.count('}')
        if depth == 0 and opens > 0:
            # look back for signature: gather up to 4 previous non-empty lines
            sig_lines = []
            j = i
            while j >= 0 and len(sig_lines) < 6:
                s = lines[j].strip()
                sig_lines.insert(0, s)
                if '::' in s and '(' in s: break
                if s.endswith(';') or s == '': break
                j -= 1
            sig = ' '.join(sig_lines)
            m = re.search(r'([A-Za-z_][A-Za-z_0-9]*)\s*::\s*(~?[A-Za-z_][A-Za-z_0-9]*)\s*\(', sig)
            name = (m.group(1)+'::'+m.group(2)) if m else None
            if name:
                cur = {'name': name, 'start': j+1, 'lines': []}
        if cur is not None:
            cur['lines'].append((i+1, rawlines[i], l))
        depth += opens - closes
        if depth <= 0:
            if cur is not None and opens+closes > 0:
                cur['end'] = i+1
                funcs.append(cur)
                cur = None
            depth = 0
    return funcs, rawlines

def masks_in(s):
    return set(re.findall(r'\b[A-Z][A-Z_0-9]*_MASK\b', s))

results = {}
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('_kokkos.cpp'): continue
    path = os.path.join(SRC, fn)
    funcs, rawlines = split_functions(path)
    finfo = []
    for f in funcs:
        kuse = {}   # field -> [lines]
        auses = {}  # atom->field
        syncs = []; mods = []
        for ln, raw, clean in f['lines']:
            for m in re.finditer(r'atomKK->k_([A-Za-z_0-9]+)', clean):
                kuse.setdefault(m.group(1), []).append(ln)
            for m in re.finditer(r'\batom->([A-Za-z_0-9]+)', clean):
                if m.group(1) in HOSTFIELDS:
                    auses.setdefault(m.group(1), []).append(ln)
            if re.search(r'->sync(_overlapping_device|_pinned)?\s*[<(]', clean):
                syncs.append((ln, clean.strip()))
            if re.search(r'->modified\s*[<(]', clean):
                mods.append((ln, clean.strip()))
        if not kuse and not auses and not syncs and not mods: continue
        sync_masks = set()
        for ln, s in syncs: sync_masks |= masks_in(s)
        mod_masks = set()
        for ln, s in mods: mod_masks |= masks_in(s)
        need = {}
        for fld, lns in kuse.items():
            mk = FIELD2MASK.get(fld)
            if mk: need.setdefault(mk, []).extend(lns)
        hneed = {}
        for fld, lns in auses.items():
            mk = FIELD2MASK.get(fld)
            if mk: hneed.setdefault(mk, []).extend(lns)
        finfo.append({
            'func': f['name'], 'start': f['start'], 'end': f.get('end'),
            'kuse': {k: v[:4] for k,v in kuse.items()},
            'atomuse': {k: v[:4] for k,v in auses.items()},
            'need_masks': {k: sorted(set(v))[:4] for k,v in need.items()},
            'host_need_masks': {k: sorted(set(v))[:4] for k,v in hneed.items()},
            'syncs': syncs, 'mods': mods,
            'sync_masks': sorted(sync_masks), 'mod_masks': sorted(mod_masks),
            'calls_other': sorted(set(re.findall(r'\b([a-z_][A-Za-z_0-9]*)\s*\(', ' '.join(c for _,_,c in f['lines'])))),
        })
    # constructor datamask
    txt = strip_comments(open(path, errors='replace').read())
    hpath = path[:-4]+'.h'
    if os.path.exists(hpath): txt += strip_comments(open(hpath, errors='replace').read())
    dmr = re.findall(r'datamask_read\s*\|?=\s*([^;]*);', txt, re.S)
    dmm = re.findall(r'datamask_modify\s*\|?=\s*([^;]*);', txt, re.S)
    results[fn] = {'funcs': finfo,
                   'dmr': [' '.join(x.split()) for x in dmr],
                   'dmm': [' '.join(x.split()) for x in dmm],
                   'dmr_masks': sorted(set().union(*[masks_in(x) for x in dmr]) if dmr else set()),
                   'dmm_masks': sorted(set().union(*[masks_in(x) for x in dmm]) if dmm else set())}

json.dump(results, open(os.path.join(OUT,'scan2.json'),'w'), indent=1)
print("files", len(results), "funcs", sum(len(v['funcs']) for v in results.values()))
