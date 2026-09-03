#!/usr/bin/env python3
"""Scan src/KOKKOS for datamask vs actual atomKK->k_<field> usage."""
import os, re, json, sys

SRC = "/home/user/lammps/src/KOKKOS"

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

def basename_style(fn):
    return fn

def collect(path):
    txt = open(path, errors='replace').read()
    lines = txt.split('\n')
    return lines

def find_files():
    out = []
    for fn in sorted(os.listdir(SRC)):
        if fn.endswith('_kokkos.cpp'):
            out.append(fn)
    return out

res = {}
for cpp in find_files():
    p = os.path.join(SRC, cpp)
    h = p[:-4]+'.h'
    lines = collect(p)
    hlines = collect(h) if os.path.exists(h) else []
    all_lines = [(cpp,i+1,l) for i,l in enumerate(lines)] + [(os.path.basename(h),i+1,l) for i,l in enumerate(hlines)]

    kfields = {}   # field -> list of (file,line,text)
    for f,i,l in all_lines:
        for m in re.finditer(r'atomKK->k_([A-Za-z_0-9]+)', l):
            kfields.setdefault(m.group(1), []).append((f,i,l.strip()))
    atomfields = {}
    for f,i,l in all_lines:
        for m in re.finditer(r'\batom->([A-Za-z_0-9]+)', l):
            atomfields.setdefault(m.group(1), []).append((f,i,l.strip()))

    # datamask assignments anywhere in cpp/h
    dmr, dmm = [], []
    for f,i,l in all_lines:
        if 'datamask_read' in l and '=' in l:
            dmr.append((f,i,l.strip()))
        if 'datamask_modify' in l and '=' in l:
            dmm.append((f,i,l.strip()))
    # gather masks mentioned (multi-line continuation)
    def gather(txtlines, key):
        # find "datamask_read = ..." possibly continued over lines until ';'
        masks = set()
        raw = []
        joined = '\n'.join(txtlines)
        for m in re.finditer(key + r'\s*(\|)?=\s*([^;]*);', joined, re.S):
            raw.append(m.group(0).replace('\n',' '))
            for mm in re.finditer(r'\b([A-Z_]+_MASK)\b', m.group(2)):
                masks.add(mm.group(1))
        return masks, raw
    mr, mr_raw = gather(lines+hlines, 'datamask_read')
    mm_, mm_raw = gather(lines+hlines, 'datamask_modify')

    syncs = []
    for f,i,l in all_lines:
        if re.search(r'->sync\b|->sync<|->sync_overlapping|->modified\b|->modified<|sync_pinned', l):
            syncs.append((f,i,l.strip()))

    res[cpp] = {
      'kfields': kfields, 'atomfields': atomfields,
      'datamask_read': sorted(mr), 'datamask_modify': sorted(mm_),
      'dmr_raw': mr_raw, 'dmm_raw': mm_raw,
      'syncs': syncs,
    }

# report missing masks
report = []
for cpp, d in sorted(res.items()):
    if not d['dmr_raw'] and not d['dmm_raw']:
        # style may not set datamask at all
        pass
    used = {}
    for fld, occ in d['kfields'].items():
        mk = FIELD2MASK.get(fld)
        if mk: used.setdefault(mk, []).append((fld, occ))
    missing_read = []
    for mk, info in sorted(used.items()):
        if mk not in d['datamask_read']:
            missing_read.append((mk, [(f, o[0][0], o[0][1], o[0][2]) for f,o in [(x[0],x[1]) for x in info]]))
    report.append({
      'file': cpp,
      'datamask_read': d['datamask_read'],
      'datamask_modify': d['datamask_modify'],
      'kfields_used': sorted(used.keys()),
      'kfield_names': sorted(d['kfields'].keys()),
      'missing_from_read': sorted([mk for mk in used if mk not in d['datamask_read']]),
      'in_read_not_used': sorted([mk for mk in d['datamask_read'] if mk not in used and mk not in ('ALL_MASK','EMPTY_MASK','ENERGY_MASK','VIRIAL_MASK')]),
      'modify_not_used': sorted([mk for mk in d['datamask_modify'] if mk not in used and mk not in ('ALL_MASK','EMPTY_MASK','ENERGY_MASK','VIRIAL_MASK')]),
      'has_dm': bool(d['dmr_raw'] or d['dmm_raw']),
      'nsync': len(d['syncs']),
    })

json.dump(res, open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_C/scan_raw.json','w'), indent=1, default=str)
json.dump(report, open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_C/scan_report.json','w'), indent=1)
print("files:", len(res))
for r in report:
    if r['missing_from_read']:
        print("MISSING_READ", r['file'], r['missing_from_read'], "have:", r['datamask_read'])
