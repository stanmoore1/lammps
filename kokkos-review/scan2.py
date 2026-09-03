import os,re,json,collections
SRC='/home/user/lammps/src/KOKKOS'
exec(open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/scan.py').read().split('groups=')[0].split("SRC=")[1].split('\n',1)[1]) if False else None
FIELD2MASK={
 'x':'X_MASK','v':'V_MASK','f':'F_MASK','tag':'TAG_MASK','type':'TYPE_MASK',
 'mask':'MASK_MASK','image':'IMAGE_MASK','q':'Q_MASK','molecule':'MOLECULE_MASK',
 'rmass':'RMASS_MASK','radius':'RADIUS_MASK','omega':'OMEGA_MASK','torque':'TORQUE_MASK',
 'angmom':'ANGMOM_MASK','mu':'MU_MASK','ellipsoid':'ELLIPSOID_MASK',
 'sp':'SP_MASK','fm':'FM_MASK','fm_long':'FML_MASK',
 'rho':'DPDRHO_MASK','dpdTheta':'DPDTHETA_MASK','uCond':'UCOND_MASK','uMech':'UMECH_MASK',
 'uChem':'UCHEM_MASK','uCG':'UCG_MASK','uCGnew':'UCGNEW_MASK','duChem':'DUCHEM_MASK',
 'nspecial':'SPECIAL_MASK','special':'SPECIAL_MASK',
 'num_bond':'BOND_MASK','bond_type':'BOND_MASK','bond_atom':'BOND_MASK',
 'num_angle':'ANGLE_MASK','angle_type':'ANGLE_MASK','angle_atom1':'ANGLE_MASK',
 'angle_atom2':'ANGLE_MASK','angle_atom3':'ANGLE_MASK',
 'num_dihedral':'DIHEDRAL_MASK','dihedral_type':'DIHEDRAL_MASK','dihedral_atom1':'DIHEDRAL_MASK',
 'dihedral_atom2':'DIHEDRAL_MASK','dihedral_atom3':'DIHEDRAL_MASK','dihedral_atom4':'DIHEDRAL_MASK',
 'num_improper':'IMPROPER_MASK','improper_type':'IMPROPER_MASK','improper_atom1':'IMPROPER_MASK',
 'improper_atom2':'IMPROPER_MASK','improper_atom3':'IMPROPER_MASK','improper_atom4':'IMPROPER_MASK',
 'dvector':'DVECTOR_MASK','ivector':'IVECTOR_MASK','iarray':'IARRAY_MASK','darray':'DARRAY_MASK',
}
def base(fn): return fn[:fn.rindex('.')]
groups=collections.defaultdict(dict)
for fn in os.listdir(SRC):
    if fn.endswith('.cpp') or fn.endswith('.h'):
        groups[base(fn)][fn[fn.rindex('.'):]]=os.path.join(SRC,fn)
out=[]
for g,files in sorted(groups.items()):
    txts={}
    for ext,path in files.items(): txts[path]=open(path,encoding='utf-8',errors='replace').read()
    full='\n'.join(txts.values())
    if 'atomKK' not in full: continue
    drs=set(); dms=set()
    for m in re.finditer(r'datamask_read\s*(\|)?=\s*([^;]*);',full):
        drs |= set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(2)))
    for m in re.finditer(r'datamask_modify\s*(\|)?=\s*([^;]*);',full):
        dms |= set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(2)))
    kf=collections.defaultdict(list)
    for path,t in txts.items():
        for i,l in enumerate(t.split('\n'),1):
            for m in re.finditer(r'atomKK->k_(\w+)',l):
                kf[m.group(1)].append((os.path.basename(path),i,l.strip()))
    needed={}
    for f,locs in kf.items():
        mk=FIELD2MASK.get(f)
        if mk: needed.setdefault(mk,[]).extend(locs)
    missing_read=[m for m in needed if m not in drs and m not in dms and m not in ('ENERGY_MASK','VIRIAL_MASK')]
    if not drs and not dms: continue
    if missing_read:
        out.append({'style':g,'drs':sorted(drs),'dms':sorted(dms),
                    'missing':{m:needed[m][:4] for m in missing_read}})
json.dump(out,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/missing.json','w'),indent=1)
print(len(out))
for o in out: print(o['style'], list(o['missing'].keys()))
