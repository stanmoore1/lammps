import os,re,json,collections

SRC='/home/user/lammps/src/KOKKOS'
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

def base(fn):
    return fn[:-4] if fn.endswith('.cpp') else fn[:-2]

groups=collections.defaultdict(dict)
for fn in os.listdir(SRC):
    if fn.endswith('.cpp') or fn.endswith('.h'):
        groups[base(fn)][fn[fn.rindex('.'):]]=os.path.join(SRC,fn)

res={}
for g,files in sorted(groups.items()):
    kfields=collections.defaultdict(list)   # field -> [(file,line,text)]
    plain=collections.defaultdict(list)
    dr=[]; dm=[]
    syncs=[]; mods=[]
    for ext,path in files.items():
        txt=open(path,encoding='utf-8',errors='replace').read().split('\n')
        for i,l in enumerate(txt,1):
            for m in re.finditer(r'atomKK->k_(\w+)', l):
                kfields[m.group(1)].append((os.path.basename(path),i,l.strip()))
            for m in re.finditer(r'\batom->(\w+)\b', l):
                plain[m.group(1)].append((os.path.basename(path),i,l.strip()))
            if 'datamask_read' in l: dr.append((os.path.basename(path),i,l.strip()))
            if 'datamask_modify' in l: dm.append((os.path.basename(path),i,l.strip()))
            if re.search(r'atomKK->sync', l): syncs.append((os.path.basename(path),i,l.strip()))
            if re.search(r'atomKK->modified', l): mods.append((os.path.basename(path),i,l.strip()))
    res[g]={'k':{k:v for k,v in kfields.items()},'plain':{k:v for k,v in plain.items()},
            'dr':dr,'dm':dm,'sync':syncs,'mod':mods}

json.dump(res,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/scan.json','w'),indent=1)
print(len(res))
