import os,re,json,collections
SRC='/home/user/lammps/src/KOKKOS'
F2M={
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
funcstart=re.compile(r'^[A-Za-z_][A-Za-z0-9_:<>,\* &~]*::[~A-Za-z0-9_]+\s*\(')
out=[]
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('.cpp'): continue
    path=os.path.join(SRC,fn)
    lines=open(path,encoding='utf-8',errors='replace').read().split('\n')
    full='\n'.join(lines)
    if 'atomKK' not in full: continue
    drs=set();dms=set()
    for m in re.finditer(r'datamask_read\s*\|?=\s*([^;]*);',full):
        drs|=set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(1)))
    for m in re.finditer(r'datamask_modify\s*\|?=\s*([^;]*);',full):
        dms|=set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(1)))
    hdr=path[:-4]+'.h'
    if os.path.exists(hdr):
        h=open(hdr,encoding='utf-8',errors='replace').read()
        for m in re.finditer(r'datamask_read\s*\|?=\s*([^;]*);',h): drs|=set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(1)))
        for m in re.finditer(r'datamask_modify\s*\|?=\s*([^;]*);',h): dms|=set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(1)))
    if 'ALL_MASK' in drs or 'ALL_MASK' in dms: pass
    # split functions
    starts=[i for i,l in enumerate(lines) if funcstart.match(l)]
    starts.append(len(lines))
    for si in range(len(starts)-1):
        a,b=starts[si],starts[si+1]
        name=lines[a].strip()
        body=lines[a:b]
        synced=set();modified=set()
        used=collections.defaultdict(list)
        plain=collections.defaultdict(list)
        for j,l in enumerate(body):
            for m in re.finditer(r'atomKK->sync(?:<[^>]*>)?\s*\(([^;]*)\)',l):
                synced|=set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(1)))
                if 'datamask_read' in m.group(1): synced|=drs
                if 'datamask_modify' in m.group(1): synced|=dms
            for m in re.finditer(r'atomKK->modified(?:<[^>]*>)?\s*\(([^;]*)\)',l):
                modified|=set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(1)))
                if 'datamask_modify' in m.group(1): modified|=dms
            for m in re.finditer(r'atomKK->k_(\w+)',l):
                used[m.group(1)].append((a+j+1,l.strip()))
            for m in re.finditer(r'\batom->(\w+)\b',l):
                if m.group(1) in F2M: plain[m.group(1)].append((a+j+1,l.strip()))
        # continuation lines for sync spanning multiple lines
        blob='\n'.join(body)
        for m in re.finditer(r'atomKK->sync(?:<[^>]*>)?\s*\(([^;]*)\)\s*;',blob,re.S):
            synced|=set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(1)))
            if 'datamask_read' in m.group(1): synced|=drs
        for m in re.finditer(r'atomKK->modified(?:<[^>]*>)?\s*\(([^;]*)\)\s*;',blob,re.S):
            modified|=set(re.findall(r'\b[A-Z_]+_MASK\b',m.group(1)))
            if 'datamask_modify' in m.group(1): modified|=dms
        if 'ALL_MASK' in synced: synced|=set(F2M.values())
        if 'ALL_MASK' in modified: modified|=set(F2M.values())
        miss={}
        for f,locs in list(used.items())+list(plain.items()):
            mk=F2M.get(f)
            if not mk: continue
            if mk in synced or mk in modified: continue
            if mk in drs or mk in dms: continue
            miss.setdefault(mk,[]).extend(locs)
        if miss:
            out.append({'file':'src/KOKKOS/'+fn,'func':name,'line':a+1,
                        'drs':sorted(drs),'dms':sorted(dms),
                        'synced':sorted(synced),'modified':sorted(modified),
                        'miss':{k:v[:3] for k,v in miss.items()}})
json.dump(out,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/fmiss.json','w'),indent=1)
print(len(out))
for o in out[:400]: print(o['file'],'|',o['func'][:70],'|',sorted(o['miss'].keys()))
