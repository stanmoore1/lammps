import re, os, glob, json
SRC='/home/user/lammps/src/KOKKOS'
# expected typedef base for each atom field, per its declared TransformView/DualView type
FIELD={
 'k_x':'kkfloat_1d_3_lr','k_v':'kkfloat_1d_3','k_f':'kkacc_1d_3',
 'k_omega':'kkfloat_1d_3','k_angmom':'kkfloat_1d_3','k_torque':'kkacc_1d_3',
 'k_mu':'kkfloat_1d_4','k_sp':'kkfloat_1d_4','k_fm':'kkacc_1d_3','k_fm_long':'kkacc_1d_3',
 'k_nspecial':'int_2d','k_special':'tagint_2d',
 'k_bond_type':'int_2d','k_bond_atom':'tagint_2d',
 'k_angle_type':'int_2d','k_angle_atom1':'tagint_2d','k_angle_atom2':'tagint_2d','k_angle_atom3':'tagint_2d',
 'k_dihedral_type':'int_2d','k_dihedral_atom1':'tagint_2d','k_dihedral_atom2':'tagint_2d',
 'k_dihedral_atom3':'tagint_2d','k_dihedral_atom4':'tagint_2d',
 'k_improper_type':'int_2d','k_improper_atom1':'tagint_2d','k_improper_atom2':'tagint_2d',
 'k_improper_atom3':'tagint_2d','k_improper_atom4':'tagint_2d',
 'k_dvector':'kkfloat_2d','k_ivector':'int_2d_lr',
 'k_tag':'tagint_1d','k_type':'int_1d','k_mask':'int_1d','k_image':'imageint_1d',
 'k_q':'kkfloat_1d','k_radius':'kkfloat_1d','k_rmass':'kkfloat_1d','k_mass':'kkfloat_1d',
 'k_molecule':'tagint_1d','k_ellipsoid':'int_1d',
}
assign=re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*atomKK->\s*(k_[A-Za-z0-9_]+)\s*\.\s*(?:template\s+)?view')
# collect declarations from headers
decl={}
for h in glob.glob(SRC+'/*.h'):
    for ln,line in enumerate(open(h,errors='replace'),1):
        m=re.search(r'(?:typename\s+)?(?:AT|DAT|HAT|ArrayTypes<[^>]*>)::t_([A-Za-z0-9_]+)\s+([A-Za-z0-9_, ]+);',line)
        if m:
            for name in m.group(2).split(','):
                decl.setdefault(os.path.basename(h)[:-2],{})[name.strip()]=(m.group(1),ln)
out=[]
for c in glob.glob(SRC+'/*.cpp')+glob.glob(SRC+'/*.h'):
    base=os.path.basename(c)
    stem=base.rsplit('.',1)[0]
    for ln,line in enumerate(open(c,errors='replace'),1):
        m=assign.match(line)
        if not m: continue
        var,field=m.group(1),m.group(2)
        if field not in FIELD: continue
        exp=FIELD[field]
        # find decl: search this stem then all
        found=None
        for k in (stem,):
            if k in decl and var in decl[k]: found=decl[k][var]
        if not found:
            for k,v in decl.items():
                if var in v and (stem.startswith(k) or k.startswith(stem)):
                    found=v[var];break
        if not found: continue
        got=found[0]
        # strip trait suffixes
        g=re.sub(r'_(const_um|const_randomread|const|um|randomread)$','',got)
        if g!=exp:
            out.append({'file':base,'line':ln,'var':var,'field':field,'declared':got,'expected':exp})
print(json.dumps(out,indent=1))
