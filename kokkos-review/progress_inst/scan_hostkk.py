import re,glob,os
base='/home/user/lammps/src/KOKKOS'
txt=open(base+'/atom_kokkos.h').read()
kind={}
for m in re.finditer(r'(DAT::)?(ttransform_\w+|tdual_\w+)\s+((?:k_\w+\s*,\s*)*k_\w+)\s*;',txt):
    t='transform' if m.group(2).startswith('ttransform') else 'dual'
    for nm in re.findall(r'k_\w+',m.group(3)):
        kind[nm]=t
print('fields classified:',len(kind))
bad=[]
for f in sorted(glob.glob(base+'/atom_vec_*_kokkos.cpp')):
    body=open(f).read()
    lines=body.split('\n')
    inhostkk=False
    for i,l in enumerate(lines,1):
        if re.search(r'\(space\s*==\s*HostKK\)',l): inhostkk=True; continue
        if inhostkk and re.match(r'\s*\}',l): inhostkk=False
        if not inhostkk: continue
        m=re.search(r'(?:atomKK->)?(k_\w+)\.(sync_host|sync_hostkk|modify_host|modify_hostkk)\(\)',l)
        if m:
            fld,op=m.group(1),m.group(2)
            k=kind.get(fld)
            if k is None: continue
            want = 'hostkk' if k=='transform' else 'host'
            got = 'hostkk' if op.endswith('hostkk') else 'host'
            if want!=got:
                print('MISMATCH',os.path.basename(f),i,l.strip(),'field kind',k)
