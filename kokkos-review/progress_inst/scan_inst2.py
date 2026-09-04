import re,glob,os
base='/home/user/lammps/src/KOKKOS'
hostcls={}
for h in sorted(glob.glob(base+'/*.h')):
    txt=open(h).read()
    guarded = 'LMP_KOKKOS_GPU' in txt.split('#else')[0] if '#else' in txt else False
    for l in txt.split('\n'):
        m=re.match(r'\s*\w+Style\(\S+?,\s*(\w+)<LMPHostType>\s*\)',l)
        if m: hostcls[m.group(1)]=os.path.basename(h)
inst={}
for f in glob.glob(base+'/*.cpp')+glob.glob(base+'/*.h'):
    t=open(f).read()
    for m in re.finditer(r'template class\s+(?:\w+::)?(\w+)\s*<\s*([^>;]+)>\s*;',t):
        inst.setdefault(m.group(1),{})[m.group(2).strip()]=os.path.basename(f)
for c,h in sorted(hostcls.items()):
    have=inst.get(c,{})
    hostkeys=[k for k in have if 'LMPHostType' in k]
    if not hostkeys: print('MISSING HOST INST:',c,h,have)
