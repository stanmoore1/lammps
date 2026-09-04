import re,glob,os
base='/home/user/lammps/src/KOKKOS'
# collect classes registered with LMPHostType
hostcls=set(); devcls=set()
for h in glob.glob(base+'/*.h'):
    for l in open(h):
        m=re.match(r'\s*\w+Style\(\S+?,\s*(\w+)<LMPHostType>\s*\)',l)
        if m: hostcls.add(m.group(1))
        m=re.match(r'\s*\w+Style\(\S+?,\s*(\w+)<LMPDeviceType>\s*\)',l)
        if m: devcls.add(m.group(1))
# collect explicit instantiations
inst={}
for f in glob.glob(base+'/*.cpp')+glob.glob(base+'/*.h'):
    t=open(f).read()
    for m in re.finditer(r'template class\s+(?:\w+::)?(\w+)\s*<\s*([^>;]+)>\s*;',t):
        inst.setdefault(m.group(1),set()).add(m.group(2).strip())
for c in sorted(hostcls):
    have=inst.get(c,set())
    if not any('LMPHostType' in x for x in have):
        print('NO HOST INSTANTIATION:',c,have)
for c in sorted(devcls):
    have=inst.get(c,set())
    if not any('LMPDeviceType' in x for x in have):
        print('NO DEVICE INSTANTIATION:',c,have)
