import re,glob,os
base='/home/user/lammps/src/KOKKOS'
for f in sorted(glob.glob(base+'/*.h'))+sorted(glob.glob(base+'/*.cpp')):
    txt=open(f).read()
    if 'template<class DeviceType>' not in txt and 'template <class DeviceType>' not in txt: continue
    lines=txt.split('\n')
    for i,l in enumerate(lines,1):
        s=l.strip()
        if s.startswith('//') or s.startswith('*'): continue
        if 'DAT::' in s or ('LMPDeviceType' in s and 'ArrayTypes<LMPDeviceType>' not in s):
            # skip registration lines and instantiation lines
            if re.match(r'^\s*\w+Style\(',s): continue
            if s.startswith('template class'): continue
            print(f"{os.path.basename(f)}:{i}: {s}")
