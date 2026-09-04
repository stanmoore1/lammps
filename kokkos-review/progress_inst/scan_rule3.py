import re,glob,os
base='/home/user/lammps/src/KOKKOS'
# collect member names declared with DAT::t_ (device-pinned view type)
decls={}  # name -> (file,line,type)
for h in glob.glob(base+'/*.h')+glob.glob(base+'/*.cpp'):
    for i,l in enumerate(open(h).read().split('\n'),1):
        m=re.match(r'\s*(?:const\s+)?DAT::(t_\w+)\s+([\w,\s]+);\s*(//.*)?$',l)
        if m:
            for nm in m.group(2).split(','):
                nm=nm.strip()
                if nm: decls.setdefault(nm,[]).append((os.path.basename(h),i,m.group(1)))
# find assignments from atomKK dual views with view<DeviceType>()
for f in glob.glob(base+'/*.cpp')+glob.glob(base+'/*.h'):
    txt=open(f).read()
    for i,l in enumerate(txt.split('\n'),1):
        m=re.search(r'(\w+)\s*=\s*(atomKK->k_\w+|\w+KK->k_\w+)\.(?:template\s+)?view<DeviceType>\(\)',l)
        if m and m.group(1) in decls:
            print('RULE3',os.path.basename(f),i,l.strip(),'DECL:',decls[m.group(1)])
