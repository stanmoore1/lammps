import re,glob,os
base='/home/user/lammps/src/KOKKOS'
for h in sorted(glob.glob(base+'/*.h')):
    stem=h[:-2]; txt=open(h).read()
    if 'DeviceType' not in txt: continue
    decls={}
    for i,l in enumerate(txt.split('\n'),1):
        m=re.match(r'\s*(?:const\s+)?typename AT::(t_\w+)\s+([\w,\s]+);',l)
        if m:
            for nm in m.group(2).split(','):
                nm=nm.strip()
                if nm: decls[nm]=(i,m.group(1))
    if not decls: continue
    srcs=[h]+[p for p in (stem+'.cpp',stem+'_impl.h') if os.path.exists(p)]
    for s in srcs:
        for i,l in enumerate(open(s).read().split('\n'),1):
            m=re.search(r'\b(\w+)\s*=\s*[^;]*?\.(?:template\s+)?(view_device\(\)|view<LMPDeviceType>\(\)|d_view|view_host\(\)|view<LMPHostType>\(\)|h_view)',l)
            if m and m.group(1) in decls:
                print('MIX',os.path.basename(s),i,l.strip(),'| decl',os.path.basename(h),decls[m.group(1)])
