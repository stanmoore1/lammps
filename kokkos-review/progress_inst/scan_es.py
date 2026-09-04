import re,glob,os
base='/home/user/lammps/src/KOKKOS'
hdrs=sorted(glob.glob(base+'/*.h'))
classre=re.compile(r'^(template\s*<[^>]*>\s*)?class\s+(\w+)\s*:\s*public\s+([\w:<>, ]+)\s*\{?',re.M)
res=[]
for h in hdrs:
    txt=open(h).read()
    # find class decls
    for m in re.finditer(r'template\s*<class DeviceType>\s*\n\s*class\s+(\w+)\s*:\s*public\s+([^\{\n]+)', txt):
        cls,parent=m.group(1),m.group(2).strip().rstrip('{').strip()
        stem=os.path.basename(h)[:-2]
        # candidate impl files
        cands=[base+'/'+stem+'.cpp', base+'/'+stem+'_impl.h', h]
        found=None
        for c in cands:
            if os.path.exists(c):
                t=open(c).read()
                for mm in re.finditer(r'execution_space\s*=\s*([^;]+);',t):
                    found=(os.path.basename(c),mm.group(1).strip())
                    break
            if found: break
        res.append((os.path.basename(h),cls,parent,found))
for r in res:
    print(r)
