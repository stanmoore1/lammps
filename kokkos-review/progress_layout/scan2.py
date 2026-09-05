import re,glob,os,json
SRC='/home/user/lammps/src/KOKKOS'
decl={}
for h in glob.glob(SRC+'/*.h'):
    b=os.path.basename(h)[:-2]
    for ln,line in enumerate(open(h,errors='replace'),1):
        m=re.search(r'(?:typename\s+)?(?:AT|DAT|HAT|ArrayTypes<[^>]*>|FFT_AT|FFT_DAT)::t(?:dual)?_([A-Za-z0-9_]+)\s+([A-Za-z0-9_, ]+);',line)
        if m:
            for n in m.group(2).split(','):
                decl.setdefault(b,{})[n.strip()]=m.group(1)
        m2=re.search(r'Kokkos::(?:Dual)?View<([^;]*?)>\s+([A-Za-z0-9_, ]+);',line)
        if m2:
            for n in m2.group(2).split(','):
                decl.setdefault(b,{})[n.strip()]='RAW['+m2.group(1)+']'
dc=re.compile(r'Kokkos::deep_copy\(\s*(?:LMPHostType\(\)\s*,\s*)?([A-Za-z0-9_>:.<()]+)\s*,\s*([A-Za-z0-9_>:.<()]+)\s*\)')
res=[]
for c in sorted(glob.glob(SRC+'/*.cpp')+glob.glob(SRC+'/*.h')):
    b=os.path.basename(c); stem=b.rsplit('.',1)[0]
    hdrs=[k for k in decl if stem.startswith(k)]
    for ln,line in enumerate(open(c,errors='replace'),1):
        m=dc.search(line)
        if not m: continue
        a,bb=m.group(1),m.group(2)
        if re.match(r'^-?[0-9.]+$',bb) or bb in ('0','0.0'): continue
        def look(v):
            v=v.split('.')[0]
            for k in hdrs:
                if v in decl[k]: return decl[k][v]
            for k,d in decl.items():
                if v in d: return d[v]
            return None
        ta,tb=look(a),look(bb)
        if ta and tb and ta!=tb:
            res.append((b,ln,a,ta,bb,tb))
for r in res: print(r)
