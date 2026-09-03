#!/usr/bin/env python3
"""Per-function: host READ of own k_* whose data is produced on device, without sync_host in that function."""
import os,re,json
SRC="/home/user/lammps/src/KOKKOS"
OUT=os.path.dirname(os.path.abspath(__file__))
def strip(t):
    t=re.sub(r'/\*.*?\*/', lambda m:'\n'*m.group(0).count('\n'), t, flags=re.S)
    return re.sub(r'//[^\n]*','',t)
SIGRE = re.compile(r'([A-Za-z_][A-Za-z_0-9]*)\s*(?:<[^<>]*>)?\s*::\s*(~?[A-Za-z_][A-Za-z_0-9]*)\s*\(')
def split_functions(path):
    clean=strip(open(path,errors='replace').read()).split('\n'); n=len(clean)
    funcs=[]; i=0
    while i<n:
        if clean[i].rstrip()=='{':
            j=i-1; sig=[]
            while j>=0 and len(sig)<8:
                s=clean[j].strip()
                if s=='' or s.startswith('#'):
                    if sig: break
                    j-=1; continue
                sig.insert(0,s)
                if SIGRE.search(' '.join(sig)): break
                j-=1
            m=SIGRE.search(' '.join(sig)); name=(m.group(1)+'::'+m.group(2)) if m else '?'
            k=i+1
            while k<n and not clean[k].startswith('}'): k+=1
            funcs.append({'name':name,'lines':[(x+1,clean[x]) for x in range(i+1,min(k+1,n))]})
            i=k+1
        else: i+=1
    return funcs
res=[]
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('.cpp'): continue
    p=os.path.join(SRC,fn)
    txt=strip(open(p,errors='replace').read())
    devmod={m.group(1) for m in re.finditer(r'(?<![>\w])\b(k_[A-Za-z_0-9]+)\s*\.\s*(?:template\s+)?(?:modify\s*<\s*(?:DeviceType|LMPDeviceType)\s*>|modify_device)',txt)}
    if not devmod: continue
    for f in split_functions(p):
        body='\n'.join(l for _,l in f['lines'])
        for n in devmod:
            reads=[ln for ln,l in f['lines'] if re.search(r'(?<![>\w])\b'+n+r'\s*\.\s*(h_view|view_host)\s*\(\s*\)?\s*[\[\(]',l)]
            if not reads: continue
            if re.search(r'(?<![>\w])\b'+n+r'\s*\.\s*(?:template\s+)?(sync\s*<\s*LMPHostType\s*>|sync_host)',body): continue
            res.append((fn,f['name'],n,reads[:4]))
json.dump(res,open(os.path.join(OUT,'scan12.json'),'w'),indent=1)
for r in res: print(r)
