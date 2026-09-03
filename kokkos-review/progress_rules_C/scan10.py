#!/usr/bin/env python3
"""Per-function: host write to own k_* dual view without modify_host in the same function."""
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
            funcs.append({'name':name,'start':i+1,'end':k+1,'lines':[(x+1,clean[x]) for x in range(i+1,min(k+1,n))]})
            i=k+1
        else: i+=1
    return funcs
res={}
WRITE=re.compile(r'(?<!>)\b(k_[A-Za-z_0-9]+)\s*\.\s*(?:h_view|view_host\s*\(\s*\))\s*[\[\(][^;]*?[\]\)]\s*(?:=[^=]|\+=|-=|\*=)')
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('.cpp'): continue
    p=os.path.join(SRC,fn)
    for f in split_functions(p):
        written={}
        mods=set()
        for ln,l in f['lines']:
            for m in WRITE.finditer(l):
                written.setdefault(m.group(1),ln)
            for m in re.finditer(r'(?<!>)\b(k_[A-Za-z_0-9]+)\s*\.\s*(?:template\s+)?(?:modify<\s*LMPHostType\s*>|modify_host)',l):
                mods.add(m.group(1))
            # h_ alias writes
        miss=[(n,ln) for n,ln in written.items() if n not in mods]
        if miss:
            res.setdefault(fn,[]).append((f['name'],miss))
json.dump(res,open(os.path.join(OUT,'scan10.json'),'w'),indent=1)
for fn,v in sorted(res.items()):
    for name,miss in v:
        print(fn,name,miss)
