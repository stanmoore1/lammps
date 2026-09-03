#!/usr/bin/env python3
"""All own k_* dual views: device modify vs host read/sync_host, per function."""
import os,re,json
SRC="/home/user/lammps/src/KOKKOS"
OUT=os.path.dirname(os.path.abspath(__file__))
def strip(t):
    t=re.sub(r'/\*.*?\*/', lambda m:'\n'*m.group(0).count('\n'), t, flags=re.S)
    return re.sub(r'//[^\n]*','',t)
res={}
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('.cpp'): continue
    p=os.path.join(SRC,fn)
    L=strip(open(p,errors='replace').read()).split('\n')
    names=set()
    for l in L:
        for m in re.finditer(r'(?<![>\w])\b(k_[A-Za-z_0-9]+)\s*\.',l):
            if 'atomKK->' in l[:m.start()] or '->k_' in l[:m.start()+2]: continue
            names.add(m.group(1))
    out={}
    for n in sorted(names):
        e={'md':[],'sh':[],'mh':[],'sd':[],'hostuse':[]}
        for i,l in enumerate(L,1):
            if re.search(n+r'\s*\.\s*(template\s+)?(modify\s*<\s*(DeviceType|LMPDeviceType)\s*>|modify_device)',l): e['md'].append(i)
            if re.search(n+r'\s*\.\s*(template\s+)?(sync\s*<\s*LMPHostType\s*>|sync_host)',l): e['sh'].append(i)
            if re.search(n+r'\s*\.\s*(template\s+)?(modify\s*<\s*LMPHostType\s*>|modify_host)',l): e['mh'].append(i)
            if re.search(n+r'\s*\.\s*(template\s+)?(sync\s*<\s*(DeviceType|LMPDeviceType)\s*>|sync_device)',l): e['sd'].append(i)
            if re.search(n+r'\s*\.\s*(h_view|view_host)',l): e['hostuse'].append(i)
        if e['md'] and e['hostuse'] and not e['sh']:
            out[n]=e
    if out: res[fn]=out
json.dump(res,open(os.path.join(OUT,'scan11.json'),'w'),indent=1)
for fn,d in sorted(res.items()):
    for n,e in sorted(d.items()):
        print(fn,n,{k:v[:5] for k,v in e.items() if v})
