#!/usr/bin/env python3
"""Error/flag DualView discipline: device write -> host read needs modify_device + sync_host."""
import os,re,json
SRC="/home/user/lammps/src/KOKKOS"
OUT=os.path.dirname(os.path.abspath(__file__))
def strip(t):
    t=re.sub(r'/\*.*?\*/', lambda m:'\n'*m.group(0).count('\n'), t, flags=re.S)
    return re.sub(r'//[^\n]*','',t)
rep={}
pat=re.compile(r'\b(k_[A-Za-z_0-9]*(?:flag|Flag|resize|scalars|error)[A-Za-z_0-9]*)\b')
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('.cpp'): continue
    p=os.path.join(SRC,fn)
    L=strip(open(p,errors='replace').read()).split('\n')
    names=set()
    for l in L:
        for m in pat.finditer(l):
            if 'atomKK->' in l[:m.start()]: continue
            names.add(m.group(1))
    d={}
    for n in names:
        e={'md':[],'sh':[],'mh':[],'sd':[],'hostread':[],'hostwrite':[],'devview':[]}
        for i,l in enumerate(L,1):
            if re.search(n+r'\s*\.\s*(template\s+)?(modify<\s*(DeviceType|LMPDeviceType)|modify_device)',l): e['md'].append(i)
            if re.search(n+r'\s*\.\s*(template\s+)?(sync<\s*LMPHostType|sync_host)',l): e['sh'].append(i)
            if re.search(n+r'\s*\.\s*(template\s+)?(modify<\s*LMPHostType|modify_host)',l): e['mh'].append(i)
            if re.search(n+r'\s*\.\s*(template\s+)?(sync<\s*(DeviceType|LMPDeviceType)|sync_device)',l): e['sd'].append(i)
            if re.search(n+r'\s*\.\s*(h_view|view_host)',l) and not re.search(n+r'\s*\.\s*(h_view|view_host)\s*\(?\)?\s*[\[\(][^;]*\)\s*=[^=]',l): e['hostread'].append(i)
            if re.search(n+r'\s*\.\s*(template\s+)?view\s*<\s*(DeviceType|LMPDeviceType)',l): e['devview'].append(i)
        d[n]=e
    if d: rep[fn]=d
json.dump(rep,open(os.path.join(OUT,'scan9.json'),'w'),indent=1)
for fn,d in sorted(rep.items()):
    for n,e in sorted(d.items()):
        prob=[]
        if e['hostread'] and not e['sh']: prob.append('HOSTREAD_NO_SYNCHOST')
        if e['devview'] and not e['md'] and e['hostread']: prob.append('DEVVIEW_NO_MODIFYDEV')
        if e['md'] and not e['sh']: prob.append('MODIFYDEV_NO_SYNCHOST')
        if prob: print(fn,n,prob,{k:v[:4] for k,v in e.items() if v})
