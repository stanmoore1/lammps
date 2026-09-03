#!/usr/bin/env python3
"""Which functions modify_host / sync<Device> each own DualView."""
import os, re, json
SRC="/home/user/lammps/src/KOKKOS"
OUT="/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_C"
import importlib.util
spec=importlib.util.spec_from_file_location("s4", os.path.join(OUT,"scan4.py"))
# reuse the splitter by copy
def strip_comments(t):
    t=re.sub(r'/\*.*?\*/', lambda m:'\n'*m.group(0).count('\n'), t, flags=re.S)
    return re.sub(r'//[^\n]*','',t)
SIGRE = re.compile(r'([A-Za-z_][A-Za-z_0-9]*)\s*(?:<[^<>]*>)?\s*::\s*(~?[A-Za-z_][A-Za-z_0-9]*)\s*\(')
def split_functions(path):
    raw=open(path,errors='replace').read(); clean=strip_comments(raw).split('\n'); n=len(clean)
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

out={}
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('_kokkos.cpp'): continue
    funcs=split_functions(os.path.join(SRC,fn))
    names=set()
    for f in funcs:
        for ln,l in f['lines']:
            for m in re.finditer(r'(?<!>)\b(k_[A-Za-z_0-9]+)\s*\.\s*(?:template\s+)?(modify|sync)', l):
                if 'atomKK->' in l[:m.start()]: continue
                names.add(m.group(1))
    d={}
    for nm in names:
        mh=[]; sd=[]; md=[]; sh=[]
        for f in funcs:
            for ln,l in f['lines']:
                if re.search(r'(?<!>)\b'+nm+r'\s*\.\s*(template\s+)?(modify<\s*LMPHostType|modify_host)', l): mh.append((f['name'],ln))
                if re.search(r'(?<!>)\b'+nm+r'\s*\.\s*(template\s+)?(sync<\s*DeviceType|sync<\s*LMPDeviceType|sync_device)', l): sd.append((f['name'],ln))
                if re.search(r'(?<!>)\b'+nm+r'\s*\.\s*(template\s+)?(modify<\s*DeviceType|modify<\s*LMPDeviceType|modify_device)', l): md.append((f['name'],ln))
                if re.search(r'(?<!>)\b'+nm+r'\s*\.\s*(template\s+)?(sync<\s*LMPHostType|sync_host)', l): sh.append((f['name'],ln))
        d[nm]={'modify_host':mh,'sync_dev':sd,'modify_dev':md,'sync_host':sh}
    out[fn]=d
json.dump(out, open(os.path.join(OUT,'scan8.json'),'w'), indent=1)

LATE={'init_one','coeff','read_restart','settings','init_style','setup','init','allocate','read_restart_settings','modify_param','post_constructor'}
print("--- modify_host in init_one/coeff/read_restart but sync_dev only in earlier-running fn ---")
for fn,d in sorted(out.items()):
    for nm,v in sorted(d.items()):
        mhf={x[0].split('::')[-1] for x in v['modify_host']}
        sdf={x[0].split('::')[-1] for x in v['sync_dev']}
        if not v['modify_host']: continue
        if not v['sync_dev']:
            print("NO_SYNC_DEV", fn, nm, sorted(mhf))
        elif mhf & {'init_one','coeff','read_restart'} and sdf and sdf <= {'init_style','settings','allocate','init'}:
            print("SYNC_BEFORE_MODIFY?", fn, nm, "modify in", sorted(mhf), "sync in", sorted(sdf))
