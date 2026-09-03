#!/usr/bin/env python3
"""Own-DualView (k_*) sync/modify discipline scan."""
import os, re, json
SRC="/home/user/lammps/src/KOKKOS"
OUT="/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_C"
def strip_comments(t):
    t=re.sub(r'/\*.*?\*/', lambda m:'\n'*m.group(0).count('\n'), t, flags=re.S)
    return re.sub(r'//[^\n]*','',t)

rep={}
for fn in sorted(os.listdir(SRC)):
    if not (fn.endswith('_kokkos.cpp') or fn.endswith('_kokkos.h')): continue
    if not fn.endswith('.cpp'): continue
    files=[os.path.join(SRC,fn)]
    h=os.path.join(SRC,fn[:-4]+'.h')
    if os.path.exists(h): files.append(h)
    lines=[]
    for p in files:
        for i,l in enumerate(strip_comments(open(p,errors='replace').read()).split('\n')):
            lines.append((os.path.basename(p),i+1,l))
    # dual view members declared in header (k_*) and NOT from atomKK
    names=set()
    for f,i,l in lines:
        for m in re.finditer(r'\b(k_[A-Za-z_0-9]+)\b', l):
            if 'atomKK->' in l[:m.start()][-12:]: continue
            names.add(m.group(1))
    # exclude the ones only ever seen as atomKK->k_x
    names = {n for n in names if any(re.search(r'(?<!>)\b'+n+r'\b', l) and 'atomKK->'+n not in l for f,i,l in lines)}
    info={}
    for n in sorted(names):
        d={'host_write':[], 'modify_host':[], 'sync_dev':[], 'modify_dev':[], 'sync_host':[],
           'dev_view':[], 'host_view':[], 'modify_any':[], 'sync_any':[]}
        # host mirror alias names, e.g. h_foo = k_foo.view_host()
        aliases=set()
        for f,i,l in lines:
            m=re.search(r'\b([A-Za-z_][A-Za-z_0-9]*)\s*=\s*'+n+r'\s*\.\s*(?:template\s+)?(view_host|h_view|view<LMPHostType>)', l)
            if m: aliases.add(m.group(1))
            m=re.search(r'\b([A-Za-z_][A-Za-z_0-9]*)\s*=\s*'+n+r'\s*\.\s*(?:template\s+)?view\s*<\s*DeviceType', l)
            if m: d['dev_view'].append((f,i))
        for f,i,l in lines:
            if re.search(n+r'\s*\.\s*(h_view|view_host)\s*\([^)]*\)\s*(\[[^\]]*\]\s*)?(=[^=]|\+=|-=)', l): d['host_write'].append((f,i,l.strip()[:110]))
            for a in aliases:
                if re.search(r'\b'+re.escape(a)+r'\s*[\(\[][^;]*?[\)\]]\s*(=[^=]|\+=|-=)', l): d['host_write'].append((f,i,l.strip()[:110]))
            if re.search(n+r'\s*\.\s*(template\s+)?modify\s*<\s*LMPHostType|'+n+r'\s*\.\s*modify_host', l): d['modify_host'].append((f,i))
            if re.search(n+r'\s*\.\s*(template\s+)?modify\s*<\s*DeviceType|'+n+r'\s*\.\s*modify_device|'+n+r'\s*\.\s*(template\s+)?modify\s*<\s*LMPDeviceType', l): d['modify_dev'].append((f,i))
            if re.search(n+r'\s*\.\s*(template\s+)?sync\s*<\s*DeviceType|'+n+r'\s*\.\s*sync_device|'+n+r'\s*\.\s*(template\s+)?sync\s*<\s*LMPDeviceType', l): d['sync_dev'].append((f,i))
            if re.search(n+r'\s*\.\s*(template\s+)?sync\s*<\s*LMPHostType|'+n+r'\s*\.\s*sync_host', l): d['sync_host'].append((f,i))
            if re.search(n+r'\s*\.\s*(template\s+)?modify', l): d['modify_any'].append((f,i))
            if re.search(n+r'\s*\.\s*(template\s+)?sync', l): d['sync_any'].append((f,i))
        if d['host_write'] or d['modify_host'] or d['modify_dev']:
            info[n]=d
    flags=[]
    for n,d in info.items():
        if d['host_write'] and not d['modify_host'] and not d['modify_any']:
            flags.append(('HOSTWRITE_NO_MODIFY', n, d['host_write'][:3]))
        if d['modify_host'] and not d['sync_dev'] and not d['sync_any']:
            flags.append(('MODIFYHOST_NO_SYNCDEV', n, d['modify_host'][:3]))
        if d['modify_dev'] and not d['sync_host'] and not d['sync_any']:
            flags.append(('MODIFYDEV_NO_SYNCHOST', n, d['modify_dev'][:3]))
    if flags: rep[fn]=flags
json.dump(rep, open(os.path.join(OUT,'scan7.json'),'w'), indent=1)
for f,fl in rep.items():
    for x in fl: print(f, x[0], x[1], x[2])
