import re,os,glob
base='/home/user/lammps/src/KOKKOS'
pat=re.compile(r'^\s*(\w+Style)\((\S+?),\s*(.+?)\);')
out=[]
for f in sorted(glob.glob(base+'/*.h')):
    lines=open(f).read().split('\n')
    for i,l in enumerate(lines,1):
        m=pat.match(l)
        if not m: continue
        macro,name,cls=m.groups()
        if '/kk' not in name and 'kk' not in name: pass
        out.append((os.path.basename(f),i,macro,name,cls))
# group by file+class base
from collections import defaultdict
byfile=defaultdict(list)
for e in out: byfile[e[0]].append(e)
for f,es in byfile.items():
    for (fn,i,macro,name,cls) in es:
        bad=None
        if name.endswith('/kk/host') and 'LMPHostType' not in cls: bad='host variant not LMPHostType'
        if name.endswith('/kk/device') and 'LMPDeviceType' not in cls: bad='device variant not LMPDeviceType'
        if name.endswith('/kk') and 'LMPDeviceType' not in cls: bad='plain /kk not LMPDeviceType'
        if bad: print('MISMATCH',fn,i,name,cls,bad)
    # check class name consistency
    clsnames=set(re.sub(r'<.*','',e[4]) for e in es)
    if len(clsnames)>1:
        print('MULTICLASS',f,[ (e[3],e[4]) for e in es])
    # check missing variants
    names=[e[3] for e in es]
    hosts=[n for n in names if n.endswith('/kk/host')]
    devs=[n for n in names if n.endswith('/kk/device')]
    plains=[n for n in names if n.endswith('/kk')]
    if len(plains)!=len(hosts) or len(plains)!=len(devs):
        print('VARIANTCOUNT',f,names)
