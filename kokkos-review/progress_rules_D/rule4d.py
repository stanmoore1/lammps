import os,re,json,collections
SRC='/home/user/lammps/src'
MAC=['PairStyle','FixStyle','ComputeStyle','BondStyle','AngleStyle','DihedralStyle','ImproperStyle',
     'KSpaceStyle','RegionStyle','MinimizeStyle','AtomStyle','NPairStyle','NBinStyle','NStencilStyle',
     'IntegrateStyle','CommandStyle','DumpStyle','ReaderStyle','BodyStyle']
rx=re.compile(r'^\s*('+'|'.join(MAC)+r')\(([^,]+),')
allst=collections.defaultdict(list)
for d,dn,fn in os.walk(SRC):
    dn[:]=[x for x in dn if x!='STUBS' and not x.startswith('Obj')]
    rel=os.path.relpath(d,SRC)
    for f in fn:
        if not f.endswith('.h'): continue
        for line in open(os.path.join(d,f),errors='replace'):
            m=rx.match(line)
            if m: allst[(m.group(1),m.group(2).strip())].append(rel+'/'+f)
bad=[]
for (mac,name),files in sorted(allst.items()):
    if not any(f.startswith('KOKKOS/') for f in files): continue
    parts=name.split('/')
    if 'kk' not in parts: continue
    i=parts.index('kk')
    base='/'.join(parts[:i])
    if not base: continue
    cand=allst.get((mac,base),[])
    cpu=[f for f in cand if not f.startswith('KOKKOS/')]
    if not cpu:
        anyn=[ (m,f) for (m,n),fl in allst.items() if n==base for f in fl]
        bad.append({'macro':mac,'kk_style':name,'kk_headers':files,'base':base,
                    'base_found_anywhere':anyn})
json.dump(bad,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_D/rule4d.json','w'),indent=1)
print('kk-suffixed style keys:',sum(1 for (m,n) in allst if 'kk' in n.split('/')))
print('base missing:',len(bad))
for b in bad: print(b['macro'],b['kk_style'],'->base',b['base'],'| hdr',b['kk_headers'],'| anywhere',b['base_found_anywhere'])
