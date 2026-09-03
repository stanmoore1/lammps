import os,re,json,collections
SRC='/home/user/lammps/src'; K=os.path.join(SRC,'KOKKOS')
MAC=['PairStyle','FixStyle','ComputeStyle','BondStyle','AngleStyle','DihedralStyle','ImproperStyle',
     'KSpaceStyle','RegionStyle','MinimizeStyle','AtomStyle','NPairStyle','NBinStyle','NStencilStyle',
     'IntegrateStyle','CommandStyle','DumpStyle','ReaderStyle','BodyStyle']
rx=re.compile(r'^\s*('+'|'.join(MAC)+r')\(([^,]+),')
# harvest ALL style registrations in the whole src tree
allstyles=collections.defaultdict(list)   # (macro,name) -> [relpath]
for d,dn,fn in os.walk(SRC):
    dn[:]=[x for x in dn if x!='STUBS' and not x.startswith('Obj')]
    rel=os.path.relpath(d,SRC)
    for f in fn:
        if not f.endswith('.h'): continue
        try: txt=open(os.path.join(d,f),errors='replace').read()
        except: continue
        for line in txt.splitlines():
            m=rx.match(line)
            if m: allstyles[(m.group(1),m.group(2).strip())].append(rel+'/'+f)
kk={k:v for k,v in allstyles.items() if k[1].endswith('/kk')}
noKKbase=[]; dups=[]; kkdups=[]
for (mac,name),files in sorted(kk.items()):
    base=name[:-3]
    kkfiles=[f for f in files if f.startswith('KOKKOS/')]
    if len(files)>1: kkdups.append((mac,name,files))
    # base must exist as CPU style with same macro somewhere
    cand=allstyles.get((mac,base))
    if not cand:
        # try any macro
        anym=[m for (m,n) in allstyles if n==base]
        noKKbase.append((mac,name,files,anym))
    else:
        cpu=[f for f in cand if not f.startswith('KOKKOS/')]
        if not cpu: noKKbase.append((mac,name,files,'only-in-KOKKOS:'+str(cand)))
# duplicates among all styles (same macro+name in 2+ headers)
for (mac,name),files in sorted(allstyles.items()):
    if len(files)>1 and any(f.startswith('KOKKOS/') for f in files): dups.append((mac,name,files))
out={'n_kk_styles':len(kk),'kk_base_missing':noKKbase,'kk_dup_registration':kkdups,'dup_involving_kokkos':dups}
json.dump(out,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_D/rule4.json','w'),indent=1)
print('kk styles',len(kk))
print('--- kk_base_missing ---')
for x in out['kk_base_missing']: print(x)
print('--- kk dup registration ---')
for x in out['kk_dup_registration']: print(x)
print('--- dup involving kokkos ---')
for x in out['dup_involving_kokkos']: print(x)
