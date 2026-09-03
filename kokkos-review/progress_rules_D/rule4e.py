import os,re,json
K='/home/user/lammps/src/KOKKOS'
MAC=['PairStyle','FixStyle','ComputeStyle','BondStyle','AngleStyle','DihedralStyle','ImproperStyle',
     'KSpaceStyle','RegionStyle','MinimizeStyle','AtomStyle','NPairStyle','NBinStyle','NStencilStyle',
     'IntegrateStyle','CommandStyle','DumpStyle','ReaderStyle','BodyStyle']
rx=re.compile(r'('+'|'.join(MAC)+r')\((.*?)\);',re.S)
bad=[]
for f in sorted(os.listdir(K)):
    if not f.endswith('.h'): continue
    txt=open(os.path.join(K,f),errors='replace').read()
    aliases=dict(re.findall(r'using\s+(\w+)\s*=\s*([^;]+);',txt))
    for m in rx.finditer(txt):
        mac=m.group(1); body=m.group(2)
        ln=txt[:m.start()].count('\n')+1
        parts=[p.strip() for p in body.split(',')]
        name=parts[0]; cls=parts[1] if len(parts)>1 else ''
        seg=name.split('/')
        if 'kk' not in seg: continue
        want=None
        tail=seg[seg.index('kk')+1:]
        if tail[:1]==['device']: want='LMPDeviceType'
        elif tail[:1]==['host']: want='LMPHostType'
        elif not tail: want='LMPDeviceType'   # bare /kk -> device
        if want is None: continue
        # resolve class: either templated inline, or an alias
        resolved=cls
        base=cls.split('<')[0]
        if '<' not in cls and base in aliases: resolved=aliases[base]
        got=set(re.findall(r'LMP(?:Device|Host)Type',resolved))
        if not got:
            bad.append((f,ln,name,cls,resolved,'no LMP*Type found'))
        elif want not in got:
            bad.append((f,ln,name,cls,resolved,'expected '+want+' got '+','.join(sorted(got))))
        elif len(got)>1:
            bad.append((f,ln,name,cls,resolved,'both types present'))
json.dump(bad,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_D/rule4e.json','w'),indent=1)
print('mismatches',len(bad))
for b in bad: print('%s:%d  %-45s cls=%-40s -> %-50s  %s'%b)
