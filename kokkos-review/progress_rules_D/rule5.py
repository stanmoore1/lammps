import os,re,glob,collections,json
SRC='/home/user/lammps/src'; DOC='/home/user/lammps/doc/src'
MACS={'PairStyle':'pair','FixStyle':'fix','ComputeStyle':'compute','BondStyle':'bond',
      'AngleStyle':'angle','DihedralStyle':'dihedral','ImproperStyle':'improper',
      'KSpaceStyle':'kspace','RegionStyle':'region','MinimizeStyle':'min','AtomStyle':'atom',
      'IntegrateStyle':'integrate','CommandStyle':'command','DumpStyle':'dump'}
rx=re.compile(r'^\s*('+'|'.join(MACS)+r')\(([^,]+),')
kk=collections.defaultdict(set)   # kind -> set of base style names having /kk
for d,dn,fn in os.walk(SRC):
    dn[:]=[x for x in dn if x!='STUBS' and not x.startswith('Obj')]
    for f in fn:
        if not f.endswith('.h'): continue
        for line in open(os.path.join(d,f),errors='replace'):
            m=rx.match(line)
            if not m: continue
            kind=MACS[m.group(1)]; name=m.group(2).strip()
            p=name.split('/')
            if 'kk' in p:
                i=p.index('kk'); base='/'.join(p[:i])
                if base and not re.search('[A-Z]',base): kk[kind].add(base)
# build style -> doc page map from Commands_*.rst and *_style.rst tables
docmap=collections.defaultdict(dict)
tables={'pair':['Commands_pair.rst'],'fix':['Commands_fix.rst'],'compute':['Commands_compute.rst'],
        'bond':['Commands_bond.rst'],'angle':['Commands_bond.rst'],'dihedral':['Commands_bond.rst'],
        'improper':['Commands_bond.rst'],'kspace':['Commands_kspace.rst'],
        'atom':['Commands_all.rst'],'region':['Commands_all.rst'],'min':['Commands_all.rst'],
        'integrate':['Commands_all.rst'],'command':['Commands_all.rst'],'dump':['Commands_all.rst']}
linkrx=re.compile(r':doc:`([^`<]+?)\s*<([^`>]+)>`')
allrst=glob.glob(DOC+'/*.rst')
for kind in kk:
    for t in tables.get(kind,[]):
        p=os.path.join(DOC,t)
        if not os.path.exists(p): continue
        for m in linkrx.finditer(open(p).read()):
            nm=m.group(1).strip().strip('*').split(' ')[0]
            docmap[kind][nm]=m.group(2)
# fallback: search all Commands_*.rst
for p in glob.glob(DOC+'/Commands_*.rst'):
    for m in linkrx.finditer(open(p).read()):
        nm=m.group(1).strip().strip('*').split(' ')[0]
        for kind in kk: docmap[kind].setdefault(nm,m.group(2))
res={'no_doc_page':[], 'no_accel_line':[], 'kk_not_listed':[], 'no_accel_include':[]}
for kind,names in sorted(kk.items()):
    for n in sorted(names):
        page=docmap[kind].get(n)
        if not page: res['no_doc_page'].append((kind,n)); continue
        f=os.path.join(DOC,page+'.rst')
        if not os.path.exists(f): res['no_doc_page'].append((kind,n,page)); continue
        txt=open(f).read()
        accl=[l for l in txt.splitlines() if l.startswith('Accelerator Variants:')]
        if not accl: res['no_accel_line'].append((kind,n,page)); continue
        joined=' '.join(accl)
        if not re.search(r'\*'+re.escape(n)+r'/kk\*',joined):
            res['kk_not_listed'].append((kind,n,page,joined[:200]))
        if '.. include:: accel_styles.rst' not in txt:
            res['no_accel_include'].append((kind,n,page))
json.dump(res,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_D/rule5.json','w'),indent=1)
print('total kk base styles:',sum(len(v) for v in kk.values()))
for k,v in res.items(): print(k,len(v))
