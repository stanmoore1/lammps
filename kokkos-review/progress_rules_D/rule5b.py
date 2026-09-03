import os,re,glob,json,collections
SRC='/home/user/lammps/src'; DOC='/home/user/lammps/doc/src'
# index type used in ".. index:: <type> <style>"
MACS={'PairStyle':'pair_style','FixStyle':'fix','ComputeStyle':'compute','BondStyle':'bond_style',
      'AngleStyle':'angle_style','DihedralStyle':'dihedral_style','ImproperStyle':'improper_style',
      'KSpaceStyle':'kspace_style','DumpStyle':'dump'}
OTHER={'RegionStyle':'region','MinimizeStyle':'min','AtomStyle':'atom','IntegrateStyle':'integrate',
       'CommandStyle':'command'}
rx=re.compile(r'^\s*(\w+)Style\(([^,]+),')
kk=collections.defaultdict(set)   # itype -> base names
other=collections.defaultdict(set)
for d,dn,fn in os.walk(SRC):
    dn[:]=[x for x in dn if x!='STUBS' and not x.startswith('Obj')]
    for f in fn:
        if not f.endswith('.h'): continue
        for line in open(os.path.join(d,f),errors='replace'):
            m=rx.match(line)
            if not m: continue
            mac=m.group(1)+'Style'; name=m.group(2).strip()
            p=name.split('/')
            if 'kk' not in p: continue
            base='/'.join(p[:p.index('kk')])
            if not base or re.search('[A-Z]',base): continue
            if mac in MACS: kk[MACS[mac]].add(base)
            elif mac in OTHER: other[OTHER[mac]].add(base)
# map ".. index:: type style" -> file
idx=collections.defaultdict(set)
ip=re.compile(r'^\.\. index:: (compute|fix|pair_style|angle_style|bond_style|dihedral_style|improper_style|kspace_style|dump)\s+([a-zA-Z0-9/_]+)\s*$')
files=sorted(glob.glob(DOC+'/*.rst'))
txtcache={}
for f in files:
    t=open(f,errors='replace').read(); txtcache[f]=t
    for line in t.splitlines():
        m=ip.match(line)
        if m: idx[(m.group(1),m.group(2))].add(f)
res={'no_index_page':[], 'no_accel_variants_line':[], 'kk_not_in_variants':[], 'no_accel_include':[]}
for itype,names in sorted(kk.items()):
    for n in sorted(names):
        pages=idx.get((itype,n),set())
        if not pages:
            res['no_index_page'].append([itype,n]); continue
        for p in sorted(pages):
            t=txtcache[p]; rel=os.path.relpath(p,'/home/user/lammps')
            accl=[l for l in t.splitlines() if l.startswith('Accelerator Variant')]
            if not accl:
                res['no_accel_variants_line'].append([itype,n,rel]); continue
            if not re.search(r'\*'+re.escape(n)+r'/kk\*',' '.join(accl)):
                res['kk_not_in_variants'].append([itype,n,rel,' | '.join(accl)[:250]])
            if '.. include:: accel_styles.rst' not in t:
                res['no_accel_include'].append([itype,n,rel])
json.dump(res,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_D/rule5b.json','w'),indent=1)
print('kk styles by index type:',{k:len(v) for k,v in kk.items()})
print('non-index kinds (region/min/atom/integrate/command):',{k:sorted(v) for k,v in other.items()})
for k,v in res.items():
    print('---',k,len(v))
    for x in v: print('   ',x)
