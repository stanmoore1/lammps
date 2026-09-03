import os,re,json,sys,collections,glob
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_parse import *
OUT=os.path.dirname(os.path.abspath(__file__))
classes=build_index()
raw=json.load(open(OUT+'/rule1_raw.json'))

def resolve(name):
    e=classes.get(name)
    return e[0] if e else None
def chain(name):
    seen={name};out=[];st=list(resolve(name)[1]) if resolve(name) else []
    while st:
        n=st.pop(0)
        if n in seen: continue
        seen.add(n)
        e=resolve(n)
        out.append(n)
        if e: st.extend(e[1])
    return out

# kokkos classes and whether their translation unit (or any kokkos ancestor's) sets copymode=1
kk={}
for h in sorted(glob.glob(SRC+"/KOKKOS/*.h")):
    cpp=h[:-2]+'.cpp'
    txt=open(h,encoding='utf-8',errors='replace').read()
    if os.path.exists(cpp): txt+=open(cpp,encoding='utf-8',errors='replace').read()
    setscm=bool(re.search(r'copymode\s*=\s*1',txt))
    for (nm,bases,path,line,body,bl) in parse_classes(h):
        kk[nm]={'h':h,'sets':setscm,'bases':bases}

def kk_sets_copymode(nm):
    if kk.get(nm,{}).get('sets'): return True
    for a in chain(nm):
        if a in kk and kk[a]['sets']: return True
    return False

users=collections.defaultdict(set)
for nm in kk:
    for a in chain(nm): users[a].add(nm)

STYLEROOT={'Pair','Fix','Compute','Bond','Angle','Dihedral','Improper','KSpace','Region','NPair','NBin','NStencil','NeighList','Command','Min'}
out=[]
for r in raw:
    cn=r['class']
    us=sorted(users[cn]) if cn in users else []
    us_copy=[u for u in us if kk_sets_copymode(u)]
    roots=set()
    for u in us: roots |= (set(chain(u))&STYLEROOT)
    if cn in kk: 
        roots|= (set(chain(cn))&STYLEROOT); us_copy = us_copy or ([cn] if kk_sets_copymode(cn) else [])
    r['users_all']=us; r['users_copymode']=us_copy; r['style_roots']=sorted(roots)
    out.append(r)
json.dump(out,open(OUT+'/rule1_filtered.json','w'),indent=1)
print("=== HOT (users set copymode=1, style-derived) ===")
for r in out:
    if r['users_copymode']:
        print('%-28s %-52s %5d mentions=%-5s roots=%s users=%s' % (r['class'],r['file'],r['line'],r['mentions_copymode'],','.join(r['style_roots']),','.join(r['users_copymode'][:5])))
print()
print("=== COLD (no kk user sets copymode=1) ===")
for r in out:
    if not r['users_copymode']:
        print('%-28s %-52s %5d roots=%s users=%s' % (r['class'],r['file'],r['line'],','.join(r['style_roots']),','.join(r['users_all'][:5])))
