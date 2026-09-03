import os,re,json,sys,collections,glob
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_parse import *
OUT=os.path.dirname(os.path.abspath(__file__))
classes=build_index(); dtors=build_dtors()
# inline dtors in headers
for name,ents in classes.items():
    for (nm,bases,path,line,body,bl) in ents:
        m=re.search(r'~'+re.escape(nm)+r'\s*\([^)]*\)\s*(?:override\s*)?\{',body)
        if m:
            idx=body.find('{',m.start()); depth=0;k=idx
            while k<len(body):
                if body[k]=='{':depth+=1
                elif body[k]=='}':
                    depth-=1
                    if depth==0:break
                k+=1
            dtors.setdefault(nm,(body[idx+1:k],path,bl+body[:idx].count('\n')))

copyfiles=[o['file'] for o in json.load(open(OUT+'/rule1c.json'))]
# classes declared in the header matching each copying file
atrisk=set()
for f in copyfiles:
    stem=f.rsplit('.',1)[0]
    for cand in (SRC+'/KOKKOS/'+stem+'.h', SRC+'/KOKKOS/'+f):
        if os.path.exists(cand):
            for t in parse_classes(cand): atrisk.add(t[0])
    # *_impl.h -> base header
    if stem.endswith('_impl'):
        h=SRC+'/KOKKOS/'+stem[:-5]+'.h'
        if os.path.exists(h):
            for t in parse_classes(h): atrisk.add(t[0])
def resolve(n):
    e=classes.get(n); return e[0] if e else None
def chain(n):
    seen={n};out=[];st=list(resolve(n)[1]) if resolve(n) else []
    while st:
        x=st.pop(0)
        if x in seen: continue
        seen.add(x); e=resolve(x); out.append(x)
        if e: st.extend(e[1])
    return out
# add all kokkos subclasses of at-risk classes (they inherit the copy behaviour)
kkall=set()
for h in glob.glob(SRC+'/KOKKOS/*.h'):
    for t in parse_classes(h): kkall.add(t[0])
grow=True
while grow:
    grow=False
    for c in list(kkall):
        if c in atrisk: continue
        if set(chain(c)) & atrisk: atrisk.add(c); grow=True

FREE=re.compile(r'destroy_kokkos|memory->destroy|memory->sfree|delete\s*\[\]|\bdelete\s+\w|->destroy\(|\bfree\s*\(|sfree|deallocate')
users=collections.defaultdict(set)
for c in atrisk:
    for a in [c]+chain(c): users[a].add(c)
res=[]
for cn,us in sorted(users.items()):
    d=dtors.get(cn)
    if not d: continue
    body,dpath,dline=d
    b2=strip_comments(body); st=b2.strip()
    if not st: continue
    if re.match(r'if\s*\(\s*(this->)?copymode\s*\)\s*return\s*;', st): continue
    if not FREE.search(b2): continue
    guarded_all = bool(re.match(r'if\s*\(\s*(allocated\s*&&\s*)?!\s*(this->)?copymode\s*\)', st)) and st.rstrip().endswith('}')
    res.append({'class':cn,'file':dpath.replace(SRC+'/','src/'),'line':dline,
                'mentions':'copymode' in b2,'wrapped_guard':guarded_all,
                'nusers':len(us),'users':sorted(us)[:6],'body':st[:1500]})
json.dump(res,open(OUT+'/rule1d_self.json','w'),indent=1)
print('at-risk kokkos classes:',len(atrisk))
print('unguarded freeing dtors in their chains:',len(res))
for r in res:
    print('%-30s %-52s %5d mentions=%-5s wrapped=%-5s n=%d %s'%(r['class'],r['file'],r['line'],r['mentions'],r['wrapped_guard'],r['nusers'],r['users'][:3]))
