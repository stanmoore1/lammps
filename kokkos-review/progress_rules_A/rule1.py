import os,re,json,sys,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_parse import *

OUT=os.path.dirname(os.path.abspath(__file__))
classes = build_index()
dtors   = build_dtors()

# also in-class inline destructors defined in headers
inline_dtors={}
for name, ents in classes.items():
    for (nm,bases,path,line,body,bodyline) in ents:
        m = re.search(r'~'+re.escape(nm)+r'\s*\([^)]*\)\s*(?:override\s*)?\{', body)
        if m:
            idx = body.find('{', m.start())
            depth=0;k=idx
            while k < len(body):
                if body[k]=='{': depth+=1
                elif body[k]=='}':
                    depth-=1
                    if depth==0: break
                k+=1
            inline_dtors.setdefault(nm,(body[idx+1:k],path,bodyline+body[:idx].count('\n')))
for k,v in inline_dtors.items():
    dtors.setdefault(k,v)

def resolve(name, prefer_non_kokkos=False):
    ents = classes.get(name)
    if not ents: return None
    if prefer_non_kokkos:
        for e in ents:
            if '/KOKKOS/' not in e[2]: return e
    return ents[0]

def chain(name):
    """BFS over base classes; returns list of (classname, entry) excluding start"""
    seen={name}; out=[]; stack=list(resolve(name)[1]) if resolve(name) else []
    while stack:
        n = stack.pop(0)
        if n in seen: continue
        seen.add(n)
        e = resolve(n)
        if not e: 
            out.append((n,None)); continue
        out.append((n,e))
        stack.extend(e[1])
    return out

FREE_PAT = re.compile(r'destroy_kokkos|memory->destroy|memory->sfree|delete\s*\[\]|\bdelete\s+\w|->destroy\(|\bfree\s*\(|sfree|deallocate')
kk_classes=[]
for h in sorted(__import__('glob').glob(SRC+"/KOKKOS/*.h")):
    for tup in parse_classes(h):
        kk_classes.append(tup)

results=[]
seen_pairs=set()
for (name,bases,path,line,body,bl) in kk_classes:
    ch = chain(name)
    for cn, e in ch:
        key=(name.split('Kokkos')[0], cn)
        if (cn,) in seen_pairs: continue
        d = dtors.get(cn)
        if not d: continue
        bodyd, dpath, dline = d
        b2 = strip_comments(bodyd)
        st = b2.strip()
        if not st: continue
        first_guard = bool(re.match(r'if\s*\(\s*copymode\s*\)\s*return\s*;', st))
        mentions = 'copymode' in b2
        frees = bool(FREE_PAT.search(b2))
        if first_guard: continue
        if not frees: continue
        seen_pairs.add((cn,))
        results.append({'class':cn,'file':dpath.replace(SRC+'/','src/'),'line':dline,
                        'mentions_copymode':mentions,'is_kokkos':'/KOKKOS/' in dpath,
                        'body':st[:2000]})
# record which kokkos classes lead to each
users=collections.defaultdict(set)
for (name,bases,path,line,body,bl) in kk_classes:
    for cn,e in chain(name): users[cn].add(name)
for r in results: r['kk_users']=sorted(users[r['class']])[:8]; r['n_kk_users']=len(users[r['class']])
json.dump(results, open(OUT+'/rule1_raw.json','w'), indent=1)
print("candidates:",len(results))
for r in results:
    print('%-30s %-55s %5d mentions=%-5s kk=%-5s users=%d' % (r['class'],r['file'],r['line'],r['mentions_copymode'],r['is_kokkos'],r['n_kk_users']))
