import os, re, sys, json, glob, subprocess
sys.path.insert(0,'/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review')
exec(open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/audit.py').read().split('# ---- 3.')[0])

SRCK = SRC+"/KOKKOS"
# classes defined in KOKKOS .cpp that set copymode = 1
copyself = set()
for cpp in glob.glob(SRCK+"/*.cpp"):
    txt = open(cpp,encoding='utf-8',errors='replace').read()
    if not re.search(r'copymode\s*=\s*1', txt): continue
    h = cpp[:-4]+'.h'
    if not os.path.exists(h): continue
    for name,bases,path,line in parse_classes(h):
        copyself.add(name)

def resolve(name):
    ents = classes.get(name)
    if not ents: return None
    return ents[0]

def chain(name):
    seen=set(); out=[]; stack=[name]
    while stack:
        n = stack.pop(0)
        if n in seen: continue
        seen.add(n)
        e = resolve(n)
        if not e: continue
        out.append((n,e))
        for b in e[0]: stack.append(b)
    return out

allchain = {}
for c in copyself:
    for n,e in chain(c):
        allchain.setdefault(n,set()).add(c)

FREE_PAT = re.compile(r'destroy_kokkos|memory->destroy|memory->sfree|delete\s*\[\]|\bdelete\s|->destroy|\bfree\(|sfree')
res=[]
for cn, users in sorted(allchain.items()):
    d = dtors.get(cn)
    if not d: continue
    body,dpath,dline = d
    b2 = re.sub(r'//[^\n]*','',body); b2 = re.sub(r'/\*.*?\*/','',b2,flags=re.S)
    st = b2.strip()
    if not st: continue
    has_first_guard = bool(re.match(r'if\s*\(\s*copymode\s*\)\s*return\s*;', st))
    mentions = 'copymode' in b2
    frees = bool(FREE_PAT.search(b2))
    if has_first_guard: continue
    if not frees: continue
    res.append({'class':cn,'file':dpath.replace(SRC+'/','src/'),'line':dline,'mentions_copymode':mentions,
                'users':sorted(users)[:6],'nusers':len(users),'body':st})
json.dump(res, open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/dtor2.json','w'), indent=1)
print("copyself classes:",len(copyself),"chain classes:",len(allchain))
for r in res:
    print('---', r['class'], r['file'], r['line'], 'mentions_copymode=',r['mentions_copymode'], 'nusers=',r['nusers'], r['users'][:3])
