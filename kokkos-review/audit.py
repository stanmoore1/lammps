import os, re, sys, json, glob
SRC = "/home/user/lammps/src"

# ---- 1. Index all class declarations in all headers under src ----
class_re = re.compile(r'^\s*class\s+([A-Za-z_]\w*)\s*(?::\s*([^{]*))?\{', re.M)
# handle multi-line: join lines up to '{'
def parse_classes(path):
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except Exception:
        return []
    # strip comments crudely
    txt = re.sub(r'//[^\n]*', '', txt)
    txt = re.sub(r'/\*.*?\*/', '', txt, flags=re.S)
    out = []
    lines = txt.split('\n')
    for i, ln in enumerate(lines):
        m = re.match(r'\s*class\s+([A-Za-z_]\w*)\b', ln)
        if not m: continue
        name = m.group(1)
        # collect until '{' or ';'
        buf = ln
        j = i
        while '{' not in buf and ';' not in buf and j+1 < len(lines) and j-i < 6:
            j += 1
            buf += ' ' + lines[j]
        if '{' not in buf:
            continue  # forward decl
        head = buf.split('{')[0]
        bases = []
        if ':' in head:
            bpart = head.split(':',1)[1]
            for b in bpart.split(','):
                b = b.strip()
                b = re.sub(r'\b(public|protected|private|virtual)\b','',b).strip()
                b = b.split('<')[0].strip()
                b = b.split('::')[-1].strip()
                if b: bases.append(b)
        out.append((name, bases, path, i+1))
    return out

classes = {}   # name -> list of (bases, header, line)
for root, dirs, files in os.walk(SRC):
    if os.sep+'STUBS' in root: continue
    for f in files:
        if f.endswith('.h'):
            p = os.path.join(root,f)
            for name,bases,path,line in parse_classes(p):
                classes.setdefault(name, []).append((bases,path,line))

# ---- 2. Index destructor bodies in .cpp (and inline in .h) ----
def find_dtors(path):
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except Exception:
        return {}
    res = {}
    for m in re.finditer(r'^([A-Za-z_]\w*)::~\1\s*\(\s*\)', txt, re.M):
        cname = m.group(1)
        # find opening brace
        idx = txt.find('{', m.end())
        if idx < 0: continue
        # match braces
        depth = 0; k = idx
        while k < len(txt):
            if txt[k]=='{': depth+=1
            elif txt[k]=='}':
                depth-=1
                if depth==0: break
            k+=1
        body = txt[idx+1:k]
        line = txt[:m.start()].count('\n')+1
        res[cname] = (body, path, line)
    return res

dtors = {}
for root, dirs, files in os.walk(SRC):
    for f in files:
        if f.endswith('.cpp') or f.endswith('.h'):
            p = os.path.join(root,f)
            for c,v in find_dtors(p).items():
                if c not in dtors or '/KOKKOS/' not in p:
                    dtors.setdefault(c, v)
                dtors[c]=v if c not in dtors else dtors[c]

# redo properly: prefer .cpp definitions
dtors = {}
for root, dirs, files in os.walk(SRC):
    for f in files:
        p = os.path.join(root,f)
        if f.endswith('.cpp'):
            for c,v in find_dtors(p).items():
                dtors[c]=v
for root, dirs, files in os.walk(SRC):
    for f in files:
        p = os.path.join(root,f)
        if f.endswith('.h'):
            for c,v in find_dtors(p).items():
                dtors.setdefault(c,v)

# also detect header-declared inline/defaulted dtors
def dtor_decl_in_header(cname):
    ents = classes.get(cname, [])
    out = []
    for bases,hpath,line in ents:
        txt = open(hpath, encoding='utf-8', errors='replace').read()
        for m in re.finditer(r'~'+re.escape(cname)+r'\s*\([^)]*\)\s*([^;{]*)([;{])', txt):
            out.append((m.group(0), hpath, txt[:m.start()].count('\n')+1))
    return out

json.dump({'nclasses':len(classes),'ndtors':len(dtors)}, sys.stdout)
print()

# ---- 3. Walk chains for kokkos classes ----
def resolve(name, prefer_nonkokkos=False):
    ents = classes.get(name)
    if not ents: return None
    if prefer_nonkokkos:
        for e in ents:
            if '/KOKKOS/' not in e[1]: return e
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
        for b in e[0]:
            stack.append(b)
    return out

FREE_PAT = re.compile(r'destroy_kokkos|memory->destroy|memory->sfree|delete\s*\[\]|\bdelete\b|->destroy|free\(')
GUARD_PAT = re.compile(r'^\s*if\s*\(\s*copymode\s*\)\s*return\s*;')

kokkos_headers = sorted(glob.glob(SRC+"/KOKKOS/*.h"))
kk_classes = []
for h in kokkos_headers:
    for name,bases,path,line in parse_classes(h):
        kk_classes.append((name,bases,path,line))

report_dtor = []
checked=set()
for name,bases,path,line in kk_classes:
    for cn, (cbases, chpath, cline) in chain(name):
        if cn in checked: continue
        checked.add(cn)
        d = dtors.get(cn)
        if not d: continue
        body, dpath, dline = d
        # strip comments
        b2 = re.sub(r'//[^\n]*','',body); b2 = re.sub(r'/\*.*?\*/','',b2,flags=re.S)
        stripped = b2.strip()
        if not stripped: continue
        has_guard = bool(GUARD_PAT.match(b2.lstrip('\n')) or re.match(r'\s*if\s*\(copymode\)\s*return;', stripped))
        frees = bool(FREE_PAT.search(b2))
        if not has_guard:
            report_dtor.append({'class':cn,'file':dpath,'line':dline,'frees':frees,
                                'first': stripped.split('\n')[0][:120],
                                'is_kokkos': '/KOKKOS/' in dpath,
                                'body_len': len(stripped.split('\n'))})

json.dump(report_dtor, open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/dtor_raw.json','w'), indent=1)
print("dtor candidates:", len(report_dtor), "frees:", sum(1 for r in report_dtor if r['frees']))
