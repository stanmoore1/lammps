import os, re, json, glob, collections
SRC = "/home/user/lammps/src"

def strip_comments(txt):
    txt = re.sub(r'/\*.*?\*/', lambda m: '\n'*m.group(0).count('\n'), txt, flags=re.S)
    txt = re.sub(r'//[^\n]*', '', txt)
    return txt

def parse_classes(path):
    """return list of (name, bases, path, line, body_text, body_startline)"""
    try:
        raw = open(path, encoding='utf-8', errors='replace').read()
    except Exception:
        return []
    txt = strip_comments(raw)
    lines = txt.split('\n')
    out = []
    for i, ln in enumerate(lines):
        m = re.match(r'\s*class\s+([A-Za-z_]\w*)\b', ln)
        if not m: continue
        name = m.group(1)
        buf = ln; j = i
        while '{' not in buf and ';' not in buf and j+1 < len(lines) and j-i < 8:
            j += 1; buf += ' ' + lines[j]
        if '{' not in buf: continue
        head = buf.split('{')[0]
        bases = []
        if ':' in head:
            for b in head.split(':',1)[1].split(','):
                b = re.sub(r'\b(public|protected|private|virtual)\b','',b).strip()
                b = b.split('<')[0].strip().split('::')[-1].strip()
                if b: bases.append(b)
        # body: find matching brace starting from the '{' after head
        # locate absolute offset
        off = 0
        for k in range(i): off += len(lines[k])+1
        bstart = txt.find('{', off)
        depth=0; k=bstart
        while k < len(txt):
            if txt[k]=='{': depth+=1
            elif txt[k]=='}':
                depth-=1
                if depth==0: break
            k+=1
        body = txt[bstart+1:k]
        bodyline = txt[:bstart].count('\n')+1
        out.append((name, bases, path, i+1, body, bodyline))
    return out

def all_headers():
    res=[]
    for root, dirs, files in os.walk(SRC):
        if os.sep+'STUBS' in root: continue
        for f in files:
            if f.endswith('.h'): res.append(os.path.join(root,f))
    return res

def build_index():
    classes = collections.defaultdict(list)
    for p in all_headers():
        for tup in parse_classes(p):
            classes[tup[0]].append(tup)
    return classes

def find_dtors(path):
    try:
        raw = open(path, encoding='utf-8', errors='replace').read()
    except Exception:
        return {}
    txt = raw
    res = {}
    for m in re.finditer(r'^([A-Za-z_]\w*)(<[^>\n]*>)?::~\1\s*\(\s*\)', txt, re.M):
        cname = m.group(1)
        idx = txt.find('{', m.end())
        if idx < 0: continue
        depth=0; k=idx
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

def build_dtors():
    dtors = {}
    for root, dirs, files in os.walk(SRC):
        for f in files:
            if f.endswith('.cpp'):
                for c,v in find_dtors(os.path.join(root,f)).items(): dtors[c]=v
    for root, dirs, files in os.walk(SRC):
        for f in files:
            if f.endswith('.h'):
                for c,v in find_dtors(os.path.join(root,f)).items(): dtors.setdefault(c,v)
    return dtors
