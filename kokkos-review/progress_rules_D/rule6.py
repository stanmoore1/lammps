import os,re,json,collections
SRC='/home/user/lammps/src'; K=os.path.join(SRC,'KOKKOS')
# index all headers by basename
hdr=collections.defaultdict(list)
allf={}
for d,dn,fn in os.walk(SRC):
    dn[:]=[x for x in dn if not x.startswith('Obj')]
    for f in fn:
        if f.endswith(('.h','.cpp')):
            allf[os.path.join(d,f)]=None
            if f.endswith('.h'): hdr[f].append(os.path.join(d,f))
incrx=re.compile(r'^\s*#\s*include\s+"([^"]+)"',re.M)
sysrx=re.compile(r'^\s*#\s*include\s+<([^>]+)>',re.M)
cache={}
def reads(p):
    if p not in cache: cache[p]=open(p,errors='replace').read()
    return cache[p]
def resolve(name,frm):
    d=os.path.dirname(frm)
    c=[os.path.join(d,name),os.path.join(SRC,name)]
    for x in c:
        if os.path.exists(x): return x
    l=hdr.get(os.path.basename(name),[])
    return l[0] if l else None
def closure(p):
    seen=set(); stack=[p]; qs=set(); ss=set()
    while stack:
        cur=stack.pop()
        if cur in seen: continue
        seen.add(cur)
        t=reads(cur)
        for s in sysrx.findall(t): ss.add(s)
        for n in incrx.findall(t):
            qs.add(os.path.basename(n))
            r=resolve(n,cur)
            if r and r not in seen: stack.append(r)
    return qs,ss
REQ=[(r'\batom->',            'atom.h'),
     (r'\batomKK->',          'atom_kokkos.h'),
     (r'\berror->',           'error.h'),
     (r'\bforce->',           'force.h'),
     (r'\bcomm->',            'comm.h'),
     (r'\bdomain->',          'domain.h'),
     (r'\bupdate->',          'update.h'),
     (r'\bmemory->',          'memory.h'),
     (r'\bmemoryKK->',        'memory_kokkos.h'),
     (r'\bmodify->',          'modify.h'),
     (r'\bneighbor->',        'neighbor.h'),
     (r'\boutput->',          'output.h'),
     (r'\bgroup->',           'group.h'),
     (r'\binput->',           'input.h'),
     (r'\buniverse->',        'universe.h'),
     (r'\btimer->',           'timer.h'),
     (r'\bMathConst::',       'math_const.h'),
     (r'\bMathSpecial::',     'math_special.h'),
     (r'\bMathExtra::',       'math_extra.h'),
     (r'\bMathSpecialKokkos::','math_special_kokkos.h'),
     (r'\bMathExtraKokkos::', 'math_extra_kokkos.h'),
     (r'\butils::',           'utils.h'),
     (r'\bfmt::',             'format.h'),
     (r'\bRespa\b',           'respa.h'),
     (r'\bNeighList\b',       'neigh_list.h'),
     (r'\bNeighRequest\b',    'neigh_request.h'),
     ]
SYS=[(r'\b(strcmp|strncmp|strcpy|strlen|memcpy|memset|strtok|strstr|memmove)\s*\(','cstring'),
     (r'\b(sqrt|fabs|pow|exp|log|sin|cos|tan|atan2|acos|asin|floor|ceil)\s*\(','cmath'),
     (r'\b(printf|fprintf|fopen|fclose|fgets|sprintf|snprintf|FILE)\b','cstdio'),
     (r'\bstd::vector\b','vector'),
     (r'\bstd::string\b','string'),
     (r'\bstd::(map|unordered_map)\b','map'),
     ]
out=[]
for f in sorted(os.listdir(K)):
    if not f.endswith(('.cpp','.h')): continue
    p=os.path.join(K,f)
    t=reads(p)
    # strip comments crudely
    body=re.sub(r'/\*.*?\*/','',t,flags=re.S)
    body=re.sub(r'//[^\n]*','',body)
    qs,ss=closure(p)
    miss=[]
    for rx,h in REQ:
        if re.search(rx,body) and h not in qs: miss.append(h)
    for rx,h in SYS:
        if re.search(rx,body) and h not in ss and h.replace('c','',1) not in ss: miss.append('<'+h+'>')
    if miss: out.append({'file':'src/KOKKOS/'+f,'missing':sorted(set(miss))})
json.dump(out,open('rule6_includes.json','w'),indent=1)
print('files with candidate missing includes:',len(out))
cnt=collections.Counter(m for o in out for m in o['missing'])
for k,v in cnt.most_common(): print('  ',k,v)
