import os,re,json,sys,glob,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_parse import *
OUT=os.path.dirname(os.path.abspath(__file__))
classes=build_index()

SKIP_NAMES=set(['if','for','while','switch','return','else','do','catch','sizeof','static_cast',
 'operator','public','private','protected','template','typedef','using','struct','class','enum'])

DECL=re.compile(r'''^[ \t]*(?P<pre>(?:(?:virtual|static|inline|explicit|constexpr|friend|KOKKOS_INLINE_FUNCTION|KOKKOS_FUNCTION|typename|template\s*<[^>]*>)\s+)*)
                    (?P<ret>[A-Za-z_][\w:<>,\s\*&]*?[\s\*&])
                    (?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^;{]*?)\)\s*
                    (?P<post>(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?(?:=\s*0\s*)?(?:=\s*default\s*)?(?:=\s*delete\s*)?)
                    [;{]''', re.X)

def methods_of(body, startline, hpath):
    """parse member function declarations in a class body (comments already stripped)"""
    res=collections.defaultdict(list)
    lines=body.split('\n')
    i=0
    # merge continuation lines so multi-line declarations are seen
    while i < len(lines):
        buf=lines[i]; j=i
        while ('(' in buf) and (')' not in buf.split('(',1)[1]) and j+1<len(lines) and j-i<6:
            j+=1; buf+=' '+lines[j].strip()
        m=DECL.match(buf.strip().join(['','']) if False else buf)
        if m:
            nm=m.group('name')
            if nm not in SKIP_NAMES and not nm.startswith('operator'):
                virt = 'virtual' in m.group('pre')
                ov   = 'override' in m.group('post') or 'final' in m.group('post')
                pure = '= 0' in m.group('post').replace(' ','= 0') or re.search(r'=\s*0',m.group('post'))
                res[nm].append({'virtual':virt,'override':ov,'pure':bool(pure),
                                'line':startline+i,'decl':' '.join(buf.split())[:160],
                                'header':hpath})
        i=j+1
    return res

# index: class -> methods
cinfo={}
for name,ents in classes.items():
    for (nm,bases,path,line,body,bl) in ents:
        key=(nm,path)
        cinfo[key]={'name':nm,'bases':bases,'path':path,'line':line,
                    'methods':methods_of(body,bl,path)}

byname=collections.defaultdict(list)
for k,v in cinfo.items(): byname[v['name']].append(v)

def resolve(n, avoid_kokkos=False):
    l=byname.get(n)
    if not l: return None
    if avoid_kokkos:
        for e in l:
            if '/KOKKOS/' not in e['path']: return e
    return l[0]

def chain(n):
    seen={n}; out=[]; e0=resolve(n)
    st=list(e0['bases']) if e0 else []
    while st:
        x=st.pop(0)
        if x in seen: continue
        seen.add(x); e=resolve(x)
        if e: out.append(e); st.extend(e['bases'])
        else: out.append({'name':x,'bases':[],'path':None,'methods':{}})
    return out

kk=[]
for h in sorted(glob.glob(SRC+'/KOKKOS/*.h')):
    for (nm,bases,path,line,body,bl) in parse_classes(h):
        kk.append(nm)

flags=[]
for nm in kk:
    e=resolve(nm)
    if not e or '/KOKKOS/' not in (e['path'] or ''): continue
    ch=chain(nm)
    for mname, decls in e['methods'].items():
        for d in decls:
            if d['override'] or d['virtual']: continue
            if d['pure']: continue
            # find base declaration of same name
            basedecls=[]
            for b in ch:
                for bd in b['methods'].get(mname,[]):
                    basedecls.append((b['name'],bd))
            if not basedecls: continue
            anyvirt=any(bd['virtual'] or bd['override'] for _,bd in basedecls)
            if not anyvirt:
                flags.append({'kk_class':nm,'kk_header':e['path'].replace(SRC+'/','src/'),
                              'method':mname,'kk_line':d['line'],'kk_decl':d['decl'],
                              'bases':[{'class':b,'line':bd['line'],'decl':bd['decl'],
                                        'header':(bd['header'] or '').replace(SRC+'/','src/')} for b,bd in basedecls]})
json.dump(flags,open(OUT+'/rule2_raw.json','w'),indent=1)
print('flags:',len(flags))
seen=set()
for f in flags:
    k=(f['kk_class'],f['method'])
    print('%-38s %-24s %s:%d  base: %s' % (f['kk_class'],f['method'],f['kk_header'],f['kk_line'],
          ', '.join('%s@%s:%d'%(b['class'],b['header'],b['line']) for b in f['bases'][:2])))
