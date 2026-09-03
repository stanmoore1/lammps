"""Independent cross-check of rule 2, line-based (no class-body parsing).
For every KOKKOS header, find the class -> base mapping via a simple regex, then for
every method-looking declaration line in that header WITHOUT override/virtual, grep the
base header chain for a declaration of the same name and report if none is virtual."""
import os,re,json,glob,sys,collections
SRC='/home/user/lammps/src'
hdrs=[]
for root,dirs,files in os.walk(SRC):
    if os.sep+'STUBS' in root: continue
    for f in files:
        if f.endswith('.h'): hdrs.append(os.path.join(root,f))
# class -> (bases, header) using raw text
cls={}
for h in hdrs:
    txt=open(h,encoding='utf-8',errors='replace').read()
    for m in re.finditer(r'\bclass\s+([A-Za-z_]\w*)\s*:\s*([^{;]*)\{', txt):
        bases=[]
        for b in m.group(2).split(','):
            b=re.sub(r'\b(public|protected|private|virtual)\b','',b).strip().split('<')[0].split('::')[-1].strip()
            if b: bases.append(b)
        cls.setdefault(m.group(1),(bases,h))
def chain(n):
    seen={n};out=[];st=list(cls.get(n,([],None))[0])
    while st:
        x=st.pop(0)
        if x in seen: continue
        seen.add(x); out.append(x); st.extend(cls.get(x,([],None))[0])
    return out
# method declaration lines per header, per class-name-region (approximate: whole header)
DECLLINE=re.compile(r'^\s*(?P<pre>(?:(?:virtual|static|inline|explicit|constexpr|friend|KOKKOS_INLINE_FUNCTION|KOKKOS_FUNCTION)\s+)*)'
                    r'(?P<ret>[A-Za-z_][\w:<>,\s\*&]*?[\s\*&])(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^;{]*)\)\s*(?P<post>[^;{]*)[;{]')
def decls(path):
    res=[]
    for i,l in enumerate(open(path,encoding='utf-8',errors='replace').read().split('\n'),1):
        m=DECLLINE.match(l)
        if not m: continue
        nm=m.group('name')
        if nm.startswith('operator') or nm in ('if','for','while','return','switch','sizeof'): continue
        res.append({'line':i,'name':nm,'virtual':'virtual' in m.group('pre'),
                    'override':'override' in m.group('post') or 'final' in m.group('post'),
                    'kif':'KOKKOS' in m.group('pre'),'text':l.strip()[:160],
                    'nargs':0 if not m.group('args').strip() else m.group('args').count(',')+1})
    return res
cache={}
def d(path):
    if path not in cache: cache[path]=decls(path)
    return cache[path]
out=[]
for h in sorted(glob.glob(SRC+'/KOKKOS/*.h')):
    txt=open(h,encoding='utf-8',errors='replace').read()
    names=[m.group(1) for m in re.finditer(r'\bclass\s+([A-Za-z_]\w*)\s*:',txt)]
    ch=set()
    for n in names: ch|=set(chain(n))
    ch-=set(names)
    basehdrs=[(b,cls[b][1]) for b in ch if b in cls and cls[b][1] and '/KOKKOS/' not in cls[b][1]]
    for dd in d(h):
        if dd['override'] or dd['virtual'] or dd['kif']: continue
        hits=[]
        for b,bh in basehdrs:
            for bd in d(bh):
                if bd['name']==dd['name'] and bd['nargs']==dd['nargs'] and not bd['kif']:
                    hits.append((b,bh,bd))
        if not hits: continue
        if any(bd['virtual'] or bd['override'] for _,_,bd in hits): continue
        out.append({'kk_header':h.replace(SRC+'/','src/'),'kk_line':dd['line'],'method':dd['name'],
                    'kk_decl':dd['text'],
                    'bases':[{'class':b,'header':bh.replace(SRC+'/','src/'),'line':bd['line'],'decl':bd['text']} for b,bh,bd in hits[:3]]})
json.dump(out,open(os.path.dirname(os.path.abspath(__file__))+'/rule2b.json','w'),indent=1)
print('cross-check hits:',len(out))
for o in out:
    print('%-52s:%-5d %-28s | %s' % (o['kk_header'],o['kk_line'],o['method'],
          '; '.join('%s %s:%d'%(b['class'],b['header'],b['line']) for b in o['bases'][:1])))
