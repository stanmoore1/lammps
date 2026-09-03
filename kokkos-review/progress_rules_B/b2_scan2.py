import re,glob,os,json
SRC='/home/user/lammps/src/KOKKOS'
BAD=[(r'\bstd::\w+','std::'),(r'\bMathSpecial::','MathSpecial::'),(r'\bMathExtra::','MathExtra::'),
     (r'(?<![\w:>.])rand\s*\(\)','rand()'),(r'\berror\s*->','error->'),(r'\butils::','utils::'),
     (r'(?<![\w:])new\s','new'),(r'(?<![\w:])delete\s','delete'),(r'\bmemory\s*->','memory->'),
     (r'\bmemoryKK\s*->','memoryKK->'),(r'\bdomain\s*->','domain->'),(r'\bforce\s*->','force->'),
     (r'\bcomm\s*->','comm->'),(r'\bupdate\s*->','update->'),(r'\bmodify\s*->','modify->'),
     (r'\bneighbor\s*->','neighbor->'),(r'\batom\s*->','atom->'),(r'\batomKK\s*->','atomKK->'),
     (r'\bMPI_','MPI_'),(r'\bfopen\b|\bfprintf\b|\bfclose\b','stdio')]
def clean(txt):
    # remove block comments & line comments & strings, keep newlines
    txt=re.sub(r'/\*.*?\*/', lambda m:'\n'*m.group(0).count('\n'), txt, flags=re.S)
    out=[]
    for l in txt.split('\n'):
        l=re.sub(r'"(\\.|[^"\\])*"','""',l); l=re.sub(r"'(\\.|[^'\\])*'","''",l)
        l=re.sub(r'//.*','',l)
        out.append(l)
    return out
hits=[]
for f in sorted(glob.glob(SRC+'/*.cpp'))+sorted(glob.glob(SRC+'/*.h')):
    raw=open(f,errors='replace').read()
    lines=clean(raw); n=len(lines)
    i=0
    while i<n:
        if re.search(r'KOKKOS_(INLINE_)?FUNCTION|KOKKOS_LAMBDA',lines[i]):
            # locate first '{' from i (skip declarations ending with ';' before any '{')
            j=i; pos=None
            while j<n:
                b=lines[j].find('{'); s=lines[j].find(';')
                if s>=0 and (b<0 or s<b): pos=None; break   # pure declaration
                if b>=0: pos=(j,b); break
                j+=1
            if pos is None: i+=1; continue
            sj,sb=pos; depth=0; k=sj; col=sb; end=None
            while k<n:
                start=col if k==sj else 0
                for c in range(start,len(lines[k])):
                    ch=lines[k][c]
                    if ch=='{': depth+=1
                    elif ch=='}':
                        depth-=1
                        if depth==0: end=k; break
                if end is not None: break
                k+=1
            if end is None: end=min(n-1,sj+300)
            for k2 in range(sj,end+1):
                for p,nm in BAD:
                    if re.search(p,lines[k2]):
                        hits.append((os.path.basename(f),k2+1,nm,raw.split('\n')[k2].strip()[:150]))
            i=sj+1
        else: i+=1
json.dump(hits,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_B/b2_hits2.json','w'),indent=1)
from collections import Counter
print(Counter(h[2] for h in hits)); print('total',len(hits))
