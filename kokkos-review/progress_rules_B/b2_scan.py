import re,os,glob,json,sys
SRC='/home/user/lammps/src/KOKKOS'
BAD = [
 (r'\bstd::(pow|exp|expf|sqrt|sqrtf|log|log2|log10|fabs|abs|sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|floor|ceil|round|cbrt|hypot|erfc|erf|isnan|isinf|copysign|fmod|trunc)\b','std-math'),
 (r'(?<![\w:])(pow|exp|sqrt|log|fabs|sin|cos|tan|atan2|floor|ceil|cbrt|erfc)\s*\(','bare-math'),
 (r'\bpowint\s*\(','powint'),
 (r'\bMathSpecial::','MathSpecial'),
 (r'\bMathExtra::','MathExtra'),
 (r'(?<![\w:>.])rand\s*\(\)','rand'),
 (r'\berror\s*->','error-ptr'),
 (r'\butils::','utils'),
 (r'\bstd::vector\b','std::vector'),
 (r'(?<![\w:])new\s+[A-Za-z_]','new'),
 (r'(?<![\w:])delete\s*(\[\])?\s','delete'),
 (r'\batom\s*->','atom->'),
 (r'\bprintf\s*\(','printf'),
 (r'\bmemory\s*->','memory->'),
 (r'\bmemoryKK\s*->','memoryKK->'),
 (r'\batomKK\s*->','atomKK->'),
]
def scan(path):
    lines=open(path,errors='replace').read().split('\n')
    n=len(lines)
    regions=[]  # (start,end,kind)
    i=0
    while i<n:
        l=lines[i]
        kind=None
        if re.search(r'\bKOKKOS_(INLINE_)?FUNCTION\b',l) and not l.strip().startswith('//'):
            kind='devfunc'
        elif re.search(r'KOKKOS_LAMBDA',l):
            kind='lambda'
        if kind:
            # find opening brace of body
            depth=0; j=i; started=False; body_start=None
            while j<n and j<i+400:
                for ch in lines[j]:
                    if ch=='{':
                        depth+=1
                        if not started:
                            started=True; body_start=j
                    elif ch=='}':
                        depth-=1
                if started and depth<=0: break
                j+=1
            if started:
                regions.append((body_start,j,kind))
                i=j+1 if kind=='devfunc' else i+1
                continue
        i+=1
    hits=[]
    for (a,b,kind) in regions:
        for k in range(a,min(b+1,n)):
            code=re.sub(r'//.*','',lines[k])
            for pat,name in BAD:
                if re.search(pat,code):
                    hits.append({'file':os.path.basename(path),'line':k+1,'kind':name,'region':(a+1,b+1,kind),'code':lines[k].strip()[:160]})
    return hits
allh=[]
for f in sorted(glob.glob(SRC+'/*.cpp'))+sorted(glob.glob(SRC+'/*.h')):
    allh+=scan(f)
json.dump(allh,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_B/b2_hits.json','w'),indent=1)
from collections import Counter
c=Counter(h['kind'] for h in allh)
print(c)
print('total',len(allh))
