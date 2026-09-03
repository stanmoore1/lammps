import os,re,json,glob
SRC='/home/user/lammps/src/KOKKOS'
funcstart=re.compile(r'^[A-Za-z_][A-Za-z0-9_:<>,\* &~]*::[~A-Za-z0-9_]+\s*\(')
out=[]
for path in sorted(glob.glob(SRC+'/*.cpp'))+sorted(glob.glob(SRC+'/*.h')):
    lines=open(path,encoding='utf-8',errors='replace').read().split('\n')
    starts=[i for i,l in enumerate(lines) if funcstart.match(l)]
    if not starts: continue
    starts.append(len(lines))
    for si in range(len(starts)-1):
        a,b=starts[si],starts[si+1]
        body=lines[a:b]
        ones=[j for j,l in enumerate(body) if re.search(r'\bcopymode\s*=\s*1',l)]
        if not ones: continue
        zeros=[j for j,l in enumerate(body) if re.search(r'\bcopymode\s*=\s*0',l)]
        first1=ones[0]
        last0=zeros[-1] if zeros else None
        problems=[]
        if last0 is None:
            problems.append('no copymode=0 in this function')
        else:
            for j in range(first1+1,last0):
                l=body[j]
                if re.search(r'^\s*return\b',l): problems.append('return at line %d: %s'%(a+j+1,l.strip()[:90]))
                if re.search(r'\berror->(all|one)\b',l): problems.append('error->all/one at line %d: %s'%(a+j+1,l.strip()[:90]))
        if problems:
            out.append({'file':'src/KOKKOS/'+os.path.basename(path),'func':lines[a].strip()[:110],
                        'func_line':a+1,'copymode1_line':a+first1+1,
                        'copymode0_line':(a+last0+1) if last0 is not None else None,
                        'problems':problems[:6]})
json.dump(out,open(os.path.dirname(os.path.abspath(__file__))+'/rule3.json','w'),indent=1)
print('functions with a copymode=1 and a suspicious path:',len(out))
for o in out:
    print('%-42s %-60s 1@%d 0@%s'%(o['file'],o['func'][:58],o['copymode1_line'],o['copymode0_line']))
    for p in o['problems']: print('      ',p)
