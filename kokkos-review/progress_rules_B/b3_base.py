import json,re,os,glob,subprocess
P='/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_B'
SRC='/home/user/lammps/src'
# collect all style names registered outside KOKKOS
base=set()
for h in glob.glob(SRC+'/**/*.h',recursive=True):
    if '/KOKKOS/' in h: continue
    try: txt=open(h,errors='replace').read()
    except: continue
    for m in re.finditer(r'^\s*(\w+Style)\(([^,]+),',txt,re.M):
        base.add((m.group(1),m.group(2).strip()))
d=json.load(open(P+'/b3_raw.json'))
missing=[]
for e in d:
    for r in e['regs']:
        n=r['name']
        if '/kk' not in n and '/KK' not in n: continue
        b=re.sub(r'/(kk|KK)(/(device|host|DEVICE|HOST))?$','',n)
        if (r['macro'],b) not in base:
            missing.append((e['header'],r['macro'],n,'->',b))
for m in sorted(set(missing)): print(m)
print('TOTAL',len(set(missing)))
