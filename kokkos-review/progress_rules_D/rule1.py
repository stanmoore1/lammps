import os,re,json,collections
SRC='/home/user/lammps/src'
K=os.path.join(SRC,'KOKKOS')
kfiles=sorted(f for f in os.listdir(K) if f.endswith(('.cpp','.h')))
# collect all files everywhere in src (core + packages)
loc=collections.defaultdict(list)
for d,dn,fn in os.walk(SRC):
    dn[:] = [x for x in dn if x not in ('STUBS','Obj_serial','Obj_mpi')]
    rel=os.path.relpath(d,SRC)
    for f in fn:
        if f.endswith(('.cpp','.h','.pyx')): loc[f].append(rel)
acts=[]
for i,line in enumerate(open(os.path.join(K,'Install.sh')),1):
    m=re.match(r'^\s*action\s+(\S+)(?:\s+(\S+))?\s*$',line)
    if m: acts.append((i,m.group(1),m.group(2)))
registered={a[1] for a in acts}
missing=[f for f in kfiles if f not in registered]
# action lines naming nonexistent kokkos file
badfirst=[a for a in acts if a[1] not in kfiles]
# dep checks
depnonexist=[]; depwrong=[]
for ln,f1,f2 in acts:
    if f1 not in kfiles: continue
    if f2:
        if f2 not in loc: depnonexist.append((ln,f1,f2))
    else:
        # single-arg: base class must live in core src/.  Derive expected base name
        base=None
        for suf in ('_kokkos.cpp','_kokkos.h'):
            if f1.endswith(suf):
                ext='.cpp' if suf.endswith('.cpp') else '.h'
                base=f1[:-len(suf)]+ext
        if base and base in loc:
            dirs=loc[base]
            if '.' not in dirs:
                depwrong.append((ln,f1,base,dirs))
out={'n_kokkos_files':len(kfiles),'n_action':len(acts),
     'missing_action':missing,'action_file_nonexistent':badfirst,
     'dep_nonexistent':depnonexist,'dep_missing_base_in_package':depwrong}
json.dump(out,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_D/rule1.json','w'),indent=1)
for k,v in out.items():
    if isinstance(v,list): print(k,len(v))
    else: print(k,v)
