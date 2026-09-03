import os,re,json
K='/home/user/lammps/src/KOKKOS'
bad=[]
for f in sorted(os.listdir(K)):
    if not f.endswith('.h'): continue
    t=open(os.path.join(K,f),errors='replace').read()
    guards=re.findall(r'^\s*#\s*ifndef\s+(LMP_\w+)\s*$',t,re.M)
    defs=re.findall(r'^\s*#\s*define\s+(LMP_\w+)\s*$',t,re.M)
    exp='LMP_'+f[:-2].upper()+'_H'
    ok=[g for g in guards if g in defs]
    n_if=len(re.findall(r'^\s*#\s*if(?:n?def|\s)',t,re.M))
    n_end=len(re.findall(r'^\s*#\s*endif',t,re.M))
    prob=[]
    if not ok: prob.append('NO include guard (ifndef/define pair): ifndef=%s define=%s'%(guards,defs))
    elif exp not in ok: prob.append('guard name %s != expected %s'%(ok,exp))
    if n_if!=n_end: prob.append('#if count %d != #endif count %d'%(n_if,n_end))
    # duplicate guard macro used in another KOKKOS header
    if prob: bad.append((f,prob))
# duplicate guard macros across files
seen={}
dups=[]
for f in sorted(os.listdir(K)):
    if not f.endswith('.h'): continue
    t=open(os.path.join(K,f),errors='replace').read()
    for g in re.findall(r'^\s*#\s*define\s+(LMP_\w+_H)\s*$',t,re.M):
        if g in seen: dups.append((g,seen[g],f))
        else: seen[g]=f
json.dump({'guard_problems':bad,'dup_guard_macros':dups},open('rule6_guards.json','w'),indent=1)
print('guard problems',len(bad))
for f,p in bad: print(' ',f,p)
print('dup guard macros',len(dups))
for d in dups: print(' ',d)
