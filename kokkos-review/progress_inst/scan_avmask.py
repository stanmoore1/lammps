import re,glob,os
base='/home/user/lammps/src/KOKKOS'
for f in sorted(glob.glob(base+'/atom_vec_*_kokkos.cpp')):
    txt=open(f).read()
    lines=txt.split('\n')
    # find function bodies of sync/modified/sync_overlapping_device
    for m in re.finditer(r'void (AtomVec\w+Kokkos)::(sync|modified|sync_overlapping_device)\(ExecutionSpace space, unsigned int mask\)',txt):
        start=m.end()
        # take until next "\n}\n" at col0
        end=txt.find('\n}\n',start)
        body=txt[start:end]
        # split by space == branches
        parts=re.split(r'\}?\s*else\s+if\s*\(space\s*==\s*(\w+)\)|if\s*\(space\s*==\s*(\w+)\)',body)
        # simpler: find each branch region
        idxs=[(mm.start(),mm.group(1) or mm.group(2)) for mm in re.finditer(r'(?:else\s+)?if\s*\(space\s*==\s*(\w+)\)',body)]
        idxs.append((len(body),None))
        branches={}
        for i in range(len(idxs)-1):
            s,name=idxs[i]; e=idxs[i+1][0]
            masks=set(re.findall(r'\b([A-Z_0-9]+_MASK)\b',body[s:e]))
            branches[name]=masks
        if len(branches)<2: continue
        keys=list(branches)
        allm=set().union(*branches.values())
        for k in keys:
            miss=allm-branches[k]
            if miss:
                print(os.path.basename(f), m.group(2), 'branch',k,'MISSING',sorted(miss))
