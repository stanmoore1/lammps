import re, os, json, glob
SRC='/home/user/lammps/src/KOKKOS'
out=[]
for h in sorted(glob.glob(SRC+'/*.h')):
    txt=open(h,errors='replace').read().split('\n')
    # is the class templated on DeviceType?
    templated = bool(re.search(r'template\s*<\s*class\s+DeviceType\s*>', '\n'.join(txt)))
    members=[]
    for i,l in enumerate(txt):
        m=re.search(r'\bDAT::(t_\w+)\s+([A-Za-z_][\w, ]*);', l)
        if m:
            names=[n.strip() for n in m.group(2).split(',')]
            for n in names:
                members.append({'name':n,'type':m.group(1),'line':i+1,'decl':l.strip()})
    if members:
        out.append({'header':os.path.basename(h),'templated':templated,'members':members})
json.dump(out,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_B/b1_members.json','w'),indent=1)
print(len(out),'headers', sum(len(o['members']) for o in out),'members')
