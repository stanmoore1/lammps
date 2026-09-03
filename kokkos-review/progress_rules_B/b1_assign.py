import re, os, glob, json
SRC='/home/user/lammps/src/KOKKOS'
# collect DAT::t_ member names per header (declaration form only: starts with DAT::t_ or typename DAT::t_ then names)
memmap={}
for h in sorted(glob.glob(SRC+'/*.h')):
    for i,l in enumerate(open(h,errors='replace')):
        m=re.match(r'\s*(?:typename\s+)?DAT::(t_\w+)\s+([A-Za-z_][\w\s,]*);\s*(//.*)?$', l)
        if m:
            for n in m.group(2).split(','):
                n=n.strip()
                if n: memmap.setdefault(os.path.basename(h),{})[n]=(i+1,m.group(1),l.rstrip())
res=[]
for hb,mems in memmap.items():
    base=hb[:-2]
    cpp=SRC+'/'+base+'.cpp'
    files=[SRC+'/'+hb]+([cpp] if os.path.exists(cpp) else [])
    for f in files:
        for i,l in enumerate(open(f,errors='replace')):
            for n,(dl,ty,decl) in mems.items():
                if re.search(r'(?<![\w.>])(?:this->)?'+re.escape(n)+r'\s*=\s*', l):
                    res.append({'header':hb,'member':n,'decl_line':dl,'decl':decl,'file':os.path.basename(f),'line':i+1,'code':l.strip()})
json.dump(res,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_B/b1_assign.json','w'),indent=1)
for r in res: print(r['header'],r['member'],'|',r['file'],r['line'],'|',r['code'][:130])
