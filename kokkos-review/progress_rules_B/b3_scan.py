import re,os,glob,json
SRC='/home/user/lammps/src/KOKKOS'
STYLE_RE=re.compile(r'^\s*(\w+Style)\(([^,]+),\s*([^)]+)\)\s*;?',re.M)
out=[]
for h in sorted(glob.glob(SRC+'/*.h')):
    txt=open(h,errors='replace').read()
    hb=os.path.basename(h)
    regs=[]
    for m in STYLE_RE.finditer(txt):
        macro,name,cls=m.group(1),m.group(2).strip(),m.group(3).strip()
        line=txt[:m.start()].count('\n')+1
        regs.append({'macro':macro,'name':name,'class':cls,'line':line})
    if not regs: continue
    # templated classes
    tmpl=set(re.findall(r'template\s*<\s*class\s+DeviceType\s*>\s*class\s+(\w+)',txt))
    cpp=h[:-2]+'.cpp'
    inst={}
    guard={}
    if os.path.exists(cpp):
        ctxt=open(cpp,errors='replace').read().split('\n')
        depth=0; guards=[]
        for i,l in enumerate(ctxt):
            ls=l.strip()
            if re.match(r'#\s*if',ls): guards.append(ls)
            elif re.match(r'#\s*endif',ls) and guards: guards.pop()
            m=re.match(r'template\s+class\s+([\w:]+)\s*<\s*(\w+)\s*>\s*;',ls)
            if m:
                inst.setdefault(m.group(1),[]).append((m.group(2),i+1,list(guards)))
    out.append({'header':hb,'cpp':os.path.basename(cpp) if os.path.exists(cpp) else None,
                'regs':regs,'templated':sorted(tmpl),'inst':inst})
json.dump(out,open('/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_B/b3_raw.json','w'),indent=1)
print(len(out),'headers with style registration')
