import os,re,glob,json
SRC='/home/user/lammps/src/KOKKOS'
out=[]
for cpp in sorted(glob.glob(SRC+'/*.cpp'))+sorted(glob.glob(SRC+'/*.h')):
    txt=open(cpp,encoding='utf-8',errors='replace').read()
    lines=txt.split('\n')
    copies=[]
    for i,l in enumerate(lines,1):
        if re.search(r'(parallel_for|parallel_reduce|parallel_scan)\s*\(', l) or re.search(r'^\s*(\*this|this->\w+)\s*,', l):
            pass
    # find *this used as functor argument (multi-line aware)
    for m in re.finditer(r'Kokkos::parallel_(?:for|reduce|scan)\s*\((?:[^;]{0,600}?)\*this', txt, re.S):
        copies.append(txt[:m.start()].count('\n')+1)
    for m in re.finditer(r'KOKKOS_CLASS_LAMBDA', txt):
        copies.append(txt[:m.start()].count('\n')+1)
    sets = [txt[:m.start()].count('\n')+1 for m in re.finditer(r'copymode\s*=\s*1', txt)]
    if copies:
        out.append({'file':os.path.basename(cpp),'copy_lines':sorted(set(copies))[:6],'ncopy':len(copies),'copymode1':sets})
json.dump(out,open(os.path.dirname(os.path.abspath(__file__))+'/rule1c.json','w'),indent=1)
print("files that copy *this into a kernel:",len(out))
for o in out:
    flag='  <== NO copymode=1' if not o['copymode1'] else ''
    print('%-45s ncopy=%-4d copymode1@%s%s'%(o['file'],o['ncopy'],o['copymode1'][:4],flag))
