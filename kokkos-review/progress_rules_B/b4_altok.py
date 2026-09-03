import re,glob,os
SRC='/home/user/lammps/src/KOKKOS'
def strip(txt):
    txt=re.sub(r'/\*.*?\*/','',txt,flags=re.S)
    out=[]
    for l in txt.split('\n'):
        l=re.sub(r'//.*','',l)
        l=re.sub(r'"(\\.|[^"\\])*"','""',l)
        l=re.sub(r"'(\\.|[^'\\])*'","''",l)
        out.append(l)
    return out
pats=[(re.compile(r'[\w\)\]]\s+(and|or|xor)\s+[\w\(!]'),'and/or/xor'),
      (re.compile(r'[\(\s&|=,]not\s+[\w\(]'),'not')]
n=0
for f in sorted(glob.glob(SRC+'/*.cpp'))+sorted(glob.glob(SRC+'/*.h')):
    lines=strip(open(f,errors='replace').read())
    for i,l in enumerate(lines):
        for p,nm in pats:
            if p.search(l):
                print(os.path.basename(f),i+1,nm,'|',l.strip()[:150]); n+=1
print('TOTAL',n)
