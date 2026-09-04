import re,sys
def norm(s):
    s=re.sub(r'\s+','',s)
    s=s.replace('knots[','K[')
    # normalize float literals like 3.0 -> 3, 2.0 -> 2
    s=re.sub(r'(?<![\w.])(\d+)\.0(?![\d])',r'\1',s)
    return s

# KK get_constants: lines 1305..? in pair_uf3_kokkos.cpp
kk=open('/home/user/lammps/src/KOKKOS/pair_uf3_kokkos.cpp').read()
cpu3=open('/home/user/lammps/src/ML-UF3/uf3_bspline_basis3.cpp').read()
cpu2=open('/home/user/lammps/src/ML-UF3/uf3_bspline_basis2.cpp').read()

def grab_kk(fnname):
    i=kk.index('PairUF3Kokkos<DeviceType>::'+fnname)
    j=kk.index('return constants;',i)
    return kk[i:j]

def assigns(text, pat):
    # returns dict index->normalized expr
    out={}
    for m in re.finditer(pat, text, re.S):
        out.setdefault(m.group(1),[]).append(norm(m.group(2)))
    return out

kkc=grab_kk('get_constants')
kkd=grab_kk('get_dnconstants')
kk_const=assigns(kkc, r'constants\[(\d+)\]\s*=\s*(.*?);\n')
kk_dn=assigns(kkd, r'constants\[(\d+)\]\s*=\s*(.*?);\n')

# CPU: c0..c3 blocks then constants[n]=cX
def cpu_expand(text):
    # replace 'c0 = ...;' sequences and 'constants[i] = cX;'
    res={}
    cur={}
    for m in re.finditer(r'\n\s*(c\d)\s*=\s*(.*?);\n', text, re.S):
        cur[m.group(1)]=norm(m.group(2))
        # record position
    return text
# simpler: walk sequentially
def cpu_map(text):
    res={}
    cur={}
    toks=re.finditer(r'\n\s*(?:(c\d)\s*=\s*(.*?);|constants\[(\d+)\]\s*=\s*(c\d);)\n', text, re.S)
    for m in toks:
        if m.group(1):
            cur[m.group(1)]=norm(m.group(2))
        else:
            res.setdefault(m.group(3),[]).append(cur.get(m.group(4),'MISSING'))
    return res

cpu3m=cpu_map(cpu3)
cpu2m=cpu_map(cpu2)

def cmp(name, a, b):
    ks=sorted(set(a)|set(b), key=int)
    bad=0
    for k in ks:
        av=a.get(k,['<none>'])[0] if k in a else '<none>'
        bv=b.get(k,['<none>'])[0] if k in b else '<none>'
        if av!=bv:
            bad+=1
            print(f"{name} MISMATCH constants[{k}]")
            print("  KK :",av[:400])
            print("  CPU:",bv[:400])
    print(f"{name}: {len(ks)} entries, {bad} mismatches")

cmp('get_constants', kk_const, cpu3m)
cmp('get_dnconstants', kk_dn, cpu2m)
