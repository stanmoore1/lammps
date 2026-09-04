import re
def norm(s):
    s=re.sub(r'\s+','',s)
    s=s.replace('knots[','K[')
    s=re.sub(r'(?<![\w.])(\d+)\.0(?![\d])',r'\1',s)
    return s

def statements(text):
    # crude: split on ';' at top level
    out=[]
    depth=0; cur=''
    for ch in text:
        if ch=='(': depth+=1
        if ch==')': depth-=1
        if ch==';' and depth==0:
            out.append(cur); cur=''
        else:
            cur+=ch
    return out

def build(text):
    res={}; cur={}
    for st in statements(text):
        st=st.strip()
        m=re.match(r'^(c\d)\s*=\s*(.*)$', st, re.S)
        if m: cur[m.group(1)]=norm(m.group(2)); continue
        m=re.match(r'^constants\[(\d+)\]\s*=\s*(.*)$', st, re.S)
        if m:
            v=m.group(2).strip()
            if re.fullmatch(r'c\d', v): res[m.group(1)]=cur.get(v,'MISSING:'+v)
            else: res[m.group(1)]=norm(v)
    return res

kk=open('/home/user/lammps/src/KOKKOS/pair_uf3_kokkos.cpp').read()
def grab_kk(fn):
    i=kk.index('PairUF3Kokkos<DeviceType>::'+fn)
    j=kk.index('return constants;',i)
    return kk[i:j]

cpu3=open('/home/user/lammps/src/ML-UF3/uf3_bspline_basis3.cpp').read()
cpu2=open('/home/user/lammps/src/ML-UF3/uf3_bspline_basis2.cpp').read()
i=cpu3.index('uf3_bspline_basis3::uf3_bspline_basis3'); cpu3b=cpu3[i:]
i=cpu2.index('uf3_bspline_basis2::uf3_bspline_basis2'); cpu2b=cpu2[i:]

for name,a,b in [('get_constants(3rd)',build(grab_kk('get_constants')),build(cpu3b)),
                 ('get_dnconstants(2nd)',build(grab_kk('get_dnconstants')),build(cpu2b))]:
    ks=sorted(set(a)|set(b),key=int); bad=0
    for k in ks:
        av=a.get(k,'<none>'); bv=b.get(k,'<none>')
        if av!=bv:
            bad+=1
            print(f"### {name} constants[{k}] MISMATCH")
            print("  KK :",av)
            print("  CPU:",bv)
    print(f"{name}: {len(ks)} entries, {bad} mismatches\n")
