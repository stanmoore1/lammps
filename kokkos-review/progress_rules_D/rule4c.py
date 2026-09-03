import re
for path in ['/home/user/lammps/src/KOKKOS/npair_halffull_kokkos.h','/home/user/lammps/src/KOKKOS/npair_kokkos.h']:
    lines=open(path).read().splitlines()
    last=None; lastln=0
    for i,l in enumerate(lines,1):
        m=re.match(r'\s*using\s+(\w+)\s*=',l)
        if m: last=m.group(1); lastln=i; continue
        m=re.match(r'\s*NPairStyle\(([^,]+),\s*$',l)
        if m and last:
            cls=lines[i].strip().rstrip(',')
            if cls!=last:
                print('%s:%d  style %-46s uses class %-46s but alias declared at line %d is %s'
                      %(path.split('/')[-1],i,m.group(1),cls,lastln,last))
