import re,sys
for path in ['/home/user/lammps/src/KOKKOS/npair_halffull_kokkos.h','/home/user/lammps/src/KOKKOS/npair_kokkos.h',
             '/home/user/lammps/src/npair_halffull.h']:
    txt=open(path).read()
    # join NPairStyle(...) blocks
    for m in re.finditer(r'NPairStyle\((.*?)\);',txt,re.S):
        blk=m.group(1); ln=txt[:m.start()].count('\n')+1
        parts=[p.strip() for p in blk.split(',',2)]
        name,cls,mask=parts[0],parts[1],parts[2].replace('\n',' ')
        toks=set(name.split('/'))
        prob=[]
        if ('tri' in toks) != ('NP_TRI' in mask): prob.append('name-tri vs NP_TRI')
        if ('trim' in toks) != ('NP_TRIM' in mask): prob.append('name-trim vs NP_TRIM')
        if ('skip' in toks) != ('NP_SKIP' in mask): prob.append('name-skip vs NP_SKIP')
        if ('ghost' in toks) != ('NP_GHOST' in mask): prob.append('name-ghost vs NP_GHOST')
        if ('newton' in toks) != ('NP_NEWTON' in mask): prob.append('name-newton vs NP_NEWTON')
        if ('newtoff' in toks) != ('NP_NEWTOFF' in mask): prob.append('name-newtoff vs NP_NEWTOFF')
        if ('host' in toks) and 'NP_KOKKOS_HOST' not in mask and 'kokkos' in path: prob.append('host mask')
        if ('device' in toks) and 'NP_KOKKOS_DEVICE' not in mask and 'kokkos' in path: prob.append('device mask')
        # alias template args
        am=re.search(r'using\s+'+re.escape(cls)+r'\s*=\s*NPairHalffullKokkos<\s*LMP(\w+)Type\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\s*>',txt)
        if am:
            dev,newton,tri,trim=am.groups()
            if ('newton' in toks)!=(newton=='1'): prob.append('name-newton vs tmpl NEWTON=%s'%newton)
            if ('tri' in toks)!=(tri=='1') and 'newtoff' not in toks: prob.append('name-tri vs tmpl TRI=%s'%tri)
            if ('trim' in toks)!=(trim=='1'): prob.append('name-trim vs tmpl TRIM=%s'%trim)
            if ('device' in toks)!=(dev=='Device'): prob.append('name-dev vs tmpl %s'%dev)
        if prob: print('%s:%d  %-52s %-46s  %s'%(path.split('/')[-1],ln,name,cls,'; '.join(prob)))
