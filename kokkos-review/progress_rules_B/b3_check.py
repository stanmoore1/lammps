import json,re,os
P='/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_B'
d=json.load(open(P+'/b3_raw.json'))
probs=[]
for e in d:
    regs=e['regs']; tmpl=set(e['templated']); inst=e['inst']
    # base class name used in registrations
    byname={}
    for r in regs:
        m=re.match(r'(\w+)\s*<\s*(\w+)\s*>',r['class'])
        r['base']=m.group(1) if m else r['class'].strip()
        r['param']=m.group(2) if m else None
        byname[r['name']]=r
    classes=set(r['base'] for r in regs)
    tclasses=classes & tmpl
    for cls in sorted(classes):
        istmpl = cls in tmpl
        rs=[r for r in regs if r['base']==cls]
        names=[r['name'] for r in rs]
        if not istmpl:
            # non templated: no <> params expected
            bad=[r for r in rs if r['param']]
            if bad: probs.append((e['header'],'nontempl-with-param',[ (r['name'],r['class'],r['line']) for r in bad]))
            if cls in inst: probs.append((e['header'],'inst-for-nontemplated',cls))
            continue
        # templated
        for r in rs:
            exp = 'LMPHostType' if r['name'].endswith('/host') else 'LMPDeviceType'
            if r['param']!=exp:
                probs.append((e['header'],'wrong-param',(r['name'],r['class'],r['line'],'expected '+exp)))
        base=[n for n in names if not (n.endswith('/device') or n.endswith('/host'))]
        has_dev=any(n.endswith('/device') for n in names)
        has_host=any(n.endswith('/host') for n in names)
        if base and not has_dev: probs.append((e['header'],'missing-kk-device',(cls,names)))
        if base and not has_host: probs.append((e['header'],'missing-kk-host',(cls,names)))
        ins=inst.get(cls,[])
        types=[t for t,l,g in ins]
        if 'LMPDeviceType' not in types:
            probs.append((e['header'],'missing-device-instantiation',(cls,e['cpp'],types)))
        hostent=[x for x in ins if x[0]=='LMPHostType']
        if not hostent:
            probs.append((e['header'],'missing-host-instantiation',(cls,e['cpp'])))
        else:
            for t,l,g in hostent:
                if not any('LMP_KOKKOS_GPU' in gg for gg in g):
                    probs.append((e['header'],'host-inst-not-guarded',(cls,e['cpp'],l,g)))
    # instantiated classes never registered
    for cls in inst:
        if cls not in classes:
            probs.append((e['header'],'instantiated-not-registered',(cls,e['cpp'])))
for p in probs: print(p)
print('TOTAL',len(probs))
