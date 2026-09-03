#!/usr/bin/env python3
import os, re, json
SRC="/home/user/lammps/src/KOKKOS"
OUT="/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_C"
F2M=json.load(open(os.path.join(OUT,'f2m.json')))

def strip_comments(txt):
    txt = re.sub(r'/\*.*?\*/', lambda m: '\n'*m.group(0).count('\n'), txt, flags=re.S)
    return re.sub(r'//[^\n]*', '', txt)
def masks_in(s): return set(re.findall(r'\b[A-Z][A-Z_0-9]*_MASK\b', s))

report={}
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('_kokkos.cpp'): continue
    base=os.path.join(SRC,fn)
    files=[base]
    if os.path.exists(base[:-4]+'.h'): files.append(base[:-4]+'.h')
    bind={}   # varname -> field
    lines_all=[]
    for p in files:
        cl=strip_comments(open(p,errors='replace').read()).split('\n')
        for i,l in enumerate(cl): lines_all.append((os.path.basename(p),i+1,l))
    for f,i,l in lines_all:
        for m in re.finditer(r'([A-Za-z_][A-Za-z_0-9]*)\s*=\s*atomKK->k_([A-Za-z_0-9]+)\s*\.\s*(?:template\s+)?view', l):
            bind.setdefault(m.group(1), set()).add(m.group(2))
        for m in re.finditer(r'([A-Za-z_][A-Za-z_0-9]*)\s*=\s*atomKK->k_([A-Za-z_0-9]+)\s*;', l):
            bind.setdefault(m.group(1), set()).add(m.group(2))
        # scatter/atomic aliases:  a_f = f;   ScatterView from d_f
        for m in re.finditer(r'\b([a-z_][A-Za-z_0-9]*)\s*=\s*([a-z_][A-Za-z_0-9]*)\s*;', l):
            pass
    # second pass: alias propagation (2 rounds)
    for _ in range(3):
        for f,i,l in lines_all:
            for m in re.finditer(r'\b([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([A-Za-z_][A-Za-z_0-9]*)\s*;', l):
                lhs,rhs=m.group(1),m.group(2)
                if rhs in bind: bind.setdefault(lhs,set()).update(bind[rhs])
            for m in re.finditer(r'\b([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(?:Kokkos::)?(?:Experimental::)?create_scatter_view[^(]*\(\s*([A-Za-z_][A-Za-z_0-9]*)', l):
                lhs,rhs=m.group(1),m.group(2)
                if rhs in bind: bind.setdefault(lhs,set()).update(bind[rhs])
    writes={}
    for f,i,l in lines_all:
        for var,flds in bind.items():
            # write pattern: var(...) = / += / -= ; or atomic_add(&var(
            if re.search(r'\b'+re.escape(var)+r'\s*\([^;]*?\)\s*(\+=|-=|\*=|/=|=[^=])', l) or \
               re.search(r'atomic_(add|sub|fetch|exchange|increment|decrement)[^;]*&\s*'+re.escape(var)+r'\s*\(', l) or \
               re.search(r'deep_copy\s*\(\s*'+re.escape(var)+r'\b', l):
                for fld in flds:
                    if fld in F2M: writes.setdefault(F2M[fld],[]).append((f,i,var,l.strip()[:120]))
    reads={}
    for f,i,l in lines_all:
        for var,flds in bind.items():
            if re.search(r'\b'+re.escape(var)+r'\s*\(', l):
                for fld in flds:
                    if fld in F2M: reads.setdefault(F2M[fld],[]).append((f,i))
    txt=''.join(open(p,errors='replace').read() for p in files)
    txt=strip_comments(txt)
    dmm=re.findall(r'datamask_modify\s*\|?=\s*([^;]*);',txt,re.S)
    dmr=re.findall(r'datamask_read\s*\|?=\s*([^;]*);',txt,re.S)
    mod_calls=set()
    for f,i,l in lines_all:
        if re.search(r'->modified\s*[<(]', l): mod_calls |= masks_in(l)
    sync_calls=set()
    for f,i,l in lines_all:
        if re.search(r'->sync(_pinned|_overlapping_device)?\s*[<(]', l): sync_calls |= masks_in(l)
    dmm_m=set().union(*[masks_in(x) for x in dmm]) if dmm else set()
    dmr_m=set().union(*[masks_in(x) for x in dmr]) if dmr else set()
    report[fn]={
      'bind':{k:sorted(v) for k,v in sorted(bind.items()) if any(x in F2M for x in v)},
      'written_masks':{k:v[:6] for k,v in sorted(writes.items())},
      'read_masks':sorted(reads.keys()),
      'dmm':[' '.join(x.split()) for x in dmm],'dmr':[' '.join(x.split()) for x in dmr],
      'dmm_masks':sorted(dmm_m),'dmr_masks':sorted(dmr_m),
      'mod_calls':sorted(mod_calls),'sync_calls':sorted(sync_calls),
      'write_not_declared':sorted([m for m in writes if m not in dmm_m and m not in mod_calls and 'ALL_MASK' not in dmm_m and 'ALL_MASK' not in mod_calls]),
      'read_not_declared':sorted([m for m in reads if m not in dmr_m and m not in sync_calls and 'ALL_MASK' not in dmr_m and 'ALL_MASK' not in sync_calls]),
      'declared_not_written':sorted([m for m in dmm_m if m not in writes and m not in ('ENERGY_MASK','VIRIAL_MASK','EMPTY_MASK','ALL_MASK')]),
    }
json.dump(report, open(os.path.join(OUT,'scan5.json'),'w'), indent=1)
print("== WRITE NOT DECLARED ==")
for k,v in report.items():
    if v['write_not_declared']: print(k, v['write_not_declared'])
print("== READ NOT DECLARED ==")
for k,v in report.items():
    if v['read_not_declared']: print(k, v['read_not_declared'])
