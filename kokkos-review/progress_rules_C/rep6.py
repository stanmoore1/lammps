import json,os,re
OUT="/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/progress_rules_C"
d=json.load(open(os.path.join(OUT,'scan4.json')))
# hooks that VerletKokkos/ModifyKokkos pre-sync with datamask_read
HOOKS=set("""compute setup setup_pre_force initial_integrate final_integrate post_force pre_force
post_integrate pre_exchange pre_neighbor post_neighbor pre_reverse end_of_step min_post_force
min_pre_force min_pre_exchange min_pre_neighbor min_post_neighbor min_pre_reverse
post_force_respa min_setup setup_pre_neighbor compute_peratom compute_scalar compute_vector
compute_array""".split())
out=[]
for fn,v in sorted(d.items()):
    dmr=set(v['dmr_masks']); dmm=set(v['dmm_masks'])
    for f in v['funcs']:
        fname=f['func'].split('::')[-1]
        cover = set(f['sync_masks'])
        if fname in HOOKS and 'EMPTY_MASK' not in dmr: cover |= dmr
        if 'ALL_MASK' in cover: continue
        miss = {m:l for m,l in f['need'].items() if m not in cover}
        hmiss = {m:l for m,l in f['hneed'].items() if m not in cover}
        if miss or hmiss:
            out.append((fn, f['func'], f['start'], f['end'], sorted(miss.items()), sorted(hmiss.items()),
                        f['sync_masks'], sorted(dmr), fname in HOOKS))
print(len(out))
for o in out:
    print(f"{o[0]}:{o[2]}-{o[3]} {o[1]} hook={o[8]} missK={o[4]} missHost={o[5]} sync={o[6]} dmr={o[7]}")
