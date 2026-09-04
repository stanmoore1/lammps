import json,sys,os
D='/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/verify/progress_v00/'
obj=json.load(open(sys.argv[1]))
objs = obj if isinstance(obj,list) else [obj]
with open(D+'done.txt','a') as d, open(D+'verdicts.jsonl','a') as v:
    for o in objs:
        d.write(o['id']+'\n')
        v.write(json.dumps(o)+'\n')
print('ok', [o['id'] for o in objs])
