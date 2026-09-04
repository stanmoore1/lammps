import json,sys,os
P='/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/verify/progress_v02/'
obj=json.load(open(sys.argv[1]))
if not isinstance(obj,list): obj=[obj]
done=set(open(P+'done.txt').read().split())
with open(P+'done.txt','a') as d, open(P+'verdicts.jsonl','a') as v:
    for o in obj:
        if o['id'] in done: 
            print('skip',o['id']); continue
        d.write(o['id']+'\n'); v.write(json.dumps(o)+'\n')
        print('added',o['id'])
