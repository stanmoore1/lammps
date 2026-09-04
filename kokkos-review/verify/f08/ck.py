import json,sys,os
P='/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/verify/progress_v08/'
o=json.load(open(sys.argv[1]))
open(P+'done.txt','a').write(o['id']+'\n')
open(P+'verdicts.jsonl','a').write(json.dumps(o)+'\n')
print('ok',o['id'])
