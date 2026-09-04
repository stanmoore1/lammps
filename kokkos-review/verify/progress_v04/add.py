import json,sys,os
P="/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/verify/progress_v04/"
obj=json.load(open(sys.argv[1]))
open(P+"verdicts.jsonl","a").write(json.dumps(obj)+"\n")
open(P+"done.txt","a").write(obj["id"]+"\n")
print("ok",obj["id"])
