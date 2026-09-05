import json,sys,os
P="/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/verify/progress_v09/"
obj=json.load(open(sys.argv[1]))
with open(P+"verdicts.jsonl","a") as f: f.write(json.dumps(obj)+"\n")
with open(P+"done.txt","a") as f: f.write(obj["id"]+"\n")
print("ok",obj["id"])
