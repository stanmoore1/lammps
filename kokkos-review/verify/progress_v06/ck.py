import json,sys,os
D="/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/verify/progress_v06/"
obj=json.load(open(sys.argv[1]))
with open(D+"verdicts.jsonl","a") as f: f.write(json.dumps(obj)+"\n")
with open(D+"done.txt","a") as f: f.write(obj["id"]+"\n")
print("ok",obj["id"])
