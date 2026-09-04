import json,sys,os
D="/tmp/claude-0/-home-user-lammps/fe5acc91-24b0-552d-9a27-dd818dd804e5/scratchpad/review/verify/progress_v01/"
obj=json.load(open(sys.argv[1]))
if isinstance(obj,dict): obj=[obj]
done=set()
if os.path.exists(D+"done.txt"):
    done=set(x.strip() for x in open(D+"done.txt") if x.strip())
with open(D+"done.txt","a") as f1, open(D+"verdicts.jsonl","a") as f2:
    for o in obj:
        if o["id"] in done: 
            print("skip",o["id"]); continue
        f1.write(o["id"]+"\n"); f1.flush()
        f2.write(json.dumps(o)+"\n"); f2.flush()
        print("added",o["id"])
