import re,sys
def norm(txt):
    out=[]
    for l in txt.split("\n"):
        l=l.strip()
        if l.startswith("//") or not l: continue
        l=re.sub(r'static_cast<KK_FLOAT>\(([^()]*)\)',r'\1',l)
        l=re.sub(r'static_cast<KK_ACC_FLOAT>\(','(',l)
        l=re.sub(r'Kokkos::','',l)
        l=re.sub(r'KK_FLOAT','double',l)
        l=re.sub(r'\bd_x\((\w+),(\d)\)',r'x[\1][\2]',l)
        l=re.sub(r'\bd_xshake\((\w+),(\d)\)',r'xshake[\1][\2]',l)
        l=re.sub(r'\ba_f\((\w+),(\d)\)',r'f[\1][\2]',l)
        l=re.sub(r'\bd_rmass\[',r'rmass[',l)
        l=re.sub(r'\bd_mass\[',r'mass[',l)
        l=re.sub(r'\bd_type\[',r'type[',l)
        l=re.sub(r'\bd_list\[',r'list[',l)
        l=re.sub(r'\bd_closest_list\((\w+),(\d)\)',r'closest_list[\1][\2]',l)
        l=re.sub(r'\bd_shake_type\((\w+),(\d)\)',r'shake_type[\1][\2]',l)
        l=re.sub(r'\bd_bond_distance\[',r'bond_distance[',l)
        l=re.sub(r'\bd_angle_distance\[',r'angle_distance[',l)
        l=re.sub(r'\bdtfsq_kk\b','dtfsq',l)
        l=re.sub(r'\btolerance_kk\b','tolerance',l)
        l=re.sub(r'\boverflow_kk\b','1e150',l)
        l=re.sub(r'rmass\.data\(\)','rmass',l)
        l=re.sub(r'\s+',' ',l)
        l=re.sub(r'\.0\b','',l)
        out.append(l)
    return out
a=norm(open(sys.argv[1]).read())
b=norm(open(sys.argv[2]).read())
import difflib
for line in difflib.unified_diff(a,b,lineterm='',n=1):
    print(line)
