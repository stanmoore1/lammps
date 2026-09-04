import re,glob,os
base='/home/user/lammps/src/KOKKOS'
# for every file that READS execution_space (sync/modified/etc), check the file (or its .h/.cpp partner) sets it
files=sorted(glob.glob(base+'/*.cpp'))+sorted(glob.glob(base+'/*.h'))
for f in files:
    t=open(f).read()
    if 'execution_space' not in t: continue
    sets=re.findall(r'execution_space\s*=\s*([^;]+);',t)
    uses=len(re.findall(r'execution_space',t))-len(sets)
    print(os.path.basename(f), 'SET=',sets, 'USES=',uses)
