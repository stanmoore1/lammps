import re,glob,os
base='/home/user/lammps/src/KOKKOS'
files=[l.strip() for l in open('/tmp/tf.txt')]
for f in files:
    p=base+'/'+f
    for i,l in enumerate(open(p).read().split('\n'),1):
        s=l.strip()
        if s.startswith('//'): continue
        for m in re.finditer(r'Kokkos::View<([^;{]*?)>\s*(\w+)?',s):
            args=m.group(1)
            # count top-level template args
            depth=0; parts=[]; cur=''
            for ch in args:
                if ch=='<': depth+=1
                if ch=='>': depth-=1
                if ch==',' and depth==0: parts.append(cur); cur=''
                else: cur+=ch
            parts.append(cur)
            parts=[x.strip() for x in parts]
            if len(parts)==1:
                print(f"{f}:{i}: {s}")
                break
