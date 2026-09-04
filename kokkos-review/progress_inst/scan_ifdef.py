import re,glob,os
base='/home/user/lammps/src/KOKKOS'
files=[l.strip() for l in open('/tmp/tf.txt')]
pat=re.compile(r'#\s*(if|ifdef|elif|ifndef).*\b(KOKKOS_ENABLE_CUDA|KOKKOS_ENABLE_HIP|KOKKOS_ENABLE_SYCL|LMP_KOKKOS_GPU|KOKKOS_ENABLE_OPENMPTARGET|LMP_KK_DEVICE_COMPILE)\b')
for f in files:
    p=base+'/'+f
    lines=open(p).read().split('\n')
    n=len(lines)
    for i,l in enumerate(lines,1):
        if pat.search(l):
            # skip trailing instantiation block
            tail='\n'.join(lines[i-1:min(n,i+6)])
            if 'template class' in tail: continue
            print(f"{f}:{i}: {l.strip()}")
