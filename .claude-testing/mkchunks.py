import subprocess,sys,os,re
S=os.environ.get('LMP_TEST_SCRATCH') or os.environ['S']; build=sys.argv[1]; tag=sys.argv[2]
CI=r'^(MolPairStyle|AtomicPairStyle|ManybodyPairStyle|EllipsoidPairStyle|KSpaceStyle|BondStyle|AngleStyle|DihedralStyle|ImproperStyle|FixTimestep|Library|FFT3D|DumpKokkos)'
out=subprocess.run(['ctest','-N','-R',CI],cwd=build,capture_output=True,text=True).stdout
names=[l.split(': ',1)[1].strip() for l in out.splitlines() if re.match(r'\s*Test\s+#\d+:',l)]
names=[n for n in names if n]
size=25
n=0
for i in range(0,len(names),size):
    n+=1
    with open(f'{S}/chunks/{tag}_{n:03d}.txt','w') as f:
        f.write('\n'.join(names[i:i+size]))
print(f'{tag}: {len(names)} tests -> {n} chunks')
