#!/bin/bash
# Resumable chunked test runner: skips chunks already marked done.
S=${LMP_TEST_SCRATCH:-/tmp/claude-0/-home-user-lammps/b7c14453-c726-5fc1-a3f6-52a175964ec4/scratchpad}
BUILD=$1; TAG=$2
# only for the cases that do not ask for a thread count themselves.  the force
# style tests set OMP_NUM_THREADS through the ctest ENVIRONMENT property, which
# overrides this, and they do so on purpose: they exercise the /omp styles and
# need a thread team to do it
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_PROC_BIND=false
mkdir -p $S/results
cd $BUILD || exit 1
for f in $S/chunks/${TAG}_*.txt; do
  base=$(basename $f .txt)
  [ -f $S/results/$base.done ] && continue
  # build an anchored alternation of the exact test names in this chunk
  RE=$(python3 "$(dirname "$0")/chunkre.py" "$f")
  ctest --output-on-failure -j 4 -R "$RE" > $S/results/$base.log 2>&1
  rc=$?
  fails=$(grep -cE '\*\*\*(Failed|Exception|Timeout)' $S/results/$base.log)
  echo "rc=$rc fails=$fails" > $S/results/$base.done
done
echo "${TAG}_CHUNKS_DONE" > $S/st-chunks-${TAG}.txt
