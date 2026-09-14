# Running the example suite on a GPU

This describes how to run every input under `examples/` against a KOKKOS GPU
build and get a result that can be compared and acted on.  It is written for
someone -- or something -- starting from a clean checkout with no other
context.

## What the example inputs give you

Most `examples/**/in.*` files run for a fixed number of steps written into the
script.  Some of those runs are long enough that a whole-tree sweep spends most
of its wall clock in a handful of inputs.

The inputs in this branch take their step counts from `index` variables:

```
variable        nsteps index 40000     # the original value, unchanged
...
run             ${nsteps}
```

`-var` on the command line overrides an `index` variable
(`doc/src/variable.rst`), and binds nothing in a script that does not declare
one.  So `lmp -var nsteps 200` shortens exactly the inputs that were long
enough to need it and leaves every other example at its own length.  Running
with no `-var` at all reproduces the original behaviour byte for byte, which
is what keeps the bundled reference logs valid.

The names follow what the tree already used: `nsteps`, and `nequil`/`nprod`
where a script has a separate equilibration and production run.  A few scripts
have `nsteps1`/`nsteps2`/`nsteps3` for successive runs, and `maxiter`/`maxeval`
where the time goes into `minimize` rather than `run`.  Set all of them.

## Build

```bash
cmake -S cmake -B build-gpu -G Ninja \
  -C cmake/presets/gcc.cmake -C cmake/presets/most.cmake \
  -D PKG_KOKKOS=on -D Kokkos_ENABLE_CUDA=on -D Kokkos_ARCH_<YOUR_ARCH>=on \
  -D BUILD_MPI=on -D FFT_KOKKOS=KISS -D CMAKE_BUILD_TYPE=Release
cmake --build build-gpu -j
```

Set `Kokkos_ARCH_*` to the target card (`Kokkos_ARCH_VOLTA70`,
`Kokkos_ARCH_AMPERE80`, `Kokkos_ARCH_HOPPER90`, ...).  For AMD use
`Kokkos_ENABLE_HIP` with the matching `Kokkos_ARCH_AMD_*`; for Intel,
`Kokkos_ENABLE_SYCL`.  Getting the arch wrong usually still builds and then
runs far slower than it should, so check it.

## Package settings

Run with the settings that put the work on the device:

```
-k on g 1 -sf kk -pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on
```

`-k on g N` sets the number of GPUs per node.  `gpu/aware on` needs a
CUDA-aware or ROCm-aware MPI; turn it off if MPI aborts on the first exchange.

Some styles reject some of these.  `neigh full` is refused by anything needing
a half list (several dihedral styles), and `newton off` is refused by `fix
srd` among others.  Those inputs stop with a clean `ERROR:` and that is the
correct outcome, not a failure of the sweep -- count them separately.

## Running the whole tree

```bash
find examples -name 'in.*' -type f | sort > /tmp/inputs.txt

VAR="-var nsteps 200 -var nequil 100 -var nprod 100 -var maxiter 200 -var maxeval 200
     -var nsteps1 200 -var nsteps2 200 -var nsteps3 200"
PK="-k on g 1 -sf kk -pk kokkos neigh full newton off comm device sort device atom/map device gpu/aware on"

while read -r f; do
  d=$(dirname "$f"); b=$(basename "$f")
  name=$(echo "$f" | sed 's|examples/||; s|/|_|g')
  ( cd "$d" && \
    printf 'timer timeout 0:02:00 every 50\ninclude %s\n' "$b" > lmpsweep.in && \
    timeout 360 mpirun -np 4 "$LMP" -in lmpsweep.in -log none $PK $VAR \
      > /tmp/out/$name.out 2>&1
    rm -f lmpsweep.in )
  echo -e "$?\t$f" >> /tmp/out/index.txt
done < /tmp/inputs.txt
```

Three things in that loop matter more than they look:

**Bound the run with `timer timeout`, not with a kill.**  LAMMPS checks the
wall limit inside the run loop and stops cleanly, so the run still shuts down
and still prints.  A `timeout` that fires sends SIGKILL and you lose the
input's output entirely -- including any end-of-run diagnostic -- and it is
recorded as a failure rather than as "never measured".  Keep the hard
`timeout` as a backstop only, for a setup that never reaches a run loop.
`timer timeout` does not survive a `clear` command, and 43 inputs use one.

**Run each input in its own directory.**  Inputs read data and potential files
by relative path, and they write dumps and logs into the working directory.

**`-np 4`, not 1.**  The exchange and border paths differ between one rank and
several, and a large share of host/device coherence bugs live there.

Expect a few hours for the tree.  Record the exit status per input: 0 is a
clean run, 1 is usually one of the graceful style rejections above, 124 is the
hard timeout, and 134/139 are an abort or a segfault and are what you are
looking for.

## Reading the result

Sort by exit status first.  Anything that died on a signal is a bug.  Anything
that reports `ERROR:` needs its message read once -- most are the expected
incompatibilities, and a new one is worth chasing.

For numerical comparison, run the same inputs and the same `-var` values
against a CPU build and diff the thermo blocks.  Use a CPU baseline generated
by the same source on the same machine rather than the bundled `log.*` files,
which come from other compilers and machines and will differ in the last
digits for reasons that have nothing to do with the GPU.

`tools/regression-tests/run_tests.py` automates the comparison and has a
`--gen-ref` mode that writes such a baseline.

## If you have a second GPU build

Building twice, once with `-D KOKKOS_PREC=mixed` and once with the default
`double`, and diffing the two runs separates precision sensitivity from
outright bugs.  Note that `KOKKOS_PREC=mixed` requires `PKG_ML-IAP=off`, so
the ML-IAP examples drop out of that comparison.
