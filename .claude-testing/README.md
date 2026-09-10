# Session test harness for the KOKKOS host-backend runs

Scripts used to run the unit test suite for the KOKKOS package at mixed and
single precision in an environment whose container is reclaimed regularly.
They exist in the repository, rather than in a scratch directory, because the
scratch directory does not survive a container restart.

* `mkchunks.py` -- split the KOKKOS CI test selection into chunks of 25 tests
* `runchunks.sh` -- run those chunks, skipping the ones already marked done, so
  an interrupted run resumes instead of starting over
* `pipe2.sh` -- the whole sequence: mixed chunks, the single precision build and
  its chunks, then the same tests again under `ctest -T memcheck`

Not part of the build, and not intended for upstream: delete this directory
before opening a pull request.
