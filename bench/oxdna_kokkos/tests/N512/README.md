# N512 benchmark case

8192-nucleotide oxDNA2 system from Erik Poppleton's oxDNA performance suite
(<https://github.com/ErikPoppleton/oxDNA_performance>, `N512`). Same model
settings and run/cross-check procedure as `../N8/README.md`.

Verified against the standalone oxDNA (oxDNA2, T=20C, salt=1.0, average seq):
- total potential energy: -1.354229 / particle (matches standalone to ~1e-5)
- per-particle force / torque vectors: ~6e-5 / ~5e-4 relative
- short Brownian-NVT run is stable

Run:
```bash
./build/oxdna_kokkos -top tests/N512/topology_N512.top -conf tests/N512/init_conf_N512.dat \
                     -model 2 -salt 1.0 -T 0.097717 -dt 0.003 -newt 103 -diff 2.5 -steps 100000
```
