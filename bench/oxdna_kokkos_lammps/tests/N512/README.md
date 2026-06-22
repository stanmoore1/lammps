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
cd tests/N512 && ../../build/oxdna_kokkos input
```
